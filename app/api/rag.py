import asyncio
import json
import logging

from fastapi import APIRouter, HTTPException, Security
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool

from app.core.config import settings
from app.core.llm import get_rag_llm
from app.core.tracing import (
    get_langfuse_handler,
    langfuse_context,
    langfuse_observation,
    update_langfuse_observation,
)
from app.core.security import require_api_key
from app.schemas.rag import RagAskRequest, RagAskResponse, RagSource
from app.core import vector_store
from app.core import session_store
from app.core.question_rewriter import rewrite_question_if_needed
from app.core.retriever import retrieve_and_gate
from app.core.rag_prompt import (
    NO_ANSWER,
    build_rag_chain,
    build_rag_text_prompt,
    format_context,
)

router = APIRouter(dependencies=[Security(require_api_key)])
logger = logging.getLogger(__name__)


@router.post("/rag/ask", response_model=RagAskResponse, summary="Hỏi đáp dựa trên tài liệu đã nạp")
async def ask_rag(request: RagAskRequest) -> RagAskResponse:
    with langfuse_observation(
        name="rag-request",
        input={
            "question": request.question,
            "top_k": request.top_k,
            "session_id": request.session_id,
        },
    ) as observation:
        response = await _answer_rag(request)
        update_langfuse_observation(
            observation,
            output={
                "answer": response.answer,
                "source_count": len(response.sources),
            },
        )
        return response


@router.post("/rag/ask/stream", summary="Hỏi đáp tài liệu dạng streaming Server-Sent Events (SSE)")
async def ask_rag_stream(request: RagAskRequest):
    """
    Streaming endpoint returning Server-Sent Events (SSE).
    Sends retrieved sources as the first event, then streams answer tokens incrementally.
    If evidence is insufficient (gating cutoff), emits a no_answer event and finishes immediately.
    """
    history = session_store.get_history(request.session_id)
    standalone_question = await rewrite_question_if_needed(request.question, history)

    try:
        with langfuse_observation(name="rag-stream-retrieval") as observation:
            matches, is_refused = await run_in_threadpool(
                retrieve_and_gate,
                standalone_question,
                request.top_k,
                settings.search_mode,
                settings.use_reranker,
                request.top_k,
                max(request.top_k, settings.reranker_pool_size),
            )
            update_langfuse_observation(
                observation,
                output={"match_count": len(matches), "refused_by_retrieval": is_refused},
            )
    except Exception as exc:
        logger.exception("RAG retrieval failed in stream endpoint")
        if isinstance(exc, vector_store.VectorStoreNotReady):
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        raise HTTPException(status_code=503, detail="RAG retrieval unavailable") from exc

    async def event_generator():
        if is_refused or not matches:
            session_store.add_turn(request.session_id, request.question, NO_ANSWER)
            yield f"data: {json.dumps({'type': 'no_answer', 'answer': NO_ANSWER}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
            return

        sources_payload = [
            {
                "source": m["source"],
                "chunk_index": m["chunk_index"],
                "score": m["score"],
                "section": m.get("section"),
                "page_number": m.get("page_number"),
            }
            for m in matches
        ]
        full_answer = []
        pending = ""
        sources_sent = False
        try:
            context = format_context(matches)
            prompt_template = build_rag_text_prompt()
            llm = get_rag_llm(temperature=0)
            chain = prompt_template | llm
            handler = get_langfuse_handler()
            callbacks = [handler] if handler else []
            with langfuse_context(session_id=request.session_id, tags=["rag", "streaming"]):
                async for chunk in chain.astream(
                    {"question": standalone_question, "context": context},
                    config={"callbacks": callbacks},
                ):
                    token = chunk.content if hasattr(chunk, "content") else str(chunk)
                    full_answer.append(token)
                    if not sources_sent:
                        pending += token
                        candidate = pending.strip()
                        if (
                            candidate
                            and (
                                NO_ANSWER.startswith(candidate)
                                or candidate.startswith(NO_ANSWER)
                            )
                        ):
                            continue
                        yield f"data: {json.dumps({'type': 'sources', 'sources': sources_payload}, ensure_ascii=False)}\n\n"
                        sources_sent = True
                        yield f"data: {json.dumps({'type': 'token', 'token': pending}, ensure_ascii=False)}\n\n"
                        pending = ""
                    else:
                        yield f"data: {json.dumps({'type': 'token', 'token': token}, ensure_ascii=False)}\n\n"
        except Exception as exc:
            logger.exception("Error during answer streaming")
            yield f"data: {json.dumps({'type': 'error', 'detail': 'Answer generation failed'}, ensure_ascii=False)}\n\n"
            return
        except (asyncio.CancelledError, GeneratorExit):
            # Preserve already generated context if the client drops the stream.
            # Mark it partial so follow-ups do not treat the answer as complete.
            partial_answer = "".join(full_answer).strip() or NO_ANSWER
            session_store.add_turn(
                request.session_id,
                request.question,
                f"[Câu trả lời bị ngắt, chưa hoàn tất]\n{partial_answer}",
            )
            raise

        complete_answer = "".join(full_answer).strip() or NO_ANSWER
        if not sources_sent:
            if complete_answer.startswith(NO_ANSWER):
                session_store.add_turn(request.session_id, request.question, NO_ANSWER)
                yield f"data: {json.dumps({'type': 'no_answer', 'answer': NO_ANSWER}, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
                return
            yield f"data: {json.dumps({'type': 'sources', 'sources': sources_payload}, ensure_ascii=False)}\n\n"
            if pending:
                yield f"data: {json.dumps({'type': 'token', 'token': pending}, ensure_ascii=False)}\n\n"
        session_store.add_turn(request.session_id, request.question, complete_answer)
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


async def _answer_rag(request: RagAskRequest) -> RagAskResponse:
    # 1. Multi-turn session context & question rewriting (TASK-AI11)
    history = session_store.get_history(request.session_id)
    standalone_question = await rewrite_question_if_needed(request.question, history)

    # 2. Retrieval & Gating (TASK-AI09, TASK-AI10)
    try:
        with langfuse_observation(
            name="rag-retrieval",
            input={
                "question": standalone_question,
                "original_question": request.question,
                "top_k": request.top_k,
            },
        ) as observation:
            matches, is_refused = await run_in_threadpool(
                retrieve_and_gate,
                standalone_question,
                request.top_k,
                settings.search_mode,
                settings.use_reranker,
                request.top_k,
                max(request.top_k, settings.reranker_pool_size),
            )
            update_langfuse_observation(
                observation,
                output={
                    "match_count": len(matches),
                    "sources": [
                        {
                            "source": match["source"],
                            "chunk_index": match["chunk_index"],
                            "score": match["score"],
                            "section": match.get("section"),
                        }
                        for match in matches
                    ],
                    "context": format_context(matches),
                },
            )
    except Exception as exc:
        logger.exception("RAG retrieval failed")
        if isinstance(exc, vector_store.VectorStoreNotReady):
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        raise HTTPException(status_code=503, detail="RAG retrieval unavailable") from exc

    if is_refused or not matches:
        session_store.add_turn(request.session_id, request.question, NO_ANSWER)
        return RagAskResponse(answer=NO_ANSWER, sources=[])

    # 3. LLM Answer Generation
    context = format_context(matches)
    try:
        handler = get_langfuse_handler()
        callbacks = [handler] if handler else []
        with langfuse_context(session_id=request.session_id, tags=["rag", "ai07"]):
            chain = await run_in_threadpool(
                build_rag_chain,
                get_rag_llm(temperature=0),
            )
            generated = await chain.ainvoke(
                {"question": standalone_question, "context": context},
                config={"callbacks": callbacks}
            )
    except Exception as exc:
        logger.exception("RAG answer generation failed")
        raise HTTPException(status_code=502, detail="RAG answer generation failed") from exc

    answer = generated.answer.strip()
    if generated.abstain or not answer or answer == NO_ANSWER:
        session_store.add_turn(request.session_id, request.question, NO_ANSWER)
        return RagAskResponse(answer=NO_ANSWER, sources=[])

    sources = list({(m["source"], m["chunk_index"]): RagSource(
        source=m["source"], chunk_index=m["chunk_index"], score=m["score"],
        page_number=m.get("page_number"), section=m.get("section")
    ) for m in matches}.values())

    final_answer = answer or NO_ANSWER
    session_store.add_turn(request.session_id, request.question, final_answer)
    return RagAskResponse(answer=final_answer, sources=sources)
