import logging

from fastapi import APIRouter, HTTPException, Security
from starlette.concurrency import run_in_threadpool

from app.core.config import settings
from app.core.llm import get_llm
from app.core.tracing import (
    get_langfuse_handler,
    langfuse_context,
    langfuse_observation,
    update_langfuse_observation,
)
from app.core.security import require_api_key
from app.schemas.rag import RagAskRequest, RagAskResponse, RagSource
from app.core import vector_store
from app.core.rag_prompt import NO_ANSWER, build_rag_messages, format_context

router = APIRouter(dependencies=[Security(require_api_key)])
logger = logging.getLogger(__name__)


@router.post("/rag/ask", response_model=RagAskResponse, summary="Hỏi đáp dựa trên tài liệu đã nạp")
async def ask_rag(request: RagAskRequest) -> RagAskResponse:
    with langfuse_observation(
        name="rag-request",
        input={"question": request.question, "top_k": request.top_k},
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


async def _answer_rag(request: RagAskRequest) -> RagAskResponse:
    try:
        with langfuse_observation(
            name="rag-retrieval",
            input={"question": request.question, "top_k": request.top_k},
        ) as observation:
            matches = await run_in_threadpool(
                vector_store.similarity_search, request.question, request.top_k
            )
            matches = [
                match for match in matches
                if match["score"] >= settings.rag_min_relevance_score
            ]
            update_langfuse_observation(
                observation,
                output={
                    "match_count": len(matches),
                    "sources": [
                        {
                            "source": match["source"],
                            "chunk_index": match["chunk_index"],
                            "score": match["score"],
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

    if not matches:
        return RagAskResponse(answer=NO_ANSWER, sources=[])

    context = format_context(matches)
    try:
        handler = get_langfuse_handler()
        callbacks = [handler] if handler else []
        with langfuse_context(session_id=request.session_id, tags=["rag", "ai07"]):
            result = await get_llm(settings.rag_model, temperature=0).ainvoke(
                build_rag_messages(request.question, context),
                config={"callbacks": callbacks}
            )
    except Exception as exc:
        logger.exception("RAG answer generation failed")
        raise HTTPException(status_code=502, detail="RAG answer generation failed") from exc

    answer = str(result.content).strip()
    if not answer or answer == NO_ANSWER:
        return RagAskResponse(answer=NO_ANSWER, sources=[])
    sources = list({(m["source"], m["chunk_index"]): RagSource(
        source=m["source"], chunk_index=m["chunk_index"], score=m["score"],
        page_number=m.get("page_number")
    ) for m in matches}.values())
    return RagAskResponse(answer=answer or NO_ANSWER, sources=sources)
