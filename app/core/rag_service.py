import asyncio
import logging

from fastapi import HTTPException, status
import httpx
import psycopg
from starlette.concurrency import run_in_threadpool

from app.core.config import settings
from app.core.llm import get_model_provider, get_rag_llm
from app.core.question_rewriter import rewrite_question
from app.core.rag_prompt import (
    NO_ANSWER,
    build_rag_chain,
    format_context,
)
from app.core.retriever import retrieve_and_gate
from app.core import session_store
from app.core.tracing import (
    get_langfuse_handler,
    langfuse_context,
    langfuse_observation,
    update_langfuse_observation,
)
from app.core import vector_store
from app.schemas.rag import RagAskRequest, RagAskResponse, RagSource

logger = logging.getLogger(__name__)

DEFAULT_CORPORA = ("demo", "user")


def _select_sources(matches: list[dict]) -> list[dict]:
    """Collapse chunks into one row per document, best chunk only."""
    if not matches:
        return []
    ranked = sorted(matches, key=lambda match: match["score"], reverse=True)
    best_by_source: dict[str, dict] = {}
    for match in ranked:
        best_by_source.setdefault(match["source"], match)
    return list(best_by_source.values())


def _session_corpus(session_id: str) -> str:
    return f"session_{session_id}"


def _resolve_target_corpus(request: RagAskRequest):
    """Resolve which corpora a request may search."""
    if request.corpus is not None:
        allowed = set(DEFAULT_CORPORA)
        if request.session_id:
            allowed.add(_session_corpus(request.session_id))
        if request.corpus not in allowed:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Không có quyền truy cập corpus được yêu cầu.",
            )
        return request.corpus
    if request.session_id:
        return [*DEFAULT_CORPORA, _session_corpus(request.session_id)]
    return list(DEFAULT_CORPORA)


def _rag_timeout_seconds() -> float:
    if get_model_provider(settings.rag_model) == "ollama":
        return settings.rag_local_model_timeout_seconds
    return settings.rag_model_timeout_seconds


def _is_timeout_error(exc: BaseException) -> bool:
    seen = set()
    current = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, (TimeoutError, httpx.TimeoutException)) or "timeout" in type(current).__name__.casefold():
            return True
        current = current.__cause__ or current.__context__
    return False


def _is_retryable_provider_error(exc: Exception) -> bool:
    current = exc
    seen = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        code = getattr(current, "status_code", None)
        if code is None:
            code = getattr(getattr(current, "response", None), "status_code", None)
        if code is None:
            code = getattr(current, "code", None)
            if callable(code):
                try:
                    code = code()
                except Exception:
                    code = None
            code = getattr(code, "value", code)
        if code in (429, 503):
            return True
        current = current.__cause__ or current.__context__
    return False


def _provider_error_code(exc: BaseException) -> int | None:
    current = exc
    seen = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        code = getattr(current, "status_code", None)
        if code is None:
            code = getattr(getattr(current, "response", None), "status_code", None)
        if code is None:
            code = getattr(current, "code", None)
            if callable(code):
                try:
                    code = code()
                except Exception:
                    code = None
            code = getattr(code, "value", code)
        try:
            if code is not None:
                return int(code)
        except (TypeError, ValueError):
            pass
        current = current.__cause__ or current.__context__
    return None


async def _invoke_rag_with_retry(chain, inputs: dict, config: dict):
    """Bound each attempt; retry one transient quota/service error only."""
    attempts = settings.rag_model_max_retries + 1
    for attempt in range(attempts):
        try:
            async with asyncio.timeout(_rag_timeout_seconds()):
                return await chain.ainvoke(inputs, config=config)
        except Exception as exc:
            if attempt + 1 >= attempts or not _is_retryable_provider_error(exc):
                raise
            logger.warning("Retrying transient RAG provider error (%s)", type(exc).__name__)
            await asyncio.sleep(0.5)
    raise RuntimeError("RAG model call did not return")


async def _retrieve_with_timeout(
    question: str,
    request: RagAskRequest,
    trace_id: str | None = None,
    raw_question: str | None = None,
):
    target_corpus = _resolve_target_corpus(request)

    async with asyncio.timeout(settings.rag_retrieval_timeout_seconds):
        if raw_question and raw_question.strip().casefold() != question.strip().casefold():
            return await run_in_threadpool(
                retrieve_and_gate,
                question,
                request.top_k,
                settings.search_mode,
                settings.use_reranker,
                request.top_k,
                max(request.top_k, settings.reranker_pool_size),
                target_corpus,
                raw_question,
            )
        return await run_in_threadpool(
            retrieve_and_gate,
            question,
            request.top_k,
            settings.search_mode,
            settings.use_reranker,
            request.top_k,
            max(request.top_k, settings.reranker_pool_size),
            target_corpus,
        )


async def answer_rag_request(request: RagAskRequest, trace_id: str | None = None) -> RagAskResponse:
    """Core RAG answer generation service."""
    headers = {"X-Trace-Id": trace_id} if trace_id else None

    # 1. Multi-turn session context & question rewriting
    history = session_store.get_history(request.session_id)
    rewrite = await rewrite_question(
        request.question,
        history,
        trace_id=trace_id,
    )
    standalone_question = rewrite.text
    dual_query = (
        request.question
        if rewrite.accent_restored and not rewrite.coref_resolved
        else None
    )

    # 2. Retrieval & Gating
    try:
        with langfuse_observation(
            name="rag-retrieval",
            trace_id=trace_id,
            input={
                "question": standalone_question,
                "original_question": request.question,
                "dual_query": dual_query,
                "accent_restored": rewrite.accent_restored,
                "coref_resolved": rewrite.coref_resolved,
                "rewrite_failed": rewrite.failed,
                "top_k": request.top_k,
            },
        ) as observation:
            matches, is_refused = await _retrieve_with_timeout(
                standalone_question, request, trace_id=trace_id, raw_question=dual_query
            )
            context = format_context(matches)
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
                    "context": context,
                },
            )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("RAG retrieval failed")
        if _is_timeout_error(exc):
            raise HTTPException(status_code=504, detail="RAG retrieval timed out", headers=headers) from exc
        if isinstance(exc, vector_store.VectorStoreNotReady):
            raise HTTPException(status_code=503, detail=str(exc), headers=headers) from exc
        if isinstance(exc, (psycopg.Error, httpx.HTTPError, OSError)):
            raise HTTPException(status_code=503, detail="RAG retrieval unavailable", headers=headers) from exc
        raise HTTPException(status_code=503, detail="RAG retrieval unavailable", headers=headers) from exc

    if is_refused or not matches:
        session_store.add_turn(request.session_id, request.question, NO_ANSWER)
        return RagAskResponse(answer=NO_ANSWER, sources=[], trace_id=trace_id)

    # 3. LLM Answer Generation
    try:
        handler = get_langfuse_handler()
        callbacks = [handler] if handler else []
        with langfuse_context(session_id=request.session_id, tags=["rag", "ai07"]):
            chain = await run_in_threadpool(
                build_rag_chain,
                get_rag_llm(temperature=0),
            )
            generated = await _invoke_rag_with_retry(
                chain,
                {"question": standalone_question, "context": context},
                {"callbacks": callbacks},
            )
    except Exception as exc:
        provider_code = _provider_error_code(exc)
        logger.exception("RAG answer generation failed (type=%s, code=%s)", type(exc).__name__, provider_code)
        if _is_timeout_error(exc):
            raise HTTPException(status_code=504, detail="RAG answer generation timed out", headers=headers) from exc
        raise HTTPException(
            status_code=502,
            detail=(
                f"RAG answer generation failed ({type(exc).__name__}, provider code {provider_code})"
                if provider_code is not None
                else f"RAG answer generation failed ({type(exc).__name__})"
            ),
            headers=headers,
        ) from exc

    answer = generated.answer.strip()
    if generated.abstain or not answer or answer == NO_ANSWER:
        session_store.add_turn(request.session_id, request.question, NO_ANSWER)
        return RagAskResponse(answer=NO_ANSWER, sources=[], trace_id=trace_id)

    sources = [
        RagSource(
            source=m["source"],
            chunk_index=m["chunk_index"],
            score=m["score"],
            page_number=m.get("page_number"),
            section=m.get("section"),
            display_name=m.get("display_name"),
            corpus=m.get("corpus"),
        )
        for m in _select_sources(matches)
    ]

    final_answer = answer or NO_ANSWER
    session_store.add_turn(request.session_id, request.question, final_answer)
    return RagAskResponse(answer=final_answer, sources=sources, trace_id=trace_id)


# Alias for backwards compatibility
_answer_rag = answer_rag_request
