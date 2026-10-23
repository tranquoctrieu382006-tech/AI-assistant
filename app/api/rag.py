import logging

from fastapi import APIRouter, HTTPException
from starlette.concurrency import run_in_threadpool

from app.core.config import settings
from app.core.llm import get_llm
from app.core.tracing import get_langfuse_handler, langfuse_context
from app.schemas.rag import RagAskRequest, RagAskResponse, RagSource

router = APIRouter()
logger = logging.getLogger(__name__)
NO_ANSWER = "Tài liệu hiện có không chứa đủ thông tin để trả lời câu hỏi này."


@router.post("/rag/ask", response_model=RagAskResponse, summary="Hỏi đáp dựa trên tài liệu đã nạp")
async def ask_rag(request: RagAskRequest) -> RagAskResponse:
    try:
        from app.core.vector_store import similarity_search
        matches = await run_in_threadpool(similarity_search, request.question, request.top_k)
    except Exception as exc:
        logger.exception("RAG retrieval failed")
        raise HTTPException(status_code=503, detail="RAG retrieval unavailable") from exc

    # Avoid fabricating an answer when retrieval has no plausible evidence.
    if not matches or matches[0]["score"] < settings.rag_min_relevance_score:
        return RagAskResponse(answer=NO_ANSWER, sources=[])

    context = "\n\n".join(
        f"[Nguồn: {m['source']} | đoạn {m['chunk_index']}]\n{m['content']}" for m in matches
    )
    prompt = (
        "Trả lời bằng tiếng Việt chỉ dựa trên ngữ cảnh. Nếu ngữ cảnh không có câu trả lời, "
        f"hãy trả lời chính xác: {NO_ANSWER}\n\nNgữ cảnh:\n{context}\n\nCâu hỏi: {request.question}"
    )
    try:
        handler = get_langfuse_handler()
        callbacks = [handler] if handler else []
        with langfuse_context(session_id=request.session_id, tags=["rag", "ai07"]):
            result = await get_llm(settings.rag_model, temperature=0).ainvoke(
                prompt, config={"callbacks": callbacks}
            )
    except Exception as exc:
        logger.exception("RAG answer generation failed")
        raise HTTPException(status_code=502, detail="RAG answer generation failed") from exc

    answer = str(result.content).strip()
    sources = list({(m["source"], m["chunk_index"]): RagSource(
        source=m["source"], chunk_index=m["chunk_index"], score=m["score"]
    ) for m in matches}.values())
    return RagAskResponse(answer=answer or NO_ANSWER, sources=sources)
