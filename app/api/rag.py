import tempfile
import asyncio
import json
import logging

import httpx
import psycopg
import hashlib
import io
from pathlib import Path
import re
import time
import unicodedata
import zipfile

from fastapi import Form, APIRouter, File, HTTPException, Response, Security, UploadFile, status
from fastapi.responses import StreamingResponse
import pypdf
from starlette.concurrency import run_in_threadpool

from app.core.config import settings
from app.core.llm import extract_message_text, get_model_provider, get_rag_llm
from app.core.tracing import (
    create_trace_id,
    get_langfuse_handler,
    langfuse_context,
    langfuse_observation,
    update_langfuse_observation,
)
from app.core.security import require_api_key
from app.schemas.rag import RagAskRequest, RagAskResponse, RagSource, RagUploadResponse
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
        status = getattr(current, "status_code", None)
        if status is None:
            status = getattr(getattr(current, "response", None), "status_code", None)
        if status is None:
            status = getattr(current, "code", None)
            if callable(status):
                try:
                    status = status()
                except Exception:
                    status = None
            status = getattr(status, "value", status)
        if status in (429, 503):
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


async def _retrieve_with_timeout(question: str, request: RagAskRequest, trace_id: str | None = None):
    # Xác định corpus hợp lệ:
    # 1. Nếu client chỉ định cụ thể request.corpus: tuân thủ yêu cầu đó (benchmark/offline eval).
    # 2. Nếu có session_id: tìm kiếm trên demo + corpus của session hiện tại.
    # 3. Mặc định: tìm kiếm trên demo.
    if request.corpus is not None:
        target_corpus = request.corpus
    elif request.session_id:
        target_corpus = ["demo", f"session_{request.session_id}"]
    else:
        target_corpus = "demo"

    async with asyncio.timeout(settings.rag_retrieval_timeout_seconds):
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


@router.post(
    "/rag/upload",
    response_model=RagUploadResponse,
    summary="Tải lên tài liệu người dùng và lập chỉ mục vào kho 'user' hoặc 'session'",
)
async def upload_document(
    file: UploadFile = File(...),
    session_id: str | None = Form(default=None),
) -> RagUploadResponse:
    # 1. Kích thước tệp (tối đa 5MB) - Chốt chặn tầng 2
    max_file_size = 5 * 1024 * 1024
    content = await file.read()
    if len(content) > max_file_size:
        raise HTTPException(
            status_code=getattr(status, "HTTP_413_CONTENT_TOO_LARGE", 413),
            detail="Kích thước tệp vượt quá giới hạn cho phép (tối đa 5MB)",
        )
    if len(content) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Tệp rỗng không có nội dung",
        )

    # 2. Định dạng tệp và chuẩn hóa Unicode NFC
    raw_name = file.filename or "uploaded_document"
    display_name = unicodedata.normalize("NFC", Path(raw_name).name)
    suffix = Path(display_name).suffix.lower()

    if suffix not in {".pdf", ".docx", ".txt", ".md"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Định dạng tệp không được hỗ trợ. Chỉ chấp nhận .pdf, .docx, .txt, .md",
        )

    # 3. Kiểm tra Magic Bytes và an toàn nội dung
    if suffix == ".pdf":
        if not content.startswith(b"%PDF-"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Tệp PDF không hợp lệ (sai magic bytes %PDF-)",
            )
        try:
            reader = pypdf.PdfReader(io.BytesIO(content))
            num_pages = len(reader.pages)
            if num_pages > 50:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Tài liệu PDF vượt quá giới hạn 50 trang (tệp có {num_pages} trang)",
                )
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Lỗi khi đọc cấu trúc tệp PDF")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Tệp PDF bị lỗi cấu trúc hoặc không thể giải mã nội dung.",
            ) from exc

    elif suffix == ".docx":
        if not content.startswith(b"PK\x03\x04"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Tệp DOCX không hợp lệ (sai magic bytes PK)",
            )
        try:
            with zipfile.ZipFile(io.BytesIO(content)) as zf:
                infolist = zf.infolist()
                if len(infolist) > 500:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Tệp DOCX chứa quá nhiều phần tử nội bộ (> 500 entries)",
                    )
                total_uncompressed = sum(info.file_size for info in infolist)
                if total_uncompressed > 25 * 1024 * 1024:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Tệp DOCX vượt quá giới hạn giải nén an toàn (tối đa 25MB)",
                    )
                names = set(zf.namelist())
                if "[Content_Types].xml" not in names and "word/document.xml" not in names:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Tệp ZIP không phải là tài liệu DOCX hợp lệ",
                    )
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Lỗi khi giải nén DOCX")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Tệp DOCX bị lỗi cấu trúc hoặc không thể giải mã nội dung.",
            ) from exc

    elif suffix in {".txt", ".md"}:
        if b"\x00" in content:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Tệp văn bản chứa ký tự null không an toàn (dấu hiệu tệp nhị phân)",
            )
        try:
            content.decode("utf-8")
        except UnicodeDecodeError:
            try:
                content.decode("latin-1")
            except Exception as exc:
                logger.exception("Lỗi giải mã văn bản")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Tệp văn bản không đúng chuẩn mã hóa hợp lệ",
                ) from exc

    # 4. Phân vùng corpus theo session (cô lập mềm)
    corpus = f"session_{session_id}" if session_id else "user"
    file_hash = hashlib.sha256(content).hexdigest()[:12]
    timestamp = int(time.time())
    safe_stem = re.sub(r"[^\w\-.]", "_", Path(display_name).stem)[:40]
    safe_filename = f"{timestamp}_{file_hash}_{safe_stem}{suffix}"

    # 5. Phân đoạn và lập chỉ mục an toàn qua NamedTemporaryFile tự hủy
    tmp_file = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp_path = Path(tmp_file.name)
    try:
        tmp_file.write(content)
        tmp_file.close()  # Đóng handle để các thư viện mở lại được trên Windows

        def _process_and_index() -> int:
            from app.core.ingestion import load_document_blocks, chunk_document_blocks
            from app.core.retriever import clear_bm25_cache

            blocks = load_document_blocks(tmp_path)
            chunks = chunk_document_blocks(blocks, strategy="heading")
            if not chunks:
                raise ValueError("Tệp rỗng hoặc không trích xuất được đoạn văn bản nào")

            source_key = f"user_uploads/{safe_filename}"
            formatted_chunks = []
            for idx, chunk in enumerate(chunks):
                formatted_chunks.append({
                    "source": source_key,
                    "chunk_index": idx,
                    "content": chunk["content"],
                    "page_number": chunk.get("page_number"),
                    "section": chunk.get("section"),
                    "display_name": display_name,
                })

            vector_store.upsert_document_chunks(
                source=source_key,
                chunks=formatted_chunks,
                corpus=corpus,
            )
            clear_bm25_cache()
            return len(formatted_chunks)

        indexed_count = await run_in_threadpool(_process_and_index)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Lỗi trong quá trình xử lý hoặc lập chỉ mục tài liệu")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Không thể phân tích hoặc lập chỉ mục tài liệu được tải lên.",
        ) from exc
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass

    return RagUploadResponse(
        status="success",
        filename=safe_filename,
        display_name=display_name,
        chunks_indexed=indexed_count,
        corpus=corpus,
    )


@router.post("/rag/ask", response_model=RagAskResponse, summary="Hỏi đáp dựa trên tài liệu đã nạp")
async def ask_rag(request: RagAskRequest, response: Response = None) -> RagAskResponse:
    trace_id = create_trace_id()
    if trace_id and response is not None:
        response.headers["X-Trace-Id"] = trace_id

    with langfuse_observation(
        name="rag-request",
        trace_id=trace_id,
        input={
            "question": request.question,
            "top_k": request.top_k,
            "session_id": request.session_id,
        },
    ) as observation:
        result = await _answer_rag(request, trace_id=trace_id)
        update_langfuse_observation(
            observation,
            output={
                "answer": result.answer,
                "source_count": len(result.sources),
            },
        )
        return result


@router.post("/rag/ask/stream", summary="Hỏi đáp tài liệu dạng streaming Server-Sent Events (SSE)")
async def ask_rag_stream(request: RagAskRequest):
    """
    Streaming endpoint returning Server-Sent Events (SSE).
    Sends retrieved sources as the first event, then streams answer tokens incrementally.
    If evidence is insufficient (gating cutoff), emits a no_answer event and finishes immediately.
    """
    trace_id = create_trace_id()
    history = session_store.get_history(request.session_id)
    standalone_question = await rewrite_question_if_needed(request.question, history)

    try:
        with langfuse_observation(name="rag-stream-retrieval", trace_id=trace_id) as observation:
            matches, is_refused = await _retrieve_with_timeout(standalone_question, request)
            context = format_context(matches)
            update_langfuse_observation(
                observation,
                output={"match_count": len(matches), "refused_by_retrieval": is_refused},
            )
    except Exception as exc:
        logger.exception("RAG retrieval failed in stream endpoint")
        if _is_timeout_error(exc):
            raise HTTPException(status_code=504, detail="RAG retrieval timed out") from exc
        if isinstance(exc, vector_store.VectorStoreNotReady):
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        raise HTTPException(status_code=503, detail="RAG retrieval unavailable") from exc

    async def event_generator():
        # First event: emit start event with trace_id immediately
        yield f"data: {json.dumps({'type': 'start', 'trace_id': trace_id}, ensure_ascii=False)}\n\n"

        if is_refused or not matches:
            session_store.add_turn(request.session_id, request.question, NO_ANSWER)
            yield f"data: {json.dumps({'type': 'no_answer', 'answer': NO_ANSWER, 'trace_id': trace_id}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
            return

        sources_payload = [
            {
                "source": m["source"],
                "chunk_index": m["chunk_index"],
                "score": m["score"],
                "section": m.get("section"),
                "page_number": m.get("page_number"),
                "display_name": m.get("display_name"),
            }
            for m in matches
        ]
        full_answer = []
        pending = ""
        sources_sent = False
        try:
            with langfuse_observation(name="rag-stream-generation", trace_id=trace_id) as gen_obs:
                prompt_template = build_rag_text_prompt()
                llm = get_rag_llm(temperature=0)
                chain = prompt_template | llm
                handler = get_langfuse_handler()
                callbacks = [handler] if handler else []
                with langfuse_context(session_id=request.session_id, tags=["rag", "streaming"]):
                    stream = chain.astream(
                        {"question": standalone_question, "context": context},
                        config={"callbacks": callbacks},
                    )
                    iterator = aiter(stream)
                    first_token = True
                    while True:
                        try:
                            wait_seconds = (
                                settings.rag_stream_first_token_timeout_seconds
                                if first_token else settings.rag_stream_idle_timeout_seconds
                            )
                            async with asyncio.timeout(wait_seconds):
                                chunk = await anext(iterator)
                        except StopAsyncIteration:
                            break
                        first_token = False
                        token = extract_message_text(chunk, strip=False)
                        if not token:
                            continue
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
                            yield f"data: {json.dumps({'type': 'sources', 'sources': sources_payload, 'trace_id': trace_id}, ensure_ascii=False)}\n\n"
                            sources_sent = True
                            yield f"data: {json.dumps({'type': 'token', 'token': pending}, ensure_ascii=False)}\n\n"
                            pending = ""
                        else:
                            yield f"data: {json.dumps({'type': 'token', 'token': token}, ensure_ascii=False)}\n\n"
                if gen_obs:
                    update_langfuse_observation(gen_obs, output={"answer": "".join(full_answer).strip()})
        except Exception as exc:
            logger.exception("Error during answer streaming")
            detail = "Answer generation timed out" if _is_timeout_error(exc) else "Answer generation failed"
            yield f"data: {json.dumps({'type': 'error', 'detail': detail, 'trace_id': trace_id}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
            return
        except (asyncio.CancelledError, GeneratorExit):
            partial_answer = "".join(full_answer).strip()
            if partial_answer:
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
                yield f"data: {json.dumps({'type': 'no_answer', 'answer': NO_ANSWER, 'trace_id': trace_id}, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
                return
            yield f"data: {json.dumps({'type': 'sources', 'sources': sources_payload, 'trace_id': trace_id}, ensure_ascii=False)}\n\n"
            if pending:
                yield f"data: {json.dumps({'type': 'token', 'token': pending}, ensure_ascii=False)}\n\n"
        session_store.add_turn(request.session_id, request.question, complete_answer)
        yield "data: [DONE]\n\n"

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    if trace_id:
        headers["X-Trace-Id"] = trace_id

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers=headers,
    )


async def _answer_rag(request: RagAskRequest, trace_id: str | None = None) -> RagAskResponse:
    # 1. Multi-turn session context & question rewriting (TASK-AI11)
    history = session_store.get_history(request.session_id)
    standalone_question = await rewrite_question_if_needed(request.question, history)

    # 2. Retrieval & Gating (TASK-AI09, TASK-AI10)
    try:
        with langfuse_observation(
            name="rag-retrieval",
            trace_id=trace_id,
            input={
                "question": standalone_question,
                "original_question": request.question,
                "top_k": request.top_k,
            },
        ) as observation:
            matches, is_refused = await _retrieve_with_timeout(standalone_question, request)
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
    except Exception as exc:
        logger.exception("RAG retrieval failed")
        if _is_timeout_error(exc):
            raise HTTPException(status_code=504, detail="RAG retrieval timed out") from exc
        if isinstance(exc, vector_store.VectorStoreNotReady):
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        if isinstance(exc, psycopg.OperationalError):
            raise HTTPException(status_code=503, detail="PostgreSQL retrieval service unavailable") from exc
        raise HTTPException(status_code=503, detail="RAG retrieval unavailable") from exc

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
            raise HTTPException(status_code=504, detail="RAG answer generation timed out") from exc
        raise HTTPException(
            status_code=502,
            detail=(
                f"RAG answer generation failed ({type(exc).__name__}, provider code {provider_code})"
                if provider_code is not None
                else f"RAG answer generation failed ({type(exc).__name__})"
            ),
        ) from exc

    answer = generated.answer.strip()
    if generated.abstain or not answer or answer == NO_ANSWER:
        session_store.add_turn(request.session_id, request.question, NO_ANSWER)
        return RagAskResponse(answer=NO_ANSWER, sources=[], trace_id=trace_id)

    sources = list({(m["source"], m["chunk_index"]): RagSource(
        source=m["source"], chunk_index=m["chunk_index"], score=m["score"],
        page_number=m.get("page_number"), section=m.get("section"),
        display_name=m.get("display_name"), corpus=m.get("corpus")
    ) for m in matches}.values())

    final_answer = answer or NO_ANSWER
    session_store.add_turn(request.session_id, request.question, final_answer)
    return RagAskResponse(answer=final_answer, sources=sources)
