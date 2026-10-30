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
    end_langfuse_observation,
    get_langfuse_handler,
    langfuse_context,
    langfuse_observation,
    start_langfuse_observation,
    update_langfuse_observation,
)
from app.core.security import require_api_key
from app.schemas.rag import RagAskRequest, RagAskResponse, RagSource, RagUploadResponse, SESSION_ID_PATTERN
from app.core import vector_store
from app.core import session_store
from app.core.question_rewriter import RewriteResult, rewrite_question  # noqa: F401
from app.core.retriever import retrieve_and_gate
from app.core.rag_prompt import (
    NO_ANSWER,
    build_rag_chain,
    build_rag_text_prompt,
    format_context,
)

router = APIRouter(dependencies=[Security(require_api_key)])

# Một tài liệu chỉ hiển thị một lần (giữ chunk điểm cao nhất). Danh sách nguồn
# phản chiếu đúng những tài liệu đã đưa vào ngữ cảnh; UI tự giới hạn khi cần.
def _select_sources(matches: list[dict]) -> list[dict]:
    """Collapse chunks into one row per document, best chunk only.

    Every distinct document present in the retrieved context is reported, so
    the source list cannot claim less evidence than the model actually saw.
    Used by both the JSON and the SSE endpoints so the two paths cannot drift
    apart.
    """
    if not matches:
        return []
    ranked = sorted(matches, key=lambda match: match["score"], reverse=True)
    best_by_source: dict[str, dict] = {}
    for match in ranked:
        best_by_source.setdefault(match["source"], match)
    return list(best_by_source.values())


logger = logging.getLogger(__name__)

DEFAULT_CORPORA = ("demo", "user")


def _session_corpus(session_id: str) -> str:
    return f"session_{session_id}"


def _resolve_target_corpus(request: RagAskRequest):
    """Resolve which corpora a request may search.

    Clients may pin `demo`, the shared `user` corpus, or their own session
    corpus. Session corpora belonging to other sessions are rejected so a
    caller cannot read another session's uploads by guessing its id.
    """
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


async def _retrieve_with_timeout(question: str, request: RagAskRequest, trace_id: str | None = None, raw_question: str | None = None):
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


@router.post(
    "/rag/upload",
    response_model=RagUploadResponse,
    summary="Tải lên tài liệu người dùng và lập chỉ mục vào kho 'user' hoặc 'session'",
)
async def upload_document(
    file: UploadFile = File(...),
    session_id: str | None = Form(default=None, max_length=128, pattern=SESSION_ID_PATTERN),
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
            content.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Tệp văn bản phải được mã hóa UTF-8 hợp lệ",
            ) from exc

    # 4. Phân vùng corpus theo session (cô lập mềm)
    if not session_id:
        logger.warning(
            "Upload không có session_id: tài liệu được lưu vào corpus 'user' dùng chung."
        )
    corpus = f"session_{session_id}" if session_id else "user"
    file_hash = hashlib.sha256(content).hexdigest()[:12]
    timestamp = int(time.time())
    safe_stem = re.sub(r"[^\w\-.]", "_", Path(display_name).stem)[:40]
    safe_filename = f"{timestamp}_{file_hash}_{safe_stem}{suffix}"

    # 5. Phân đoạn và lập chỉ mục an toàn qua NamedTemporaryFile tự hủy
    # Register before writing rows so a concurrent startup sweep cannot mistake
    # this brand-new session corpus for an orphan.
    session_store.touch_session(session_id)

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
            clear_bm25_cache(corpus=corpus)
            return len(formatted_chunks)

        indexed_count = await run_in_threadpool(_process_and_index)
    except HTTPException:
        raise
    except ValueError as exc:
        logger.warning("Tài liệu tải lên không hợp lệ: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Không thể phân tích tài liệu: nội dung trống hoặc không hợp lệ.",
        ) from exc
    except vector_store.VectorStoreNotReady as exc:
        logger.exception("Kho vector chưa sẵn sàng để lập chỉ mục tài liệu")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Dịch vụ lập chỉ mục tài liệu chưa sẵn sàng.",
        ) from exc
    except (psycopg.Error, httpx.HTTPError, OSError, TimeoutError) as exc:
        logger.exception("Lỗi hạ tầng khi lập chỉ mục tài liệu")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Dịch vụ lập chỉ mục tài liệu tạm thời không khả dụng.",
        ) from exc
    except Exception as exc:
        logger.exception("Lỗi không mong đợi khi xử lý hoặc lập chỉ mục tài liệu")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Không thể xử lý tài liệu do lỗi không mong đợi.",
        ) from exc
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass

    return RagUploadResponse(
        status="success",
        filename=display_name,
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
    stream_root = start_langfuse_observation(
        "rag-stream",
        trace_id=trace_id,
        input={"question": request.question, "session_id": request.session_id},
    )
    root_span_id = getattr(stream_root, "id", None)

    try:
        history = session_store.get_history(request.session_id)
        rewrite = await rewrite_question(
            request.question,
            history,
            trace_id=trace_id,
            parent_span_id=root_span_id,
        )
        standalone_question = rewrite.text
        dual_query = (
            request.question
            if rewrite.accent_restored and not rewrite.coref_resolved
            else None
        )
        update_langfuse_observation(
            stream_root,
            metadata={
                "standalone_question": standalone_question,
                "accent_restored": rewrite.accent_restored,
                "coref_resolved": rewrite.coref_resolved,
                "rewrite_failed": rewrite.failed,
                "used_dual_retrieval": dual_query is not None,
            },
        )
        with langfuse_observation(
            name="rag-stream-retrieval",
            trace_id=trace_id,
            parent_span_id=root_span_id,
            input={
                "question": standalone_question,
                "original_question": request.question,
                "dual_query": dual_query,
            },
        ) as observation:
            matches, is_refused = await _retrieve_with_timeout(
                standalone_question, request, trace_id=trace_id, raw_question=dual_query
            )
            context = format_context(matches)
            update_langfuse_observation(
                observation,
                output={"match_count": len(matches), "refused_by_retrieval": is_refused},
            )
    except HTTPException:
        end_langfuse_observation(stream_root)
        raise
    except Exception as exc:
        end_langfuse_observation(stream_root)
        logger.exception("RAG retrieval failed in stream endpoint")
        if _is_timeout_error(exc):
            raise HTTPException(status_code=504, detail="RAG retrieval timed out") from exc
        if isinstance(exc, vector_store.VectorStoreNotReady):
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        if isinstance(exc, (psycopg.Error, httpx.HTTPError, OSError)):
            raise HTTPException(status_code=503, detail="RAG retrieval unavailable") from exc
        raise HTTPException(status_code=503, detail="RAG retrieval unavailable") from exc

    async def _stream_events():
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
            for m in _select_sources(matches)
        ]
        full_answer = []
        pending = ""
        sources_sent = False
        gen_obs = start_langfuse_observation(
            "rag-stream-generation",
            trace_id=trace_id,
            parent_span_id=root_span_id,
        )
        try:
            prompt_template = build_rag_text_prompt()
            llm = get_rag_llm(temperature=0)
            chain = prompt_template | llm
            generation_span_id = getattr(gen_obs, "id", None) or root_span_id
            handler = get_langfuse_handler(
                trace_context=(
                    {"trace_id": trace_id, "parent_span_id": generation_span_id}
                    if trace_id and generation_span_id
                    else None
                )
            )
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
        finally:
            # End the generation span in every exit path, including client
            # disconnects that close this async generator mid-yield.
            end_langfuse_observation(gen_obs)

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

    async def event_generator():
        events = _stream_events()
        try:
            async for event in events:
                yield event
        finally:
            # Closing the client connection must also run the inner generator's
            # GeneratorExit handler so a partial answer is persisted.
            try:
                await events.aclose()
            finally:
                end_langfuse_observation(stream_root)

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

    # 2. Retrieval & Gating (TASK-AI09, TASK-AI10)
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
            raise HTTPException(status_code=504, detail="RAG retrieval timed out") from exc
        if isinstance(exc, vector_store.VectorStoreNotReady):
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        if isinstance(exc, (psycopg.Error, httpx.HTTPError, OSError)):
            raise HTTPException(status_code=503, detail="RAG retrieval unavailable") from exc
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

    sources = [RagSource(
        source=m["source"], chunk_index=m["chunk_index"], score=m["score"],
        page_number=m.get("page_number"), section=m.get("section"),
        display_name=m.get("display_name"), corpus=m.get("corpus")
    ) for m in _select_sources(matches)]

    final_answer = answer or NO_ANSWER
    session_store.add_turn(request.session_id, request.question, final_answer)
    return RagAskResponse(answer=final_answer, sources=sources)
