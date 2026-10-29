"""
Streamlit Web UI - Mini AI Assistant (RAG & NLQ).

Hỗ trợ:
1. Phân loại câu hỏi tự động (Smart Intent Router: RAG / NLQ / Out of scope).
2. Hiển thị streaming và các bước đang chạy (st.status).
3. Hỏi đáp tài liệu doanh nghiệp qua RAG (streaming Server-Sent Events, hiển thị nguồn và trace_id).
4. Truy vấn dữ liệu tự nhiên NLQ (sinh SQL an toàn, bảng dữ liệu pandas, biểu đồ trực quan).
5. Tải lên và đánh chỉ mục tài liệu mới qua API backend /api/rag/upload (hoặc trực tiếp qua pgvector + bge-m3).
6. Tùy chọn vai trò mô hình ngôn ngữ (fast, strong, local).
7. Phản hồi người dùng (👍 / 👎 qua st.feedback) gắn chặt với trace_id gửi về Langfuse.
"""

import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import streamlit as st

# Đảm bảo đường dẫn import tới app core
CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.core.config import settings
from app.core.router import route_question, RouteCategory
from app.core.tracing import get_langfuse_client, flush_langfuse
from app.core.ingestion import ingest_directory


# ---------------------------------------------------------
# CẤU HÌNH TRANG STREAMLIT
# ---------------------------------------------------------
st.set_page_config(
    page_title="Mini AI Assistant - RAG & NLQ",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

BACKEND_URL = os.getenv("BACKEND_URL", f"http://127.0.0.1:{settings.app_port}")
API_KEY = settings.app_api_key


# ---------------------------------------------------------
# QUẢN LÝ SESSION STATE
# ---------------------------------------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = f"session-{uuid.uuid4().hex[:8]}"

if "messages" not in st.session_state:
    st.session_state.messages = []

if "feedback_given" not in st.session_state:
    st.session_state.feedback_given = {}


def reset_session():
    st.session_state.session_id = f"session-{uuid.uuid4().hex[:8]}"
    st.session_state.messages = []
    st.session_state.feedback_given = {}
    st.rerun()


# ---------------------------------------------------------
# HÀM GỬI FEEDBACK VỀ BACKEND / LANGFUSE
# ---------------------------------------------------------
def submit_feedback(msg_idx: int, score: float, trace_id: str | None = None, comment: str = ""):
    """Gửi đánh giá người dùng qua API /api/feedback (gắn trace_id) hoặc SDK Langfuse."""
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": API_KEY,
    }
    payload = {
        "score": score,
        "trace_id": trace_id,
        "session_id": st.session_state.session_id,
        "comment": comment,
    }

    recorded = False
    # 1. Gọi API backend /api/feedback
    try:
        resp = httpx.post(f"{BACKEND_URL}/api/feedback", json=payload, headers=headers, timeout=5.0)
        if resp.status_code == 200:
            recorded = True
    except Exception:
        pass

    # 2. Fallback trực tiếp qua SDK nếu backend chưa bật
    if not recorded:
        try:
            client = get_langfuse_client()
            if client:
                if trace_id:
                    client.create_score(name="user_feedback", value=score, trace_id=trace_id, comment=comment)
                else:
                    client.create_score(name="user_feedback", value=score, session_id=st.session_state.session_id, comment=comment)
                flush_langfuse()
                recorded = True
        except Exception as e:
            st.error(f"Lỗi ghi nhận phản hồi: {e}")

    st.session_state.feedback_given[msg_idx] = score
    if recorded:
        st.toast(f"✅ Đã ghi nhận phản hồi {'👍 Hài lòng' if score == 1.0 else '👎 Chưa đúng'}!", icon="🎉")
    else:
        st.toast("Đã ghi nhận phản hồi cục bộ.", icon="ℹ️")


# ---------------------------------------------------------
# SIDEBAR: CẤU HÌNH & TÍNH NĂNG PHỤ
# ---------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Cấu Hình & Công Cụ")

    # Kiểm tra trạng thái Backend
    try:
        resp = httpx.get(f"{BACKEND_URL}/api/health/live", timeout=2.0)
        if resp.status_code == 200:
            st.success(f"🟢 Backend: Sẵn sàng ({BACKEND_URL})")
        else:
            st.warning(f"🟡 Backend: Trả về HTTP {resp.status_code}")
    except Exception:
        st.error(f"🔴 Backend chưa bật tại {BACKEND_URL}")
        st.caption("Khởi động: `uvicorn app.main:app --port 8000`")

    st.divider()

    # Chọn chế độ trợ lý
    st.subheader("🎯 Chế độ trợ lý")
    mode = st.radio(
        "Chọn luồng xử lý câu hỏi:",
        options=[
            "🤖 Tự động thông minh (Router)",
            "📑 Hỏi đáp tài liệu (RAG)",
            "📊 Truy vấn CSDL bán hàng (NLQ)",
        ],
        index=0,
    )

    # Chọn mô hình xử lý (DoD: chọn mô hình)
    st.subheader("🧠 Chọn mô hình")
    selected_model_role = st.selectbox(
        "Vai trò mô hình (Role):",
        options=[
            "fast (gemini-3.5-flash-lite)",
            "strong (gemini-3.6-flash)",
            "local (ollama/gemma3:12b)",
        ],
        index=0,
        help="Cấu hình vai trò mô hình thực thi cho RAG & NLQ",
    )

    st.caption(f"**Phiên hiện tại:** `{st.session_state.session_id}`")
    if st.button("🔄 Tạo phiên trò chuyện mới", use_container_width=True):
        reset_session()

    st.divider()

    # Upload tài liệu mới qua API backend /api/rag/upload
    st.subheader("📤 Tải lên tài liệu mới")
    st.caption("Tài liệu sẽ được chunking và index tự động vào PostgreSQL/pgvector.")
    uploaded_file = st.file_uploader(
        "Chọn file tài liệu:",
        type=["docx", "pdf", "md", "txt"],
        help="Hỗ trợ văn bản chính sách, quy định, hướng dẫn bán hàng/bảo hành.",
    )

    if uploaded_file is not None:
        if st.button("⚡ Đánh chỉ mục (Index) ngay", use_container_width=True):
            with st.spinner("Đang tải lên và đánh chỉ mục tài liệu vào PGVector..."):
                file_bytes = uploaded_file.getvalue()
                uploaded_via_api = False

                # 1. Thử gửi qua endpoint /api/rag/upload của Backend
                try:
                    upload_resp = httpx.post(
                        f"{BACKEND_URL}/api/rag/upload",
                        headers={"X-API-Key": API_KEY},
                        files={"file": (uploaded_file.name, file_bytes, uploaded_file.type or "application/octet-stream")},
                        timeout=120.0,
                    )
                    if upload_resp.status_code == 200:
                        data = upload_resp.json()
                        st.success(
                            f"✅ Đã nạp thành công **{data.get('chunks')} chunks** "
                            f"từ **{data.get('files')} tài liệu** qua API!"
                        )
                        st.info("Bây giờ bạn có thể đặt câu hỏi về tài liệu này ngay lập tức.")
                        uploaded_via_api = True
                except Exception:
                    pass

                # 2. Fallback nạp trực tiếp nếu backend offline
                if not uploaded_via_api:
                    try:
                        target_dir = Path(settings.rag_document_dir).resolve()
                        target_dir.mkdir(parents=True, exist_ok=True)
                        save_path = target_dir / uploaded_file.name
                        with open(save_path, "wb") as f:
                            f.write(file_bytes)

                        result = ingest_directory(directory=target_dir)
                        st.success(
                            f"✅ Đã nạp trực tiếp thành công **{result.get('chunks')} chunks** "
                            f"từ **{result.get('files')} tài liệu**!"
                        )
                        st.info("Bây giờ bạn có thể đặt câu hỏi về tài liệu này.")
                    except Exception as ex:
                        st.error(f"Lỗi nạp tài liệu: {ex}")

    st.divider()
    st.caption(f"Default model: `{settings.default_model}`")
    st.caption(f"Selected role: `{selected_model_role.split()[0]}`")
    st.caption("Mini AI Assistant v2.0 · ITSS Intern AI")


# ---------------------------------------------------------
# HÀM HIỂN THỊ CHI TIẾT TIN NHẮN CHAT
# ---------------------------------------------------------
def render_chat_message(message: dict, idx: int | None = None):
    # Huy hiệu định tuyến nếu có
    if message.get("route_info"):
        st.caption(f"🎯 {message['route_info']}")

    # Nội dung câu trả lời chính
    st.markdown(message.get("content", ""))

    # Hiển thị SQL query (nếu là kết quả NLQ)
    if message.get("sql"):
        with st.expander("🔍 Xem câu lệnh SQL đã thực thi"):
            st.code(message["sql"], language="sql")
            if message.get("repaired"):
                st.warning("⚠️ Câu lệnh đã được hệ thống tự động sửa lỗi cú pháp trước khi chạy.")

    # Hiển thị bảng số liệu (nếu có dữ liệu bảng)
    if message.get("data") is not None:
        df = message["data"]
        if isinstance(df, pd.DataFrame):
            with st.expander("📋 Bảng số liệu chi tiết", expanded=True):
                if df.empty:
                    st.info("ℹ️ Truy vấn thành công nhưng không có dòng dữ liệu nào thỏa mãn.")
                else:
                    st.dataframe(df, use_container_width=True)

                    # Tự động vẽ biểu đồ cột nếu có cột số và chuỗi
                    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
                    text_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
                    if num_cols and text_cols and len(df) <= 50:
                        try:
                            x_col = text_cols[0]
                            y_col = num_cols[0]
                            chart_df = df.set_index(x_col)[[y_col]]
                            st.caption(f"📈 Biểu đồ cột: {y_col} theo {x_col}")
                            st.bar_chart(chart_df)
                        except Exception:
                            pass

    # Hiển thị nguồn trích dẫn (nếu là kết quả RAG)
    if message.get("sources"):
        with st.expander(f"📚 Nguồn trích dẫn ({len(message['sources'])} đoạn trích)"):
            for s_idx, src in enumerate(message["sources"], start=1):
                sec = f" | Mục: *{src.get('section')}*" if src.get("section") else ""
                page = f" | Trang: *{src.get('page_number')}*" if src.get("page_number") else ""
                st.markdown(
                    f"**{s_idx}. `{src.get('source')}`** (Độ tương đồng: `{src.get('score', 0):.3f}`){page}{sec}"
                )

    # Nút Feedback (👍 / 👎) sử dụng st.feedback native
    if message.get("role") == "assistant" and idx is not None:
        trace_id = message.get("trace_id")
        already_voted = idx in st.session_state.feedback_given

        if already_voted:
            v = st.session_state.feedback_given[idx]
            trace_str = f" | Trace: `{trace_id[:16]}...`" if trace_id else ""
            st.caption(f"Đã phản hồi: {'👍 Hài lòng' if v == 1.0 else '👎 Chưa đúng'}{trace_str}")
        else:
            fb = st.feedback("thumbs", key=f"fb_{idx}")
            if fb is not None:
                score = 1.0 if fb == 1 else 0.0
                submit_feedback(idx, score=score, trace_id=trace_id)
                st.rerun()
            elif trace_id:
                st.caption(f"Trace ID: `{trace_id}`")


# ---------------------------------------------------------
# GIAO DIỆN CHÍNH (CHAT)
# ---------------------------------------------------------
st.header("🤖 Mini AI Assistant")
st.markdown(
    "*Hệ thống trợ lý AI hỗ trợ tra cứu chính sách doanh nghiệp (RAG) và phân tích số liệu kinh doanh (NLQ).*"
)

# 1. Hiển thị toàn bộ lịch sử hội thoại hiện có
for idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message.get("role", "user")):
        render_chat_message(message, idx=idx)

# 2. Xử lý nhập câu hỏi mới từ thanh chat_input
prompt = st.chat_input("Nhập câu hỏi của bạn (ví dụ: 'Chính sách đổi trả hàng?' hoặc 'Top 5 sản phẩm bán chạy nhất?')...")

if prompt:
    # A. Hiển thị câu hỏi của user và lưu vào session
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # B. Xác định luồng xử lý
    target_pipeline = "rag"
    route_info_str = None

    if "Tự động thông minh" in mode:
        decision = route_question(prompt)
        if decision.category == RouteCategory.OUT_OF_SCOPE:
            target_pipeline = "out_of_scope"
            route_info_str = f"Phân loại: Ngoài phạm vi kinh doanh ({decision.reasoning})"
        elif decision.category == RouteCategory.NLQ:
            target_pipeline = "nlq"
            route_info_str = f"Tự động định tuyến: **Truy vấn CSDL (NLQ)** – {decision.reasoning}"
        else:
            target_pipeline = "rag"
            route_info_str = f"Tự động định tuyến: **Hỏi đáp tài liệu (RAG)** – {decision.reasoning}"
    elif "NLQ" in mode:
        target_pipeline = "nlq"
        route_info_str = "Chế độ thủ công: **Truy vấn CSDL (NLQ)**"
    else:
        target_pipeline = "rag"
        route_info_str = "Chế độ thủ công: **Hỏi đáp tài liệu (RAG)**"

    # C. Xử lý từng luồng phản hồi của trợ lý
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": API_KEY,
    }

    # -------------------------------------------------
    # LUỒNG 0: NGOÀI PHẠM VI (Chào hỏi / Hướng dẫn)
    # -------------------------------------------------
    if target_pipeline == "out_of_scope":
        out_of_scope_reply = (
            "Xin chào! Tôi là trợ lý AI chuyên về:\n"
            "- 📑 **Hỏi đáp tài liệu nội bộ (RAG)**: Chính sách bán hàng, đổi trả, bảo hành, lưu kho, vận chuyển.\n"
            "- 📊 **Truy vấn số liệu bán hàng (NLQ)**: Thống kê doanh thu, số lượng đơn hàng, top sản phẩm, khách hàng.\n\n"
            "Bạn hãy nhập câu hỏi liên quan đến tài liệu hoặc số liệu kinh doanh để được hỗ trợ tốt nhất nhé!"
        )
        assistant_msg = {
            "role": "assistant",
            "content": out_of_scope_reply,
            "route_info": route_info_str,
        }
        with st.chat_message("assistant"):
            render_chat_message(assistant_msg, idx=len(st.session_state.messages))
        st.session_state.messages.append(assistant_msg)

    # -------------------------------------------------
    # LUỒNG 1: HỎI ĐÁP TÀI LIỆU (RAG - STREAMING SSE)
    # -------------------------------------------------
    elif target_pipeline == "rag":
        payload = {
            "question": prompt,
            "session_id": st.session_state.session_id,
            "top_k": 5,
        }

        with st.chat_message("assistant"):
            if route_info_str:
                st.caption(f"🎯 {route_info_str}")

            with st.status("🔍 Đang tìm kiếm tài liệu RAG...", expanded=True) as status_box:
                st.write("1️⃣ Tiếp nhận câu hỏi & phân loại intent")
                st.write("2️⃣ Tìm kiếm ngữ cảnh liên quan (Hybrid Search: PGVector + BM25)")
                st.write("3️⃣ Khởi tạo luồng sinh phản hồi streaming...")

            answer_placeholder = st.empty()
            full_response = ""
            sources_captured = []
            trace_id_captured = None

            try:
                with httpx.Client(timeout=60.0) as client:
                    with client.stream(
                        "POST",
                        f"{BACKEND_URL}/api/rag/ask/stream",
                        json=payload,
                        headers=headers,
                    ) as response:
                        if response.status_code != 200:
                            err_msg = f"Lỗi máy chủ (HTTP {response.status_code}): {response.read().decode('utf-8', errors='ignore')}"
                            answer_placeholder.error(err_msg)
                            assistant_msg = {"role": "assistant", "content": err_msg, "route_info": route_info_str}
                        else:
                            for line in response.iter_lines():
                                line_str = line.strip()
                                if not line_str.startswith("data:"):
                                    continue
                                raw_data = line_str[5:].strip()
                                if raw_data == "[DONE]":
                                    break

                                try:
                                    event = json.loads(raw_data)
                                    ev_type = event.get("type")

                                    if ev_type == "token":
                                        full_response += event.get("token", "")
                                        answer_placeholder.markdown(full_response + " ▌")
                                    elif ev_type == "sources":
                                        sources_captured = event.get("sources", [])
                                        trace_id_captured = event.get("trace_id", trace_id_captured)
                                    elif ev_type == "no_answer":
                                        full_response = event.get("answer", "Tôi không tìm thấy thông tin phù hợp trong tài liệu.")
                                        trace_id_captured = event.get("trace_id", trace_id_captured)
                                        answer_placeholder.markdown(full_response)
                                    elif ev_type == "error":
                                        full_response += f"\n\n*Lỗi: {event.get('detail')}*"
                                        answer_placeholder.markdown(full_response)
                                except Exception:
                                    pass

                            answer_placeholder.markdown(full_response)
                            status_box.update(label="✅ Hoàn tất câu trả lời", state="complete", expanded=False)

                            assistant_msg = {
                                "role": "assistant",
                                "content": full_response,
                                "sources": sources_captured,
                                "trace_id": trace_id_captured,
                                "route_info": route_info_str,
                            }
                            # Hiển thị nguồn trích dẫn
                            if sources_captured:
                                with st.expander(f"📚 Nguồn trích dẫn ({len(sources_captured)} đoạn trích)"):
                                    for s_idx, src in enumerate(sources_captured, start=1):
                                        sec = f" | Mục: *{src.get('section')}*" if src.get("section") else ""
                                        page = f" | Trang: *{src.get('page_number')}*" if src.get("page_number") else ""
                                        st.markdown(
                                            f"**{s_idx}. `{src.get('source')}`** (Độ tương đồng: `{src.get('score', 0):.3f}`){page}{sec}"
                                        )

            except Exception as e:
                err_text = f"Không thể kết nối API RAG ({BACKEND_URL}): {e}"
                answer_placeholder.error(err_text)
                status_box.update(label="❌ Lỗi xử lý", state="error", expanded=False)
                assistant_msg = {"role": "assistant", "content": err_text, "route_info": route_info_str}

            st.session_state.messages.append(assistant_msg)

    # -------------------------------------------------
    # LUỒNG 2: TRUY VẤN DỮ LIỆU CSDL (NLQ)
    # -------------------------------------------------
    else:
        assistant_msg = None
        with st.chat_message("assistant"):
            if route_info_str:
                st.caption(f"🎯 {route_info_str}")

            with st.status("📊 Đang phân tích CSDL và sinh truy vấn SQL...", expanded=True) as status_box:
                st.write("1️⃣ Phân tích schema 9 bảng CSDL & liên kết thực thể (Schema Linker)")
                st.write("2️⃣ Sinh câu lệnh SQL với Text-to-SQL Guard an toàn")
                st.write("3️⃣ Thực thi câu lệnh SQL chỉ đọc trên PostgreSQL & trích xuất bảng số liệu")

                payload = {
                    "question": prompt,
                    "session_id": st.session_state.session_id,
                    "user_id": "streamlit_user",
                }

                try:
                    resp = httpx.post(
                        f"{BACKEND_URL}/api/nlq/ask",
                        json=payload,
                        headers=headers,
                        timeout=60.0,
                    )
                    if resp.status_code != 200:
                        err_msg = f"Lỗi máy chủ NLQ (HTTP {resp.status_code}): {resp.text}"
                        status_box.update(label="❌ Lỗi truy vấn CSDL", state="error", expanded=False)
                        assistant_msg = {"role": "assistant", "content": err_msg, "route_info": route_info_str}
                    else:
                        data = resp.json()
                        answer_text = data.get("answer", "Đã thực thi truy vấn thành công.")
                        sql_query = data.get("sql", "")
                        columns = data.get("columns", [])
                        rows = data.get("rows", [])
                        repaired = data.get("repaired", False)
                        trace_id = data.get("trace_id")

                        df = None
                        if columns and rows:
                            df = pd.DataFrame(rows, columns=columns)

                        status_box.update(label="✅ Hoàn tất truy vấn CSDL", state="complete", expanded=False)

                        assistant_msg = {
                            "role": "assistant",
                            "content": answer_text,
                            "sql": sql_query,
                            "data": df,
                            "repaired": repaired,
                            "trace_id": trace_id,
                            "route_info": route_info_str,
                        }

                except Exception as e:
                    err_text = f"Không thể kết nối API NLQ ({BACKEND_URL}): {e}"
                    status_box.update(label="❌ Lỗi kết nối CSDL", state="error", expanded=False)
                    assistant_msg = {"role": "assistant", "content": err_text, "route_info": route_info_str}

            # Hiển thị kết quả ra ngoài status_box để người dùng đọc được ngay
            if assistant_msg:
                st.markdown(assistant_msg.get("content", ""))
                if assistant_msg.get("sql"):
                    with st.expander("🔍 Xem câu lệnh SQL đã thực thi"):
                        st.code(assistant_msg["sql"], language="sql")
                        if assistant_msg.get("repaired"):
                            st.warning("⚠️ Câu lệnh đã được hệ thống tự động sửa lỗi cú pháp trước khi chạy.")
                if assistant_msg.get("data") is not None:
                    df = assistant_msg["data"]
                    if isinstance(df, pd.DataFrame) and not df.empty:
                        with st.expander("📋 Bảng số liệu chi tiết", expanded=True):
                            st.dataframe(df, use_container_width=True)
                            num_cols = df.select_dtypes(include=["number"]).columns.tolist()
                            text_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
                            if num_cols and text_cols and len(df) <= 50:
                                try:
                                    x_col = text_cols[0]
                                    y_col = num_cols[0]
                                    chart_df = df.set_index(x_col)[[y_col]]
                                    st.caption(f"📈 Biểu đồ cột: {y_col} theo {x_col}")
                                    st.bar_chart(chart_df)
                                except Exception:
                                    pass
                if assistant_msg.get("trace_id"):
                    st.caption(f"Trace ID: `{assistant_msg['trace_id']}`")

        if assistant_msg:
            st.session_state.messages.append(assistant_msg)
