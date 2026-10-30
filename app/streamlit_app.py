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

import httpx
import pandas as pd
import streamlit as st

# Đảm bảo đường dẫn import tới app core
CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.core.config import settings
# Router logic được điều phối tập trung tại backend (/api/ask)


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


def format_api_error(status_code: int, headers, body: str, label: str) -> str:
    """Thông báo lỗi thân thiện kèm Trace ID, không dán raw body vào UI."""
    detail = ""
    try:
        payload = json.loads(body) if body else {}
        detail = payload.get("detail") or payload.get("answer") or ""
        if isinstance(detail, list):
            detail = "; ".join(
                str(item.get("msg", item)) if isinstance(item, dict) else str(item)
                for item in detail
            )
    except Exception:
        detail = ""

    trace_id = headers.get("X-Trace-Id") if headers is not None else None
    message = f"{label} (HTTP {status_code})"
    if detail:
        message += f": {detail}"
    if trace_id:
        message += f"\n\n🔎 Mã tra cứu (Trace ID): `{trace_id}`"
    else:
        message += "\n\n🔎 Mã tra cứu: không có (Langfuse chưa được cấu hình trên máy chủ)."
    return message


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


def render_data_chart(df: pd.DataFrame) -> None:
    """Tự động trực quan hóa dữ liệu bảng với nhãn trục X xoay ngang (labelAngle=0)."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty or len(df) > 50:
        return

    plot_df = df.copy()

    # 1. Chuyển đổi các cột số bị lưu dưới dạng chuỗi/Decimal sang số thực
    for col in plot_df.columns:
        converted = pd.to_numeric(plot_df[col], errors="coerce")
        if converted.notna().sum() > 0 and (converted.notna().sum() / len(plot_df)) >= 0.8:
            plot_df[col] = converted

    # 2. Nhận diện cột thước đo (metrics) và cột chiều phân tích (dimensions)
    numeric_cols = plot_df.select_dtypes(include=["number"]).columns.tolist()
    if not numeric_cols:
        return

    id_keywords = ["id", "ma_", "code", "stt", "year", "month", "day", "nam", "thang", "ngay", "quy", "quarter"]
    metric_cols = [c for c in numeric_cols if not any(k in c.lower() for k in id_keywords)]

    if not metric_cols:
        if len(numeric_cols) >= 2:
            dim_col = numeric_cols[0]
            val_col = numeric_cols[-1]
        else:
            return
    else:
        val_col = metric_cols[0]
        non_metric_cols = [c for c in plot_df.columns if c != val_col]
        text_cols = plot_df.select_dtypes(include=["object", "string", "category", "datetime"]).columns.tolist()
        if text_cols:
            dim_col = text_cols[0]
        elif non_metric_cols:
            dim_col = non_metric_cols[0]
        else:
            return

    plot_df[dim_col] = plot_df[dim_col].astype(str)

    try:
        import altair as alt

        st.caption(f"📈 Biểu đồ cột: **{val_col}** theo **{dim_col}**")
        chart = (
            alt.Chart(plot_df)
            .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, color="#38bdf8")
            .encode(
                x=alt.X(
                    f"{dim_col}:N",
                    title=dim_col,
                    axis=alt.Axis(
                        labelAngle=0,       # Đảm bảo các con số/nhãn ở dưới luôn xoay ngang (0 độ), không bị quay dọc
                        labelOverlap=False,
                        labelLimit=120,
                    ),
                    sort=None,
                ),
                y=alt.Y(
                    f"{val_col}:Q",
                    title=val_col,
                    axis=alt.Axis(format="~s"),
                ),
                tooltip=[
                    alt.Tooltip(f"{dim_col}:N", title=dim_col),
                    alt.Tooltip(f"{val_col}:Q", title=val_col, format=","),
                ],
            )
            .properties(height=320)
        )
        st.altair_chart(chart, use_container_width=True)
    except Exception:
        try:
            chart_df = plot_df.set_index(dim_col)[[val_col]]
            st.caption(f"📈 Biểu đồ cột: {val_col} theo {dim_col}")
            st.bar_chart(chart_df)
        except Exception:
            pass


# ---------------------------------------------------------
# HÀM GỬI FEEDBACK VỀ BACKEND / LANGFUSE
# ---------------------------------------------------------
def submit_feedback(msg_idx: int, score: float, trace_id: str | None = None, comment: str = ""):
    """Gửi đánh giá người dùng qua API duy nhất /api/feedback (gắn trace_id)."""
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
    try:
        resp = httpx.post(f"{BACKEND_URL}/api/feedback", json=payload, headers=headers, timeout=5.0)
        if resp.status_code == 200:
            data = resp.json()
            recorded = data.get("recorded", False)
        elif resp.status_code == 422:
            st.error(
                format_api_error(
                    resp.status_code, resp.headers, resp.text,
                    "Lỗi xác thực dữ liệu feedback",
                )
            )
    except Exception as exc:
        st.warning(f"Không thể kết nối đến máy chủ backend để gửi phản hồi: {exc}")

    st.session_state.feedback_given[msg_idx] = score
    if recorded:
        st.toast(f"✅ Đã ghi nhận phản hồi {'👍 Hài lòng' if score == 1.0 else '👎 Chưa đúng'} về Langfuse!", icon="🎉")
    else:
        st.toast("ℹ️ Phản hồi chưa được lưu (Langfuse chưa được cấu hình trên máy chủ).", icon="⚠️")


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

                # 1. Gửi qua endpoint /api/rag/upload của Backend (cô lập corpus 'user')
                try:
                    upload_resp = httpx.post(
                        f"{BACKEND_URL}/api/rag/upload",
                        headers={"X-API-Key": API_KEY},
                        files={"file": (uploaded_file.name, file_bytes, uploaded_file.type or "application/octet-stream")},
                        data={"session_id": st.session_state.session_id},
                        timeout=120.0,
                    )
                    if upload_resp.status_code == 200:
                        data = upload_resp.json()
                        chunks = data.get("chunks_indexed") or data.get("chunks", 0)
                        fname = data.get("display_name") or data.get("filename") or uploaded_file.name
                        st.success(
                            f"✅ Đã nạp thành công **{chunks} chunks** "
                            f"cho tài liệu **{fname}** (kho 'user')!"
                        )
                        st.info("Bây giờ bạn có thể đặt câu hỏi về tài liệu này ngay lập tức.")
                        uploaded_via_api = True
                    else:
                        st.error(
                            format_api_error(
                                upload_resp.status_code,
                                upload_resp.headers,
                                upload_resp.text,
                                "❌ Backend từ chối tải tệp",
                            )
                        )
                        uploaded_via_api = True
                except Exception as ex:
                    st.warning(f"⚠️ Không thể kết nối tới Backend API ({ex}).")

                # 2. Thông báo nếu không qua được API
                if not uploaded_via_api:
                    st.error("Không thể tải tài liệu lên backend. Vui lòng kiểm tra lại dịch vụ backend.")

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

                    # Tự động vẽ biểu đồ cột với nhãn xoay ngang
                    render_data_chart(df)

    # Hiển thị nguồn trích dẫn (nếu là kết quả RAG)
    if message.get("sources"):
        with st.expander(f"📚 Nguồn trích dẫn ({len(message['sources'])} đoạn trích)"):
            for s_idx, src in enumerate(message["sources"], start=1):
                sec = f" | Mục: *{src.get('section')}*" if src.get("section") else ""
                page = f" | Trang: *{src.get('page_number')}*" if src.get("page_number") else ""
                st.markdown(
                    f"**{s_idx}. `{src.get('source')}`** (Độ tương đồng: `{src.get('score', 0):.3f}`){page}{sec}"
                )

    # Nút Feedback (👍 / 👎) sử dụng st.feedback native; chỉ hiển thị khi có
    # trace_id để phản hồi luôn gắn được với một trace trên Langfuse.
    if message.get("role") == "assistant" and idx is not None and message.get("trace_id"):
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

    # B. Xử lý phản hồi từ trợ lý
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": API_KEY,
    }
    # Chế độ thủ công (RAG/NLQ) không đi qua router nên không có thông tin route.
    route_info_str = ""

    # -------------------------------------------------
    # LUỒNG 1: TỰ ĐỘNG THÔNG MINH QUA BACKEND ORCHESTRATOR (/api/ask)
    # -------------------------------------------------
    if "Tự động thông minh" in mode:
        payload = {
            "question": prompt,
            "session_id": st.session_state.session_id,
            "user_id": "streamlit_user",
            "mode": "auto",
            "top_k": 5,
        }
        with st.chat_message("assistant"):
            with st.status("🎯 Đang phân tích và điều phối xử lý qua Backend...", expanded=True) as status_box:
                try:
                    resp = httpx.post(f"{BACKEND_URL}/api/ask", json=payload, headers=headers, timeout=60.0)
                    if resp.status_code != 200:
                        err_msg = format_api_error(
                            resp.status_code, resp.headers, resp.text,
                            "Lỗi máy chủ khi xử lý câu hỏi",
                        )
                        status_box.update(label="❌ Lỗi xử lý", state="error", expanded=False)
                        st.error(err_msg)
                        assistant_msg = {"role": "assistant", "content": err_msg}
                    else:
                        data = resp.json()
                        route = data.get("route")
                        status_val = data.get("status", "ok")
                        ans = data.get("answer", "")
                        sql_query = data.get("sql", "")
                        columns = data.get("columns", [])
                        rows = data.get("rows", [])
                        sources_captured = data.get("sources", [])
                        fallback_from = data.get("fallback_from")
                        trace_id = data.get("trace_id")
                        reasoning = data.get("reasoning", "")
                        route_reason = data.get("route_reason")

                        df = None
                        if columns and rows:
                            df = pd.DataFrame(rows, columns=columns)

                        if fallback_from == "nlq":
                            status_box.update(label="ℹ️ Đã tự động chuyển sang tra cứu tài liệu (RAG)", state="complete", expanded=False)
                            st.info("ℹ️ CSDL không có dữ liệu này. Hệ thống đã tự động chuyển sang tra cứu tài liệu quy định.")
                        elif status_val == "restricted":
                            status_box.update(label="🔒 Dữ liệu bị hạn chế quyền riêng tư", state="complete", expanded=False)
                        elif status_val == "unsupported_action":
                            status_box.update(label="⚠️ Thao tác không được hỗ trợ", state="complete", expanded=False)
                        elif status_val == "out_of_scope":
                            if route_reason == "greeting":
                                status_box.update(label="👋 Chào bạn!", state="complete", expanded=False)
                            else:
                                status_box.update(label="ℹ️ Câu hỏi ngoài phạm vi", state="complete", expanded=False)
                        elif route == "nlq":
                            status_box.update(label="✅ Hoàn tất truy vấn CSDL", state="complete", expanded=False)
                        else:
                            status_box.update(label="✅ Hoàn tất câu trả lời", state="complete", expanded=False)

                        st.markdown(ans)
                        if sql_query:
                            with st.expander("🔍 Xem câu lệnh SQL đã thực thi"):
                                st.code(sql_query, language="sql")
                        if df is not None and not df.empty:
                            with st.expander("📋 Bảng số liệu chi tiết", expanded=True):
                                st.dataframe(df, use_container_width=True)

                                # Tự động vẽ biểu đồ cột với nhãn xoay ngang
                                render_data_chart(df)
                        if sources_captured:
                            with st.expander(f"📚 Nguồn trích dẫn ({len(sources_captured)} đoạn trích)"):
                                for s_idx, src in enumerate(sources_captured, start=1):
                                    sec = f" | Mục: *{src.get('section')}*" if src.get("section") else ""
                                    page = f" | Trang: *{src.get('page_number')}*" if src.get("page_number") else ""
                                    st.markdown(f"**{s_idx}. `{src.get('source')}`** (Độ tương đồng: `{src.get('score', 0):.3f}`){page}{sec}")

                        route_label = f"Tự động: {route.upper()}" + (f" (Fallback từ {fallback_from.upper()})" if fallback_from else "")
                        assistant_msg = {
                            "role": "assistant",
                            "content": ans,
                            "sql": sql_query,
                            "data": df,
                            "sources": sources_captured,
                            "trace_id": trace_id,
                            "route_info": f"{route_label} – {reasoning}" if reasoning else route_label,
                        }
                except Exception as e:
                    err_text = f"Không thể kết nối API ({BACKEND_URL}): {e}"
                    status_box.update(label="❌ Lỗi kết nối", state="error", expanded=False)
                    st.error(err_text)
                    assistant_msg = {"role": "assistant", "content": err_text}

            if assistant_msg:
                st.session_state.messages.append(assistant_msg)

    # -------------------------------------------------
    # LUỒNG 1: HỎI ĐÁP TÀI LIỆU (RAG - STREAMING SSE)
    # -------------------------------------------------
    elif "RAG" in mode:
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
                            body = response.read().decode("utf-8", errors="ignore")
                            err_msg = format_api_error(
                                response.status_code, response.headers, body,
                                "Lỗi máy chủ khi tra cứu tài liệu",
                            )
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
                        err_msg = format_api_error(
                            resp.status_code, resp.headers, resp.text,
                            "Lỗi khi truy vấn CSDL",
                        )
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

                        status_val = data.get("status", "ok")
                        if status_val == "restricted":
                            status_box.update(label="🔒 Dữ liệu bị hạn chế quyền riêng tư", state="complete", expanded=False)
                        elif status_val == "out_of_scope":
                            status_box.update(label="ℹ️ Câu hỏi ngoài phạm vi CSDL", state="complete", expanded=False)
                        elif status_val == "unsupported_action":
                            status_box.update(label="⚠️ Thao tác không được hỗ trợ", state="complete", expanded=False)
                        elif not sql_query and not rows:
                            status_box.update(label="ℹ️ Không có truy vấn phù hợp", state="complete", expanded=False)
                        else:
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
                            render_data_chart(df)
                if assistant_msg.get("trace_id"):
                    st.caption(f"Trace ID: `{assistant_msg['trace_id']}`")

        if assistant_msg:
            st.session_state.messages.append(assistant_msg)
