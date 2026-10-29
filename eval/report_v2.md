# Báo Cáo Đánh Giá Tổng Hợp v2 — AI Assistant (2026-10-28)

Báo cáo này tổng hợp kết quả đánh giá hệ thống trợ lý ảo Mini AI Assistant phiên bản v2 (tuần 2), so sánh sự nâng cấp từ phiên bản v1, hiệu năng giữa các lớp mô hình (Local, Fast, Strong) và đưa ra khuyến nghị kiến trúc cho từng thành phần (Router, RAG, NLQ).

---

## 1. Môi Trường & Dữ Liệu Thực Nghiệm

- **Hạ tầng cơ sở dữ liệu:** PostgreSQL 18.6 với extension `pgvector 0.8.7`. Database bán hàng gồm 9 bảng (`customers`, `orders`, `order_items`, `products`, `inventory`, `warehouses`, `categories`, `suppliers`, `stock_movements`).
- **Embedding Model:** `BAAI/bge-m3` (1024 chiều) chạy qua Ollama (`http://localhost:11434`).
- **Tập dữ liệu kiểm thử (Golden Sets):**
  - **RAG:** 65 ca kiểm thử (`data/rag_golden_set.json`), gồm 41 ca có đáp án trong tài liệu, 19 ca trong phạm vi nhưng thiếu dữ liệu (cần từ chối), và 5 ca hoàn toàn ngoài phạm vi.
  - **NLQ:** 30 ca kiểm thử câu hỏi tự nhiên sang SQL (`data/nlq_golden_set.json`), bao gồm các câu hỏi có JOIN nhiều bảng, GROUP BY, tính tổng doanh thu và loại trừ đơn hủy.
- **Quan sát & Tracing:** Toàn bộ request và điểm đánh giá được ghi nhận qua Langfuse Cloud SDK v5.
- **Tập tin kết quả thực nghiệm v2 trực tiếp:** `eval/v2_nlq_eval_run_20261028.json` (chạy thực tế qua `scripts/evaluate_nlq_model.py` trên hạ tầng cục bộ).

---

## 2. So Sánh Bước Nhảy Chất Lượng: Phiên Bản v1 ➔ v2

| Thành phần | Phiên bản v1 (Tuần 1) | Phiên bản v2 (Tuần 2) | Hiệu quả cải tiến |
| :--- | :--- | :--- | :--- |
| **Tìm kiếm RAG** | Chỉ dùng Vector đơn thuần (`V5`) | Hybrid Search (`BM25 + PGVector`) kết hợp Reciprocal Rank Fusion (`RRF`) | `Hit@3` tăng từ **0.9512 ➔ 0.9756** (+2.44%). Bắt chính xác các từ khóa mã sản phẩm, thuật ngữ viết tắt tiếng Việt. |
| **Reranker** | Không có reranker | Tích hợp `bge-reranker-v2-m3` qua ContextualCompression | Tinh chọn ngữ cảnh sát nhất trước khi đưa vào context prompt của LLM. |
| **Kiến trúc NLQ** | Single-turn Prompt-to-SQL một lượt | LangGraph StateGraph (5 nodes: `schema_link` ➔ `generate` ➔ `validate` ➔ `execute` ➔ `interpret`) | Có vòng lặp **tự sửa lỗi (self-repair)** tối đa 2 lần khi SQL lỗi cú pháp; an toàn 2 lớp (`sqlglot` + DB read-only user). |
| **Định tuyến (Router)** | Chưa có bộ định tuyến tự động | Module Router (`app/core/router.py`) phân loại 3 hướng: `rag`, `nlq`, `out_of_scope` | Đạt **100% độ chính xác** trên bộ test phân loại mẫu; hỗ trợ cả Fake LLM trong test. |
| **Giao diện & Tương tác**| Chỉ có API Swagger | Ứng dụng Web Chat Streamlit đầy đủ: Streaming SSE, bảng dữ liệu, biểu đồ, upload index tài liệu tức thì, nút feedback 👍/👎 | Người dùng cuối trải nghiệm mượt mà, theo dõi được nguồn trích dẫn và câu lệnh SQL. |

---

## 3. So Sánh Hiệu Năng Giữa Các Lớp Mô Hình (Model Benchmarks)

Thực nghiệm đo đạc độ trễ (latency), số lượng token và chi phí ước tính trên 3 vai trò mô hình (`local`, `fast`, `strong`):

| Vai trò (Role) | Mô hình cấu hình | Nhà cung cấp | Độ trễ trung bình | Tokens (In / Out) | Chi phí ước tính / 1.000 lượt | Đánh giá ưu / nhược điểm |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`fast`** | `gemini-3.5-flash-lite` | Google GenAI | **~1.1s** | 29 / 159 | **~$0.02** | ⚡ **Cực nhanh, siêu rẻ**, rất phù hợp cho streaming chat và định tuyến intent. |
| **`strong`** | `gemini-3.6-flash` / `gpt-4o` | Google / OpenAI | **~7.5s** | 29 / 1.104 | **~$0.35** | 🧠 **Lập luận sâu, cấu trúc câu trả lời giàu thông tin**, sinh SQL phức tạp chuẩn xác. |
| **`local`** | `gemma3:12b` | Ollama (Local) | **~60 - 90s** (CPU) | 29 / ~200 | **$0.00** | 🔒 **Bảo mật tuyệt đối, không tốn quota API**, nhưng đòi hỏi phần cứng GPU để đạt độ trễ thời gian thực. |

---

## 4. Đánh Giá LLM-as-a-Judge & Chấm Tay $\ge 20\%$ Mẫu (Langfuse Traces)

Hệ thống thiết lập bộ đánh giá kết hợp giữa tự động qua Langfuse LLM-as-a-Judge và chấm tay con người (Human-in-the-loop) với tỷ lệ $\ge 20\%$ trên tập Golden Sets:

### 4.1. Tiêu Chí & Cơ Chế Đánh Giá
- **Faithfulness (Độ trung thực)**: Đánh giá câu trả lời RAG có hoàn toàn dựa trên bằng chứng (evidence context) được trích xuất từ tài liệu hay không, loại trừ ảo giác (hallucination).
- **Relevance (Độ liên quan)**: Đánh giá câu trả lời có trực tiếp giải quyết đúng câu hỏi của người dùng, không lan man thừa thông tin.
- **Correctness (Độ chính xác SQL & Số liệu)**: So sánh kết quả bảng thực thi SQL trên database với ground truth mẫu (`nlq_golden_set.json`).

### 4.2. Kết Quả Thẩm Định Chấm Mẫu ($\ge 20\%$ Dataset)
- **Tập mẫu RAG (13/65 ca ~ 20%)**:
  - Faithfulness trung bình: **0.94 / 1.0** (các đoạn trích chính sách bảo hành, đổi trả được trích dẫn chính xác).
  - Relevance trung bình: **0.96 / 1.0**.
  - Tỷ lệ từ chối câu hỏi thiếu dữ liệu / ngoài phạm vi (Abstention Recall): **100%** (trả về đúng thông báo từ chối tiêu chuẩn, không bịa đặt số liệu).
- **Tập mẫu NLQ (6/30 ca ~ 20%)**:
  - SQL Syntax Validity: **100%** (vượt qua kiểm duyệt cú pháp AST của `sqlglot`).
  - Execution Correctness: **100%** trên tập mẫu (khớp chính xác số lượng sản phẩm, khách hàng, tổng doanh thu theo trạng thái đơn hàng).

---

## 5. Khuyến Nghị Lựa Chọn Mô Hình Cho Từng Thành Phần

Dựa trên số liệu thực nghiệm về độ trễ, chi phí và độ chính xác:

1. **Bộ định tuyến (Router Node):**
   - **Khuyến nghị:** Sử dụng role **`fast`** (hoặc rule-based regex fallback).
   - **Lý do:** Router cần đưa ra quyết định chuyển hướng trong vòng $< 1.2$ giây để không làm gián đoạn trải nghiệm người dùng trước khi bắt đầu stream câu trả lời.
2. **Hỏi đáp tài liệu (RAG Pipeline):**
   - **Khuyến nghị:** Sử dụng **`bge-m3`** cho Vector + BM25, kết hợp role **`fast`** để sinh văn bản trả lời.
   - **Lý do:** `bge-m3` xử lý ngữ nghĩa tiếng Việt đa ngữ cực tốt; mô hình `fast` stream SSE với tốc độ từ đầu tiên $< 1.5$s, đáp ứng trọn vẹn yêu cầu tra cứu chính sách hàng ngày.
3. **Truy vấn CSDL (NLQ Pipeline):**
   - **Khuyến nghị:** Sử dụng role **`strong`** (hoặc `fast` có cơ chế LangGraph retry).
   - **Lý do:** Text-to-SQL yêu cầu tuân thủ nghiêm ngặt schema và quan hệ khóa ngoại (JOIN). Mô hình `strong` có tỷ lệ sinh đúng cú pháp ngay từ lần đầu cao hơn, giảm số lần phải kích hoạt vòng lặp repair.

---

## 6. Kết Quả Kiểm Thử (Unit Tests & Độ Ổn Định)

Toàn bộ các module mới đều được bảo vệ bởi bộ test tự động độc lập mạng, không gọi LLM bên ngoài:
- `tests/test_router.py`: **7/7 test passed** (phân loại RAG, NLQ, Out-of-scope; kiểm thử với `FakeListChatModel` và regex fallback).
- `tests/test_response_contracts.py`: **6/6 test passed** (kiểm định tính toàn vẹn của Pydantic request/response schemas RAG/NLQ).
- `tests/test_api_client.py`: **8/8 test passed** (kiểm định bảo mật `X-API-Key`, mã lỗi 401, 422, mock route RAG, NLQ, feedback endpoint, và router classification endpoint qua `httpx.ASGITransport`).
- `tests/test_chunking.py`: **8/8 test passed** (kiểm thử chiến lược chunk_text, chunk_document_blocks theo heading & fixed size, xử lý tiếng Việt có dấu).
- **Tổng số test mới Ngày 29:** **29/29 pass 100%** trong ~2.6 giây.
- **Toàn bộ Test Suite hệ thống:** **219/219 test pass 100%** trên toàn bộ 22 files test trong 13.45 giây.

---

## 7. Kết Luận

Hệ thống Mini AI Assistant v2 đã đạt đầy đủ các tiêu chí Definition of Done (DoD) của Ngày 29:
1. Giao diện Streamlit hoạt động đa năng, kết nối mượt mà giữa RAG streaming và NLQ bảng biểu.
2. Nạp tài liệu mới và phản hồi người dùng tích hợp trực tiếp vào PGVector và Langfuse.
3. Báo cáo đánh giá v2 cung cấp đầy đủ số liệu so sánh, định lượng chi phí và định hướng cấu hình tối ưu sẵn sàng cho ngày bàn giao sản phẩm.
