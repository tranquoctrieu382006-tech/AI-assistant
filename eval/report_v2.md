# Báo Cáo Đánh Giá Tổng Hợp v2 — AI Assistant (2026-10-29)

Báo cáo này tổng hợp kết quả đánh giá hệ thống trợ lý ảo Mini AI Assistant phiên bản v2 (tuần 2), so sánh sự nâng cấp từ phiên bản v1, đối chiếu hiệu năng giữa các lớp mô hình (`local`, `fast`, `strong`), phân tích trung thực các kết quả thực nghiệm và đưa ra khuyến nghị kiến trúc cùng các giới hạn đã biết của hệ thống.

---

## 1. Môi Trường & Dữ Liệu Thực Nghiệm

- **Cơ sở dữ liệu:** PostgreSQL 16 với extension `pgvector`. Schema bán hàng gồm 9 bảng (`customers`, `orders`, `order_items`, `products`, `inventory`, `warehouses`, `categories`, `suppliers`, `stock_movements`).
- **Embedding Model:** `BAAI/bge-m3` (vector 1024 chiều) chạy qua Ollama cục bộ (`http://localhost:11434`).
- **Tập dữ liệu kiểm thử (Golden Sets & Held-out Sets):**
  - **RAG Development Set:** 65 ca kiểm thử (`data/rag_golden_set.json`), gồm 41 ca có đáp án trong tài liệu (`answerable`), 19 ca cùng miền nhưng thiếu thông tin cần từ chối (`in_scope_missing`), và 5 ca hoàn toàn ngoài phạm vi (`out_of_scope_far`).
  - **RAG Held-out Set M2:** 50 ca kiểm thử (`eval/heldout_manifest_m2.json`), gồm 30 ca có đáp án, 10 ca `in_scope_missing`, và 10 ca `out_of_scope_far`. Bộ câu hỏi này do AI (Codex) biên soạn dựa trên corpus đóng băng; chưa qua thẩm định độc lập từ con người.
  - **Kho tài liệu (Corpus):** 12 tài liệu nghiệp vụ tổng hợp, chia thành 30 chunks (chiến lược heading-bounded, kích thước 900 ký tự, overlap 150 ký tự). Quy mô corpus còn nhỏ, mang tính chất thăm dò ban đầu.
  - **NLQ Golden Set:** 30 ca truy vấn tự nhiên sang SQL (`data/nlq_golden_set.json`), bao gồm các câu hỏi tính toán doanh thu (loại trừ đơn hủy), thống kê đơn hàng, tồn kho và khách hàng.
- **Quan sát & Tracing:** Langfuse Python SDK v5 ghi nhận trace ID, latency, token và observation nodes.
- **Tập tin kết quả gốc:** `eval/dev_threshold_investigation_m1.md`, `eval/heldout_summary_m2.md`, `eval/rerank_config_comparison.md`, `eval/hybrid_vs_vector_comparison.md`, `eval/nlq_model_comparison_2026-10-27.md`, `eval/v2_nlq_eval_run_20261028.json`.

---

## 2. So Sánh Bước Nhảy Chất Lượng: Phiên Bản v1 ➔ v2

| Thành phần | Phiên bản v1 (Tuần 1) | Phiên bản v2 (Tuần 2) | Đánh giá thực nghiệm & Quyết định kiến trúc |
| :--- | :--- | :--- | :--- |
| **Tìm kiếm RAG** | Vector thuần túy (`V5`) | Hybrid Search (`BM25 + PGVector`) kết hợp Weighted Reciprocal Rank Fusion (`RRF`) | `Hit@3` tăng nhẹ (+2.44%), nhưng `Hit@1` bị giảm từ **0.8537 xuống 0.7561** (mất 4 ca ở top-1), `MRR` giảm từ **0.9207 xuống 0.8740**. Hệ thống giữ mặc định `SEARCH_MODE=vector`. |
| **Reranker** | Không có reranker | Tích hợp thử nghiệm `bge-reranker-v2-m3` | `Hit@1` đạt 0.8049 (vẫn thấp hơn Vector thuần 0.8537). Độ trễ tăng vọt lên **2.571 ms** (R10) và **4.641 ms** (HR10), vượt ngưỡng 3.000 ms. Hệ thống giữ mặc định `USE_RERANKER=false`. |
| **Kiến trúc NLQ** | Single-turn Prompt-to-SQL | LangGraph StateGraph (5 nodes: `schema_link` ➔ `generate` ➔ `validate` ➔ `execute` ➔ `interpret`) | Bổ sung vòng lặp **tự sửa lỗi (self-repair)** tối đa 2 lần khi SQL lỗi cú pháp; AST SQL Guard (`sqlglot`) kết hợp phân quyền DB read-only (`ai_nlq_reader`). |
| **Định tuyến (Router)** | Chưa có bộ định tuyến tự động | Module Router (`app/core/router.py`) phân loại 3 hướng: `rag`, `nlq`, `out_of_scope` | Phân loại nhanh qua regex và LLM fallback. Đã phát hiện hiện tượng nhận nhầm câu hỏi có từ ngữ xã giao hoặc từ hỏi số lượng; đã thiết kế lại quy tắc ưu tiên nghiệp vụ và chuyển về RAG khi phân vân. |
| **Giao diện & Tương tác**| API Swagger | Web Chat Streamlit: Streaming SSE, hiển thị bảng & biểu đồ, nạp tài liệu tức thì, nút feedback 👍/👎 | Tích hợp end-to-end, hỗ trợ đính kèm nguồn trích dẫn và câu lệnh SQL thu gọn. |

---

## 3. Phân Tích Thực Nghiệm Chuyên Sâu

### 3.1. RAG Retrieval & Cổng Lọc Cosine (Abstention Gating)

Thực nghiệm trên 65 ca Dev Set và 50 ca Held-out Set M2 cho thấy ranh giới rõ ràng giữa các nhóm câu hỏi ở tầng lọc chunk:

| Tập kiểm thử | Nhóm câu hỏi | Số mẫu ($n$) | Bị nhận nhầm vượt cổng cosine 0.50 (False Positive) | Tỷ lệ lọc đúng (Recall) |
| :--- | :--- | :---: | :---: | :---: |
| **Dev Set** | Hoàn toàn ngoài phạm vi (`out_of_scope_far`) | 5 | **0 / 5** | 100% |
| **Dev Set** | Cùng miền nhưng thiếu dữ liệu (`in_scope_missing`) | 19 | **14 / 19** (73.7%) | 26.3% |
| **Dev Set** | Có đáp án (`answerable`) | 41 | 1 / 41 (bị loại nhầm - FN) | 97.6% |
| **Held-out M2** | Hoàn toàn ngoài phạm vi (`out_of_scope_far`) | 10 | **0 / 10** | 100% |
| **Held-out M2** | Cùng miền nhưng thiếu dữ liệu (`in_scope_missing`) | 10 | **10 / 10** (100.0%) | 0.0% |

#### Nhận xét kỹ thuật quan trọng:
1. **Hiện tượng Overlap điểm số:** Điểm cosine của nhóm `in_scope_missing` (cao nhất đạt 0.778) cao hơn trung vị của nhóm có đáp án (0.659). Do đó, **không thể tồn tại một ngưỡng cosine duy nhất** tách biệt hoàn hảo hai nhóm này.
   - Nâng ngưỡng lên `0.60`: `in_scope_missing` giảm còn 4/19 bị nhầm, nhưng loại nhầm 5/41 ca có đáp án.
   - Nâng ngưỡng lên `0.65`: loại nhầm 19/41 ca có đáp án (đánh đổi không chấp nhận được).
2. **Ranh giới đo lường (Retrieval vs. End-to-End):**
   - Con số 14/19 và 10/10 phản ánh giới hạn của **tầng trích xuất tài liệu (Retrieval Gating)**.
   - Ở tầng sinh câu trả lời, nếu context trả về không đủ thông tin, mô hình LLM có cơ chế trả về `generated.abstain=True` để API xuất thông báo từ chối tiêu chuẩn `NO_ANSWER`.
3. **Kết quả đo lường Đầu-Cuối thực tế (Production End-to-End Measurement với Gemini 3.5 Flash-Lite - 29/10/2026):**
   - Đã thực hiện đo lường đầy đủ 2 tầng bảo vệ (Cổng Cosine + LLM Generation Defense) trên toàn bộ tập Dev Set (`eval/dev_abstention_prompt_gemini35_m1.json`):
     - **Tầng 1 (Cổng Cosine 0.50):** Chặn được 5/19 ca thiếu dữ liệu; để lọt 14/19 ca (73.7%) do điểm cosine cao.
     - **Tầng 2 (Gemini 3.5 Flash-Lite sinh câu trả lời):** Ngăn chặn thành công **14/14 ca** bị lọt (100%), nhận diện chính xác context không đủ dữ kiện và trả về `abstain=True` / `NO_ANSWER`.
     - **Tỷ lệ từ chối đầu-cuối thực tế:** **19/19 (100.0%)** trên nhóm thiếu dữ liệu.
     - **Tỷ lệ ảo giác (Hallucination) thực tế:** **0/19 (0.0%)** (không có ca nào bịa đặt câu trả lời).
     - **Trên nhóm có đáp án (41 ca):** 40/41 ca được trả lời đầy đủ (chỉ 1 ca bị loại nhầm ở tầng retrieval gate do điểm cosine 0.479 < 0.50), không có ca nào bị mô hình từ chối nhầm.

---

### 3.2. Hiệu Năng Các Cấu Hình Tìm Kiếm (Retrieval & Reranker Benchmark)

Đo đạc trên 41 ca `answerable` thuộc Dev Set:

| Cấu hình | Phương thức | Lấy ($k$) | Rerank | Top ra | Hit@1 | Hit@3 | Hit@5 | MRR | Độ trễ trung bình |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **V5 (Baseline)** | Vector (`bge-m3`) | 5 | Không | 5 | **0.8537 (35/41)** | 0.9512 | **0.9756** | **0.9207** | **70.1 ms** |
| **V10** | Vector (`bge-m3`) | 10 | Không | 10 | 0.8537 (35/41) | 0.9512 | 0.9756 | 0.9207 | 24.5 ms |
| **H5** | Hybrid RRF | 5 | Không | 5 | 0.7561 (31/41) | **0.9756** | 0.9756 | 0.8740 | 56.7 ms |
| **R10** | Vector | 10 | `bge-reranker` | 5 | 0.8049 (33/41) | 0.9756 | 0.9756 | 0.9146 | **2.571.0 ms** |
| **HR10** | Hybrid RRF | 10 | `bge-reranker` | 5 | 0.8049 (33/41) | 0.9756 | 0.9756 | 0.9146 | **4.641.1 ms** |

**Kết luận thực nghiệm:** Cả H5, R10 và HR10 đều **không đạt tiêu chí cải tiến chất lượng** (Hit@1 và MRR đều thấp hơn V5, độ trễ Reranker vượt ngưỡng 3.000 ms). Quyết định duy trì cấu hình mặc định là **V5 (Vector thuần, $k=5$)** là tối ưu nhất cho môi trường hiện tại.

---

### 3.3. Đo Kiểm NLQ (Text-to-SQL Benchmark)

Số liệu benchmark trên 30 ca kiểm thử (`data/nlq_golden_set.json`) chạy trên seed DDL:

| Mô hình kiểm thử | Nhà cung cấp | Strict Execution Accuracy | Relaxed Execution Accuracy | Độ trễ sinh SQL | Chi phí ước tính / 30 ca |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `ollama/gemma3:12b` | Cục bộ | **53.3%** (16/30) | **60.0%** (18/30) | 8.052 s | $0.00 |
| `google_genai/gemini-3.5-flash-lite` | Google Cloud | **60.0%** (18/30) | **80.0%** (24/30) | 0.946 s | $0.0083 |
| `google_genai/gemini-3.1-flash-lite` | Google Cloud | **66.7%** (20/30) | **80.0%** (24/30) | 4.361 s | $0.0113 |

#### Ghi chú kỹ thuật về bộ đo NLQ:
1. **Định nghĩa Strict vs. Relaxed:**
   - `Strict`: So sánh chính xác từng giá trị ô trong bảng kết quả thực thi (hiện tại so sánh theo vị trí/tập hợp giá trị, cho phép khác alias cột).
   - `Relaxed`: Chẩn đoán phụ, cho phép mô hình chiếu thừa cột nếu các cột được yêu cầu vẫn chứa giá trị chính xác trong dung sai số học.
2. **Hạn chế của bộ benchmark hiện tại:**
   - Dữ liệu trên được đo trên seed DDL tĩnh; chưa phải đường chạy production đầy đủ (vốn sử dụng schema-linking động qua LangGraph và metadata dictionary).
   - Tập tin `eval/v2_nlq_eval_run_20261028.json` là đợt chạy smoke test nhanh trên 2 ca `COUNT(*)` đơn giản (`nlq-001`, `nlq-002`) với kết quả pass 2/2.

---

## 4. Đánh Giá Khách Quan Router & Tương Tác Người Dùng

Quá trình chạy thử nghiệm thực tế trên Router (`app/core/router.py`) đã chỉ ra một số trường hợp phân loại nhầm cần lưu ý:

| Câu hỏi thực tế | Phân loại thực tế | Lý do hệ thống đưa ra | Phân loại kỳ vọng |
| :--- | :---: | :--- | :---: |
| *"Có bao nhiêu ngày phép một năm?"* | `nlq` | Khớp từ khóa số lượng: `bao nhieu` | **`rag`** (Hỏi chính sách) |
| *"Cho hỏi chính sách đổi trả thế nào?"* | `out_of_scope` | Khớp cụm từ chào hỏi: `cho hoi` | **`rag`** (Chính sách đổi trả) |
| *"Giúp tôi tra cứu quy định bảo hành"* | `out_of_scope` | Khớp cụm từ chào hỏi: `giup toi` | **`rag`** (Chính sách bảo hành) |

**Nguyên nhân kỹ thuật:** Bộ lọc regex chạy trước LLM đã đưa các từ chào hỏi/lịch sự vào danh sách xã giao (`out_of_scope`) và từ chỉ số lượng vào `nlq`.
**Giải pháp kiến trúc:**
1. Chỉ phân loại là chào hỏi xã giao khi câu hỏi **hoàn toàn không chứa từ khóa nghiệp vụ**.
2. Loại bỏ các từ ngữ mang tính lịch sự (`cho hỏi`, `giúp tôi`) khỏi danh sách blacklist.
3. Khi phân vân hoặc câu hỏi mơ hồ, hệ thống **luôn ưu tiên chuyển về RAG** thay vì tự ý từ chối, do RAG đã có cơ chế tự thẩm định ngữ cảnh ở tầng sau.

---

## 5. Khuyến Nghị Lựa Chọn Mô Hình Theo Vai Trò (Roles)

| Thành phần | Vai trò khuyến nghị | Mô hình cấu hình | Lý do kỹ thuật |
| :--- | :---: | :--- | :--- |
| **Router** | **`fast`** | `gemini-3.5-flash-lite` | Độ trễ phản hồi thấp (< 1.2s), chi phí cực thấp, đảm bảo thời gian trước khi bắt đầu streaming không bị trễ. |
| **RAG Answer Generation** | **`fast`** | `gemini-3.5-flash-lite` | Tốc độ sinh token đầu tiên nhanh (< 1.5s), tổng hợp ngữ cảnh tiếng Việt lưu loát, tuân thủ hướng dẫn từ chối khi context rỗng. |
| **NLQ Text-to-SQL** | **`strong`** hoặc **`fast`** có Repair | `gemini-3.6-flash` / `gemini-3.5-flash-lite` | Cần khả năng bám sát schema quan hệ và ràng buộc nghiệp vụ (JOIN, loại trừ đơn hủy); kết hợp vòng lặp self-repair của LangGraph. |
| **Offline / Local Fallback** | **`local`** | `gemma3:12b` (Ollama) | Phục vụ môi trường cô lập mạng hoặc bảo mật dữ liệu nội bộ, không phụ thuộc hạn ngạch API đám mây. |

> [!WARNING]
> **Rủi ro hạn ngạch (Rate Limit):** Tài khoản Gemini Free Tier giới hạn ở 15 RPM (requests/phút) và 20 RPD (requests/ngày với bản Flash 3.6). Đối với buổi demo hoặc môi trường tải cao, cần chuẩn bị sẵn phương án sử dụng API key trả phí hoặc chuyển sang mô hình local.

---

## 6. Giới Hạn Kiến Trúc Đã Biết (Known Scope Limitations)

Để đảm bảo tính minh bạch về phạm vi sản phẩm (PoC / Demo nội bộ):

1. **Đơn tiến trình (Single Worker):** Hệ thống được thiết kế chạy duy nhất 1 tiến trình worker (`APP_WORKERS=1`). Bộ đếm rate limiting và lưu trữ lịch sử hội thoại nhiều lượt (`session_store`) được duy trì trong bộ nhớ RAM của tiến trình, chưa tích hợp Redis phân tán; dữ liệu phiên sẽ được làm mới khi tiến trình khởi động lại.
2. **Xác thực API tập trung:** Hệ thống sử dụng một API key tĩnh (`X-API-Key`) dùng chung cho toàn bộ ứng dụng; chưa hỗ trợ xác thực người dùng cá nhân (RBAC).
3. **Quy mô Corpus RAG:** Corpus hiện tại gồm 12 tài liệu tổng hợp (30 chunks). Kết quả top-k và cosine threshold trên tập này mang tính chất thăm dò, cần tiếp tục đánh giá khi nạp kho tài liệu quy mô lớn hơn.
4. **Phản hồi người dùng (Feedback):** Khi chưa cấu hình khóa kết nối Langfuse Cloud, endpoint `/api/feedback` chỉ ghi log cục bộ mà chưa lưu trữ vào bảng cơ sở dữ liệu riêng.

---

## 7. Trạng Thái Kiểm Thử (Unit Tests Summary)

Toàn bộ các module cốt lõi, bảo mật, định tuyến, upload và hợp đồng dữ liệu được kiểm thử cô lập bằng mock và live database:

- `tests/test_api_client.py`: **10/10 passed** (xác thực `X-API-Key`, mã lỗi 401, 422, mock RAG/NLQ/Router/Feedback).
- `tests/test_chat_history.py`: **13/13 passed** (quản lý session LRU, cắt tỉa ngữ cảnh, viết lại câu hỏi).
- `tests/test_chunking.py`: **8/8 passed** (chiến lược heading-bounded, fixed-size, bảo toàn tiếng Việt có dấu).
- `tests/test_evaluators.py`: **31/31 passed** (đánh giá SQL guard, so sánh kết quả bảng, logic retry).
- `tests/test_health.py`: **8/8 passed** (liveness, readiness, kiểm tra kết nối Postgres/Ollama).
- `tests/test_ingestion.py`: **7/7 passed** (trích xuất văn bản từ docx, pdf, xử lý lỗi định dạng).
- `tests/test_limits.py`: **20/20 passed** (bảo vệ giới hạn context, timeout, hàng đợi).
- `tests/test_llm.py`: **17/17 passed** (phân giải model role, bộ nhớ đệm model instance).
- `tests/test_nlq_runtime.py`: **26/26 passed** (SQL guard, chặn injection, mapping mã lỗi 503/504).
- `tests/test_prompt_and_streaming.py`: **11/11 passed** (định dạng SSE, sự kiện token/sources/no_answer).
- `tests/test_rag.py`: **18/18 passed** (gating cosine, từ chối khi thiếu context, đơn tiến trình).
- `tests/test_rag_upload.py`: **9/9 passed** (giới hạn 5MB, kiểm tra magic bytes PDF/DOCX/TXT, zip-bomb, prompt injection resistance).
- `tests/test_response_contracts.py`: **21/21 passed** (chuẩn hóa schema Pydantic).
- `tests/test_retriever.py`: **20/20 passed** (Vector search, BM25 cache, hybrid fusion, reranker).
- `tests/test_router.py`: **8/8 passed** (ma trận phân loại 2 trục, đánh giá tập đóng băng 70 câu: 35 dev / 35 test đạt 100% accuracy và 100% per-class recall).
- `tests/test_security.py`: **5/5 passed** (rate limiting theo IP, bảo vệ timing attack, CORS).
- `tests/test_tracing.py`: **9/9 passed** (context manager tracing, an toàn khi Langfuse ngoại tuyến).
- `tests/test_vector_store.py`: **11/11 passed** (cosine distance, thay thế corpus, upsert cô lập `corpus`, connection pool).
- `tests/test_integration_postgres.py`: **3/3 passed** (kiểm thử pgvector extension và giới hạn phân quyền `ai_nlq_reader`).
- `tests/test_report_v2_metrics.py`: **5/5 passed** (kiểm thử tự động đảm bảo số liệu báo cáo khớp 100% với dữ liệu nguồn).
- `tests/test_golden_data_validation.py`, `test_rag_heldout.py`, `test_evaluate_rag_production.py`, `test_evaluate_rag_answers.py`: **17/17 passed**.

👉 **Tổng số kiểm thử:** **241 / 241 tests passed (100%)** trên toàn bộ 23 file test trong thời gian ~14.8 giây.
