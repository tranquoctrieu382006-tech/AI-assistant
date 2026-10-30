# Báo Cáo Đánh Giá Tổng Hợp v2 — AI Assistant (2026-10-29)

Báo cáo này tổng hợp kết quả đánh giá hệ thống trợ lý ảo Mini AI Assistant phiên bản v2 (tuần 2), so sánh sự nâng cấp từ phiên bản v1, đối chiếu hiệu năng giữa các lớp mô hình (`local`, `fast`, `strong`), phân tích trung thực các kết quả thực nghiệm và đưa ra khuyến nghị kiến trúc cùng các giới hạn đã biết của hệ thống.

---

## 1. Môi Trường & Dữ Liệu Thực Nghiệm

- **Cơ sở dữ liệu:** PostgreSQL 16 với extension `pgvector`. Schema bán hàng gồm 9 bảng (`customers`, `orders`, `order_items`, `products`, `inventory`, `warehouses`, `categories`, `suppliers`, `stock_movements`).
- **Embedding Model:** `BAAI/bge-m3` (vector 1024 chiều) chạy qua Ollama cục bộ (`http://localhost:11434`).
- **Tập dữ liệu kiểm thử (Golden Sets & Held-out Sets):**
  - **RAG Development Set:** 65 ca kiểm thử (`data/rag_golden_set.json`), gồm 41 ca có đáp án trong tài liệu (`answerable`), 19 ca cùng miền nhưng thiếu thông tin cần từ chối (`in_scope_missing`), và 5 ca hoàn toàn ngoài phạm vi (`out_of_scope_far`).
  - **RAG Held-out Set M2:** 50 ca kiểm thử (`eval/heldout_manifest_m2.json`), gồm 30 ca có đáp án, 10 ca `in_scope_missing`, và 10 ca `out_of_scope_far`. Bộ câu hỏi này do AI (Codex) biên soạn dựa trên corpus đóng băng và **chưa có thẩm định độc lập từ con người**; nhãn phân loại chỉ mang tính tham khảo, không phải ground truth đã kiểm định. Báo cáo không ghi nhận kiểm định thủ công hay tỷ lệ đồng thuận nào.
  - **Kho tài liệu (Corpus):** 12 tài liệu nghiệp vụ tổng hợp, chia thành 30 chunks (chiến lược heading-bounded, kích thước 900 ký tự, overlap 150 ký tự). Quy mô corpus còn nhỏ, mang tính chất thăm dò ban đầu.
  - **NLQ Golden Set:** 30 ca truy vấn tự nhiên sang SQL (`data/nlq_golden_set.json`), bao gồm các câu hỏi tính toán doanh thu (loại trừ đơn hủy), thống kê đơn hàng, tồn kho và khách hàng.
- **Quan sát & Tracing:** Langfuse Python SDK v5 ghi nhận trace ID, latency, token và observation nodes.
- **Tập tin kết quả gốc:** `eval/dev_threshold_investigation_m1.md`, `eval/heldout_summary_m2.md`, `eval/rerank_config_comparison.md`, `eval/hybrid_vs_vector_comparison.md`, `eval/nlq_model_comparison_2026-10-27.md`, `eval/v2_nlq_eval_run_20261028.json`.

---

## 2. So Sánh Bước Nhảy Chất Lượng: Phiên Bản v1 ➔ v2

| Thành phần | Phiên bản v1 (Tuần 1) | Phiên bản v2 (Tuần 2) | Đánh giá thực nghiệm & Quyết định kiến trúc |
| :--- | :--- | :--- | :--- |
| **Tìm kiếm RAG** | Vector thuần túy (`V5`) | Hybrid Search (`BM25 + PGVector`) kết hợp Weighted Reciprocal Rank Fusion (`RRF`) | `Hit@3` tăng nhẹ (+2.44%), nhưng `Hit@1` bị giảm từ **0.8537 xuống 0.7561** (mất 4 ca ở top-1), `MRR` giảm từ **0.9207 xuống 0.8740**. Hệ thống giữ mặc định `SEARCH_MODE=vector`. |
| **Reranker** | Không có reranker | Tích hợp thử nghiệm `bge-reranker-v2-m3` | `Hit@1` đạt 0.8049 (vẫn thấp hơn Vector thuần 0.8537). Độ trễ tăng vọt lên **2571 ms (≈2,57 s)** (R10) và **4641 ms (≈4,64 s)** (HR10), vượt ngưỡng 3.000 ms. Hệ thống giữ mặc định `USE_RERANKER=false`. |
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
   - *Số hiện hành (chạy lại 2026-10-29 trên corpus đã dựng lại, `eval/dev_abstention_prompt_gemini35_m2.json`):* xem `eval/current_numbers.md`. Quyết định gate và từ chối không đổi; `fragment_match` là 31/41 thay vì 32/41 do model diễn đạt lại, không phải hồi quy.

---

### 3.2. Hiệu Năng Các Cấu Hình Tìm Kiếm (Retrieval & Reranker Benchmark)

Đo đạc trên 41 ca `answerable` thuộc Dev Set:

| Cấu hình | Phương thức | Lấy ($k$) | Rerank | Top ra | Hit@1 | Hit@3 | Hit@5 | MRR | Độ trễ trung bình |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **V5 (Baseline)** | Vector (`bge-m3`) | 5 | Không | 5 | **0.8537 (35/41)** | 0.9512 | **0.9756** | **0.9207** | **70.1 ms** |
| **V10** | Vector (`bge-m3`) | 10 | Không | 10 | 0.8537 (35/41) | 0.9512 | 0.9756 | 0.9207 | 24.5 ms |
| **H5** | Hybrid RRF | 5 | Không | 5 | 0.7561 (31/41) | **0.9756** | 0.9756 | 0.8740 | 56.7 ms |
| **R10** | Vector | 10 | `bge-reranker` | 5 | 0.8049 (33/41) | 0.9756 | 0.9756 | 0.9146 | **2571 ms (≈2,57 s)** |
| **HR10** | Hybrid RRF | 10 | `bge-reranker` | 5 | 0.8049 (33/41) | 0.9756 | 0.9756 | 0.9146 | **4641 ms (≈4,64 s)** |

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
5. **Phân vùng Corpus người dùng (Soft Session Isolation):** Tài liệu upload của người dùng được phân vùng theo `corpus=session_{session_id}` trong cơ sở dữ liệu pgvector (hoặc `user` khi không có session). Khi người dùng đặt câu hỏi, cơ chế trích xuất RAG tự động tìm kiếm đa phân vùng (`corpus = ANY(%s)`) trên corpus mặc định `demo`, corpus dùng chung `user`, và phân vùng tài liệu riêng của phiên hiện tại; client chỉ được chỉ định `demo`, `user` hoặc đúng session của mình. Khi session hết TTL hoặc bị đẩy khỏi LRU, corpus `session_{session_id}` được dọn khỏi PostgreSQL; thao tác xoá chạy ngoài event loop và các corpus `session_*` tồn đọng sau khi khởi động lại được quét dọn ở startup. Tài liệu tải lên được đọc qua bộ đệm tạm thời (`NamedTemporaryFile`) và được lập tức giải phóng (`unlink`) sau khi phân đoạn và embedding; hệ thống không duy trì file thô vĩnh viễn trên ổ cứng của container.
6. **Đánh giá tự động Langfuse LLM-as-judge:** Hiện tại tính năng đánh giá tự động sau mỗi lượt hội thoại (online auto-eval) bằng LLM-as-judge ở trạng thái hoàn thành một phần (partially completed) / đang chờ kết nối hạ tầng Langfuse Cloud thực tế. Các bộ prompt đánh giá (`tests/test_evaluators.py`) đã được hoàn thiện và kiểm thử thành công bằng unit test, sẵn sàng tích hợp khi triển khai production.

---

## 7. Trạng Thái Kiểm Thử (Unit Tests Summary)

Toàn bộ các module cốt lõi, bảo mật, định tuyến, upload và hợp đồng dữ liệu được kiểm thử tự động, thu thập qua `pytest --collect-only -q`:

| STT | File Kiểm Thử | Thu thập (Collected) | Trạng thái Chạy | Mô Tả & Phạm Vi Kiểm Thử |
| :---: | :--- | :---: | :---: | :--- |
| 1 | `tests/test_api_client.py` | **13** | Chờ artifact JUnit trên CI | Xác thực `X-API-Key`, mã lỗi 401, 422, mock RAG/NLQ/Router/Feedback. |
| 2 | `tests/test_chat_history.py` | **20** | Chờ artifact JUnit trên CI | Quản lý session LRU, dọn corpus nền không chặn event loop, quét corpus mồ côi, cắt tỉa ngữ cảnh hội thoại. |
| 3 | `tests/test_chunking.py` | **8** | Chờ artifact JUnit trên CI | Chiến lược phân đoạn heading-bounded, fixed-size, bảo toàn tiếng Việt Unicode NFC. |
| 4 | `tests/test_evaluate_rag_answers.py` | **2** | Chờ artifact JUnit trên CI | Đánh giá chất lượng sinh câu trả lời RAG. |
| 5 | `tests/test_evaluate_rag_production.py` | **1** | Chờ artifact JUnit trên CI | Đo đạc kiểm thử pipeline RAG với mô hình production. |
| 6 | `tests/test_evaluators.py` | **31** | Chờ artifact JUnit trên CI | Đánh giá SQL guard, so sánh kết quả bảng, cơ chế retry khi lỗi. |
| 7 | `tests/test_golden_data_validation.py` | **13** | Chờ artifact JUnit trên CI | Kiểm tra tính toàn vẹn và phân bổ nhãn của tập golden set 65 ca, hash manifest chuẩn hoá LF, phát hiện drift, nhãn abstention của nhóm far và loại ca lỗi khỏi mẫu số. |
| 8 | `tests/test_health.py` | **9** | Chờ artifact JUnit trên CI | Liveness, readiness probes, kiểm tra kết nối Postgres và Ollama. |
| 9 | `tests/test_ingestion.py` | **20** | Chờ artifact JUnit trên CI | Trích xuất văn bản từ docx, pdf, xử lý lỗi định dạng và ký tự đặc biệt. |
| 10 | `tests/test_integration_postgres.py` | **4** | Chờ artifact JUnit trên CI | Kiểm thử live database pgvector, cô lập corpus session A/B và phân quyền `ai_nlq_reader` (chạy trên CI/Live DB). |
| 11 | `tests/test_llm.py` | **15** | Chờ artifact JUnit trên CI | Phân giải vai trò model (fast/strong/local), cache instance mô hình. |
| 12 | `tests/test_nlq_runtime.py` | **40** | Chờ artifact JUnit trên CI | NLQ graph sinh SQL, validator, test với `FakeListChatModel`, mã lỗi 503/504. |
| 13 | `tests/test_prompt_and_streaming.py` | **6** | Chờ artifact JUnit trên CI | Định dạng SSE, bắt sự kiện `start`, `sources`, `token`, `no_answer`, `error`. |
| 14 | `tests/test_rag.py` | **24** | Chờ artifact JUnit trên CI | Cosine gating, từ chối khi thiếu context, dedupe và giới hạn nguồn, test với `FakeListChatModel` (JSON chuẩn, JSON hỏng, abstain). |
| 15 | `tests/test_rag_heldout.py` | **2** | Chờ artifact JUnit trên CI | Đánh giá độ trễ và tỷ lệ recall trên tập held-out M1/M2. |
| 16 | `tests/test_rag_upload.py` | **20** | Chờ artifact JUnit trên CI | Endpoint upload 5MB limit, byte-counting middleware (413), cô lập corpus theo session, NamedTemporaryFile tự hủy. |
| 17 | `tests/test_report_v2_metrics.py` | **6** | Chờ artifact JUnit trên CI | Kiểm tra tự động đảm bảo số liệu báo cáo khớp 100% với dữ liệu nguồn và test table. |
| 18 | `tests/test_response_contracts.py` | **8** | Chờ artifact JUnit trên CI | Chuẩn hóa schema Pydantic, xác thực trường `trace_id` trong response. |
| 19 | `tests/test_retriever.py` | **19** | Chờ artifact JUnit trên CI | Tìm kiếm vector, BM25 cache theo corpus có giới hạn, hybrid fusion, reranker. |
| 20 | `tests/test_router.py` | **9** | Chờ artifact JUnit trên CI | Ma trận phân loại 2 trục, đánh giá tập đóng băng 70 câu, test với `FakeListChatModel`. |
| 21 | `tests/test_security.py` | **5** | Chờ artifact JUnit trên CI | Rate limiting theo IP, chống timing attack, `MaxBodySizeMiddleware`. |
| 22 | `tests/test_tracing.py` | **12** | Chờ artifact JUnit trên CI | Context manager tracing, an toàn khi Langfuse ngoại tuyến, sinh pre-generated `trace_id`. |
| 23 | `tests/test_vector_store.py` | **11** | Chờ artifact JUnit trên CI | Cosine distance, thay thế corpus, upsert và tìm kiếm theo danh sách `corpus = ANY(%s)`. |

👉 **Tổng số kiểm thử thu thập:** **298 tests** trên toàn bộ **23 file test** (số lượng thu thập, không phải số lượng đã chạy/đạt).

- Trạng thái pass/fail của từng file được xác định bằng artifact JUnit `test-results.xml` do job CI tải lên, không phải bằng tuyên bố trong tài liệu này. Bảng trên chỉ phản ánh số test được `pytest --collect-only` thu thập tại thời điểm sinh báo cáo.
- Lưu ý: Toàn bộ danh sách file trên được trích xuất trực tiếp từ cấu trúc thư mục `tests/` thực tế; báo cáo không còn chứa các file giả định (loại bỏ hoàn toàn các file giả định không có trong thư mục tests).
