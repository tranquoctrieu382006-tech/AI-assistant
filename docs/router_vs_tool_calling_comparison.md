# So Sánh Kiến Trúc Định Tuyến (Router) và Tác Tử Gọi Công Cụ (Tool-Calling Agent)

> **Mã công việc:** `TASK-AI17` | **Dự án:** Mini Enterprise AI Assistant (RAG & NLQ)  
> **Tác giả:** Trần Quốc Triệu — Intern AI  
> **Thời điểm đánh giá:** Tuần 2 (Tháng 10/2026)

---

## 1. Bối Cảnh & Đặt Vấn Đề

Trong quá trình phát triển Trợ lý AI hợp nhất có khả năng phục vụ đồng thời hai luồng nghiệp vụ cốt lõi:
1. **RAG (Hỏi đáp tài liệu nội bộ):** Quy trình, quy định bảo hành, đổi trả, hướng dẫn công việc.
2. **NLQ (Truy vấn dữ liệu số CSDL):** Báo cáo doanh thu, tồn kho, sản phẩm, khách hàng từ PostgreSQL.

Yêu cầu đặt ra tại nhiệm vụ `TASK-AI17` là nghiên cứu, thử nghiệm và so sánh hiệu quả giữa 2 trường phái thiết kế:
- **Phương án 1 (Classifier Router - Lựa chọn chính):** Sử dụng một node phân loại ý định (Intent Classifier) chuyên biệt trong LangGraph để phân luồng người dùng vào đúng graph con tương ứng.
- **Phương án 2 (Autonomous Tool-Calling Agent - Phương án thử nghiệm):** Gắn trực tiếp 2 công cụ (`search_documents` và `query_data`) vào LLM và để mô hình tự suy luận (ReAct / Function Calling) xem khi nào cần gọi công cụ nào.

---

## 2. Mô Tả Cơ Chế Của Từng Phương Án

```
PHƯƠNG ÁN 1: LANGGRAPH CLASSIFIER ROUTER
Client Prompt ──> [ Chốt chặn an toàn & Regex Pre-filter ]
                       │
                       ▼
                 [ LLM Classifier (Structured Enum) ]
                       │
         ┌─────────────┼─────────────┐
         ▼             ▼             ▼
      [ RAG ]       [ NLQ ]    [ Out-of-Scope ]
  (Vector+BM25)  (AST Validate)   (Abstain)

---------------------------------------------------------------------------------

PHƯƠNG ÁN 2: AUTONOMOUS TOOL-CALLING AGENT
Client Prompt + [Tools Schema] ──> [ LLM Call 1 ]
                                         │
                                         ▼ (Sinh Tool Calls)
                                   [ Agent Executor ]
                                   ┌─────┴─────┐
                                   ▼           ▼
                           [search_docs]  [query_data]
                                   └─────┬─────┘
                                         ▼ (Tool Result)
Client Response <───────────────── [ LLM Call 2: Tổng hợp ]
```

---

## 3. Bảng So Sánh Chi Tiết (Evaluation Matrix)

> ⚠️ **Ghi chú về phương pháp đánh giá & nguồn số liệu:**
> - **Số liệu Phương án 1 (Router):** Con số 100% (70/70 câu trên bộ frozen dev/test) được đo kiểm bởi `tests/test_router.py` trên **bộ quy tắc regex tiền xử lý (`_rule_based_classify`)**, do đó cần lưu ý về nguy cơ overfit trên tập dữ liệu đóng băng này. Trong runtime thực tế, LLM structured classifier chỉ đóng vai trò dự phòng phân loại ngữ nghĩa cho các câu hỏi không khớp mẫu regex. Về độ trễ, khoảng **0.001s – 0.8s** chỉ phản ánh **thời gian xử lý của riêng bước định tuyến (Routing Step Latency)**; độ trễ toàn trình của cả pipeline RAG/NLQ được đo lường riêng biệt tại [`eval/current_numbers.md`](../eval/current_numbers.md) §5 (P50 = 15.4s, P95 = 69.8s).
> - **Số liệu Phương án 2 (Tool-Calling Agent):** Đây là **ước tính lý thuyết dựa trên phân tích đặc tính kiến trúc (Theoretical Architectural Estimation)** xuất phát từ bản chất vòng lặp ReAct / Tool-calling (tối thiểu 2 lượt round-trip LLM, kích thước token của tool schemas và hành vi gọi thừa công cụ trên các câu hỏi ranh giới), do dự án không duy trì runtime agent ReAct nội bộ mà đóng gói công cụ ra chuẩn mở MCP.

| Tiêu Chí So Sánh | Phương Án 1: Classifier Router (LangGraph) | Phương Án 2: Autonomous Tool-Calling Agent | Đánh Giá Ưu Thế |
|---|---|---|:---:|
| **Độ trễ bước định tuyến (Routing Step Latency)** | **0.001s – 0.8s (Chỉ tính bước phân loại)**<br>• Regex rule-based: ~0.001s.<br>• LLM Classifier: ~0.4s – 0.8s (Gemini Flash-Lite).<br>*(Độ trễ toàn trình pipeline: xem `eval/current_numbers.md`)* | **Ước tính 3.8s – 6.5s (Cả vòng lặp agent)**<br>• Bắt buộc tối thiểu 2 lượt gọi LLM: Lượt 1 quyết định tool, Lượt 2 đọc kết quả và tổng hợp câu trả lời. | 🏆 **Router**<br>(Bước phân loại tức thì) |
| **Tiêu thụ Token & Chi phí API** | **Rất thấp (~180 prompt tokens cho router)**<br>• Prompt router chỉ nhận câu hỏi, không phải nhồi schema CSDL hay mô tả công cụ chi tiết.<br>• Phân luồng nào chỉ nạp context luồng đó. | **Ước tính cao gấp 3–4 lần**<br>• Mỗi câu hỏi đều phải gửi toàn bộ tool JSON schemas.<br>• Khi tool NLQ trả về bảng số liệu, toàn bộ dữ liệu thô bị dồn vào context vòng 2, nhanh chóng chạm hạn mức 15 RPM của Gemini Free Tier. | 🏆 **Router**<br>(Tiết kiệm chi phí, tránh lỗi 429) |
| **Bảo mật & Rào chắn (Safety & Guardrails)** | **Kiểm soát chặt chẽ 100% (Deterministic)**<br>• Câu hỏi NLQ bắt buộc đi qua bộ AST Validator (`sqlglot`), kiểm tra `allowed_tables`, `allowed_columns`, và chốt PII trước khi chạy.<br>• Câu hỏi mang tính phá hoại bị chặn ngay tại chốt router. | **Rủi ro rò rỉ & Prompt Injection**<br>• Nếu văn bản RAG chứa câu chỉ thị độc hại (Prompt Injection), LLM có thể bị đánh lừa tự sinh tool call tấn công CSDL.<br>• Khó ép chốt chặn tiền kiểm (pre-execution checks). | 🏆 **Router**<br>(Đáp ứng tiêu chuẩn Zero-Trust) |
| **Độ chính xác phân loại (Accuracy)** | **100% (70/70 câu frozen dev/test)**<br>• Đo trên bộ luật regex tiền xử lý (`_rule_based_classify`). Có rủi ro overfit trên bộ 70 câu; LLM classifier đóng vai trò xử lý các câu hỏi mở. | **Ước tính lý thuyết 88% – 92%**<br>• Các câu hỏi ranh giới (ví dụ: *"Chính sách bảo hành cho đơn hàng DH102 là gì?"*) thường bị LLM gọi thừa cả 2 công cụ hoặc lúng túng chọn nhầm thứ tự. | 🏆 **Router**<br>(Độ ổn định cao) |
| **Khả năng quan sát & Gỡ lỗi (Observability)** | **Tường minh qua `reason_code` và span riêng biệt**<br>• Mỗi quyết định định tuyến đều gắn mã lý do (`greetings`, `write_attempt_blocked`, `sql_count_pattern`, `nlq_semantic`). | **Khó gỡ lỗi**<br>• Khó xác định nguyên nhân khi LLM quyết định bịa câu trả lời (hallucination) thay vì gọi tool. | 🏆 **Router** |
| **Khả năng mở rộng đa bước (Multi-step Reasoning)** | **Hạn chế đối với câu hỏi cần kết hợp đồng thời 2 nguồn**<br>• Không thể tự động lấy kết quả SQL rồi tra tiếp vào RAG trong cùng 1 request mà cần cơ chế fallback 1-step. | **Tự nhiên, linh hoạt**<br>• Có thể gọi `query_data` lấy mã sản phẩm, sau đó gọi `search_documents` để đọc chính sách cho sản phẩm đó. | 🏆 **Tool-Calling** |

---

## 4. Kết Luận & Quyết Định Kiến Trúc

1. **Kiến trúc áp dụng cho sản phẩm nội bộ (Production System):**
   - Lựa chọn **Classifier Router (LangGraph Node)** làm xương sống điều phối. Quyết định này giúp hệ thống đạt độ trễ thấp, tiết kiệm quota API Gemini, loại bỏ hoàn toàn nguy cơ Prompt Injection vượt quyền vào CSDL bán hàng, và vượt qua 100% các bài test bảo mật.
2. **Vai trò của Tool-Calling:**
   - Thay vì để LLM nội bộ tự chạy vòng lặp agent không kiểm soát, dự án đóng gói chức năng RAG và NLQ thành chuẩn **Model Context Protocol (MCP)** tại [`app/mcp_server.py`](../app/mcp_server.py).
   - Mô hình này cho phép các Agent bên ngoài (Claude Desktop, Cursor, Antigravity IDE) đóng vai trò Tool-Calling Client, trong khi hệ thống nội bộ đóng vai trò Server cung cấp công cụ an toàn và chuẩn hóa.
