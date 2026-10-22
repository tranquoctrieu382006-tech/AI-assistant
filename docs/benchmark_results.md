# Báo cáo Benchmark LLM – Ngày 22/10/2026

## 1. Môi trường thử nghiệm
- **Framework**: LangChain `init_chat_model` + `langchain-ollama`
- **Observability**: Langfuse Cloud (`https://cloud.langfuse.com`)
- **Model local**: `ollama/gemma3:12b` (chạy trên local machine qua Ollama)
- **User ID**: `intern_tran_quoc_trieu`

## 2. Kết quả đo lường chi tiết

| STT | Category | Prompt tóm tắt | Model | Latency | Tokens (In/Out/Total) | Status | Langfuse Trace |
|:---:|---|---|---|:---:|:---:|:---:|:---:|
| 1 | **RAG-Concept** | Giải thích cơ chế & lợi ích RAG trong 3 câu | `ollama/gemma3:12b` | **22.19s** | 38 + 133 = **171** | SUCCESS | ✅ Ghi nhận |
| 2 | **NLQ-SQL** | Viết câu lệnh SQL tìm top 3 khách hàng chi tiêu 2026 | `ollama/gemma3:12b` | **81.83s** | 51 + 491 = **542** | SUCCESS | ✅ Ghi nhận |
| 3 | **Reasoning** | So sánh PostgreSQL+pgvector vs Pinecone cho RAG nội bộ | `ollama/gemma3:12b` | **263.20s** | 35 + 1491 = **1526** | SUCCESS | ✅ Ghi nhận |

## 3. Nhận xét & Đánh giá
1. **Chất lượng câu trả lời**:
   - `gemma3:12b` trả lời tiếng Việt trôi chảy, đúng định dạng yêu cầu.
   - Câu lệnh SQL sinh ra đúng cú pháp, có `SUM()`, `GROUP BY`, `ORDER BY` và `LIMIT 3`.
   - Bài so sánh pgvector vs Pinecone rất đầy đủ về tính toàn vẹn dữ liệu, chi phí vận hành, bảo mật nội bộ và độ phức tạp mở rộng.
2. **Độ trễ & Tài nguyên**:
   - Do chạy local trên GPU/CPU cá nhân, các câu trả lời dài (>1000 tokens) mất từ 1 đến 4 phút.
   - Đối với các tác vụ yêu cầu phản hồi nhanh trong API (như Router phân loại câu hỏi), nên ưu tiên mô hình cloud nhẹ như `gpt-4o-mini` hoặc `gemini-2.0-flash` khi có API key.
3. **Observability**:
   - 100% các lượt gọi đều có mặt trên Langfuse Dashboard kèm `session_id`, `user_id`, `tags` (`benchmark`, `local`, `<category>`).
