# Báo cáo benchmark LLM

**Ngày chạy:** 22/10/2026

**Dataset:** 3 prompt (RAG, NLQ-SQL, so sánh)

**Trace:** Langfuse Cloud, session `benchmark-session-1792654439`
**Kết quả chi tiết:** [`benchmark_20261022_143955.csv`](benchmark_runs/benchmark_20261022_143955.csv)

## Tổng hợp

| Vai trò | Model | Thành công | Độ trễ trung bình | Token vào | Token ra | Chi phí ước tính |
|---|---|---:|---:|---:|---:|---:|
| Nhanh | Groq `openai/gpt-oss-20b` | 3/3 | 2,26 giây | 306 | 2.939 | $0,000905 |
| Mạnh | Groq `openai/gpt-oss-120b` | 3/3 | 3,82 giây | 306 | 3.872 | $0,002369 |
| Local | Ollama `gemma3:12b` | 3/3 | 111,41 giây | 124 | 2.111 | Không tính phí API |

Chi phí Groq tổng ước tính cho 6 lượt gọi là **$0,003274**. Mức ước tính dựa trên token đầu vào/đầu ra và giá niêm yết của Groq lúc kiểm tra; hóa đơn thực tế có thể khác. Model Groq 20B dùng mức $0,075/$0,30 trên một triệu token đầu vào/đầu ra; 120B dùng $0,15/$0,60. Nguồn: https://console.groq.com/docs/models.

## Kết quả theo prompt

| Prompt | Local (giây) | Groq nhanh (giây) | Groq mạnh (giây) |
|---|---:|---:|---:|
| Giải thích RAG | 21,19 | 1,04 | 1,12 |
| Truy vấn top 3 khách hàng | 73,79 | 1,48 | 1,74 |
| So sánh pgvector và Pinecone | 239,25 | 4,26 | 8,60 |

## Ghi chú

- Cả 9 lượt gọi thành công và Langfuse đã flush trace sau khi benchmark.
- Thời gian local phụ thuộc phần cứng và lần chạy nóng/lạnh; prompt so sánh dài mất nhiều thời gian hơn.
- Bộ đo ghi số token theo metadata trả về từ model; CSV lưu từng lượt, session ID và trạng thái.
- Benchmark này đo độ trễ, token và chi phí ước tính; chưa chấm điểm chất lượng câu trả lời tự động.
- Đây là snapshot lịch sử one-shot ngày 22/10: chưa có warm-up hay nhiều mẫu, nên các độ trễ (đặc biệt lần Ollama đầu) không đại diện cho median ổn định. Script hiện tại đã warm-up và lấy median của ba mẫu cho mỗi prompt; benchmark lần mới sẽ tạo thêm lượt gọi và trace. Chi phí chỉ ước tính cho hai model Groq đã có bảng giá trong code; model khác để trống.
