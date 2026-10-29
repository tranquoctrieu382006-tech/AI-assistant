# RAG API (Retrieval-Augmented Generation)

Tài liệu đặc tả các giao diện lập trình ứng dụng (API) phục vụ hỏi đáp trên tài liệu nội bộ doanh nghiệp.

---

## 1. Xác thực & Bảo mật

Mọi endpoint RAG (ngoại trừ `/api/health/*`) đều yêu cầu xác thực qua HTTP Header:

```http
X-API-Key: <APP_API_KEY>
```

- Giá trị `APP_API_KEY` được cấu hình trong `.env` (tối thiểu 32 ký tự ngẫu nhiên).
- Nếu thiếu hoặc sai header, API trả về `HTTP 401 Unauthorized`.
- Nếu chưa cấu hình `APP_API_KEY` trên server, API trả về `HTTP 503 Service Unavailable`.

---

## 2. Danh Sách Endpoint

| Phương thức | Đường dẫn | Chức năng | Định dạng phản hồi |
| :--- | :--- | :--- | :--- |
| `POST` | `/api/rag/ask` | Hỏi đáp tài liệu đồng bộ | `application/json` |
| `POST` | `/api/rag/ask/stream` | Hỏi đáp tài liệu dạng Streaming (SSE) | `text/event-stream` |
| `POST` | `/api/rag/upload` | Tải lên tài liệu và lập chỉ mục tức thì | `application/json` |

---

## 3. Chi Tiết Từng Endpoint

### 3.1. `POST /api/rag/ask` (Đồng bộ)

Hỏi đáp và nhận toàn bộ câu trả lời kèm danh sách nguồn trích dẫn trong một phản hồi JSON duy nhất.

#### Request Body (`application/json`):
```json
{
  "question": "Chính sách đổi trả hàng hóa quy định trong bao nhiêu ngày?",
  "top_k": 5,
  "session_id": "session-12345"
}
```

- `question` *(string, bắt buộc)*: Nội dung câu hỏi (2 - 2000 ký tự).
- `top_k` *(integer, tùy chọn, mặc định: 5)*: Số lượng đoạn trích tối đa trả về (1 - 20).
- `session_id` *(string, tùy chọn)*: Định danh phiên hội thoại để lưu lịch sử và kích hoạt bộ viết lại câu hỏi (coreference rewrite).

#### Response (`HTTP 200 OK`):
```json
{
  "answer": "Theo chính sách đổi trả, khách hàng được quyền đổi trả hàng trong vòng 30 ngày kể từ ngày nhận hàng với điều kiện sản phẩm còn nguyên tem mác.",
  "sources": [
    {
      "source": "04_chinh_sach_doi_tra.docx",
      "chunk_index": 0,
      "score": 0.884,
      "page_number": 1,
      "section": "Điều kiện đổi trả"
    }
  ],
  "trace_id": "rag-134758ef1a7e"
}
```

---

### 3.2. `POST /api/rag/ask/stream` (Server-Sent Events - SSE)

Truyền câu trả lời theo thời gian thực (từng token) qua giao thức Server-Sent Events (SSE).

#### Request Body (`application/json`):
```json
{
  "question": "Quy định bảo hành màn hình vi tính?",
  "top_k": 5,
  "session_id": "session-12345"
}
```

#### Cấu trúc luồng sự kiện SSE (`text/event-stream`):

Mỗi gói tin tuân thủ chuẩn SSE: `data: <JSON>\n\n`. Luồng stream bao gồm các loại sự kiện (`type`) sau:

1. **Sự kiện `sources` (Nguồn trích dẫn):**  
   Được phát ra đầu tiên ngay khi các đoạn trích vượt qua ngưỡng lọc tương đồng (`score >= RAG_MIN_RELEVANCE_SCORE`):
   ```sse
   data: {"type": "sources", "sources": [{"source": "02_chinh_sach_bao_hanh.md", "chunk_index": 1, "score": 0.825, "section": "Màn hình", "page_number": null}], "trace_id": "rag-a1b2c3d4e5f6"}
   ```

2. **Sự kiện `token` (Từng mảnh từ câu trả lời):**  
   Được phát liên tục khi LLM sinh từ:
   ```sse
   data: {"type": "token", "token": "Thời"}

   data: {"type": "token", "token": " hạn"}

   data: {"type": "token", "token": " bảo hành là 24 tháng."}
   ```

3. **Sự kiện `no_answer` (Từ chối trả lời khi thiếu căn cứ):**  
   Được phát khi không tìm thấy đoạn trích nào đạt ngưỡng tin cậy, hoặc mô hình từ chối:
   ```sse
   data: {"type": "no_answer", "answer": "Tài liệu hiện có không chứa đủ thông tin để trả lời câu hỏi này.", "trace_id": "rag-a1b2c3d4e5f6"}
   ```

4. **Sự kiện `error` (Lỗi xử lý):**  
   Được phát khi xảy ra ngoại lệ trong quá trình tạo sinh:
   ```sse
   data: {"type": "error", "detail": "Mô hình không phản hồi kịp thời."}
   ```

5. **Kết thúc luồng stream (`[DONE]`):**  
   Gói tin chốt cuối cùng báo hiệu kết thúc phiên stream:
   ```sse
   data: [DONE]
   ```

---

### 3.3. `POST /api/rag/upload` (Tải lên & Lập chỉ mục)

Cho phép tải trực tiếp file tài liệu mới lên để chunking, embedding và nạp vào vector store.

#### Request: `multipart/form-data`
- `file`: File nhị phân (hỗ trợ `.pdf`, `.docx`, `.md`, `.txt`).

#### Response (`HTTP 200 OK`):
```json
{
  "status": "success",
  "filename": "chinh_sach_moi.pdf",
  "chunks_indexed": 14,
  "strategy": "heading"
}
```

---

## 4. Quản Lý Phiên Hội Thoại & Viết Lại Câu Hỏi (`session_id`)

- **Bộ nhớ hội thoại:** Hệ thống duy trì tối đa 40 tin nhắn gần nhất cho mỗi `session_id`, lưu trong bộ nhớ LRU với thời gian hết hạn TTL (mặc định 60 phút).
- **Coreference Rewriter:**  
  Khi người dùng hỏi tiếp các câu hỏi tỉnh lược (ví dụ: *"Còn điều kiện thì sao?"* sau câu *"Chính sách bảo hành là gì?"*), module `question_rewriter` sẽ tự động hợp nhất ngữ cảnh thành: *"Điều kiện bảo hành sản phẩm là gì?"* trước khi thực hiện truy xuất tài liệu.

---

## 5. Danh Mục Mã Lỗi HTTP

| Mã lỗi | Tên lỗi | Nguyên nhân & Hướng xử lý |
| :--- | :--- | :--- |
| `401` | `Unauthorized` | Thiếu hoặc sai header `X-API-Key`. |
| `422` | `Unprocessable Entity` | Dữ liệu đầu vào không hợp lệ (câu hỏi rỗng hoặc vượt quá 2000 ký tự). |
| `429` | `Too Many Requests` | Vượt hạn mức 60 req/phút theo IP hoặc hàng đợi **Admission Gate** bị đầy (trả về kèm header `Retry-After`). |
| `502` | `Bad Gateway` | Mô hình LLM bên ngoài (Gemini/OpenAI) trả về lỗi hoặc ngắt kết nối đột ngột. |
| `503` | `Service Unavailable` | `APP_API_KEY` chưa được cấu hình hoặc PostgreSQL pgvector chưa sẵn sàng. |
| `504` | `Gateway Timeout` | Quá trình truy xuất hoặc tạo sinh vượt quá ngân sách thời gian (`RAG_CLOUD_REQUEST_DEADLINE_SECONDS`). |
