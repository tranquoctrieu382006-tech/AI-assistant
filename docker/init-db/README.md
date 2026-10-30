# Khởi Tạo Cơ Sở Dữ Liệu PostgreSQL (Docker Init DB)

Thư mục này chứa các script được Docker Compose nạp tự động vào thư mục `/docker-entrypoint-initdb.d/` của container PostgreSQL khi khởi động lần đầu:

1. `01_init.sql`:
   - Kích hoạt extension `vector` (`pgvector`) trên CSDL `ai_assistant_demo`.
2. `02_schema_seed.sql`:
   - **Ghi chú về thứ tự:** Tệp `02` được ánh xạ trực tiếp từ tệp dữ liệu nguồn `./data/demo_sales_warehouse/schema_seed.sql` thông qua cấu hình volume trong `docker-compose.yml`:
     ```yaml
     - ./data/demo_sales_warehouse/schema_seed.sql:/docker-entrypoint-initdb.d/02_schema_seed.sql:ro
     ```
   - Việc ánh xạ trực tiếp này giúp tái sử dụng tệp seed gốc (2.346 dòng) mà không cần nhân bản tạo thêm file trùng lặp trong thư mục `docker/init-db/`.
3. `03_create_reader_role.sh`:
   - Tạo role bảo mật `ai_nlq_reader` dùng cho tính năng NLQ.
   - Cấp quyền chỉ đọc (`SELECT`) trên các bảng dữ liệu bán hàng.
   - Giới hạn quyền trên bảng `customers` (chỉ cho phép đọc `customer_id, city, member_tier, joined_on`, tuyệt đối cấm truy cập cột PII `email`, `phone`, `full_name`).
   - Cấm truy cập bảng nội bộ vector `rag_chunks`.
