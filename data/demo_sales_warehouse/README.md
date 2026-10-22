# Bộ demo RAG và NLQ bán hàng kho

Gồm 12 tài liệu tiếng Việt (6 MD, 3 DOCX, 3 PDF), từ điển dữ liệu, nguồn tham khảo và script PostgreSQL có schema cùng dữ liệu giả lập.

## Nạp PostgreSQL

Tạo database rỗng rồi chạy `psql -d ten_database -f schema_seed.sql`.

Seed cố định: 9 bảng quan hệ, tổng cộng 2.346 dòng dữ liệu: 8 danh mục, 15 nhà cung cấp, 80 sản phẩm, 200 khách hàng, 3 kho, 300 đơn, 900 dòng đơn, 240 tồn kho, 600 biến động kho.

## Ghi chú

Toàn bộ tên khách/email trong database là giả lập; email dùng miền `example.test`. Tài liệu là chính sách mô phỏng để thử nghiệm RAG/NLQ, không phải quy chế pháp lý dùng cho doanh nghiệp thật. CSDL được sinh từ `D:\ITSS\build_demo_pack.py` với seed Python 20261022.

Chi tiết cột, quan hệ và ý nghĩa nghiệp vụ nằm trong `data_dictionary.md`. Nguồn tham khảo công khai và lưu ý kiểm tra giấy phép ở `nguon_dataset.md`.
