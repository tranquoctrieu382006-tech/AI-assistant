# So sánh thực nghiệm 5 cấu hình Retrieval & Reranker

> **Dataset:** `data/rag_golden_set.json` (65 cases)
> **Model Reranker:** `BAAI/bge-reranker-v2-m3`

| Cấu hình | Retrieval | k lấy | Rerank | Top n | Hit@1 | Hit@3 | Hit@5 | MRR | FP (in-missing) | FP (out-far) | FN (ans) | Độ trễ TB |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **V5** | vector | 5 | Không | 5 | 0.8537 | 0.9512 | 0.9756 | 0.9207 | 14/19 | 0/5 | 1/41 | 70.1 ms |
| **V10** | vector | 10 | Không | 10 | 0.8537 | 0.9512 | 0.9756 | 0.9207 | 14/19 | 0/5 | 1/41 | 24.5 ms |
| **H5** | hybrid | 5 | Không | 5 | 0.7561 | 0.9756 | 0.9756 | 0.8740 | 14/19 | 0/5 | 1/41 | 56.7 ms |
| **R10** | vector | 10 | bge-reranker | 5 | 0.8049 | 0.9756 | 0.9756 | 0.9146 | 14/19 | 0/5 | 1/41 | 2571.0 ms |
| **HR10** | hybrid | 10 | bge-reranker | 5 | 0.8049 | 0.9756 | 0.9756 | 0.9146 | 14/19 | 0/5 | 1/41 | 4641.1 ms |

## Tiêu chuẩn chọn cấu hình mặc định
1. **Hit@1 và MRR không giảm** so với baseline V5.
2. **False positive ở nhóm câu ngoài phạm vi (out_of_scope_far)** không tăng.
3. **Độ trễ trung bình < 3.000 ms** (3 giây) trên môi trường local.
4. *Lưu ý:* Đây là kết quả thăm dò trên development set để tìm kiếm cấu hình tối ưu.

## Phân tích & Kết luận
- **V5/V10:** Hit@1 và MRR cao nhất trong các cấu hình đã đo (35/41, MRR 0.9207). V10 không cải thiện retrieval so với V5. Độ trễ 70.1 ms so với 24.5 ms không đáng tin để so sánh: script chạy một lượt theo thứ tự cấu hình, không warm-up riêng và lấy trung bình các câu; có thể bị ảnh hưởng bởi khởi động nguội.
- **H5:** Hit@3 tăng đúng **1/41 câu** (39 lên 40), trong khi Hit@1 giảm **4/41 câu** (35 xuống 31) và MRR giảm 0.0467. Kết quả này chưa chứng minh BM25 cải thiện retrieval nói chung.
- **R10/HR10:** Hit@3 tăng 1/41 câu, Hit@1 giảm 2/41 câu và MRR giảm 0.0061. R10 mất trung bình **2.571 giây** mỗi câu; HR10 mất **4.641 giây**. Đơn vị “ms” ở phần mô tả cũ bị sai 1.000 lần.
- FP out-of-scope là 0/5 ở **mọi cấu hình**, do cùng dùng cổng cosine; không thể quy kết điều này cho reranker hay gọi reranker là bảo đảm an toàn. Nhóm chỉ có 5 câu nên đây cũng không phải bằng chứng an toàn tuyệt đối.

### Quyết định

Theo tiêu chí đã đặt trước (Hit@1 và MRR không giảm so với V5), **H5, R10 và HR10 đều không đạt**. Giữ V5 làm baseline tạm thời; không khuyến nghị bật hybrid/reranker dựa trên bộ số liệu này. Đây là phép đo một lượt trên development set, không có warm-up hoặc lặp nhiều lần. Hơn nữa, các số liệu được tạo bởi phiên bản hybrid trước khi sửa cosine gate theo từng chunk; cần chạy lại trước khi so sánh cấu hình của code hiện tại.
