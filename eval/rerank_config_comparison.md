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
- **Baseline V5 / V10:** Đạt Hit@1 cao nhất (**85.37%**) và MRR cao nhất (**0.9207**) với độ trễ cực thấp (< 75 ms). Khi tăng k từ 5 lên 10 không đổi Hit@1/Hit@3 do corpus 12 văn bản có độ tập trung cao.
- **Hybrid H5:** Giúp tăng Hit@3 lên **97.56%** (+2.44% so với V5) với độ trễ chỉ 56.7 ms, tuy nhiên Hit@1 giảm xuống 75.61% do điểm RRF bình đẳng thứ hạng giữa BM25 và Vector.
- **Reranker R10 & HR10:** Cả 2 cấu hình đều đạt Hit@3 = **97.56%** và Hit@5 = **97.56%**, giữ MRR ở mức cao (**0.9146**). Cả hai đều kiểm soát an toàn tuyệt đối các câu ngoài phạm vi (`FP out_of_scope_far = 0/5`).
  - Về độ trễ: **R10** (2.571 ms) nằm trong giới hạn an toàn (< 3s). **HR10** (4.641 ms) vượt quá giới hạn 3 giây khi chạy CPU inference.
- **Cấu hình mặc định đề xuất:**
  - Giữ mặc định hệ thống: `search_mode = "vector"`, `use_reranker = False` (tối ưu hóa độ trễ 70ms và Hit@1 cao nhất 85.37%).
  - Hỗ trợ bật cấu hình `use_reranker = True` (R10) hoặc `search_mode = "hybrid"` (H5) thông qua config / query param khi nghiệp vụ yêu cầu tối đa hóa Recall/Hit@3 (97.56%).
