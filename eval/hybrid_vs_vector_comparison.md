# So sánh Retrieval: Vector-only vs. Hybrid (BM25 + PGVector)

> **Dataset:** `data/rag_golden_set.json` (65 cases: 41 answerable, 19 in_scope_missing, 5 out_of_scope_far)
> **Model Embedding:** `bge-m3`
> **Gating Strategy:** Cosine threshold (`score >= 0.5`) checked for each final hybrid chunk
> **Hybrid Fusion:** Weighted Reciprocal Rank Fusion ([BM25, vector] weights `[0.5, 0.5]`), keyed by `(source, chunk_index)`

## Bảng chỉ số tổng hợp

| Chỉ số | Vector-only (V5) | Hybrid RRF (H5) | Chênh lệch (H5 - V5) |
|---|:---:|:---:|:---:|
| **Hit@1** | 0.8537 (35/41) | 0.7561 (31/41) | -0.0976 |
| **Hit@3** | 0.9512 (39/41) | 0.9756 (40/41) | +0.0244 |
| **Hit@5** | 0.9756 (40/41) | 0.9756 (40/41) | +0.0000 |
| **MRR** | 0.9207 | 0.8740 | -0.0467 |
| **Source Coverage@5** | 0.9756 | 0.9756 | +0.0000 |
| **False Negatives (Answerable bị từ chối)** | 1/41 | 1/41 | +0 |
| **False Positives (in_scope_missing)** | 14/19 | 14/19 | +0 |
| **False Positives (out_of_scope_far)** | 0/5 | 0/5 | +0 |

## Nhận xét & Phân tích
- Chênh lệch trên development set nhỏ: Hybrid có thêm 1 câu trong top 3, nhưng mất 4 câu ở top 1; Hit@5 không đổi. Chỉ số tổng hợp này chưa cho thấy BM25 cải thiện chung hay xác nhận nguyên nhân do khớp từ khóa.
- Các số FP bằng vector-only phản ánh cổng cosine dùng chung trong lần chạy cũ; không phải hiệu ứng độc lập của BM25. Implementation hiện tại chấm cosine từng chunk sau fusion, nên số liệu này cần được chạy lại trước khi dùng làm kết luận.
- Đây là một lượt đo trên corpus 30 chunk và golden set development, không phải kết quả held-out.
