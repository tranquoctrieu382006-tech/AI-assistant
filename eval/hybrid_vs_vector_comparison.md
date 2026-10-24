# So sánh Retrieval: Vector-only vs. Hybrid (BM25 + PGVector)

> **Dataset:** `data/rag_golden_set.json` (65 cases: 41 answerable, 19 in_scope_missing, 5 out_of_scope_far)
> **Model Embedding:** `bge-m3`
> **Gating Strategy:** Cosine threshold (`score >= 0.5`) from vector branch
> **Hybrid Fusion:** Reciprocal Rank Fusion (EnsembleRetriever with weights [0.5, 0.5], id_key="chunk_id", truncated to top-k)

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
- **Cổng từ chối (Gating):** Do hybrid dùng chung cổng cosine từ nhánh vector, số lượng False Positives của cả 2 nhóm `in_scope_missing` (14/19) và `out_of_scope_far` (0/5) là đồng nhất.
- **Thứ hạng tài liệu:** BM25 bổ trợ các câu chứa từ khóa chính xác, trong khi vector bge-m3 đảm bảo ngữ nghĩa.
