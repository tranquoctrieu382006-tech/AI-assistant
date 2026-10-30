# Số hiện hành (chốt ngày 2026-10-29)

Bảng này ghi lại con số **đang dùng được**, kèm commit và phiên bản dataset, để
tránh so sánh chéo giữa các bộ dữ liệu khác nhau. Mọi giá trị dưới đây được
**tính lại từ artifact đã commit**, không phải từ một lần chạy mới.

| Mốc | Commit | Ghi chú |
|---|---|---|
| HEAD khi lập bảng | `4fe95e0` | nhánh `feature/TASK-AI21-AI23-chat-ui-eval-unit-test`; **274** test |
| Bản zip đang lưu hành | `5b8a5d4` | thiếu 4 commit follow-up; **268** test |

## 1. RAG abstention — 65 ca

Nguồn chính: `eval/dev_abstention_prompt_gemini35_m2.json` — chạy lại ngày
2026-10-29 trên corpus `demo` đã dựng lại (30 chunk / 12 nguồn), Gemini 3.5
Flash-Lite, 54 lần gọi API, **0 ca lỗi**.

Nhãn đúng theo `docs/rag_eval_protocol.md`: **cả** `in_scope_missing` **và**
`out_of_scope_far` là expected abstain.

| Nhóm | n | Gate chặn | Từ chối cuối | Ghi chú |
|---|---:|---:|---:|---|
| answerable | 41 | 1 | 1 | `rag-029` bị từ chối nhầm ở tầng gate |
| in_scope_missing | 19 | 5 | 19 | 14 ca lọt gate đều bị LLM chặn |
| out_of_scope_far | 5 | 5 | 5 | chỉ báo count vì n < 10 |
| **Tổng** | **65** | **11** | **25** | |

- `abstention_accuracy` = **64/65 = 0.985** (65/65 ca quyết định được).
- Quyết định gate và quyết định từ chối **giống hệt** artifact M1 từng ca: dựng
  lại corpus không làm đổi hành vi, chỉ làm số liệu đo trên corpus đúng.
- Artifact M1 ghi `0.908` (59/65) là **sai nhãn**: 5 câu `out_of_scope_far` bị
  chặn đúng lại bị tính là trả lời sai. Không dùng lại số này.
- `fragment_match` = **31/41 = 75.6%**. Mẫu số là 41 ca answerable, không phải 46.
  Chênh 1 ca so với M1 (32/41) **không phải hồi quy**: cả 3 ca đổi kết quả chỉ do
  model diễn đạt lại, ví dụ `08:30–17:30` thành `08:30 đến 17:30` và
  `1.000.000 đồng trước giảm giá` thành `1.000.000 đồng (trước giảm giá)`. Phép
  so khớp chuỗi con chính xác vì thế chỉ nên đọc là chỉ báo, sai số ±1–2 ca.
- Hallucination thật trên hai nhóm âm: **0/24**.
- `latency_ms` của artifact M2 đo **sau** sleep, ca bị gate trả `null`: n = 54,
  min 1.444 ms, median 26.901 ms, max 120.500 ms. Provider Gemini 3.5 Flash-Lite
  suy giảm mạnh đúng lúc đo (gọi thử 1 câu mất 26,37 s), nên **không dùng số này
  làm baseline độ trễ**. Artifact M1 cũ đo cả 4,2 s pacing nên cũng không dùng.

## 2. Retrieval và reranker

Nguồn: `eval/report_v2.md` §3.2 và `eval/rerank_config_comparison.md`, cùng
dataset 65 ca (41 answerable).

| Cấu hình | Hit@1 | Hit@5 | MRR | Độ trễ TB |
|---|---:|---:|---:|---:|
| V5 (mặc định) | 0.8537 (35/41) | 0.9756 | 0.9207 | 70.1 ms |
| R10 | 0.8049 (33/41) | 0.9756 | 0.9146 | 2571 ms (≈2,57 s) |
| HR10 | 0.8049 (33/41) | 0.9756 | 0.9146 | 4641 ms (≈4,64 s) |

- `avg_latency_ms` lưu **mili-giây**; script đo bằng giây rồi nhân 1000. 2571 ms
  và 2,571 s là cùng một giá trị.
- **Hit@5 = 1.000 thuộc bộ 47 ca** (39 answerable), không phải bộ 65 ca. Hai số
  không mâu thuẫn, chỉ khác phiên bản dataset — phải ghi kèm phiên bản.
- FP `out_of_scope_far` = 0/5 ở mọi cấu hình, nhưng n = 5 nên không phải bằng
  chứng an toàn.
- Kết quả đo một lượt, không warm-up, và R10/HR10 dùng 10 ứng viên (không khớp
  cấu hình production hiện tại). Chưa có số `P-current`.

### Hạn mức model (kiểm ngày 2026-10-29)

- `gemini-3.6-flash` free tier: **20 request/ngày**. Chạy 65 ca bị
  `429 RESOURCE_EXHAUSTED` sau 20 lần gọi; phải chờ reset (~15h42m).
- `gemini-3.5-flash-lite` gọi được trở lại sau đợt 503, nhưng lúc đo rất chậm:
  median 26,9 s/lần gọi, cá biệt 120,5 s. Không phải lỗi code.
- Vì vậy phép đo 65 ca phải dùng model còn quota, và script đã tách riêng ca
  lỗi khỏi mẫu số thay vì tính là "đã trả lời".

## 3. NLQ
- `eval/v2_nlq_eval_run_20261028.json`: **2 ca**, `execution_accuracy = 1.0`.
  Đây là artifact có thật nhưng **không đại diện** cho golden set 30 ca.
- Số cũ 16/18/20 không so sánh được với golden set hiện hành: dataset đã đổi và
  ca 021/027 đã có tie-break ngay trong câu hỏi. Cần rerun trước khi báo cáo.

## 4. Hội thoại nhiều lượt

- `eval/conversation_dev_evaluation.json`: **9/23** lượt `answer_matched`, không
  có chỉ số tổng hợp.
- `conv-009` lượt 1 trả lời về mật khẩu nhưng vẫn bị đánh `matched` — nhãn sai,
  cần sửa trước khi dùng.

## 5. Độ trễ

- `p95 = 69.823 s` là **giá trị lớn nhất của 11 mẫu** (nearest-rank), không phải
  "lớn thứ nhì". Chỉ nên ghi `max/11 mẫu`.
- RAG chưa có timing từng giai đoạn. NLQ đã log stage tại
  `app/core/nlq_graph.py` (`NLQ stage completed: <name> in <s>`).

## 6. Trạng thái manifest (chưa freeze lại)

`eval/development_manifest_m1.json` và `eval/heldout_manifest_m2.json` được tạo
bởi phiên bản hash **byte thô** (không có trường `hash_format`), nên phụ thuộc
line ending của máy checkout. Kiểm tra bằng:

```powershell
.venv\Scripts\python.exe -m scripts.report_manifest_drift
```

Tại thời điểm lập bảng, dưới hash chuẩn hoá LF:

- M1: **12/20** entry lệch. Trong đó **5 lệch nội dung thật**
  (`scripts/evaluate_rag_retrieval.py`, `scripts/write_development_manifest.py`,
  `tests/test_golden_data_validation.py`, `docs/rag_eval_protocol.md`,
  `data/nlq_golden_set.json`) và 7 chỉ lệch line ending.
- M2: **3/4** entry lệch. Trong đó **2 lệch nội dung thật**
  (`scripts/evaluate_rag_heldout.py`, `scripts/write_heldout_manifest.py`) và 1
  chỉ lệch line ending (`eval/rag_heldout_m1.json`).

Vì vậy **không sửa M2**. Việc cần làm: review từng file lệch nội dung, commit
thay đổi có chủ đích, freeze **M1'** (`--label M1-prime`) rồi **M3** trỏ vào M1'
sau khi có người review held-out set.

## 7. Trạng thái corpus `demo`

Corpus `demo` trong DB local đã được dựng lại từ đúng 12 tài liệu của
`data/demo_sales_warehouse` (heading chunking): **30 chunk / 12 nguồn**, khớp
manifest. Trước khi dựng lại, corpus có 49 chunk / 13 nguồn — thừa một file
`Smart_Waste_Sorter_hoan_chinh_v4.pptx.pdf`, làm mọi phép đo retrieval sai.

`POST /api/rag/ask` và `/api/rag/ask/stream` giờ gộp nguồn theo tài liệu (mỗi
nguồn một dòng, giữ chunk điểm cao nhất), cắt theo biên 0,10 điểm và tối đa 3
nguồn. `top_k` vẫn là số đoạn trích đưa vào ngữ cảnh, không phải số nguồn hiển
thị.

## Việc còn lại trước khi freeze lại baseline

1. Review từng file lệch nội dung (mục 6), commit thay đổi có chủ đích.
2. Freeze M1' và M3:

   ```powershell
   .venv\Scripts\python.exe -m scripts.write_development_manifest `
       --output eval/development_manifest_m1_prime.json --label M1-prime
   .venv\Scripts\python.exe -m scripts.write_heldout_manifest `
       --output eval/heldout_manifest_m3.json --label M3 `
       --m1-manifest eval/development_manifest_m1_prime.json `
       --m1-commit <commit-chứa-M1'>
   ```

3. Chỉ sau đó mới chạy `scripts.evaluate_rag_heldout` (evaluator đã dùng hash
   chuẩn hoá LF nên sẽ từ chối M1/M2 cũ).

## Cách tái lập nhanh

```powershell
.venv\Scripts\python.exe -m scripts.report_manifest_drift   # trạng thái hash
.venv\Scripts\python.exe -m pytest -q                       # 279 test sau khi thêm test drift, nhãn far, ca lỗi, dedupe nguồn
```
