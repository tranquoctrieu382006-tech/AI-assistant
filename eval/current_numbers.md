# Số hiện hành (chốt ngày 2026-10-30 — Bàn giao Tuần 2)

Bảng này ghi lại con số **đang dùng được**, kèm commit và phiên bản dataset, để
tránh so sánh chéo giữa các bộ dữ liệu khác nhau. Mọi giá trị dưới đây được
**tính lại từ artifact đã commit**, không phải từ một lần chạy mới.

| Mốc | Commit | Ghi chú |
|---|---|---|
| HEAD bàn giao Tuần 2 | `fc9facdddc593ccc8772e5cb89577ef1aba3a110` (`fc9facd`) | nhánh `feature/TASK-AI24-AI26-testing-handover-demo` & `release/week-2-demo`; **400** test passed 100% (400/400) |
| Kết quả chạy kiểm thử | `eval/pytest_run_log.txt` | Lưu chi tiết toàn bộ 400 test case thực thi đạt 100% trong 19.88s |
| Bản đóng gói bàn giao | Mã nguồn hoàn thiện | Đầy đủ mã nguồn, tài liệu, kịch bản demo và log kiểm thử độc lập |

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

### Chạy lại end-to-end 2026-10-30 (đo sàn nhiễu)

Hai lần chạy độc lập V0 (`eval/rag_variants_answers_v0_run1.json`,
`eval/rag_variants_answers_v0_run2.json`), cùng cấu hình production
(`search_mode=vector`, `top_k=5`, gate 0.5), cùng 65 ca:

| Chỉ số | Run 1 | Run 2 | Artifact M2 cũ |
|---|---:|---:|---:|
| answerable đã trả lời | 40/41 | 40/41 | 40/41 |
| `fragment_match` | 32/41 | 32/41 | 31/41 |
| hallucination trên nhóm âm | 0/24 | 0/24 | 0/24 |
| lần gọi LLM | 54 | 54 | 54 |

- **Trạng thái trả lời/từ chối không đổi ca nào** giữa run 1 và run 2. Sàn nhiễu
  **của retrieval** = 0 ca đổi qua 3 lần chạy V0 (retrieval tất định).
- Nhưng `fragment_match` **lật ở 4/41 ca** giữa hai lần chạy (`rag-018`,
  `rag-037`, `rag-039`, `rag-041`) và 16/65 câu có text khác nhau, dù tổng vẫn
  32/41 nhờ hai ca lên và hai ca xuống. Sàn nhiễu của phép khớp chuỗi con là
  **±2 ca**, không phải 0.
- `fragment_match` 32 so với 31 của M2 nằm trong sàn nhiễu đó; không ca
  answerable nào đổi trạng thái trả lời/từ chối.

### So sánh lượt trả lời V0 vs V2/V3/V4 (2026-10-30)

Nguồn: `eval/rag_variants_answers_v2v4.json` (cùng 65 ca, cùng cấu hình
production, cùng cache viết lại với V0). Dòng V0 lấy từ hai lần chạy
(`eval/rag_variants_answers_v0_run1.json` và `..._run2.json`), hai lần này có
cùng trạng thái trả lời/từ chối ở từng ca.

| Biến thể | Answerable đã trả lời | `fragment_match` | Hallucination | Lần gọi LLM |
|---|---:|---:|---:|---:|
| V0 | 40/41 | 32/41 | 0/24 | 54 |
| V2 | 40/41 | 32/41 | 0/24 | 53 |
| V3 | 40/41 | 30/41 | 0/24 | 53 |
| V4 | 37/38 ca quyết định | 30/38 ca quyết định | 0/19 ca quyết định | 53 |

- **V2 khớp V0 về trạng thái trả lời/từ chối ở cả 41 ca answerable**; tổng
  `fragment_match` cũng bằng đúng V0 (32/41) nhưng có 2 ca lật ngược chiều
  (`rag-023` từ không khớp thành khớp, `rag-037` ngược lại) — vẫn trong sàn
  4 ca đã đo giữa hai lần chạy V0.
- **V3 = V2 về truy xuất**: cả 65 ca có đúng cùng 5 chunk (`results` giống nhau
  từng ca), chỉ khác metadata `dual_query`. Vậy mà `fragment_match` vẫn lật ở
  **4/41 ca** (`rag-018`, `rag-023`, `rag-039`, `rag-040`) và tổng đổi −2 — đúng
  bằng sàn nhiễu, nên **không kết luận V3 kém hơn V2**. Chỉ coi là cải
  thiện/hồi quy khi vượt quá 4 ca lật hoặc khi đổi trạng thái trả lời/từ chối.
- **V4 chưa đủ dữ liệu**: 8 ca lỗi `GoogleRateLimitError` (429) — 3 answerable
  (`rag-036`, `rag-041`, `rag-048`) và 5 ca âm (`rag-051`, `rag-052`, `rag-056`,
  `rag-059`, `rag-060`). Chạy lại riêng V4 với pacing 8 s
  (`eval/rag_variants_answers_v4_retry.json`) vẫn 429 đúng 8 ca đó, nên đây là
  hạn mức free tier, không phải lỗi pipeline. Trên 38 ca quyết định:
  37/38 đã trả lời (97,4% so với 40/41 = 97,6% của V0) và 30/38 khớp chuỗi con
  (78,9% so với 78,0%) — **không có bằng chứng hồi quy**, nhưng **không thể kết
  luận** cho tới khi chạy bù đủ 41 ca answerable và 24 ca âm.
- `llm_answer_calls` 53 thay vì 54 vì `rag-055` bị gate chặn ở V2/V3/V4.

## 2. Retrieval và reranker

Nguồn: `eval/report_v2.md` §3.2 và `eval/rerank_config_comparison.md`, cùng
dataset 65 ca (41 answerable).

### Biến thể câu hỏi không dấu (2026-10-30)

Nguồn: `eval/rag_variants_dev.json` (cùng corpus, gate, `top_k`; V0 chạy 3 lần,
sàn nhiễu retrieval = 0 ca).

| Biến thể | Mô tả | Hit@1 | Hit@5 | MRR | False refusal | Gate leak |
|---|---|---:|---:|---:|---:|---:|
| V0 | câu gốc có dấu | 0.8537 (35/41) | 0.9756 | 0.9207 | 1 | 14/24 |
| V1 | bỏ dấu, không xử lý | 0.0000 (0/41) | 0.0000 | 0.0000 | 41 | 0/24 |
| V2 | V1 + LLM khôi phục dấu | 0.8537 (35/41) | 0.9756 | 0.9207 | 1 | 13/24 |
| V3 | V2 + truy xuất kép có điều kiện | 0.8537 (35/41) | 0.9756 | 0.9207 | 1 | 13/24 |
| V4 | V3 + hybrid (vector + BM25 bỏ dấu) | 0.8537 (35/41) | 0.9756 | 0.9268 | 1 | 13/24 |

- V1 sụp hoàn toàn vì `bge-m3` rất nhạy dấu: `rag-001` được 0.6302 khi có dấu
  nhưng chỉ 0.3674 khi bỏ dấu, dưới gate 0.5.
- So với V0, V2/V3/V4 chỉ đổi **1 ca**: `rag-055` (nhóm âm) từ lọt gate thành bị
  chặn, tức cải thiện; **không ca answerable nào đổi**. V3 không khác V2 vì mọi
  match của câu không dấu đều dưới gate.
- V4 tăng MRR nhờ hạ trọng số BM25 trong RRF từ `(0.5, 0.5)` xuống `(0.2, 0.8)`;
  đo quét trọng số cho Hit@1 26/41 → 35/41. BM25 chỉ dùng khi
  `search_mode=hybrid`, production hiện vẫn `vector`.
- Đây là số development trên cùng corpus đã dùng để chọn ngưỡng, không phải
  held-out.

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

`POST /api/rag/ask` và `/api/rag/ask/stream` gộp nguồn theo tài liệu (mỗi tài
liệu một dòng, giữ chunk điểm cao nhất) và báo **mọi tài liệu được truy xuất**
đã nằm trong ngữ cảnh; không còn cắt theo biên 0,10 điểm hay tối đa 3 nguồn.
`top_k` vẫn là số đoạn trích đưa vào ngữ cảnh, không phải số dòng nguồn hiển thị;
UI tự giới hạn khi cần.

Corpus session có thêm mốc `last_seen` lưu trong PostgreSQL (bảng
`rag_corpus_activity`). Khi khởi động lại, sweep chỉ xoá corpus `session_*` có
`last_seen` cũ hơn TTL phiên; bản ghi thiếu mốc được backfill bằng thời điểm
quét. `clear_session`/dọn phiên xoá cả chunks lẫn mốc `last_seen`. Lịch sử hội
thoại vẫn chỉ nằm trong RAM, nên nhiều instance vẫn cần sticky session hoặc
session store dùng chung.

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
.venv\Scripts\python.exe -m pytest -q                       # 400 test thu thập
.venv\Scripts\python.exe -m scripts.check_current_numbers   # đối chiếu bảng này với artifact
```
