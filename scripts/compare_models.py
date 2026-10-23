"""
Script so sánh các mô hình LLM và đẩy trace về Langfuse.

Đo lường:
- Độ trễ (latency)
- Số lượng token (input, output, total)
- Chi phí ước tính (cost)
- Tỉ lệ thành công và chất lượng phản hồi
"""

import io
import sys
import time
import csv
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any

# Đảm bảo in UTF-8 không lỗi trên Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from app.core.config import settings
from app.core.llm import get_llm, MODEL_REGISTRY, LLMInitError
from app.core.tracing import (
    get_langfuse_handler,
    is_langfuse_configured,
    langfuse_context,
    init_tracing,
    flush_langfuse,
)

BENCHMARK_PROMPTS = [
    {
        "id": "prompt-1",
        "category": "RAG-Concept",
        "prompt": "Giải thích ngắn gọn cơ chế RAG (Retrieval-Augmented Generation) và lợi ích lớn nhất của nó trong 3 câu.",
    },
    {
        "id": "prompt-2",
        "category": "NLQ-SQL",
        "prompt": "Cho bảng orders(id, customer_id, order_date, total_amount). Viết câu lệnh SQL lấy top 3 khách hàng chi tiêu nhiều nhất trong năm 2026.",
    },
    {
        "id": "prompt-3",
        "category": "Reasoning-Comparison",
        "prompt": "So sánh ưu và nhược điểm khi dùng PostgreSQL + pgvector so với Pinecone cho hệ thống RAG nội bộ.",
    },
]


def _response_text(response: Any) -> str:
    """Normalize string and multimodal/text-block response content."""
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(parts)
    return str(content)


def _write_results(results: list[dict[str, Any]], session_id: str) -> Path:
    """Persist a CSV snapshot so the benchmark can be reviewed after the run."""
    output_dir = Path("docs") / "benchmark_runs"
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"benchmark_{stamp}.csv"
    fields = [
        "session_id", "prompt_id", "category", "role", "model",
        "duration_sec", "samples", "tokens_in", "tokens_out", "total_tokens", "cost_usd", "status",
    ]
    with output_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in results:
            writer.writerow({"session_id": session_id, **row})
    return output_path


def run_benchmark():
    init_tracing()
    print("=" * 70)
    print("🚀 BẮT ĐẦU CHƯƠNG TRÌNH BENCHMARK SO SÁNH CÁC MÔ HÌNH LLM")
    print("=" * 70)

    langfuse_active = is_langfuse_configured()
    if langfuse_active:
        print(f"✅ Langfuse tracing: ĐÃ KÍCH HOẠT (Host: {settings.langfuse_host})")
    else:
        print("⚠️ Langfuse tracing: CHƯA CÓ KEY (Chạy benchmark cục bộ, không gửi trace)")

    roles_to_test = ["local"]
    provider_keys = {
        "openai": settings.openai_api_key,
        "google_genai": settings.google_api_key,
        "groq": settings.groq_api_key,
    }
    for role in ("fast", "strong"):
        model_name = MODEL_REGISTRY.get(role, "")
        provider = model_name.split("/", 1)[0] if "/" in model_name else ""
        api_key = provider_keys.get(provider, "")
        if api_key and not api_key.endswith("..."):
            roles_to_test.append(role)
        else:
            print(f"ℹ️ Bỏ qua role '{role}': chưa có API key hợp lệ cho provider '{provider or 'mặc định'}'.")

    results: list[dict[str, Any]] = []
    unavailable_roles: set[str] = set()
    warmed_roles: set[str] = set()
    session_id = f"benchmark-session-{int(time.time())}"

    for prompt_item in BENCHMARK_PROMPTS:
        p_id = prompt_item["id"]
        p_cat = prompt_item["category"]
        p_text = prompt_item["prompt"]

        print(f"\n📌 [{p_id}] [{p_cat}]")
        print(f"Câu hỏi: {p_text}")
        print("-" * 70)

        for role in roles_to_test:
            model_name = MODEL_REGISTRY.get(role, role)
            if role in unavailable_roles:
                results.append({
                    "prompt_id": p_id, "category": p_cat, "role": role,
                    "model": model_name, "duration_sec": 0, "tokens_in": 0,
                    "tokens_out": 0, "total_tokens": 0,
                    "status": "SKIPPED: lỗi cấu hình/quota ở lượt trước",
                })
                continue
            print(f"▶ Đang gọi role='{role}' (model='{model_name}')...", end="", flush=True)

            tags = ["benchmark", role, p_cat]
            metadata = {
                "prompt_id": p_id,
                "category": p_cat,
                "model_role": role,
                "model_name": model_name,
            }

            try:
                llm = get_llm(role, temperature=0.0, timeout=300.0 if role == "local" else 120.0)
                if role not in warmed_roles:
                    llm.invoke(p_text)  # warm-up excluded from measured samples and tracing
                    warmed_roles.add(role)
                samples = []
                durations = []
                for _ in range(3):
                    # A fresh callback instance keeps each request's trace isolated.
                    handler = get_langfuse_handler()
                    callbacks = [handler] if handler else []
                    start_time = time.perf_counter()
                    with langfuse_context(session_id=session_id, tags=tags, metadata=metadata):
                        response = llm.invoke(p_text, config={"callbacks": callbacks} if callbacks else None)
                    durations.append(time.perf_counter() - start_time)
                    usage = getattr(response, "usage_metadata", None) or {}
                    samples.append((usage.get("input_tokens", 0), usage.get("output_tokens", 0)))
                duration = statistics.median(durations)

                content = _response_text(response)

                tokens_in = round(statistics.mean(pair[0] for pair in samples))
                tokens_out = round(statistics.mean(pair[1] for pair in samples))

                model_id = model_name.split("/", 1)[1] if model_name.startswith("groq/") else ""
                groq_rates = {
                    "openai/gpt-oss-120b": (0.15, 0.60),
                    "openai/gpt-oss-20b": (0.075, 0.30),
                }
                rates = groq_rates.get(model_id)
                cost_usd = round((tokens_in * rates[0] + tokens_out * rates[1]) / 1_000_000, 8) if rates else None

                cost_label = f", cost≈${cost_usd:.6f}" if cost_usd is not None else ""
                print(f" XONG ({duration:.2f}s, tokens: {tokens_in} in / {tokens_out} out{cost_label})")
                sample_text = content.replace("\n", " ")[:100]
                print(f"   Trích đoạn câu trả lời: \"{sample_text}...\"")

                results.append({
                    "prompt_id": p_id,
                    "category": p_cat,
                    "role": role,
                    "model": model_name,
                    "duration_sec": round(duration, 2),
                    "samples": len(samples),
                    "tokens_in": tokens_in,
                    "tokens_out": tokens_out,
                    "total_tokens": tokens_in + tokens_out,
                    "cost_usd": cost_usd,
                    "status": "SUCCESS",
                })

            except LLMInitError as e:
                print(f" LỖI KHỞI TẠO: {e}")
                unavailable_roles.add(role)
                results.append({
                    "prompt_id": p_id,
                    "category": p_cat,
                    "role": role,
                    "model": model_name,
                    "duration_sec": 0,
                    "tokens_in": 0,
                    "tokens_out": 0,
                    "total_tokens": 0,
                    "cost_usd": None,
                    "status": f"INIT_ERROR: {e}",
                })
            except Exception as e:
                print(f" LỖI RUNTIME: {e}")
                if any(token in str(e).lower() for token in ("resource_exhausted", "quota", "api key required", "authentication")):
                    unavailable_roles.add(role)
                results.append({
                    "prompt_id": p_id,
                    "category": p_cat,
                    "role": role,
                    "model": model_name,
                    "duration_sec": 0,
                    "tokens_in": 0,
                    "tokens_out": 0,
                    "total_tokens": 0,
                    "cost_usd": None,
                    "status": f"RUN_ERROR: {e}",
                })

    # Flush toàn bộ trace lên Langfuse Cloud
    print("\n⏳ Đang flush traces lên Langfuse Cloud...")
    flush_langfuse()
    print("✅ Đã flush traces thành công!")

    # In bảng tổng hợp
    print("\n" + "=" * 70)
    print("📊 BẢNG TỔNG HỢP KẾT QUẢ BENCHMARK")
    print("=" * 70)
    print(f"{'Role':<8} | {'Model':<20} | {'Category':<15} | {'Latency':<8} | {'Tokens':<10} | {'Status'}")
    print("-" * 70)
    for r in results:
        tok_str = f"{r['tokens_in']}+{r['tokens_out']}"
        lat_str = f"{r['duration_sec']}s"
        print(f"{r['role']:<8} | {r['model']:<20} | {r['category']:<15} | {lat_str:<8} | {tok_str:<10} | {r['status']}")

    output_path = _write_results(results, session_id)
    print(f"\n💾 Đã lưu kết quả CSV: {output_path}")

    return results


if __name__ == "__main__":
    run_benchmark()
