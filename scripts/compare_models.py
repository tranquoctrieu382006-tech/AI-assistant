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


def run_benchmark():
    print("=" * 70)
    print("🚀 BẮT ĐẦU CHƯƠNG TRÌNH BENCHMARK SO SÁNH CÁC MÔ HÌNH LLM")
    print("=" * 70)

    langfuse_active = is_langfuse_configured()
    if langfuse_active:
        print(f"✅ Langfuse tracing: ĐÃ KÍCH HOẠT (Host: {settings.langfuse_base_url or settings.langfuse_host})")
    else:
        print("⚠️ Langfuse tracing: CHƯA CÓ KEY (Chạy benchmark cục bộ, không gửi trace)")

    roles_to_test = ["local"]
    if settings.openai_api_key and not settings.openai_api_key.startswith("sk-..."):
        roles_to_test.extend(["fast", "strong"])
    else:
        print("ℹ️ Đang chạy với role 'local' (Ollama). Khi có OpenAI/Google API key, sẽ tự test thêm fast/strong.")

    results: list[dict[str, Any]] = []
    session_id = f"benchmark-session-{int(time.time())}"
    user_id = "intern_tran_quoc_trieu"

    handler = get_langfuse_handler()

    for prompt_item in BENCHMARK_PROMPTS:
        p_id = prompt_item["id"]
        p_cat = prompt_item["category"]
        p_text = prompt_item["prompt"]

        print(f"\n📌 [{p_id}] [{p_cat}]")
        print(f"Câu hỏi: {p_text}")
        print("-" * 70)

        for role in roles_to_test:
            model_name = MODEL_REGISTRY.get(role, role)
            print(f"▶ Đang gọi role='{role}' (model='{model_name}')...", end="", flush=True)

            tags = ["benchmark", role, p_cat]
            metadata = {
                "prompt_id": p_id,
                "category": p_cat,
                "model_role": role,
                "model_name": model_name,
            }

            try:
                llm = get_llm(role, temperature=0.0)
                callbacks = [handler] if handler else []

                start_time = time.time()
                with langfuse_context(
                    session_id=session_id,
                    user_id=user_id,
                    tags=tags,
                    metadata=metadata,
                ):
                    response = llm.invoke(p_text, config={"callbacks": callbacks} if callbacks else None)
                duration = time.time() - start_time

                content = response.content if hasattr(response, "content") else str(response)

                tokens_in = 0
                tokens_out = 0
                if hasattr(response, "usage_metadata") and response.usage_metadata:
                    tokens_in = response.usage_metadata.get("input_tokens", 0)
                    tokens_out = response.usage_metadata.get("output_tokens", 0)

                print(f" XONG ({duration:.2f}s, tokens: {tokens_in} in / {tokens_out} out)")
                sample_text = content.replace("\n", " ")[:100]
                print(f"   Trích đoạn câu trả lời: \"{sample_text}...\"")

                results.append({
                    "prompt_id": p_id,
                    "category": p_cat,
                    "role": role,
                    "model": model_name,
                    "duration_sec": round(duration, 2),
                    "tokens_in": tokens_in,
                    "tokens_out": tokens_out,
                    "total_tokens": tokens_in + tokens_out,
                    "status": "SUCCESS",
                })

            except LLMInitError as e:
                print(f" LỖI KHỞI TẠO: {e}")
                results.append({
                    "prompt_id": p_id,
                    "category": p_cat,
                    "role": role,
                    "model": model_name,
                    "duration_sec": 0,
                    "tokens_in": 0,
                    "tokens_out": 0,
                    "total_tokens": 0,
                    "status": f"INIT_ERROR: {e}",
                })
            except Exception as e:
                print(f" LỖI RUNTIME: {e}")
                results.append({
                    "prompt_id": p_id,
                    "category": p_cat,
                    "role": role,
                    "model": model_name,
                    "duration_sec": 0,
                    "tokens_in": 0,
                    "tokens_out": 0,
                    "total_tokens": 0,
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

    return results


if __name__ == "__main__":
    run_benchmark()
