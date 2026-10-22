"""
Kiểm tra kết nối và gọi thử mô hình local (Ollama).
"""

import sys
import io

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from app.core.llm import get_llm, verify_model, check_ollama_status

if __name__ == "__main__":
    print("=" * 60)
    print("1. Kiểm tra trạng thái Ollama:")
    status = check_ollama_status()
    print("   Status:", status)

    print("\n2. Gọi verify_model('local'):")
    result = verify_model("local")
    print("   Role:", result.get("role"))
    print("   Model:", result.get("model"))
    print("   Status:", result.get("status"))
    print("   Response preview:", result.get("response_preview"))
    print("=" * 60)
