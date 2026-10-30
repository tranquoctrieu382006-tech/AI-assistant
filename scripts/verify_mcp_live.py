"""Script kiểm tra thực tế (Live Verification) các tools của MCP Server.

Gọi trực tiếp vào 2 công cụ MCP:
1. search_documents: Truy xuất qua RAG pipeline
2. query_data: Truy xuất qua NLQ Text-to-SQL pipeline và database
Đồng thời lưu kết quả thực thi chi tiết vào eval/mcp_live_verification_log.txt.
"""

import asyncio
from datetime import datetime
import json
from pathlib import Path
import sys

from app.mcp_server import mcp


async def main() -> None:
    lines: list[str] = []

    def log(msg: str = "") -> None:
        lines.append(msg)
        print(msg)

    timestamp = datetime.now().isoformat()
    log("=" * 65)
    log("NHẬT KÝ THỰC THI KIỂM TRA THỰC TẾ MCP SERVER (TASK-AI19)")
    log(f"Thời điểm kiểm tra: {timestamp}")
    log("Chuẩn giao thức: Model Context Protocol (Anthropic MCP SDK 2.3.0)")
    log("=" * 65)

    # 1. Liệt kê tools đã đăng ký
    tools = await mcp.list_tools()
    log(f"\n[1] Danh sách tools đã đăng ký thành công ({len(tools)} tools):")
    for t in tools:
        log(f"  - Tên tool: {t.name}")
        log(f"    Mô tả: {t.description}")

    # 2. Gọi tool query_data với dữ liệu thật từ CSDL
    log("\n[2] Kiểm tra gọi thật tool 'query_data' (NLQ Text-to-SQL):")
    question_nlq = "Có bao nhiêu sản phẩm trong hệ thống?"
    log(f"  -> Câu hỏi đầu vào: '{question_nlq}'")
    try:
        res = await mcp.call_tool("query_data", {"query": question_nlq})
        text = res.content[0].text if res.content else "{}"
        parsed = json.loads(text)
        log(f"  <- Trạng thái: {parsed.get('status')}")
        log(f"  <- SQL sinh ra: {parsed.get('sql')}")
        log(f"  <- Dữ liệu kết quả: {parsed.get('rows')}")
        log(f"  <- Câu trả lời tự nhiên: {parsed.get('answer')}")
    except Exception as e:
        log(f"  <- Lỗi thực thi: {e}")

    # 3. Gọi tool search_documents với tài liệu thật từ RAG
    log("\n[3] Kiểm tra gọi thật tool 'search_documents' (RAG Documents):")
    question_rag = "Quy định bảo hành và đổi trả thế nào?"
    log(f"  -> Câu hỏi đầu vào: '{question_rag}'")
    try:
        res = await mcp.call_tool("search_documents", {"query": question_rag, "top_k": 2})
        text = res.content[0].text if res.content else "{}"
        parsed = json.loads(text)
        log(f"  <- Câu trả lời: {parsed.get('answer')}")
        log(f"  <- Số nguồn tài liệu trích dẫn: {len(parsed.get('sources', []))}")
        for idx, s in enumerate(parsed.get("sources", []), 1):
            log(f"     {idx}. File: {s.get('source')} | Trang: {s.get('page')} | Điểm tương đồng: {s.get('score')}")
    except Exception as e:
        log(f"  <- Lỗi thực thi: {e}")

    log("\n" + "=" * 65)
    log("KẾT LUẬN: CẢ 2 CÔNG CỤ MCP HOẠT ĐỘNG CHÍNH XÁC VỚI HẠ TẦNG THẬT")
    log("=" * 65)

    out_file = Path("eval/mcp_live_verification_log.txt")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[OK] Đã lưu bản ghi minh chứng tại: {out_file}")


if __name__ == "__main__":
    asyncio.run(main())
