"""Script kiểm tra thực tế (Live Verification) các tools của MCP Server.

Gọi trực tiếp vào 2 công cụ MCP:
1. search_documents: Truy xuất qua RAG pipeline
2. query_data: Truy xuất qua NLQ Text-to-SQL pipeline và database
"""

import asyncio
import json
import os
import sys

from app.mcp_server import mcp


async def main() -> None:
    print("=" * 60)
    print("KIỂM TRA THỰC TẾ MCP SERVER (TASK-AI19)")
    print("=" * 60)

    # 1. Liệt kê tools đã đăng ký
    tools = await mcp.list_tools()
    print(f"\n[1] Danh sách tools đã đăng ký ({len(tools)} tools):")
    for t in tools:
        print(f"  - {t.name}: {t.description[:80]}...")

    # 2. Gọi tool query_data với dữ liệu thật từ CSDL
    print("\n[2] Gọi tool 'query_data' (NLQ Text-to-SQL):")
    print("  -> Câu hỏi: 'Có bao nhiêu sản phẩm trong hệ thống?'")
    try:
        res = await mcp.call_tool("query_data", {"query": "Có bao nhiêu sản phẩm trong hệ thống?"})
        text = res.content[0].text if res.content else "{}"
        parsed = json.loads(text)
        print(f"  <- Trạng thái: {parsed.get('status')}")
        print(f"  <- SQL sinh ra: {parsed.get('sql')}")
        print(f"  <- Dữ liệu: {parsed.get('rows')}")
        print(f"  <- Câu trả lời: {parsed.get('answer')}")
    except Exception as e:
        print(f"  <- Lỗi: {e}")

    # 3. Gọi tool search_documents với tài liệu thật từ RAG
    print("\n[3] Gọi tool 'search_documents' (RAG Documents):")
    print("  -> Câu hỏi: 'Quy định bảo hành và đổi trả thế nào?'")
    try:
        res = await mcp.call_tool("search_documents", {"query": "Quy định bảo hành và đổi trả thế nào?", "top_k": 2})
        text = res.content[0].text if res.content else "{}"
        parsed = json.loads(text)
        print(f"  <- Câu trả lời: {parsed.get('answer')[:120]}...")
        print(f"  <- Số nguồn trích dẫn: {len(parsed.get('sources', []))}")
        for s in parsed.get("sources", []):
            print(f"     * Nguồn: {s.get('source')} (Trang {s.get('page')}, Score: {s.get('score')})")
    except Exception as e:
        print(f"  <- Lỗi: {e}")

    print("\n" + "=" * 60)
    print("HOÀN TẤT KIỂM TRA MCP SERVER")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
