"""Model Context Protocol (MCP) Server for Enterprise AI Assistant.

Exposes RAG (document search) and NLQ (database text-to-SQL query) capabilities
as standard MCP Tools conforming to the Model Context Protocol specification.
Compatible with external AI clients like Claude Desktop, Cursor, and Antigravity IDE.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

try:
    from mcp.server.mcpserver import MCPServer
except ImportError:
    from mcp.server.fastmcp import FastMCP as MCPServer

from app.schemas.rag import RagAskRequest
from app.core.rag_service import answer_rag_request
from app.core.nlq_graph import ask_nlq

logger = logging.getLogger("mcp_server")

# Initialize the MCP Server
mcp = MCPServer("Enterprise-AI-Assistant")


@mcp.tool()
async def search_documents(query: str, top_k: int = 5) -> str:
    """Tra cứu tài liệu quy trình, quy định bảo hành, đổi trả và chính sách nội bộ thông qua RAG (Retrieval-Augmented Generation).

    Args:
        query: Câu hỏi hoặc từ khóa cần tra cứu trong tài liệu nội bộ.
        top_k: Số lượng đoạn trích liên quan tối đa cần truy xuất (mặc định: 5).

    Returns:
        Câu trả lời tổng hợp kèm theo danh sách nguồn tài liệu trích dẫn chi tiết (tên file, số trang, điểm tương đồng).
    """
    req = RagAskRequest(question=query, top_k=top_k)
    resp = await answer_rag_request(req)

    result: dict[str, Any] = {
        "answer": resp.answer,
        "sources": [
            {
                "source": s.source,
                "page": s.page_number,
                "score": round(s.score, 4),
                "section": s.section,
            }
            for s in resp.sources
        ],
    }
    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool()
async def query_data(query: str) -> str:
    """Truy vấn dữ liệu số, báo cáo doanh thu, sản phẩm và khách hàng từ cơ sở dữ liệu quan hệ (NLQ Text-to-SQL).

    Args:
        query: Câu hỏi tự nhiên bằng tiếng Việt yêu cầu trích xuất số liệu từ CSDL bán hàng.

    Returns:
        Câu trả lời giải thích số liệu kèm câu lệnh SQL an toàn đã sinh và dữ liệu chi tiết dạng bảng (rows/columns).
    """
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, ask_nlq, query)

    rows = result.get("rows", [])
    formatted: dict[str, Any] = {
        "answer": result.get("answer", ""),
        "sql": result.get("sql", ""),
        "status": result.get("status", "success"),
        "columns": result.get("columns", []),
        "rows": rows[:20] if isinstance(rows, list) else [],
        "total_rows": len(rows) if isinstance(rows, list) else 0,
    }
    return json.dumps(formatted, ensure_ascii=False, indent=2)


def main() -> None:
    """Run the MCP server over stdio transport."""
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting Enterprise AI Assistant MCP Server...")
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
