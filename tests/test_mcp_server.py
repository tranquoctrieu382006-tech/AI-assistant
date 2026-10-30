"""Unit tests for Enterprise AI Assistant MCP Server."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from app.mcp_server import mcp
from app.schemas.rag import RagAskResponse, RagSource


@pytest.mark.asyncio
async def test_mcp_server_lists_tools() -> None:
    """Kiểm tra MCP Server đăng ký đủ 2 tools chuẩn: search_documents và query_data."""
    tools = await mcp.list_tools()
    tool_names = [tool.name for tool in tools]
    assert "search_documents" in tool_names
    assert "query_data" in tool_names

    # Kiểm tra mô tả tool
    search_tool = next(t for t in tools if t.name == "search_documents")
    assert "RAG" in (search_tool.description or "")

    query_tool = next(t for t in tools if t.name == "query_data")
    assert "NLQ" in (query_tool.description or "") or "SQL" in (query_tool.description or "")


@pytest.mark.asyncio
async def test_mcp_search_documents_success() -> None:
    """Kiểm tra gọi tool search_documents trả về định dạng JSON đúng schema."""
    fake_rag_resp = RagAskResponse(
        answer="Bảo hành 12 tháng kể từ ngày mua.",
        sources=[
            RagSource(
                source="chinh_sach_bao_hanh.pdf",
                chunk_index=0,
                score=0.9123,
                page_number=3,
                section="Thời hạn bảo hành",
            )
        ],
        trace_id="test-trace-mcp",
    )

    with patch("app.mcp_server.answer_rag_request", new=AsyncMock(return_value=fake_rag_resp)):
        result = await mcp.call_tool("search_documents", {"query": "Quy định bảo hành như thế nào?"})
        assert not result.is_error
        text = result.content[0].text
        data = json.loads(text)
        assert data["answer"] == "Bảo hành 12 tháng kể từ ngày mua."
        assert len(data["sources"]) == 1
        assert data["sources"][0]["source"] == "chinh_sach_bao_hanh.pdf"
        assert data["sources"][0]["page"] == 3


@pytest.mark.asyncio
async def test_mcp_query_data_success() -> None:
    """Kiểm tra gọi tool query_data trả về kết quả số liệu, SQL và danh sách bản ghi."""
    fake_nlq_result = {
        "answer": "Có tổng cộng 150 sản phẩm trong hệ thống.",
        "sql": "SELECT count(*) FROM products;",
        "columns": ["count"],
        "rows": [[150]],
        "status": "success",
    }

    with patch("app.mcp_server.ask_nlq", return_value=fake_nlq_result):
        result = await mcp.call_tool("query_data", {"query": "Có bao nhiêu sản phẩm?"})
        assert not result.is_error
        text = result.content[0].text
        data = json.loads(text)
        assert data["answer"] == "Có tổng cộng 150 sản phẩm trong hệ thống."
        assert data["sql"] == "SELECT count(*) FROM products;"
        assert data["columns"] == ["count"]
        assert data["rows"] == [[150]]
        assert data["status"] == "success"
