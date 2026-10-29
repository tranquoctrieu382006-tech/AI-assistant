"""
Unit tests for data contracts, Pydantic schemas, and response formatting.
"""

import pytest
from pydantic import ValidationError

from app.schemas.rag import RagAskRequest, RagAskResponse, RagSource
from app.schemas.nlq import NlqAskRequest, NlqAskResponse
from app.core.router import RouteDecision, RouteCategory


def test_rag_ask_request_valid():
    """Kiểm tra khởi tạo RagAskRequest hợp lệ."""
    req = RagAskRequest(question="Chính sách bảo hành?", top_k=5, session_id="sess-123")
    assert req.question == "Chính sách bảo hành?"
    assert req.top_k == 5
    assert req.session_id == "sess-123"


def test_rag_ask_request_invalid_length():
    """Kiểm tra RagAskRequest từ chối câu hỏi quá ngắn."""
    with pytest.raises(ValidationError):
        RagAskRequest(question="a")


def test_rag_ask_response_structure():
    """Kiểm tra cấu trúc và serialize của RagAskResponse."""
    src = RagSource(
        source="04_chinh_sach_doi_tra.docx",
        chunk_index=2,
        score=0.88,
        section="Điều 1",
        page_number=1,
    )
    res = RagAskResponse(
        answer="Thời hạn đổi trả hàng là 7 ngày.",
        sources=[src],
    )
    data = res.model_dump()
    assert data["answer"] == "Thời hạn đổi trả hàng là 7 ngày."
    assert len(data["sources"]) == 1
    assert data["sources"][0]["score"] == 0.88
    assert data["sources"][0]["section"] == "Điều 1"


def test_nlq_ask_request_defaults():
    """Kiểm tra NlqAskRequest nhận diện đúng các trường mặc định."""
    req = NlqAskRequest(question="Doanh thu tháng 9?")
    assert req.question == "Doanh thu tháng 9?"
    assert req.session_id is None
    assert req.user_id is None


def test_nlq_ask_response_with_tabular_data():
    """Kiểm tra cấu trúc NlqAskResponse cho kết quả bảng số liệu."""
    res = NlqAskResponse(
        sql="SELECT customer_name, total_spent FROM customers ORDER BY total_spent DESC LIMIT 5;",
        columns=["customer_name", "total_spent"],
        rows=[
            {"customer_name": "Công ty A", "total_spent": 150000000},
            {"customer_name": "Công ty B", "total_spent": 95000000},
        ],
        answer="Top khách hàng chi tiêu nhiều nhất gồm Công ty A và Công ty B.",
        repaired=False,
        truncated=False,
    )
    dumped = res.model_dump()
    assert len(dumped["rows"]) == 2
    assert dumped["columns"] == ["customer_name", "total_spent"]
    assert dumped["repaired"] is False


def test_route_decision_schema():
    """Kiểm tra schema RouteDecision."""
    decision = RouteDecision(category=RouteCategory.RAG, reasoning="Hỏi chính sách")
    assert decision.category == "rag"
    assert decision.reasoning == "Hỏi chính sách"
