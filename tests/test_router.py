"""
Unit tests for the Router module using FakeListChatModel (no real LLM or API keys).
"""

import pytest
from langchain_community.chat_models import FakeListChatModel

from app.core.router import (
    RouteCategory,
    RouteDecision,
    route_question,
    _rule_based_classify,
)


def test_rule_based_rag_classification():
    """Kiểm tra phân loại chính sách, tài liệu vào RAG."""
    rag_samples = [
        "Chính sách đổi trả hàng áp dụng trong bao nhiêu ngày?",
        "Quy định bảo hành sản phẩm điện tử như thế nào?",
        "Điều kiện để được miễn phí vận chuyển là gì?",
        "Thủ tục hoàn tiền cho khách hàng cần những giấy tờ gì?",
        "Tài liệu hướng dẫn sử dụng phần mềm quản lý kho",
    ]
    for question in rag_samples:
        decision = _rule_based_classify(question)
        assert decision.category == RouteCategory.RAG, f"Failed on: {question}"


def test_rule_based_nlq_classification():
    """Kiểm tra phân loại câu hỏi số liệu, bảng biểu vào NLQ."""
    nlq_samples = [
        "Tổng doanh thu tháng 9 năm 2026 là bao nhiêu?",
        "Top 5 khách hàng mua nhiều nhất quý vừa qua?",
        "Sản phẩm nào có số lượng tồn kho thấp nhất?",
        "Đơn hàng nào có giá trị cao nhất trong tháng 8?",
        "Có bao nhiêu đơn hàng bị hủy trong tuần này?",
    ]
    for question in nlq_samples:
        decision = _rule_based_classify(question)
        assert decision.category == RouteCategory.NLQ, f"Failed on: {question}"


def test_rule_based_out_of_scope():
    """Kiểm tra phân loại chào hỏi và câu ngoài phạm vi vào OUT_OF_SCOPE."""
    oos_samples = [
        "Xin chào",
        "Hello bạn",
        "Bạn là ai?",
        "",
        "   ",
        "abc",
    ]
    for question in oos_samples:
        decision = _rule_based_classify(question)
        assert decision.category == RouteCategory.OUT_OF_SCOPE, f"Failed on: {question}"


def test_router_with_fake_chat_model_rag():
    """Kiểm tra router tích hợp FakeListChatModel của LangChain trả về RAG."""
    fake_llm = FakeListChatModel(responses=['{"category": "rag", "reasoning": "Hỏi về bảo hành"}'])
    decision = route_question("Bảo hành tủ lạnh mấy năm?", llm=fake_llm)
    assert decision.category == RouteCategory.RAG


def test_router_with_fake_chat_model_nlq():
    """Kiểm tra router tích hợp FakeListChatModel của LangChain trả về NLQ."""
    fake_llm = FakeListChatModel(responses=['{"category": "nlq", "reasoning": "Hỏi doanh thu"}'])
    decision = route_question("Doanh thu hôm nay bao nhiêu?", llm=fake_llm)
    assert decision.category == RouteCategory.NLQ


def test_router_with_fake_chat_model_out_of_scope():
    """Kiểm tra router tích hợp FakeListChatModel trả về OUT_OF_SCOPE."""
    fake_llm = FakeListChatModel(responses=['{"category": "out_of_scope", "reasoning": "Chào hỏi"}'])
    decision = route_question("Chào buổi sáng!", llm=fake_llm)
    assert decision.category == RouteCategory.OUT_OF_SCOPE


def test_router_empty_input():
    """Kiểm tra router xử lý chuỗi rỗng an toàn mà không gọi model."""
    decision = route_question("")
    assert decision.category == RouteCategory.OUT_OF_SCOPE
    assert "rỗng" in decision.reasoning.lower()
