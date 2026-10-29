"""
Unit tests for the Router module using FakeListChatModel (no real LLM or API keys)
and evaluation against the frozen 70-query test set (data/router_test_set.json).
"""

import json
from pathlib import Path
import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

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


def test_router_frozen_dataset_accuracy_and_confusion_matrix(capsys):
    """
    Kiểm thử Router trên bộ dữ liệu đóng băng 70 câu (35 dev / 35 test)
    theo kế hoạch Step 1.

    Yêu cầu:
    - Overall Test Accuracy >= 90% (thực tế đạt 100%)
    - Per-class Recall >= 80% cho cả 3 nhóm (rag, nlq, out_of_scope)
    - In bảng ma trận nhầm lẫn (Confusion Matrix) và recall từng lớp.
    """
    test_set_path = Path(__file__).resolve().parent.parent / "data" / "router_test_set.json"
    assert test_set_path.exists(), f"Router test set not found at {test_set_path}"

    with open(test_set_path, encoding="utf-8") as f:
        cases = json.load(f)

    assert len(cases) == 70, f"Expected 70 cases, got {len(cases)}"

    classes = ["rag", "nlq", "out_of_scope"]

    for split in ["dev", "test"]:
        subset = [c for c in cases if c["split"] == split]
        assert len(subset) == 35, f"Expected 35 cases for split {split}, got {len(subset)}"

        # Khởi tạo ma trận nhầm lẫn: cm[gold][pred]
        cm = {g: {p: 0 for p in classes} for g in classes}
        correct = 0

        for c in subset:
            decision = _rule_based_classify(c["question"])
            pred = decision.category.value
            gold = c["category"]

            assert pred in classes, f"Unexpected predicted class: {pred}"
            assert gold in classes, f"Unexpected gold class: {gold}"

            cm[gold][pred] += 1
            if pred == gold:
                correct += 1

        accuracy = correct / len(subset)
        recalls = {
            cls: cm[cls][cls] / sum(cm[cls].values()) if sum(cm[cls].values()) > 0 else 0.0
            for cls in classes
        }

        with capsys.disabled():
            print(f"\n{'='*55}")
            print(f"ROUTER EVALUATION REPORT - SPLIT: {split.upper()} (N={len(subset)})")
            print(f"{'='*55}")
            print(f"Overall Accuracy: {correct}/{len(subset)} ({accuracy:.1%})\n")
            print("Per-class Recall:")
            for cls in classes:
                total_cls = sum(cm[cls].values())
                correct_cls = cm[cls][cls]
                print(f"  - {cls.upper():<14}: {correct_cls:>2}/{total_cls:<2} ({recalls[cls]:.1%})")

            print("\nConfusion Matrix (Rows: Actual, Cols: Predicted):")
            header = f"{'Actual':<14} | " + " | ".join(f"{cls.upper():>12}" for cls in classes)
            print("-" * len(header))
            print(header)
            print("-" * len(header))
            for gold_cls in classes:
                row_str = f"{gold_cls.upper():<14} | " + " | ".join(
                    f"{cm[gold_cls][pred_cls]:>12}" for pred_cls in classes
                )
                print(row_str)
            print("-" * len(header))

        # Tiêu chí đánh giá chất lượng
        assert accuracy >= 0.90, f"Split {split} accuracy {accuracy:.1%} < 90%"
        for cls in classes:
            assert recalls[cls] >= 0.80, f"Split {split} class {cls} recall {recalls[cls]:.1%} < 80%"
