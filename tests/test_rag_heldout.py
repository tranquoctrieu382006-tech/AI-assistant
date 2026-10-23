import json
from collections import Counter
from pathlib import Path

from scripts.evaluate_rag_heldout import score_cases
from scripts.validate_golden_data import validate_rag_cases


ROOT = Path(__file__).resolve().parents[1]


def test_heldout_set_has_ten_cases_per_category_and_valid_evidence():
    cases = json.loads((ROOT / "eval/rag_heldout_m1.json").read_text(encoding="utf-8"))
    assert len(cases) == 50
    assert Counter(case["category"] for case in cases) == {
        "simple": 10,
        "paraphrase": 10,
        "multi_document": 10,
        "out_of_scope_far": 10,
        "in_scope_missing": 10,
    }
    assert validate_rag_cases(cases) == []


def test_heldout_scorer_emits_only_aggregate_metrics():
    cases = [
        {
            "id": "answerable", "category": "simple", "question": "Evidence?",
            "source": "a.md", "answer_contains": "the required passage",
            "evidence": [{"source": "a.md", "text": "the required passage"}],
        },
        {
            "id": "missing", "category": "in_scope_missing", "question": "Missing?",
            "source": None, "answer_contains": "abstain",
        },
        {
            "id": "far", "category": "out_of_scope_far", "question": "Far?",
            "source": None, "answer_contains": "abstain",
        },
    ]

    def retriever(question, k):
        assert k == 5
        if question == "Evidence?":
            return [{"source": "a.md", "chunk_index": 0, "content": "the required passage", "score": 0.8}]
        if question == "Missing?":
            return [{"source": "neighbor.md", "chunk_index": 0, "content": "related", "score": 0.7}]
        return [{"source": "other.md", "chunk_index": 0, "content": "unrelated", "score": 0.3}]

    result = score_cases(cases, retriever)
    assert result["primary_metric"]["estimate"] == 1
    assert result["risk_metric"]["by_abstention_type"]["in_scope_missing"]["false_positive_count"] == 1
    assert result["risk_metric"]["by_abstention_type"]["out_of_scope_far"]["false_positive_rate"] is None
    assert result["per_case_output"] == "omitted"
    assert not {"id", "question", "case_results"} & set(result)
