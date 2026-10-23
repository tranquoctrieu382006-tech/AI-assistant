from datetime import date
from decimal import Decimal

from scripts.evaluate_nlq_golden import is_read_only_query, results_match
from scripts.evaluate_rag_retrieval import source_matches
from scripts.evaluate_nlq_model import (
    _aggregate_nlq_run,
    _execution_results_match,
    _load_schema_ddl,
    _sql_features,
    upload_langfuse_experiment,
)
from scripts.evaluate_rag_retrieval import _category_metrics, _retrieval_metrics
from scripts.evaluate_rag_experiment import _aggregate_run, _make_evaluators
from scripts.evaluate_rag_experiment import _ensure_dataset
from scripts.evaluate_rag_answers import evaluate as evaluate_rag_answers
from langfuse import Evaluation
from types import SimpleNamespace


def test_nlq_accepts_select_and_with_queries_in_read_only_transaction():
    assert is_read_only_query("SELECT 1")
    assert is_read_only_query("-- note\n WITH rows AS (SELECT 1) SELECT * FROM rows")
    assert not is_read_only_query("DELETE FROM products")


def test_nlq_compares_decimal_and_date_values_with_json_values():
    actual = [{"amount": Decimal("1.25"), "sold_on": date(2026, 10, 23)}]
    expected = [{"amount": 1.25, "sold_on": "2026-10-23"}]
    assert results_match(actual, expected)


def test_retrieval_source_matching_uses_full_relative_path():
    assert source_matches("policies/leave.md", "policies/leave.md")
    assert not source_matches("archive/leave.md", "policies/leave.md")
    assert not source_matches("archive/leave.md", "leave.md")


def test_model_sql_comparison_ignores_aliases_and_row_order():
    actual = [{"label": "Bạc", "total": 50}, {"label": "Vàng", "total": 50}]
    expected = [{"member_tier": "Vàng", "customer_count": 50},
                {"member_tier": "Bạc", "customer_count": 50}]
    assert _execution_results_match(actual, expected)


def test_model_sql_comparison_preserves_order_for_order_by_queries():
    actual = [{"name": "Bạc"}, {"name": "Vàng"}]
    expected = [{"tier": "Vàng"}, {"tier": "Bạc"}]
    assert not _execution_results_match(actual, expected, ordered=True)
    assert _execution_results_match(actual, expected, ordered=False)


def test_sql_features_ignore_semicolons_in_literals_and_comments():
    count, ordered = _sql_features(
        "SELECT 'value; still literal' AS note /* ; */ ORDER BY note; -- trailing comment"
    )
    assert count == 1
    assert ordered


def test_sql_features_handle_escaped_and_dollar_quoted_literals():
    assert _sql_features(r"SELECT E'escaped \'; value'") == (1, False)
    assert _sql_features("SELECT $body$semi;colon$body$") == (1, False)


def test_sql_features_detect_multiple_statements_but_ignore_nested_order_by():
    assert _sql_features("SELECT (SELECT 1 ORDER BY 1)") == (1, False)
    assert _sql_features("SELECT 1; SELECT 2")[0] == 2
    assert _sql_features("WITH rows AS (SELECT 2) SELECT * FROM rows ORDER BY 1") == (1, True)


def test_model_evaluator_uses_seed_create_table_ddl():
    schema = _load_schema_ddl()
    assert "CREATE TABLE products" in schema
    assert "CREATE TABLE stock_movements" in schema
    assert "INSERT INTO products" not in schema


def test_rag_golden_set_new_cases_have_no_replacement_question_marks():
    import json
    from pathlib import Path

    cases = json.loads(Path("data/rag_golden_set.json").read_text(encoding="utf-8"))
    for case in cases:
        for field in ("question", "answer_contains"):
            value = case.get(field)
            values = value if isinstance(value, list) else [value]
            assert all(
                not isinstance(item, str) or "?" not in item.rstrip("?")
                for item in values
            ), case["id"]


def test_nlq_langfuse_aggregate_reports_accuracy_and_status_denominators():
    results = [
        SimpleNamespace(output={"status": "pass"}),
        SimpleNamespace(output={"status": "fail"}),
        SimpleNamespace(output={"status": "rejected"}),
        SimpleNamespace(output={"status": "error"}),
    ]

    scores = {
        score.name: score.value
        for score in _aggregate_nlq_run(item_results=results)
    }

    assert scores == {
        "execution_accuracy": 0.25,
        "pass_cases": 1,
        "fail_cases": 1,
        "rejected_cases": 1,
        "error_cases": 1,
    }


def test_nlq_langfuse_upload_replays_saved_results_without_model_calls(monkeypatch):
    from unittest.mock import MagicMock, patch

    result = {
        "model_role": "local",
        "cases": 1,
        "results": [{
            "id": "nlq-001",
            "question": "Có bao nhiêu sản phẩm?",
            "generated_sql": "SELECT COUNT(*) FROM products",
            "status": "pass",
            "expected": [{"count": 80}],
            "actual": [{"count": 80}],
        }],
    }
    item = SimpleNamespace(metadata={"case_id": "nlq-001"})
    experiment = SimpleNamespace(
        item_results=[item],
        experiment_url="https://langfuse.test/experiment",
        run_evaluations=[],
    )
    client = MagicMock()
    client.run_experiment.return_value = experiment
    monkeypatch.setattr("scripts.evaluate_nlq_model.is_langfuse_configured", lambda: True)
    with (
        patch("scripts.evaluate_nlq_model.init_tracing"),
        patch("scripts.evaluate_nlq_model.get_client", return_value=client),
        patch("scripts.evaluate_nlq_model._ensure_nlq_dataset", return_value=[item]),
    ):
        uploaded = upload_langfuse_experiment(result, "test-nlq-dataset")

    call = client.run_experiment.call_args.kwargs
    output = call["task"](item=item)
    score = call["evaluators"][0](output=output)
    aggregate = call["run_evaluators"][0](
        item_results=[SimpleNamespace(output=output)]
    )

    assert uploaded["items"] == 1
    assert output["generated_sql"] == "SELECT COUNT(*) FROM products"
    assert score.value is True
    assert {item.name: item.value for item in aggregate}["execution_accuracy"] == 1
    client.flush.assert_called_once()


def test_retrieval_metrics_require_all_expected_sources_for_multi_document_cases():
    record = {
        "category": "multi_document",
        "expected_sources": ["a.md", "b.md"],
        "ranks": [1],
        "results": [{"source": "a.md"}, {"source": "other.md"}],
    }
    metrics = _retrieval_metrics([record], 5)
    assert metrics["hit@1"] == 0
    assert metrics["hit@3"] == 0
    assert metrics["expected_source_coverage@5"] == 0.5


def test_retrieval_metrics_return_empty_values_for_categories_without_answers():
    metrics = _retrieval_metrics([], 5)
    assert metrics["answerable_cases"] == 0
    assert metrics["hit@5"] is None


def test_category_metrics_include_out_of_scope_abstention_counts():
    metrics = _category_metrics(
        [],
        [
            {"category": "out_of_scope", "correct": True},
            {"category": "out_of_scope", "correct": False},
        ],
        5,
    )
    assert metrics["out_of_scope"]["answerable_cases"] == 0
    assert metrics["out_of_scope"]["abstention_cases"] == 2
    assert metrics["out_of_scope"]["abstention_accuracy"] == 0.5


def test_rag_quality_evaluator_scores_faithfulness_and_relevance():
    class FakeJudge:
        def invoke(self, prompt):
            assert "CONTEXT" in prompt
            return SimpleNamespace(content=(
                '{"faithfulness":{"score":0.9,"reason":"Supported"},'
                '"answer_relevancy":{"score":0.8,"reason":"Direct"}}'
            ))

    answer_contains, abstention, quality = _make_evaluators(FakeJudge(), "groq/judge")
    input_value = {
        "question": "Bao lâu?",
        "context": "Thời hạn là bảy ngày.",
    }
    expected = {"answer_contains": "bảy ngày"}
    output = {"answer": "Khách có bảy ngày.", "latency_ms": 25.0}
    assert answer_contains(output=output, expected_output=expected).value is True
    assert abstention(
        input=input_value, output=output, expected_output=expected,
        metadata={"expected_sources": ["returns.pdf"]},
    ).value is True
    scores = quality(input=input_value, output=output)
    assert [(score.name, score.value) for score in scores] == [
        ("faithfulness", 0.9),
        ("answer_relevancy", 0.8),
    ]
    assert scores[0].metadata == {"judge_model": "groq/judge"}


def test_rag_quality_judge_stops_repeating_calls_after_rate_limit():
    class RateLimitError(Exception):
        status_code = 429

    class FakeJudge:
        def __init__(self):
            self.calls = 0

        def invoke(self, prompt):
            self.calls += 1
            raise RateLimitError()

    judge = FakeJudge()
    _, _, quality = _make_evaluators(judge, "groq/judge")
    input_value = {"question": "Question?", "context": "Evidence"}
    output = {"answer": "Answer"}

    first = quality(input=input_value, output=output)
    second = quality(input=input_value, output=output)

    assert first.name == "judge_rate_limited"
    assert second.name == "judge_skipped_after_rate_limit"
    assert judge.calls == 1


def test_rag_experiment_aggregates_metric_denominators():
    item_result = SimpleNamespace(
        item=SimpleNamespace(metadata={"category": "simple"}),
        output={"latency_ms": 25.0},
        evaluations=[
            Evaluation(name="faithfulness", value=0.8, data_type="NUMERIC"),
            Evaluation(name="answer_contains", value=True, data_type="BOOLEAN"),
        ],
    )

    scores = {
        score.name: score.value
        for score in _aggregate_run(item_results=[item_result])
    }

    assert scores["simple_faithfulness_mean"] == 0.8
    assert scores["simple_faithfulness_count"] == 1
    assert scores["simple_answer_contains_count"] == 1
    assert scores["simple_latency_ms_mean"] == 25.0


def test_langfuse_dataset_creation_is_idempotent_for_existing_items():
    import uuid
    from unittest.mock import MagicMock, patch

    case_id = "rag-001"
    item_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"ai-assistant-rag-v3-{case_id}"))
    existing_item = SimpleNamespace(
        id=item_id,
        input={"question": "question", "context": "frozen context"},
        expected_output={"answer_contains": "answer"},
        metadata={
            "category": "simple",
            "expected_sources": ["policy.md"],
        },
    )
    dataset = SimpleNamespace(items=[existing_item])
    client = MagicMock()
    client.get_dataset.return_value = dataset

    with patch("app.core.vector_store.similarity_search") as search:
        items = _ensure_dataset(
            client,
            "test-rag-dataset",
            [{
                "id": case_id,
                "question": "question",
                "answer_contains": "answer",
                "source": "policy.md",
            }],
        )

    assert items == [existing_item]
    client.create_dataset_item.assert_not_called()
    search.assert_not_called()
    assert client.get_dataset.call_count == 2


def test_langfuse_dataset_rejects_stale_existing_golden_item():
    import uuid
    from unittest.mock import MagicMock, patch

    item_id = str(uuid.uuid5(
        uuid.NAMESPACE_URL, "ai-assistant-rag-v3-rag-001"
    ))
    stale_item = SimpleNamespace(
        id=item_id,
        input={"question": "old question", "context": "old context"},
        expected_output={"answer_contains": "old answer"},
        metadata={"category": "simple", "expected_sources": ["policy.md"]},
    )
    client = MagicMock()
    client.get_dataset.return_value = SimpleNamespace(items=[stale_item])

    with patch("app.core.vector_store.similarity_search") as search:
        try:
            _ensure_dataset(
                client,
                "test-rag-dataset",
                [{
                    "id": "rag-001",
                    "question": "corrected question",
                    "answer_contains": "correct answer",
                    "source": "policy.md",
                }],
            )
        except RuntimeError as exc:
            assert "new dataset version/name" in str(exc)
        else:
            raise AssertionError("Stale Langfuse item should be rejected")

    search.assert_not_called()
    client.create_dataset_item.assert_not_called()


def test_rag_answer_smoke_evaluator_authenticates_to_protected_api(tmp_path, monkeypatch):
    from unittest.mock import MagicMock, patch
    import json

    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "rag_golden_set.json").write_text(
        json.dumps([{
            "id": "case-1",
            "category": "simple",
            "question": "Question?",
            "answer_contains": "answer",
        }]),
        encoding="utf-8",
    )
    monkeypatch.setattr("scripts.evaluate_rag_answers.settings.app_api_key", "x" * 40)
    response = MagicMock()
    response.json.return_value = {"answer": "Expected answer"}
    client = MagicMock()
    client.__enter__.return_value = client
    client.post.return_value = response
    with patch("scripts.evaluate_rag_answers.httpx.Client", return_value=client):
        result = evaluate_rag_answers("http://example.test")

    assert result["answer_contains_passed"] == 1
    assert client.post.call_args.kwargs["headers"] == {"X-API-Key": "x" * 40}
