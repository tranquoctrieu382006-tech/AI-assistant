from datetime import date
from decimal import Decimal

from scripts.evaluate_nlq_golden import is_read_only_query, results_match
from scripts.evaluate_rag_retrieval import source_matches
from scripts.evaluate_nlq_model import (
    _execution_results_match,
    _load_schema_ddl,
    _sql_features,
)
from scripts.evaluate_rag_retrieval import _category_metrics, _retrieval_metrics


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
    count, trailing, ordered = _sql_features(
        "SELECT 'value; still literal' AS note /* ; */ ORDER BY note; -- trailing comment"
    )
    assert count == 1
    assert not trailing
    assert ordered


def test_sql_features_handle_escaped_and_dollar_quoted_literals():
    assert _sql_features(r"SELECT E'escaped \'; value'") == (0, False, False)
    assert _sql_features("SELECT $body$semi;colon$body$") == (0, False, False)


def test_sql_features_detect_multiple_statements_but_ignore_nested_order_by():
    assert _sql_features("SELECT (SELECT 1 ORDER BY 1)") == (0, False, False)
    assert _sql_features("SELECT 1; SELECT 2")[:2] == (1, True)


def test_model_evaluator_uses_seed_create_table_ddl():
    schema = _load_schema_ddl()
    assert "CREATE TABLE products" in schema
    assert "CREATE TABLE stock_movements" in schema
    assert "INSERT INTO products" not in schema


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
