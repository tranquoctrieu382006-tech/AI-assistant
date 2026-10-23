from datetime import date
from decimal import Decimal

from scripts.evaluate_nlq_golden import is_read_only_query, results_match
from scripts.evaluate_rag_retrieval import source_matches
from scripts.evaluate_nlq_model import _execution_results_match


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
