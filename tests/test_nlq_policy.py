"""
Unit tests for app.core.nlq_policy.
Kiểm tra chốt chặn chính sách NLQ, phân biệt PII cá nhân vs thông tin liên hệ cơ sở/tổ chức,
miễn trừ câu hỏi đếm tổng hợp (COUNT aggregate), và phân cấp ngoại lệ có kiểu (typed exceptions).
"""

import pytest

from app.core.nlq_policy import (
    ALLOWED_CUSTOMER_COLUMNS,
    InvalidSqlError,
    NlqValidationError,
    PolicyViolation,
    PrivacyViolation,
    SchemaMismatch,
    get_allowed_columns_for_table,
    has_sensitive_mention,
    is_human_pii_query,
)


def test_typed_exceptions_hierarchy():
    """Kiểm tra cây kế thừa ngoại lệ typed exceptions kế thừa NlqValidationError và ValueError."""
    assert issubclass(NlqValidationError, ValueError)
    assert issubclass(PrivacyViolation, NlqValidationError)
    assert issubclass(SchemaMismatch, NlqValidationError)
    assert issubclass(InvalidSqlError, NlqValidationError)
    assert PolicyViolation is PrivacyViolation

    priv = PrivacyViolation("Lỗi vi phạm chính sách")
    assert priv.error_code == "policy_denied"

    schema = SchemaMismatch("Cột ngoài schema")
    assert schema.error_code == "unanswerable"

    syntax = InvalidSqlError("Cú pháp SQL không hợp lệ")
    assert syntax.error_code == "exec_error"


def test_human_pii_detection_blocks_personal_contact_fields():
    """Chặn các câu hỏi đòi thông tin liên hệ / danh tính cá nhân của con người (PII)."""
    pii_queries = [
        "Email của khách hàng VIP là gì?",
        "Cho tôi số điện thoại của khách hàng Nguyen Van A",
        "Địa chỉ nhà của khách hàng này ở đâu?",
        "Mật khẩu của người dùng admin là gì?",
        "Số CCCD của nhân viên giao hàng",
        "Thông tin tài khoản ngân hàng của khách hàng",
        "SĐT của thành viên",
    ]
    for q in pii_queries:
        assert is_human_pii_query(q) is True, f"Kỳ vọng {q} là PII query"
        assert has_sensitive_mention(q) is True


def test_human_pii_exempts_facility_and_business_contacts():
    """Không chặn thông tin liên hệ công khai của tổ chức, kho hàng, chi nhánh."""
    non_pii_queries = [
        "Số điện thoại kho hàng là bao nhiêu?",
        "Hotline công ty là gì?",
        "Địa chỉ chi nhánh công ty ở đâu?",
        "Số điện thoại tổng đài hỗ trợ kỹ thuật?",
    ]
    for q in non_pii_queries:
        assert is_human_pii_query(q) is False, f"Kỳ vọng {q} KHÔNG phải PII query"
        assert has_sensitive_mention(q) is False


def test_human_pii_exempts_count_aggregates():
    """Miễn trừ câu hỏi đếm số lượng tổng hợp (COUNT aggregate), cho phép truy vấn CSDL."""
    count_queries = [
        "Có bao nhiêu khách hàng có email?",
        "Đếm số lượng khách hàng có số điện thoại",
        "Thống kê khách hàng có email",
        "Có mấy user có số điện thoại?",
        "Tổng số khách hàng có email hợp lệ",
        "Tỷ lệ khách hàng có số điện thoại",
    ]
    for q in count_queries:
        assert is_human_pii_query(q) is False, f"Kỳ vọng câu hỏi aggregate {q} không bị chặn PII"


def test_get_allowed_columns_filters_sensitive_fields():
    """Đảm bảo danh sách cột được phép loại bỏ triệt để các cột nhạy cảm khỏi schema."""
    raw_schema = [
        {
            "name": "customers",
            "columns": [
                {"name": "customer_id"},
                {"name": "email"},
                {"name": "phone"},
                {"name": "password_hash"},
                {"name": "city"},
                {"name": "member_tier"},
            ],
        }
    ]
    allowed = get_allowed_columns_for_table(raw_schema, table_name="customers")
    assert "mã khách hàng" in allowed
    assert "thành phố" in allowed
    assert "hạng thành viên" in allowed
    assert "email" not in allowed
    assert "phone" not in allowed
    assert "password_hash" not in allowed


def test_get_allowed_columns_fallback_defaults():
    """Khi schema rỗng, trả về các cột mặc định an toàn của bảng customers."""
    allowed = get_allowed_columns_for_table([], table_name="customers")
    assert allowed == list(ALLOWED_CUSTOMER_COLUMNS.values())
