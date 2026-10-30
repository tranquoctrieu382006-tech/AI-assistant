import re
from typing import Any

# Biểu thức chính quy cho cột và bảng nhạy cảm trong CSDL (Single Source of Truth)
_SENSITIVE_PATTERN = r"password|passwd|secret|token|email|phone|mobile|address|account_number|national_id|full_name"
_SENSITIVE = re.compile(_SENSITIVE_PATTERN, re.IGNORECASE)

# Các trường thông tin liên hệ / danh tính cá nhân nhạy cảm
_PII_FIELD_PATTERN = (
    r"\b(?:số\s+điện\s+thoại|so\s+dien\s+thoai|sđt|sdt|email|thư\s+điện\s+tử|thu\s+dien\s+tu|"
    r"mật\s+khẩu|mat\s+khau|địa\s+chỉ(?:\s+nhà)?|dia\s+chi(?:\s+nha)?|cmnd|cccd|căn\s+cước|can\s+cuoc|"
    r"tài\s+khoản\s+ngân\s+hàng|tai\s+khoan\s+ngan\s+hang|password|phone)\b"
)
_PII_FIELD_RE = re.compile(_PII_FIELD_PATTERN, re.IGNORECASE)

# Các thực thể người (human/person entities) cần bảo vệ PII
_HUMAN_ENTITY_PATTERN = (
    r"\b(?:khách\s+hàng|khach\s+hang|người\s+dùng|nguoi\s+dung|user|nhân\s+viên|nhan\s+vien|"
    r"cá\s+nhân|ca\s+nhan|tài\s+khoản|tai\s+khoan|thành\s+viên|thanh\s+vien)\b"
)
_HUMAN_ENTITY_RE = re.compile(_HUMAN_ENTITY_PATTERN, re.IGNORECASE)

# Không có miễn trừ cho câu đếm tổng hợp. Nếu vị ngữ của câu dựa vào một
# thuộc tính nhạy cảm (email, số điện thoại...) thì cột đó không nằm trong dữ
# liệu được cấp quyền: model không thể đếm theo điều kiện đó và rất dễ trả về
# tổng số bản ghi như thể là câu trả lời đúng. Vì vậy mọi cách diễn đạt
# ("bao nhiêu", "đếm", "thống kê"...) đều bị chặn như nhau.

# Nhãn tiếng Việt cho các cột thường gặp
COLUMN_DISPLAY_LABELS = {
    "customer_id": "mã khách hàng",
    "city": "thành phố",
    "member_tier": "hạng thành viên",
    "joined_on": "ngày tham gia",
    "order_id": "mã đơn hàng",
    "order_date": "ngày đặt hàng",
    "delivered_date": "ngày giao hàng",
    "status": "trạng thái",
    "shipping_fee": "phí vận chuyển",
    "product_id": "mã sản phẩm",
    "name": "tên sản phẩm",
    "sku": "mã SKU",
    "unit_price": "giá bán",
}

# Danh sách cột mặc định của bảng customers theo đúng schema và quyền DB (schema_seed.sql)
ALLOWED_CUSTOMER_COLUMNS = {
    "customer_id": "mã khách hàng",
    "city": "thành phố",
    "member_tier": "hạng thành viên",
    "joined_on": "ngày tham gia",
}


def is_human_pii_query(question: str) -> bool:
    """
    Nhận diện câu hỏi có yêu cầu truy xuất dữ liệu cá nhân (PII) của con người hay không.
    Điều kiện:
    - Có chứa trường PII (số điện thoại, email, địa chỉ, mật khẩu, CCCD, tài khoản).
    - Đi kèm thực thể người (khách hàng, người dùng, nhân viên, thành viên).
    - Không miễn trừ cách diễn đạt: "bao nhiêu", "đếm", "thống kê"... đều bị
      chặn nếu vị ngữ dựa vào trường nhạy cảm, vì trường đó không có trong dữ
      liệu được cấp quyền.
    """
    if not question:
        return False

    has_pii_field = bool(_PII_FIELD_RE.search(question))
    if not has_pii_field:
        return False

    has_human_entity = bool(_HUMAN_ENTITY_RE.search(question))
    if not has_human_entity:
        # Nếu hỏi thực thể phi cá nhân (ví dụ: 'Số điện thoại kho hàng', 'Hotline công ty') thì không phải PII người
        return False

    # Không miễn trừ câu đếm tổng hợp: nếu vị ngữ dựa vào cột nhạy cảm thì cột
    # đó không được cấp quyền, nên "đếm" cũng không thể trả lời đúng.
    return True


def has_sensitive_mention(question: str) -> bool:
    """Kiểm tra câu hỏi có chứa yêu cầu PII cá nhân hay không (dùng làm chốt chặn an toàn)."""
    return is_human_pii_query(question)


def restricted_answer(columns: list[str] | None = None) -> str:
    """Thông báo thống nhất khi câu hỏi cần một trường nhạy cảm không được cấp quyền."""
    allowed = ", ".join(columns) if columns else ", ".join(ALLOWED_CUSTOMER_COLUMNS.values())
    return (
        "Câu hỏi cần một trường nhạy cảm (số điện thoại, email, địa chỉ...) "
        "không có trong dữ liệu được phép truy vấn, nên bị hạn chế để bảo vệ "
        f"quyền riêng tư. Bạn có thể tra cứu các trường được cấp quyền như: {allowed}."
    )


def get_allowed_columns_for_table(schema: list[dict] | None = None, table_name: str = "customers") -> list[str]:
    """Lấy danh sách các cột được phép từ schema CSDL thật (loại trừ các cột nhạy cảm)."""
    if schema:
        for table in schema:
            if table.get("name") == table_name:
                cols = [
                    COLUMN_DISPLAY_LABELS.get(col["name"], col["name"])
                    for col in table.get("columns", [])
                    if not _SENSITIVE.search(col["name"])
                ]
                if cols:
                    return cols
    return list(ALLOWED_CUSTOMER_COLUMNS.values())


# ==============================================================================
# HỆ THỐNG NGOẠI LỆ CÓ KIỂU (TYPED EXCEPTIONS)
# ==============================================================================

class NlqValidationError(ValueError):
    """Lớp cha cho các ngoại lệ kiểm tra SQL trong NLQ (kế thừa ValueError để tương thích test cũ)."""
    error_code: str = "validation_error"


class PrivacyViolation(NlqValidationError):
    """
    Vi phạm quyền riêng tư và an ninh:
    - Chọn cột nhạy cảm (email, phone, password, full_name...).
    - Truy cập bảng cấm (rag_chunks).
    - Thao tác ghi/sửa/xóa dữ liệu (INSERT, UPDATE, DELETE, DROP, TRUNCATE...).
    Hành động: DỪNG NGAY LẬP TỨC (không vào repair_sql), trả về status='restricted', error_code='policy_denied'.
    """
    error_code: str = "policy_denied"


# Alias tương thích
PolicyViolation = PrivacyViolation


class InvalidSqlError(NlqValidationError):
    """
    Model sinh SQL không hợp lệ về mặt kỹ thuật/cú pháp:
    - Chứa nhiều câu lệnh (multiple statements).
    - Gọi hàm ngoài whitelist cho phép.
    - Chọn cả dòng qua alias bảng (SELECT c FROM customers c).
    - SELECT INTO hoặc khóa dòng.
    Hành động: Cho phép repair_sql. Nếu hết lượt repair thì trả về status='error', error_code='exec_error'.
    """
    error_code: str = "exec_error"


class SchemaMismatch(NlqValidationError):
    """
    Truy vấn tham chiếu bảng hoặc cột ngoài schema được cấp quyền:
    - Bảng ngoài schema.
    - Cột ngoài schema.
    - SELECT * (cần chọn cột cụ thể).
    Hành động: Cho phép repair_sql. Nếu hết lượt repair thì trả về status='out_of_scope', error_code='unanswerable'.
    Được phép fallback sang RAG nếu câu hỏi không chứa cờ PII.
    """
    error_code: str = "unanswerable"
