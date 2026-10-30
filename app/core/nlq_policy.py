import re
from typing import Any

# Biểu thức chính quy cho cột và bảng nhạy cảm (Single Source of Truth)
_SENSITIVE_PATTERN = r"password|passwd|secret|token|email|phone|mobile|address|account_number|national_id|full_name"
_SENSITIVE = re.compile(_SENSITIVE_PATTERN, re.IGNORECASE)

# Biểu thức chính quy nhận diện từ khóa đề cập thuộc tính nhạy cảm trong câu hỏi của người dùng
_SENSITIVE_MENTION_RE = re.compile(
    r"\b(?:số\s+điện\s+thoại|so\s+dien\s+thoai|sđt|sdt|email|thư\s+điện\s+tử|thu\s+dien\s+tu|"
    r"mật\s+khẩu|mat\s+khau|địa\s+chỉ|dia\s+chi|cmnd|cccd|căn\s+cước|can\s+cuoc|"
    r"tài\s+khoản\s+ngân\s+hàng|tai\s+khoan\s+ngan\s+hang|password|phone)\b",
    re.IGNORECASE,
)

# Danh sách cột được phép thực tế của bảng customers theo đúng schema CSDL (schema_seed.sql & GRANT)
# Lưu ý: Bảng customers chỉ có (customer_id, full_name, email, city, member_tier, joined_on)
# Trong đó full_name và email đã bị hạn chế truy cập bởi quyền DB.
ALLOWED_CUSTOMER_COLUMNS = {
    "customer_id": "mã khách hàng",
    "city": "thành phố",
    "member_tier": "hạng thành viên",
    "joined_on": "ngày tham gia",
}


def has_sensitive_mention(question: str) -> bool:
    """
    Kiểm tra xem câu hỏi có đề cập đến các thuộc tính nhạy cảm hay không.
    Được dùng làm chốt chặn an toàn: Nếu câu hỏi có cờ này mà CSDL từ chối,
    tuyệt đối không cho phép fallback sang RAG để tránh rò rỉ thông tin cá nhân.
    """
    if not question:
        return False
    return bool(_SENSITIVE_MENTION_RE.search(question))


class NlqValidationError(ValueError):
    """Lớp cha cho các ngoại lệ kiểm tra tính an toàn hoặc hợp lệ của truy vấn NLQ (kế thừa ValueError để tương thích)."""
    pass


class PolicyViolation(NlqValidationError):
    """
    Vi phạm chính sách bảo mật:
    - Truy vấn chọn cột nhạy cảm (email, phone, password...).
    - Thao tác ghi/sửa dữ liệu (INSERT, UPDATE, DELETE, DROP...).
    - Truy cập bảng hệ thống bị cấm (rag_chunks...).
    Hành động: Ngắt ngay lập tức, KHÔNG đưa vào vòng repair, KHÔNG fallback sang RAG.
    """
    pass


class SchemaMismatch(NlqValidationError):
    """
    Tham chiếu bảng hoặc cột không tồn tại trong schema được cấp quyền.
    Hành động: Cho phép repair; nếu hết lượt repair thì chuyển thành kết cục out_of_scope/unanswerable.
    Được phép fallback sang RAG nếu câu hỏi không chứa cờ nhạy cảm.
    """
    pass
