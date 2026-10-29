"""
Unit tests for text and document chunking strategies.
Kiểm thử chuyên sâu các hàm chunk_text, chunk_document_blocks và xử lý ngữ nghĩa tiếng Việt.
"""

import pytest

from app.core.ingestion import (
    chunk_text,
    chunk_document_blocks,
)


def test_chunk_text_default_parameters():
    """Kiểm tra chunk_text với tham số mặc định."""
    text = "Đây là văn bản thử nghiệm nhằm kiểm tra thuật toán chia nhỏ tài liệu." * 10
    chunks = chunk_text(text)
    assert len(chunks) >= 1
    assert all(isinstance(c, str) for c in chunks)


def test_chunk_text_respects_custom_size():
    """Kiểm tra mỗi chunk không vượt quá giới hạn size đã cấu hình."""
    text = "Từ " * 200
    size = 100
    overlap = 20
    chunks = chunk_text(text, size=size, overlap=overlap)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= size


def test_chunk_text_preserves_word_boundaries():
    """Đảm bảo không cắt ngang giữa chừng một từ tiếng Việt."""
    words = ["chính", "sách", "bảo", "hành", "doanh", "nghiệp", "vận", "chuyển", "kho"]
    text = " ".join(words * 15)
    chunks = chunk_text(text, size=80, overlap=15)
    for chunk in chunks:
        assert not chunk.startswith(" ")
        assert not chunk.endswith(" ")


def test_chunk_text_vietnamese_diacritics():
    """Kiểm tra văn bản tiếng Việt có dấu đầy đủ không bị lỗi font hay mất ký tự."""
    vietnamese_text = (
        "Chính sách đổi trả hàng hóa áp dụng trong vòng 07 ngày làm việc kể từ ngày nhận hàng. "
        "Khách hàng cần giữ nguyên tem mác, hóa đơn mua hàng và phụ kiện đi kèm. "
        "Các trường hợp sản phẩm bị trầy xước do người dùng sẽ không được áp dụng đổi mới 100%."
    )
    chunks = chunk_text(vietnamese_text, size=120, overlap=20)
    reconstructed = " ".join(chunks)
    assert "đổi trả hàng hóa" in reconstructed
    assert "07 ngày làm việc" in reconstructed
    assert "không được áp dụng đổi mới 100%" in reconstructed


def test_chunk_text_empty_and_whitespace():
    """Kiểm tra xử lý an toàn cho chuỗi rỗng hoặc chỉ có khoảng trắng."""
    assert chunk_text("") == []
    assert chunk_text("   \n\t  ") == []


def test_chunk_text_short_text():
    """Kiểm tra văn bản ngắn hơn kích thước chunk_size trả về đúng 1 chunk."""
    short = "Quy định bảo mật thông tin nội bộ."
    chunks = chunk_text(short, size=200, overlap=30)
    assert len(chunks) == 1
    assert chunks[0] == short


def test_chunk_document_blocks_heading_strategy():
    """Kiểm tra chiến lược heading lưu trữ section/tiêu đề và nội dung phân đoạn tương ứng."""
    blocks = [
        (1, "I. Điều kiện đổi trả", "Sản phẩm phải còn nguyên hộp và tem niêm phong."),
        (2, "II. Thời hạn bảo hành", "Thời gian bảo hành tiêu chuẩn là 12 tháng kể từ ngày kích hoạt."),
    ]
    chunks = chunk_document_blocks(blocks, strategy="heading")
    assert len(chunks) >= 2
    assert any(c.get("section") == "I. Điều kiện đổi trả" for c in chunks)
    assert any(c.get("section") == "II. Thời hạn bảo hành" for c in chunks)
    assert any("Sản phẩm phải còn nguyên hộp" in c["content"] for c in chunks)


def test_chunk_document_blocks_fixed_size_strategy():
    """Kiểm tra chiến lược fixed_size chia nhỏ các block đồng đều."""
    blocks = [
        (1, None, "Nội dung đoạn văn bản dài cần được phân đoạn theo kích thước cố định. " * 5),
    ]
    chunks = chunk_document_blocks(blocks, strategy="fixed_size")
    assert len(chunks) >= 1
    assert all("page_number" in c for c in chunks)
    assert all("content" in c for c in chunks)
