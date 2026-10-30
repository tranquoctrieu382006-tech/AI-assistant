"""Centralized text normalization utilities for Vietnamese NLP and retrieval."""
from __future__ import annotations

import re
import unicodedata

# Common Vietnamese unaccented functional and domain words
COMMON_VN_UNACCENTED = {
    "khong", "duoc", "nhu", "the", "nao", "sao", "thoi", "han", "chinh", "sach",
    "quy", "trinh", "don", "hang", "doi", "tra", "bao", "hanh", "nhap", "xuat",
    "kho", "san", "pham", "khach", "nhan", "vien", "tien", "phi", "van", "chuyen",
    "giao", "mua", "ban", "ton", "khi", "neu", "lam", "gi", "ai", "dau", "may",
    "nhieu", "cua", "trong", "cho", "ve", "cac", "nhung", "nguoi", "ngay", "thang", "nam",
    "dia", "chi", "thanh", "pho", "hop", "dong", "kpi", "thu", "tuc", "dieu", "khoan",
}

# Common English stop words
ENGLISH_STOPS = {
    "what", "when", "where", "which", "who", "whom", "why", "how", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "do", "does",
    "did", "the", "a", "an", "and", "or", "but", "if", "because", "as", "until",
    "while", "of", "at", "by", "for", "with", "about", "between", "into", "through",
    "during", "before", "after", "above", "below", "to", "from", "in", "out", "on",
    "can", "could", "should", "would", "will", "select", "from", "where",
}


def strip_accents(text: str) -> str:
    """
    Remove all Vietnamese accents/diacritics and normalize to lowercase ASCII.
    Correctly handles 'đ'/'Đ' which are distinct Unicode code points (U+0110/U+0111).
    """
    if not text:
        return ""
    nfkd = unicodedata.normalize("NFKD", text)
    stripped = "".join(c for c in nfkd if not unicodedata.combining(c))
    return stripped.replace("đ", "d").replace("Đ", "D").casefold()


def ascii_fold(text: str) -> str:
    """Alias for strip_accents for consistent cross-module naming."""
    return strip_accents(text)


def is_probably_unaccented_vietnamese(text: str) -> bool:
    """
    Fast, deterministic heuristic to detect whether an ASCII string is unaccented Vietnamese.
    - Must be valid ASCII (no diacritics yet).
    - Contains at least 2 words.
    - Not predominantly English stopwords.
    - Contains at least one characteristic unaccented Vietnamese word.
    """
    if not text or not text.isascii():
        return False
    words = re.findall(r"[a-zA-Z]+", text.casefold())
    if len(words) < 2:
        return False
    eng_count = sum(1 for w in words if w in ENGLISH_STOPS)
    if eng_count / len(words) >= 0.35:
        return False
    vn_count = sum(1 for w in words if w in COMMON_VN_UNACCENTED)
    return vn_count >= 1


def tokenize_with_unaccented(text: str) -> list[str]:
    """
    Tokenize text for BM25 retrieval, emitting only unaccented tokens.

    Both documents and queries go through this function, so accented and
    unaccented spellings of the same word produce identical tokens. Emitting
    only one form (instead of both) keeps BM25 term frequency and document
    length comparable between accented and unaccented queries.
    """
    if not text:
        return []
    normalized = unicodedata.normalize("NFKC", text).casefold()
    tokens = re.findall(r"\w+", normalized, flags=re.UNICODE)
    return [strip_accents(token) for token in tokens]
