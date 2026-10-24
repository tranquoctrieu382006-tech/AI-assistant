"""Portable SHA-256 helpers for frozen evaluation manifests."""
from __future__ import annotations

import hashlib
from pathlib import Path

_BINARY_SUFFIXES = {".docx", ".pdf", ".png", ".jpg", ".jpeg", ".zip"}


def canonical_bytes(path: Path) -> bytes:
    """Hash text with LF endings across platforms; hash binary inputs verbatim."""
    data = path.read_bytes()
    if path.suffix.lower() in _BINARY_SUFFIXES:
        return data
    return data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(canonical_bytes(path)).hexdigest()
