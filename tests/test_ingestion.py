from pathlib import Path

from app.core.ingestion import chunk_text, load_text


def test_chunk_text_respects_boundaries_and_overlap():
    chunks = chunk_text("a" * 60 + "\n" + "b" * 60, size=80, overlap=10)
    assert len(chunks) == 2
    assert len(chunks[0]) <= 80
    assert len(chunks[1]) <= 80
    assert chunks[1].startswith("a"[-10:])


def test_chunk_text_splits_long_paragraphs():
    chunks = chunk_text("0123456789" * 10, size=30, overlap=5)
    assert all(len(chunk) <= 30 for chunk in chunks)
    assert len(chunks) > 1


def test_load_markdown(tmp_path: Path):
    source = tmp_path / "policy.md"
    source.write_text("# Tiêu đề\nNội dung", encoding="utf-8")
    assert load_text(source) == "# Tiêu đề\nNội dung"
