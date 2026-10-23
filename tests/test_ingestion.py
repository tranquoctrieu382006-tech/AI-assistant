from pathlib import Path
import sys
import types

from app.core.ingestion import chunk_text, load_text


def test_chunk_text_respects_boundaries_and_overlap():
    chunks = chunk_text("a" * 60 + "\n" + "b" * 60, size=80, overlap=10)
    assert len(chunks) == 2
    assert len(chunks[0]) <= 80
    assert len(chunks[1]) <= 80
    assert chunks[1].startswith("a" * 10)


def test_chunk_text_splits_long_paragraphs():
    chunks = chunk_text("0123456789" * 10, size=30, overlap=5)
    assert all(len(chunk) <= 30 for chunk in chunks)
    assert len(chunks) > 1


def test_load_markdown(tmp_path: Path):
    source = tmp_path / "policy.md"
    source.write_text("# Tiêu đề\nNội dung", encoding="utf-8")
    assert load_text(source) == "# Tiêu đề\nNội dung"


def test_ingest_directory_sends_current_relative_source_set(tmp_path: Path, monkeypatch):
    (tmp_path / "policy.md").write_text("current policy", encoding="utf-8")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "guide.txt").write_text("current guide", encoding="utf-8")
    captured = {}

    def replace_corpus_chunks(chunks, sources):
        captured["chunks"] = chunks
        captured["sources"] = sources
        return len(chunks)

    fake_store = types.ModuleType("app.core.vector_store")
    fake_store.replace_corpus_chunks = replace_corpus_chunks
    monkeypatch.setitem(sys.modules, "app.core.vector_store", fake_store)

    from app.core.ingestion import ingest_directory
    result = ingest_directory(tmp_path)

    assert result["files"] == 2
    assert captured["sources"] == ["nested/guide.txt", "policy.md"]
    assert {item["source"] for item in captured["chunks"]} == set(captured["sources"])
