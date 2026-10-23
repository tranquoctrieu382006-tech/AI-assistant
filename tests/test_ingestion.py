from pathlib import Path
import sys
import types

from app.core.ingestion import chunk_text, load_text


def test_chunk_text_respects_boundaries_and_overlap():
    chunks = chunk_text(" ".join(["alpha"] * 20) + "\n" + " ".join(["bravo"] * 10), size=80, overlap=10)
    assert len(chunks) >= 2
    assert len(chunks[0]) <= 80
    assert len(chunks[1]) <= 80
    assert "alpha" in chunks[0] and "alpha" in chunks[1]


def test_chunk_text_never_splits_words_and_keeps_heading_with_content():
    chunks = chunk_text("# Heading\n" + "important " * 10, size=38, overlap=5)
    assert all(not chunk.endswith("impor") for chunk in chunks)
    assert chunks[0].startswith("# Heading") and "important" in chunks[0]


def test_chunk_text_splits_long_paragraphs():
    chunks = chunk_text("0123456789" * 10, size=30, overlap=5)
    assert all(len(chunk) <= 30 for chunk in chunks)
    assert len(chunks) > 1


def test_load_markdown(tmp_path: Path):
    source = tmp_path / "policy.md"
    source.write_text("# Tiêu đề\nNội dung", encoding="utf-8")
    assert load_text(source) == "# Tiêu đề\nNội dung"


def test_load_text_strips_utf8_bom(tmp_path: Path):
    source = tmp_path / "bom.txt"
    source.write_bytes("\ufeffNội dung".encode("utf-8"))
    assert load_text(source) == "Nội dung"


def test_docx_preserves_paragraph_table_order_and_deduplicates_merged_cells(tmp_path: Path):
    from docx import Document

    from app.core.ingestion import load_text
    source = tmp_path / "ordered.docx"
    document = Document()
    document.add_paragraph("Before table")
    table = document.add_table(rows=1, cols=2)
    table.cell(0, 0).text = "Merged text"
    table.cell(0, 0).merge(table.cell(0, 1))
    document.add_paragraph("After table")
    document.save(source)
    text = load_text(source)
    assert text.index("Before table") < text.index("Merged text") < text.index("After table")
    assert text.count("Merged text") == 1


def test_pdf_loader_keeps_one_based_page_metadata(monkeypatch, tmp_path: Path):
    class Page:
        def __init__(self, text): self.text = text
        def extract_text(self): return self.text

    class Reader:
        def __init__(self, path): self.pages = [Page("first page"), Page("second page")]

    fake_pypdf = types.ModuleType("pypdf")
    fake_pypdf.PdfReader = Reader
    monkeypatch.setitem(sys.modules, "pypdf", fake_pypdf)
    from app.core.ingestion import load_document_pages
    assert load_document_pages(tmp_path / "sample.pdf") == [(1, "first page"), (2, "second page")]


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
