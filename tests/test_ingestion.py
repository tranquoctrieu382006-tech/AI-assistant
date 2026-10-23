from pathlib import Path
import sys
import types

from app.core.ingestion import chunk_document_blocks, chunk_text, load_text


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


def test_paragraph_block_strategy_preserves_extracted_boundaries_and_headings():
    blocks = [
        (None, "Policy", "Policy"),
        (None, "Policy", "First paragraph."),
        (None, "Policy", "Second paragraph."),
    ]
    packed = chunk_document_blocks(blocks)
    separated = chunk_document_blocks(blocks, strategy="paragraph_blocks")

    assert len(packed) == 1
    assert len(separated) == 2
    assert separated[0]["content"] == "Policy\nFirst paragraph."
    assert separated[1]["content"] == "Second paragraph."
    assert all(chunk["section"] == "Policy" for chunk in separated)


def test_chunk_document_blocks_rejects_unknown_strategy():
    try:
        chunk_document_blocks([(None, None, "text")], strategy="unknown")
    except ValueError as exc:
        assert "Unsupported chunking strategy" in str(exc)
    else:
        raise AssertionError("Expected unsupported strategy to fail")


def test_load_markdown(tmp_path: Path):
    source = tmp_path / "policy.md"
    source.write_text("# Tiêu đề\nNội dung", encoding="utf-8")
    assert load_text(source) == "# Tiêu đề\nNội dung"


def test_load_text_strips_utf8_bom(tmp_path: Path):
    source = tmp_path / "bom.txt"
    source.write_bytes("\ufeffNội dung".encode("utf-8"))
    assert load_text(source) == "Nội dung"


def test_load_text_preserves_markdown_whitespace(tmp_path: Path):
    source = tmp_path / "whitespace.md"
    source.write_text("# Heading\n\nParagraph\n", encoding="utf-8")

    assert load_text(source) == "# Heading\n\nParagraph\n"


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
    from app.core.ingestion import load_document_blocks, load_document_pages
    assert load_document_pages(tmp_path / "sample.pdf") == [(1, "first page"), (2, "second page")]
    assert load_document_blocks(tmp_path / "sample.pdf") == [
        (1, None, "first page"), (2, None, "second page")
    ]


def test_markdown_sections_are_kept_on_chunks(tmp_path: Path):
    from app.core.ingestion import load_document_blocks

    source = tmp_path / "policy.md"
    source.write_text(
        "# Policy\nIntroduction\n## Leave\nAnnual leave details\n## Other\nOther details",
        encoding="utf-8",
    )

    chunks = chunk_document_blocks(load_document_blocks(source))

    assert [chunk["section"] for chunk in chunks] == [
        "Policy", "Policy > Leave", "Policy > Other",
    ]
    assert "## Leave" in chunks[1]["content"]


def test_docx_heading_styles_are_kept_on_chunks(tmp_path: Path):
    from docx import Document

    from app.core.ingestion import chunk_document_blocks, load_document_blocks
    source = tmp_path / "sections.docx"
    document = Document()
    document.add_heading("Policies", level=1)
    document.add_heading("Leave", level=2)
    document.add_paragraph("Annual leave details")
    document.save(source)

    chunks = chunk_document_blocks(load_document_blocks(source))

    assert chunks[-1]["section"] == "Policies > Leave"
    assert "Annual leave details" in chunks[-1]["content"]


def test_ingest_directory_sends_current_relative_source_set(tmp_path: Path, monkeypatch):
    (tmp_path / "policy.md").write_text("current policy", encoding="utf-8")
    (tmp_path / "README.md").write_text("repository notes", encoding="utf-8")
    (tmp_path / "data_dictionary.md").write_text("schema notes", encoding="utf-8")
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
    assert all("section" in item for item in captured["chunks"])
