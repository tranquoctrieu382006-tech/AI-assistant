"""Load documents and split them into bounded, word-safe chunks."""
from pathlib import Path
import re

from app.core.config import settings

_HEADING = re.compile(r"^#{1,6}\s+.+")


def load_document_pages(path: Path) -> list[tuple[int | None, str]]:
    """Return ordered document blocks, with one-based PDF page metadata."""
    suffix = path.suffix.lower()
    if suffix in {".md", ".txt"}:
        return [(None, path.read_text(encoding="utf-8-sig"))]
    if suffix == ".docx":
        from docx import Document as DocxDocument
        from docx.table import Table
        from docx.text.paragraph import Paragraph

        document = DocxDocument(path)
        blocks = []
        for child in document.element.body.iterchildren():
            if child.tag.endswith("}p"):
                text = Paragraph(child, document).text.strip()
                if text:
                    blocks.append((None, text))
            elif child.tag.endswith("}tbl"):
                table = Table(child, document)
                for row in table.rows:
                    seen = set()
                    cells = []
                    for cell in row.cells:
                        identity = id(cell._tc)
                        value = cell.text.strip()
                        if identity not in seen and value:
                            cells.append(value)
                        seen.add(identity)
                    if cells:
                        blocks.append((None, " | ".join(cells)))
        return blocks
    if suffix == ".pdf":
        from pypdf import PdfReader
        return [(number, page.extract_text() or "")
                for number, page in enumerate(PdfReader(str(path)).pages, 1)]
    return []


def load_text(path: Path) -> str:
    """Compatibility helper returning all extracted text in reading order."""
    return "\n".join(text for _, text in load_document_pages(path))


def _split_long(text: str, size: int, overlap: int) -> list[str]:
    words = text.split()
    chunks: list[str] = []
    current: list[str] = []
    for word in words:
        if len(word) > size:
            if current:
                chunks.append(" ".join(current)); current = []
            # A single pathological token is the only case where a word must be split.
            for start in range(0, len(word), size - overlap):
                part = word[start:start + size]
                chunks.append(part)
            continue
        candidate = " ".join([*current, word])
        if len(candidate) <= size:
            current.append(word)
        else:
            chunks.append(" ".join(current))
            tail: list[str] = []
            for previous in reversed(current):
                if len(" ".join([previous, *tail])) > overlap:
                    break
                tail.insert(0, previous)
            current = [*tail, word]
    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_text(text: str, size: int | None = None, overlap: int | None = None) -> list[str]:
    size = settings.rag_chunk_size if size is None else size
    overlap = settings.rag_chunk_overlap if overlap is None else overlap
    if size < 1 or overlap < 0 or overlap >= size:
        raise ValueError("Chunk size must be positive and overlap must be smaller than size")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    # Attach Markdown headings to the following paragraph so a chunk boundary
    # cannot strand a heading from the content it describes.
    paragraphs: list[str] = []
    pending_heading = ""
    for line in lines:
        if _HEADING.match(line):
            if pending_heading:
                paragraphs.append(pending_heading)
            pending_heading = line
            continue
        paragraphs.append(f"{pending_heading}\n{line}" if pending_heading else line)
        pending_heading = ""
    if pending_heading:
        paragraphs.append(pending_heading)
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        parts = _split_long(paragraph, size, overlap)
        for part in parts:
            candidate = f"{current}\n{part}" if current else part
            if len(candidate) <= size:
                current = candidate
                continue
            if current:
                chunks.append(current)
            tail_words = []
            for word in reversed(current.split()):
                if len(" ".join([word, *tail_words])) > overlap:
                    break
                tail_words.insert(0, word)
            current = " ".join([*tail_words, part]).strip()
            if len(current) > size:
                chunks.extend(_split_long(current, size, overlap)[:-1])
                current = _split_long(current, size, overlap)[-1]
    if current:
        chunks.append(current)
    return chunks


def ingest_directory(directory: str | Path | None = None) -> dict:
    from app.core.vector_store import replace_corpus_chunks
    root = Path(directory or settings.rag_document_dir)
    excluded = {name.casefold() for name in settings.rag_exclude_files}
    files = sorted(
        p for p in root.rglob("*")
        if p.suffix.lower() in {".md", ".txt", ".docx", ".pdf"}
        and p.name.casefold() not in excluded
    )
    records: list[dict] = []
    sources = [path.relative_to(root).as_posix() for path in files]
    for path in files:
        source = path.relative_to(root).as_posix()
        index = 0
        for page_number, text in load_document_pages(path):
            for chunk in chunk_text(text):
                records.append({"content": chunk, "source": source,
                                "chunk_index": index, "page_number": page_number})
                index += 1
    stored = replace_corpus_chunks(records, sources)
    return {"files": len(files), "chunks": stored, "directory": str(root)}
