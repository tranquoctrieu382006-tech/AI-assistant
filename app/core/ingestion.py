"""Document loading and deterministic paragraph-aware chunking."""
from pathlib import Path

from app.core.config import settings


def load_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".md" or suffix == ".txt":
        return path.read_text(encoding="utf-8")
    if suffix == ".docx":
        from docx import Document as DocxDocument
        document = DocxDocument(path)
        blocks = [p.text for p in document.paragraphs if p.text.strip()]
        blocks.extend(
            " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
            for table in document.tables
            for row in table.rows
            if any(cell.text.strip() for cell in row.cells)
        )
        return "\n".join(blocks)
    if suffix == ".pdf":
        from pypdf import PdfReader
        return "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
    return ""


def chunk_text(text: str, size: int | None = None, overlap: int | None = None) -> list[str]:
    size = size or settings.rag_chunk_size
    overlap = settings.rag_chunk_overlap if overlap is None else overlap
    if size < 1 or overlap < 0 or overlap >= size:
        raise ValueError("Chunk size must be positive and overlap must be smaller than size")
    # Keep paragraph boundaries where possible, while retaining a hard upper bound.
    paragraphs = [p.strip() for p in text.splitlines() if p.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        while len(paragraph) > size:
            if current:
                chunks.append(current)
                current = ""
            chunks.append(paragraph[:size])
            paragraph = paragraph[size - overlap:]
        candidate = f"{current}\n{paragraph}" if current else paragraph
        if len(candidate) <= size:
            current = candidate
        else:
            chunks.append(current)
            tail_size = min(overlap, max(0, size - len(paragraph) - 1))
            tail = current[-tail_size:] if tail_size else ""
            current = f"{tail}\n{paragraph}".strip()
    if current:
        chunks.append(current)
    return chunks


def ingest_directory(directory: str | Path | None = None) -> dict:
    from app.core.vector_store import replace_corpus_chunks
    root = Path(directory or settings.rag_document_dir)
    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in {".md", ".txt", ".docx", ".pdf"})
    records: list[dict] = []
    sources = [path.relative_to(root).as_posix() for path in files]
    for path in files:
        content = load_text(path)
        for index, chunk in enumerate(chunk_text(content)):
            records.append({"content": chunk, "source": path.relative_to(root).as_posix(), "chunk_index": index})
    stored = replace_corpus_chunks(records, sources)
    return {"files": len(files), "chunks": stored, "directory": str(root)}

