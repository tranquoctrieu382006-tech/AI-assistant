"""Load documents and split them into bounded, word-safe chunks."""
from pathlib import Path
import logging
import re

from app.core.config import settings

_HEADING = re.compile(r"^#{1,6}\s+.+")
_MARKDOWN_HEADING = re.compile(r"^(#{1,6})\s+(.+)$")
logger = logging.getLogger(__name__)


def load_document_blocks(path: Path) -> list[tuple[int | None, str | None, str]]:
    """Return ordered text blocks with PDF page and structural section metadata."""
    suffix = path.suffix.lower()
    if suffix in {".md", ".txt"}:
        blocks = []
        sections: list[tuple[int, str]] = []
        from langchain_community.document_loaders import TextLoader

        loaded = TextLoader(str(path), encoding="utf-8-sig").load()
        for line in loaded[0].page_content.splitlines():
            text = line.strip()
            if not text:
                continue
            heading = _MARKDOWN_HEADING.match(text)
            if heading:
                level = len(heading.group(1))
                sections = [item for item in sections if item[0] < level]
                sections.append((level, heading.group(2).strip()))
            section = " > ".join(title for _, title in sections) or None
            blocks.append((None, section, text))
        return blocks
    if suffix == ".docx":
        from docx import Document as DocxDocument
        from docx.table import Table
        from docx.text.paragraph import Paragraph

        document = DocxDocument(path)
        blocks = []
        sections: list[tuple[int, str]] = []
        for child in document.element.body.iterchildren():
            if child.tag.endswith("}p"):
                paragraph = Paragraph(child, document)
                text = paragraph.text.strip()
                if text:
                    style_name = getattr(paragraph.style, "name", "") or ""
                    heading = re.match(r"Heading\s+(\d+)$", style_name, re.IGNORECASE)
                    if heading:
                        level = int(heading.group(1))
                        sections = [item for item in sections if item[0] < level]
                        sections.append((level, text))
                    section = " > ".join(title for _, title in sections) or None
                    blocks.append((None, section, text))
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
                        section = " > ".join(title for _, title in sections) or None
                        blocks.append((None, section, " | ".join(cells)))
        return blocks
    if suffix == ".pdf":
        from langchain_community.document_loaders import PyPDFLoader

        pages = PyPDFLoader(str(path)).load()
        blocks = []
        for page in pages:
            page_number = page.metadata.get("page")
            if not isinstance(page_number, int) or page_number < 0:
                raise ValueError(
                    f"LangChain PDF loader returned an invalid page number for {path}."
                )
            blocks.append((page_number + 1, None, page.page_content or ""))
        return blocks
    return []


def load_document_pages(path: Path) -> list[tuple[int | None, str]]:
    """Compatibility helper returning ordered text and one-based PDF page metadata."""
    return [(page_number, text) for page_number, _, text in load_document_blocks(path)]


def load_text(path: Path) -> str:
    """Compatibility helper returning all extracted text in reading order."""
    if path.suffix.lower() in {".md", ".txt"}:
        return path.read_text(encoding="utf-8-sig")
    return "\n".join(text for _, text in load_document_pages(path))


def _split_long(text: str, size: int, overlap: int) -> list[str]:
    words = text.split()
    chunks: list[str] = []
    current: list[str] = []
    for word in words:
        if len(word) > size:
            if current:
                chunks.append(" ".join(current))
                current = []
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
                candidate_tail = " ".join([previous, *tail])
                if len(candidate_tail) > overlap:
                    break
                if len(candidate_tail) + 1 + len(word) > size:
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
            pending_heading = f"{pending_heading}\n{line}" if pending_heading else line
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
                candidate_tail = " ".join([word, *tail_words])
                if len(candidate_tail) > overlap:
                    break
                if len(candidate_tail) + 1 + len(part) > size:
                    break
                tail_words.insert(0, word)
            current = " ".join([*tail_words, part]).strip()
    if current:
        chunks.append(current)
    return chunks


def chunk_document_blocks(
    blocks: list[tuple[int | None, str | None, str]],
    strategy: str = "heading",
) -> list[dict]:
    """Chunk by fixed size or within extracted heading boundaries."""
    if strategy == "paragraph":
        strategy = "heading"
    if strategy not in {"fixed_size", "heading", "paragraph_blocks"}:
        raise ValueError(f"Unsupported chunking strategy: {strategy}")
    if strategy == "paragraph_blocks":
        chunks = []
        pending_heading = None
        previous_section = None
        for page_number, section, raw_text in blocks:
            text = raw_text.strip()
            if not text:
                continue
            section_name = section.split(" > ")[-1] if section else None
            is_heading = bool(_HEADING.match(text)) or (
                section is not None
                and section != previous_section
                and text == section_name
            )
            if is_heading:
                if pending_heading:
                    page, heading_section, heading = pending_heading
                    chunks.extend(
                        {"content": content, "section": heading_section, "page_number": page}
                        for content in chunk_text(heading)
                    )
                pending_heading = (page_number, section, text)
                previous_section = section
                continue
            if pending_heading:
                if pending_heading[:2] == (page_number, section):
                    _, _, heading = pending_heading
                    text = f"{heading}\n{text}"
                else:
                    page, heading_section, heading = pending_heading
                    chunks.extend(
                        {
                            "content": content,
                            "section": heading_section,
                            "page_number": page,
                        }
                        for content in chunk_text(heading)
                    )
                pending_heading = None
            chunks.extend(
                {"content": content, "section": section, "page_number": page_number}
                for content in chunk_text(text)
            )
            previous_section = section
        if pending_heading:
            page, section, heading = pending_heading
            chunks.extend(
                {"content": content, "section": section, "page_number": page}
                for content in chunk_text(heading)
            )
        return _add_page_boundary_overlap(chunks)

    if strategy == "fixed_size":
        runs: list[tuple[int | None, set[str | None], str]] = []
        for page_number, section, text in blocks:
            if not text.strip():
                continue
            if runs and runs[-1][0] == page_number:
                page, sections, current_text = runs[-1]
                sections.add(section)
                runs[-1] = (page, sections, f"{current_text}\n{text.strip()}")
            else:
                runs.append((page_number, {section}, text.strip()))
        chunks = []
        for page_number, sections, text in runs:
            section = next(iter(sections)) if len(sections) == 1 else None
            chunks.extend(
                {"content": content, "section": section, "page_number": page_number}
                for content in chunk_text(text)
            )
        return _add_page_boundary_overlap(chunks)

    runs: list[tuple[int | None, str | None, str]] = []
    for page_number, section, text in blocks:
        if not text.strip():
            continue
        if runs and runs[-1][:2] == (page_number, section):
            page, current_section, current_text = runs[-1]
            runs[-1] = (page, current_section, f"{current_text}\n{text.strip()}")
        else:
            runs.append((page_number, section, text.strip()))

    def is_heading_only(section: str | None, text: str) -> bool:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return False
        section_name = section.split(" > ")[-1] if section else None
        return all(
            bool(_HEADING.match(line)) or (section_name is not None and line == section_name)
            for line in lines
        )

    # Heading-aware extraction can create a run containing only a parent
    # heading when the next line starts a child section. Carry that heading
    # forward so it describes the content rather than becoming its own vector.
    merged_runs: list[tuple[int | None, str | None, str]] = []
    for page_number, section, text in runs:
        if (
            merged_runs
            and is_heading_only(merged_runs[-1][1], merged_runs[-1][2])
            and merged_runs[-1][0] == page_number
            and (
                merged_runs[-1][1] is None
                or section == merged_runs[-1][1]
                or (section is not None and section.startswith(merged_runs[-1][1] + " > "))
            )
        ):
            previous_page, previous_section, previous_text = merged_runs.pop()
            text = f"{previous_text}\n{text}"
            section = section or previous_section
        merged_runs.append((page_number, section, text))
    runs = merged_runs

    chunks = []
    for page_number, section, text in runs:
        chunks.extend(
            {"content": content, "section": section, "page_number": page_number}
            for content in chunk_text(text)
        )
    return _add_page_boundary_overlap(chunks)


def _add_page_boundary_overlap(chunks: list[dict]) -> list[dict]:
    """Copy a bounded tail of the prior PDF page into the next page's first chunk."""
    output: list[dict] = []
    overlap = settings.rag_chunk_overlap
    size = settings.rag_chunk_size
    for chunk in chunks:
        page = chunk.get("page_number")
        previous = output[-1] if output else None
        previous_page = previous.get("page_number") if previous else None
        if (
            not isinstance(page, int)
            or not isinstance(previous_page, int)
            or page <= previous_page
            or overlap <= 0
        ):
            output.append(chunk)
            continue

        tail = previous["content"][-overlap:]
        if " " in tail:
            tail = tail[tail.find(" ") + 1:]
        content = chunk["content"]
        budget = size - len(tail) - 1
        # If the entire next-page chunk fits, retain it whole. Cutting at its
        # last whitespace would otherwise emit the final word as a tiny chunk.
        if len(content) <= budget:
            output.append({**chunk, "content": f"{tail}\n{content}".strip()})
            continue
        cut = content.rfind(" ", 0, budget + 1)
        if not tail or cut <= 0:
            output.append(chunk)
            continue
        merged = {**chunk, "content": f"{tail}\n{content[:cut]}".strip()}
        output.append(merged)
        remainder = content[cut + 1:].strip()
        if remainder:
            output.append({**chunk, "content": remainder})
    return output


def build_corpus_chunks(
    directory: str | Path | None = None,
    strategy: str = "heading",
) -> tuple[list[dict], list[str]]:
    """Build indexed chunks without changing the vector store."""
    root = Path(directory or settings.rag_document_dir)
    if not root.exists():
        raise FileNotFoundError(f"Document directory does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Document path is not a directory: {root}")
    excluded = {name.casefold() for name in settings.rag_exclude_files}
    files = sorted(
        p for p in root.rglob("*")
        if p.suffix.lower() in {".md", ".txt", ".docx", ".pdf"}
        and p.name.casefold() not in excluded
    )
    records: list[dict] = []
    sources = [path.relative_to(root).as_posix() for path in files]
    if not files:
        raise ValueError(f"No supported documents found in {root}")
    for path in files:
        source = path.relative_to(root).as_posix()
        file_chunks = chunk_document_blocks(load_document_blocks(path), strategy=strategy)
        if not file_chunks:
            logger.warning("Document produced no text chunks (possibly scanned/empty): %s", path)
        for index, chunk in enumerate(file_chunks):
            records.append({"source": source, "chunk_index": index, **chunk})
    if not records:
        raise ValueError(f"No text chunks could be extracted from documents in {root}")
    return records, sources


def ingest_directory(
    directory: str | Path | None = None,
    *,
    strategy: str = "heading",
) -> dict:
    from app.core.vector_store import replace_corpus_chunks
    root = Path(directory or settings.rag_document_dir)
    records, sources = build_corpus_chunks(root, strategy=strategy)
    stored = replace_corpus_chunks(records)
    from app.core.retriever import clear_bm25_cache
    clear_bm25_cache()
    return {
        "files": len(sources),
        "chunks": stored,
        "directory": str(root),
        "chunking_strategy": strategy,
    }
