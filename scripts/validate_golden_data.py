"""Validate checked-in RAG/NLQ benchmark data without external services."""
import json
import re
import unicodedata
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CORPUS_DIR = DATA_DIR / "demo_sales_warehouse"
ALLOWED_RAG_CATEGORIES = {
    "simple", "paraphrase", "multi_document",
    "out_of_scope_far", "in_scope_missing",
}
_REPLACEMENT_QUESTION_MARK = re.compile(r"[^\W\d_]\?[^\W\d_]", re.UNICODE)


def _source_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".md", ".txt"}:
        return path.read_text(encoding="utf-8-sig")
    if suffix == ".docx":
        from docx import Document

        document = Document(path)
        parts = [paragraph.text for paragraph in document.paragraphs]
        parts.extend(cell.text for table in document.tables for row in table.rows for cell in row.cells)
        return "\n".join(parts)
    if suffix == ".pdf":
        from pypdf import PdfReader

        return "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    return ""


def _normalized(value: str) -> str:
    return " ".join(unicodedata.normalize("NFC", value).casefold().split())


def validate_text(value: str, label: str) -> list[str]:
    errors = []
    if unicodedata.normalize("NFC", value) != value:
        errors.append(f"{label}: text is not Unicode NFC")
    if _REPLACEMENT_QUESTION_MARK.search(value):
        errors.append(f"{label}: question mark appears between word characters")
    return errors


def validate_rag_cases(cases: list[dict], corpus_dir: Path = CORPUS_DIR) -> list[str]:
    errors = []
    seen = set()
    source_text_cache = {}
    for index, case in enumerate(cases):
        label = case.get("id", f"row {index + 1}")
        case_id = case.get("id")
        if not case_id or case_id in seen:
            errors.append(f"{label}: missing or duplicate id")
        seen.add(case_id)
        for field in ("question", "answer_contains"):
            value = case.get(field)
            if isinstance(value, list):
                for item_index, item in enumerate(value):
                    if isinstance(item, str):
                        errors.extend(validate_text(item, f"{label}.{field}[{item_index}]"))
            elif isinstance(value, str):
                errors.extend(validate_text(value, f"{label}.{field}"))
            else:
                errors.append(f"{label}: missing/invalid {field}")
        category = case.get("category")
        if category not in ALLOWED_RAG_CATEGORIES:
            errors.append(f"{label}: invalid category {category!r}")
        sources = case.get("expected_sources")
        if sources is None:
            sources = [case["source"]] if case.get("source") else []
        if category in {"out_of_scope_far", "in_scope_missing"} and sources:
            errors.append(f"{label}: abstention case must not have expected sources")
        if category not in {"out_of_scope_far", "in_scope_missing"} and not sources:
            errors.append(f"{label}: answerable case needs expected sources")
        for source in sources:
            if not (corpus_dir / source).is_file():
                errors.append(f"{label}: source does not exist: {source}")
        if category == "multi_document" and len(set(sources)) < 2:
            errors.append(f"{label}: multi_document case needs evidence from at least two sources")
        if sources:
            evidence = case.get("evidence")
            if not isinstance(evidence, list) or not evidence:
                errors.append(f"{label}: answerable case needs evidence text")
            else:
                evidence_sources = {item.get("source") for item in evidence if isinstance(item, dict)}
                if set(sources) != evidence_sources:
                    errors.append(f"{label}: evidence sources must match expected sources")
                for evidence_index, item in enumerate(evidence):
                    if not isinstance(item, dict) or not isinstance(item.get("text"), str) or not item["text"].strip():
                        errors.append(f"{label}: evidence[{evidence_index}] needs non-empty text")
                    else:
                        errors.extend(validate_text(item["text"], f"{label}.evidence[{evidence_index}].text"))
                        if not item.get("source"):
                            errors.append(f"{label}: evidence[{evidence_index}] needs a source")
                    if isinstance(item, dict) and item.get("source") and not (corpus_dir / item["source"]).is_file():
                        errors.append(f"{label}: evidence source does not exist: {item['source']}")
                    elif isinstance(item, dict) and item.get("source") and isinstance(item.get("text"), str):
                        source = item["source"]
                        if source not in source_text_cache:
                            source_text_cache[source] = _normalized(_source_text(corpus_dir / source))
                        if _normalized(item["text"]) not in source_text_cache[source]:
                            errors.append(f"{label}: evidence text not found in source {source}")
    missing_count = sum(case.get("category") == "in_scope_missing" for case in cases)
    if missing_count < 10:
        errors.append(f"in_scope_missing has {missing_count} cases; need at least 10 before reporting a rate")
    return errors


def validate_nlq_cases(cases: list[dict]) -> list[str]:
    errors = []
    seen = set()
    for index, case in enumerate(cases):
        label = case.get("id", f"row {index + 1}")
        if not case.get("id") or case["id"] in seen:
            errors.append(f"{label}: missing or duplicate id")
        seen.add(case.get("id"))
        for field in ("question", "sql"):
            if not isinstance(case.get(field), str) or not case[field].strip():
                errors.append(f"{label}: missing/invalid {field}")
        if "expected" not in case:
            errors.append(f"{label}: missing expected result")
    return errors


def main() -> int:
    rag_path = DATA_DIR / "rag_golden_set.json"
    nlq_path = DATA_DIR / "nlq_golden_set.json"
    rag = json.loads(rag_path.read_text(encoding="utf-8"))
    nlq = json.loads(nlq_path.read_text(encoding="utf-8"))
    errors = validate_rag_cases(rag) + validate_nlq_cases(nlq)
    if errors:
        print("Golden data validation failed:")
        print("\n".join(f"- {error}" for error in errors))
        return 1
    print(f"Golden data valid: RAG {len(rag)} cases, NLQ {len(nlq)} cases")
    print(f"in_scope_missing: {sum(c['category'] == 'in_scope_missing' for c in rag)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
