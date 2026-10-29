"""Build a compact, policy-filtered schema context for NLQ."""
from __future__ import annotations

import re
import copy
import hashlib
import logging
import threading
import time
import unicodedata
from pathlib import Path

import psycopg
from psycopg import sql

from app.core.config import settings

_SENSITIVE = re.compile(r"password|passwd|secret|token|email|phone|mobile|address|account_number|national_id|full_name", re.I)
_ENUM_CANDIDATE = re.compile(r"(?:^|_)(?:status|type|tier|category|level|method)(?:_|$)", re.I)
_STOP = {"cho", "cua", "trong", "nhung", "nao", "theo", "bao", "nhieu", "hay", "va", "la", "co", "tu", "den", "thang", "nam", "ngay", "hang", "san", "pham", "don", "khach", "tong", "trung", "binh"}
_CACHE_LOCK = threading.Lock()
_SCHEMA_CACHE: tuple[str, float, list[dict]] | None = None
_VIETNAMESE_SCHEMA_ALIASES = {
    "doanh thu": {"orders", "order_items"}, "doanh so": {"orders", "order_items"},
    "khach hang": {"customers"}, "don hang": {"orders"},
    "san pham": {"products"}, "danh muc": {"categories"},
    "nha cung cap": {"suppliers"}, "kho hang": {"warehouses"},
    "ton kho": {"inventory"}, "bien dong": {"stock_movements"},
    "nhap kho": {"stock_movements"}, "xuat kho": {"stock_movements"},
    "hang ton": {"inventory"}, "gia ban": {"products"},
    "phi giao hang": {"orders"}, "van chuyen": {"orders"},
    "trang thai": {"orders", "stock_movements"}, "don": {"orders"},
    "khach": {"customers"}, "so luong": {"order_items", "inventory", "stock_movements"},
    "hang vang": {"customers"}, "hang bac": {"customers"},
    "hang thanh vien": {"customers"}, "thanh pho": {"customers", "suppliers", "warehouses"},
    "nhap": {"stock_movements"}, "xuat": {"stock_movements"},
    "giao dich": {"orders"},
    "ai mua": {"customers", "orders"}, "nguoi mua": {"customers", "orders"},
    "mua nhieu": {"customers", "orders", "order_items"}, "dat hang": {"orders", "order_items"},
}


def _ascii_fold(value: str) -> str:
    value = value.casefold().replace("đ", "d")
    return "".join(
        char for char in unicodedata.normalize("NFKD", value)
        if not unicodedata.combining(char)
    )


def _question_schema_terms(question: str) -> set[str]:
    normalized = _ascii_fold(question)
    terms = set(re.findall(r"[a-z0-9_]+", normalized))
    for phrase, tables in _VIETNAMESE_SCHEMA_ALIASES.items():
        if phrase in normalized:
            terms.update(tables)
    return terms


def _connect():
    if not settings.nlq_database_url.strip():
        raise RuntimeError("NLQ_DATABASE_URL is not configured; set it to a PostgreSQL read-only account.")
    return psycopg.connect(
        settings.nlq_database_url,
        connect_timeout=max(1, round(settings.database_connection_timeout_seconds)),
    )


def _load_schema_from_database() -> list[dict]:
    """Read public table metadata and sample enum values using the NLQ account."""
    with _connect() as conn:
        conn.execute("SET TRANSACTION READ ONLY")
        rows = conn.execute("""
            SELECT c.table_name, c.column_name, c.data_type, c.is_nullable,
                   c.column_default, tc.constraint_type, kcu2.table_name,
                   kcu2.column_name
            FROM information_schema.columns c
            LEFT JOIN information_schema.key_column_usage kcu
              ON c.table_schema=kcu.table_schema AND c.table_name=kcu.table_name
             AND c.column_name=kcu.column_name
            LEFT JOIN information_schema.table_constraints tc
              ON kcu.constraint_catalog=tc.constraint_catalog
             AND kcu.constraint_schema=tc.constraint_schema
             AND kcu.constraint_name=tc.constraint_name
            LEFT JOIN information_schema.constraint_column_usage kcu2
              ON tc.constraint_catalog=kcu2.constraint_catalog
             AND tc.constraint_schema=kcu2.constraint_schema
             AND tc.constraint_name=kcu2.constraint_name
            WHERE c.table_schema='public' AND c.table_name <> 'rag_chunks'
            ORDER BY c.table_name, c.ordinal_position
        """).fetchall()
        tables: dict[str, dict] = {}
        for table, column, dtype, nullable, default, constraint, ref_table, ref_column in rows:
            if _SENSITIVE.search(table) or _SENSITIVE.search(column):
                continue
            entry = tables.setdefault(table, {"name": table, "columns": [], "relations": set()})
            tags = []
            if constraint == "PRIMARY KEY":
                tags.append("PK")
            elif constraint == "FOREIGN KEY" and ref_table:
                tags.append(f"FK → {ref_table}.{ref_column}")
                entry["relations"].add(ref_table)
            entry["columns"].append(
                {"name": column, "type": dtype, "nullable": nullable == "YES", "tags": tags}
            )
        result = list(tables.values())
        for table in result:
            table["relations"] = sorted(name for name in table["relations"] if name in tables)
        # Sample short values from low-cardinality columns in the same read-only
        # connection used for metadata, avoiding a second connection per refresh.
        conn.execute("SELECT set_config('statement_timeout', %s, true)", (f"{settings.nlq_statement_timeout_ms}ms",))
        for table in result:
            for column in table["columns"]:
                if (
                    column["type"] in {"text", "character varying", "USER-DEFINED"}
                    and _ENUM_CANDIDATE.search(column["name"])
                ):
                    query = sql.SQL(
                        "SELECT DISTINCT {column} FROM {table} "
                        "WHERE {column} IS NOT NULL ORDER BY {column} LIMIT 6"
                    ).format(
                        column=sql.Identifier(column["name"]),
                        table=sql.Identifier(table["name"]),
                    )
                    samples = [str(row[0])[:64] for row in conn.execute(query).fetchall()]
                    if samples:
                        column["samples"] = samples
        return result


def load_schema() -> list[dict]:
    """Return a defensive copy of cached metadata, refreshing it after its TTL."""
    global _SCHEMA_CACHE
    ttl = settings.nlq_schema_cache_ttl_seconds
    fingerprint = hashlib.sha256(settings.nlq_database_url.encode("utf-8")).hexdigest()
    now = time.monotonic()
    with _CACHE_LOCK:
        if (
            ttl > 0 and _SCHEMA_CACHE is not None
            and _SCHEMA_CACHE[0] == fingerprint and now - _SCHEMA_CACHE[1] < ttl
        ):
            return copy.deepcopy(_SCHEMA_CACHE[2])
        schema = _load_schema_from_database()
        if ttl > 0:
            _SCHEMA_CACHE = (fingerprint, time.monotonic(), copy.deepcopy(schema))
        else:
            _SCHEMA_CACHE = None
        return schema


def clear_schema_cache() -> None:
    """Clear process-local schema metadata cache (primarily for tests/admin)."""
    global _SCHEMA_CACHE
    with _CACHE_LOCK:
        _SCHEMA_CACHE = None


def schema_catalog(tables: list[dict]) -> str:
    dictionary_path = Path(settings.nlq_schema_dictionary_path)
    dictionary = dictionary_path.read_text(encoding="utf-8") if dictionary_path.exists() else ""
    descriptions: dict[str, str] = {}
    # The project dictionary stores table meanings in a Markdown table, after
    # the Mermaid ERD. Read only actual table rows to avoid treating ERD edges
    # as descriptions.
    for name in (table["name"] for table in tables):
        meanings = []
        for line in dictionary.splitlines():
            if not line.lstrip().startswith("|"):
                continue
            cells = [cell.strip().strip("` ") for cell in line.strip().strip("|").split("|")]
            if len(cells) >= 4 and cells[0].casefold() == name.casefold():
                meaning = cells[-1]
                if meaning and meaning != "Ý nghĩa":
                    meanings.append(meaning)
        if meanings:
            descriptions[name] = "; ".join(dict.fromkeys(meanings))
    lines = []
    for table in tables:
        cols = ", ".join(
            f"{col['name']} {col['type']}"
            + (f" [{', '.join(col['tags'])}]" if col["tags"] else "")
            + (f" values={col['samples']}" if col.get("samples") else "")
            for col in table["columns"]
        )
        relations = f"; joins: {', '.join(table['relations'])}" if table["relations"] else ""
        description = f"; {descriptions[table['name']]}" if table["name"] in descriptions else ""
        lines.append(f"{table['name']} ({cols}){relations}{description}")
    return "\n".join(lines)


def select_relevant_tables(
    question: str, tables: list[dict], llm=None, callbacks: list | None = None
) -> list[dict]:
    """LLM schema linking with a deterministic lexical fallback and FK expansion."""
    if not tables:
        raise ValueError("No queryable public tables are available to the NLQ account.")
    selected: list[str] = []
    use_lexical_fallback = llm is None
    if llm is not None:
        catalog = "\n".join(
            f"{t['name']}: " + ", ".join(
                c["name"] + (f" values={c['samples']}" if c.get("samples") else "")
                for c in t["columns"]
            )
            for t in tables
        )
        prompt = (
            "Choose the minimum table names needed to answer the question. "
            "Return only a JSON array of exact table names from the catalog; "
            "return [] if none apply.\nCATALOG:\n" + catalog + "\nQUESTION:\n" + question
        )
        try:
            from langchain_core.messages import HumanMessage
            response = llm.invoke(
                [HumanMessage(content=prompt)],
                config={"callbacks": callbacks or []},
            )
            from app.core.llm import extract_message_text
            raw = extract_message_text(response).strip()
            # Extract JSON array even if surrounded by markdown code fences or conversational text
            array_match = re.search(r"\[.*?\]", raw, re.DOTALL)
            payload = array_match.group(0) if array_match else raw.strip("`")
            if payload.startswith("json"):
                payload = payload[4:].strip()
            import json
            names = json.loads(payload)
            if isinstance(names, list):
                allowed = {t["name"] for t in tables}
                if not names:
                    return []
                selected = [n for n in names if isinstance(n, str) and n in allowed]
                use_lexical_fallback = not selected
            else:
                use_lexical_fallback = True
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning(
                "NLQ schema linking failed; using lexical fallback (%s)", type(exc).__name__
            )
            selected = []
            use_lexical_fallback = True
    if not selected and use_lexical_fallback:
        words = {w for w in _question_schema_terms(question) if len(w) > 2 and w not in _STOP}
        ranked = []
        for table in tables:
            vocab = set(re.findall(r"[a-z0-9_]+", (table["name"] + " " + " ".join(c["name"] for c in table["columns"])).casefold()))
            score = len(words & vocab)
            if score:
                ranked.append((score, table["name"]))
        if ranked:
            best = max(score for score, _ in ranked)
            selected = [name for score, name in ranked if score == best]
        # Do not guess unrelated tables when there is no lexical evidence.
    by_name = {t["name"]: t for t in tables}
    normalized_question = _ascii_fold(question)
    asks_revenue = "doanh thu" in normalized_question or "doanh so" in normalized_question
    if asks_revenue and set(selected).intersection({"categories", "products", "order_items", "orders"}):
        # Revenue semantics require the order status table as well as line items.
        for required_table in ("orders", "order_items"):
            if required_table in by_name and required_table not in selected:
                selected.append(required_table)
    expanded = set(selected)
    for name in selected:
        if name in by_name:
            expanded.update(by_name[name].get("relations", []))
    return [t for t in tables if t["name"] in expanded]
