"""Build a compact, policy-filtered schema context for NLQ."""
from __future__ import annotations

import re
import copy
import logging
import threading
import time
import unicodedata
from pathlib import Path

from psycopg_pool import ConnectionPool
from psycopg import sql

from app.core.config import settings

_SENSITIVE = re.compile(r"password|passwd|secret|token|email|phone|mobile|address|account_number|national_id|full_name", re.I)
_ENUM_CANDIDATE = re.compile(r"(?:^|_)(?:status|type|tier|category|level|method)(?:_|$)", re.I)
_STOP = {
    "cho", "cua", "trong", "nhung", "nao", "theo", "bao", "nhieu", "hay",
    "va", "la", "co", "tu", "den", "thang", "nam", "ngay", "hang", "san",
    "pham", "don", "khach", "tong", "trung", "binh", "la", "bao", "nhieu",
}
_VIETNAMESE_TABLE_SYNONYMS = {
    "doanh thu": ("orders", "order_items"),
    "khach hang": ("customers",),
    "don hang": ("orders",),
    "san pham": ("products",),
    "danh muc": ("categories",),
    "nha cung cap": ("suppliers",),
    "ton kho": ("inventory",),
    "bien dong": ("stock_movements",),
    "kho": ("warehouses", "inventory", "stock_movements"),
}
_SCHEMA_CACHE_TTL_SECONDS = 600
_SCHEMA_CACHE_LOCK = threading.Lock()
_SCHEMA_CACHE: list[dict] | None = None
_SCHEMA_CACHE_EXPIRES_AT = 0.0
logger = logging.getLogger(__name__)
_NLQ_POOL: ConnectionPool | None = None
_NLQ_POOL_URL: str | None = None
_NLQ_POOL_LOCK = threading.Lock()


def _normalize_text(value: str) -> str:
    decomposed = unicodedata.normalize("NFD", value.casefold().replace("đ", "d"))
    return "".join(char for char in decomposed if unicodedata.category(char) != "Mn")


def _tokens(value: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", _normalize_text(value)))


def _connect():
    global _NLQ_POOL, _NLQ_POOL_URL
    if not settings.nlq_database_url.strip():
        raise RuntimeError("NLQ_DATABASE_URL is not configured; set it to a PostgreSQL read-only account.")
    timeout = max(1, round(settings.database_connection_timeout_seconds))
    if _NLQ_POOL is None or _NLQ_POOL_URL != settings.nlq_database_url:
        with _NLQ_POOL_LOCK:
            if _NLQ_POOL is None or _NLQ_POOL_URL != settings.nlq_database_url:
                if _NLQ_POOL is not None:
                    _NLQ_POOL.close()
                _NLQ_POOL = ConnectionPool(
                    conninfo=settings.nlq_database_url,
                    min_size=0,
                    max_size=8,
                    timeout=timeout,
                    kwargs={"connect_timeout": timeout},
                )
                _NLQ_POOL_URL = settings.nlq_database_url
    return _NLQ_POOL.connection()


def close_nlq_pool() -> None:
    global _NLQ_POOL, _NLQ_POOL_URL
    with _NLQ_POOL_LOCK:
        if _NLQ_POOL is not None:
            _NLQ_POOL.close()
            _NLQ_POOL = None
            _NLQ_POOL_URL = None


def _load_schema_uncached() -> list[dict]:
    """Read public table metadata and sample enum values using the NLQ account."""
    with _connect() as conn:
        conn.execute("SET TRANSACTION READ ONLY")
        rows = conn.execute("""
            SELECT rel.relname, att.attname,
                   pg_catalog.format_type(att.atttypid, att.atttypmod),
                   att.attnotnull, pg_catalog.pg_get_expr(def.adbin, def.adrelid)
            FROM pg_catalog.pg_class rel
            JOIN pg_catalog.pg_namespace ns ON ns.oid = rel.relnamespace
            JOIN pg_catalog.pg_attribute att ON att.attrelid = rel.oid
            LEFT JOIN pg_catalog.pg_attrdef def
              ON def.adrelid = rel.oid AND def.adnum = att.attnum
            WHERE ns.nspname = 'public'
              AND rel.relname <> 'rag_chunks'
              AND rel.relkind IN ('r', 'p', 'v', 'm')
              AND att.attnum > 0 AND NOT att.attisdropped
            ORDER BY rel.relname, att.attnum
        """).fetchall()
        constraints = conn.execute("""
            SELECT rel.relname, att.attname, con.contype,
                   referenced.relname, referenced_att.attname
            FROM pg_catalog.pg_constraint con
            JOIN pg_catalog.pg_class rel ON rel.oid = con.conrelid
            JOIN pg_catalog.pg_namespace ns ON ns.oid = rel.relnamespace
            CROSS JOIN LATERAL unnest(con.conkey) WITH ORDINALITY AS key(attnum, ord)
            JOIN pg_catalog.pg_attribute att
              ON att.attrelid = rel.oid AND att.attnum = key.attnum
            LEFT JOIN pg_catalog.pg_class referenced ON referenced.oid = con.confrelid
            LEFT JOIN LATERAL unnest(con.confkey) WITH ORDINALITY AS refkey(attnum, ord)
              ON refkey.ord = key.ord
            LEFT JOIN pg_catalog.pg_attribute referenced_att
              ON referenced_att.attrelid = referenced.oid
             AND referenced_att.attnum = refkey.attnum
            WHERE ns.nspname = 'public'
              AND rel.relname <> 'rag_chunks'
              AND con.contype IN ('p', 'f')
            ORDER BY rel.relname, att.attnum, con.contype
        """).fetchall()
    tables: dict[str, dict] = {}
    for table, column, dtype, not_null, default in rows:
        if _SENSITIVE.search(table) or _SENSITIVE.search(column):
            continue
        entry = tables.setdefault(table, {"name": table, "columns": [], "relations": set()})
        entry["columns"].append({
            "name": column,
            "type": dtype,
            "nullable": not not_null,
            "default": default,
            "tags": [],
        })
    columns_by_table = {
        table_name: {column["name"]: column for column in table["columns"]}
        for table_name, table in tables.items()
    }
    # pg_constraint is visible to read-only roles and returns each key column once,
    # including composite PK/FKs that information_schema joins can duplicate.
    for table, column, kind, ref_table, ref_column in constraints:
        if table not in tables or column not in columns_by_table[table]:
            continue
        tags = columns_by_table[table][column]["tags"]
        if kind == "p" and "PK" not in tags:
            tags.append("PK")
        elif kind == "f" and ref_table and ref_column:
            tag = f"FK → {ref_table}.{ref_column}"
            if tag not in tags:
                tags.append(tag)
            tables[table]["relations"].add(ref_table)
            if ref_table in tables:
                tables[ref_table]["relations"].add(table)
    result = list(tables.values())
    for table in result:
        table["relations"] = sorted(name for name in table["relations"] if name in tables)
    # Only sample short values from likely low-cardinality columns. The
    # read-only transaction and quoted identifiers bound this metadata query.
    with _connect() as conn:
        conn.execute("SET TRANSACTION READ ONLY")
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


def load_schema(*, force_refresh: bool = False) -> list[dict]:
    """Return a defensive copy of the schema, refreshing it every 10 minutes."""
    global _SCHEMA_CACHE, _SCHEMA_CACHE_EXPIRES_AT
    now = time.monotonic()
    if not force_refresh and _SCHEMA_CACHE is not None and now < _SCHEMA_CACHE_EXPIRES_AT:
        return copy.deepcopy(_SCHEMA_CACHE)
    with _SCHEMA_CACHE_LOCK:
        now = time.monotonic()
        if force_refresh or _SCHEMA_CACHE is None or now >= _SCHEMA_CACHE_EXPIRES_AT:
            _SCHEMA_CACHE = _load_schema_uncached()
            _SCHEMA_CACHE_EXPIRES_AT = time.monotonic() + _SCHEMA_CACHE_TTL_SECONDS
        return copy.deepcopy(_SCHEMA_CACHE)


def clear_schema_cache() -> None:
    """Invalidate cached metadata after schema changes or in tests."""
    global _SCHEMA_CACHE, _SCHEMA_CACHE_EXPIRES_AT
    with _SCHEMA_CACHE_LOCK:
        _SCHEMA_CACHE = None
        _SCHEMA_CACHE_EXPIRES_AT = 0.0


def schema_catalog(tables: list[dict]) -> str:
    dictionary_path = Path(settings.nlq_schema_dictionary_path)
    dictionary = dictionary_path.read_text(encoding="utf-8") if dictionary_path.exists() else ""
    if not dictionary:
        logger.warning("NLQ schema dictionary is missing or empty: %s", dictionary_path)
    descriptions: dict[str, list[str]] = {table["name"]: [] for table in tables}
    in_table_section = False
    for line in dictionary.splitlines():
        heading = line.strip().casefold()
        if heading.startswith("## "):
            in_table_section = heading == "## bảng và cột"
            continue
        if not in_table_section or not line.strip().startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 4 or set(cells[0]) <= {"-", ":"}:
            continue
        table_name, column_name, _, meaning = cells[:4]
        if (
            table_name not in descriptions
            or table_name.casefold() == "bảng"
            or _SENSITIVE.search(table_name)
            or _SENSITIVE.search(column_name)
            or not meaning
        ):
            continue
        item = f"{column_name}: {meaning}"
        if item not in descriptions[table_name]:
            descriptions[table_name].append(item)
    lines = []
    for table in tables:
        cols = ", ".join(
            f"{col['name']} {col['type']}"
            + (f" [{', '.join(col['tags'])}]" if col["tags"] else "")
            + (f" values={col['samples']}" if col.get("samples") else "")
            for col in table["columns"]
        )
        relations = f"; joins: {', '.join(table['relations'])}" if table["relations"] else ""
        description_text = "; ".join(descriptions.get(table["name"], []))[:700]
        description = f"; mô tả: {description_text}" if description_text else ""
        lines.append(f"{table['name']} ({cols}){relations}{description}")
    return "\n".join(lines)


def select_relevant_tables(
    question: str, tables: list[dict], llm=None, callbacks: list | None = None
) -> list[dict]:
    """LLM schema linking with a deterministic lexical fallback and FK expansion."""
    if not tables:
        raise ValueError("No queryable public tables are available to the NLQ account.")
    selected: list[str] = []
    llm_resolved = False
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
            raw = extract_message_text(response).strip().strip("`")
            if raw.startswith("json"):
                raw = raw[4:].strip()
            import json
            names = json.loads(raw)
            if isinstance(names, list):
                allowed = {t["name"] for t in tables}
                selected = [n for n in names if isinstance(n, str) and n in allowed]
                # An explicit empty list means the model judged the question out of scope.
                if not names:
                    return []
                llm_resolved = bool(selected)
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning(
                "NLQ schema linking failed; using lexical fallback (%s)", type(exc).__name__
            )
            selected = []
    if not selected and not llm_resolved:
        normalized_question = _normalize_text(question)
        words = {word for word in _tokens(question) if len(word) > 2 and word not in _STOP}
        ranked = []
        for table in tables:
            vocabulary = " ".join(
                [table["name"]]
                + [column["name"] for column in table["columns"]]
                + [sample for column in table["columns"] for sample in column.get("samples", [])]
            )
            score = len(words & _tokens(vocabulary))
            ranked.append((score, table["name"]))
        available = {table["name"] for table in tables}
        for phrase, targets in _VIETNAMESE_TABLE_SYNONYMS.items():
            if re.search(rf"\b{re.escape(phrase)}\b", normalized_question):
                for target in targets:
                    if target in available:
                        ranked = [
                            (score + (2 if name == target else 0), name)
                            for score, name in ranked
                        ]
        if ranked:
            best = max(score for score, _ in ranked)
            selected = [name for score, name in ranked if score == best and score > 0]
    if not selected:
        return []
    by_name = {t["name"]: t for t in tables}
    expanded = set(selected)
    for name in selected:
        expanded.update(by_name[name]["relations"])
    return [t for t in tables if t["name"] in expanded]
