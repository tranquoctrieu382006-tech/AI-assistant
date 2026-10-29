"""
Integration tests with a live PostgreSQL + pgvector instance.
Được thiết kế để chạy trong CI (GitHub Actions service container) hoặc môi trường dev có DB thật.
Nếu không có kết nối DB thật, test sẽ tự động skip an toàn để không làm ảnh hưởng unit test thông thường.
"""

import os
import hashlib
import pytest
import psycopg
from psycopg import errors

from app.core.config import settings


def get_live_db_connection():
    """Thử kết nối tới database nếu DATABASE_URL được cấu hình."""
    url = settings.database_url or os.environ.get("DATABASE_URL", "")
    if not url.strip():
        return None
    try:
        conn = psycopg.connect(url, connect_timeout=2)
        return conn
    except Exception:
        return None


def get_deterministic_embedding(text: str, dim: int = 1024) -> list[float]:
    """
    Tạo vector embedding giả định có tính xác định (deterministic) dựa trên SHA-256.
    Không gọi mạng hay Ollama, chuẩn hóa độ dài vector để kiểm thử khoảng cách cosine trên pgvector thật.
    """
    seed_hash = hashlib.sha256(text.encode("utf-8")).digest()
    floats = []
    for i in range(dim):
        byte_val = seed_hash[i % len(seed_hash)]
        floats.append((byte_val - 128) / 128.0)
    norm = sum(x * x for x in floats) ** 0.5 or 1.0
    return [round(x / norm, 6) for x in floats]


@pytest.fixture
def live_conn():
    conn = get_live_db_connection()
    if conn is None:
        pytest.skip("Bỏ qua integration test: Không có kết nối tới PostgreSQL thật.")
    try:
        yield conn
    finally:
        conn.close()
        from app.core.vector_store import close_pool

        close_pool()


def test_postgres_pgvector_extension_exists(live_conn):
    """Kiểm tra extension pgvector đã được cài đặt và kích hoạt thành công."""
    with live_conn.cursor() as cur:
        cur.execute("SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';")
        row = cur.fetchone()
        assert row is not None, "Extension vector chưa được cài đặt trong PostgreSQL."
        assert row[0] == "vector"


def test_postgres_vector_similarity_search_live(live_conn):
    """
    Kiểm tra luồng lưu trữ vector và truy vấn tương đồng cosine (<=>) trên pgvector thật
    với embedding giả lập xác định.
    """
    dim = 1024
    with live_conn.cursor() as cur:
        # Tạo bảng tạm kiểm thử vector
        cur.execute(f"""
            CREATE TEMP TABLE test_vector_search (
                id SERIAL PRIMARY KEY,
                source TEXT NOT NULL,
                content TEXT NOT NULL,
                embedding vector({dim}) NOT NULL
            );
        """)

        # Nạp 3 đoạn văn bản với vector xác định
        texts = [
            ("doc1", "Chính sách đổi trả hàng hóa trong 7 ngày."),
            ("doc2", "Quy định bảo hành sản phẩm điện tử 12 tháng."),
            ("doc3", "Hướng dẫn sử dụng hệ thống kho bãi logistics."),
        ]
        for src, text in texts:
            emb = get_deterministic_embedding(text, dim=dim)
            cur.execute(
                "INSERT INTO test_vector_search (source, content, embedding) VALUES (%s, %s, %s::vector)",
                (src, text, emb),
            )

        # Truy vấn tìm đoạn gần nhất với chính doc1
        query_text = "Chính sách đổi trả hàng hóa trong 7 ngày."
        query_emb = get_deterministic_embedding(query_text, dim=dim)

        cur.execute("""
            SELECT source, content, (1 - (embedding <=> %s::vector)) as cosine_similarity
            FROM test_vector_search
            ORDER BY embedding <=> %s::vector
            LIMIT 1;
        """, (query_emb, query_emb))

        top_match = cur.fetchone()
        assert top_match is not None
        assert top_match[0] == "doc1"
        assert top_match[2] >= 0.9999  # Chính xác tuyệt đối


def test_postgres_session_corpus_isolation(live_conn):
    """Session A must not retrieve chunks that belong to session B."""
    from uuid import uuid4

    from app.core import vector_store

    if not settings.database_url.strip():
        pytest.skip("DATABASE_URL chưa được nạp vào settings cho vector store.")

    vector_store.ensure_vector_schema()
    suffix = uuid4().hex
    corpus_a = f"session_a_{suffix}"
    corpus_b = f"session_b_{suffix}"
    dim = settings.embedding_dimensions
    embedding_a = get_deterministic_embedding(f"session-a-{suffix}", dim)
    embedding_b = get_deterministic_embedding(f"session-b-{suffix}", dim)

    def vector_literal(values: list[float]) -> str:
        return "[" + ",".join(f"{value:.8f}" for value in values) + "]"

    with live_conn.cursor() as cur:
        for corpus, content, embedding in (
            (corpus_a, "Tài liệu riêng của phiên A", embedding_a),
            (corpus_b, "Tài liệu riêng của phiên B", embedding_b),
        ):
            cur.execute(
                """INSERT INTO rag_chunks
                       (content, source, chunk_index, corpus, embedding)
                   VALUES (%s, %s, 0, %s, %s::vector)""",
                (content, f"isolation_{suffix}.txt", corpus, vector_literal(embedding)),
            )
    live_conn.commit()

    try:
        hits_a = vector_store.similarity_search(
            "phiên A", k=5, embedding=embedding_a, corpus=corpus_a
        )
        assert [hit["content"] for hit in hits_a] == ["Tài liệu riêng của phiên A"]
        assert {hit["corpus"] for hit in hits_a} == {corpus_a}

        hits_b = vector_store.similarity_search(
            "phiên B", k=5, embedding=embedding_b, corpus=corpus_b
        )
        assert [hit["content"] for hit in hits_b] == ["Tài liệu riêng của phiên B"]
        assert {hit["corpus"] for hit in hits_b} == {corpus_b}

        chunks_a = vector_store.get_all_chunks(corpus=corpus_a)
        assert {chunk["corpus"] for chunk in chunks_a} == {corpus_a}
        assert all(chunk["content"] == "Tài liệu riêng của phiên A" for chunk in chunks_a)
    finally:
        with live_conn.cursor() as cur:
            cur.execute(
                "DELETE FROM rag_chunks WHERE corpus IN (%s, %s)",
                (corpus_a, corpus_b),
            )
        live_conn.commit()


def test_nlq_reader_role_security_boundaries():
    """
    Kiểm tra tài khoản ai_nlq_reader bị giới hạn quyền truy cập an toàn trên DB thật.
    """
    nlq_url = settings.nlq_database_url or os.environ.get("NLQ_DATABASE_URL", "")
    if not nlq_url.strip() or "ai_nlq_reader" not in nlq_url:
        pytest.skip("Bỏ qua test: NLQ_DATABASE_URL chưa cấu hình với tài khoản ai_nlq_reader.")

    try:
        conn = psycopg.connect(nlq_url, connect_timeout=2)
    except Exception as exc:
        pytest.skip(f"Không thể kết nối bằng ai_nlq_reader: {exc}")

    try:
        with conn.cursor() as cur:
            # 1. SELECT bảng kinh doanh được phép
            cur.execute("SELECT count(*) FROM orders;")
            assert cur.fetchone() is not None

            # 2. SELECT cột được phép của customers
            cur.execute("SELECT customer_id, member_tier FROM customers LIMIT 1;")
            assert cur.fetchone() is not None

            # 3. Tuyệt đối KHÔNG được SELECT cột nhạy cảm (email, full_name, phone)
            with pytest.raises(errors.InsufficientPrivilege):
                cur.execute("SELECT email FROM customers LIMIT 1;")
            conn.rollback()

            # 4. Tuyệt đối KHÔNG được ghi (INSERT/UPDATE/DELETE)
            with pytest.raises((errors.InsufficientPrivilege, errors.ReadOnlySqlTransaction)):
                cur.execute("INSERT INTO categories (name) VALUES ('Hàng lậu');")
            conn.rollback()

            # 5. Tuyệt đối KHÔNG được đọc bảng rag_chunks
            with pytest.raises((errors.InsufficientPrivilege, errors.UndefinedTable)):
                cur.execute("SELECT count(*) FROM rag_chunks;")
            conn.rollback()
    finally:
        conn.close()
