from unittest.mock import MagicMock, patch
import sys
import types

from app.core import vector_store


def setup_function():
    vector_store._STORE_READY = None


def test_replace_corpus_removes_stale_sources_and_replaces_current_sources(monkeypatch):
    monkeypatch.setattr(vector_store.settings, "embedding_dimensions", 2)
    connection = MagicMock()
    connection.__enter__.return_value = connection
    cursor_context = connection.cursor.return_value
    cursor = cursor_context.__enter__.return_value
    chunks = [{"content": "new text", "source": "policy.md", "chunk_index": 0}]
    embeddings = MagicMock()
    embeddings.embed_documents.return_value = [[0.1, 0.2]]

    with (patch.object(vector_store, "ensure_vector_schema"),
          patch.object(vector_store, "get_embeddings", return_value=embeddings),
          patch.object(vector_store, "_connect", return_value=connection),
          patch.object(vector_store, "register_vector") as register,
          patch.object(vector_store, "Vector", side_effect=lambda value: value)):
        stored = vector_store.replace_corpus_chunks(chunks)

    assert stored == 1
    register.assert_called_once_with(connection)
    assert connection._vector_adapter_registered is True
    assert cursor.execute.call_args_list[0].args[0] == "DELETE FROM rag_chunks"
    assert cursor.execute.call_count == 1
    cursor.executemany.assert_called_once()


def test_replace_empty_corpus_is_rejected_without_touching_database():
    with (patch.object(vector_store, "ensure_vector_schema") as ensure,
          patch.object(vector_store, "_connect") as connect):
        try:
            vector_store.replace_corpus_chunks([])
        except ValueError as exc:
            assert "empty chunk list" in str(exc)
        else:
            raise AssertionError("Replacing the corpus with no chunks must be rejected")

    ensure.assert_not_called()
    connect.assert_not_called()


def test_embeddings_are_requested_in_configured_batches(monkeypatch):
    monkeypatch.setattr(vector_store.settings, "embedding_dimensions", 2)
    connection = MagicMock()
    connection.__enter__.return_value = connection
    cursor = connection.cursor.return_value.__enter__.return_value
    embeddings = MagicMock()
    embeddings.embed_documents.side_effect = [[[0.1, 0.2]], [[0.2, 0.3]]]
    chunks = [
        {"content": "one", "source": "a.md", "chunk_index": 0},
        {"content": "two", "source": "a.md", "chunk_index": 1},
    ]
    monkeypatch.setattr(vector_store.settings, "rag_embedding_batch_size", 1)
    with (patch.object(vector_store, "ensure_vector_schema"),
          patch.object(vector_store, "get_embeddings", return_value=embeddings),
          patch.object(vector_store, "_connect", return_value=connection),
          patch.object(vector_store, "register_vector"),
          patch.object(vector_store, "Vector", side_effect=lambda value: value)):
        assert vector_store.replace_corpus_chunks(chunks) == 2
    assert embeddings.embed_documents.call_count == 2
    cursor.executemany.assert_called_once()


def test_connection_pool_is_reused_and_closed(monkeypatch):
    created = []

    class FakePool:
        def __init__(self, *args, **kwargs):
            self.args, self.kwargs, self.closed = args, kwargs, False
            created.append(self)
        def connection(self): return "pooled-connection"
        def close(self): self.closed = True

    monkeypatch.setitem(sys.modules, "psycopg_pool", types.SimpleNamespace(ConnectionPool=FakePool))
    monkeypatch.setattr(vector_store.settings, "database_url", "postgresql://local/db")
    monkeypatch.setattr(vector_store.settings, "database_connection_timeout_seconds", 3)
    monkeypatch.setattr(vector_store, "_POOL", None)
    assert vector_store._pool() is vector_store._pool()
    assert len(created) == 1
    assert created[0].kwargs["timeout"] == 3
    assert created[0].kwargs["kwargs"]["connect_timeout"] == 3
    assert vector_store._connect() == "pooled-connection"
    vector_store.close_pool()
    assert created[0].closed


def test_pool_connection_skips_vector_registration_before_extension_exists():
    connection = MagicMock()
    connection.execute.return_value.fetchone.return_value = (None,)
    with patch.object(vector_store, "register_vector") as register:
        vector_store._configure_connection(connection)
    register.assert_not_called()
    connection.commit.assert_called_once()


def test_pool_connection_registers_vector_when_extension_exists():
    connection = MagicMock()
    connection.execute.return_value.fetchone.return_value = ("vector",)
    with patch.object(vector_store, "register_vector") as register:
        vector_store._configure_connection(connection)
    register.assert_called_once_with(connection)
    assert connection._vector_adapter_registered is True
    connection.commit.assert_called_once()


def test_similarity_search_registers_vector_once_after_extension_is_added(monkeypatch):
    monkeypatch.setattr(vector_store.settings, "embedding_dimensions", 2)
    connection = MagicMock()
    connection.__enter__.return_value = connection
    connection.execute.side_effect = [
        MagicMock(fetchone=MagicMock(return_value=("vector", "rag_chunks"))),
        MagicMock(fetchall=MagicMock(return_value=[])),
        MagicMock(fetchone=MagicMock(return_value=("vector", "rag_chunks"))),
        MagicMock(fetchall=MagicMock(return_value=[])),
    ]
    embeddings = MagicMock()
    embeddings.embed_query.return_value = [0.1, 0.2]

    with (patch.object(vector_store, "get_embeddings", return_value=embeddings),
          patch.object(vector_store, "_connect", return_value=connection),
          patch.object(vector_store, "register_vector") as register,
          patch.object(vector_store, "Vector", side_effect=lambda value: value)):
        assert vector_store.similarity_search("question") == []
        assert vector_store.similarity_search("second question") == []

    register.assert_called_once_with(connection)
    assert connection.execute.call_count == 3


def test_similarity_search_returns_section_metadata(monkeypatch):
    monkeypatch.setattr(vector_store.settings, "embedding_dimensions", 2)
    connection = MagicMock()
    connection.__enter__.return_value = connection
    connection.execute.side_effect = [
        MagicMock(fetchone=MagicMock(return_value=("vector", "rag_chunks"))),
        MagicMock(fetchall=MagicMock(return_value=[
            ("policy text", "policy.md", 2, None, "Leave > Annual", 0.9)
        ])),
    ]
    embeddings = MagicMock()
    embeddings.embed_query.return_value = [0.1, 0.2]

    with (patch.object(vector_store, "get_embeddings", return_value=embeddings),
          patch.object(vector_store, "_connect", return_value=connection),
          patch.object(vector_store, "register_vector"),
          patch.object(vector_store, "Vector", side_effect=lambda value: value)):
        results = vector_store.similarity_search("question")

    assert results[0]["section"] == "Leave > Annual"
    assert results[0]["score"] == 0.9


def test_score_chunks_returns_real_cosine_scores_for_requested_keys(monkeypatch):
    monkeypatch.setattr(vector_store.settings, "embedding_dimensions", 2)
    connection = MagicMock()
    connection.__enter__.return_value = connection
    connection.execute.side_effect = [
        MagicMock(fetchone=MagicMock(return_value=("vector", "rag_chunks"))),
        MagicMock(fetchall=MagicMock(return_value=[("policy.md", 7, 0.62)])),
    ]
    embeddings = MagicMock()
    embeddings.embed_query.return_value = [0.1, 0.2]

    with (
        patch.object(vector_store, "get_embeddings", return_value=embeddings),
        patch.object(vector_store, "_connect", return_value=connection),
        patch.object(vector_store, "register_vector"),
        patch.object(vector_store, "Vector", side_effect=lambda value: value),
    ):
        scores = vector_store.score_chunks("question", [("policy.md", 7)])

    assert scores == {("policy.md", 7): 0.62}
    sql, params = connection.execute.call_args_list[1].args
    assert "WHERE (source, chunk_index) IN (VALUES (%s, %s))" in sql
    assert params == [[0.1, 0.2], "policy.md", 7]


def test_get_corpus_revision_uses_count_and_latest_chunk_id():
    connection = MagicMock()
    connection.__enter__.return_value = connection
    connection.execute.side_effect = [
        MagicMock(fetchone=MagicMock(return_value=("vector", "rag_chunks"))),
        MagicMock(fetchone=MagicMock(return_value=(12, 98))),
    ]

    with (
        patch.object(vector_store, "_connect", return_value=connection),
        patch.object(vector_store, "register_vector"),
    ):
        revision = vector_store.get_corpus_revision()

    assert revision == (12, 98)
