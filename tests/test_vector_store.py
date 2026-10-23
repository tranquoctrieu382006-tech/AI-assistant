from unittest.mock import MagicMock, patch
import sys
import types

from app.core import vector_store


def test_replace_corpus_removes_stale_sources_and_replaces_current_sources():
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
          patch.object(vector_store, "register_vector"),
          patch.object(vector_store, "Vector", side_effect=lambda value: value)):
        stored = vector_store.replace_corpus_chunks(chunks, ["policy.md"])

    assert stored == 1
    assert cursor.execute.call_args_list[0].args[0] == "DELETE FROM rag_chunks WHERE source <> ALL(%s)"
    assert cursor.execute.call_args_list[0].args[1] == (["policy.md"],)
    assert cursor.execute.call_args_list[1].args[0] == "DELETE FROM rag_chunks WHERE source = ANY(%s)"
    cursor.executemany.assert_called_once()


def test_replace_empty_corpus_clears_all_indexed_chunks():
    connection = MagicMock()
    connection.__enter__.return_value = connection
    cursor = connection.cursor.return_value.__enter__.return_value

    with (patch.object(vector_store, "ensure_vector_schema"),
          patch.object(vector_store, "_connect", return_value=connection),
          patch.object(vector_store, "register_vector")):
        assert vector_store.replace_corpus_chunks([], []) == 0

    cursor.execute.assert_called_once_with("DELETE FROM rag_chunks")
    cursor.executemany.assert_not_called()


def test_embeddings_are_requested_in_configured_batches(monkeypatch):
    connection = MagicMock()
    connection.__enter__.return_value = connection
    cursor = connection.cursor.return_value.__enter__.return_value
    embeddings = MagicMock()
    embeddings.embed_documents.side_effect = [[[0.1]], [[0.2]]]
    chunks = [
        {"content": "one", "source": "a.md", "chunk_index": 0},
        {"content": "two", "source": "a.md", "chunk_index": 1},
    ]
    monkeypatch.setattr(vector_store.settings, "rag_embedding_batch_size", 1)
    with (patch.object(vector_store, "ensure_vector_schema"),
          patch.object(vector_store, "get_embeddings", return_value=embeddings),
          patch.object(vector_store, "_connect", return_value=connection),
          patch.object(vector_store, "Vector", side_effect=lambda value: value)):
        assert vector_store.replace_corpus_chunks(chunks, ["a.md"]) == 2
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
    monkeypatch.setattr(vector_store, "_POOL", None)
    assert vector_store._pool() is vector_store._pool()
    assert len(created) == 1
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
    connection.commit.assert_called_once()
