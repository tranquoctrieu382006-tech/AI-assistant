from unittest.mock import MagicMock, patch

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
