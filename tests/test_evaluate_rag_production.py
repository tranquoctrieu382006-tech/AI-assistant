from scripts.evaluate_rag_production import evaluate_cases


def test_production_evaluator_uses_retrieve_gate_config_and_strict_multisource_hit(monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "rag_top_k", 5)
    monkeypatch.setattr(settings, "search_mode", "hybrid")
    monkeypatch.setattr(settings, "use_reranker", True)
    monkeypatch.setattr(settings, "rag_min_relevance_score", 0.5)
    calls = []

    def retriever(question, **kwargs):
        calls.append((question, kwargs))
        if question == "multi?":
            return ([
                {"source": "a.md", "chunk_index": 0, "content": "evidence a", "score": 0.8},
                {"source": "noise.md", "chunk_index": 0, "content": "noise", "score": 0.7},
            ], False)
        if question == "answerable?":
            return ([
                {"source": "a.md", "chunk_index": 0, "content": "evidence a", "score": 0.8},
            ], False)
        return ([], True)

    cases = [
        {"id": "multi", "question": "multi?", "category": "multi_document",
         "expected_sources": ["a.md", "b.md"], "evidence": []},
        {"id": "answer", "question": "answerable?", "category": "simple",
         "source": "a.md", "evidence": []},
        {"id": "reject", "question": "reject?", "category": "in_scope_missing",
         "source": None, "evidence": []},
    ]
    report = evaluate_cases(cases, retriever=retriever)

    assert len(calls) == 3
    assert calls[0][1] == {
        "top_k": 5, "mode": "hybrid", "use_reranker": True,
        "reranker_top_n": 5, "reranker_pool_size": 20,
    }
    assert report["metrics"]["source_hit@5"] == 0.5
    assert report["metrics"]["first_relevant_source_mrr"] == 1.0
    assert report["abstention_risk"]["in_scope_missing"]["false_positives"] == 0
