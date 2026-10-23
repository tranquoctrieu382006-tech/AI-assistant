"""Compare section-packed and paragraph-preserving chunking on the RAG golden set."""
import argparse
import json
import math
from pathlib import Path, PurePosixPath

from app.core.config import settings
from app.core.ingestion import build_corpus_chunks
from app.core.vector_store import get_embeddings


def _matches_source(actual: str, expected: str) -> bool:
    return PurePosixPath(actual.replace("\\", "/")) == PurePosixPath(
        expected.replace("\\", "/")
    )


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return numerator / (left_norm * right_norm)


def _expected_sources(case: dict) -> list[str]:
    return case.get("expected_sources", [case["source"]] if case.get("source") else [])


def _evaluate_strategy(strategy: str, cases: list[dict], top_k: int) -> dict:
    chunks, sources = build_corpus_chunks(strategy=strategy)
    if not chunks:
        raise RuntimeError("The configured document corpus contains no chunks.")

    embeddings = get_embeddings()
    chunk_vectors = embeddings.embed_documents([chunk["content"] for chunk in chunks])
    if len(chunk_vectors) != len(chunks):
        raise RuntimeError("Embedding provider returned an unexpected number of vectors.")

    records = []
    query_results = {}
    for case in cases:
        query_vector = embeddings.embed_query(case["question"])
        ranked = sorted(
            (
                {
                    "source": chunk["source"],
                    "chunk_index": chunk["chunk_index"],
                    "score": _cosine_similarity(query_vector, vector),
                }
                for chunk, vector in zip(chunks, chunk_vectors)
            ),
            key=lambda result: result["score"],
            reverse=True,
        )[:top_k]
        expected = _expected_sources(case)
        source_ranks = []
        if expected:
            source_ranks = [
                next(
                    (
                        rank
                        for rank, result in enumerate(ranked, 1)
                        if _matches_source(result["source"], source)
                    ),
                    None,
                )
                for source in expected
            ]
            records.append({
                "id": case["id"],
                "category": case.get("category", "simple"),
                "expected_sources": expected,
                "ranks": source_ranks,
                "results": ranked,
            })
        else:
            top_score = ranked[0]["score"] if ranked else None
            records.append({
                "id": case["id"],
                "category": case.get("category", "out_of_scope"),
                "expected_sources": [],
                "ranks": [],
                "results": ranked,
                "abstained": top_score is None or top_score < settings.rag_min_relevance_score,
            })
        query_results[case["id"]] = {
            "top_sources": [result["source"] for result in ranked],
            "top_chunk": (
                [ranked[0]["source"], ranked[0]["chunk_index"]] if ranked else None
            ),
            "top_score": ranked[0]["score"] if ranked else None,
            "expected_source_ranks": source_ranks,
        }

    answerable = [record for record in records if record["expected_sources"]]
    negatives = [record for record in records if not record["expected_sources"]]
    metrics = {}
    for cutoff in (1, 3, 5):
        metrics[f"hit@{cutoff}"] = (
            sum(
                all(rank is not None and rank <= cutoff for rank in record["ranks"])
                for record in answerable
            ) / len(answerable)
            if answerable else None
        )
    metrics["mrr"] = (
        sum(
            1 / min((rank for rank in record["ranks"] if rank is not None), default=math.inf)
            for record in answerable
        ) / len(answerable)
        if answerable else None
    )
    metrics[f"expected_source_coverage@{top_k}"] = (
        sum(
            sum(rank is not None and rank <= top_k for rank in record["ranks"])
            / len(record["expected_sources"])
            for record in answerable
        ) / len(answerable)
        if answerable else None
    )
    metrics["abstention_accuracy"] = (
        sum(record["abstained"] for record in negatives) / len(negatives)
        if negatives else None
    )

    failures = []
    for record in records:
        if record["expected_sources"]:
            failed = not all(
                rank is not None and rank <= top_k for rank in record["ranks"]
            )
        else:
            failed = not record["abstained"]
        if failed:
            failures.append({
                "id": record["id"],
                "category": record["category"],
                "expected_sources": record["expected_sources"],
                "expected_source_ranks": record["ranks"],
                "top_sources": record["results"],
            })

    return {
        "chunks": len(chunks),
        "sources": len(sources),
        "answerable_cases": len(answerable),
        "out_of_scope_cases": len(negatives),
        "metrics": metrics,
        "failure_examples": failures,
        "query_results": query_results,
    }


def evaluate() -> dict:
    cases = json.loads(Path("data/rag_golden_set.json").read_text(encoding="utf-8"))
    top_k = max(settings.rag_top_k, 5)
    strategies = {
        strategy: _evaluate_strategy(strategy, cases, top_k)
        for strategy in ("paragraph", "paragraph_blocks")
    }
    strategy_pairs = (
        ("paragraph", strategies["paragraph"]["query_results"]),
        ("paragraph_blocks", strategies["paragraph_blocks"]["query_results"]),
    )
    baseline, candidate = (pair[1] for pair in strategy_pairs)
    paired_ids = sorted(baseline.keys() & candidate.keys())
    top_score_deltas = [
        abs(baseline[item]["top_score"] - candidate[item]["top_score"])
        for item in paired_ids
        if baseline[item]["top_score"] is not None
        and candidate[item]["top_score"] is not None
    ]
    comparison = {
        "paired_cases": len(paired_ids),
        "same_top_source": sum(
            (baseline[item]["top_sources"] or [None])[0]
            == (candidate[item]["top_sources"] or [None])[0]
            for item in paired_ids
        ),
        "same_ordered_top_k_sources": sum(
            baseline[item]["top_sources"] == candidate[item]["top_sources"]
            for item in paired_ids
        ),
        "same_top_chunk": sum(
            baseline[item]["top_chunk"] == candidate[item]["top_chunk"]
            for item in paired_ids
        ),
        "changed_expected_source_ranks": sum(
            baseline[item]["expected_source_ranks"]
            != candidate[item]["expected_source_ranks"]
            for item in paired_ids
            if baseline[item]["expected_source_ranks"]
        ),
        "mean_absolute_top_score_delta": (
            sum(top_score_deltas) / len(top_score_deltas) if top_score_deltas else None
        ),
    }
    for result in strategies.values():
        del result["query_results"]
    return {
        "corpus_directory": settings.rag_document_dir,
        "golden_cases": len(cases),
        "top_k": top_k,
        "abstention_threshold": settings.rag_min_relevance_score,
        "strategies": strategies,
        "paired_comparison": comparison,
        "limitations": (
            "Compares section-packed text with extracted paragraph boundaries using "
            "the configured embedding model; "
            "not a held-out quality estimate. The active vector store is not modified."
        ),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("eval/chunking_comparison_2026-10-23.json"),
    )
    args = parser.parse_args()
    result = evaluate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
