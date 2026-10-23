"""Deterministic retrieval metrics over annotated evidence text."""
import random
import unicodedata


def normalize_evidence_text(value: str) -> str:
    value = unicodedata.normalize("NFC", value).casefold()
    return " ".join(value.split())


def _retrieved_context_runs(results: list[dict], k: int) -> list[str]:
    """Join adjacent retrieved chunks from the same source to handle split evidence."""
    grouped: dict[str, list[dict]] = {}
    for item in results[:k]:
        grouped.setdefault(item["source"], []).append(item)
    runs = []
    for source_items in grouped.values():
        ordered = sorted(source_items, key=lambda item: item.get("chunk_index", 0))
        current = []
        previous_index = None
        for item in ordered:
            index = item.get("chunk_index", 0)
            if current and index > previous_index + 1:
                runs.append("\n".join(part["content"] for part in current))
                current = []
            current.append(item)
            previous_index = index
        if current:
            runs.append("\n".join(part["content"] for part in current))
    return runs


def context_recall_at_k(case: dict, results: list[dict], k: int) -> float | None:
    """Fraction of evidence annotations present in top-k retrieved context.

    Evidence is matched within a chunk or a run of adjacent chunks from one source,
    allowing a passage that crosses an overlapping chunk boundary to be recovered.
    """
    evidence = case.get("evidence", [])
    if not evidence:
        return None
    runs_by_source: dict[str, list[str]] = {}
    selected = results[:k]
    for source in {item["source"] for item in selected}:
        runs_by_source[source] = _retrieved_context_runs(
            [item for item in selected if item["source"] == source], k
        )
    found = 0
    for item in evidence:
        needle = normalize_evidence_text(item["text"])
        source_runs = runs_by_source.get(item["source"], [])
        if any(needle in normalize_evidence_text(run) for run in source_runs):
            found += 1
    return found / len(evidence)


def mean_context_recall_at_k(cases_and_results: list[tuple[dict, list[dict]]], k: int) -> float | None:
    scores = [context_recall_at_k(case, results, k) for case, results in cases_and_results]
    scores = [score for score in scores if score is not None]
    return sum(scores) / len(scores) if scores else None


def paired_bootstrap_interval(
    baseline: list[float], candidate: list[float], *,
    samples: int = 10_000, confidence: float = 0.95, seed: int = 20261023,
) -> dict:
    """Percentile interval for candidate-minus-baseline on matched cases."""
    if len(baseline) != len(candidate) or not baseline:
        raise ValueError("Paired scores must have the same non-zero length")
    if samples < 1 or not 0 < confidence < 1:
        raise ValueError("samples must be positive and confidence must be between 0 and 1")
    differences = [right - left for left, right in zip(baseline, candidate)]
    estimate = sum(differences) / len(differences)
    rng = random.Random(seed)
    boot = sorted(
        sum(rng.choices(differences, k=len(differences))) / len(differences)
        for _ in range(samples)
    )
    tail = (1 - confidence) / 2
    lower_index = int(tail * (samples - 1))
    upper_index = int((1 - tail) * (samples - 1))
    return {
        "estimate": estimate,
        "confidence": confidence,
        "lower": boot[lower_index],
        "upper": boot[upper_index],
        "paired_cases": len(differences),
        "samples": samples,
        "seed": seed,
        "note": "Exploratory case-level paired interval; questions sharing documents may be dependent.",
    }
