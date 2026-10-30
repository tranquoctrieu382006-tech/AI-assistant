"""Evaluate RAG retrieval/answer variants V0-V4 for unaccented questions.

Variants (same corpus, same gate, same top_k):

- V0: original accented question (baseline).
- V1: question with accents stripped, no processing.
- V2: V1 + LLM diacritic restoration.
- V3: V2 + dual retrieval, only when the rewrite actually restored accents and
      did not resolve coreference.
- V4: V3 with hybrid retrieval (vector + unaccented BM25).

Retrieval metrics are cheap and deterministic, so they can be repeated at will.
End-to-end answers (`--answers`) call the generation LLM and are cached per
variant/case in the output artifact; use `--reuse-answers` to merge a previous
artifact instead of paying for the same calls twice.

Examples:
    python -m scripts.evaluate_rag_variants --variants V0,V1 --repeats 3
    python -m scripts.evaluate_rag_variants --variants V0,V1,V2,V3,V4 --answers
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import sys
import time
import unicodedata
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from app.core.config import settings  # noqa: E402
from app.core.text_utils import strip_accents  # noqa: E402

GOLDEN_PATH = ROOT / "data" / "rag_golden_set.json"
DEFAULT_OUTPUT = ROOT / "eval" / "rag_variants_dev.json"
REWRITE_CACHE_DIR = ROOT / "eval"
ABSTAIN_CATEGORIES = frozenset({"in_scope_missing", "out_of_scope_far"})
VARIANTS = ("V0", "V1", "V2", "V3", "V4")


def normalize_answer(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text or "").casefold().split())


def unaccented_question(question: str) -> str:
    """V1 input: keep punctuation/case shape but remove Vietnamese diacritics."""
    return strip_accents(question)


# ---------------------------------------------------------------------------
# Rewrite cache
# ---------------------------------------------------------------------------

def _prompt_version() -> str:
    from app.core import question_rewriter

    prompt = getattr(question_rewriter, "REWRITE_SYSTEM_PROMPT", "")
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]


def rewrite_cache_path(model_role: str) -> Path:
    return REWRITE_CACHE_DIR / f"rewrite_cache_{model_role}_{_prompt_version()}.json"


def load_rewrite_cache(model_role: str) -> dict:
    path = rewrite_cache_path(model_role)
    if not path.exists():
        return {"model_role": model_role, "prompt_version": _prompt_version(), "entries": {}}
    data = json.loads(path.read_text(encoding="utf-8"))
    data.setdefault("entries", {})
    return data


def save_rewrite_cache(model_role: str, cache: dict) -> Path:
    path = rewrite_cache_path(model_role)
    path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def rewrite_with_cache(
    case: dict,
    *,
    model_role: str,
    cache: dict,
    pacing_seconds: float,
) -> dict:
    """Rewrite the unaccented variant of one case and cache the result."""
    raw = unaccented_question(case["question"])
    entry = cache["entries"].get(case["id"])
    if entry and entry.get("input") == raw:
        return entry

    from app.core.question_rewriter import rewrite_question

    if pacing_seconds > 0:
        time.sleep(pacing_seconds)
    started = time.perf_counter()
    result = asyncio.run(
        rewrite_question(raw, history=[], model_role=model_role)
    )
    elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
    rewritten = result.text
    entry = {
        "input": raw,
        "output": rewritten,
        "accent_restored": result.accent_restored,
        "coref_resolved": result.coref_resolved,
        "latency_ms": elapsed_ms,
        "failed": result.failed,
    }
    cache["entries"][case["id"]] = entry
    save_rewrite_cache(model_role, cache)
    return entry


# ---------------------------------------------------------------------------
# Retrieval and metrics
# ---------------------------------------------------------------------------

def source_matches(actual: str, expected: str) -> bool:
    return actual.replace("\\", "/") == expected.replace("\\", "/")


def expected_sources(case: dict) -> list[str]:
    sources = case.get("expected_sources")
    if sources is not None:
        return sources
    return [case["source"]] if case.get("source") else []


def retrieval_metrics(cases: list[dict], per_case: dict[str, dict]) -> dict:
    answerable = [c for c in cases if expected_sources(c)]
    negatives = [c for c in cases if not expected_sources(c)]

    metrics: dict = {"answerable_cases": len(answerable)}
    for cutoff in (1, 3, 5):
        hits = 0
        for case in answerable:
            record = per_case[case["id"]]
            sources = [item["source"] for item in record["results"][:cutoff]]
            if all(any(source_matches(s, exp) for s in sources) for exp in expected_sources(case)):
                hits += 1
        metrics[f"hit@{cutoff}"] = hits / len(answerable) if answerable else None

    reciprocal_ranks = []
    for case in answerable:
        ranks = per_case[case["id"]].get("ranks") or []
        reciprocal_ranks.append(1 / min(ranks) if ranks else 0)
    metrics["mrr"] = sum(reciprocal_ranks) / len(answerable) if answerable else None

    coverage = 0.0
    for case in answerable:
        sources = [item["source"] for item in per_case[case["id"]]["results"][:5]]
        expected = expected_sources(case)
        coverage += sum(any(source_matches(s, exp) for s in sources) for exp in expected) / len(expected)
    metrics["expected_source_coverage@5"] = coverage / len(answerable) if answerable else None

    metrics["answerable_retrieval_false_refusals"] = sum(
        not per_case[c["id"]]["retrieval_passed"] for c in answerable
    )
    metrics["negative_cases"] = len(negatives)
    metrics["negative_gate_leaks"] = sum(
        per_case[c["id"]]["retrieval_passed"] for c in negatives
    )
    metrics["negative_gate_leak_ids"] = [
        c["id"] for c in negatives if per_case[c["id"]]["retrieval_passed"]
    ]
    return metrics


def answer_metrics(cases: list[dict], per_case: dict[str, dict]) -> dict:
    answerable = [c for c in cases if expected_sources(c)]
    negatives = [c for c in cases if not expected_sources(c)]

    def usable(records):
        return [r for r in records if not r.get("errored")]

    ans_records = usable([per_case[c["id"]] for c in answerable])
    neg_records = usable([per_case[c["id"]] for c in negatives])
    answered = sum(not r["final_abstain"] for r in ans_records)
    fragments = sum(r.get("fragment_match") is True for r in ans_records)
    false_abstentions = sum(r["final_abstain"] for r in ans_records)
    hallucinations = sum(not r["final_abstain"] for r in neg_records)

    result = {
        "answerable_total": len(answerable),
        "answerable_decided": len(ans_records),
        "answerable_answered": answered,
        "answerable_fragment_matches": fragments,
        "answerable_false_abstentions": false_abstentions,
        "negative_total": len(negatives),
        "negative_decided": len(neg_records),
        "negative_hallucinated_answers": hallucinations,
        "negative_false_refusals": sum(r["final_abstain"] for r in neg_records),
    }
    decided = len(ans_records) + len(neg_records)
    correct = fragments + sum(r["final_abstain"] for r in neg_records)
    result["decided_cases"] = decided
    result["correct_cases"] = correct
    result["abstention_accuracy"] = round(correct / decided, 3) if decided else None
    return result


def latency_summary(values: list[float | None]) -> dict:
    clean = sorted(v for v in values if v is not None)
    if not clean:
        return {"n": 0, "p50_ms": None, "p95_ms": None}
    def percentile(p: float) -> float:
        index = min(len(clean) - 1, max(0, math.ceil(p * len(clean)) - 1))
        return clean[index]
    return {
        "n": len(clean),
        "p50_ms": round(median(clean), 1),
        "p95_ms": round(percentile(0.95), 1),
    }


# ---------------------------------------------------------------------------
# Variant execution
# ---------------------------------------------------------------------------

def variant_plan(variant: str, case: dict, rewrite: dict | None) -> tuple[str, str | None, str]:
    """Return (retrieval_query, raw_query_for_dual, retrieval_mode)."""
    original = case["question"]
    if variant == "V0":
        return original, None, settings.search_mode
    if variant == "V1":
        return unaccented_question(original), None, settings.search_mode
    if rewrite is None:
        raise ValueError(f"{variant} requires a rewrite entry")
    restored = rewrite["output"]
    if variant == "V2":
        return restored, None, settings.search_mode
    dual_raw = None
    if rewrite["accent_restored"] and not rewrite["coref_resolved"]:
        dual_raw = rewrite["input"]
    mode = "hybrid" if variant == "V4" else settings.search_mode
    return restored, dual_raw, mode


def run_variant(
    variant: str,
    cases: list[dict],
    *,
    rewrites: dict[str, dict],
    answers: bool,
    answer_chain=None,
    pacing_seconds: float = 0.0,
    previous: dict[str, dict] | None = None,
) -> dict:
    from app.core.rag_prompt import NO_ANSWER, format_context
    from app.core.retriever import retrieve_and_gate

    previous = previous or {}
    per_case: dict[str, dict] = {}
    llm_calls = 0

    for case in cases:
        cid = case["id"]
        rewrite = rewrites.get(cid)
        query, dual_raw, mode = variant_plan(variant, case, rewrite)

        started = time.perf_counter()
        matches, _abstained_at_gate = retrieve_and_gate(
            query,
            top_k=settings.rag_top_k,
            mode=mode,
            use_reranker=settings.use_reranker,
            corpus="demo",
            raw_query=dual_raw,
        )
        retrieval_ms = round((time.perf_counter() - started) * 1000, 1)

        gated = [m for m in matches if m["score"] >= settings.rag_min_relevance_score]
        retrieval_passed = bool(gated)
        ranks = [
            next((i for i, item in enumerate(matches, 1)
                  if source_matches(item["source"], exp)), None)
            for exp in expected_sources(case)
        ]
        record = {
            "id": cid,
            "category": case.get("category", "simple"),
            "query": query,
            "dual_query": dual_raw,
            "retrieval_mode": mode,
            "rewrite": rewrite,
            "top_score": matches[0]["score"] if matches else None,
            "retrieval_passed": retrieval_passed,
            "retrieval_ms": retrieval_ms,
            "results": [
                {
                    "source": m["source"],
                    "chunk_index": m["chunk_index"],
                    "score": m["score"],
                }
                for m in matches
            ],
            "ranks": [r for r in ranks if r is not None],
            "hit": bool(ranks) and all(r is not None for r in ranks),
        }

        if answers:
            prior = previous.get((variant, cid)) or {}
            can_reuse = (
                prior.get("answer") is not None
                and prior.get("query") == query
                and prior.get("dual_query") == dual_raw
                and prior.get("retrieval_mode") == mode
                and not prior.get("errored")
            )
            if can_reuse:
                record.update({
                    "answer": prior["answer"],
                    "model_abstain": prior.get("model_abstain"),
                    "final_abstain": prior.get("final_abstain"),
                    "fragment_match": prior.get("fragment_match"),
                    "errored": prior.get("errored", False),
                    "generation_ms": prior.get("generation_ms"),
                    "reused_answer": True,
                })
            elif not retrieval_passed:
                record.update({
                    "answer": NO_ANSWER,
                    "model_abstain": True,
                    "final_abstain": True,
                    "fragment_match": None,
                    "errored": False,
                    "generation_ms": None,
                    "reused_answer": False,
                })
            else:
                context = format_context(gated)
                if pacing_seconds > 0:
                    time.sleep(pacing_seconds)
                llm_calls += 1
                gen_started = time.perf_counter()
                try:
                    result = answer_chain.invoke({"question": query, "context": context})
                    raw_answer = (result.answer or "").strip()
                    model_abstain = bool(result.abstain)
                    answer = raw_answer
                except Exception as exc:  # pragma: no cover - network failures
                    raw_answer = f"ERROR: {type(exc).__name__}: {exc}"
                    model_abstain = None
                    answer = raw_answer
                generation_ms = round((time.perf_counter() - gen_started) * 1000, 1)
                final_abstain = bool(model_abstain)
                expected_abstain = case.get("category") in ABSTAIN_CATEGORIES
                parts = case.get("answer_contains", [])
                if isinstance(parts, str):
                    parts = [parts]
                fragment_match = (
                    all(normalize_answer(p) in normalize_answer(answer) for p in parts)
                    if (not expected_abstain and not final_abstain and not raw_answer.startswith("ERROR:"))
                    else None
                )
                record.update({
                    "answer": answer,
                    "model_abstain": model_abstain,
                    "final_abstain": final_abstain,
                    "fragment_match": fragment_match,
                    "errored": model_abstain is None,
                    "generation_ms": generation_ms,
                    "reused_answer": False,
                })

        per_case[cid] = record

    retrieval = retrieval_metrics(cases, per_case)
    output = {
        "variant": variant,
        "metrics": retrieval,
        "retrieval_latency": latency_summary([r["retrieval_ms"] for r in per_case.values()]),
        "llm_answer_calls": llm_calls,
        "per_case": per_case,
    }
    if answers:
        output["answer_metrics"] = answer_metrics(cases, per_case)
        output["generation_latency"] = latency_summary(
            [r.get("generation_ms") for r in per_case.values()]
        )
    return output


def pairwise(left: dict, right: dict, cases: list[dict]) -> dict:
    changed = []
    for case in cases:
        cid = case["id"]
        a = left["per_case"][cid]
        b = right["per_case"][cid]
        if expected_sources(case):
            a_ok, b_ok = a["hit"], b["hit"]
        else:
            # For expected-abstain cases, passing retrieval is the failure mode.
            a_ok, b_ok = not a["retrieval_passed"], not b["retrieval_passed"]
        if a_ok != b_ok:
            changed.append({
                "id": cid,
                "category": case.get("category", "simple"),
                "from": a_ok,
                "to": b_ok,
                "question": case["question"],
            })
    improved = sum(1 for item in changed if item["to"])
    return {
        "definition": "good outcome: answerable case hit, or expected-abstain case blocked by the gate",
        "changed_cases": len(changed),
        "improved": improved,
        "regressed": len(changed) - improved,
        "changes": changed,
    }


def noise_floor(runs: list[dict], cases: list[dict]) -> dict:
    if len(runs) < 2:
        return {"note": "Cần ít nhất 2 lần chạy để đo sàn nhiễu.", "unstable_case_ids": []}
    unstable = set()
    for case in cases:
        cid = case["id"]
        outcomes = {
            (run["per_case"][cid]["hit"] if expected_sources(case)
             else run["per_case"][cid]["retrieval_passed"])
            for run in runs
        }
        if len(outcomes) > 1:
            unstable.add(cid)
    return {
        "repeats": len(runs),
        "unstable_cases": len(unstable),
        "unstable_case_ids": sorted(unstable),
        "max_case_delta_between_runs": len(unstable),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", default="V0,V1", help="Comma-separated subset of V0..V4.")
    parser.add_argument("--repeats", type=int, default=1, help="Runs per variant (noise floor).")
    parser.add_argument("--answers", action="store_true", help="Call the generation LLM too.")
    parser.add_argument("--reuse-answers", type=Path, help="Merge cached answers from this artifact.")
    parser.add_argument("--model", default=settings.question_rewriter_model, help="Rewrite/model role.")
    parser.add_argument("--pacing", type=float, default=4.2, help="Seconds between LLM calls.")
    parser.add_argument("--limit", type=int, default=0, help="Only run the first N cases (smoke test).")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    variants = [v.strip().upper() for v in args.variants.split(",") if v.strip()]
    unknown = [v for v in variants if v not in VARIANTS]
    if unknown:
        parser.error(f"Unknown variants: {unknown}")

    cases = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    if args.limit > 0:
        cases = cases[: args.limit]
    cache = load_rewrite_cache(args.model)

    rewrites: dict[str, dict] = {}
    if any(v in {"V2", "V3", "V4"} for v in variants):
        print(f"Rewriting {len(cases)} unaccented questions with role={args.model}...")
        for index, case in enumerate(cases, 1):
            entry = rewrite_with_cache(
                case,
                model_role=args.model,
                cache=cache,
                pacing_seconds=args.pacing if case["id"] not in cache["entries"] else 0,
            )
            rewrites[case["id"]] = entry
            flags = f"accent={entry['accent_restored']} coref={entry['coref_resolved']}"
            print(f"  [{index:02d}/{len(cases)}] {case['id']}: {flags}")

    previous_answers: dict[tuple[str, str], dict] = {}
    if args.reuse_answers and args.reuse_answers.exists():
        reused = json.loads(args.reuse_answers.read_text(encoding="utf-8"))
        for variant_name, variant_data in reused.get("variants", {}).items():
            for cid, record in variant_data.get("per_case", {}).items():
                if record.get("answer") is not None and record.get("final_abstain") is not None:
                    previous_answers[(variant_name, cid)] = record

    answer_chain = None
    if args.answers:
        from app.core.llm import get_llm
        from app.core.rag_prompt import build_rag_chain

        answer_chain = build_rag_chain(get_llm(args.model, temperature=0))

    output = {
        "golden_set": str(GOLDEN_PATH.relative_to(ROOT)),
        "model_role": args.model,
        "config": {
            "search_mode": settings.search_mode,
            "top_k": settings.rag_top_k,
            "gate": settings.rag_min_relevance_score,
            "use_reranker": settings.use_reranker,
            "corpus": "demo",
        },
        "repeats": args.repeats,
        "variants": {},
        "pairwise": {},
    }

    try:
        for variant in variants:
            runs = []
            best = None
            for repeat in range(args.repeats):
                print(f"Running {variant} repeat {repeat + 1}/{args.repeats}...")
                run = run_variant(
                    variant,
                    cases,
                    rewrites=rewrites,
                    answers=args.answers,
                    answer_chain=answer_chain,
                    pacing_seconds=args.pacing,
                    previous=previous_answers,
                )
                runs.append(run)
                if best is None:
                    best = run
                print(f"  hit@5={run['metrics']['hit@5']} "
                      f"neg_leaks={run['metrics']['negative_gate_leaks']} "
                      f"llm_calls={run['llm_answer_calls']}")
            best = dict(best)
            best["noise_floor"] = noise_floor(runs, cases)
            if args.repeats > 1:
                best["runs"] = [
                    {
                        "metrics": run["metrics"],
                        "answer_metrics": run.get("answer_metrics"),
                        "llm_answer_calls": run["llm_answer_calls"],
                    }
                    for run in runs
                ]
            output["variants"][variant] = best
            if variant == "V0" and len(runs) > 1:
                output["noise_floor_V0"] = best["noise_floor"]

        if "V0" in output["variants"]:
            for variant in variants:
                if variant == "V0":
                    continue
                output["pairwise"][f"V0_vs_{variant}"] = pairwise(
                    output["variants"]["V0"], output["variants"][variant], cases
                )
    finally:
        from app.core.vector_store import close_pool

        close_pool()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved {args.output}")
    for variant, data in output["variants"].items():
        metrics = data["metrics"]
        print(f"{variant}: hit@1={metrics['hit@1']} hit@5={metrics['hit@5']} "
              f"mrr={metrics['mrr']} false_refusals={metrics['answerable_retrieval_false_refusals']} "
              f"gate_leaks={metrics['negative_gate_leaks']}/{metrics['negative_cases']}")
        if "answer_metrics" in data:
            answers = data["answer_metrics"]
            print(f"   answered={answers['answerable_answered']}/{answers['answerable_total']} "
                  f"fragments={answers['answerable_fragment_matches']} "
                  f"hallucinations={answers['negative_hallucinated_answers']}")


if __name__ == "__main__":
    main()
