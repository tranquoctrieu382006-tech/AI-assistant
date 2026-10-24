"""Compare one conservative abstention-prompt variant on the RAG dev set only."""
import json
from pathlib import Path
import random
import time
import unicodedata
import argparse

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.core.config import settings
from app.core.llm import get_llm
from app.core.rag_prompt import NO_ANSWER, build_rag_chain, build_rag_prompt, format_context
from app.schemas.rag import RagGeneratedAnswer

ROOT = Path(__file__).resolve().parents[1]
CASES_PATH = ROOT / "data/rag_golden_set.json"
OUTPUT_PATH = ROOT / "eval/dev_abstention_prompt_experiment_m1.json"
FOLLOWUP_OUTPUT_PATH = ROOT / "eval/dev_abstention_prompt_missing_followup_m1.json"
ANSWERABLE_FOLLOWUP_OUTPUT_PATH = ROOT / "eval/dev_abstention_prompt_answerable_followup_m1.json"
CANDIDATE_RULE = (
    "Trước khi trả lời, hãy kiểm tra riêng từng thông tin mà câu hỏi yêu cầu. "
    "Chỉ trả lời nếu CONTEXT nêu rõ và hỗ trợ tất cả các thông tin thiết yếu đó. "
    "Nếu thiếu một con số, mốc thời gian, đối tượng, điều kiện hoặc bước thực hiện "
    "được hỏi, hãy từ chối; không suy ra từ thông tin cùng chủ đề."
)


def normalize(value: str) -> str:
    return " ".join(unicodedata.normalize("NFC", value).casefold().split())


def make_candidate_chain(llm):
    base_prompt = build_rag_prompt()
    parser = PydanticOutputParser(pydantic_object=RagGeneratedAnswer)
    system_template = base_prompt.messages[0].prompt.template
    candidate_prompt = ChatPromptTemplate.from_messages([
        ("system", system_template + "\n\n" + CANDIDATE_RULE),
        base_prompt.messages[1],
    ]).partial(format_instructions=parser.get_format_instructions())
    return candidate_prompt | llm | parser


def select_cases(cases: list[dict], counts: dict[str, int], seed: int) -> list[dict]:
    rng = random.Random(seed)
    selected_ids = set()
    for category, count in counts.items():
        members = [case for case in cases if case.get("category", "simple") == category]
        rng.shuffle(members)
        selected_ids.update(case["id"] for case in members[:count])
    return [case for case in cases if case["id"] in selected_ids]


def run(*, sample_counts: dict[str, int] | None = None, seed: int = 20261023,
        case_ids: set[str] | None = None, output_path: Path = OUTPUT_PATH) -> dict:
    from app.core.vector_store import close_pool, similarity_search

    full_cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    if case_ids is not None:
        cases = [case for case in full_cases if case["id"] in case_ids]
    else:
        cases = select_cases(full_cases, sample_counts, seed) if sample_counts else full_cases
    llm = get_llm(settings.rag_model, temperature=0)
    baseline_chain = build_rag_chain(llm)
    candidate_chain = make_candidate_chain(llm)
    records = []
    try:
        for index, case in enumerate(cases):
            retrieved = similarity_search(case["question"], k=settings.rag_top_k)
            context_matches = [
                match for match in retrieved
                if match["score"] >= settings.rag_min_relevance_score
            ]
            context = format_context(context_matches)
            outputs = {}
            for variant, chain in (("baseline", baseline_chain), ("candidate", candidate_chain)):
                started = time.perf_counter()
                if not context:
                    answer, abstain, status = NO_ANSWER, True, "no_context"
                else:
                    try:
                        result = chain.invoke({"question": case["question"], "context": context})
                        answer, abstain, status = result.answer.strip(), result.abstain, "ok"
                    except Exception as exc:
                        answer, abstain, status = "", None, type(exc).__name__
                outputs[variant] = {
                    "answer": answer,
                    "abstain": abstain,
                    "status": status,
                    "latency_ms": round((time.perf_counter() - started) * 1000, 1),
                }

            parts = case.get("answer_contains", [])
            if isinstance(parts, str):
                parts = [parts]
            answerable = bool(case.get("evidence"))
            expected_abstain = not answerable
            for output in outputs.values():
                output["abstention_correct"] = (
                    output["abstain"] is not None
                    and output["abstain"] == expected_abstain
                )
                output["answer_contains_normalized"] = (
                    all(normalize(part) in normalize(output["answer"]) for part in parts)
                    if answerable and output["status"] == "ok" else None
                )
            records.append({
                "id": case["id"],
                "category": case.get("category", "simple"),
                "question": case["question"],
                "answerable": answerable,
                "top_score": retrieved[0]["score"] if retrieved else None,
                "context_chunk_count": len(context_matches),
                "variants": outputs,
            })
            print(f"{case['id']} {case.get('category', 'simple')}: "
                  f"base={outputs['baseline']['abstain']}/{outputs['baseline']['status']} "
                  f"candidate={outputs['candidate']['abstain']}/{outputs['candidate']['status']}")
    finally:
        close_pool()

    summary = {}
    for variant in ("baseline", "candidate"):
        variant_records = [r["variants"][variant] for r in records]
        answerable_records = [r["variants"][variant] for r in records if r["answerable"]]
        missing_records = [r["variants"][variant] for r in records if r["category"] == "in_scope_missing"]
        summary[variant] = {
            "abstention_correct": sum(r["abstention_correct"] is True for r in variant_records),
            "answerable_false_abstentions": sum(r["abstain"] is True for r in answerable_records),
            "answerable_expected_fragment_match": sum(r["answer_contains_normalized"] is True for r in answerable_records),
            "answerable_cases": len(answerable_records),
            "in_scope_missing_false_answers": sum(r["abstain"] is False for r in missing_records),
            "in_scope_missing_cases": len(missing_records),
            "generation_errors": sum(r["status"] not in {"ok", "no_context"} for r in variant_records),
        }
    output = {
        "dataset": "development M1 only",
        "case_count": len(records),
        "full_development_case_count": len(full_cases),
        "sample_counts": sample_counts,
        "sample_seed": seed if sample_counts else None,
        "model": settings.local_model if settings.rag_model == "local" else settings.rag_model,
        "retrieval": {
            "top_k": settings.rag_top_k,
            "cutoff": settings.rag_min_relevance_score,
            "embedding_model": settings.embedding_model,
        },
        "candidate_rule": CANDIDATE_RULE,
        "summary": summary,
        "paired_case_changes": {
            "candidate_fixes_missing_false_answers": sum(
                r["variants"]["baseline"]["abstain"] is False
                and r["variants"]["candidate"]["abstain"] is True
                for r in records if r["category"] == "in_scope_missing"
            ),
            "candidate_adds_answerable_false_abstentions": sum(
                r["variants"]["baseline"]["abstain"] is False
                and r["variants"]["candidate"]["abstain"] is True
                for r in records if r["answerable"]
            ),
        },
        "interpretation_limit": (
            "Development-only prompt experiment, exploratory; one candidate, synthetic corpus, "
            "expected-fragment matching is not semantic answer evaluation. No production config changed."
        ),
        "records": records,
    }
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output_path), "summary": summary,
                      "paired_case_changes": output["paired_case_changes"]},
                     ensure_ascii=False, indent=2))
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", action="store_true",
                        help="Use the predefined stratified development sample (24 cases).")
    parser.add_argument("--missing-followup", action="store_true",
                        help="Run in_scope_missing cases not present in the first sample.")
    parser.add_argument("--answerable-followup", action="store_true",
                        help="Run answerable cases not present in the first sample.")
    args = parser.parse_args()
    selected_modes = sum((args.sample, args.missing_followup, args.answerable_followup))
    if selected_modes > 1:
        parser.error("Choose only one of --sample, --missing-followup, or --answerable-followup")
    if args.missing_followup:
        prior = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
        previously_run = {record["id"] for record in prior["records"]}
        all_cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
        missing_ids = {
            case["id"] for case in all_cases
            if case.get("category") == "in_scope_missing" and case["id"] not in previously_run
        }
        run(case_ids=missing_ids, output_path=FOLLOWUP_OUTPUT_PATH)
        raise SystemExit(0)
    if args.answerable_followup:
        prior = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
        previously_run = {record["id"] for record in prior["records"]}
        all_cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
        answerable_ids = {
            case["id"] for case in all_cases
            if case.get("evidence") and case["id"] not in previously_run
        }
        run(case_ids=answerable_ids, output_path=ANSWERABLE_FOLLOWUP_OUTPUT_PATH)
        raise SystemExit(0)
    run(sample_counts=(
        {"in_scope_missing": 8, "simple": 8, "paraphrase": 4,
         "multi_document": 2, "out_of_scope_far": 2}
        if args.sample else None
    ))
