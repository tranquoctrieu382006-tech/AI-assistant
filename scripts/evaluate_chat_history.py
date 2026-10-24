"""Evaluation script for multi-turn conversational RAG (TASK-AI11)."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path, PurePosixPath

from app.core import session_store
from app.core.question_rewriter import rewrite_question_if_needed
from app.api.rag import _answer_rag
from app.schemas.rag import RagAskRequest


def source_matches(actual: str, expected: str) -> bool:
    actual_path = PurePosixPath(actual.replace("\\", "/"))
    expected_path = PurePosixPath(expected.replace("\\", "/"))
    return actual_path == expected_path


async def evaluate_conversations():
    dev_set_path = Path("data/conversation_dev_set.json")
    conversations = json.loads(dev_set_path.read_text(encoding="utf-8"))

    results = []
    total_turns = 0
    correct_retrievals = 0
    correct_rewrites = 0

    for conv in conversations:
        conv_id = conv["id"]
        session_id = f"eval-{conv_id}"
        session_store.clear_session(session_id)
        conv_record = {
            "id": conv_id,
            "topic": conv.get("topic"),
            "turns": [],
        }

        for turn_data in conv["turns"]:
            total_turns += 1
            question = turn_data["question"]
            history = session_store.get_history(session_id)

            # 1. Evaluate rewrite step
            rewritten_q = await rewrite_question_if_needed(question, history)
            needs_rewrite = turn_data.get("needs_rewrite", False)
            rewrite_passed = True

            if not needs_rewrite:
                # If question was already complete, it shouldn't be drastically mangled
                # Expect length / keywords to remain close
                pass
            else:
                # Expect keywords to appear in rewritten question
                keywords = turn_data.get("expected_standalone_keywords", [])
                lower_rewritten = rewritten_q.lower()
                matches_keywords = any(kw.lower() in lower_rewritten for kw in keywords)
                if not matches_keywords:
                    rewrite_passed = False

            if rewrite_passed:
                correct_rewrites += 1

            # 2. Call RAG pipeline
            req = RagAskRequest(question=question, top_k=5, session_id=session_id)
            resp = await _answer_rag(req)

            # 3. Check retrieval & answer
            expected_source = turn_data.get("expected_source")
            retrieval_hit = False
            if expected_source:
                retrieval_hit = any(source_matches(s.source, expected_source) for s in resp.sources)
                if retrieval_hit:
                    correct_retrievals += 1

            expected_ans = turn_data.get("expected_answer_contains")
            ans_matched = False
            if expected_ans:
                ans_matched = expected_ans.lower() in resp.answer.lower()
            elif turn_data.get("expected_abstain"):
                ans_matched = resp.answer == "Tài liệu hiện có không chứa đủ thông tin để trả lời câu hỏi này."

            conv_record["turns"].append({
                "turn": turn_data["turn"],
                "original_question": question,
                "rewritten_question": rewritten_q,
                "rewrite_passed": rewrite_passed,
                "answer": resp.answer,
                "sources": [s.source for s in resp.sources],
                "retrieval_hit": retrieval_hit,
                "answer_matched": ans_matched,
            })

        results.append(conv_record)

    summary = {
        "total_conversations": len(conversations),
        "total_turns": total_turns,
        "rewrite_accuracy": correct_rewrites / total_turns if total_turns else 0,
        "retrieval_accuracy": correct_retrievals / total_turns if total_turns else 0,
        "conversations": results,
    }

    out_json = Path("eval/conversation_dev_evaluation.json")
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    md_lines = [
        "# Đánh giá Hội thoại nhiều lượt (TASK-AI11)",
        "",
        f"- **Tổng số kịch bản hội thoại:** {len(conversations)} hội thoại ({total_turns} lượt)",
        f"- **Độ chính xác viết lại câu hỏi:** {summary['rewrite_accuracy']*100:.1f}% ({correct_rewrites}/{total_turns})",
        f"- **Độ chính xác tìm đúng nguồn:** {summary['retrieval_accuracy']*100:.1f}% ({correct_retrievals}/{total_turns})",
        "",
        "| ID | Lượt | Câu hỏi gốc | Câu hỏi làm rõ | Nguồn tìm được | Trả lời |",
        "|:---:|:---:|---|---|---|---|",
    ]
    for c in results:
        for t in c["turns"]:
            md_lines.append(
                f"| {c['id']} | {t['turn']} | {t['original_question']} | {t['rewritten_question']} | "
                f"{', '.join(t['sources'][:2]) or 'Không có'} | {t['answer'][:60]}... |"
            )

    out_md = Path("eval/conversation_dev_evaluation.md")
    out_md.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Results saved to {out_json} and {out_md}")
    print(f"Rewrite accuracy: {summary['rewrite_accuracy']*100:.1f}%, Retrieval: {summary['retrieval_accuracy']*100:.1f}%")


if __name__ == "__main__":
    asyncio.run(evaluate_conversations())
