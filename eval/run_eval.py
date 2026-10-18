"""Score retrieval hit rate, answer correctness (LLM-graded) and latency on eval/questions.json."""
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import CHAT_MODEL, client  # noqa: E402
from ask import answer  # noqa: E402

QUESTIONS = Path(__file__).parent / "questions.json"
RESULTS = Path(__file__).parent / "results.json"

GRADER_PROMPT = """You are grading a question-answering system.
Question: {question}
Reference answer: {reference}
System answer: {candidate}

Does the system answer convey the same essential information as the reference answer,
without contradicting it or adding unsupported claims? Reply with exactly one word: CORRECT or INCORRECT."""


def grade(question: str, reference: str, candidate: str) -> bool:
    resp = client.chat.completions.create(
        model=CHAT_MODEL, temperature=0,
        messages=[{"role": "user", "content": GRADER_PROMPT.format(
            question=question, reference=reference, candidate=candidate)}],
    )
    return resp.choices[0].message.content.strip().upper().startswith("CORRECT")


def main():
    items = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    results, hits, correct, times = [], 0, 0, []

    for item in items:
        out = answer(item["question"])
        times.append(out["seconds"])

        if item["expected_doc"] is None:
            hit = out["answer"].startswith("I could not find")
        else:
            hit = any(s["doc"] == item["expected_doc"] and s["page"] in item["expected_pages"]
                      for s in out["sources"])
        hits += hit

        ok = grade(item["question"], item["reference_answer"], out["answer"])
        correct += ok

        results.append({**item, "answer": out["answer"], "retrieval_hit": hit, "correct": ok,
                        "seconds": round(out["seconds"], 2)})
        print(f"{'HIT ' if hit else 'MISS'} {'OK ' if ok else 'BAD'}  {item['question'][:70]}")

    n = len(items)
    summary = {
        "n": n,
        "retrieval_hit_rate": round(hits / n, 3),
        "answer_correctness": round(correct / n, 3),
        "median_seconds": round(statistics.median(times), 2),
    }
    RESULTS.write_text(json.dumps({"summary": summary, "results": results}, indent=1),
                       encoding="utf-8")
    print("\n" + json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
