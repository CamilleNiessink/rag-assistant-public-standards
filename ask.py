"""Answer a question from the indexed documents, with citations."""
import sys
import time

import faiss

from common import (CHAT_MODEL, INDEX_FILE, MIN_SCORE, TOP_K, client, embed_texts,
                    load_meta)

SYSTEM_PROMPT = """You answer questions using ONLY the provided document excerpts.
Rules:
- If the excerpts do not contain the answer, say exactly: "I could not find this in the documents."
- Never add information that is not in the excerpts.
- Answer the specific question asked; do not substitute a related requirement.
- If several excerpts are relevant, combine them and cite each one.
- Note when a requirement applies only to a specific sector (e.g. health) rather than all entities.
- Cite every claim with the excerpt label in square brackets, e.g. [1] or [2][3].
- Be concise."""


def retrieve(question: str, k: int = TOP_K) -> list[dict]:
    index = faiss.read_index(str(INDEX_FILE))
    meta = load_meta()
    qvec = embed_texts([question])
    scores, ids = index.search(qvec, k * 4)
    hits, seen = [], set()
    for score, idx in zip(scores[0], ids[0]):
        if idx < 0:
            continue
        key = (meta[idx]["doc"], meta[idx]["page"])
        if key in seen:
            continue
        seen.add(key)
        hit = dict(meta[idx])
        hit["score"] = float(score)
        hits.append(hit)
        if len(hits) == k:
            break

        DOC_HINTS = {
            "wa-ai-policy.pdf": ["wa government ai policy", "government ai policy"],
            "wa-ai-health-policy.pdf": ["health ai policy", "wa health ai policy"],
            "wa-ai-assurance-framework.pdf": ["assurance framework"],
            "wa-health-ai-transparency-statement.pdf": ["transparency statement"],
            "wa-health-ai-digital-strategy-2020-2030.pdf": ["digital strategy"],
        }
        q = question.lower()
        for h in hits:
            if any(hint in q for hint in DOC_HINTS.get(h["doc"], [])):
                h["score"] += 0.10
        hits.sort(key=lambda h: h["score"], reverse=True)
    return hits


def answer(question: str, k: int = TOP_K) -> dict:
    t0 = time.time()
    hits = retrieve(question, k)
    if not hits or hits[0]["score"] < MIN_SCORE:
        return {"answer": "I could not find this in the documents.", "sources": hits,
                "seconds": time.time() - t0}

    context = "\n\n".join(
        f"[{i}] ({h['doc']}, p.{h['page']})\n{h['text']}" for i, h in enumerate(hits, start=1)
    )
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Excerpts:\n\n{context}\n\nQuestion: {question}"},
        ],
    )
    return {"answer": resp.choices[0].message.content.strip(), "sources": hits,
            "seconds": time.time() - t0}


def main():
    if len(sys.argv) < 2:
        raise SystemExit('Usage: python ask.py "your question"')
    question = " ".join(sys.argv[1:])
    result = answer(question)
    print("\n" + result["answer"] + "\n")
    print("Sources:")
    for i, h in enumerate(result["sources"], start=1):
        print(f"  [{i}] {h['doc']} p.{h['page']}  (score {h['score']:.2f})")
    print(f"\n{result['seconds']:.1f}s")


if __name__ == "__main__":
    main()
