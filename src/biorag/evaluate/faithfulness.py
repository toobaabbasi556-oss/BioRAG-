"""CLI: generate answers for manual faithfulness scoring.

Usage:  PYTHONPATH=src python -m biorag.evaluate.faithfulness --n 20
Writes: results/faithfulness.md  (score the blank column by hand)

Uses the OFF-THE-SHELF retriever, not the fine-tuned one: the ablation in
results/retrieval.md showed fine-tuning lowered MRR (0.698 -> 0.595), so
serving the fine-tuned model would knowingly ship the worse system.

Resumable, because the free-tier daily quota is small enough that a run will
sometimes stop partway. Completed answers are cached to disk and reused.
"""

import argparse
import json
import time
from pathlib import Path

from biorag.generate.answer import answer_question
from biorag.index.dense import load_index
from biorag.ingest.build_corpus import load_corpus
from biorag.retrieve.dense import DenseRetriever
from biorag.train.pairs import load_questions

OUT_PATH = Path("results/faithfulness.md")
CACHE_PATH = Path("results/.faithfulness_cache.jsonl")
SLEEP = 6.0  # stay under the per-minute free-tier limit


def _load_cache() -> dict[str, dict]:
    if not CACHE_PATH.exists():
        return {}
    return {
        row["id"]: row for row in (json.loads(l) for l in CACHE_PATH.open())
    }


def main(n: int) -> None:
    chunks = load_corpus()
    retriever = DenseRetriever(chunks, index=load_index(Path("data/base.faiss")))

    cache = _load_cache()
    if cache:
        print(f"resuming: {len(cache)} answers already generated")

    # Deliberately stratified, not the first n: the unanswerable questions sit
    # at the end of the file, and a slice would never exercise the refusal path
    # — which is the failure mode most worth measuring.
    all_q = load_questions()
    unanswerable = [q for q in all_q if q["type"] == "unanswerable"]
    answerable = [q for q in all_q if q["type"] != "unanswerable"]
    questions = answerable[: max(n - len(unanswerable), 0)] + unanswerable
    for q in questions:
        if q["id"] in cache:
            continue
        result = answer_question(q["question"], retriever)
        row = {
            "id": q["id"],
            "type": q["type"],
            "question": q["question"],
            "answer": result.text,
            "retrieved": [c.pmid for c in result.chunks],
            "expected": q["relevant_pmids"],
        }
        with CACHE_PATH.open("a") as f:
            f.write(json.dumps(row) + "\n")
        cache[q["id"]] = row
        print(f"[{q['id']}] {result.text[:70]}", flush=True)
        time.sleep(SLEEP)

    sections = [
        "# Faithfulness review\n",
        "Answers generated with the off-the-shelf retriever (the fine-tuned "
        "model ranked worse — see retrieval.md).\n",
        "Score each as: grounded / partially grounded / hallucinated\n",
        "**Summary:** _fill in after scoring_\n",
    ]
    for q in questions:
        row = cache[q["id"]]
        sections.append(
            f"## {row['id']} ({row['type']})\n\n"
            f"**Q:** {row['question']}\n\n"
            f"**Answer:** {row['answer']}\n\n"
            f"**Retrieved:** {', '.join(row['retrieved']) or '(none)'}\n\n"
            f"**Expected:** {', '.join(row['expected']) or '(none — should refuse)'}\n\n"
            f"**Score:** \n"
        )

    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text("\n".join(sections))
    print(f"\nwrote {OUT_PATH} — fill in the Score lines by hand")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20)
    main(parser.parse_args().n)
