"""Drafting aid for the evaluation set. NOT an automatic labeler.

Generates candidate questions from corpus abstracts so a human has something
concrete to review, edit and approve. The committed `eval/questions.jsonl` is
only trustworthy after that review — an unreviewed eval set measures nothing.

Design choices that matter for validity:
- The model is told to PARAPHRASE, never to reuse abstract wording. Questions
  that echo the abstract's phrasing hand the win to BM25 and make the whole
  retriever comparison meaningless.
- `multi` questions are drawn from pairs of abstracts that are near neighbours
  in embedding space, so a genuinely shared answer is plausible.
- `unanswerable` questions are hardcoded, not generated: they must be reliably
  outside the corpus, which is easier to guarantee by hand than to prompt for.

Usage:  PYTHONPATH=src python eval/draft_questions.py
"""

import json
import random
import time
from pathlib import Path

from biorag.generate.llm import complete
from biorag.index.dense import load_index
from biorag.retrieve.dense import DenseRetriever
from biorag.ingest.build_corpus import load_corpus

OUT_PATH = Path("eval/questions.draft.jsonl")
N_SINGLE = 25
N_MULTI = 10
SLEEP = 6.0  # pace under the per-minute limit; daily quota is the real ceiling

SINGLE_PROMPT = """Read this biomedical abstract and write ONE question that it answers.

Rules:
- Phrase it the way a clinician or graduate student would ask, NOT using the
  abstract's own wording. Do not copy distinctive phrases from the abstract.
- It must be answerable from this abstract alone.
- Output ONLY the question text, nothing else.

Title: {title}
Abstract: {text}"""

MULTI_PROMPT = """Read these TWO biomedical abstracts and write ONE question that
requires BOTH of them to answer well.

Rules:
- Phrase it naturally, NOT using either abstract's wording.
- It must genuinely need information from both, not just one.
- Output ONLY the question text, nothing else.

Abstract A - {title_a}
{text_a}

Abstract B - {title_b}
{text_b}"""

# Plausible biomedical questions with no answer in a biomechanics corpus.
UNANSWERABLE = [
    "What is the recommended first-line antibiotic for uncomplicated cystitis?",
    "How does metformin lower blood glucose at the cellular level?",
    "What are the diagnostic criteria for diabetic retinopathy?",
    "Which vaccines are recommended during the first year of life?",
    "What is the mechanism of action of SSRIs in treating depression?",
]


def _load_done() -> tuple[list[dict], set[str]]:
    """Resume support: the free-tier daily quota is small enough that a run
    WILL be interrupted. Each row is flushed as it is produced, so a quota
    failure costs one question rather than the whole run."""
    if not OUT_PATH.exists():
        return [], set()
    rows = [json.loads(line) for line in OUT_PATH.open()]
    keys = {"|".join(r["relevant_pmids"]) for r in rows if r["relevant_pmids"]}
    print(f"resuming: {len(rows)} rows already drafted")
    return rows, keys


def _append(row: dict) -> None:
    with OUT_PATH.open("a") as f:
        f.write(json.dumps(row) + "\n")


def main() -> None:
    chunks = load_corpus()
    # One representative chunk per abstract; questions are labeled by PMID.
    by_pmid = {}
    for chunk in chunks:
        by_pmid.setdefault(chunk.pmid, chunk)
    pmids = sorted(by_pmid)

    random.seed(0)  # reproducible sample
    picked = random.sample(pmids, N_SINGLE + N_MULTI * 2)

    rows, done = _load_done()

    for i, pmid in enumerate(picked[:N_SINGLE]):
        if pmid in done:
            continue
        c = by_pmid[pmid]
        question = complete(
            SINGLE_PROMPT.format(title=c.title, text=c.text[:2000])
        ).strip()
        row = {
            "id": f"q{len(rows) + 1:03d}",
            "question": question,
            "relevant_pmids": [pmid],
            "type": "single",
        }
        rows.append(row)
        _append(row)
        print(f"[single {i + 1}/{N_SINGLE}] {question[:80]}", flush=True)
        time.sleep(SLEEP)

    # Pair each remaining abstract with its nearest neighbour in embedding
    # space, so a shared answer is plausible rather than contrived. Pairing
    # random abstracts would produce questions no retriever could satisfy,
    # which would measure nothing.
    retriever = DenseRetriever(chunks, index=load_index(Path("data/base.faiss")))
    for pmid in picked[N_SINGLE:][:N_MULTI]:
        a = by_pmid[pmid]
        neighbours = [
            c for c in retriever.search(f"{a.title}. {a.text[:500]}", k=6)
            if c.pmid != a.pmid
        ]
        if not neighbours:
            print(f"  no neighbour for {a.pmid}, skipping")
            continue
        b = neighbours[0]
        if "|".join(sorted({a.pmid, b.pmid})) in done:
            continue
        question = complete(
            MULTI_PROMPT.format(
                title_a=a.title, text_a=a.text[:1200],
                title_b=b.title, text_b=b.text[:1200],
            )
        ).strip()
        row = {
            "id": f"q{len(rows) + 1:03d}",
            "question": question,
            "relevant_pmids": sorted({a.pmid, b.pmid}),
            "type": "multi",
        }
        rows.append(row)
        _append(row)
        print(f"[multi {a.pmid}+{b.pmid}] {question[:70]}", flush=True)
        time.sleep(SLEEP)

    if not any(r["type"] == "unanswerable" for r in rows):
        for question in UNANSWERABLE:
            row = {
                "id": f"q{len(rows) + 1:03d}",
                "question": question,
                "relevant_pmids": [],
                "type": "unanswerable",
            }
            rows.append(row)
            _append(row)

    print(f"\n{len(rows)} draft questions in {OUT_PATH}")
    print("REVIEW AND EDIT these before renaming to eval/questions.jsonl")


if __name__ == "__main__":
    main()
