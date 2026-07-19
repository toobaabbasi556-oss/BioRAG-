"""CLI: evaluate all three retrievers on the question set.

Usage:  PYTHONPATH=src python -m biorag.evaluate.run_retrieval
Writes: results/retrieval.md
"""

from pathlib import Path
from statistics import mean

from biorag.evaluate.metrics import mrr, recall_at_k
from biorag.index.dense import load_index
from biorag.ingest.build_corpus import load_corpus
from biorag.retrieve.bm25 import BM25Retriever
from biorag.retrieve.dense import DenseRetriever
from biorag.train.pairs import load_questions

OUT_PATH = Path("results/retrieval.md")


def _distinct_pmids(chunks: list, limit: int) -> list[str]:
    """Collapse chunks to distinct PMIDs, best-ranked first.

    ~2/3 of abstracts split into multiple chunks, so a top-10 chunk list holds
    only ~9 distinct PMIDs. Ground truth is labeled by PMID, so without this,
    "recall@10" would silently mean a different k on every query.
    """
    seen: list[str] = []
    for chunk in chunks:
        if chunk.pmid not in seen:
            seen.append(chunk.pmid)
        if len(seen) == limit:
            break
    return seen


def evaluate(retriever, questions: list[dict]) -> dict[str, float]:
    """Score one retriever. Unanswerable questions are excluded — recall is
    undefined without a relevant set, and they are scored separately by hand."""
    scored = [q for q in questions if q["relevant_pmids"]]
    r5, r10, rr = [], [], []

    for q in scored:
        # Over-fetch chunks so that 10 DISTINCT pmids are reliably available.
        hits = retriever.search(q["question"], k=30)
        retrieved = _distinct_pmids(hits, limit=10)
        relevant = set(q["relevant_pmids"])
        r5.append(recall_at_k(retrieved, relevant, k=5))
        r10.append(recall_at_k(retrieved, relevant, k=10))
        rr.append(mrr(retrieved, relevant))

    return {"recall@5": mean(r5), "recall@10": mean(r10), "MRR": mean(rr)}


def main() -> None:
    chunks = load_corpus()
    questions = load_questions()

    retrievers = {
        "BM25 (baseline)": BM25Retriever(chunks),
        "MiniLM (off-the-shelf)": DenseRetriever(
            chunks, index=load_index(Path("data/base.faiss"))
        ),
        "MiniLM (fine-tuned)": DenseRetriever(
            chunks,
            model_name="models/finetuned-minilm",
            index=load_index(Path("data/finetuned.faiss")),
        ),
    }

    rows = {name: evaluate(r, questions) for name, r in retrievers.items()}

    n_scored = len([q for q in questions if q["relevant_pmids"]])
    lines = [
        "# Retrieval results\n",
        f"Corpus: {len(chunks)} chunks from "
        f"{len({c.pmid for c in chunks})} abstracts. "
        f"Scored on {n_scored} answerable questions "
        f"({len(questions) - n_scored} unanswerable questions are excluded — "
        "recall is undefined without a relevant set).\n",
        "| Retriever | recall@5 | recall@10 | MRR |",
        "|---|---|---|---|",
    ]
    for name, m in rows.items():
        lines.append(
            f"| {name} | {m['recall@5']:.3f} | {m['recall@10']:.3f} | {m['MRR']:.3f} |"
        )

    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
