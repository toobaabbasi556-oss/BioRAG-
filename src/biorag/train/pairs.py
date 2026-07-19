"""Weakly-supervised training pair generation for embedding fine-tuning.

MultipleNegativesRankingLoss needs only positive pairs — negatives come from
other examples in the batch — so the pairs here are all (query, positive) shaped.
"""

import json
from pathlib import Path

from biorag.models import Chunk

QUESTIONS_PATH = Path("eval/questions.jsonl")


def load_questions(path: Path = QUESTIONS_PATH) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f]


def held_out_pmids(questions: list[dict]) -> set[str]:
    """Every PMID cited by the eval set. These must never appear in training."""
    return {pmid for q in questions for pmid in q["relevant_pmids"]}


def build_title_pairs(
    chunks: list[Chunk], held_out: set[str]
) -> list[tuple[str, str]]:
    """(title, body) pairs — a title is a natural short query for its abstract."""
    return [
        (c.title, c.text)
        for c in chunks
        if c.pmid not in held_out and c.text.strip() and c.title.strip()
    ]
