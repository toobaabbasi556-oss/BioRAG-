"""Lexical baseline. This is the number the fine-tuned model must beat."""

import re

from rank_bm25 import BM25Okapi

from biorag.models import Chunk


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25Retriever:
    def __init__(self, chunks: list[Chunk]) -> None:
        self._chunks = chunks
        # Title carries strong signal in abstracts, so index it alongside the body.
        corpus = [_tokenize(f"{c.title} {c.text}") for c in chunks]
        self._bm25 = BM25Okapi(corpus)

    def search(self, query: str, k: int = 5) -> list[Chunk]:
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self._chunks[i] for i in ranked[:k]]
