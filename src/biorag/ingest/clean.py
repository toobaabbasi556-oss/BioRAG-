"""Cleaning and chunking of raw PubMed abstract text."""

import re

from biorag.models import Chunk

# Publisher boilerplate carries no retrievable information and pollutes embeddings.
_BOILERPLATE = re.compile(
    r"(copyright|©)\s*\d{4}.*$|all rights reserved\.?\s*$",
    re.IGNORECASE | re.DOTALL,
)


def clean_text(raw: str) -> str:
    """Collapse whitespace and strip publisher boilerplate."""
    text = _BOILERPLATE.sub("", raw)
    return re.sub(r"\s+", " ", text).strip()


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p]


def chunk_abstract(
    pmid: str,
    title: str,
    text: str,
    journal: str,
    year: str,
    max_words: int = 220,
) -> list[Chunk]:
    """Return one Chunk per abstract, splitting only if it exceeds max_words.

    Splits on sentence boundaries with a one-sentence overlap so a fact spanning
    a boundary is still retrievable from at least one chunk.
    """
    text = clean_text(text)

    if len(text.split()) <= max_words:
        return [Chunk(id=pmid, pmid=pmid, title=title, text=text,
                      journal=journal, year=year)]

    sentences = _split_sentences(text)
    chunks: list[str] = []
    current: list[str] = []

    for sentence in sentences:
        current.append(sentence)
        if sum(len(s.split()) for s in current) >= max_words:
            chunks.append(" ".join(current))
            current = [current[-1]]  # one-sentence overlap

    remaining = [s for s in current if s]
    # Avoid emitting a trailing chunk that is only the overlap sentence.
    if remaining and (len(remaining) > 1 or not chunks):
        chunks.append(" ".join(remaining))

    return [
        Chunk(id=f"{pmid}-{i}", pmid=pmid, title=title, text=body,
              journal=journal, year=year)
        for i, body in enumerate(chunks)
    ]
