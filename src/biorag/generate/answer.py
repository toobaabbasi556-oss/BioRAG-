"""Prompt assembly and grounded answer generation."""

from dataclasses import dataclass

from biorag.generate.llm import complete
from biorag.models import Chunk
from biorag.retrieve.base import Retriever

NOT_FOUND = "Not found in corpus."

PROMPT = """You are answering questions using only the provided biomedical abstracts.

Rules:
- Use ONLY the abstracts below. Do not use outside knowledge.
- Cite the PMID in square brackets after each claim, e.g. [12345678].
- If the abstracts do not answer the question, reply exactly: {not_found}

Abstracts:
{context}

Question: {question}

Answer:"""


@dataclass
class Answer:
    text: str
    chunks: list[Chunk]


def _format_context(chunks: list[Chunk]) -> str:
    return "\n\n".join(
        f"[{c.pmid}] {c.title}\n{c.text}" for c in chunks
    )


def answer_question(question: str, retriever: Retriever, k: int = 5) -> Answer:
    """Retrieve, then generate a grounded answer with citations."""
    chunks = retriever.search(question, k=k)
    if not chunks:
        return Answer(text=NOT_FOUND, chunks=[])

    prompt = PROMPT.format(
        not_found=NOT_FOUND, context=_format_context(chunks), question=question
    )
    return Answer(text=complete(prompt).strip(), chunks=chunks)
