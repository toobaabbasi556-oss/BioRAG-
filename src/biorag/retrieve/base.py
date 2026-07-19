"""The single retrieval interface every implementation satisfies.

Keeping this narrow is what makes the ablation a config change: eval code and
generation code depend on `Retriever`, never on a concrete class.
"""

from typing import Protocol

from biorag.models import Chunk


class Retriever(Protocol):
    def search(self, query: str, k: int = 5) -> list[Chunk]:
        """Return the k most relevant chunks, best first."""
        ...
