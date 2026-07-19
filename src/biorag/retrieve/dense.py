# src/biorag/retrieve/dense.py
"""Embedding-based retriever.

`model_name` is a constructor argument, which is the whole point: the off-the-shelf
run and the fine-tuned run are the same class with a different path.
"""

import faiss
from sentence_transformers import SentenceTransformer

from biorag.index.dense import BASE_MODEL, build_index
from biorag.models import Chunk


class DenseRetriever:
    def __init__(
        self,
        chunks: list[Chunk],
        model_name: str = BASE_MODEL,
        index: faiss.Index | None = None,
    ) -> None:
        self._chunks = chunks
        self._model = SentenceTransformer(model_name)
        self._index = index if index is not None else build_index(chunks, model_name)

    def search(self, query: str, k: int = 5) -> list[Chunk]:
        vector = self._model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(vector)
        _scores, ids = self._index.search(vector, k)
        return [self._chunks[i] for i in ids[0] if i != -1]
