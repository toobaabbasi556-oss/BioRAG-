# src/biorag/index/dense.py
"""Embedding and FAISS index construction."""

from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from biorag.models import Chunk

BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def embed(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    """Embed texts and L2-normalise, so inner product == cosine similarity."""
    vectors = model.encode(
        texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True
    )
    faiss.normalize_L2(vectors)
    return vectors


def build_index(chunks: list[Chunk], model_name: str = BASE_MODEL) -> faiss.Index:
    """Build an exact inner-product index. At ~3k chunks, exact search is fast
    and avoids tuning an approximate index for no measurable benefit."""
    model = SentenceTransformer(model_name)
    texts = [f"{c.title}. {c.text}" for c in chunks]
    vectors = embed(model, texts)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def save_index(index: faiss.Index, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path))


def load_index(path: Path) -> faiss.Index:
    return faiss.read_index(str(path))
