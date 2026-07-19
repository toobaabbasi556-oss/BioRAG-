import pytest

from biorag.models import Chunk
from biorag.retrieve.bm25 import BM25Retriever
from biorag.retrieve.dense import DenseRetriever

CHUNKS = [
    Chunk(id="1", pmid="1", title="A", text="lumbar spine compressive load during lifting",
          journal="J", year="2020"),
    Chunk(id="2", pmid="2", title="B", text="knee joint kinematics in running athletes",
          journal="J", year="2021"),
    Chunk(id="3", pmid="3", title="C", text="shoulder muscle activation during throwing",
          journal="J", year="2022"),
]


def test_bm25_ranks_lexically_matching_chunk_first():
    r = BM25Retriever(CHUNKS)
    hits = r.search("lumbar compressive load", k=2)
    assert hits[0].id == "1"


def test_bm25_respects_k():
    r = BM25Retriever(CHUNKS)
    assert len(r.search("knee running", k=2)) == 2


def test_bm25_returns_chunks_not_indices():
    r = BM25Retriever(CHUNKS)
    assert isinstance(r.search("knee", k=1)[0], Chunk)


@pytest.mark.slow
def test_dense_retrieves_semantically_related_chunk():
    """Query shares no keywords with the target — only BM25 would miss this."""
    r = DenseRetriever(CHUNKS)
    hits = r.search("how does the lower back handle heavy weight?", k=1)
    assert hits[0].id == "1"
