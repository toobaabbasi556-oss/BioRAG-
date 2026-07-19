from biorag.models import Chunk
from biorag.retrieve.bm25 import BM25Retriever

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
