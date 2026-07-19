from biorag.ingest.clean import clean_text, chunk_abstract
from biorag.models import Chunk


def test_clean_text_collapses_whitespace():
    assert clean_text("hello   \n\n world  ") == "hello world"


def test_clean_text_strips_copyright_boilerplate():
    raw = "Spinal load increased. Copyright 2019 Elsevier Ltd. All rights reserved."
    assert clean_text(raw) == "Spinal load increased."


def test_chunk_abstract_short_returns_single_chunk():
    chunks = chunk_abstract(
        pmid="111", title="Gait study", text="Short abstract body.",
        journal="J Biomech", year="2020", max_words=100,
    )
    assert len(chunks) == 1
    assert chunks[0].id == "111"
    assert chunks[0].pmid == "111"
    assert chunks[0].text == "Short abstract body."


def test_chunk_abstract_long_splits_with_overlap():
    text = ". ".join(f"Sentence number {i}" for i in range(1, 13)) + "."
    chunks = chunk_abstract(
        pmid="222", title="T", text=text, journal="J", year="2021", max_words=12,
    )
    assert len(chunks) > 1
    assert [c.id for c in chunks] == [f"222-{i}" for i in range(len(chunks))]
    # overlap: last sentence of chunk 0 reappears at the start of chunk 1
    assert chunks[0].text.split(". ")[-1].rstrip(".") in chunks[1].text


def test_chunk_abstract_returns_chunk_objects():
    chunks = chunk_abstract(
        pmid="333", title="T", text="Body.", journal="J", year="2022", max_words=100,
    )
    assert isinstance(chunks[0], Chunk)
