import pytest

from biorag.models import Chunk
from biorag.train.pairs import build_title_pairs, held_out_pmids

CHUNKS = [
    Chunk(id="1", pmid="1", title="Lumbar loading in lifting", text="Body one.",
          journal="J", year="2020"),
    Chunk(id="2", pmid="2", title="Knee kinematics in running", text="Body two.",
          journal="J", year="2021"),
]

QUESTIONS = [{"id": "q1", "question": "x", "relevant_pmids": ["2"], "type": "single"}]


def test_held_out_pmids_collects_all_cited_pmids():
    assert held_out_pmids(QUESTIONS) == {"2"}


def test_held_out_pmids_ignores_unanswerable_questions():
    qs = [{"id": "q9", "question": "x", "relevant_pmids": [], "type": "unanswerable"}]
    assert held_out_pmids(qs) == set()


def test_build_title_pairs_excludes_held_out_chunks():
    pairs = build_title_pairs(CHUNKS, held_out={"2"})
    assert len(pairs) == 1
    assert pairs[0] == ("Lumbar loading in lifting", "Body one.")


def test_build_title_pairs_without_exclusion_uses_all():
    assert len(build_title_pairs(CHUNKS, held_out=set())) == 2


def test_no_eval_pmid_survives_into_training_pairs():
    """The guard: evaluation abstracts must never become training data."""
    held = held_out_pmids(QUESTIONS)
    pairs = build_title_pairs(CHUNKS, held_out=held)
    leaked = [p for p in pairs if "Knee kinematics" in p[0] or "Body two" in p[1]]
    assert leaked == []


def test_build_title_pairs_skips_empty_text():
    chunks = [Chunk(id="3", pmid="3", title="T", text="", journal="J", year="2020")]
    assert build_title_pairs(chunks, held_out=set()) == []
