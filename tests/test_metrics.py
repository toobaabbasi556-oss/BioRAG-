import pytest

from biorag.evaluate.metrics import mrr, recall_at_k


def test_recall_hit_within_k():
    assert recall_at_k(retrieved=["a", "b", "c"], relevant={"b"}, k=3) == 1.0


def test_recall_miss_outside_k():
    assert recall_at_k(retrieved=["a", "b", "c"], relevant={"c"}, k=2) == 0.0


def test_recall_partial_with_multiple_relevant():
    # two relevant docs, one retrieved within k -> 0.5
    assert recall_at_k(retrieved=["a", "b"], relevant={"b", "z"}, k=2) == 0.5


def test_recall_all_relevant_found():
    assert recall_at_k(retrieved=["a", "b"], relevant={"a", "b"}, k=2) == 1.0


def test_recall_empty_relevant_raises():
    # A question with no ground truth is a data bug, not a zero score.
    with pytest.raises(ValueError):
        recall_at_k(retrieved=["a"], relevant=set(), k=1)


def test_mrr_first_position():
    assert mrr(retrieved=["a", "b"], relevant={"a"}) == 1.0


def test_mrr_second_position():
    assert mrr(retrieved=["a", "b"], relevant={"b"}) == 0.5


def test_mrr_third_position():
    assert mrr(retrieved=["a", "b", "c"], relevant={"c"}) == pytest.approx(1 / 3)


def test_mrr_no_hit_is_zero():
    assert mrr(retrieved=["a", "b"], relevant={"z"}) == 0.0


def test_mrr_uses_first_relevant_only():
    assert mrr(retrieved=["a", "b", "c"], relevant={"b", "c"}) == 0.5
