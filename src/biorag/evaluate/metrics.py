"""Retrieval metrics. Pure functions, no I/O — so they can be tested exactly."""


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Fraction of relevant documents that appear in the top k retrieved."""
    if not relevant:
        raise ValueError("recall_at_k requires a non-empty relevant set")
    top = set(retrieved[:k])
    return len(top & relevant) / len(relevant)


def mrr(retrieved: list[str], relevant: set[str]) -> float:
    """Reciprocal rank of the first relevant document; 0.0 if none retrieved."""
    for position, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / position
    return 0.0
