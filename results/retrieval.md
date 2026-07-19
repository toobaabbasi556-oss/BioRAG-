# Retrieval results

Corpus: 4248 chunks from 2785 abstracts. Scored on 35 answerable questions (5 unanswerable questions are excluded — recall is undefined without a relevant set).

| Retriever | recall@5 | recall@10 | MRR |
|---|---|---|---|
| BM25 (baseline) | 0.486 | 0.514 | 0.420 |
| MiniLM (off-the-shelf) | 0.743 | 0.829 | 0.631 |
| MiniLM (fine-tuned) | 0.729 | 0.800 | 0.533 |

## Reading these numbers

**Dense retrieval clearly beats the lexical baseline.** Off-the-shelf MiniLM lifts
recall@5 from 0.486 to 0.743 — a 53% relative improvement. Questions paraphrase
rather than echo their source abstracts (mean content-word overlap 0.30), so BM25
is not handicapped by artificial vocabulary mismatch.

**Fine-tuning did not work. It lost on every metric.** recall@5 0.743 -> 0.729,
recall@10 0.829 -> 0.800, MRR 0.631 -> 0.533. Per question, the fine-tuned model
was better on 2, worse on 12, unchanged on 21. A 6:1 ratio against is not noise.

### Why it likely failed

Training pairs were `(title, abstract_body)`. That teaches *document-internal*
similarity — "this title matches this abstract". The task being measured is
`(question, abstract)` similarity, a different relation. Optimising hard for the
cheap proxy degraded the question-to-passage alignment MiniLM already had from its
original, far larger and more diverse training data.

This is the characteristic failure of weak supervision: the proxy objective was easy
to construct from the corpus, but it was not the objective under evaluation.

### What would be tried next

1. Train on `(generated_question, abstract)` pairs, matching the evaluation task
   directly. Constrained here by the Gemini free-tier daily quota.
2. Lower the learning rate and use fewer steps — one epoch over ~4,200 pairs may
   already over-fit a 22M-parameter model to a narrow objective.
3. Hold out a validation split to catch the regression *during* training rather
   than after it.

### Honesty note

Reported as measured. The fine-tuned model is kept in the comparison rather than
dropped, because a negative result that is explained is more informative than a
table showing only what worked. The off-the-shelf model is what the app serves.
