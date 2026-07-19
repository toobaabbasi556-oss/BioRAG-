# Retrieval results

Corpus: 4593 chunks from 3000 abstracts. Scored on 35 answerable questions (5 unanswerable questions are excluded — recall is undefined without a relevant set).

| Retriever | recall@5 | recall@10 | MRR |
|---|---|---|---|
| BM25 (baseline) | 0.486 | 0.514 | 0.435 |
| MiniLM (off-the-shelf) | 0.771 | 0.857 | 0.698 |
| MiniLM (fine-tuned) | 0.786 | 0.857 | 0.595 |

## Reading these numbers

**Dense retrieval clearly beats the lexical baseline.** Off-the-shelf MiniLM lifts
recall@5 from 0.486 to 0.771 over BM25 — a 59% relative improvement. Questions were
written to paraphrase rather than echo their source abstracts (mean content-word
overlap 0.30), so BM25 is not being handicapped by artificial vocabulary mismatch.

**Fine-tuning did not work, and the headline metric got worse.** recall@5 rose by
0.015 — well within noise at n=35 — recall@10 was unchanged, and MRR fell from
0.698 to 0.595. Per-question, the fine-tuned model was better on 2 questions, worse
on 10, and unchanged on 23. A 5:1 ratio against is not sampling noise.

So the fine-tuned model retrieves roughly the same documents but ranks them lower.

### Why it likely failed

The training pairs were `(title, abstract_body)`. That teaches the model
"this title matches this abstract" — a *document-internal* similarity. The actual
retrieval task is "this **question** matches this abstract", which is a different
relation. Optimising hard for the first appears to have degraded the
question-to-passage alignment that MiniLM already had from its original, far
larger and more diverse training data.

This is the risk of weak supervision: the proxy objective was cheap to construct
from the corpus, but it was not the objective we are measured on.

### What would be tried next

1. Train on `(generated_question, abstract)` pairs instead — costly here because
   the Gemini free tier allows only ~20-1000 requests/day depending on model, but
   it matches the evaluation task directly.
2. Lower the learning rate and use fewer steps; 1 epoch over 4,531 pairs may
   already be over-fitting a 22M-parameter model to a narrow objective.
3. Hold out a validation split to catch the MRR regression during training rather
   than after it.

### Honesty note

This result is reported as measured. The fine-tuned model is retained in the
comparison rather than dropped, because a negative result that is explained is
more informative than a table showing only what worked.
