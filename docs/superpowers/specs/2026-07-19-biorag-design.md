# BioRAG — Retrieval-Augmented QA over Musculoskeletal Biomechanics Literature

**Date:** 2026-07-19
**Author:** Tooba Maqsood
**Status:** Approved design, ready for implementation planning

## Purpose

Build a retrieval-augmented question answering system over PubMed abstracts in
musculoskeletal and spinal biomechanics, and **measure it**.

The project exists to fill three specific gaps in the author's CV for ML/AI engineer
roles: no LLM or vector-search work, no trained model with a reported metric, and no
deployed artifact. The deliverable that carries CV value is not the chatbot — it is a
results table showing that a fine-tuned embedding model retrieves better than both a
keyword baseline and an off-the-shelf embedding model.

Domain choice is deliberate: it aligns with the author's ongoing biomechanics research
at NUST, so the CV reads as one coherent track rather than unrelated projects.

## Constraints

- **Time budget: 5 days at 5–6 hours per day (~28 hours).** This is the binding
  constraint and drives every scope decision below.
- **No paid API access.** Generation uses Groq's free tier (Gemini free tier as
  fallback). Embedding and fine-tuning run locally on CPU; free Colab GPU is an
  acceptable escape hatch if fine-tuning is too slow.
- Author has no prior LLM, vector-search, or deployment experience. Day 2 carries the
  highest risk of overrun.

## Success Criteria

The project is a success if, at the end of day 5, the repository contains:

1. A working end-to-end pipeline: question in, cited answer out.
2. An evaluation set of ~40 hand-written questions with labeled ground-truth abstracts.
3. A three-row ablation table reporting recall@5, recall@10, and MRR.
4. A fine-tuned embedding model that measurably beats the off-the-shelf model on at
   least one retrieval metric — **or**, if it does not, a README that says so and
   analyses why. A negative result honestly reported is acceptable; an unmeasured
   system is not.
5. A README containing the numbers and a stated list of failure modes.

It is **not** a success criterion that the fine-tuned model wins. It is a success
criterion that the comparison is real and correctly measured.

## Architecture

Five components with one-directional data flow. Each is independently testable and
writes its output to disk, so a failure in a later stage never forces re-running an
earlier one.

```
ingest/     PubMed fetch (Entrez API) -> clean -> chunk -> corpus.jsonl
index/      chunks -> embeddings -> FAISS index;  chunks -> BM25 index
retrieve/   query -> top-k chunks   [Retriever interface, 3 implementations]
train/      corpus -> synthetic pairs -> fine-tuned embedding model
generate/   query + retrieved chunks -> prompt -> LLM -> answer with citations
eval/       question set -> retrievers + pipeline -> metrics tables
```

### The key design decision

`retrieve/` exposes a single interface:

```python
class Retriever(Protocol):
    def search(self, query: str, k: int) -> list[Chunk]: ...
```

with three implementations: `BM25Retriever`, `DenseRetriever` (off-the-shelf
embeddings), and `DenseRetriever` loaded with the fine-tuned model. The ablation is
therefore a config change, not a rewrite, and `eval/` runs all three through identical
code. This is the design decision most worth being able to explain in an interview.

`generate/` depends only on the `Retriever` interface, never on a concrete retriever.

## Components

### ingest/
Fetches ~3,000 abstracts via the Entrez E-utilities API (free, no key required for low
request rates; requires an email in the request header). Query terms target spinal
loading, gait, and musculoskeletal biomechanics.

Chunking: abstracts are short, so **one chunk per abstract** is the default. No
sliding-window chunking unless abstracts exceed the embedding model's 256-token limit,
in which case split on sentence boundaries with a 1-sentence overlap.

Output: `data/corpus.jsonl`, one record per chunk with `id`, `pmid`, `title`, `text`,
`journal`, `year`.

Written once on day 1 and then treated as read-only. Committed to the repo if under
GitHub's file size limits; otherwise the fetch script is committed and the data is not.

### index/
Embeds chunks with `sentence-transformers/all-MiniLM-L6-v2` (384-dim, CPU-friendly)
and builds a FAISS `IndexFlatIP` over L2-normalised vectors, so inner product equals
cosine similarity. At ~3,000 chunks, an exact index is fast and avoids tuning an
approximate index for no benefit.

Also builds a BM25 index (`rank_bm25`) over the same chunks for the baseline.

### train/
Fine-tunes the embedding model with `MultipleNegativesRankingLoss` from
sentence-transformers, which needs only positive pairs — in-batch negatives supply the
rest.

Training pairs are generated from the corpus by weak supervision: for each abstract,
form `(title, abstract_body)` pairs, plus `(generated_question, abstract)` pairs where
questions are produced by the LLM from the abstract text. Target ~3,000–5,000 pairs.

**Critical rule:** abstracts used in the evaluation set are excluded from training pair
generation. Training on evaluation data would invalidate the entire results table.
This exclusion is enforced in code, not by hand, and is unit tested.

Training config: 1–2 epochs, batch size 16–32, default learning rate. Small model,
small dataset — this runs in minutes to low hours on CPU. If it exceeds ~1 hour,
move to free Colab GPU rather than reducing the dataset.

### generate/
Assembles a prompt from the query plus top-k retrieved chunks, instructing the model to
answer only from the provided context and to cite PMIDs. Calls Groq's free-tier API.

Handles the two failure modes that matter: no relevant chunks retrieved (answer "not
found in corpus" rather than hallucinating), and API errors (retry once, then fail
loudly rather than silently returning an empty answer).

### eval/
Two evaluation layers.

**Retrieval (automated).** For each of ~40 questions with labeled ground-truth abstract
IDs, compute recall@5, recall@10, and MRR for each of the three retrievers. Output a
markdown table.

**Generation (manual).** For 20 questions, the author scores the generated answer as
`grounded` / `partially grounded` / `hallucinated` against the retrieved context, and
records the counts. Manual scoring is honest at this scale; an automated LLM-judge
would add a day and its own unmeasured error.

### Interface
A minimal Streamlit page: text input, answer, and the cited abstracts. If time runs
short this is replaced by a CLI script — the evaluation is what has CV value, not the UI.

## Evaluation Set

~40 questions written by hand against the corpus, each labeled with the PMID(s) of the
abstract(s) that answer it. Stored as `eval/questions.jsonl`.

Question mix, deliberately including cases the system should fail on:
- ~25 factual questions answerable from a single abstract
- ~10 questions requiring synthesis across two or more abstracts
- ~5 questions with no answer in the corpus, to test refusal behaviour

Writing this set is the single most time-consuming manual task in the project and is
budgeted a half day. It is also the component that most distinguishes the project from
a tutorial, so it is never the thing that gets cut.

## Testing

Not full TDD given the time budget, but two areas are non-negotiable because bugs there
silently produce false results:

- **Metric functions** (`recall@k`, `MRR`) — tested against hand-computed expected
  values on small fixtures.
- **Train/eval separation** — a test asserting that no evaluation abstract appears in
  the generated training pairs.

Chunking gets a smoke test. The LLM call is not unit tested; it is exercised manually.

## Error Handling

- **Entrez API:** rate-limited to 3 requests/second, retried with backoff. Fetch is
  resumable — already-fetched PMIDs are skipped, so an interrupted day-1 run does not
  restart from zero.
- **LLM API:** one retry, then a clear error. Never silently return an empty answer.
- **Empty retrieval:** explicit "not found in corpus" path, exercised by the 5
  unanswerable evaluation questions.

## Scope: Priority Ladder

Implemented in this order. If the schedule slips, cut from the bottom without
renegotiating.

**Never cut:**
1. End-to-end retrieval + generation
2. Evaluation set and retrieval metrics
3. Fine-tuned embeddings and the three-row ablation table
4. README with honest numbers and stated failure modes

**Cut in this order:**
5. Docker
6. Streamlit page (CLI script substitutes)
7. Faithfulness scoring on 20 answers, reduced to 10

**Explicitly out of scope:** cross-encoder reranking, agents, multi-topic corpora,
cloud deployment, LLM-as-judge evaluation, authentication, conversation history.

## Day Plan

| Day | Work | Risk |
|-----|------|------|
| 1 | Repo setup, Entrez fetch, cleaning, chunking, `corpus.jsonl` | Low |
| 2 | Embeddings, FAISS index, BM25 index, retrieval working end-to-end | **Highest — new tooling** |
| 3 | Generation pipeline, prompt, citations, minimal Streamlit page | Medium |
| 4 | Evaluation set, metric functions + tests, fine-tuning, ablation table | Medium |
| 5 | Docker, README with results, buffer | Low |

Day 5 is buffer by design. If day 2 overruns, Docker and Streamlit are dropped
immediately rather than compressing day 4 — day 4 contains all the CV value.

## CV Framing

The CV entry should lead with the measured result, not the technology stack:

> **BioRAG — Retrieval-Augmented QA over Biomechanics Literature**
> Built and evaluated a RAG system over ~3,000 PubMed abstracts. Fine-tuned a
> sentence-transformer embedding model with contrastive learning, improving recall@5
> from X% to Y% over an off-the-shelf model and Z% over a BM25 baseline. Evaluated on a
> hand-labeled 40-question set (recall@k, MRR) with manual faithfulness scoring.
> Python, PyTorch, sentence-transformers, FAISS, Docker.

The X/Y/Z figures are filled in from actual results. If fine-tuning does not improve
retrieval, the entry reports that honestly and describes the analysis — which is still
a stronger interview story than an unmeasured system.
