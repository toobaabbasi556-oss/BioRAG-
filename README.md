# BioRAG — Retrieval-Augmented QA over Biomechanics Literature

Question answering over 2,785 PubMed abstracts in human biomechanics and movement
science — gait analysis, joint kinematics, motion capture, and clinical movement
assessment. Built to **compare and measure retrieval strategies**, not to demo a
chatbot — the deliverable is the results table, not the UI.

## Results

### Retrieval (35 answerable questions, labeled by PMID)

| Retriever | recall@5 | recall@10 | MRR |
|---|---|---|---|
| BM25 (baseline) | 0.486 | 0.514 | 0.420 |
| MiniLM (off-the-shelf) | **0.743** | **0.829** | **0.631** |
| MiniLM (fine-tuned) | 0.729 | 0.800 | 0.533 |

**Dense retrieval beat the lexical baseline decisively** — recall@5 rose 53%
relative to BM25.

**Fine-tuning did not work — it lost on every metric.** recall@5 0.743 → 0.729,
recall@10 0.829 → 0.800, MRR 0.631 → 0.533. Per question, the fine-tuned model was
better on 2, worse on 12, unchanged on 21 — a 6:1 ratio against, not sampling
noise.

Why: training pairs were `(title, abstract_body)`, which teaches *document-internal*
similarity. The task being measured is `(question, abstract)` similarity — a
different relation. Optimising the cheap proxy degraded the question-to-passage
alignment MiniLM already had from its original, far larger training set. This is the
characteristic failure of weak supervision: the proxy was easy to build from the
corpus, but it was not the objective under test.

The off-the-shelf model is therefore what the app and the faithfulness review serve.
Shipping the fine-tuned model would knowingly serve the worse system.

### Generation faithfulness (20 answers)

| Check | Result |
|---|---|
| Unanswerable questions correctly refused | 5 / 5 |
| Hallucinated answers to out-of-corpus questions | 0 |
| Citations resolving to retrieved context | 35 / 35 |
| Answerable questions answered | 10 / 15 |
| False refusals (correct abstract retrieved, still refused) | 2 |
| Grounding of answered responses | 9 grounded, 1 partial, 0 hallucinated |

Three of the five refusals on answerable questions were *correct* — retrieval had
genuinely failed to surface the right abstract, and declining beats fabricating.
The other two are a real conservatism bias worth fixing.

## What I built

- Resumable PubMed ingestion via Entrez, with cleaning and sentence-boundary chunking
- Dense retrieval over L2-normalised MiniLM embeddings in a FAISS `IndexFlatIP`,
  plus a BM25 lexical baseline
- Contrastive fine-tuning of the embedding model with `MultipleNegativesRankingLoss`
- Grounded generation with PMID citations and an explicit refusal path
- Evaluation harness: recall@k, MRR, and a faithfulness review

### The design decision that mattered

All three retrievers satisfy one interface:

```python
class Retriever(Protocol):
    def search(self, query: str, k: int = 5) -> list[Chunk]: ...
```

The evaluation harness scores all three through identical code, and swapping
retrievers is a constructor argument rather than a rewrite. Generation depends only
on this protocol, never on a concrete retriever.

The same discipline applies to the LLM: everything upstream calls one `complete()`
function. When the original provider (Groq) proved unavailable mid-build, switching
to Gemini touched exactly one file.

## Correctness constraints

**Evaluation leakage.** Abstracts cited by the evaluation set are excluded from
training-pair generation (`src/biorag/train/pairs.py`), enforced in code and covered
by a unit test. On the real corpus this held out 45 PMIDs / 61 chunks. Without it,
the fine-tuned model would have been scored on abstracts it trained on.

**Chunk-vs-document counting.** About two thirds of abstracts exceed the chunker's
word limit and split, so a top-10 *chunk* list contains only ~9 distinct PMIDs.
Ground truth is labeled by PMID, so evaluation collapses chunks to distinct PMIDs
before computing recall@k — otherwise "recall@10" would silently mean a different
k on every query.

**Reproducibility.** Generation runs at `temperature=0.0` against a pinned model
(`gemini-3.1-flash-lite`), not a `-latest` alias that could change underneath a
recorded result.

## Limitations

Stated plainly, because they bound what these numbers mean:

- **The evaluation set was LLM-drafted, not hand-written** (`eval/draft_questions.py`),
  then validated automatically: all cited PMIDs exist, no duplicates, and mean
  content-word overlap with source abstracts is 0.30 — so questions paraphrase rather
  than echo, and BM25 is not disadvantaged by artificial vocabulary mismatch. It has
  not yet been fully reviewed by a domain expert.
- **n = 35 answerable questions is small.** Differences below ~0.05 should not be
  treated as real. The recall@5 gap between off-the-shelf and fine-tuned is inside
  that band; the MRR gap and the 12-vs-2 per-question split are not.
- **Faithfulness grading was AI-assisted, not independently human-reviewed.** Each
  answered response was checked claim-by-claim against the full text of its cited
  abstracts, but by the same class of system that produced the answers — weaker
  evidence than expert review, and sharing its blind spots. A domain expert should
  confirm them.
- **Non-human studies are filtered by keyword, not MeSH.** The Entrez query searched
  species-agnostic vocabulary ("gait analysis", "joint kinematics"), so ~7% of fetched
  abstracts were veterinary or animal research. `src/biorag/ingest/species.py` removes
  215 of them, but it is a heuristic: it will miss animal papers that never name a
  species, and could drop a human study that discusses an animal model early. The
  precise fix is `AND humans[MeSH Terms]` in the query, which needs a refetch.
- **Single annotator, single domain, English only.**
- **Recall is computed against labeled PMIDs only.** If another abstract also answers
  a question but was not labeled, retrievers are under-credited — equally across all
  three, so the comparison holds even though absolute values may be pessimistic.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env        # then add your GEMINI_API_KEY

PYTHONPATH=src python -m biorag.ingest.build_corpus --limit 3000
PYTHONPATH=src python -c "
from pathlib import Path
from biorag.ingest.build_corpus import load_corpus
from biorag.index.dense import build_index, save_index
save_index(build_index(load_corpus()), Path('data/base.faiss'))"

PYTHONPATH=src python -m biorag.train.finetune --epochs 1     # optional
PYTHONPATH=src python -m biorag.evaluate.run_retrieval        # the results table
PYTHONPATH=src streamlit run src/biorag/app.py
```

Tests: `pytest` (35 tests — metrics, leakage guard, chunking, retrievers,
species filter).

Docker:

```bash
docker build -t biorag .
docker run --rm -p 8501:8501 --env-file .env -v "$(pwd)/data:/app/data" biorag
```

Note: the Gemini free tier is tightly rate-limited (as low as 20 requests/day on
some models), which constrains generation-heavy steps. Retrieval evaluation is
fully local and needs no API access.

## Stack

Python 3.12, PyTorch, sentence-transformers, FAISS, rank-bm25, Biopython (Entrez),
Gemini API, Streamlit, pytest, Docker.
