"""CLI: fine-tune the embedding model on the corpus.

Usage:  PYTHONPATH=src python -m biorag.train.finetune --epochs 1
"""

import argparse

from sentence_transformers import InputExample, SentenceTransformer, losses
from torch.utils.data import DataLoader

from biorag.index.dense import BASE_MODEL
from biorag.ingest.build_corpus import load_corpus
from biorag.train.pairs import build_title_pairs, held_out_pmids, load_questions

OUT_DIR = "models/finetuned-minilm"


def main(epochs: int, batch_size: int) -> None:
    chunks = load_corpus()
    held = held_out_pmids(load_questions())
    pairs = build_title_pairs(chunks, held_out=held)

    print(f"{len(chunks)} chunks, {len(held)} held-out pmids, {len(pairs)} pairs")
    if not pairs:
        raise SystemExit("no training pairs — check the corpus and question set")
    if not held:
        raise SystemExit(
            "no held-out pmids — the leakage guard is not wired up, and any "
            "result measured after training would be invalid"
        )

    model = SentenceTransformer(BASE_MODEL)
    examples = [InputExample(texts=[q, p]) for q, p in pairs]
    loader = DataLoader(examples, shuffle=True, batch_size=batch_size)
    # Needs only positive pairs: every other example in the batch serves as a
    # negative, which is why no negative mining step is required here.
    loss = losses.MultipleNegativesRankingLoss(model)

    model.fit(
        train_objectives=[(loader, loss)],
        epochs=epochs,
        warmup_steps=int(0.1 * len(loader) * epochs),
        show_progress_bar=True,
    )
    model.save(OUT_DIR)
    print(f"saved to {OUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    main(epochs=args.epochs, batch_size=args.batch_size)
