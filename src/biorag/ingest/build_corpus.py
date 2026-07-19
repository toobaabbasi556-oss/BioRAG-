"""CLI: fetch PubMed records, clean and chunk them into data/corpus.jsonl.

Usage:  python -m biorag.ingest.build_corpus --limit 3000
"""

import argparse
import json
from pathlib import Path

from biorag.ingest.clean import chunk_abstract
from biorag.ingest.fetch import fetch_records, search_pmids

RAW_PATH = Path("data/raw.jsonl")
CORPUS_PATH = Path("data/corpus.jsonl")


def build(limit: int) -> None:
    pmids = search_pmids(retmax=limit)
    print(f"search returned {len(pmids)} pmids")
    fetch_records(pmids, RAW_PATH)

    chunks = []
    with RAW_PATH.open() as f:
        for line in f:
            rec = json.loads(line)
            chunks.extend(
                chunk_abstract(
                    pmid=rec["pmid"], title=rec["title"], text=rec["text"],
                    journal=rec["journal"], year=rec["year"],
                )
            )

    with CORPUS_PATH.open("w") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk.to_dict()) + "\n")

    print(f"wrote {len(chunks)} chunks to {CORPUS_PATH}")


def load_corpus(path: Path = CORPUS_PATH) -> list:
    """Load the corpus as Chunk objects. Used by every downstream module."""
    from biorag.models import Chunk

    with path.open() as f:
        return [Chunk.from_dict(json.loads(line)) for line in f]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=3000)
    build(parser.parse_args().limit)
