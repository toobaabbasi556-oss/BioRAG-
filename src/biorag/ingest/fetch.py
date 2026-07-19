"""Resumable PubMed fetching via Entrez E-utilities."""

import json
import time
from pathlib import Path

from Bio import Entrez

# NCBI requires a contact email with every request.
Entrez.email = "Toobaabbasi556@gmail.com"

SEARCH_QUERY = (
    '("spinal load"[All Fields] OR "spine biomechanics"[All Fields] '
    'OR "gait analysis"[All Fields] OR "musculoskeletal model"[All Fields] '
    'OR "lumbar spine biomechanics"[All Fields] OR "joint kinematics"[All Fields] '
    'OR "ground reaction force"[All Fields] OR "electromyography"[Title] '
    'OR "motion capture"[All Fields]) AND hasabstract'
)


def search_pmids(query: str = SEARCH_QUERY, retmax: int = 3000) -> list[str]:
    """Return up to retmax PMIDs matching the query."""
    handle = Entrez.esearch(db="pubmed", term=query, retmax=retmax)
    record = Entrez.read(handle)
    handle.close()
    return list(record["IdList"])


def _extract(article: dict) -> dict | None:
    """Pull the fields we need out of one Entrez article record."""
    try:
        citation = article["MedlineCitation"]
        art = citation["Article"]
        abstract_parts = art["Abstract"]["AbstractText"]
        # AbstractText entries are str subclasses; structured abstracts have several.
        text = " ".join(str(p) for p in abstract_parts).strip()
        if not text:
            return None
        pub_date = (
            art.get("Journal", {}).get("JournalIssue", {}).get("PubDate", {})
        )
        # PubDate carries either a Year or only a MedlineDate string like "2019 Jul-Aug".
        year = str(pub_date.get("Year") or pub_date.get("MedlineDate", "unknown"))[:4]
        return {
            "pmid": str(citation["PMID"]),
            "title": str(art.get("ArticleTitle", "")),
            "text": text,
            "journal": str(art.get("Journal", {}).get("Title", "unknown")),
            "year": year,
        }
    except (KeyError, IndexError):
        # Records without abstracts or with unexpected shape are skipped, not fatal.
        return None


def fetch_records(pmids: list[str], out_path: Path, batch_size: int = 100) -> int:
    """Fetch abstracts for pmids, appending JSON lines to out_path.

    Resumable: PMIDs already present in out_path are skipped, so an interrupted
    run continues rather than restarting.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    if out_path.exists():
        with out_path.open() as f:
            for line in f:
                seen.add(json.loads(line)["pmid"])

    todo = [p for p in pmids if p not in seen]
    print(f"{len(seen)} already fetched, {len(todo)} to go")

    written = 0
    with out_path.open("a") as f:
        for i in range(0, len(todo), batch_size):
            batch = todo[i : i + batch_size]
            for attempt in range(3):
                try:
                    handle = Entrez.efetch(
                        db="pubmed", id=",".join(batch), retmode="xml"
                    )
                    records = Entrez.read(handle)
                    handle.close()
                    break
                except Exception as e:  # network/rate-limit failures are transient
                    if attempt == 2:
                        raise
                    print(f"  batch failed ({e}), retrying in {2 ** attempt}s")
                    time.sleep(2**attempt)

            for article in records.get("PubmedArticle", []):
                rec = _extract(article)
                if rec:
                    f.write(json.dumps(rec) + "\n")
                    written += 1
            f.flush()
            print(f"  fetched {i + len(batch)}/{len(todo)}")
            time.sleep(0.4)  # NCBI allows 3 req/s without an API key

    return written
