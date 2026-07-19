"""Shared data types. Every module imports Chunk from here."""

from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class Chunk:
    """One retrievable unit of text. For this corpus, one chunk == one abstract."""

    id: str  # unique chunk id, e.g. "12345678" or "12345678-1" when split
    pmid: str
    title: str
    text: str
    journal: str
    year: str

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Chunk":
        return Chunk(**d)
