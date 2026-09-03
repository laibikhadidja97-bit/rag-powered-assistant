"""Split pages into overlapping chunks small enough to embed and to fit in a prompt.

Chunk size is the main retrieval knob in a RAG system. Too small and a chunk
loses the context that makes it interpretable; too large and the embedding
averages several topics together, so it matches everything weakly and nothing
strongly. ~1200 characters (roughly 250 tokens, comfortably inside the 512-token
window of the default embedding model) with 200 characters of overlap is a
reasonable default for prose-heavy PDFs like papers or lecture notes.

The splitter is recursive: it tries to break on the largest natural boundary that
fits — paragraph, then sentence, then word — so chunks rarely start mid-sentence.
"""
import re
from dataclasses import dataclass, asdict

from .loaders import Page

CHUNK_CHARS = 1200
OVERLAP_CHARS = 200


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    page: int
    index: int  # position of this chunk within its page

    def as_metadata(self):
        """Chroma stores metadata as flat scalars, so drop id/text."""
        data = asdict(self)
        data.pop("id")
        data.pop("text")
        return data


def _split_points(text, separator):
    """Offsets just past each separator match, i.e. candidate break positions."""
    return [match.end() for match in re.finditer(separator, text)]


def split_text(text, chunk_chars=CHUNK_CHARS, overlap_chars=OVERLAP_CHARS):
    """Greedily cut `text` into <= chunk_chars pieces that overlap by ~overlap_chars.

    For each piece we take the furthest natural boundary that still fits inside
    the budget. If a single "sentence" is longer than the budget (tables, long
    formulas, dense reference lists) we fall back to a hard cut, which is correct
    but rare.
    """
    text = text.strip()
    if len(text) <= chunk_chars:
        return [text] if text else []

    # Preference order: paragraph break, sentence end, then any whitespace.
    boundaries = (
        _split_points(text, r"\n\n+"),
        _split_points(text, r"(?<=[.!?])[\s]+"),
        _split_points(text, r"\s+"),
    )

    chunks = []
    start = 0
    while start < len(text):
        limit = start + chunk_chars
        if limit >= len(text):
            chunks.append(text[start:].strip())
            break

        end = 0
        for candidates in boundaries:
            # furthest boundary that is inside the budget and makes progress
            fitting = [p for p in candidates if start < p <= limit]
            if fitting:
                end = fitting[-1]
                break
        if end <= start:
            end = limit  # nothing to break on: hard cut

        chunks.append(text[start:end].strip())
        # Step back by the overlap so context spanning a boundary is not lost,
        # but always move forward to guarantee termination.
        start = max(end - overlap_chars, start + 1)

    return [c for c in chunks if c]


def chunk_pages(pages, chunk_chars=CHUNK_CHARS, overlap_chars=OVERLAP_CHARS, min_chars=80):
    """Chunk a list of Pages into Chunks with stable, content-independent ids.

    Chunks shorter than `min_chars` are dropped: on paper PDFs these are almost
    always page headers, figure captions stranded on their own, or page numbers,
    and they pollute retrieval because short strings embed close to everything.
    """
    chunks = []
    for page in pages:
        assert isinstance(page, Page)
        for index, text in enumerate(split_text(page.text, chunk_chars, overlap_chars)):
            if len(text) < min_chars:
                continue
            chunks.append(Chunk(
                id=f"{page.source}:{page.page}:{index}",
                text=text,
                source=page.source,
                page=page.page,
                index=index,
            ))
    return chunks
