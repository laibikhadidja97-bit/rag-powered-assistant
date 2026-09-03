"""Turn files on disk into plain text, one record per page.

Keeping the page as the unit of loading (rather than concatenating a whole PDF
into one string) is what lets an answer cite "bert.pdf p.4" later on. Plain-text
files have no pages, so they are loaded as a single page numbered 1.
"""
import os
import re
from dataclasses import dataclass

SUPPORTED = (".pdf", ".txt", ".md")


@dataclass
class Page:
    source: str  # filename, e.g. "bert.pdf"
    page: int    # 1-based; always 1 for .txt/.md
    text: str


def clean(text):
    """Normalise the whitespace that PDF text extraction leaves behind.

    Two things matter for retrieval quality: hyphens that split a word across a
    line break become one word again, and single newlines inside a paragraph
    become spaces while blank lines (the real paragraph breaks) survive. Without
    the second step every chunk boundary lands mid-sentence.
    """
    text = text.replace("­", "")            # soft hyphens
    text = re.sub(r"-\n(?=[a-z])", "", text)      # re-join hyphenated line breaks
    text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)
    text = re.sub(r"(?<![\n])\n(?![\n])", " ", text)  # single newline -> space
    text = re.sub(r"\n{2,}", "\n\n", text)        # collapse blank-line runs
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def load_pdf(path):
    import pymupdf  # imported lazily so .txt-only users need not install it

    pages = []
    with pymupdf.open(path) as doc:
        for number, page in enumerate(doc, start=1):
            text = clean(page.get_text("text"))
            if text:
                pages.append(Page(os.path.basename(path), number, text))
    return pages


def load_text(path):
    with open(path, encoding="utf-8", errors="replace") as handle:
        text = clean(handle.read())
    return [Page(os.path.basename(path), 1, text)] if text else []


def load_file(path):
    extension = os.path.splitext(path)[1].lower()
    if extension == ".pdf":
        return load_pdf(path)
    if extension in (".txt", ".md"):
        return load_text(path)
    raise ValueError(f"unsupported file type: {path}")


def load_dir(directory):
    """Load every supported file in a directory, sorted for reproducible ids."""
    if not os.path.isdir(directory):
        raise FileNotFoundError(
            f"{directory} does not exist — run `python get_data.py` first, "
            "or point --docs at your own folder of PDFs/notes."
        )

    names = sorted(n for n in os.listdir(directory) if n.lower().endswith(SUPPORTED))
    if not names:
        raise FileNotFoundError(f"no {'/'.join(SUPPORTED)} files found in {directory}")

    pages = []
    for name in names:
        pages.extend(load_file(os.path.join(directory, name)))
    return pages
