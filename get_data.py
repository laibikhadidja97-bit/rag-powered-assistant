"""
Fetch the demo corpus — ten foundational machine-learning papers — into data/docs/

The assistant is document-agnostic: drop any .pdf, .txt or .md file into data/docs/
and re-run `python ingest.py`. This script just gives the project something real
and reproducible to answer questions about out of the box, so the pipeline can be
demonstrated without shipping a private set of notes.

The papers are pulled from arXiv, which serves the author-submitted PDFs openly.
Versions are pinned so the page numbers quoted in README.md stay correct. arXiv
asks automated clients to identify themselves and to space requests out, so this
script sets a descriptive User-Agent and sleeps between downloads; please do not
remove either.

    python get_data.py

Roughly 25 MB and ~450 pages in total. Files already present are skipped, so the
script is safe to re-run after an interrupted download.
"""
import os
import sys
import time
import urllib.error
import urllib.request

OUT_DIR = os.path.join("data", "docs")

# arXiv id (version-pinned) -> output filename. The mix is deliberate: enough
# overlap between papers that questions can span several of them, and enough
# variety that retrieval has to actually discriminate.
PAPERS = [
    ("1706.03762v7", "attention-is-all-you-need.pdf"),
    ("1810.04805v2", "bert.pdf"),
    ("2005.14165v4", "gpt3-few-shot-learners.pdf"),
    ("2005.11401v4", "rag-knowledge-intensive-nlp.pdf"),
    ("1512.03385v1", "resnet.pdf"),
    ("1412.6980v9", "adam-optimizer.pdf"),
    ("2010.11929v2", "vision-transformer.pdf"),
    ("2106.09685v2", "lora.pdf"),
    ("1502.03167v3", "batch-normalization.pdf"),
    ("2203.02155v1", "instructgpt.pdf"),
]

# export.arxiv.org is the mirror arXiv points automated clients at.
BASE = "https://export.arxiv.org/pdf/"
UA = "RAG-powered-Assistant/1.0 (course project; contact via GitHub repo)"
DELAY = 3.0  # seconds between requests, per arXiv's robot guidance


def download(arxiv_id, local):
    """Fetch one PDF, writing through a .part file so partial downloads never
    masquerade as complete ones on a re-run."""
    if os.path.exists(local):
        print(f"  {os.path.basename(local):<36} cached")
        return False

    tmp = local + ".part"
    request = urllib.request.Request(BASE + arxiv_id, headers={"User-Agent": UA})
    with urllib.request.urlopen(request, timeout=60) as response, open(tmp, "wb") as out:
        total = int(response.headers.get("Content-Length") or 0)
        read = 0
        while True:
            block = response.read(64 * 1024)
            if not block:
                break
            out.write(block)
            read += len(block)
            bar = f"{read / 1e6:5.1f} MB" if not total else f"{read / total:6.1%}"
            sys.stdout.write(f"\r  {os.path.basename(local):<36} {bar}")
            sys.stdout.flush()

    os.rename(tmp, local)  # only becomes the real file once fully written
    print()
    return True


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Fetching {len(PAPERS)} papers into {OUT_DIR}/")

    fetched = 0
    for arxiv_id, name in PAPERS:
        local = os.path.join(OUT_DIR, name)
        try:
            if download(arxiv_id, local):
                fetched += 1
                time.sleep(DELAY)
        except (urllib.error.URLError, TimeoutError) as exc:
            print(f"\n  !! {name} failed ({exc}) — re-run to retry just this one")
            if os.path.exists(local + ".part"):
                os.remove(local + ".part")

    have = len([n for _, n in PAPERS if os.path.exists(os.path.join(OUT_DIR, n))])
    size = sum(os.path.getsize(os.path.join(OUT_DIR, n))
               for _, n in PAPERS if os.path.exists(os.path.join(OUT_DIR, n)))
    print(f"\n{have}/{len(PAPERS)} papers on disk ({size / 1e6:.0f} MB), {fetched} newly downloaded.")
    if have < len(PAPERS):
        print("Re-run `python get_data.py` to retry the missing ones.")
        return 1
    print("Next: python ingest.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
