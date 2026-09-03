"""Build the vector index: load documents -> chunk -> embed -> store.

    python ingest.py                       # index data/docs into data/chroma
    python ingest.py --docs ~/my-notes     # index your own folder
    python ingest.py --rebuild             # start from an empty store

Ingestion is incremental by default. Chunk ids are derived from filename, page and
position, so re-running after adding a file updates the store in place rather than
duplicating everything. Use --rebuild after changing the chunk size or the
embedding model, since old vectors would otherwise sit in the store alongside new
ones that are not comparable to them.
"""
import argparse
import time

from rag.chunking import CHUNK_CHARS, OVERLAP_CHARS, chunk_pages
from rag.embeddings import DEFAULT_MODEL, Embedder
from rag.loaders import load_dir
from rag.store import DEFAULT_DIR, VectorStore


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--docs", default="data/docs", help="folder of .pdf/.txt/.md files")
    parser.add_argument("--store", default=DEFAULT_DIR, help="where to persist the index")
    parser.add_argument("--embed-model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="auto", help="auto | cpu | mps | cuda")
    parser.add_argument("--chunk-chars", type=int, default=CHUNK_CHARS)
    parser.add_argument("--overlap-chars", type=int, default=OVERLAP_CHARS)
    parser.add_argument("--rebuild", action="store_true", help="delete the store first")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.overlap_chars >= args.chunk_chars:
        raise SystemExit("--overlap-chars must be smaller than --chunk-chars")

    if args.rebuild:
        print(f"Removing existing store at {args.store}")
        VectorStore.wipe(args.store)

    print(f"Loading documents from {args.docs}/")
    started = time.perf_counter()
    pages = load_dir(args.docs)
    by_source = {}
    for page in pages:
        by_source[page.source] = by_source.get(page.source, 0) + 1
    for source, count in sorted(by_source.items()):
        print(f"  {source:<36} {count:>4} pages")
    print(f"  {len(pages)} pages from {len(by_source)} documents "
          f"in {time.perf_counter() - started:.1f}s")

    chunks = chunk_pages(pages, args.chunk_chars, args.overlap_chars)
    if not chunks:
        raise SystemExit("no text survived chunking — are these scanned image-only PDFs?")
    average = sum(len(c.text) for c in chunks) / len(chunks)
    print(f"\nChunked into {len(chunks)} chunks "
          f"(target {args.chunk_chars} chars, {args.overlap_chars} overlap, "
          f"mean {average:.0f} chars)")

    print(f"\nEmbedding with {args.embed_model}")
    embedder = Embedder(model_name=args.embed_model, device=args.device)
    print(f"  device={embedder.device}  dimension={embedder.dimension}")
    started = time.perf_counter()
    vectors = embedder.embed_documents([c.text for c in chunks], show_progress=True)
    elapsed = time.perf_counter() - started
    print(f"  {len(vectors)} vectors in {elapsed:.1f}s ({len(vectors) / elapsed:.0f} chunks/s)")

    print(f"\nWriting to {args.store}")
    store = VectorStore(persist_dir=args.store)
    store.add(chunks, vectors)
    print(f"  store now holds {len(store)} chunks from {len(store.sources())} documents")
    print("\nNext: python ask.py \"your question\"   (or: python app.py for the web UI)")


if __name__ == "__main__":
    main()
