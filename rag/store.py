"""Persistent vector store on top of ChromaDB.

Chroma was chosen over FAISS for one practical reason: it keeps the vectors, the
chunk text and the metadata (source file, page number) together in one on-disk
collection. With FAISS you get an index of vectors and must maintain a parallel
sidecar mapping row -> chunk yourself, which is an easy thing to let drift out of
sync. Here `ingest.py` writes the store and `ask.py` opens it read-only.

Embeddings are computed by our own Embedder and passed in explicitly, rather than
letting Chroma manage an embedding function. That keeps the choice of model in one
place and makes it impossible to index with one model and query with another
without noticing.
"""
import os
import shutil

DEFAULT_DIR = os.path.join("data", "chroma")
COLLECTION = "documents"


class VectorStore:
    def __init__(self, persist_dir=DEFAULT_DIR, collection=COLLECTION):
        import chromadb
        from chromadb.config import Settings

        self.persist_dir = persist_dir
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )
        self.collection = self.client.get_or_create_collection(
            name=collection,
            # Our vectors are L2-normalised, so cosine distance is just 1 - dot.
            metadata={"hnsw:space": "cosine"},
        )

    def __len__(self):
        return self.collection.count()

    def add(self, chunks, vectors, batch_size=512):
        """Insert chunks and their vectors. Ids are stable, so re-ingesting the
        same document updates in place instead of duplicating it."""
        if len(chunks) != len(vectors):
            raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")

        for start in range(0, len(chunks), batch_size):
            batch = chunks[start:start + batch_size]
            self.collection.upsert(
                ids=[c.id for c in batch],
                documents=[c.text for c in batch],
                metadatas=[c.as_metadata() for c in batch],
                embeddings=vectors[start:start + batch_size].tolist(),
            )

    def search(self, vector, k=5, source=None):
        """Return the k nearest chunks, optionally restricted to one source file."""
        result = self.collection.query(
            query_embeddings=[vector.tolist()],
            n_results=k,
            where={"source": source} if source else None,
            include=["documents", "metadatas", "distances"],
        )

        hits = []
        for text, metadata, distance in zip(
            result["documents"][0], result["metadatas"][0], result["distances"][0]
        ):
            hits.append({
                "text": text,
                "source": metadata["source"],
                "page": metadata["page"],
                # Chroma reports cosine *distance*; similarity is the intuitive number.
                "score": 1.0 - float(distance),
            })
        return hits

    def sources(self):
        """Distinct source filenames currently indexed."""
        if not len(self):
            return []
        metadatas = self.collection.get(include=["metadatas"])["metadatas"]
        return sorted({m["source"] for m in metadatas})

    @staticmethod
    def wipe(persist_dir=DEFAULT_DIR):
        if os.path.isdir(persist_dir):
            shutil.rmtree(persist_dir)
