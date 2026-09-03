"""Turn text into vectors with a local sentence-transformers model.

The default is BAAI/bge-small-en-v1.5: 33M parameters, 384 dimensions, ~130 MB on
disk, and consistently stronger on retrieval benchmarks than the more familiar
all-MiniLM-L6-v2 at a similar size. It runs comfortably on CPU and is fast on
Apple Silicon via MPS.

One detail that is easy to miss and costs real accuracy: BGE models are trained
asymmetrically. Passages are embedded bare, but *queries* must be prefixed with a
short instruction. Embedding a query as if it were a passage silently degrades
recall, so the prefix is applied here rather than left to the caller.
"""
import numpy as np

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"

# Instruction prefixes that a model expects on the query side only.
QUERY_PREFIXES = {
    "bge": "Represent this sentence for searching relevant passages: ",
    "e5": "query: ",
}


def pick_device(requested=None):
    import torch

    if requested and requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class Embedder:
    def __init__(self, model_name=DEFAULT_MODEL, device=None, batch_size=64):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.device = pick_device(device)
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name, device=self.device)
        self.query_prefix = self._query_prefix(model_name)

    @staticmethod
    def _query_prefix(model_name):
        lowered = model_name.lower()
        for family, prefix in QUERY_PREFIXES.items():
            # "intfloat/e5-*" and "*/bge-*" both need their own query instruction.
            if f"/{family}-" in lowered or lowered.startswith(f"{family}-"):
                return prefix
        return ""

    @property
    def dimension(self):
        # Renamed in sentence-transformers 6; support both so the project runs
        # against whichever version is already installed.
        getter = getattr(self.model, "get_embedding_dimension", None) or \
            self.model.get_sentence_embedding_dimension
        return getter()

    def _encode(self, texts, show_progress=False):
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,  # unit vectors => dot product == cosine
            show_progress_bar=show_progress,
        )
        return np.asarray(vectors, dtype=np.float32)

    def embed_documents(self, texts, show_progress=False):
        return self._encode(list(texts), show_progress=show_progress)

    def embed_query(self, text):
        return self._encode([self.query_prefix + text])[0]
