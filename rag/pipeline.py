"""The end-to-end assistant: retrieve, then generate an answer grounded in what
was retrieved.

`Assistant` owns the three pieces (embedder, store, generator) and exposes two
entry points. `retrieve()` is the retrieval half on its own, which is what
`evaluate.py` measures and what `ask.py --show-only` prints. `answer()` runs the
whole thing. The generator is created lazily, so retrieval-only work never pays
the cost of loading a language model.
"""
import re
from dataclasses import dataclass, field

from .embeddings import Embedder
from .generate import build_prompt, make_generator
from .store import VectorStore


@dataclass
class Answer:
    question: str
    text: str
    hits: list = field(default_factory=list)

    def cited(self):
        """The 1-based passage numbers the model actually referenced.

        Useful as a cheap grounding check: an answer that cites nothing while
        claiming facts is a red flag worth surfacing to the user.
        """
        numbers = {int(n) for n in re.findall(r"\[(\d+)\]", self.text)}
        return sorted(n for n in numbers if 1 <= n <= len(self.hits))

    def format(self, show_passages=False):
        if not self.hits:
            return self.text
        lines = [self.text, ""]
        cited = set(self.cited())
        lines.append("Sources:")
        for number, hit in enumerate(self.hits, start=1):
            mark = "*" if number in cited else " "
            lines.append(f" {mark}[{number}] {hit['source']} p.{hit['page']}  (similarity {hit['score']:.3f})")
            if show_passages:
                snippet = hit["text"].replace("\n", " ")
                lines.append(f"      {snippet[:300]}{'...' if len(snippet) > 300 else ''}")
        if cited:
            lines.append(" * = cited in the answer above")
        return "\n".join(lines)


def diversify(hits, k, max_per_source=3):
    """Trim an over-fetched hit list to k, capping how many come from one document.

    Without this, a question whose wording happens to match one paper's phrasing
    fills every slot from that paper, and the model never sees that three other
    documents also discuss it. Over-fetching and then capping costs nothing extra
    at query time — the vector search already returned the candidates.
    """
    kept, per_source = [], {}
    for hit in hits:
        if per_source.get(hit["source"], 0) >= max_per_source:
            continue
        kept.append(hit)
        per_source[hit["source"]] = per_source.get(hit["source"], 0) + 1
        if len(kept) == k:
            break

    # If the cap was so tight that we came up short, top up with what was skipped.
    if len(kept) < k:
        already = {(h["source"], h["page"], h["text"][:60]) for h in kept}
        for hit in hits:
            key = (hit["source"], hit["page"], hit["text"][:60])
            if key not in already:
                kept.append(hit)
                already.add(key)
            if len(kept) == k:
                break
    return kept


class Assistant:
    def __init__(self, persist_dir=None, embed_model=None, device=None,
                 backend="hf", model_name=None, max_new_tokens=400,
                 max_context_chars=6000, max_per_source=3):
        store_kwargs = {"persist_dir": persist_dir} if persist_dir else {}
        self.store = VectorStore(**store_kwargs)
        if not len(self.store):
            raise RuntimeError(
                f"the vector store at {self.store.persist_dir} is empty — "
                "run `python ingest.py` first."
            )

        embedder_kwargs = {"device": device}
        if embed_model:
            embedder_kwargs["model_name"] = embed_model
        self.embedder = Embedder(**embedder_kwargs)

        self.backend = backend
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.max_context_chars = max_context_chars
        self.max_per_source = max_per_source
        self._generator = None

    @property
    def generator(self):
        if self._generator is None:
            self._generator = make_generator(
                self.backend, self.model_name,
                device=self.device, max_new_tokens=self.max_new_tokens,
            )
        return self._generator

    def retrieve(self, question, k=5, source=None):
        vector = self.embedder.embed_query(question)
        # Over-fetch so diversify() has candidates to choose between.
        candidates = self.store.search(vector, k=max(k * 4, k), source=source)
        return diversify(candidates, k, self.max_per_source)

    def answer(self, question, k=5, source=None):
        hits = self.retrieve(question, k=k, source=source)
        if not hits:
            return Answer(question, "The provided documents do not cover this.", [])
        system, user = build_prompt(question, hits, self.max_context_chars)
        return Answer(question, self.generator(system, user), hits)
