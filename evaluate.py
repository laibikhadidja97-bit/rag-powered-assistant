"""Measure how well retrieval actually works, before blaming the language model.

When a RAG answer is wrong, the cause is usually retrieval, not generation: if the
right passage never reaches the prompt, no model can recover it. So this script
scores the retrieval half on its own against a small hand-labelled set — for each
question we know which paper must be found, and we check whether it appears among
the top-k chunks.

    python evaluate.py                 # retrieval metrics only (no LLM, fast)
    python evaluate.py --generate      # also check the answers stay grounded
    python evaluate.py -k 10 --verbose

Reported metrics:
  hit@k  fraction of questions where the expected document is in the top k
  MRR    mean reciprocal rank of the first correct document (1.0 = always first)

The --generate pass adds two grounding checks: in-scope answers must cite at least
one passage, and the out-of-scope questions (whose answer is nowhere in the corpus)
must trigger the refusal rather than an invented answer.
"""
import argparse
import time

from rag.embeddings import DEFAULT_MODEL
from rag.generate import BACKENDS
from rag.pipeline import Assistant
from rag.store import DEFAULT_DIR

REFUSAL = "do not cover this"

# (question, expected source file). "expected" is the document that genuinely
# answers the question; other documents may mention the topic in passing.
QUESTIONS = [
    ("What is multi-head attention and why is it used instead of a single attention head?",
     "attention-is-all-you-need.pdf"),
    ("What learning rate schedule and warmup did the original Transformer use?",
     "attention-is-all-you-need.pdf"),
    ("What are the two pre-training objectives used by BERT?", "bert.pdf"),
    ("How many parameters does the largest GPT-3 model have?",
     "gpt3-few-shot-learners.pdf"),
    ("What is the difference between RAG-Sequence and RAG-Token?",
     "rag-knowledge-intensive-nlp.pdf"),
    ("Which retriever does the RAG model use to fetch supporting documents?",
     "rag-knowledge-intensive-nlp.pdf"),
    ("What degradation problem do residual connections solve in very deep networks?",
     "resnet.pdf"),
    ("What are the default values of beta1 and beta2 in the Adam optimizer?",
     "adam-optimizer.pdf"),
    ("How does the Vision Transformer turn an image into a sequence of tokens?",
     "vision-transformer.pdf"),
    ("How does LoRA reduce the number of trainable parameters during fine-tuning?",
     "lora.pdf"),
    ("What is internal covariate shift?", "batch-normalization.pdf"),
    ("How was reinforcement learning from human feedback used to train InstructGPT?",
     "instructgpt.pdf"),
]

# Plausible-sounding questions the corpus genuinely cannot answer. A grounded
# assistant refuses these; an ungrounded one confabulates.
OUT_OF_SCOPE = [
    "What was the closing share price of Nvidia last Friday?",
    "What is the recommended dosage of amoxicillin for a child?",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--store", default=DEFAULT_DIR)
    parser.add_argument("--embed-model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--generate", action="store_true", help="also score the answers")
    parser.add_argument("--backend", default="hf", choices=sorted(BACKENDS))
    parser.add_argument("--model", default=None)
    parser.add_argument("--verbose", action="store_true", help="print every retrieved source")
    return parser.parse_args()


def rank_of(hits, expected):
    """1-based rank of the first chunk from the expected document, else None."""
    for rank, hit in enumerate(hits, start=1):
        if hit["source"] == expected:
            return rank
    return None


def main():
    args = parse_args()
    assistant = Assistant(
        persist_dir=args.store, embed_model=args.embed_model, device=args.device,
        backend=args.backend, model_name=args.model,
    )
    print(f"{len(assistant.store)} chunks from {len(assistant.store.sources())} documents")
    print(f"Retrieval: {args.embed_model} (k={args.k})\n")

    ranks, latencies = [], []
    print(f"{'rank':>4}  {'question':<62} expected")
    print("-" * 100)
    for question, expected in QUESTIONS:
        started = time.perf_counter()
        hits = assistant.retrieve(question, k=args.k)
        latencies.append(time.perf_counter() - started)
        rank = rank_of(hits, expected)
        ranks.append(rank)
        label = str(rank) if rank else "MISS"
        print(f"{label:>4}  {question[:60]:<62} {expected}")
        if args.verbose:
            for number, hit in enumerate(hits, start=1):
                print(f"        [{number}] {hit['source']} p.{hit['page']} ({hit['score']:.3f})")

    total = len(ranks)
    print("\n" + "=" * 100)
    for cutoff in sorted({c for c in (1, 3, args.k) if c <= args.k}):
        got = sum(1 for r in ranks if r is not None and r <= cutoff)
        print(f"  hit@{cutoff:<3} {got}/{total} = {got / total:.1%}")
    mrr = sum(1 / r for r in ranks if r) / total
    print(f"  MRR    {mrr:.3f}")
    print(f"  median retrieval latency {sorted(latencies)[len(latencies) // 2] * 1000:.0f} ms")

    if not args.generate:
        print("\n(Add --generate to also score answer grounding.)")
        return

    print(f"\nGeneration: {args.backend} / "
          f"{args.model or BACKENDS[args.backend][1]}")
    print("-" * 100)
    grounded = 0
    for question, _ in QUESTIONS:
        answer = assistant.answer(question, k=args.k)
        citations = answer.cited()
        grounded += bool(citations)
        marker = "ok  " if citations else "BARE"
        print(f"{marker}  {question[:60]:<62} cites {citations or '-'}")

    refused = 0
    for question in OUT_OF_SCOPE:
        answer = assistant.answer(question, k=args.k)
        did_refuse = REFUSAL in answer.text.lower()
        refused += did_refuse
        print(f"{'ok  ' if did_refuse else 'LEAK'}  {question[:60]:<62} "
              f"{'refused' if did_refuse else answer.text[:40] + '...'}")

    print("\n" + "=" * 100)
    print(f"  answers citing a passage      {grounded}/{len(QUESTIONS)}")
    print(f"  out-of-scope refusals         {refused}/{len(OUT_OF_SCOPE)}")


if __name__ == "__main__":
    main()
