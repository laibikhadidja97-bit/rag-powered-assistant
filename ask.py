"""Ask the assistant a question from the command line.

    python ask.py "What is multi-head attention?"
    python ask.py                      # interactive REPL
    python ask.py -k 8 --passages "How does LoRA reduce trainable parameters?"
    python ask.py --show-only "batch normalization"     # retrieval only, no LLM
    python ask.py --source lora.pdf "What rank did they use?"
    python ask.py --backend ollama --model llama3 "Compare BERT and GPT-3 pretraining"
"""
import argparse
import sys

from rag.embeddings import DEFAULT_MODEL
from rag.generate import BACKENDS
from rag.pipeline import Assistant
from rag.store import DEFAULT_DIR


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question", nargs="*", help="question; omit for an interactive REPL")
    parser.add_argument("-k", type=int, default=5, help="passages to retrieve (default 5)")
    parser.add_argument("--store", default=DEFAULT_DIR)
    parser.add_argument("--embed-model", default=DEFAULT_MODEL)
    parser.add_argument("--backend", default="hf", choices=sorted(BACKENDS))
    parser.add_argument("--model", default=None, help="generation model id for the backend")
    parser.add_argument("--device", default="auto", help="auto | cpu | mps | cuda")
    parser.add_argument("--source", default=None, help="restrict retrieval to one filename")
    parser.add_argument("--max-new-tokens", type=int, default=400)
    parser.add_argument("--passages", action="store_true", help="print retrieved text too")
    parser.add_argument("--show-only", action="store_true",
                        help="retrieve and print passages without calling the LLM")
    return parser.parse_args()


def show_retrieval(assistant, question, k, source):
    hits = assistant.retrieve(question, k=k, source=source)
    if not hits:
        print("No passages matched.")
        return
    for number, hit in enumerate(hits, start=1):
        snippet = hit["text"].replace("\n", " ")
        print(f"[{number}] {hit['source']} p.{hit['page']}  similarity {hit['score']:.3f}")
        print(f"    {snippet[:400]}{'...' if len(snippet) > 400 else ''}\n")


def main():
    args = parse_args()
    assistant = Assistant(
        persist_dir=args.store, embed_model=args.embed_model, device=args.device,
        backend=args.backend, model_name=args.model, max_new_tokens=args.max_new_tokens,
    )
    print(f"{len(assistant.store)} chunks from {len(assistant.store.sources())} documents "
          f"| retrieval {args.embed_model} | generation {args.backend}",
          file=sys.stderr)

    def handle(question):
        if args.show_only:
            show_retrieval(assistant, question, args.k, args.source)
            return
        answer = assistant.answer(question, k=args.k, source=args.source)
        print(answer.format(show_passages=args.passages))

    if args.question:
        handle(" ".join(args.question))
        return

    print("Interactive mode — ask a question, or Ctrl-D to quit.", file=sys.stderr)
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not question:
            continue
        if question in {"exit", "quit"}:
            return
        print()
        handle(question)


if __name__ == "__main__":
    main()
