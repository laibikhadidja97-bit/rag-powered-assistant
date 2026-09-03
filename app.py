"""A small Gradio front end for the assistant.

    python app.py                       # http://127.0.0.1:7860
    python app.py --backend ollama --model llama3
    python app.py --share               # temporary public link

The UI shows the retrieved passages next to the answer on purpose. The whole point
of RAG is that an answer is checkable: being able to read the source passage and
the page it came from is what separates this from asking a chatbot and hoping.
"""
import argparse
from html import escape

import gradio as gr

from rag.embeddings import DEFAULT_MODEL
from rag.generate import BACKENDS
from rag.pipeline import Assistant
from rag.store import DEFAULT_DIR

EXAMPLES = [
    "What is multi-head attention and why is it better than a single attention head?",
    "What are the two pre-training objectives used by BERT?",
    "How does LoRA reduce the number of trainable parameters?",
    "What is the difference between RAG-Sequence and RAG-Token?",
    "What are the default beta1 and beta2 values for Adam?",
    "How do BERT and GPT-3 differ in how they are pre-trained?",
]

CSS = """
.passage { border-left: 3px solid #888; padding-left: 12px; margin-bottom: 14px; }
footer { display: none !important; }
"""


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store", default=DEFAULT_DIR)
    parser.add_argument("--embed-model", default=DEFAULT_MODEL)
    parser.add_argument("--backend", default="hf", choices=sorted(BACKENDS))
    parser.add_argument("--model", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=400)
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()


def render_passages(hits, cited):
    """Render hits as HTML. Passage text comes straight out of a PDF, so it is
    escaped rather than trusted — a stray "<" in a formula would otherwise eat the
    rest of the panel."""
    if not hits:
        return "_Nothing retrieved._"

    parts = []
    for number, hit in enumerate(hits, start=1):
        mark = " ✅ cited" if number in cited else ""
        text = escape(hit["text"].replace("\n", " "))
        parts.append(
            f"<div class='passage'>"
            f"<b>[{number}] {escape(hit['source'])} &middot; page {hit['page']}</b>"
            f" &nbsp; <code>similarity {hit['score']:.3f}</code>{mark}<br>"
            f"<small>{text[:900]}{'&hellip;' if len(text) > 900 else ''}</small>"
            f"</div>"
        )
    return "\n".join(parts)


def build_ui(assistant, model_label):
    sources = assistant.store.sources()

    def respond(question, k, source):
        question = (question or "").strip()
        if not question:
            return "Ask something first.", ""
        chosen = None if source in (None, "All documents") else source
        answer = assistant.answer(question, k=int(k), source=chosen)
        cited = set(answer.cited())
        warning = ""
        if not cited and "do not cover this" not in answer.text.lower():
            warning = ("\n\n> ⚠️ The model did not cite any passage — treat this "
                       "answer with suspicion and check the sources below.")
        return answer.text + warning, render_passages(answer.hits, cited)

    with gr.Blocks(title="RAG-Powered Assistant", css=CSS) as demo:
        gr.Markdown(
            "# 📚 RAG-Powered Assistant\n"
            f"Answers grounded in **{len(assistant.store)} chunks** from "
            f"**{len(sources)} documents**, retrieved with `{assistant.embedder.model_name}` "
            f"and answered by `{model_label}`. Every answer cites the passages it used."
        )

        with gr.Row():
            with gr.Column(scale=3):
                question = gr.Textbox(
                    label="Question", lines=2, autofocus=True,
                    placeholder="Ask anything about the indexed documents…",
                )
            with gr.Column(scale=1):
                k = gr.Slider(1, 10, value=5, step=1, label="Passages to retrieve (k)")
                source = gr.Dropdown(
                    ["All documents"] + sources, value="All documents",
                    label="Search within",
                )

        ask = gr.Button("Ask", variant="primary")
        answer_box = gr.Markdown(label="Answer")
        with gr.Accordion("Retrieved passages", open=True):
            passages_box = gr.HTML()

        gr.Examples(examples=EXAMPLES, inputs=question, label="Try one of these")

        for trigger in (ask.click, question.submit):
            trigger(respond, inputs=[question, k, source],
                    outputs=[answer_box, passages_box])

    return demo


def main():
    args = parse_args()
    print("Loading assistant…")
    assistant = Assistant(
        persist_dir=args.store, embed_model=args.embed_model, device=args.device,
        backend=args.backend, model_name=args.model, max_new_tokens=args.max_new_tokens,
    )
    model_label = args.model or BACKENDS[args.backend][1]
    # Warm the generator now so the first question is not the one that pays the
    # multi-second model load.
    print(f"Loading generator ({args.backend} / {model_label})…")
    _ = assistant.generator

    build_ui(assistant, model_label).launch(
        server_port=args.port, share=args.share, show_api=False,
    )


if __name__ == "__main__":
    main()
