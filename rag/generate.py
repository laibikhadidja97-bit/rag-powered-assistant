"""The generation half of RAG: turn retrieved passages plus a question into an answer.

Three interchangeable backends are provided so the project runs anywhere:

  hf      local Hugging Face transformers (the default — no API key, no server)
  ollama  a running Ollama daemon on localhost
  openai  any OpenAI-compatible chat-completions endpoint

They all take the same `(system, user)` pair and return a string, so `pipeline.py`
never has to know which one is in use.

The default local model is Qwen2.5-1.5B-Instruct: about 3 GB, runs at usable speed
on an M-series laptop, and follows grounding instructions well enough to keep the
citations honest. Larger models are a flag away (`--model Qwen/Qwen2.5-7B-Instruct`)
and the prompt does not change.
"""
import os

DEFAULT_HF_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_OLLAMA_MODEL = "llama3"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = """You are a precise research assistant. Answer the user's question using ONLY the numbered context passages provided.

Rules:
- Ground every claim in the passages. Do not add outside knowledge.
- Cite the passages you used inline, like [1] or [2][3].
- If the passages do not contain the answer, say exactly: "The provided documents do not cover this." Do not guess.
- Be concise and specific. Prefer the numbers, names and definitions given in the passages over paraphrase."""

USER_TEMPLATE = """Context passages:
{context}

Question: {question}

Answer (with citations):"""


def format_context(hits, max_chars=6000):
    """Render retrieved hits as a numbered block, trimmed to a prompt budget.

    Hits arrive best-first, so truncating from the end drops the weakest evidence.
    The character budget keeps the prompt inside a small model's context window
    even when k is large.
    """
    blocks, used = [], 0
    for number, hit in enumerate(hits, start=1):
        block = f"[{number}] ({hit['source']}, p.{hit['page']})\n{hit['text']}"
        if used + len(block) > max_chars and blocks:
            break
        blocks.append(block)
        used += len(block)
    return "\n\n".join(blocks)


def build_prompt(question, hits, max_context_chars=6000):
    return SYSTEM_PROMPT, USER_TEMPLATE.format(
        context=format_context(hits, max_context_chars) or "(no passages retrieved)",
        question=question.strip(),
    )


class HuggingFaceGenerator:
    """Local generation through transformers. The model is loaded once and reused."""

    name = "hf"

    def __init__(self, model_name=DEFAULT_HF_MODEL, device=None, max_new_tokens=400):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from .embeddings import pick_device

        self.model_name = model_name
        self.device = pick_device(device)
        self.max_new_tokens = max_new_tokens
        # fp16 on accelerators, fp32 on CPU where half precision is slow and unstable.
        dtype = torch.float16 if self.device in ("mps", "cuda") else torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
        except TypeError:
            # transformers < 4.56 spells this argument torch_dtype.
            self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
        self.model.to(self.device)
        self.model.eval()

    def __call__(self, system, user):
        import torch

        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": user}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,  # greedy: grounded answers should be reproducible
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        # Slice off the prompt so only the freshly generated tokens are decoded.
        generated = output[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()


class OllamaGenerator:
    """Generation through a local Ollama daemon (`ollama serve`)."""

    name = "ollama"

    def __init__(self, model_name=DEFAULT_OLLAMA_MODEL, host=None, max_new_tokens=400):
        self.model_name = model_name
        self.host = (host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
        self.max_new_tokens = max_new_tokens

    def __call__(self, system, user):
        import json
        import urllib.error
        import urllib.request

        payload = json.dumps({
            "model": self.model_name,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": self.max_new_tokens},
        }).encode()

        request = urllib.request.Request(
            f"{self.host}/api/chat", data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                return json.load(response)["message"]["content"].strip()
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"could not reach Ollama at {self.host} ({exc}). "
                f"Start it with `ollama serve` and `ollama pull {self.model_name}`."
            ) from exc


class OpenAIGenerator:
    """Generation through any OpenAI-compatible chat-completions endpoint."""

    name = "openai"

    def __init__(self, model_name=DEFAULT_OPENAI_MODEL, max_new_tokens=400):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("pip install openai to use --backend openai") from exc
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("set OPENAI_API_KEY to use --backend openai")

        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.client = OpenAI()

    def __call__(self, system, user):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=0.0,
            max_tokens=self.max_new_tokens,
        )
        return response.choices[0].message.content.strip()


BACKENDS = {
    "hf": (HuggingFaceGenerator, DEFAULT_HF_MODEL),
    "ollama": (OllamaGenerator, DEFAULT_OLLAMA_MODEL),
    "openai": (OpenAIGenerator, DEFAULT_OPENAI_MODEL),
}


def make_generator(backend="hf", model_name=None, **kwargs):
    if backend not in BACKENDS:
        raise ValueError(f"unknown backend {backend!r}; choose from {sorted(BACKENDS)}")
    cls, default_model = BACKENDS[backend]
    if backend != "hf":
        kwargs.pop("device", None)  # only the local backend has a device
    return cls(model_name or default_model, **kwargs)
