"""Model loading, retrieval and streamed generation for the GhostLM web app."""

from __future__ import annotations

import json
import re
import subprocess
import threading
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Iterator, List, Optional

import torch

from ghostlm.config import GhostLMConfig
from ghostlm.model import GhostLM
from ghostlm.rag import HybridRetriever, extract_ids
from ghostlm.tokenizer import GhostTokenizer

# Generation runs on the CPU so the app never competes with training for the GPU.
DEVICE = "cpu"
STOP_STRINGS = ("\nQuestion:", "\n\nQuestion", "<|ghost_eos|>")


@dataclass
class ModelSpec:
    key: str
    label: str
    source: str  # "hf:<repo>" or a local checkpoint path
    description: str
    min_free_memory: int = 15

    def available(self) -> bool:
        return self.source.startswith("hf:") or Path(self.source).exists()


def default_models(root: Path) -> List[ModelSpec]:
    return [
        ModelSpec("ghost-small-gen", "ghost-small-gen (45M)", "hf:Ghostgim/ghost-small-gen",
                  "Published 45M generalist. Small and fast; leans on retrieved sources."),
        ModelSpec("ghost-base-live", "ghost-base (in training)",
                  str(root / "checkpoints/ghost_base_mac/best_model.pt"),
                  "Best checkpoint of the running 349M pretrain. Improves as training goes.",
                  min_free_memory=25),
    ]


def free_memory_percent() -> Optional[int]:
    out = subprocess.run(["memory_pressure"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if "free percentage" in line:
            return int(line.rsplit(" ", 1)[-1].rstrip("%"))
    return None


def _config_from_dict(raw: dict) -> GhostLMConfig:
    valid = {f.name for f in fields(GhostLMConfig)}
    cfg = GhostLMConfig(**{k: v for k, v in raw.items() if k in valid})
    cfg.device, cfg.dropout, cfg.gradient_checkpointing = DEVICE, 0.0, False
    return cfg


def load_model(spec: ModelSpec) -> tuple[GhostLM, dict]:
    if spec.source.startswith("hf:"):
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        repo = spec.source[3:]
        cfg = _config_from_dict(json.loads(Path(hf_hub_download(repo, "config.json")).read_text()))
        state = load_file(hf_hub_download(repo, "model.safetensors"))
        meta = {"step": None}
    else:
        ckpt = torch.load(spec.source, map_location="cpu", weights_only=False, mmap=True)
        cfg = _config_from_dict(ckpt["config"])
        state = ckpt["model_state_dict"]
        meta = {"step": ckpt.get("step"), "val_loss": ckpt.get("val_loss")}
    model = GhostLM(cfg)
    # Saved files may omit the tied lm_head; strict=False tolerates that.
    model.load_state_dict({k: v.float() for k, v in state.items()}, strict=False)
    model.lm_head.weight = model.token_embedding.weight
    return model.eval(), meta


def build_prompt(question: str, passages: List[dict], history: List[dict], passage_chars: int = 600) -> str:
    parts = []
    if passages:
        refs = "\n".join(f"[{i + 1}] ({p['source']} {p['ref']}) {_trim(p['text'], passage_chars)}"
                         for i, p in enumerate(passages))
        parts.append(f"Reference passages:\n{refs}\n")
    for turn in history[-4:]:
        parts.append(f"Question: {turn['user']}\nAnswer: {turn['assistant']}\n")
    parts.append(f"Question: {question}\nAnswer:")
    return "\n".join(parts)


def key_facts(passages: List[dict], max_sentences: int = 2) -> List[dict]:
    """Quoted opening sentences of each exact-ID match: factual by construction."""
    facts = []
    for i, p in enumerate(passages):
        if p.get("match") != "id":
            continue
        text = " ".join(p["text"].split())
        sentences = re.split(r"(?<=[.!?])\s+", text)
        facts.append({"cite": i + 1, "ref": p["ref"], "text": " ".join(sentences[:max_sentences])})
    return facts


def _trim(text: str, n: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[:n].rsplit(" ", 1)[0] + "..."


def _sample(logits: torch.Tensor, temperature: float, top_k: int, top_p: float,
            recent: List[int], repetition_penalty: float) -> int:
    logits = logits.clone()
    if repetition_penalty != 1.0 and recent:
        idx = torch.tensor(sorted(set(recent)))
        vals = logits[idx]
        logits[idx] = torch.where(vals > 0, vals / repetition_penalty, vals * repetition_penalty)
    if temperature <= 0:
        return int(logits.argmax())
    logits = logits / temperature
    if top_k:
        kth = torch.topk(logits, min(top_k, logits.numel())).values[-1]
        logits[logits < kth] = float("-inf")
    probs = torch.softmax(logits, dim=-1)
    if top_p < 1.0:
        sorted_p, order = torch.sort(probs, descending=True)
        cut = torch.cumsum(sorted_p, 0) - sorted_p > top_p
        sorted_p[cut] = 0
        probs = torch.zeros_like(probs).scatter(0, order, sorted_p)
    return int(torch.multinomial(probs / probs.sum(), 1))


class Engine:
    """Holds one loaded model at a time plus the shared retriever."""

    def __init__(self, root: Path, rag_dir: Optional[Path] = None):
        self.root = root
        self.models = {m.key: m for m in default_models(root)}
        self.tokenizer = GhostTokenizer()
        self.rag_dir = rag_dir or root / "data/rag"
        self._model: Optional[GhostLM] = None
        self._model_key: Optional[str] = None
        self._meta: dict = {}
        self._retriever: Optional[HybridRetriever] = None
        self._lock = threading.Lock()

    @property
    def retriever(self) -> Optional[HybridRetriever]:
        if self._retriever is None and (self.rag_dir / "index.npy").exists():
            self._retriever = HybridRetriever.load(self.rag_dir)
        return self._retriever

    def model_list(self) -> List[dict]:
        return [{"key": m.key, "label": m.label, "description": m.description,
                 "available": m.available(), "loaded": m.key == self._model_key,
                 "meta": self._meta if m.key == self._model_key else {}} for m in self.models.values()]

    def _ensure_model(self, key: str) -> None:
        if key == self._model_key:
            return
        spec = self.models[key]
        if not spec.available():
            raise ValueError(f"{spec.label} is not available yet")
        free = free_memory_percent()
        if free is not None and free < spec.min_free_memory:
            raise MemoryError(f"Only {free}% memory free; {spec.label} needs {spec.min_free_memory}%.")
        self._model, self._model_key = None, None
        self._model, self._meta = load_model(spec)
        self._model_key = key

    def retrieve(self, question: str, k: int = 4) -> List[dict]:
        r = self.retriever
        return r.search(question, k=k) if r else []

    def stream_answer(self, question: str, model_key: str, history: List[dict], use_retrieval: bool = True,
                      temperature: float = 0.7) -> Iterator[dict]:
        """Yield {"type": "tool"|"sources"|"token"|"done"} events."""
        with self._lock:
            self._ensure_model(model_key)
            model, tok = self._model, self.tokenizer
            ctx = model.config.context_length
            # A 512-token model cannot hold four full passages plus an answer.
            small = ctx <= 512
            k, passage_chars, max_new_tokens = (2, 320, 160) if small else (4, 600, 256)

            for ident in extract_ids(question):
                yield {"type": "tool", "name": "exact_lookup", "detail": ident}
            passages = self.retrieve(question, k=k) if use_retrieval else []
            yield {"type": "sources", "passages": [
                {"source": p["source"], "ref": p["ref"], "match": p["match"],
                 "score": round(p["score"], 3), "text": _trim(p["text"], 900)} for p in passages]}

            facts = key_facts(passages)
            if facts:
                yield {"type": "facts", "facts": facts}

            ids_in_question = extract_ids(question)
            # Base models continue text better than they answer questions; start the
            # answer on the asked-about entity so the continuation stays on topic.
            lead = f" {ids_in_question[0]} is" if len(ids_in_question) == 1 else ""
            prompt = build_prompt(question, passages, history[-2:] if small else history, passage_chars) + lead
            prompt_ids = tok.encode(prompt)
            prompt_ids = prompt_ids[-(ctx - max_new_tokens):]
            ids = torch.tensor([prompt_ids])
            past, new_ids, emitted = None, [], ""
            if lead:
                yield {"type": "token", "text": lead.strip()}
                emitted_lead = lead.strip()
            else:
                emitted_lead = ""
            eos = tok.eos_id
            with torch.no_grad():
                inp = ids
                for _ in range(max_new_tokens):
                    logits, _, past = model(inp, past_kv=past, use_cache=True)
                    nxt = _sample(logits[0, -1], temperature, 40, 0.95, new_ids[-64:], 1.15)
                    if nxt == eos:
                        break
                    new_ids.append(nxt)
                    text = tok.decode(new_ids)
                    stop = min((text.find(s) for s in STOP_STRINGS if s in text), default=-1)
                    if stop >= 0:
                        text = text[:stop]
                    if len(text) > len(emitted) and not text.endswith("�"):
                        yield {"type": "token", "text": text[len(emitted):]}
                        emitted = text
                    if stop >= 0 or len(prompt_ids) + len(new_ids) >= ctx:
                        break
                    inp = torch.tensor([[nxt]])
            yield {"type": "done", "answer": (emitted_lead + emitted).strip(), "model": model_key, "facts": facts}
