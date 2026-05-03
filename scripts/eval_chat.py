#!/usr/bin/env python3
"""Held-out chat eval — run a fixed prompt set through a chat-tuned checkpoint.

Generates a transcript for each prompt without sampling randomness (top_k=1
greedy by default for reproducibility) so we can compare side-by-side with
the underlying pretrain checkpoint or future chat-tuned versions.
"""

import argparse
from dataclasses import fields
from pathlib import Path

import torch

from ghostlm.config import GhostLMConfig
from ghostlm.model import GhostLM
from ghostlm.tokenizer import GhostTokenizer, load_tokenizer
from scripts.chat import generate_until_end, resolve_device


PROMPTS = [
    # --- Small-talk / identity ---
    "hi",
    "what are you",
    "who made you",
    "are you ChatGPT",
    "what can you do",

    # --- OOD refusals (model should redirect) ---
    "what's the weather",
    "tell me a joke",
    "give me a recipe for lasagna",
    "what stocks should I buy",

    # --- Cybersec basics (should produce coherent answers) ---
    "what is XSS",
    "what is SQL injection",
    "what is a CVE",
    "what is MITRE ATT&CK",
    "explain buffer overflow",
    "what is a CTF",

    # --- Specific items (testing memorization / generalization) ---
    "explain CVE-2021-44228",
    "what is T1059",
    "what is CAPEC-66",
    "explain the EternalBlue exploit",

    # --- Help / open-ended ---
    "where do I start with cybersecurity",
    "I want to learn pentesting",
    "what's the best ctf platform for beginners",

    # --- Limits ---
    "are you accurate",
    "should I trust your answers",

    # --- Edge cases ---
    "?",
    "thanks",
    "bye",
]


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    p = argparse.ArgumentParser(description="GhostLM chat eval — held-out prompt set")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--max-tokens", type=int, default=200)
    p.add_argument("--repetition-penalty", type=float, default=1.25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tokenizer", default=None,
                   help="Optional path to a v0.5 tokenizer.json")
    p.add_argument("--out", default=None,
                   help="Optional path to write transcript to (defaults to stdout)")
    return p.parse_args()


def load_model(ckpt_path: str, device: str):
    """Load a GhostLM checkpoint into eval mode."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg_raw = ckpt["config"]
    if isinstance(cfg_raw, dict):
        cfg = GhostLMConfig(**{
            f.name: cfg_raw[f.name]
            for f in fields(GhostLMConfig)
            if f.name in cfg_raw
        })
    else:
        cfg = cfg_raw
    model = GhostLM(cfg)
    state = ckpt.get("model_state_dict", ckpt.get("model"))
    model.load_state_dict(state, strict=False)
    model.eval()
    return model.to(device), cfg


def main() -> None:
    """Run all PROMPTS through the model and print transcripts."""
    args = parse_args()
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    model, cfg = load_model(args.checkpoint, device)
    tokenizer = load_tokenizer(args.tokenizer) if args.tokenizer else GhostTokenizer()
    end_id = tokenizer._special_tokens[tokenizer.END]

    use_chat = cfg.vocab_size >= tokenizer.vocab_size
    out_lines = []
    out_lines.append(f"# Eval — checkpoint: {args.checkpoint}")
    out_lines.append(f"# device={device} chat_format={use_chat} "
                     f"temp={args.temperature} top_k={args.top_k} top_p={args.top_p} seed={args.seed}")
    out_lines.append("")

    for prompt in PROMPTS:
        if use_chat:
            ids = tokenizer.format_chat_prompt([{"role": "user", "content": prompt}])
            new = generate_until_end(
                model, ids, end_id=end_id, max_new_tokens=args.max_tokens,
                temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
                device=device, repetition_penalty=args.repetition_penalty,
            )
            reply = tokenizer.decode(new).strip()
        else:
            ids = tokenizer.encode(prompt)
            x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
            with torch.no_grad():
                out = model.generate(x, max_new_tokens=args.max_tokens,
                                     temperature=args.temperature,
                                     top_k=args.top_k if args.top_k > 0 else None)
            text = tokenizer.decode(out[0].tolist())
            reply = text[len(prompt):].strip() if text.startswith(prompt) else text.strip()

        out_lines.append(f"USER : {prompt}")
        out_lines.append(f"GHOST: {reply}")
        out_lines.append("")

    text = "\n".join(out_lines)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"Wrote transcript to {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
