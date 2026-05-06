#!/usr/bin/env python3
"""Side-by-side completion comparison across multiple chat checkpoints.

Cross-bench numbers are useful but anonymous. This script renders the
same prompts through any number of chat checkpoints so the qualitative
read sits next to the bench numbers. Useful at every release inflection
point for "is the model actually getting smarter" intuition.

Default prompts are five fact-recall probes (CVE / CWE / MITRE /
crypto / web) where the v0.9 cross-bench result predicts an improvement
relative to v0.7. Use ``--prompts-file`` to pass a custom JSONL
(``{"prompt": "..."}`` per line).

Output: one block per prompt × checkpoint to stdout, plus a JSON
summary if ``--out-json`` is set.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path

import torch

from ghostlm.config import GhostLMConfig
from ghostlm.model import GhostLM
from ghostlm.tokenizer import GhostTokenizer, load_tokenizer


DEFAULT_PROMPTS = [
    "What is the CVE identifier for EternalBlue?",
    "Which CWE category covers SQL injection vulnerabilities?",
    "What MITRE ATT&CK technique describes credential dumping via LSASS?",
    "What does the TLS 1.3 ChaCha20-Poly1305 cipher suite provide that AES-GCM does not?",
    "How does SameSite=Strict on a session cookie defeat CSRF attacks?",
]


def parse_args() -> argparse.Namespace:
    """CLI args."""
    p = argparse.ArgumentParser(description="Side-by-side chat completion compare")
    p.add_argument("--checkpoints", nargs="+", required=True,
                   help="Format: label1=path1 label2=path2 ... ")
    p.add_argument("--tokenizer", default=None,
                   help="Optional tokenizer path; default = bundled GhostTokenizer")
    p.add_argument("--prompts-file", default=None,
                   help="JSONL with {\"prompt\": \"...\"} per line")
    p.add_argument("--device", default="mps")
    p.add_argument("--max-tokens", type=int, default=180)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-json", default=None)
    return p.parse_args()


def parse_checkpoints(specs: list[str]) -> list[tuple[str, str]]:
    """Parse 'label=path' specs into (label, path) tuples."""
    out = []
    for s in specs:
        if "=" not in s:
            raise SystemExit(f"--checkpoints expects label=path, got: {s}")
        label, path = s.split("=", 1)
        out.append((label.strip(), path.strip()))
    return out


def load_model(path: str, device: str) -> tuple[GhostLM, GhostLMConfig]:
    """Load a checkpoint with its embedded config."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    saved = ckpt["config"]
    cfg = GhostLMConfig(**{
        f.name: saved[f.name] for f in fields(GhostLMConfig) if f.name in saved
    })
    cfg.device = device
    model = GhostLM(cfg).to(device)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt["model"]
    model.load_state_dict(state, strict=False)
    model.eval()
    return model, cfg


def chat_completion(model: GhostLM, tokenizer: GhostTokenizer,
                    prompt: str, *, device: str,
                    max_tokens: int, temperature: float,
                    top_k: int) -> str:
    """One-shot chat completion for `prompt`."""
    ids = tokenizer.format_chat_prompt([{"role": "user", "content": prompt}])
    x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        out_ids = model.generate(
            x,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k if top_k > 0 else None,
        )
    new_ids = out_ids[0, len(ids):].tolist()
    text = tokenizer.decode(new_ids)
    # Trim at the end-of-turn marker if present
    return text.split("<|ghost_end|>")[0].strip()


def main() -> None:
    """Run the comparison."""
    args = parse_args()
    torch.manual_seed(args.seed)

    pairs = parse_checkpoints(args.checkpoints)
    tokenizer = load_tokenizer(args.tokenizer) if args.tokenizer else GhostTokenizer()

    if args.prompts_file:
        prompts = []
        with open(args.prompts_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("prompt"):
                    prompts.append(rec["prompt"])
    else:
        prompts = list(DEFAULT_PROMPTS)
    print(f"Comparing {len(pairs)} checkpoints across {len(prompts)} prompts.\n")

    results: list[dict] = []
    for prompt_idx, prompt in enumerate(prompts):
        print(f"==== prompt {prompt_idx + 1}/{len(prompts)} ====")
        print(f"USER: {prompt}\n")
        prompt_record: dict = {"prompt": prompt, "completions": {}}
        for label, path in pairs:
            model, _ = load_model(path, args.device)
            completion = chat_completion(
                model, tokenizer, prompt,
                device=args.device,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
            )
            print(f"--- {label} ({path}) ---")
            print(completion)
            print()
            prompt_record["completions"][label] = completion
            del model
            if args.device == "mps":
                torch.mps.empty_cache()
        results.append(prompt_record)
        print()

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({
            "checkpoints": [{"label": l, "path": p} for l, p in pairs],
            "sampling": {
                "temperature": args.temperature,
                "top_k": args.top_k,
                "max_tokens": args.max_tokens,
                "seed": args.seed,
            },
            "results": results,
        }, indent=2))
        print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
