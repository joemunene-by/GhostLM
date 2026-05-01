"""GhostLM interactive chat — multi-turn terminal interface using chat role markers.

For chat-tuned checkpoints (Phase 5+), this loads the model with the expanded
vocabulary (50264) and feeds the conversation history through
``tokenizer.format_chat_prompt`` so each generation begins from a
``<|ghost_assistant|>`` token. Generation stops the moment a ``<|ghost_end|>``
token is sampled.

Pretrain-only checkpoints (Phase 4 ghost-small, vocab 50261) still work:
``--no-chat-format`` falls back to the original raw-completion behavior.
"""

import argparse
import sys
from dataclasses import fields
from pathlib import Path

import torch
import torch.nn.functional as F

from ghostlm.config import GhostLMConfig
from ghostlm.model import GhostLM
from ghostlm.tokenizer import GhostTokenizer


def parse_args():
    """Parse CLI arguments for the chat REPL."""
    p = argparse.ArgumentParser(description="GhostLM Interactive Chat")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to checkpoint .pt file")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=50, help="0 disables top-k")
    p.add_argument("--top-p", type=float, default=0.95,
                   help="Nucleus sampling cutoff. Set 1.0 to disable.")
    p.add_argument("--max-tokens", type=int, default=300,
                   help="Max tokens per assistant reply")
    p.add_argument("--device", default="auto")
    p.add_argument("--no-chat-format", action="store_true",
                   help="Disable chat role markers — use raw completion mode")
    p.add_argument("--system", type=str, default=None,
                   help="Optional system prefix prepended as the first user turn")
    return p.parse_args()


def resolve_device(arg: str) -> str:
    """Pick a device from --device, honoring 'auto'."""
    if arg != "auto":
        return arg
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(checkpoint_path: str, device: str) -> tuple:
    """Load a GhostLM checkpoint, returning (model, config)."""
    if checkpoint_path is None or not Path(checkpoint_path).exists():
        print("  No checkpoint provided — using random ghost-tiny weights.")
        config = GhostLMConfig.from_preset("ghost-tiny")
        config.vocab_size = 50264
        config.context_length = 128
        model = GhostLM(config)
        model.eval()
        return model.to(device), config

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    saved_config = ckpt["config"]
    if isinstance(saved_config, dict):
        config = GhostLMConfig(**{
            f.name: saved_config[f.name]
            for f in fields(GhostLMConfig)
            if f.name in saved_config
        })
    else:
        config = saved_config

    model = GhostLM(config)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt["model"]
    model.load_state_dict(state, strict=False)
    model.eval()
    return model.to(device), config


def sample_next(logits: torch.Tensor, temperature: float, top_k: int, top_p: float) -> int:
    """Sample one token id from logits using temperature + optional top-k / top-p."""
    logits = logits / max(temperature, 1e-6)
    if top_k and top_k > 0:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[..., -1:]] = float("-inf")
    if top_p and top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cum = probs.cumsum(dim=-1)
        cutoff = cum > top_p
        # Always keep the top token.
        cutoff[..., 0] = False
        sorted_logits[cutoff] = float("-inf")
        logits = torch.full_like(logits, float("-inf")).scatter(
            -1, sorted_idx, sorted_logits
        )
    probs = F.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


def generate_until_end(
    model: GhostLM,
    prompt_ids: list,
    *,
    end_id: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    device: str,
) -> list:
    """Greedy-or-sampled generation that stops the moment ``end_id`` is sampled.

    Returns only the *newly generated* token ids (excluding the prompt).
    """
    ids = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
    new_ids: list = []
    ctx = model.config.context_length
    with torch.no_grad():
        for _ in range(max_new_tokens):
            cond = ids[:, -ctx:]
            logits, _ = model(cond)
            next_logits = logits[:, -1, :].squeeze(0).clone()
            tok = sample_next(next_logits, temperature, top_k, top_p)
            if tok == end_id:
                break
            new_ids.append(tok)
            ids = torch.cat([ids, torch.tensor([[tok]], device=device)], dim=1)
    return new_ids


def chat_loop_chat_format(model, tokenizer: GhostTokenizer, args, device: str) -> None:
    """Multi-turn loop using <|ghost_user|>/<|ghost_assistant|>/<|ghost_end|>."""
    history: list = []
    if args.system:
        history.append({"role": "user", "content": args.system})
        history.append({
            "role": "assistant",
            "content": "Got it — I'll keep that in mind.",
        })
    end_id = tokenizer._special_tokens[tokenizer.END]
    print()
    print("Chat mode (chat-tuned). Commands: 'quit', 'exit', 'reset'.")
    print()
    while True:
        try:
            user_input = input("You > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
            return
        if user_input.lower() in ("quit", "exit"):
            return
        if user_input.lower() == "reset":
            history = []
            print("(history cleared)")
            continue
        if not user_input:
            continue

        history.append({"role": "user", "content": user_input})
        prompt_ids = tokenizer.format_chat_prompt(history)

        # Trim history if the prompt overflows context. Keep system/identity
        # turn (index 0) if there is one and drop oldest turns until the prompt
        # fits.
        ctx_budget = model.config.context_length - args.max_tokens - 4
        while len(prompt_ids) > ctx_budget and len(history) > 1:
            # Drop the oldest user/assistant pair (preserve system if present).
            drop_from = 2 if args.system else 0
            if len(history) > drop_from + 1:
                del history[drop_from:drop_from + 2]
                prompt_ids = tokenizer.format_chat_prompt(history)
            else:
                break

        new_ids = generate_until_end(
            model,
            prompt_ids,
            end_id=end_id,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k if args.top_k > 0 else 0,
            top_p=args.top_p,
            device=device,
        )
        reply = tokenizer.decode(new_ids).strip()
        history.append({"role": "assistant", "content": reply})
        print(f"\nGhostLM > {reply}\n")


def chat_loop_completion(model, tokenizer: GhostTokenizer, args, device: str) -> None:
    """Original raw-completion mode for non-chat-tuned checkpoints."""
    print()
    print("Completion mode (no chat formatting). Commands: 'quit', 'exit'.")
    print()
    while True:
        try:
            user_input = input("Ghost > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
            return
        if user_input.lower() in ("quit", "exit"):
            return
        if not user_input:
            continue
        ids = tokenizer.encode(user_input)
        x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
        with torch.no_grad():
            out = model.generate(
                x,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k if args.top_k > 0 else None,
            )
        text = tokenizer.decode(out[0].tolist())
        if text.startswith(user_input):
            text = text[len(user_input):]
        print(f"\nGhostLM > {text.strip()}\n")


def main():
    """Run the chat REPL."""
    args = parse_args()
    device = resolve_device(args.device)
    model, config = load_model(args.checkpoint, device)
    tokenizer = GhostTokenizer()

    print()
    print("╔══════════════════════════════════════╗")
    print("║         GhostLM Chat                 ║")
    print("║   Cybersecurity Language Model       ║")
    print("╚══════════════════════════════════════╝")
    print(f"  Device: {device}")
    print(f"  Vocab:  model={config.vocab_size}, tokenizer={tokenizer.vocab_size}")
    print(f"  Params: {model.num_params():,}")

    use_chat = not args.no_chat_format and config.vocab_size >= tokenizer.vocab_size
    if not use_chat and not args.no_chat_format:
        print("  (model vocab smaller than tokenizer — falling back to completion mode)")

    if use_chat:
        chat_loop_chat_format(model, tokenizer, args, device)
    else:
        chat_loop_completion(model, tokenizer, args, device)


if __name__ == "__main__":
    main()
