"""Checkpoint-driven runner for GhostAgent.

Wires a trained GhostLM checkpoint (or random ghost-tiny weights for
smoke tests) into ``GhostAgent``, exposing a CLI:

    python -m ghostlm.agent --query "What is CVE-2017-0144?"
    python -m ghostlm.agent --query "..." --checkpoint runs/v09chat/best.pt
    python -m ghostlm.agent --query "..." --offline
    python -m ghostlm.agent --query "..." --max-iters 4 --json

If ``--checkpoint`` is omitted, the runner spins up random ghost-tiny
weights so the loop can be smoke-tested without a trained model. The
output will be gibberish but the loop's mechanics (system prompt,
bet-1 stop-sequence, tool dispatch, max-iterations safety) all
exercise correctly.

If a real checkpoint is provided that wasn't trained on the bet 1
SFT (e.g. v0.9 chat), the model won't reliably emit valid
``<|tool_call|>`` blocks; the loop will terminate via answer_emitted
on the first turn or hit max-iterations. That's the expected
behaviour today; ghost-base trained on synth_v1.jsonl will exercise
the loop properly.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import fields
from pathlib import Path
from typing import List, Tuple

from .messages import AgentMessage, MessageRole
from .runtime import GhostAgent, RuntimeConfig


# ---------------------------------------------------------------------------
# Checkpoint loading + generation
# ---------------------------------------------------------------------------


def load_model(checkpoint_path: str | None, device: str):
    """Load a GhostLM checkpoint into eval mode. Mirrors scripts/chat.py
    for consistency with the rest of the codebase."""
    import torch  # local import so the agent module is importable
    from ghostlm.config import GhostLMConfig
    from ghostlm.model import GhostLM

    if checkpoint_path is None or not Path(checkpoint_path).exists():
        config = GhostLMConfig.from_preset("ghost-tiny")
        config.vocab_size = 50264
        config.context_length = 128
        model = GhostLM(config)
        model.eval()
        return model.to(device), config, True

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    saved = ckpt["config"]
    if isinstance(saved, dict):
        config = GhostLMConfig(**{
            f.name: saved[f.name] for f in fields(GhostLMConfig)
            if f.name in saved
        })
    else:
        config = saved

    model = GhostLM(config)
    state = ckpt.get("model_state_dict") or ckpt["model"]
    model.load_state_dict(state, strict=False)
    model.eval()
    return model.to(device), config, False


def _resolve_device(arg: str) -> str:
    import torch
    if arg != "auto":
        return arg
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _history_to_chat_turns(history: List[AgentMessage]) -> List[dict]:
    """Translate AgentMessage history into the {role, content} list
    that ``GhostTokenizer.format_chat_prompt`` expects.

    Mapping:
      SYSTEM    -> a leading user turn carrying the system instructions,
                   followed by a synthetic assistant ack (mirrors
                   scripts/chat.py).
      USER      -> user
      ASSISTANT -> assistant
      TOOL      -> user, content is the raw <|tool_response|>...</...>
                   string already produced by AgentMessage.tool().
                   For a bet-1-trained model this is the expected
                   shape; v0.9 chat will see it as user input.
    """
    turns: List[dict] = []
    for m in history:
        if m.role == MessageRole.SYSTEM:
            turns.append({"role": "user", "content": m.content})
            turns.append({"role": "assistant",
                          "content": "Got it — I'll keep that in mind."})
        elif m.role == MessageRole.USER:
            turns.append({"role": "user", "content": m.content})
        elif m.role == MessageRole.ASSISTANT:
            turns.append({"role": "assistant", "content": m.content})
        elif m.role == MessageRole.TOOL:
            turns.append({"role": "user", "content": m.content})
    # The prompt must end on a user turn so format_chat_prompt can
    # append the assistant marker. If the last role is assistant
    # (shouldn't happen at the start of a generate-call) collapse it
    # into a no-op user nudge.
    if turns and turns[-1]["role"] == "assistant":
        turns.append({"role": "user", "content": "(continue)"})
    return turns


def make_generator(
    checkpoint_path: str | None,
    device_arg: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
):
    """Build a Generator callable wrapping a real GhostLM checkpoint.

    Loads the checkpoint and constructs a fresh tokenizer, then
    returns a closure that the GhostAgent runtime can drive. Returns
    ``(generator, is_random_weights)``. Use this from the CLI runner.

    For callers that already have a model + tokenizer in memory (e.g.
    the MCP server, which keeps the model loaded for non-agent
    direct-chat tools), use ``make_generator_from_loaded`` instead to
    avoid a second checkpoint load.
    """
    device = _resolve_device(device_arg)
    model, config, is_random = load_model(checkpoint_path, device)
    from ghostlm.tokenizer import GhostTokenizer
    tokenizer = GhostTokenizer()
    return make_generator_from_loaded(
        model, config, tokenizer, device,
        max_new_tokens, temperature, top_p, top_k, repetition_penalty,
    ), is_random


def make_generator_from_loaded(
    model,
    config,
    tokenizer,
    device: str,
    max_new_tokens: int = 384,
    temperature: float = 0.6,
    top_p: float = 0.9,
    top_k: int = 0,
    repetition_penalty: float = 1.15,
):
    """Build a Generator callable from an already-loaded model.

    Same body as ``make_generator`` but accepts the model / tokenizer /
    device that the caller has already loaded. Useful for the MCP
    server, tests, or any other place where the model is shared
    between the agent loop and other code paths.
    """
    import torch
    import torch.nn.functional as F

    end_id = tokenizer._special_tokens[tokenizer.END]

    def _sample_next(logits: torch.Tensor, prev_ids: list) -> int:
        if prev_ids and repetition_penalty != 1.0:
            for tok in set(prev_ids):
                if logits[tok] > 0:
                    logits[tok] = logits[tok] / repetition_penalty
                else:
                    logits[tok] = logits[tok] * repetition_penalty
        logits = logits / max(temperature, 1e-6)
        if top_k and top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[..., -1:]] = float("-inf")
        if top_p and top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            probs = F.softmax(sorted_logits, dim=-1)
            cum = probs.cumsum(dim=-1)
            cutoff = cum > top_p
            cutoff[..., 0] = False
            sorted_logits[cutoff] = float("-inf")
            logits = torch.full_like(logits, float("-inf")).scatter(
                -1, sorted_idx, sorted_logits,
            )
        probs = F.softmax(logits, dim=-1)
        return int(torch.multinomial(probs, num_samples=1).item())

    def generate(history: List[AgentMessage]) -> str:
        turns = _history_to_chat_turns(history)
        prompt_ids = tokenizer.format_chat_prompt(turns)

        # Trim if the prompt overflows context.
        ctx_budget = config.context_length - max_new_tokens - 4
        while len(prompt_ids) > ctx_budget and len(turns) > 2:
            del turns[0:2]  # drop oldest user/assistant pair
            prompt_ids = tokenizer.format_chat_prompt(turns)

        ids = torch.tensor(prompt_ids, dtype=torch.long,
                            device=device).unsqueeze(0)
        new_ids: list = []
        ctx = config.context_length
        # Stop on either the chat end token OR a decoded "<|/tool_call|>"
        # tail so the loop can dispatch immediately. Decoding every step
        # is cheap relative to the forward pass.
        with torch.no_grad():
            for _ in range(max_new_tokens):
                cond = ids[:, -ctx:]
                logits, _ = model(cond)
                next_logits = logits[:, -1, :].squeeze(0).clone()
                tok = _sample_next(next_logits, new_ids[-128:])
                if tok == end_id:
                    break
                new_ids.append(tok)
                ids = torch.cat(
                    [ids, torch.tensor([[tok]], device=device)], dim=1,
                )
                # Cheap suffix check: only decode the trailing window.
                if len(new_ids) >= 6:
                    tail = tokenizer.decode(new_ids[-32:])
                    if "<|/tool_call|>" in tail:
                        break

        return tokenizer.decode(new_ids).strip()

    return generate


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m ghostlm.agent",
        description=(
            "Run GhostAgent against a checkpoint. The agent decides "
            "which tool to call, executes it, feeds the result back, "
            "and emits a cite-tagged final answer."
        ),
    )
    p.add_argument("--query", required=True,
                    help="The user query to answer.")
    p.add_argument("--checkpoint", default=None,
                    help="Path to a GhostLM checkpoint .pt. "
                         "Omit for random ghost-tiny smoke test.")
    p.add_argument("--device", default="auto",
                    help="Device: auto/cuda/mps/cpu.")
    p.add_argument("--max-iters", type=int, default=8,
                    help="Cap on model -> tool round-trips.")
    p.add_argument("--max-new-tokens", type=int, default=384,
                    help="Generation budget per assistant message.")
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--repetition-penalty", type=float, default=1.15)
    p.add_argument("--system-prompt", default=None,
                    help="Override the default system prompt. "
                         "Pass empty string to disable.")
    p.add_argument("--offline", action="store_true",
                    help="Force tool backends to use offline caches "
                         "even if NVD is reachable.")
    p.add_argument("--json", action="store_true",
                    help="Emit the full trace as JSON instead of "
                         "human-readable form.")
    return p.parse_args()


def cli_main() -> int:
    args = parse_args()
    if args.offline:
        os.environ["GHOST_AGENT_OFFLINE"] = "1"

    generator, is_random = make_generator(
        args.checkpoint,
        args.device,
        args.max_new_tokens,
        args.temperature,
        args.top_p,
        args.top_k,
        args.repetition_penalty,
    )

    cfg = RuntimeConfig(
        max_iters=args.max_iters,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    if args.system_prompt is not None:
        cfg.system_prompt = args.system_prompt

    agent = GhostAgent(generator, cfg)
    trace = agent.run(args.query)

    if args.json:
        print(trace.to_json(indent=2))
        return 0

    if is_random:
        print("[note] Using random ghost-tiny weights, so output will "
              "be noise. Pass --checkpoint to load real weights.")
    print()
    print(f"Q: {trace.query}")
    print()
    for msg in trace.history:
        print(f"--- {msg.role.value} ---")
        snippet = msg.content if len(msg.content) < 800 \
            else msg.content[:800] + " …"
        print(snippet)
        print()
    print("=" * 60)
    print(f"Final answer ({trace.terminated_reason}, "
          f"{trace.iterations} iters, {trace.total_latency_ms} ms):")
    print(trace.final_answer)
    return 0


if __name__ == "__main__":
    sys.exit(cli_main())
