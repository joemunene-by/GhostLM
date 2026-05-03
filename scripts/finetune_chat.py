#!/usr/bin/env python3
"""GhostLM chat-tuning fine-tune — SFT on top of the Phase 4 ghost-small checkpoint.

Steps:
1. Build the chat tokenizer (now includes <|ghost_user|>, <|ghost_assistant|>,
   <|ghost_end|>) and the chat dataset.
2. Load the Phase 4 ghost-small checkpoint.
3. Expand the token-embedding rows from 50261 → 50264 to accommodate the three
   new chat-role tokens, copying existing weights and initializing the new rows
   with small Gaussian noise. Because lm_head is weight-tied to token_embedding,
   re-tying is enough — no separate head expansion needed.
4. Train with SFT-appropriate hyperparameters (lower LR, fewer steps, the loss
   is already masked to assistant tokens by ChatDataset's target -1 fill).
5. Save to ``checkpoints/phase5_chat/`` with periodic checkpoints + best-val.

This script does not modify ``scripts/train.py`` or ``ghostlm/trainer.py`` —
it reuses GhostTrainer's training loop directly with overridden config.
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn

from ghostlm.chat_dataset import build_chat_dataloaders
from ghostlm.config import GhostLMConfig
from ghostlm.model import GhostLM
from ghostlm.tokenizer import GhostTokenizer, GhostTokenizerV05, load_tokenizer
from ghostlm.trainer import GhostTrainer


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for chat-tuning."""
    p = argparse.ArgumentParser(description="GhostLM chat-tuning (SFT)")
    p.add_argument("--checkpoint", required=True,
                   help="Pretrain checkpoint to fine-tune (e.g. checkpoints/phase4_ghost_small/best_model.pt)")
    p.add_argument("--train-data", default="data/processed/chat_train.jsonl")
    p.add_argument("--val-data", default="data/processed/chat_val.jsonl")
    p.add_argument("--run-name", default="phase5_chat",
                   help="Subdir under checkpoints/ and logs/")
    p.add_argument("--learning-rate", type=float, default=2e-5,
                   help="Lower than pretrain (3e-4) — SFT is delicate")
    p.add_argument("--max-steps", type=int, default=4000)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum-steps", type=int, default=4)
    p.add_argument("--context-length", type=int, default=1024)
    p.add_argument("--eval-interval", type=int, default=200)
    p.add_argument("--save-interval", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--tokenizer", default=None,
                   help="Optional path to a v0.5 tokenizer.json. When provided, "
                        "uses the 32K BPE; otherwise falls back to the legacy "
                        "tiktoken GPT-2 BPE (50261 base vocab + 7 specials).")
    return p.parse_args()


def expand_token_embedding(model: GhostLM, new_vocab_size: int) -> None:
    """Resize the model's tied embedding to ``new_vocab_size``.

    Copies existing rows verbatim. New rows are initialized to the *mean* of the
    existing embeddings plus tiny N(0, 0.001²) jitter — sub-100M models destabilize
    when N(0, 0.02²) noise tokens hit the residual stream cold (research call:
    SmolLM2 retrospective + Komatsuzaki et al. on warm-start). Re-ties lm_head.
    """
    old_emb = model.token_embedding
    old_vocab, d_model = old_emb.weight.shape
    if new_vocab_size == old_vocab:
        return
    if new_vocab_size < old_vocab:
        raise ValueError(
            f"Refusing to shrink vocab: old={old_vocab}, new={new_vocab_size}"
        )

    new_emb = nn.Embedding(new_vocab_size, d_model)
    with torch.no_grad():
        new_emb.weight[:old_vocab] = old_emb.weight
        mean_emb = old_emb.weight.mean(dim=0, keepdim=True)
        n_new = new_vocab_size - old_vocab
        new_emb.weight[old_vocab:] = mean_emb.expand(n_new, -1)
        new_emb.weight[old_vocab:].add_(torch.randn(n_new, d_model) * 0.001)

    new_emb = new_emb.to(old_emb.weight.device).to(old_emb.weight.dtype)
    model.token_embedding = new_emb

    # Re-tie the language-model head to the new embedding weight. The original
    # lm_head Linear's weight pointer still references the *old* tensor; rebuild
    # so out_features matches and tying is preserved.
    new_head = nn.Linear(d_model, new_vocab_size, bias=False).to(
        old_emb.weight.device
    ).to(old_emb.weight.dtype)
    new_head.weight = model.token_embedding.weight
    model.lm_head = new_head

    # Update the config so downstream code that reads vocab_size sees the new value.
    model.config.vocab_size = new_vocab_size


def main() -> None:
    """Load Phase 4 checkpoint, expand vocab, and run SFT chat-tuning."""
    args = parse_args()

    torch.manual_seed(args.seed)

    # ---- Tokenizer ----
    # When --tokenizer points at a v0.5 tokenizer.json, use the 32K BPE;
    # otherwise fall back to the legacy tiktoken GPT-2 wrapper.
    tokenizer = load_tokenizer(args.tokenizer) if args.tokenizer else GhostTokenizer()
    print(f"Tokenizer: {tokenizer}")
    print(f"  Special tokens: {sorted(tokenizer._special_tokens.items(), key=lambda kv: kv[1])}")

    # ---- Load checkpoint ----
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # The checkpoint stores a config dict; rebuild the config so we know the
    # original vocab_size, then override training-specific fields.
    base_cfg = ckpt.get("config", None)
    if base_cfg is None:
        raise RuntimeError("Checkpoint missing 'config' — cannot infer architecture")
    if isinstance(base_cfg, dict):
        config = GhostLMConfig(**base_cfg)
    else:
        config = base_cfg

    # Apply SFT training overrides
    config.batch_size = args.batch_size
    config.grad_accum_steps = args.grad_accum_steps
    config.learning_rate = args.learning_rate
    config.warmup_steps = args.warmup_steps
    config.max_steps = args.max_steps
    config.eval_interval = args.eval_interval
    config.save_interval = args.save_interval
    config.context_length = args.context_length
    config.checkpoint_dir = f"checkpoints/{args.run_name}"
    config.log_dir = f"logs/{args.run_name}"
    config.device = args.device

    print(f"Pretrain config vocab_size: {config.vocab_size}")

    # ---- Build model and load pretrain weights at original vocab size ----
    model = GhostLM(config)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt["model"]

    # Pos embedding can be larger in the checkpoint (pretrain ctx 1024) than in
    # the current model (SFT at smaller ctx). Slice down to match — chat data
    # rarely needs > 512 tokens and shrinking ctx halves attention memory.
    pe_key = "pos_embedding.weight"
    if pe_key in state:
        ckpt_pe = state[pe_key]
        ctx_now = model.pos_embedding.weight.shape[0]
        if ckpt_pe.shape[0] > ctx_now:
            print(f"  Slicing pos_embedding {ckpt_pe.shape[0]} → {ctx_now} for SFT ctx")
            state[pe_key] = ckpt_pe[:ctx_now]

    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        print(f"  Unexpected keys (ignored): {unexpected}")
    if missing:
        print(f"  Missing keys (will be reinitialized): {missing}")

    # ---- Expand vocab to fit chat tokens ----
    new_vocab = tokenizer.vocab_size
    print(f"Expanding token embedding {config.vocab_size} → {new_vocab}")
    expand_token_embedding(model, new_vocab)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ---- Data ----
    train_loader, val_loader = build_chat_dataloaders(
        args.train_data, args.val_data, tokenizer, config,
    )

    # ---- Trainer ----
    trainer = GhostTrainer(model, config)
    print(f"Device: {trainer.device}")
    print(f"Saving to: {config.checkpoint_dir}")
    print(f"Logs to:   {config.log_dir}")
    print()

    # Persist tokenizer alongside checkpoint so chat.py can load both.
    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    tok_save_path = f"{config.checkpoint_dir}/tokenizer.json"
    if isinstance(tokenizer, GhostTokenizerV05):
        # V05 backend: copy the canonical tokenizer.json verbatim. The
        # save method on GhostTokenizer only writes special-token metadata,
        # which doesn't capture the BPE state — for V05 we need the raw
        # HuggingFace tokenizers JSON.
        import shutil
        shutil.copy2(args.tokenizer, tok_save_path)
    else:
        tokenizer.save(tok_save_path)

    trainer.train(train_loader, val_loader)

    # Persist the final config (with the updated vocab size) for inference loaders.
    with open(f"{config.checkpoint_dir}/config.json", "w") as f:
        json.dump({k: getattr(config, k) for k in vars(config)}, f, indent=2)

    print()
    print("Chat-tuning complete.")


if __name__ == "__main__":
    main()
