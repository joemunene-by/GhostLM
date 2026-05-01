#!/usr/bin/env python3
"""GhostLM LoRA fine-tuning — adapt the chat-tuned base on a custom corpus.

Lets community contributors fine-tune ghost-small on their own cybersecurity
data without touching base weights or running a full SFT. Uses HuggingFace
`peft` to inject low-rank adapters into the attention QKV / output and FFN
projections; the resulting adapter is ~0.5–1.5 MB depending on rank — small
enough to host dozens of variants on HF Hub.

Defaults follow the 2026 best-practice for sub-100M base models:
- DoRA = True (weight-decomposed LoRA — reliably +1-2 pts over plain LoRA at
  the same rank)
- rank = 8, alpha = 16
- targets = c_qkv + proj + ffn.fc1 + ffn.fc2 (full attention + FFN coverage)
- dropout = 0.05

The dataset format is the same as ``finetune_chat.py`` — a chat JSONL with
``{"turns": [...]}`` lines. Loss is masked to assistant tokens by ChatDataset.

Output: ``checkpoints/lora/<run_name>/`` containing the adapter weights and a
small adapter card describing the rank, targets, and training data. To use the
adapter at inference, ``scripts/chat.py`` can be wrapped or modified to apply
``PeftModel.from_pretrained`` over the base.
"""

import argparse
import json
from dataclasses import fields
from pathlib import Path

import torch
import torch.nn as nn

from ghostlm.chat_dataset import build_chat_dataloaders
from ghostlm.config import GhostLMConfig
from ghostlm.model import GhostLM
from ghostlm.tokenizer import GhostTokenizer
from ghostlm.trainer import GhostTrainer

try:
    from peft import LoraConfig, get_peft_model, TaskType
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "ghostlm-lora requires the 'peft' package: pip install peft"
    ) from e


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for LoRA fine-tuning."""
    p = argparse.ArgumentParser(description="GhostLM LoRA fine-tuning")
    p.add_argument("--checkpoint", required=True,
                   help="Base checkpoint (typically phase5_chat/best_model.pt)")
    p.add_argument("--train-data", required=True,
                   help="Chat JSONL — same format as chat_train.jsonl")
    p.add_argument("--val-data", required=True)
    p.add_argument("--run-name", default="custom",
                   help="Subdir under checkpoints/lora/")
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--alpha", type=int, default=16)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--use-dora", action="store_true", default=True,
                   help="Use weight-decomposed LoRA (default True)")
    p.add_argument("--no-dora", dest="use_dora", action="store_false")
    p.add_argument("--targets", nargs="+",
                   default=["c_qkv", "proj", "fc1", "fc2"],
                   help="Linear modules to inject LoRA into")
    p.add_argument("--learning-rate", type=float, default=1e-4,
                   help="LoRA tolerates a higher LR than full SFT")
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum-steps", type=int, default=4)
    p.add_argument("--context-length", type=int, default=1024)
    p.add_argument("--eval-interval", type=int, default=100)
    p.add_argument("--save-interval", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--description", default="",
                   help="Description for the adapter card")
    return p.parse_args()


def freeze_base_model(model: nn.Module) -> None:
    """Freeze every parameter outside the LoRA layers.

    PEFT sets ``requires_grad=False`` on the base by default, but we double-
    check here to make sure no rogue parameter slips through.
    """
    trainable = 0
    total = 0
    for n, p in model.named_parameters():
        total += p.numel()
        if "lora_" in n.lower() or p.requires_grad:
            trainable += p.numel()
    print(f"  Trainable params: {trainable:,} / {total:,} "
          f"({100.0 * trainable / total:.2f}%)")


def main() -> None:
    """Run LoRA fine-tuning end-to-end."""
    args = parse_args()
    torch.manual_seed(args.seed)

    tokenizer = GhostTokenizer()

    # ---- Load base model ----
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg_raw = ckpt["config"]
    if isinstance(cfg_raw, dict):
        config = GhostLMConfig(**{
            f.name: cfg_raw[f.name]
            for f in fields(GhostLMConfig)
            if f.name in cfg_raw
        })
    else:
        config = cfg_raw

    # SFT trainer overrides
    config.batch_size = args.batch_size
    config.grad_accum_steps = args.grad_accum_steps
    config.learning_rate = args.learning_rate
    config.warmup_steps = args.warmup_steps
    config.max_steps = args.max_steps
    config.eval_interval = args.eval_interval
    config.save_interval = args.save_interval
    config.context_length = args.context_length
    config.checkpoint_dir = f"checkpoints/lora/{args.run_name}"
    config.log_dir = f"logs/lora/{args.run_name}"
    config.device = args.device

    model = GhostLM(config)
    state = ckpt.get("model_state_dict", ckpt.get("model"))
    model.load_state_dict(state, strict=False)
    print(f"Loaded base: {sum(p.numel() for p in model.parameters()):,} params")

    # ---- Wrap with PEFT LoRA ----
    lora_cfg = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        target_modules=list(args.targets),
        lora_dropout=args.dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        use_dora=args.use_dora,
    )
    model = get_peft_model(model, lora_cfg)
    print(f"LoRA injected — rank={args.rank}, alpha={args.alpha}, "
          f"dora={args.use_dora}, targets={args.targets}")
    freeze_base_model(model)

    # ---- Data ----
    train_loader, val_loader = build_chat_dataloaders(
        args.train_data, args.val_data, tokenizer, config,
    )

    # ---- Train ----
    trainer = GhostTrainer(model, config)
    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    trainer.train(train_loader, val_loader)

    # ---- Save adapter ----
    adapter_dir = Path(config.checkpoint_dir) / "adapter"
    model.save_pretrained(str(adapter_dir))

    # ---- Adapter card ----
    card = {
        "base_checkpoint": args.checkpoint,
        "rank": args.rank,
        "alpha": args.alpha,
        "use_dora": args.use_dora,
        "targets": list(args.targets),
        "dropout": args.dropout,
        "learning_rate": args.learning_rate,
        "max_steps": args.max_steps,
        "train_data": args.train_data,
        "val_data": args.val_data,
        "description": args.description,
    }
    with open(adapter_dir / "adapter_card.json", "w") as f:
        json.dump(card, f, indent=2)

    print()
    print(f"LoRA adapter saved to: {adapter_dir}")
    print(f"Adapter card:          {adapter_dir / 'adapter_card.json'}")


if __name__ == "__main__":
    main()
