#!/usr/bin/env python3
"""ghost-base pretraining with MLX on Apple Silicon.

A drop-in for ``train_ghost_base.py`` on Macs: same flags (minus CUDA-only
ones), same data loaders and curriculum, same LR schedule, gradient
clipping, accumulation and eval semantics as ``GhostTrainer``. Checkpoints
use the PyTorch layout (``model_state_dict`` as torch tensors) so evals, RAG,
the agent and ``--resume`` across frameworks keep working; MLX optimizer
state rides along under ``mlx_optimizer_state``. SIGTERM saves and exits.

``--dtype bfloat16`` keeps fp32 master weights and optimizer state and runs
forward/backward in bf16, which is about twice as fast as fp32 on an M4.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import signal
import time
from dataclasses import asdict
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import torch
from mlx.utils import tree_flatten, tree_map, tree_unflatten

from ghostlm.config import GhostLMConfig
from ghostlm.curriculum import DEFAULT_GENERALIST_CURRICULUM, parse_curriculum_spec
from ghostlm.dataset import build_curriculum_train_loader, build_dataloaders
from ghostlm.mlx_model import GhostLMMLX, from_torch_state
from ghostlm.model import GhostLM
from ghostlm.tokenizer import GhostTokenizer

MIN_LR = 1e-5


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GhostLM ghost-base pretrain on MLX")
    p.add_argument("--train-data", default="data/processed/train.bin")
    p.add_argument("--val-data", default="data/processed/val.bin")
    p.add_argument("--run-name", default="ghost_base_mlx")
    p.add_argument("--max-steps", type=int, default=30_000)
    p.add_argument("--warmup-steps", type=int, default=2_000)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--batch-size", type=int, default=64,
                   help="Sequences per optimizer step (split into --grad-accum-steps micro-batches)")
    p.add_argument("--grad-accum-steps", type=int, default=32)
    p.add_argument("--context-length", type=int, default=1024)
    p.add_argument("--eval-interval", type=int, default=500)
    p.add_argument("--save-interval", type=int, default=1_500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    p.add_argument("--resume", default=None)
    p.add_argument("--grad-checkpoint", action="store_true")
    p.add_argument("--no-intra-doc-mask", action="store_true")
    p.add_argument("--curriculum-manifest", default=None)
    p.add_argument("--curriculum-spec", default=None)
    p.add_argument("--lr-schedule", default="cosine", choices=["cosine", "wsd"])
    p.add_argument("--wsd-decay-frac", type=float, default=0.2)
    p.add_argument("--best-weights-only", action="store_true")
    p.add_argument("--attn-gate", action="store_true")
    p.add_argument("--value-residual", action="store_true")
    p.add_argument("--optimizer", default="adamw", choices=["adamw", "muon", "normuon"])
    p.add_argument("--cautious-wd", action="store_true")
    p.add_argument("--memory-limit-gb", type=float, default=9.0,
                   help="MLX memory guideline; also caps the allocator cache at 1GB.")
    return p.parse_args()


def ghost_base_config(args, tokenizer) -> GhostLMConfig:
    """The same shape and settings ``train_ghost_base.py`` builds, with dropout 0."""
    c = GhostLMConfig.from_preset("ghost-small-v0.5")
    c.n_layers, c.d_model, c.n_heads, c.n_kv_heads, c.d_ff = 30, 960, 15, 5, 3936
    c.use_qk_norm = True
    c.use_flash_attention = True
    c.vocab_size = tokenizer.vocab_size
    c.eos_token_id = tokenizer.eos_id
    c.intra_doc_mask = not args.no_intra_doc_mask
    c.context_length = args.context_length
    c.batch_size, c.grad_accum_steps = args.batch_size, args.grad_accum_steps
    c.learning_rate, c.warmup_steps, c.max_steps = args.learning_rate, args.warmup_steps, args.max_steps
    c.eval_interval, c.save_interval = args.eval_interval, args.save_interval
    c.checkpoint_dir, c.log_dir = f"checkpoints/{args.run_name}", f"logs/{args.run_name}"
    c.device, c.dtype, c.dropout = "mps", args.dtype, 0.0
    c.optimizer, c.lr_schedule, c.wsd_decay_frac = args.optimizer, args.lr_schedule, args.wsd_decay_frac
    c.best_weights_only, c.attn_gate, c.value_residual = args.best_weights_only, args.attn_gate, args.value_residual
    c.gradient_checkpointing = args.grad_checkpoint
    c.cautious_wd = args.cautious_wd
    return c


def lr_at(step: int, cfg: GhostLMConfig) -> float:
    """Identical to GhostTrainer.get_lr."""
    base, warmup, max_steps = cfg.learning_rate, cfg.warmup_steps, cfg.max_steps
    if step < warmup:
        return base * (step + 1) / warmup
    if cfg.lr_schedule == "wsd":
        decay_start = int(max_steps * (1 - cfg.wsd_decay_frac))
        if step < decay_start:
            return base
        frac = min(1.0, (step - decay_start) / max(1, max_steps - decay_start))
        return base + (MIN_LR - base) * frac
    ratio = min((step - warmup) / max(1, max_steps - warmup), 1.0)
    return MIN_LR + (base - MIN_LR) * 0.5 * (1.0 + math.cos(math.pi * ratio))


def newton_schulz5(g: mx.array, steps: int = 5) -> mx.array:
    """Same quintic iteration as ghostlm.muon.zeropower_via_newtonschulz5 (fp32)."""
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.astype(mx.float32)
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    x = x / (mx.linalg.norm(x) + 1e-7)
    for _ in range(steps):
        A = x @ x.T
        B = b * A + c * (A @ A)
        x = a * x + B @ x
    return x.T if transposed else x


class MuonMLX:
    """MLX port of ghostlm.muon.MuonWithAuxAdam (Muon/NorMuon on hidden matrices, AdamW elsewhere)."""

    def __init__(self, cfg: GhostLMConfig, muon_names: set, decay: dict):
        self.cfg = cfg
        self.muon_names = muon_names
        self.decay = decay
        self.normuon = cfg.optimizer == "normuon"
        self.cautious = getattr(cfg, "cautious_wd", False)
        self.momentum = getattr(cfg, "muon_momentum", 0.95)
        self.state: dict = {}

    def _decay(self, p, update, amount):
        if amount == 0:
            return p
        if self.cautious:
            return p - p * ((update * p) > 0) * amount
        return p * (1 - amount)

    def apply(self, grads: dict, params: dict, lr: float) -> dict:
        b1, b2, eps = self.cfg.beta1, self.cfg.beta2, 1e-8
        out = {}
        for name, p in params.items():
            g = grads[name]
            st = self.state.setdefault(name, {})
            wd = lr * self.cfg.weight_decay if self.decay.get(name) else 0.0
            if name in self.muon_names:
                buf = st.get("momentum_buffer", mx.zeros_like(p)) * self.momentum + g
                st["momentum_buffer"] = buf
                update = newton_schulz5(g + self.momentum * buf)
                if self.normuon:
                    row = st.get("row_sq", mx.zeros((p.shape[0], 1)))
                    row = row + (1 - b2) * (mx.mean(update * update, axis=1, keepdims=True) - row)
                    st["row_sq"] = row
                    update = update / (mx.sqrt(row) + eps)
                    update = update * (0.2 * p.size ** 0.5 / (mx.linalg.norm(update) + eps))
                else:
                    update = update * (0.2 * max(p.shape) ** 0.5)
            else:
                step = st.get("step", 0) + 1
                st["step"] = step
                m = st.get("exp_avg", mx.zeros_like(p))
                v = st.get("exp_avg_sq", mx.zeros_like(p))
                m = m + (1 - b1) * (g - m)
                v = v * b2 + (1 - b2) * g * g
                st["exp_avg"], st["exp_avg_sq"] = m, v
                update = (m / (1 - b1 ** step)) / (mx.sqrt(v / (1 - b2 ** step)) + eps)
            out[name] = self._decay(p, update, wd) - lr * update
        return out

    def state_arrays(self) -> dict:
        flat = {}
        for name, st in self.state.items():
            for k, v in st.items():
                flat[f"{name}/{k}"] = np.asarray(v)
        return flat

    def load_state_arrays(self, flat: dict) -> None:
        self.state = {}
        for key, v in flat.items():
            name, k = key.rsplit("/", 1)
            self.state.setdefault(name, {})[k] = int(v) if k == "step" else mx.array(v)


def decay_mask(model: GhostLMMLX) -> dict:
    """True for Linear weights (decayed), False for embeddings, norms, biases and scalars."""
    linear_weights = {f"{name}.weight" for name, m in model.named_modules() if isinstance(m, nn.Linear)}
    return {name: name in linear_weights for name, _ in tree_flatten(model.parameters())}


class Trainer:
    def __init__(self, cfg: GhostLMConfig, model: GhostLMMLX, compute_dtype):
        self.cfg = cfg
        self.model = model
        self.compute_dtype = compute_dtype
        self.master = tree_map(lambda p: p.astype(mx.float32), model.parameters())
        self.decay = decay_mask(model)
        # MLX defaults to no bias correction; torch.optim.AdamW applies it, and
        # early updates would otherwise be ~0.45x the size.
        self.opt = optim.AdamW(learning_rate=cfg.learning_rate, betas=[cfg.beta1, cfg.beta2],
                               eps=1e-8, weight_decay=0.0, bias_correction=True)
        self.opt.init(self.master)
        self.muon = None
        if cfg.optimizer in ("muon", "normuon"):
            # Same split as GhostLM.configure_optimizers: 2D decayed (Linear) weights.
            muon_names = {n for n, v in tree_flatten(self.master) if self.decay.get(n) and v.ndim == 2}
            self.muon = MuonMLX(cfg, muon_names, self.decay)
        self.step = 0
        self.best_val_loss = float("inf")
        self.stop_requested = False
        self.log = []
        self.ckpt_dir = Path(cfg.checkpoint_dir)
        self.log_dir = Path(cfg.log_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._grad_fn = nn.value_and_grad(model, lambda m, x, y: m.loss(x, y))
        self._sync_model()

    def _sync_model(self):
        self.model.update(tree_map(lambda p: p.astype(self.compute_dtype), self.master))

    def train_step(self, x: np.ndarray, y: np.ndarray) -> float:
        # Same split as GhostTrainer: the batch is divided, not multiplied.
        micro = max(1, len(x) // self.cfg.grad_accum_steps)
        grads_acc, total = None, 0.0
        chunks = [(x[i:i + micro], y[i:i + micro]) for i in range(0, len(x), micro)]
        for mx_, my_ in chunks:
            loss, grads = self._grad_fn(self.model, mx.array(mx_), mx.array(my_))
            grads = tree_map(lambda g: g.astype(mx.float32) / len(chunks), grads)
            grads_acc = grads if grads_acc is None else tree_map(mx.add, grads_acc, grads)
            mx.eval(loss, grads_acc)
            total += loss.item()

        lr = lr_at(self.step, self.cfg)
        grads_acc, _ = optim.clip_grad_norm(grads_acc, self.cfg.grad_clip)
        flat = dict(tree_flatten(self.master))
        if self.muon is not None:
            updated = self.muon.apply(dict(tree_flatten(grads_acc)), flat, lr)
            self.master = tree_unflatten(list(updated.items()))
            mx.eval(self.master, [v for st in self.muon.state.values() for v in st.values()
                                  if isinstance(v, mx.array)])
        else:
            wd = lr * self.cfg.weight_decay
            if wd:
                flat = {k: (v * (1 - wd) if self.decay.get(k) else v) for k, v in flat.items()}
            self.opt.learning_rate = lr
            self.master = self.opt.apply_gradients(grads_acc, tree_unflatten(list(flat.items())))
            mx.eval(self.master, self.opt.state)
        self._sync_model()
        self.step += 1
        return total / len(chunks)

    def eval_loss(self, val_loader, num_batches: int = 20) -> float:
        total, count = 0.0, 0
        for i, (x, y) in enumerate(val_loader):
            if i >= num_batches:
                break
            size = max(1, len(x) // self.cfg.grad_accum_steps)
            losses = [self.model.loss(mx.array(x[j:j + size].numpy()), mx.array(y[j:j + size].numpy())).item()
                      for j in range(0, len(x), size)]
            total += sum(losses) / len(losses)
            count += 1
        return total / max(1, count)

    def _state(self, val_loss: float, with_optimizer: bool = True) -> dict:
        # np.asarray on an evaluated MLX array is zero-copy (unified memory), so a
        # save does not briefly double the multi-GB weight and optimizer state.
        model_state = {k: torch.from_numpy(np.asarray(v)) for k, v in tree_flatten(self.master)}
        model_state["lm_head.weight"] = model_state["token_embedding.weight"]
        out = {
            "step": self.step, "val_loss": val_loss, "best_val_loss": self.best_val_loss,
            "model_state_dict": model_state, "config": asdict(self.cfg), "framework": "mlx",
        }
        if with_optimizer:
            out["mlx_optimizer_state"] = (self.muon.state_arrays() if self.muon is not None else
                                          {k: np.asarray(v) for k, v in tree_flatten(self.opt.state)})
        return out

    def save(self, val_loss: float) -> None:
        is_best = val_loss < self.best_val_loss
        if is_best:
            self.best_val_loss = val_loss
        _atomic_save(self._state(val_loss), self.ckpt_dir / f"checkpoint_step_{self.step}.pt")
        print(f"  Saved checkpoint: {self.ckpt_dir / f'checkpoint_step_{self.step}.pt'}", flush=True)
        if is_best:
            _atomic_save(self._state(val_loss, not self.cfg.best_weights_only), self.ckpt_dir / "best_model.pt")
            print(f"  New best model saved (val_loss={val_loss:.4f})", flush=True)

    def load(self, path: str) -> None:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        from_torch_state(self.model, ck["model_state_dict"])
        self.master = tree_map(lambda p: p.astype(mx.float32), self.model.parameters())
        if "mlx_optimizer_state" in ck and self.muon is not None:
            self.muon.load_state_arrays(ck["mlx_optimizer_state"])
        elif "mlx_optimizer_state" in ck:
            self.opt.state = tree_unflatten([(k, mx.array(v)) for k, v in ck["mlx_optimizer_state"].items()])
        else:
            print("  (no MLX optimizer state in checkpoint; AdamW moments restart from zero)")
        self.step = ck["step"]
        self.best_val_loss = ck.get("best_val_loss", ck["val_loss"])
        self._sync_model()
        print(f"Loaded checkpoint from step {self.step} (best val_loss={self.best_val_loss:.4f})", flush=True)

    def _log(self, row: dict) -> None:
        self.log.append(row)
        (self.log_dir / "training_log.json").write_text(json.dumps(self.log, indent=2))

    def train(self, train_loader, val_loader) -> None:
        log_path = self.log_dir / "training_log.json"
        if log_path.exists():
            self.log = [e for e in json.loads(log_path.read_text()) if e.get("step", 0) <= self.step]
        it = iter(_cycle(train_loader))
        print(f"Training from step {self.step} to {self.cfg.max_steps}", flush=True)
        while self.step < self.cfg.max_steps:
            t0 = time.time()
            x, y = next(it)
            loss = self.train_step(x.numpy(), y.numpy())
            dt = time.time() - t0
            print(f"step {self.step} loss {loss:.4f} lr {lr_at(self.step - 1, self.cfg):.2e} "
                  f"dt {dt:.1f}s tok/s {x.numel() / dt:,.0f} "
                  f"mem {mx.get_active_memory() / 1e9:.1f}/{mx.get_peak_memory() / 1e9:.1f}GB", flush=True)
            eval_due = self.step % self.cfg.eval_interval == 0
            save_due = self.step % self.cfg.save_interval == 0
            if eval_due or save_due:
                val_loss = self.eval_loss(val_loader)
            if eval_due:
                print(f"  Step {self.step} | val_loss={val_loss:.4f} | train_loss={loss:.4f}", flush=True)
                self._log({"step": self.step, "train_loss": loss, "val_loss": val_loss,
                           "lr": lr_at(self.step - 1, self.cfg), "time": dt})
            if save_due:
                self.save(val_loss)
            if self.cfg.lr_schedule == "wsd" and self.step == int(self.cfg.max_steps * (1 - self.cfg.wsd_decay_frac)):
                _atomic_save(self._state(float("inf")), self.ckpt_dir / "pre_decay.pt")
                print(f"  Saved pre-decay checkpoint at step {self.step}", flush=True)
            if self.stop_requested:
                print(f"Stop requested; saving step {self.step} and exiting.", flush=True)
                self.save(float("inf"))
                return
        val_loss = self.eval_loss(val_loader)
        print(f"Final val_loss: {val_loss:.4f}", flush=True)
        self.save(val_loss)
        self._log({"step": self.step, "train_loss": loss, "val_loss": val_loss,
                   "lr": lr_at(self.step - 1, self.cfg), "time": dt, "status": "complete"})


def _cycle(loader):
    while True:
        yield from loader


def _atomic_save(obj, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def main() -> None:
    args = parse_args()
    mx.set_memory_limit(int(args.memory_limit_gb * 1e9))
    mx.set_cache_limit(int(1e9))
    mx.random.seed(args.seed)
    torch.manual_seed(args.seed)
    tokenizer = GhostTokenizer()
    cfg = ghost_base_config(args, tokenizer)
    print(cfg, flush=True)

    # Initialise through the PyTorch model so the init scheme (incl. the
    # depth-scaled residual projections) is identical to train_ghost_base.py.
    model = GhostLMMLX(cfg, gradient_checkpointing=args.grad_checkpoint)
    init_model = GhostLM(cfg)
    from_torch_state(model, init_model.state_dict())
    del init_model
    gc.collect()
    n_params = sum(v.size for _, v in tree_flatten(model.parameters()))
    print(f"Model parameters: {n_params:,} ({n_params / 1e6:.1f}M), dtype {args.dtype}", flush=True)

    compute = mx.bfloat16 if args.dtype == "bfloat16" else mx.float32
    trainer = Trainer(cfg, model, compute)

    train_loader, val_loader = build_dataloaders(args.train_data, args.val_data, tokenizer, cfg)
    if args.curriculum_manifest:
        manifest = json.loads(Path(args.curriculum_manifest).read_text())
        curriculum = (parse_curriculum_spec(args.curriculum_spec)
                      if args.curriculum_spec else DEFAULT_GENERALIST_CURRICULUM)
        print(f"Curriculum:   {len(manifest)} domains from {args.curriculum_manifest}", flush=True)
        train_loader = build_curriculum_train_loader(
            manifest, cfg, curriculum, progress_fn=lambda: trainer.step / max(1, cfg.max_steps))

    if args.resume:
        trainer.load(args.resume)

    def request_stop(signum, frame):
        trainer.stop_requested = True

    signal.signal(signal.SIGTERM, request_stop)
    trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    main()
