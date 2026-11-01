#!/usr/bin/env python3
"""ghost-base pretraining in JAX, for Cloud TPU (and CPU smoke tests).

    python -m ghostlm.tpu.train --curriculum-manifest data/processed/domains.json \
        --run-name ghost_base_tpu --max-steps 30000 --batch-size 256 --grad-accum-steps 1

Same flags, schedule, optimizer and eval semantics as ``scripts/train_ghost_base_mlx.py``.
The batch is sharded across all local devices (data parallel, parameters replicated).
Checkpoints are directories of safetensors plus ``meta.json``; ``--sync-to gs://...``
mirrors the checkpoint folder to a bucket after every save, and ``--resume`` accepts a
checkpoint directory or a PyTorch ``.pt`` (weights only). SIGTERM saves and exits.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax.traverse_util import flatten_dict, unflatten_dict
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from safetensors.numpy import load_file, save_file

from ghostlm.config import GhostLMConfig
from ghostlm.curriculum import DEFAULT_GENERALIST_CURRICULUM, parse_curriculum_spec
from ghostlm.tpu.data import CurriculumSampler, SingleStreamSampler, load_manifest, val_batches
from ghostlm.tpu.model import GhostLMFlax, cross_entropy, from_torch_state, init_params, param_count, to_torch_state
from ghostlm.tpu.optim import Optimizer, clip_by_global_norm, lr_at


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GhostLM ghost-base pretrain in JAX")
    p.add_argument("--train-data", default="data/processed/train.bin")
    p.add_argument("--val-data", default="data/processed/val.bin")
    p.add_argument("--curriculum-manifest", default=None)
    p.add_argument("--curriculum-spec", default=None)
    p.add_argument("--run-name", default="ghost_base_tpu")
    p.add_argument("--checkpoint-root", default="checkpoints")
    p.add_argument("--sync-to", default=None, help="gs://bucket/path mirrored with gsutil rsync after each save")
    p.add_argument("--max-steps", type=int, default=30_000)
    p.add_argument("--warmup-steps", type=int, default=2_000)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--batch-size", type=int, default=64, help="sequences per optimizer step")
    p.add_argument("--grad-accum-steps", type=int, default=1)
    p.add_argument("--context-length", type=int, default=1024)
    p.add_argument("--eval-interval", type=int, default=500)
    p.add_argument("--save-interval", type=int, default=1_500)
    p.add_argument("--eval-batches", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    p.add_argument("--resume", default=None)
    p.add_argument("--grad-checkpoint", action="store_true")
    p.add_argument("--no-intra-doc-mask", action="store_true")
    p.add_argument("--lr-schedule", default="cosine", choices=["cosine", "wsd"])
    p.add_argument("--wsd-decay-frac", type=float, default=0.2)
    p.add_argument("--attn-gate", action="store_true")
    p.add_argument("--value-residual", action="store_true")
    p.add_argument("--optimizer", default="adamw", choices=["adamw", "muon", "normuon"])
    p.add_argument("--cautious-wd", action="store_true")
    p.add_argument("--n-layers", type=int, default=30)
    p.add_argument("--d-model", type=int, default=960)
    p.add_argument("--n-heads", type=int, default=15)
    p.add_argument("--n-kv-heads", type=int, default=5)
    p.add_argument("--d-ff", type=int, default=3936)
    p.add_argument("--vocab-size", type=int, default=50264)
    p.add_argument("--eos-token-id", type=int, default=50258)
    p.add_argument("--save-torch", action="store_true", help="also write checkpoint_step_N.pt for the PyTorch tools")
    return p.parse_args()


def build_config(a: argparse.Namespace) -> GhostLMConfig:
    c = GhostLMConfig.from_preset("ghost-small-v0.5")
    c.n_layers, c.d_model, c.n_heads, c.n_kv_heads, c.d_ff = a.n_layers, a.d_model, a.n_heads, a.n_kv_heads, a.d_ff
    c.use_qk_norm = True
    c.vocab_size, c.eos_token_id = a.vocab_size, a.eos_token_id
    c.intra_doc_mask = not a.no_intra_doc_mask
    c.context_length, c.batch_size, c.grad_accum_steps = a.context_length, a.batch_size, a.grad_accum_steps
    c.learning_rate, c.warmup_steps, c.max_steps = a.learning_rate, a.warmup_steps, a.max_steps
    c.eval_interval, c.save_interval = a.eval_interval, a.save_interval
    c.checkpoint_dir, c.log_dir = f"{a.checkpoint_root}/{a.run_name}", f"logs/{a.run_name}"
    c.device, c.dtype, c.dropout, c.seed = "tpu", a.dtype, 0.0, a.seed
    c.optimizer, c.lr_schedule, c.wsd_decay_frac = a.optimizer, a.lr_schedule, a.wsd_decay_frac
    c.attn_gate, c.value_residual, c.cautious_wd = a.attn_gate, a.value_residual, a.cautious_wd
    c.gradient_checkpointing = a.grad_checkpoint
    return c


class Trainer:
    def __init__(self, cfg: GhostLMConfig, params: dict):
        self.cfg = cfg
        self.compute_dtype = jnp.bfloat16 if cfg.dtype == "bfloat16" else jnp.float32
        self.model = GhostLMFlax(cfg, dtype=self.compute_dtype)
        self.opt = Optimizer(cfg, params)
        self.devices = jax.devices()
        self.mesh = Mesh(np.array(self.devices), ("data",))
        replicated = NamedSharding(self.mesh, P())
        self.batch_sharding = NamedSharding(self.mesh, P("data"))
        self.params = jax.device_put(params, replicated)
        self.opt_state = jax.device_put(self.opt.init(params), replicated)
        self.step = 0
        self.best_val_loss = float("inf")
        self.stop_requested = False
        self.log: list = []
        self.ckpt_dir = Path(cfg.checkpoint_dir)
        self.log_dir = Path(cfg.log_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.sync_to = None
        self.save_torch = False
        self._train_step = jax.jit(self._train_step_impl, donate_argnums=(0, 1))
        self._eval_step = jax.jit(self._eval_step_impl)

    def _loss(self, params, x, y):
        return cross_entropy(self.model.apply({"params": params}, x), y)

    def _train_step_impl(self, params, opt_state, x, y, lr):
        accum = self.cfg.grad_accum_steps
        micro = x.shape[0] // accum
        xs = x.reshape(accum, micro, -1)
        ys = y.reshape(accum, micro, -1)
        grad_fn = jax.value_and_grad(self._loss)

        def body(carry, batch):
            grads_acc, loss_acc = carry
            loss, grads = grad_fn(params, *batch)
            grads = jax.tree_util.tree_map(lambda g: g.astype(jnp.float32) / accum, grads)
            return (jax.tree_util.tree_map(jnp.add, grads_acc, grads), loss_acc + loss / accum), None

        zero = jax.tree_util.tree_map(lambda p: jnp.zeros(p.shape, jnp.float32), params)
        (grads, loss), _ = jax.lax.scan(body, (zero, jnp.zeros((), jnp.float32)), (xs, ys))
        grads, grad_norm = clip_by_global_norm(grads, self.cfg.grad_clip)
        params, opt_state = self.opt.update(params, grads, opt_state, lr)
        return params, opt_state, loss, grad_norm

    def _eval_step_impl(self, params, x, y):
        return self._loss(params, x, y)

    def train_step(self, x: np.ndarray, y: np.ndarray) -> float:
        lr = lr_at(self.step, self.cfg)
        x = jax.device_put(jnp.asarray(x), self.batch_sharding)
        y = jax.device_put(jnp.asarray(y), self.batch_sharding)
        self.params, self.opt_state, loss, _ = self._train_step(self.params, self.opt_state, x, y, jnp.float32(lr))
        self.step += 1
        return float(loss)

    def eval_loss(self, val_path: str, num_batches: int) -> float:
        size = max(1, self.cfg.batch_size // self.cfg.grad_accum_steps)
        losses = [float(self._eval_step(self.params, jnp.asarray(x), jnp.asarray(y)))
                  for x, y in val_batches(val_path, self.cfg.context_length, size, num_batches)]
        return sum(losses) / max(1, len(losses))

    def _flat_state_arrays(self) -> dict:
        """Optimizer state as ``{"blocks_0/attn/proj/kernel/exp_avg": array, "step": array}``."""
        out = {}
        for path, entry in self.opt_state.items():
            if path == ("step",):
                out["step"] = np.asarray(entry)
            else:
                for name, v in entry.items():
                    out["/".join(path) + "/" + name] = np.asarray(v)
        return out

    @staticmethod
    def _opt_state_from_arrays(flat: dict) -> dict:
        state = {}
        for key, v in flat.items():
            parts = key.split("/")
            if parts == ["step"]:
                state[("step",)] = jnp.asarray(v)
            else:
                state.setdefault(tuple(parts[:-1]), {})[parts[-1]] = jnp.asarray(v)
        return state

    def save(self, val_loss: float) -> None:
        is_best = val_loss < self.best_val_loss
        if is_best:
            self.best_val_loss = val_loss
        target = self.ckpt_dir / f"checkpoint_step_{self.step}"
        tmp = target.with_name(target.name + ".tmp")
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        params_np = {"/".join(path): np.asarray(v) for path, v in flatten_dict(self.params).items()}
        save_file(params_np, str(tmp / "params.safetensors"))
        save_file(self._flat_state_arrays(), str(tmp / "optimizer.safetensors"))
        (tmp / "meta.json").write_text(json.dumps({
            "step": self.step, "val_loss": val_loss, "best_val_loss": self.best_val_loss,
            "config": asdict(self.cfg), "framework": "jax"}, indent=2))
        if target.exists():
            shutil.rmtree(target)
        os.replace(tmp, target)
        print(f"  Saved checkpoint: {target}", flush=True)
        if is_best:
            best = self.ckpt_dir / "best_model"
            if best.exists() or best.is_symlink():
                best.unlink() if best.is_symlink() else shutil.rmtree(best)
            best.symlink_to(target.name)
            print(f"  New best model (val_loss={val_loss:.4f})", flush=True)
        if self.save_torch:
            self._save_torch(target, val_loss)
        if self.sync_to:
            subprocess.Popen(["gsutil", "-m", "-q", "rsync", "-r", str(self.ckpt_dir), self.sync_to])

    def _save_torch(self, target: Path, val_loss: float) -> None:
        import torch
        state = {k: torch.from_numpy(v.copy()) for k, v in to_torch_state(jax.device_get(self.params)).items()}
        torch.save({"step": self.step, "val_loss": val_loss, "best_val_loss": self.best_val_loss,
                    "model_state_dict": state, "config": asdict(self.cfg), "framework": "jax"},
                   target.with_suffix(".pt"))

    def load(self, path: str) -> None:
        path = Path(path)
        replicated = NamedSharding(self.mesh, P())
        if path.suffix == ".pt":
            import torch
            ck = torch.load(path, map_location="cpu", weights_only=False)
            self.params = jax.device_put(from_torch_state(jax.device_get(self.params), ck["model_state_dict"]), replicated)
            self.step = int(ck.get("step", 0))
            self.best_val_loss = float(ck.get("best_val_loss", ck.get("val_loss", float("inf"))))
            print(f"Loaded PyTorch weights from step {self.step}; optimizer moments restart from zero", flush=True)
            return
        params = unflatten_dict({tuple(k.split("/")): jnp.asarray(v) for k, v in load_file(str(path / "params.safetensors")).items()})
        opt_state = self._opt_state_from_arrays(load_file(str(path / "optimizer.safetensors")))
        meta = json.loads((path / "meta.json").read_text())
        self.params = jax.device_put(params, replicated)
        self.opt_state = jax.device_put(opt_state, replicated)
        self.step = int(meta["step"])
        self.best_val_loss = float(meta.get("best_val_loss", meta["val_loss"]))
        print(f"Loaded checkpoint from step {self.step} (best val_loss={self.best_val_loss:.4f})", flush=True)

    def _log(self, row: dict) -> None:
        self.log.append(row)
        (self.log_dir / "training_log.json").write_text(json.dumps(self.log, indent=2))

    def train(self, sampler, val_path: str, eval_batches: int) -> None:
        cfg = self.cfg
        log_path = self.log_dir / "training_log.json"
        if log_path.exists():
            self.log = [e for e in json.loads(log_path.read_text()) if e.get("step", 0) <= self.step]
        print(f"Training from step {self.step} to {cfg.max_steps} on {len(self.devices)} device(s)", flush=True)
        loss, dt = float("nan"), 0.0
        while self.step < cfg.max_steps:
            t0 = time.time()
            x, y = sampler.batch(self.step * cfg.batch_size, cfg.batch_size, self.step / max(1, cfg.max_steps))
            loss = self.train_step(x, y)
            dt = time.time() - t0
            print(f"step {self.step} loss {loss:.4f} lr {lr_at(self.step - 1, cfg):.2e} "
                  f"dt {dt:.1f}s tok/s {x.size / dt:,.0f}", flush=True)
            eval_due = self.step % cfg.eval_interval == 0
            save_due = self.step % cfg.save_interval == 0
            if eval_due or save_due:
                val_loss = self.eval_loss(val_path, eval_batches)
            if eval_due:
                print(f"  Step {self.step} | val_loss={val_loss:.4f} | train_loss={loss:.4f}", flush=True)
                self._log({"step": self.step, "train_loss": loss, "val_loss": val_loss,
                           "lr": lr_at(self.step - 1, cfg), "time": dt})
            if save_due:
                self.save(val_loss)
            if cfg.lr_schedule == "wsd" and self.step == int(cfg.max_steps * (1 - cfg.wsd_decay_frac)):
                pre = self.ckpt_dir / "pre_decay"
                self.save(float("inf"))
                if pre.exists() or pre.is_symlink():
                    pre.unlink()
                pre.symlink_to(f"checkpoint_step_{self.step}")
            if self.stop_requested:
                print(f"Stop requested; saving step {self.step} and exiting.", flush=True)
                self.save(float("inf"))
                return
        val_loss = self.eval_loss(val_path, eval_batches)
        print(f"Final val_loss: {val_loss:.4f}", flush=True)
        if self.step % cfg.save_interval:
            self.save(val_loss)
        self._log({"step": self.step, "train_loss": loss, "val_loss": val_loss,
                   "lr": lr_at(self.step - 1, cfg), "time": dt, "status": "complete"})


def main() -> None:
    a = parse_args()
    cfg = build_config(a)
    print(cfg, flush=True)
    params = init_params(cfg, a.seed)
    print(f"Model parameters: {param_count(params):,} ({param_count(params) / 1e6:.1f}M), compute dtype {a.dtype}, "
          f"devices {[d.platform for d in jax.devices()]}", flush=True)
    trainer = Trainer(cfg, params)
    trainer.sync_to = a.sync_to
    trainer.save_torch = a.save_torch
    if a.curriculum_manifest:
        curriculum = parse_curriculum_spec(a.curriculum_spec) if a.curriculum_spec else DEFAULT_GENERALIST_CURRICULUM
        sampler = CurriculumSampler(load_manifest(a.curriculum_manifest), cfg.context_length, curriculum, seed=a.seed)
    else:
        sampler = SingleStreamSampler(a.train_data, cfg.context_length, seed=a.seed)
    if a.resume:
        trainer.load(a.resume)

    def request_stop(signum, frame):
        trainer.stop_requested = True

    signal.signal(signal.SIGTERM, request_stop)
    trainer.train(sampler, a.val_data, a.eval_batches)


if __name__ == "__main__":
    main()
