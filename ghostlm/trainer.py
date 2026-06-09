"""GhostLM trainer — handles the full training loop, evaluation, checkpointing, and logging."""

import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Tuple

import torch
from tqdm import tqdm

from ghostlm.config import GhostLMConfig
from ghostlm.model import GhostLM


class GhostTrainer:
    """Manages the GhostLM training loop with evaluation, checkpointing, and logging.

    Handles device placement, optimizer setup, cosine learning rate scheduling
    with warmup, gradient clipping, periodic evaluation, checkpoint saving,
    and JSON-based training log persistence. Supports mixed precision (AMP)
    training on CUDA devices for faster throughput and lower memory usage.
    """

    def __init__(self, model: GhostLM, config: GhostLMConfig, use_amp: Optional[bool] = None):
        """Initialize the trainer.

        Args:
            model: GhostLM model instance to train.
            config: GhostLMConfig with training hyperparameters and paths.
            use_amp: Enable mixed precision (AMP) training. Defaults to True
                when running on CUDA, False otherwise. AMP is only supported
                on CUDA devices — setting True on CPU/MPS will be ignored.
        """
        self.model = model
        self.config = config

        # Resolve device
        if config.device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = config.device

        # Distributed training support (issue #8). Detect whether we are
        # running inside torchrun / torch.distributed.launch by reading the
        # standard env vars; if so, set the local-rank device and wrap the
        # model in DistributedDataParallel after moving to device.
        # Single-GPU / CPU training is the default and unchanged.
        self.is_distributed = (
            "RANK" in os.environ
            and "WORLD_SIZE" in os.environ
            and int(os.environ.get("WORLD_SIZE", "1")) > 1
        )
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.global_rank = int(os.environ.get("RANK", "0"))
        self.is_main_process = self.global_rank == 0

        if self.is_distributed:
            import torch.distributed as dist
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            if not dist.is_initialized():
                dist.init_process_group(backend=backend)
            if torch.cuda.is_available():
                torch.cuda.set_device(self.local_rank)
                self.device = f"cuda:{self.local_rank}"

        self.model = self.model.to(self.device)

        # Mixed precision (AMP), only effective on CUDA
        if use_amp is None:
            self.use_amp = self.device.startswith("cuda")
        else:
            self.use_amp = use_amp and self.device.startswith("cuda")

        # AMP dtype. bfloat16 has the same exponent range as float32, so
        # it needs no loss scaling and is the stabler choice for
        # pretraining on Ampere+ GPUs. Honour an explicit
        # ``config.dtype: "float16"`` request; otherwise prefer bf16
        # whenever the hardware supports it.
        if self.use_amp and config.dtype != "float16" and torch.cuda.is_bf16_supported():
            self.amp_dtype = torch.bfloat16
        else:
            self.amp_dtype = torch.float16

        # GradScaler is only needed for float16 (bf16 doesn't overflow
        # gradients the way fp16 does). With enabled=False it's a no-op
        # passthrough, so the train-step code stays uniform.
        self.grad_scaler = torch.amp.GradScaler(
            "cuda", enabled=self.use_amp and self.amp_dtype == torch.float16,
        )

        # Optimizer (built BEFORE wrapping in DDP so param groups see raw modules)
        self.optimizer = self.model.configure_optimizers(config)

        # DDP wrap. Each rank now sees a self.model that does the all-reduce
        # transparently in backward(). Other code paths that touch
        # self.model.* still work because DDP forwards attribute access.
        if self.is_distributed:
            from torch.nn.parallel import DistributedDataParallel as DDP
            ddp_kwargs = {}
            if torch.cuda.is_available():
                ddp_kwargs["device_ids"] = [self.local_rank]
                ddp_kwargs["output_device"] = self.local_rank
            # MoE: experts that receive no tokens in a micro-batch get no
            # gradient, which DDP treats as an error unless told to scan
            # for unused parameters each backward.
            if getattr(config, "use_moe", False):
                ddp_kwargs["find_unused_parameters"] = True
            self.model = DDP(self.model, **ddp_kwargs)

        # Create directories
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.log_dir = Path(config.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # State
        self.step = 0
        self.accum_steps = getattr(config, 'grad_accum_steps', 4)
        self.best_val_loss = float("inf")
        self.log: list = []

    def get_lr(self) -> float:
        """Compute the current learning rate using cosine decay with linear warmup.

        During the warmup phase (step < warmup_steps), the learning rate scales
        linearly from 0 to config.learning_rate. After warmup, it follows a
        cosine decay schedule down to a minimum of 1e-5.

        Returns:
            Current learning rate as a float.
        """
        step = self.step
        warmup = self.config.warmup_steps
        max_steps = self.config.max_steps
        base_lr = self.config.learning_rate
        min_lr = 1e-5

        if step < warmup:
            return base_lr * (step + 1) / warmup

        decay_ratio = (step - warmup) / max(1, max_steps - warmup)
        decay_ratio = min(decay_ratio, 1.0)

        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + (base_lr - min_lr) * cosine_decay

    def _set_lr(self) -> None:
        """Apply the current learning rate from get_lr() to all optimizer parameter groups."""
        lr = self.get_lr()
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    def train_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> float:
        """Execute a single training step with gradient accumulation and optional AMP.

        Accumulates gradients over self.accum_steps micro-steps before
        updating weights, effectively multiplying the batch size without
        increasing memory usage. When AMP is enabled, the forward pass runs
        in float16 and the GradScaler handles loss scaling for stable training.

        Args:
            batch: Tuple of (input_ids, target_ids) tensors.

        Returns:
            Training loss as a float.
        """
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)

        self.model.train()

        # Split batch into micro-batches for gradient accumulation
        micro_x = x.split(max(1, x.size(0) // self.accum_steps), dim=0)
        micro_y = y.split(max(1, y.size(0) // self.accum_steps), dim=0)

        total_loss = 0.0

        for mx, my in zip(micro_x, micro_y):
            with torch.amp.autocast("cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                _, loss = self.model(mx, targets=my)
                # Scale loss by number of accumulation steps
                scaled_loss = loss / len(micro_x)

            self.grad_scaler.scale(scaled_loss).backward()
            total_loss += loss.item()

        # Apply this step's scheduled LR BEFORE the optimizer update.
        # (Setting it afterwards would make every update use the previous
        # step's LR — and the very first update would run at the full
        # base LR instead of the warmup floor.)
        self._set_lr()

        # Gradient clipping and optimizer step after accumulation
        self.grad_scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()
        self.optimizer.zero_grad(set_to_none=True)

        self.step += 1

        return total_loss / len(micro_x)

    def eval_step(self, val_loader, num_batches: int = 20) -> float:
        """Run evaluation over a number of validation batches.

        Args:
            val_loader: DataLoader yielding (input_ids, target_ids) batches.
            num_batches: Maximum number of batches to evaluate over.

        Returns:
            Average validation loss as a float.
        """
        self.model.eval()
        total_loss = 0.0
        count = 0

        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                if i >= num_batches:
                    break
                x, y = batch
                x = x.to(self.device)
                y = y.to(self.device)

                with torch.amp.autocast("cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                    _, loss = self.model(x, targets=y)
                total_loss += loss.item()
                count += 1

        return total_loss / max(count, 1)

    def save_checkpoint(self, val_loss: float) -> None:
        """Save a model checkpoint to disk.

        Saves the current step, validation loss, model state dict, optimizer
        state dict, and config. Also saves as "best_model.pt" if the current
        validation loss is the best seen so far.

        Under distributed training, only rank 0 writes; the saved state_dict
        unwraps DDP so checkpoints remain compatible with single-GPU loading.

        Args:
            val_loss: Current validation loss for comparison.
        """
        # Only rank 0 writes checkpoints in DDP runs
        if getattr(self, "is_distributed", False) and not self.is_main_process:
            return

        # Unwrap DDP to keep checkpoints loadable on a single GPU
        raw_model = self.model.module if hasattr(self.model, "module") else self.model

        checkpoint = {
            "step": self.step,
            "val_loss": val_loss,
            "model_state_dict": raw_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "grad_scaler_state_dict": self.grad_scaler.state_dict(),
            "config": asdict(self.config),
        }

        filename = f"checkpoint_step_{self.step}.pt"
        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
        print(f"  Saved checkpoint: {path}")

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"  New best model saved: {best_path} (val_loss={val_loss:.4f})")

    def load_checkpoint(self, path: str) -> None:
        """Load a model checkpoint from disk.

        Restores the model state dict, optimizer state dict, training step,
        and best validation loss from the saved checkpoint file.

        Args:
            path: File path to the checkpoint .pt file.
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        # Load into the raw model (works for DDP-wrapped or single-GPU)
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        raw_model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "grad_scaler_state_dict" in checkpoint:
            self.grad_scaler.load_state_dict(checkpoint["grad_scaler_state_dict"])
        self.step = checkpoint["step"]
        self.best_val_loss = checkpoint["val_loss"]

        print(f"Loaded checkpoint from step {self.step} (val_loss={self.best_val_loss:.4f})")

    def _log(self, data: dict) -> None:
        """Append a data dict to the training log and persist it.

        Each entry is appended to ``training_log.jsonl`` as it arrives
        (O(1) per entry), and the legacy ``training_log.json`` array is
        rewritten so existing consumers (plot scripts) keep working.
        Only rank 0 writes under distributed training.

        Args:
            data: Dictionary of metrics and metadata to log.
        """
        self.log.append(data)
        if not self.is_main_process:
            return
        with open(self.log_dir / "training_log.jsonl", "a") as f:
            f.write(json.dumps(data) + "\n")
        with open(self.log_dir / "training_log.json", "w") as f:
            json.dump(self.log, f, indent=2)

    def train(self, train_loader, val_loader) -> None:
        """Run the main training loop.

        Iterates from the current step to config.max_steps, performing training
        steps with a tqdm progress bar. Evaluates periodically at config.eval_interval
        and saves checkpoints at config.save_interval. Performs a final evaluation
        and saves the final checkpoint at the end of training.

        Args:
            train_loader: DataLoader yielding (input_ids, target_ids) training batches.
            val_loader: DataLoader yielding (input_ids, target_ids) validation batches.
        """
        if self.is_main_process:
            print(f"Training on device: {self.device}")
            amp_desc = (
                f"enabled ({'bf16' if self.amp_dtype == torch.bfloat16 else 'fp16'})"
                if self.use_amp else "disabled"
            )
            print(f"Mixed precision (AMP): {amp_desc}")
            print(f"Model size: {self.model.num_params():,} parameters")
            print(f"Training from step {self.step} to {self.config.max_steps}")

        # Create iterator that cycles through train_loader. Under DDP the
        # DistributedSampler must be told the epoch number, otherwise
        # every pass (and every rank reseed) reuses the epoch-0 shuffle.
        def cycle(loader):
            epoch = 0
            while True:
                sampler = getattr(loader, "sampler", None)
                if hasattr(sampler, "set_epoch"):
                    sampler.set_epoch(epoch)
                for batch in loader:
                    yield batch
                epoch += 1

        train_iter = cycle(train_loader)

        with tqdm(initial=self.step, total=self.config.max_steps, desc="Training",
                  disable=not self.is_main_process) as pbar:
            while self.step < self.config.max_steps:
                t0 = time.time()

                # Training step
                batch = next(train_iter)
                loss = self.train_step(batch)

                dt = time.time() - t0
                lr = self.get_lr()

                pbar.set_postfix(loss=f"{loss:.4f}", lr=f"{lr:.2e}", dt=f"{dt:.3f}s")
                pbar.update(1)

                # Periodic evaluation / checkpointing. When a step hits
                # both intervals, evaluate once and reuse the result.
                eval_due = self.step % self.config.eval_interval == 0
                save_due = self.step % self.config.save_interval == 0

                if eval_due or save_due:
                    val_loss = self.eval_step(val_loader)

                if eval_due:
                    if self.is_main_process:
                        print(f"\n  Step {self.step} | val_loss={val_loss:.4f} | train_loss={loss:.4f}")

                    self._log({
                        "step": self.step,
                        "train_loss": loss,
                        "val_loss": val_loss,
                        "lr": lr,
                        "time": dt,
                    })

                if save_due:
                    self.save_checkpoint(val_loss)

        # Final evaluation and checkpoint
        if self.is_main_process:
            print("\nTraining complete. Running final evaluation...")
        val_loss = self.eval_step(val_loader)
        if self.is_main_process:
            print(f"Final val_loss: {val_loss:.4f}")
        self.save_checkpoint(val_loss)

        self._log({
            "step": self.step,
            "train_loss": loss,
            "val_loss": val_loss,
            "lr": lr,
            "time": dt,
            "status": "complete",
        })

        if self.is_main_process:
            print(f"Training log saved to {self.log_dir / 'training_log.json'}")
