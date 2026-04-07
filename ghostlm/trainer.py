"""GhostLM trainer — handles the full training loop, evaluation, checkpointing, and logging."""

import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional, Tuple

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

        self.model = self.model.to(self.device)

        # Mixed precision (AMP) — only effective on CUDA
        if use_amp is None:
            self.use_amp = self.device == "cuda"
        else:
            self.use_amp = use_amp and self.device == "cuda"

        self.grad_scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        # Optimizer
        self.optimizer = self.model.configure_optimizers(config)

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
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                _, loss = self.model(mx, targets=my)
                # Scale loss by number of accumulation steps
                scaled_loss = loss / len(micro_x)

            self.grad_scaler.scale(scaled_loss).backward()
            total_loss += loss.item()

        # Gradient clipping and optimizer step after accumulation
        self.grad_scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()
        self.optimizer.zero_grad(set_to_none=True)

        self.step += 1
        self._set_lr()

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

                with torch.amp.autocast("cuda", enabled=self.use_amp):
                    _, loss = self.model(x, targets=y)
                total_loss += loss.item()
                count += 1

        return total_loss / max(count, 1)

    def save_checkpoint(self, val_loss: float) -> None:
        """Save a model checkpoint to disk.

        Saves the current step, validation loss, model state dict, optimizer
        state dict, and config. Also saves as "best_model.pt" if the current
        validation loss is the best seen so far.

        Args:
            val_loss: Current validation loss for comparison.
        """
        checkpoint = {
            "step": self.step,
            "val_loss": val_loss,
            "model_state_dict": self.model.state_dict(),
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

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "grad_scaler_state_dict" in checkpoint:
            self.grad_scaler.load_state_dict(checkpoint["grad_scaler_state_dict"])
        self.step = checkpoint["step"]
        self.best_val_loss = checkpoint["val_loss"]

        print(f"Loaded checkpoint from step {self.step} (val_loss={self.best_val_loss:.4f})")

    def _log(self, data: dict) -> None:
        """Append a data dict to the training log and persist as JSON.

        Args:
            data: Dictionary of metrics and metadata to log.
        """
        self.log.append(data)
        log_path = self.log_dir / "training_log.json"
        with open(log_path, "w") as f:
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
        print(f"Training on device: {self.device}")
        print(f"Mixed precision (AMP): {'enabled' if self.use_amp else 'disabled'}")
        print(f"Model size: {self.model.num_params():,} parameters")
        print(f"Training from step {self.step} to {self.config.max_steps}")

        # Create iterator that cycles through train_loader
        def cycle(loader):
            while True:
                for batch in loader:
                    yield batch

        train_iter = cycle(train_loader)

        with tqdm(initial=self.step, total=self.config.max_steps, desc="Training") as pbar:
            while self.step < self.config.max_steps:
                t0 = time.time()

                # Training step
                batch = next(train_iter)
                loss = self.train_step(batch)

                dt = time.time() - t0
                lr = self.get_lr()

                pbar.set_postfix(loss=f"{loss:.4f}", lr=f"{lr:.2e}", dt=f"{dt:.3f}s")
                pbar.update(1)

                # Periodic evaluation
                if self.step % self.config.eval_interval == 0:
                    val_loss = self.eval_step(val_loader)
                    print(f"\n  Step {self.step} | val_loss={val_loss:.4f} | train_loss={loss:.4f}")

                    self._log({
                        "step": self.step,
                        "train_loss": loss,
                        "val_loss": val_loss,
                        "lr": lr,
                        "time": dt,
                    })

                # Periodic checkpoint
                if self.step % self.config.save_interval == 0:
                    val_loss = self.eval_step(val_loader)
                    self.save_checkpoint(val_loss)

        # Final evaluation and checkpoint
        print("\nTraining complete. Running final evaluation...")
        val_loss = self.eval_step(val_loader)
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

        print(f"Training log saved to {self.log_dir / 'training_log.json'}")
