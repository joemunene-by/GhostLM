"""End-to-end training smoke test (issue #16).

Existing tests cover model forward pass, attention mechanics, tokenizer
behavior, and weight tying. None of them actually exercise the full
training loop. If a PR breaks the backward pass, the optimizer step,
the LR schedule, or the gradient flow in a subtle way, the unit tests
all pass and we don't catch it.

This smoke test runs a tiny end-to-end training loop on dummy data
and asserts loss decreases. It uses the smallest possible config so
it stays under a few seconds on CPU and won't spook CI.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ghostlm.config import GhostLMConfig
from ghostlm.model import GhostLM


def test_loss_decreases_on_dummy_data():
    """5 optimizer steps on synthetic data should drive loss down.

    A flat or rising loss means something in the train loop is broken:
    the backward pass, the optimizer.step, the LR schedule, or the
    forward pass itself. Catching that here is the cheapest possible
    end-to-end signal.
    """
    torch.manual_seed(0)

    cfg = GhostLMConfig()
    cfg.vocab_size = 256
    cfg.n_layers = 2
    cfg.d_model = 64
    cfg.n_heads = 4
    cfg.d_ff = 128
    cfg.context_length = 32
    cfg.dropout = 0.0
    cfg.use_rope = False

    model = GhostLM(cfg)
    model.train()

    optim = torch.optim.AdamW(model.parameters(), lr=1e-2)

    seq_len = 32
    batch = 4
    x = torch.randint(0, cfg.vocab_size, (batch, seq_len))
    y = torch.randint(0, cfg.vocab_size, (batch, seq_len))

    losses = []
    for _ in range(5):
        logits, _ = model(x)
        loss = F.cross_entropy(
            logits.reshape(-1, cfg.vocab_size),
            y.reshape(-1),
        )
        optim.zero_grad()
        loss.backward()
        optim.step()
        losses.append(loss.item())

    assert all(t == t for t in losses), f"NaN loss in trajectory: {losses}"
    assert losses[-1] < losses[0], (
        "Training loss did not decrease over 5 steps. "
        f"start={losses[0]:.4f} end={losses[-1]:.4f} full={losses}"
    )


def test_grad_flows_through_all_blocks():
    """Every parameter that requires grad should receive a non-None grad
    after a single backward pass. Catches accidental disconnections in
    the residual stream (e.g. forgetting to wire a new norm into the
    block forward)."""
    torch.manual_seed(0)

    cfg = GhostLMConfig()
    cfg.vocab_size = 256
    cfg.n_layers = 2
    cfg.d_model = 64
    cfg.n_heads = 4
    cfg.d_ff = 128
    cfg.context_length = 16
    cfg.dropout = 0.0
    cfg.use_rope = False

    model = GhostLM(cfg)
    model.train()

    x = torch.randint(0, cfg.vocab_size, (2, 16))
    logits, _ = model(x)
    loss = logits.mean()
    loss.backward()

    missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, f"Parameters with no gradient: {missing}"
