# Bet 5 (MoE) training smoke result

## What this validates

The aux-loss wiring landed in commit `b1015b3` with single-forward-pass
tests: does the model construct, does total loss equal CE plus
`coef * sum(aux)`, do gradients reach the gate. None of those tests
ran the optimizer. They didn't tell us whether 100 backward-and-step
iterations stay stable, whether the router's load-balancing pressure
holds, or whether expert weights drift in a way that breaks training.

This smoke fills that gap. It runs 100 SGD steps on a tiny synthetic
dataset using `from_preset("ghost-tiny")`-shaped MoE config (4
layers, 64 d_model, 4 heads, 4 experts top-2). The data is fixed
random tokens, intentional: the point isn't to learn anything
language-y, it's to see whether the training loop is well-formed.

## Headline

**PASS across all four criteria** (commit `XXX`, 2026-05-08, CPU,
seed 42, ~5.6 min wall clock):

| Metric | Step 1 | Step 100 | Verdict |
|---|---:|---:|:---:|
| Total loss (CE + 0.01 * sum(aux)) | 5.632 | 0.843 | PASS |
| Cross-entropy alone | 5.552 | 0.762 | PASS |
| Mean aux per layer | 2.004 | 2.001 | PASS |
| Mean gate gradient norm | 2.4e-3 | 2.0e-2 | PASS |

The model successfully memorises the random target (CE drops 86%).
Aux stays glued to ~2.0 across all 100 steps, which is exactly the
uniform-routing equilibrium for 4 experts top-2: with `f_i = 0.5`
(half of tokens reach each expert through the top-2 path) and
`p_i = 0.25` (uniform router probabilities), the formula
`(f_i * p_i).sum() * n_experts = 4 * (4 * 0.125) = 2.0`. Random
inputs give the gate no signal to specialise, so the load-balancing
pressure keeps it pinned at uniform. That is the right behaviour
for this no-signal stress test.

The gate gradients grow ~10x from step 1 to step 100 (2.4e-3 to
2.0e-2), confirming the optimizer is actively shaping the router,
not just the experts.

## What this does NOT tell us

This is a unit-test grade smoke, not a training validation. It
specifically does not tell us:

- Whether MoE wins over a dense FFN at any param count (needs a real
  language pretrain run with a held-out eval).
- Whether the router specialises on real cybersec data (needs at
  least a few thousand steps on the actual corpus + a routing
  histogram analysis per expert).
- Whether `n_experts=4 top_k=2` is the right shape for ghost-1B
  (needs an architecture-search ablation or a confidence call from
  the literature; both are out of scope for this smoke).

What it does establish: the wiring is correct end-to-end, the
optimizer can step a MoE model without NaNs or aux-loss explosions,
and the router gets a learning signal it can act on. That's the
floor that needed to be on the floor before the user spends real
GPU hours on bet 5.

## Reproducing

```bash
PYTHONPATH=. python3 scripts/smoke_train_moe.py --device cpu
# or --device mps on Mac, --device cuda on a GPU host
```

The script writes a per-step log to `logs/moe_smoke_<timestamp>.md`
plus the raw per-step JSON next to it for plotting. Re-running with
a different `--seed` should land the same qualitative shape (CE
drops monotonically, aux stays near 2.0); regression-checking a
future code change against this is one of the smoke's main jobs.

## Pass/fail criteria, in code

```python
ce_dropped         = last["ce"] < first["ce"] - 0.1
aux_finite         = all(0 < r["aux_mean"] < 100 for r in log)
grads_nonzero      = all(r["gate_grad_norm_mean"] > 0 for r in log)
aux_at_end_lower   = last["aux_mean"] <= first["aux_mean"] + 0.5
```

The thresholds are intentionally generous; the smoke is meant to
catch broken-broken (NaN, exploding aux, dead gates), not subtle
regressions. Subtle regressions are the job of a real eval on real
data, which is gated on real GPU time and not covered here.
