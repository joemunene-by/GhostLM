# Training on Cloud TPU

`ghostlm/tpu/` is a JAX port of the ghost-base model and trainer, written for the
TPU Research Cloud run. It reproduces `scripts/train_ghost_base_mlx.py`: the same
architecture switches (RoPE, SwiGLU, RMSNorm, GQA, QK-norm, intra-document masking,
attention gate, value residual), the same warmup plus cosine or WSD schedule, the
same AdamW, Muon and NorMuon optimizers with cautious weight decay, the same
curriculum sampler with per-sample seeding, and the same eval and checkpoint cadence.
MoE is not ported yet.

Weights convert by name: `ghostlm.tpu.model.to_torch_state` and `from_torch_state`
move a checkpoint between this trainer, the PyTorch tools and the MLX trainer.
`tests/test_tpu_port.py` checks the logits and the loss against the PyTorch model on
a tiny shape, and the sampler against `MultiDomainBinDataset`.

## Environment

JAX, Flax and PyTorch do not share a wheel set with MLX, so the port lives in its
own environment. Python 3.13 (3.14 has no JAX or PyTorch wheels yet):

```bash
/opt/homebrew/bin/python3.13 -m venv .venv-jax
.venv-jax/bin/pip install "jax[cpu]" flax optax safetensors torch tiktoken pytest
.venv-jax/bin/pytest tests/test_tpu_port.py
```

On a TPU VM replace `jax[cpu]` with `jax[tpu]`.

## CPU smoke test

A tiny shape on the local token files, to prove the loop end to end:

```bash
.venv-jax/bin/python -m ghostlm.tpu.train --run-name smoke --n-layers 2 --d-model 64 \
    --n-heads 4 --n-kv-heads 2 --d-ff 128 --context-length 64 --batch-size 8 \
    --max-steps 20 --warmup-steps 2 --eval-interval 10 --save-interval 10 --dtype float32
```

## Free TPU before the grant

Kaggle gives 20 hours a week on a TPU v3-8 and Colab sometimes offers a v2-8. Clone
the repo in the notebook, install `jax[tpu]`, copy a few hundred megabytes of `.bin`
shards into the VM, and run the same command with the ghost-base shape and a small
step count. The point is to see all eight cores in `jax.devices()`, a sane
tokens-per-second figure, and a checkpoint written and resumed.

## On the TRC slice

1. Copy the tokenised shards from the bucket to the VM's local disk: `gsutil -m rsync -r gs://<bucket>/processed data/processed`. The sampler memory-maps local files.
2. Write a manifest `{"general_web": "train.general_web.bin", ...}` next to the shards.
3. Launch under `nohup` or a systemd unit so an SSH drop does not stop it:

```bash
python -m ghostlm.tpu.train --curriculum-manifest data/processed/domains.json \
    --run-name ghost_base_tpu --lr-schedule wsd --optimizer normuon --cautious-wd \
    --attn-gate --value-residual --batch-size 256 --grad-accum-steps 1 \
    --max-steps 60000 --warmup-steps 2000 --learning-rate 5e-4 \
    --save-interval 500 --eval-interval 250 --sync-to gs://<bucket>/checkpoints/ghost_base_tpu
```

4. Preemption: `--sync-to` mirrors every checkpoint to the bucket; on a new VM, rsync the
   folder back and pass `--resume checkpoints/ghost_base_tpu/checkpoint_step_N`.

Checkpoints are directories: `params.safetensors`, `optimizer.safetensors`, `meta.json`.
`--save-torch` additionally writes `checkpoint_step_N.pt` in the PyTorch layout so
`scripts/scorecard.py` and the serving stack can read it without conversion.

## Not done yet

- MoE experts (the BTX merge step) in JAX.
- Multi-host slices (v4-32 and larger) need `jax.distributed.initialize` and a process-sharded sampler; the single-host data-parallel path is what exists.
- Gradient checkpointing is wired through `--grad-checkpoint` but untested at the 3B shape.
