# Distributed training (issue #8)

GhostLM's trainer supports `torch.nn.parallel.DistributedDataParallel`
(DDP) when launched via `torchrun`. Single-GPU and CPU training paths
are unchanged: the trainer only enters DDP mode when the standard
torchrun env vars (`RANK`, `WORLD_SIZE`, `LOCAL_RANK`) are present
with `WORLD_SIZE > 1`.

## Quick launch

Single-node, 4 GPUs:

```bash
torchrun \
  --standalone \
  --nproc-per-node 4 \
  scripts/train.py \
  --preset ghost-small \
  --device cuda \
  --batch-size 16  # per-GPU batch
```

Multi-node, 2 nodes × 8 GPUs each (run on each node, with the right
`--node-rank`):

```bash
torchrun \
  --nnodes 2 \
  --nproc-per-node 8 \
  --node-rank ${NODE_RANK} \
  --master-addr ${MASTER_ADDR} \
  --master-port 29500 \
  scripts/train.py \
  --preset ghost-base \
  --device cuda \
  --batch-size 8
```

## What the trainer does in DDP mode

- Reads `RANK`, `WORLD_SIZE`, `LOCAL_RANK` from env on `__init__`.
- Calls `dist.init_process_group(backend="nccl")` (or `"gloo"` on CPU).
- Pins the CUDA device to `LOCAL_RANK` and wraps the model in DDP.
- Only rank 0 writes checkpoints. The saved `state_dict` is unwrapped
  (no `module.` prefix) so checkpoints stay compatible with
  single-GPU loading via `scripts/run_bench.py`, etc.
- The dataloader is shared across ranks for now; for very large
  corpora you'll want to pass a `DistributedSampler` to
  `build_dataloaders` so each rank sees a different shard.

## Limitations

- Apple Silicon MPS does not support DDP. Use single-process MPS for M4.
- The `DistributedSampler` integration is not wired into
  `build_dataloaders` yet, so each rank sees the same data; for short
  ablation runs this is fine, for serious multi-node training it is
  not. Open follow-up: data-parallel sampling.
- `scripts/finetune_chat.py` and the `train_v05/06/07.py` launchers
  inherit the trainer's DDP support, but their CLI arg parsing has
  not been audited for DDP-specific overrides.

## When to use this

- ghost-base (~350M params) on rented multi-GPU hardware.
- ghost-1B at the long-term rung.
- Anything where a single GPU is no longer enough to hold the model
  + activations + optimizer state at a reasonable batch size.
