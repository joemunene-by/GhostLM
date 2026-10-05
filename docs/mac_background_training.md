# Training ghost-base in the background on a Mac

The no-budget path to ghost-base: a launchd service that trains on the M4
whenever the machine is otherwise idle, for as many weeks as it takes.

## What to expect

Measured on a 16GB M4 mini, ghost-base (349M) at 65,536 tokens per step
(batch 64 x 1024 context, as 32 micro-batches of 2; `--grad-accum-steps`
splits `--batch-size`, it does not multiply it):

| Backend | Step time | Tokens/s | Memory |
|---|---:|---:|---:|
| MLX (`train_ghost_base_mlx.py`, bf16 compute, fp32 master weights) | 81 s | 805 | 7.3GB, 8.1GB peak |
| PyTorch MPS (`train_ghost_base.py`, fp32) | 160 s | 410 | 7.5GB+ |

The supervisor uses MLX by default (`"backend": "mlx"`). At about 70M tokens per
day of active training, the default 15,000-step run (about 1B tokens) takes
roughly two weeks. A rented H100 does the same in a few hours; this path
trades time for zero cost.

## Setup

```bash
python3 -m venv .venv && .venv/bin/pip install -e '.[train,data,dev]'
caffeinate -i zsh scripts/background/collect_corpus.sh &   # hours; resumable
scripts/background/install.sh                              # starts at every login
```

`collect_corpus.sh` pulls every source, rebuilds the generalist mix, removes
near-duplicates (`scripts/dedup_near.py`), strips benchmark overlap
(`scripts/decontaminate.py`) and pretokenizes by domain. Rerunning it skips
finished sources. The supervisor waits until `data/processed/train.bin`
exists, then starts training on its own.

## What the supervisor does

- Pauses (the trainer checkpoints and exits on SIGTERM) while a wine/cellar
  game runs or `.bg/PAUSE` exists, and resumes 5 minutes after.
- Pauses when free memory stays under 10% for 30 s (checked every 5 s) and
  resumes once it has been above 20% for 5 minutes, so the Mac stays usable.
- Every 1,500 steps, stops briefly to score the latest checkpoint
  (`scripts/scorecard.py`), appends to `.bg/evals.jsonl` and redraws
  `.bg/progress.png`.
- Keeps one rolling checkpoint plus `best_model.pt` (weights only) and, with
  the WSD schedule, `pre_decay.pt` for extending the run later.
- Saves bf16 weight snapshots through the final decay phase and averages them
  into `checkpoints/ghost_base_mac/averaged.pt` when training ends.
- Backs off after crashes, sends one evening progress notification and a
  notification on completion.
- With `backup_repo` set, uploads `best_model.pt` (weights only), `pre_decay.pt`,
  `averaged.pt`, the training log and evals to a private Hugging Face repo
  once a day, only re-sending files that changed (`.bg/logs/backup.log`).
- Once the corpus exists, runs `prepare_post_training.sh` on the CPU: RAG
  index (cybersec + knowledge), chat SFT set, RAFT set and tool-use set.

## Checking on it

- `cat .bg/STATUS.txt`
- Dashboard: `http://<mac-ip>:8090` from the same network, or from anywhere
  through a Twingate resource covering the Mac. Pause/resume asks for the
  token in `.bg/dashboard_token`.
- Logs: `.bg/logs/train.log`, `.bg/logs/eval.log`, `.bg/logs/supervisor.log`.

## Configuration

`.bg/train_config.json` overrides any key of `DEFAULT_CONFIG` in
`scripts/background/supervisor.py` (optimizer, schedule, steps, eval
cadence, `attn_gate`, `value_residual`, ...). It is re-read every minute;
changes that affect the model shape only make sense before the first
checkpoint.

## Extending a finished run

Set a larger `max_steps` and resume from `checkpoints/ghost_base_mac/pre_decay.pt`:
the WSD schedule keeps the learning rate flat until the new decay point.
