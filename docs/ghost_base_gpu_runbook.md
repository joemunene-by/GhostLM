# ghost-base GPU run — rented-box runbook

Step-by-step for executing the ghost-base v1.0 pretrain on a rented
GPU instance, written so the meter isn't running while we figure
things out. Pair with [`ghost_base_spec.md`](ghost_base_spec.md) (what
we're training and why), [`hardware_pathway.md`](hardware_pathway.md)
(the longer-term hardware picture), and [`distributed.md`](distributed.md)
(only needed if we rent more than one GPU, which this run does not).

Context that shaped this plan: the local Mac no longer holds the full
v1.0 corpus (`data/raw` has the 42 long-tail sources, ~560 MB, but the
seven big files — primus_seed, primus_fineweb, fineweb_edu,
math_reasoning, code_corpus, cve_full, arxiv_full — were never on this
disk; see CORPUS.md). The Mac is also tight on free space. So the
corpus is rebuilt **on the rented box**, which has more disk and far
better bandwidth, and only the checkpoint comes back.

## Budget

| Phase | Wall clock | Cost driver |
|---|---|---|
| Corpus pull + rebuild + pretokenize | 2-4 h | mostly bandwidth; NVD is the long pole |
| Smoke run (200 steps) | ~10 min | |
| Real run (30k steps ≈ 2.0B tokens) | 4-8 h on one H100 | ~$2-3/h |
| Evals + artifact download | <1 h | |
| **Total** | **one working day** | **~$20-40, padded** |

30k steps × 64 seqs/step × 1024 ctx = ~2.0B tokens ≈ 4.7 epochs over
the 422M-token corpus. 6·N·D ≈ 6 × 0.349e9 × 2e9 ≈ 4.2e18 FLOPs; an
H100 at 35-45% MFU does that in roughly 3-4 h of pure math, so 4-8 h
wall clock with eval/checkpoint overhead is the honest range.

## Phase 0 — before renting anything (free, do at leisure)

1. **NVD API key.** Request at
   <https://nvd.nist.gov/developers/request-an-api-key> (free, emailed).
   Without it the full CVE pull is rate-limited to ~5 requests/30s and
   takes many hours; with it, well under an hour. This is the single
   biggest schedule risk, so get the key before booking the box.
2. **wandb API key** ready (`wandb.ai/authorize`). Live loss curves are
   how we decide to kill a bad run early instead of paying for it.
3. **SSH key** registered with the provider.
4. Repo pushed to GitHub (the box clones from there, not from the Mac).
5. Pre-pack the long-tail raw data on the Mac so the upload is one file:

   ```bash
   cd ~/Desktop/GhostLM
   tar czf /tmp/ghostlm_raw.tar.gz data/raw data/code_corpus_repos.json
   ls -lh /tmp/ghostlm_raw.tar.gz   # expect roughly 150-250 MB
   ```

## Phase 1 — provision

- **One H100 80GB** (RunPod secure cloud, Lambda on-demand, or Vast
  with a high-reliability host). A100 80GB also works, expect ~1.7x the
  step time. Spot/community pricing is fine **only because** the
  trainer checkpoints every 1500 steps and `--resume` works — but
  on-demand at ~$2.5/h costs maybe $10 more in total and removes the
  babysitting; take on-demand unless the spot discount is large.
- **Disk: 150 GB+.** Raw pulls + processed corpus + .bin files +
  checkpoints add up to ~40-60 GB; 150 leaves room for carelessness.
- **Image:** any recent PyTorch 2.x / CUDA 12.x image, or plain Ubuntu
  22.04 + Python 3.10+.
- First command on the box: `tmux`. Every long step below runs inside
  it so an SSH drop costs nothing.

## Phase 2 — setup (~10 min)

```bash
tmux
git clone https://github.com/joemunene-by/GhostLM.git && cd GhostLM
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install torch                      # CUDA wheel from PyPI default index
pip install -e ".[train,export,data,dev]"
python -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.is_available())"
wandb login        # paste the key from Phase 0
export NVD_API_KEY=...   # from Phase 0
```

Do NOT use `make install` here — its torch line pins the CPU wheel
(it exists for CI). Quick sanity: `make test` should pass (~1 min).

## Phase 3 — upload the long-tail raw data (~5 min)

From the **Mac**:

```bash
scp /tmp/ghostlm_raw.tar.gz <box>:~/GhostLM/
```

On the **box**:

```bash
cd ~/GhostLM && tar xzf ghostlm_raw.tar.gz && ls data/raw | wc -l   # expect ~45
```

## Phase 4 — pull the seven missing big sources (1-3 h)

Run the slow ones in parallel tmux panes; none depend on each other.

```bash
# Pane 1 — NVD full pull (the long pole; needs NVD_API_KEY exported)
python scripts/collect_nvd_full.py

# Pane 2 — PRIMUS seed + fineweb (HuggingFace datasets)
python scripts/collect_primus.py

# Pane 3 — FineWeb-Edu + open-web-math (HF, sequential is fine)
python scripts/collect_fineweb_edu.py
python scripts/collect_math_reasoning.py

# Pane 4 — code corpus (clones ~120 repos; bursty network)
python scripts/collect_code_corpus.py --config data/code_corpus_repos.json

# Pane 5 — arXiv cs.CR full text (PDF pulls, polite rate limit)
python scripts/collect_arxiv_full.py
```

Expected outputs in `data/raw/`: `cve_full.jsonl`, `primus_seed.jsonl`,
`primus_fineweb.jsonl`, `fineweb_edu.jsonl`, `math_reasoning.jsonl`,
`code_corpus.jsonl`, `arxiv_full.jsonl`.

## Phase 5 — rebuild + audit (~15 min)

```bash
python scripts/rebuild_corpus.py --max-cve-tokens 6000000
```

The `--max-cve-tokens` cap is not optional — without it NVD's ~27M
tokens swamp the mix (see CORPUS.md). Gate checks before proceeding:

- ~768K train / ~40K val records (v0.9.32 reference: 768,741 / 40,429;
  drift of a few percent is fine since live sources move)
- the printed **leakage check must be 0**
- `python scripts/data_stats.py` per-source shares roughly match the
  CORPUS.md v0.9.32 table (code ~11-12%, cybersec ~65%)

## Phase 6 — pretokenize (~15 min)

```bash
python scripts/pretokenize.py   # defaults: data/processed/{train,val}.jsonl -> .bin
```

Produces `data/processed/train.bin` + `val.bin` with `.meta.json`
sidecars; the trainer memory-maps these.

## Phase 7 — smoke, then the real run

Smoke (~10 min, no wandb noise):

```bash
python scripts/train_ghost_base.py \
  --train-data data/processed/train.bin --val-data data/processed/val.bin \
  --max-steps 200 --eval-interval 100 --run-name ghost_base_smoke
```

Check the startup config print shows **n_kv_heads 5, use_qk_norm True,
use_flash_attn True, ~349M params** (the v0.9.35+ launcher sets these;
if any is missing, the box has a stale checkout). Watch `nvidia-smi`:
if memory is comfortably under 40 GB at `--batch-size 16`, try 32 with
`--grad-accum-steps 2` (same 64-seq effective batch, fewer optimizer
stalls). Then:

```bash
python scripts/train_ghost_base.py \
  --train-data data/processed/train.bin --val-data data/processed/val.bin \
  --wandb --compile --dtype bfloat16 \
  --run-name ghost_base_v1
```

- Checkpoints land in `checkpoints/ghost_base_v1/` every 1500 steps;
  resume after any interruption with `--resume <last ckpt>.pt`.
- First step stalls ~1-2 min on torch.compile; that's normal.
- Sanity curve: loss starts ~10.8 (ln 50264) and should be well under
  4 by a few thousand steps. A flat or NaN curve in the first 500
  steps means kill it and diagnose — that's exactly what wandb is for.

## Phase 8 — acceptance evals (on the box, GPU makes them quick)

The spec's gate is any one of:

```bash
# >=40% per-perm avg, debiased CTIBench
python scripts/eval_debiased.py --checkpoint checkpoints/ghost_base_v1/best_model.pt

# >=30% on the 50-question fact-recall set (the truth metric)
python scripts/eval_fact_recall_v2.py --checkpoints checkpoints/ghost_base_v1/best_model.pt

# >=65% on the in-repo CTF eval
python scripts/run_bench.py   # see script header for checkpoint args
```

## Phase 9 — bring the artifacts home

From the **Mac** (best_model.pt at this scale is ~1.4 GB fp32 /
~0.7 GB bf16 — check local free space first, the Mac is tight):

```bash
rsync -avP <box>:~/GhostLM/checkpoints/ghost_base_v1/best_model.pt \
  ~/Desktop/GhostLM/checkpoints/ghost_base_v1/
rsync -avP <box>:~/GhostLM/logs/ghost_base_v1/ ~/Desktop/GhostLM/logs/ghost_base_v1/
```

Optionally also pull `data/processed/{train,val}.bin` later once the
SATA SSD is installed, so future local work has the corpus; do not pull
them to the internal disk now.

## Phase 10 — teardown checklist

1. Eval numbers recorded (RESULTS.md entry drafted or raw numbers saved).
2. best_model.pt + training log rsynced and **verified openable
   locally** (`torch.load` of the checkpoint on the Mac).
3. wandb run finished/synced (it lives in the cloud regardless).
4. Instance **destroyed**, not just stopped — stopped instances still
   bill for disk.

## Known pitfalls

- `make install` installs CPU torch; use the Phase 2 commands instead.
- Single GPU means no `torchrun` — plain `python` invocation; DDP only
  activates under torchrun env vars.
- The NVD pull without an API key can eat half a day. Get the key
  before renting (Phase 0).
- If the smoke run's param count prints ~347M instead of ~349M or
  n_kv_heads is missing from the config dump, the checkout predates
  the GQA launcher change — `git pull`.
