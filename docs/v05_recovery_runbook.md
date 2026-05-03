# v0.5 Recovery Runbook

After the 24h pretrain extension finishes (~20:00 EAT 2026-05-03),
follow this sequence to land the chat-tune that beats v0.4 chat-v3.

## Goal

Lift v0.5 chat from CTIBench MCQ **32.5%** → **40-48%** (best-effort
target 50%+). Beat v0.4 chat-v3 (**36.9%**) cleanly.

## Diagnosis (from research-agent report 2026-05-02)

- **Tokenizer collision** on small-talk: domain BPE splits "hi" / "you"
  / "thanks" character-level, no gradient signal connects them to
  assistant-mode behavior.
- **Pretrain undertraining**: 28 tok/param < 80-200 floor for sub-100M
  models per SmolLM2 / Pythia-2025 / MobiLlama / BabyLLaMA-2.
- **MCQ records too parlor-trick**: letter-only training teaches "after
  Answer: emit a letter" not the underlying knowledge — doesn't transfer.

## Steps

### 0. Verify pretrain finished cleanly

```bash
ssh ghostlm-mac 'tail -3 /tmp/v05_pretrain_ext.log; ls -la ~/Desktop/GhostLM/checkpoints/phase6_v05_pretrain/best_model.pt'
```

Expected: `v0.5 pretrain complete.` in log; best_model.pt mtime
matches the end of the run.

### 1. Restart Ollama (Qwen-14B already pulled)

(Killed earlier to free MPS for pretrain — safe to restart now.)

```bash
ssh ghostlm-mac 'open -a Ollama'
sleep 30  # let it start
ssh ghostlm-mac 'ollama list | grep qwen2.5'
```

Verify `qwen2.5:14b` is listed. No pull needed — already cached
(~8 GB on disk).

### 2. Generate CoT-templated MCQ data

```bash
ssh ghostlm-mac 'cd ~/Desktop/GhostLM && PYTHONPATH=. python3 scripts/build_mcq_cot_data.py \
    --in-mcq data/raw/chat/mcq.jsonl \
    --out data/raw/chat/mcq_cot.jsonl \
    --model qwen2.5:14b'
```

Resume-safe (skips already-done records). Expect ~2.5h for 1.8K records
on M4 + Qwen-14B at ~15-20 tok/sec. The 14B's narrative quality is
markedly better than 7B's for this justification task — worth the wait.

### 3. Tokenizer surgery — add chat anchors

```bash
ssh ghostlm-mac 'cd ~/Desktop/GhostLM && PYTHONPATH=. python3 scripts/tokenizer_surgery.py \
    --in-tokenizer data/tokenizer_v05/tokenizer.json \
    --out-dir data/tokenizer_v05_surgery'
```

Adds ~30 chat anchor tokens. Output: `data/tokenizer_v05_surgery/tokenizer.json`.
Vocab grows from 32,000 → ~32,030.

### 4. Rebuild chat dataset using CoT MCQs (replace `mcq.jsonl` → `mcq_cot.jsonl`)

```bash
ssh ghostlm-mac 'cd ~/Desktop/GhostLM && PYTHONPATH=. python3 scripts/build_chat_dataset.py \
    --mcq-jsonl data/raw/chat/mcq_cot.jsonl \
    --mcq-multiplier 1 \
    --small-talk-multiplier 30'
```

Note: `--mcq-multiplier 1` (down from 2) per research-agent recommendation —
CoT records at 1× beat raw MCQ at 5×.

### 5. SFT on top of extended pretrain

```bash
ssh ghostlm-mac 'cd ~/Desktop/GhostLM && nohup env PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 \
    PYTHONPATH=. python3 -u scripts/finetune_chat.py \
    --checkpoint checkpoints/phase6_v05_pretrain/best_model.pt \
    --tokenizer data/tokenizer_v05_surgery/tokenizer.json \
    --run-name phase8_chat_v05_recovered \
    --max-steps 1500 --warmup-steps 100 \
    --learning-rate 3e-5 \
    --batch-size 4 --grad-accum-steps 8 --context-length 512 \
    --eval-interval 100 --save-interval 500 \
    > /tmp/v05_chat_recovered.log 2>&1 < /dev/null & echo "PID=$!"'
```

**Conservative settings.** 1500 steps max (overcooked beyond — chat-long
collapsed at 4000). lr 3e-5 (not 5e-5). Monitor for first val plateau,
stop early if needed.

### 6. Bench

```bash
ssh ghostlm-mac 'cd ~/Desktop/GhostLM && PYTHONPATH=. python3 scripts/run_bench.py \
    --checkpoint checkpoints/phase8_chat_v05_recovered/best_model.pt \
    --tokenizer data/tokenizer_v05_surgery/tokenizer.json \
    --label "ghost-small-v0.5 chat-recovered (extended pretrain + CoT MCQ + tok surgery)" \
    --device mps --bench ctibench-mcq \
    --out-json logs/phase8_chat_v05_recovered/bench_ctibench.json'
```

### 7. Decision point

| Result | Action |
|---|---|
| > 50% | 🎉 Push to HF, update Space, update MCP, ship as canonical |
| 40-50% | Push as v0.5.0 alongside v0.4 chat-v3 (canonical) — clear improvement, ship |
| 36.9-40% | Tied / slight edge over v0.4 — push as v0.5-experimental, keep v0.4 canonical |
| < 36.9% | Accept v0.4 as canonical, ship v0.5 work as research artifact, document the lessons honestly |

### 8. If shipping (≥ 36.9%)

```bash
# Update HF model + Space + MCP per docs/chat_tuning.md and docs/mcp.md
# Re-bench should also feed into RESULTS.md automatically via run_bench.py
# Don't forget: re-register MCP server pointing at new checkpoint
```

## Failure modes to watch

- **OOM during SFT**: Pretrain extended at the same ctx 512 / batch 4 / accum 8
  settings, so SFT should fit. If it doesn't, drop batch to 2 / accum 16.
- **Mac reboot mid-SFT**: Resume via `--resume checkpoints/phase8_chat_v05_recovered/checkpoint_step_<N>.pt`
  (note `--resume` is on `train_v05.py`, not `finetune_chat.py` — add the same logic
  if needed, or just re-run from scratch since SFT is short).
- **Ollama hangs on a record during CoT gen**: script is resume-safe, just
  Ctrl-C and re-run.
- **CoT records have wrong letters**: build_mcq_cot_data.py preserves the
  original letter from build_mcq_data.py — Qwen only writes the
  justification. The letter math doesn't depend on Qwen.
