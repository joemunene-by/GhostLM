# Keeping GhostLM current after chat SFT

A 349M model trained on a Mac cannot improve itself from its own outputs. What
it can have is a loop around it that keeps its knowledge current every night
and folds real corrections back into the weights every few weeks, with an
evaluation gate so the live model never gets worse. This note defines that
loop. It builds on pieces that already exist; the new work is the glue and the
gate.

Scope: runs on the same 16GB M4 under the existing supervisor
(`scripts/background/supervisor.py`), after the `chat_sft` phase has produced
`checkpoints/ghost_base_chat/`. Nothing here trains on unreviewed text; the
review gate in `scripts/review_chat.py` stays the only way into training.

## 1. Why three layers

| Layer | What changes | Cadence | Cost | Expected effect |
|---|---|---|---|---|
| Knowledge refresh | RAG index only | nightly | minutes | model answers about things published today |
| Feedback fine-tune | weights (LoRA) | monthly or on data threshold | hours | fewer repeated mistakes, better tone on real questions |
| Verifiable self-play | weights (LoRA) | experiment only | days | narrow skills with a mechanical checker |

Layer 1 is where almost all of the "adapts with time" value lives for a model
this size. Its weights will never hold facts reliably (the pretrain is about
7x under Chinchilla), so facts belong in the index, and the index is cheap to
rebuild. Layer 2 is the real "gets better" loop. Layer 3 is deliberately last:
self-instruct and model-as-judge only work when the model can generate data
better than what it was trained on, which is not true at GPT-2-small quality.

## 2. Layer 1: nightly knowledge refresh

Sources that already have collectors and change daily or weekly:

- `scripts/collect_nvd_full.py` (CVEs; NVD API key in the environment keeps it under rate limits)
- `scripts/collect_cisa_kev.py` and `scripts/collect_cisa_advisories.py`
- `scripts/collect_mitre_full.py` (ATT&CK, CWE, CAPEC releases)
- `scripts/collect_misp_feeds.py`, `scripts/collect_exploitdb.py`, `scripts/collect_security_blogs.py`, `scripts/collect_vendor_research.py`

Design:

1. Each collector gets an incremental mode (`--since DATE`) writing only new or
   changed records to `data/raw/<source>/delta-YYYY-MM-DD.jsonl`. NVD already
   exposes `lastModStartDate`; KEV and ATT&CK are small enough to re-pull whole
   and diff by id.
2. `scripts/build_rag_index.py --append DELTA` embeds only the delta with the
   same bge-small encoder and appends rows to `data/rag/index.npy` and
   `data/rag/chunks.jsonl`. Changed records replace their old chunks by
   `ref` so a CVE whose CVSS was revised shows the new score, not both.
3. The hybrid retriever (`ghostlm/rag.py`) already does exact ID lookup for
   CVE, CWE, CAPEC and ATT&CK ids, so a fresh CVE is findable the moment its
   chunk is in the index.
4. A new supervisor phase `refresh_index` runs at 03:00 local when no training
   step is in flight, pauses under the same memory and game rules as training,
   and writes `.bg/refresh_status.json` (sources pulled, rows added, time).
   The app (`ghostlm/app/engine.py`) re-opens the memory-mapped index when
   its mtime changes, the same way it reloads the live checkpoint.
5. Full rebuild monthly to purge replaced chunks and keep the matrix dense.

Disk: at 384 float32 dims a chunk costs 1.5 KB in the matrix; 2,000 new CVE
chunks a day is about 3 MB a day, so a year of deltas is about 1 GB before the
monthly rebuild compacts it.

## 3. Layer 2: learning from use

### 3.1 Capture

The app (`ghostlm/app/server.py`) gains:

- `POST /api/feedback` with `{chat_id, turn, verdict: up|down, correction?}`.
  A correction is the user's rewrite of the assistant turn.
- A thumbs up, thumbs down and "edit this answer" control under each
  assistant message in the static UI.

Every feedback event is appended to `data/chat_gen/feedback.jsonl` in the
same `{turns, source, teacher}` shape the review gate reads, with
`teacher: "user"` and the original answer kept under `original` for the
reviewer to compare. A thumbs down without a correction is logged to
`data/feedback/negatives.jsonl` for the eval set (section 3.4) but never
becomes a training example on its own.

Ghost Code (`ghostlm/code/`) and the MCP server can post to the same endpoint
when a user edits a generated answer, so terminal use feeds the loop too.

### 3.2 Review

Feedback goes through `scripts/review_chat.py next` and `apply` like any
other area. Reviewers apply `docs/chat_review.md` unchanged. One addition:
the original wrong answer, if useful as a contrast, can be kept as context
with `"train": false` on that turn (the encoder masks it; see
`tests/test_chat_turn_mask.py`), so the model learns the correction without
learning the mistake.

### 3.3 Fine-tune

Trigger: at least 300 newly approved conversations since the last round, or
30 days, whichever first. Both numbers are config in
`.bg/train_config.json` under a `continual` block.

Recipe, new supervisor phase `continual_sft`:

1. Build the round's set with `scripts/build_chat_mix.py --since LAST_ROUND
   --replay 0.3`: all newly approved conversations plus a 30 percent-sized
   random replay slice of the previously approved pool, so the round does not
   overwrite what earlier rounds taught (plain catastrophic-forgetting
   insurance; the pool is small enough that this stays cheap).
2. Train a LoRA adapter with `scripts/finetune_lora.py` from the current live
   chat checkpoint, not from the original SFT checkpoint, so rounds compound.
   Rank 16, two epochs, learning rate 1e-4, the same micro-batch and
   grad-checkpoint settings the chat SFT phase used on this machine. A full
   fine-tune is not an option on 16GB once the model is the roughly 1B MoE;
   LoRA keeps it to adapter-sized memory.
3. Merge the adapter into a candidate checkpoint
   `checkpoints/ghost_base_chat/candidate.pt`. Keep the adapter file as the
   audit trail of what each round changed.

### 3.4 Gate and promote

A candidate becomes live only if it passes all of:

- `scripts/scorecard.py` on the standard benches: no bench drops by more than
  its bootstrap confidence interval, and the aggregate does not drop.
- `scripts/eval_chat.py` on a held-out chat validation set that includes every
  past negative from `data/feedback/negatives.jsonl` with its reviewed
  correct answer: the pass rate must not drop and the newest negatives should
  improve.
- `scripts/eval_security.py` refusal and over-refusal checks unchanged within
  noise, so feedback cannot quietly make the model refuse defensive questions
  or stop refusing harmful ones.
- `scripts/eval_format_compliance.py` unchanged within noise.

Pass: `candidate.pt` is renamed to `best_model.pt`, the previous live
checkpoint is kept as `previous.pt` for one round, and the app's live reload
picks it up. Fail: the candidate is deleted, the round's data stays approved
for the next attempt, and the failure is written to `.bg/continual_log.jsonl`
with the per-bench numbers, so a bad round is visible rather than silent.

Rollback is a file rename back to `previous.pt`.

### 3.5 Why this does not repeat the October replay bug

The round-9 overfit (train loss 0.84, val 3.40) came from an unnoticed data
replay plus no gate. Here every round has a fixed, logged data manifest, the
replay slice is sampled with the round id as seed, and nothing goes live
without the eval gate. The gate is the point of the design; the fine-tune is
the easy part.

## 4. Layer 3: verifiable self-play (later, experiment only)

Only for tasks with a mechanical checker, where the checker and not the model
supplies the training signal:

- Sigma or YARA rule writing, checked by running the rule against a labelled
  log or sample set.
- Small code fixes, checked by the repository's own tests.
- CVE to ATT&CK mapping, checked against the MITRE mapping tables.

Loop: the model attempts the task many times with sampling, the checker keeps
the attempts that pass, a reviewer reads a sample of the passing set, and
those enter the same review gate and continual round as everything else.
Attempts the checker rejects are discarded, not used as negatives. Run as a
one-off experiment on one task first and compare scorecard deltas before
making it a phase. Do not use the model as its own judge.

## 5. Order of work

1. Feedback capture in the app (section 3.1). Smallest piece, and the data
   starts accumulating from day one.
2. Incremental collectors plus index append and the `refresh_index` phase
   (section 2). Deliverable: a CVE published yesterday is answerable today.
3. `build_chat_mix.py --since/--replay`, the `continual_sft` phase and the
   gate (sections 3.3, 3.4). Deliverable: the first gated round, even if its
   data is only the first few hundred feedback conversations plus replay.
4. Layer 3 experiment on one verifiable task.

Items 1 and 2 can be built before chat SFT finishes, since neither depends on
the chat checkpoint. Item 3 waits for `checkpoints/ghost_base_chat/` to
exist. All phases obey the existing one-heavy-job rule: refresh and
fine-tune never run while a training step is in flight, and they pause for
games and memory pressure like the trainer does.
