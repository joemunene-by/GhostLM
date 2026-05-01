# GhostLM chat tuning (Phase 5)

A supervised fine-tune that turns the Phase 4 ghost-small completion model
into a conversational cybersecurity assistant. The base model's weights are
preserved as the starting point — only three new chat-role tokens are added
to the embedding (vocab 50261 → 50264) and the model is fine-tuned on a
small instruction dataset with assistant-only loss masking.

The chat tune is the v0.5 milestone; the v0.4 ghost-small pretrain remains
canonical for raw completion / per-source perplexity work.

## Chat format

Three new special tokens, appended after the existing four (BOS/EOS/PAD/UNK):

- `<|ghost_user|>`       — start of a user turn
- `<|ghost_assistant|>`  — start of an assistant turn
- `<|ghost_end|>`        — end of any turn

A two-turn conversation looks like::

    <|ghost_user|>What is XSS?<|ghost_end|>
    <|ghost_assistant|>Cross-Site Scripting is a vulnerability...<|ghost_end|>

`GhostTokenizer.encode_chat(turns)` produces both the token ids and a per-token
loss mask (1 on assistant content + the assistant's trailing `<|ghost_end|>`,
0 everywhere else). `format_chat_prompt(turns)` is the inference-time helper
that ends the sequence with `<|ghost_assistant|>` ready for generation.

## Building the dataset

`scripts/build_chat_dataset.py` walks the pretrain corpus and applies
per-source templates to generate ~10K instruction pairs (NVD, MITRE ATT&CK,
CAPEC, Exploit-DB, CTFtime writeups, synthetic CTF). It then merges in the
hand-written `data/raw/chat/small_talk.jsonl` seed, oversampling the
small-talk pairs to balance the mix.

```bash
PYTHONPATH=. python3 scripts/build_chat_dataset.py \
    --small-talk-multiplier 30 \
    --out-train data/processed/chat_train.jsonl \
    --out-val data/processed/chat_val.jsonl
```

The `--small-talk-multiplier` flag exists because the v1 attempt used 1× and
ended up with small_talk at 1.6% of training — the model never learned to
follow instructions because templated cybersec answers swamped the chat-shape
signal. 30× brings small_talk to ~30% of the training mix.

## Training

`scripts/finetune_chat.py` loads the Phase 4 ghost-small checkpoint, expands
the token embedding by three rows, re-ties `lm_head`, and runs the standard
GhostTrainer with SFT-appropriate hyperparameters. Loss is masked to the
assistant's content tokens via `target_id = -1` everywhere else — the model's
existing `cross_entropy(..., ignore_index=-1)` does the rest.

```bash
PYTHONPATH=. python3 scripts/finetune_chat.py \
    --checkpoint checkpoints/phase4_ghost_small/best_model.pt \
    --run-name phase5_chat_v2 \
    --max-steps 1500 --warmup-steps 100 \
    --learning-rate 3e-5 \
    --eval-interval 75 --save-interval 500
```

Outputs land in `checkpoints/phase5_chat_v2/` (best_model.pt + checkpoints +
tokenizer.json + config.json). Wall-clock on M4 MPS: ~45 minutes for 1500
steps at batch_size=8 × grad_accum=4.

## Inference

`scripts/chat.py` is the multi-turn REPL. It maintains conversation history,
formats each turn with the role markers, and stops generation the moment a
`<|ghost_end|>` token is sampled.

```bash
PYTHONPATH=. python3 scripts/chat.py \
    --checkpoint checkpoints/phase5_chat_v2/best_model.pt \
    --temperature 0.7 --top-k 40 --top-p 0.95 \
    --repetition-penalty 1.25
```

A repetition penalty (default 1.25) is applied to tokens recently emitted —
without it, small models occasionally degenerate into "Wifi Wifi Wifi…"
loops. `--no-chat-format` falls back to the original raw-completion mode for
pretrain-only checkpoints.

## Eval

`scripts/eval_chat.py` runs a fixed 27-prompt held-out suite (small-talk,
identity, OOD refusals, cybersec basics, specific items, edge cases) and
writes a transcript file for side-by-side comparison across runs.

```bash
PYTHONPATH=. python3 scripts/eval_chat.py \
    --checkpoint checkpoints/phase5_chat_v2/best_model.pt \
    --out logs/phase5_chat_v2/eval_transcript.txt
```

## What worked, what didn't

The chat tune ran through three iterations on the way to a model that's
actually useful:

- **v1 (`phase5_chat`)** — small-talk at 1.6% of training. The model emitted
  chat-format-shaped output but ignored the user prompt: every input
  returned a randomly-sampled cybersec answer. CTIBench score: not run
  because the model wasn't usable.
- **v2 (`phase5_chat_v2`)** — small-talk × 30 → 30% of training. Identity,
  OOD-refusal, and chat-shape all started working. CTIBench: 475 / 2500
  (19.0%), below random.
- **v3 (`phase5_chat_v3`)** — added 1,802 templated MCQ examples (NVD
  CWE-class, MITRE tactic, acronym definitions) at 2× oversampling. The
  assistant turn is the bare letter A/B/C/D, occasionally followed by a
  one-line justification. CTIBench: **922 / 2500 (36.9%) — +17.9pp over
  v2**, well above random. **This is the current canonical chat model.**

The MCQ-format examples are the larger of the two unlocks. The chat
training data was teaching free-form answer shape (CVE description, MITRE
explanation, etc.) and never teaching "here's a question with four options,
output one letter." Without that signal the model defaults to the most
common first-letter token — typically A — regardless of the question.

## Build the v3 dataset

```bash
# Step 1: generate the MCQ-format chat examples (~1.8K records)
PYTHONPATH=. python3 scripts/build_mcq_data.py

# Step 2: build the combined chat training set
PYTHONPATH=. python3 scripts/build_chat_dataset.py \
    --small-talk-multiplier 30 \
    --mcq-multiplier 2

# Step 3: SFT
PYTHONPATH=. python3 scripts/finetune_chat.py \
    --checkpoint checkpoints/phase4_ghost_small/best_model.pt \
    --run-name phase5_chat_v3 \
    --max-steps 1800 --warmup-steps 120 \
    --learning-rate 3e-5 \
    --eval-interval 100 --save-interval 600
```

Held-out eval + benchmark::

    PYTHONPATH=. python3 scripts/eval_chat.py \
        --checkpoint checkpoints/phase5_chat_v3/best_model.pt
    PYTHONPATH=. python3 scripts/run_bench.py \
        --checkpoint checkpoints/phase5_chat_v3/best_model.pt \
        --device mps --bench ctibench-mcq

Limits of a 45M chat tune (acknowledged up front):

- **No general world knowledge** — the model only knows what's in its 12.5M
  cybersecurity tokens. Outside that domain it's wrong, repetitive, or both.
- **Specific facts are unreliable** — exact CVE numbers, dates, CVSS scores,
  technique IDs are memorized incompletely. Always verify against the NVD,
  MITRE ATT&CK, or vendor advisories. The RAG layer (see `docs/rag.md`) is
  the path to better factual grounding.
- **Short coherence window** — 1024 ctx and 45M params mean long multi-turn
  conversations drift; the chat REPL trims old turns when the prompt overflows.
- **CTIBench at 36.9% is well above random but well below larger models.**
  Public reference points: GPT-4 / Claude / Gemini routinely score
  85-95% on CTIBench MCQ. We're at 36.9% from 45M parameters trained on
  12.56M tokens of pure cybersecurity text. Two clear paths to lift this:
  (1) the v0.4.2 corpus expansion + v0.5 retrain bundled in
  `docs/v0.5_architecture.md`; (2) RAFT-style retrieval-aware fine-tune
  on top of the existing RAG index (the index already retrieves the
  right documents — the model just needs supervision to use them).
