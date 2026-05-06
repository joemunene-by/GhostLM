---
title: GhostLM
emoji: 🔐
colorFrom: purple
colorTo: gray
sdk: gradio
app_file: app.py
pinned: false
license: apache-2.0
short_description: From-scratch cybersecurity LM, interactive demo
---

# GhostLM Demo

Interactive Gradio UI for any GhostLM checkpoint. Two tabs: a
single-checkpoint **Generate** view with curated prompt presets and a
generation history, and an optional **Compare** tab that runs the
same prompt through two checkpoints side-by-side (e.g. an early
ghost-tiny vs the v0.9 chat to show the trajectory).

The current canonical chat checkpoint is
`checkpoints/phase19_chat_v09/best_model.pt` (v0.9, 81M wide,
trained on the 273M-token PRIMUS + CWE + OWASP + RFCs corpus).
Ghost-base (~360M, v1.0 target) is pending GPU access; once the v1.0
checkpoint lands the demo's default will switch to it.

This file is dual-purpose:

- **In the GitHub repo** (`demo/README.md`) — documents the demo and
  the deploy steps.
- **As an HF Space README** — the YAML frontmatter at the top is parsed
  by Hugging Face Spaces as the Space metadata. Keep it intact when
  copying this file to a Space repo.

## Run locally

From the repo root:

```bash
pip install -r demo/requirements.txt
PYTHONPATH=. python3 demo/app.py
```

Open `http://localhost:7860`. Pass `--checkpoint` to load any saved
model; the v0.9 chat checkpoint is the recommended default:

```bash
PYTHONPATH=. python3 demo/app.py --checkpoint checkpoints/phase19_chat_v09/best_model.pt
```

To enable the Compare tab, add a second checkpoint. Useful pairings:
v0.4 chat-v3 vs v0.9 chat to see the corpus-density trajectory, or
the canonical v0.4 base vs ghost-base when v1.0 ships.

```bash
PYTHONPATH=. python3 demo/app.py \
  --checkpoint checkpoints/phase19_chat_v09/best_model.pt \
  --compare-checkpoint checkpoints/phase5_chat_v3/best_model.pt
```

The same `--share` flag Gradio supports works:

```bash
PYTHONPATH=. python3 demo/app.py --share
```

## Deploy to Hugging Face Spaces

A Space is a separate git repo on huggingface.co. The demo here lives
under `demo/` in the GhostLM repo so the source stays in one place; to
deploy you copy the demo files plus the `ghostlm/` package and a
checkpoint into a fresh Space repo.

### 1. Create the Space

Either via the Hugging Face web UI (New → Space, SDK = Gradio) or via
CLI:

```bash
pip install huggingface_hub
huggingface-cli login
huggingface-cli repo create ghostlm --type space --space-sdk gradio
```

Replace `ghostlm` with your preferred Space name.

### 2. Clone the Space repo and stage files

```bash
git clone https://huggingface.co/spaces/<your-user>/ghostlm hf-space
cd hf-space

# Track the checkpoint via LFS (it's ~177 MB)
git lfs install
git lfs track "*.pt"

# Copy the demo + the ghostlm package + the canonical chat checkpoint
cp ../demo/app.py .
cp ../demo/requirements.txt .
cp ../demo/README.md .
cp -r ../ghostlm .
mkdir -p checkpoints/phase19_chat_v09
cp ../checkpoints/phase19_chat_v09/best_model.pt checkpoints/phase19_chat_v09/

git add .
git commit -m "Initial GhostLM Space deploy"
git push
```

The Space will start building automatically; first build takes ~3–5
minutes (gradio + torch wheel install + checkpoint LFS pull). The
README's frontmatter tells HF this is a Gradio Space, sets the colors,
and pins `app_file: app.py`.

### 3. Optional, include a second checkpoint for the Compare tab

If you want the Compare tab live in the Space, copy a second
checkpoint (e.g. v0.4 chat-v3 to compare against v0.9) and set the
env var in the Space's Settings page:

```bash
mkdir -p checkpoints/phase5_chat_v3
cp ../checkpoints/phase5_chat_v3/best_model.pt checkpoints/phase5_chat_v3/
git add checkpoints/phase5_chat_v3
git commit -m "Add v0.4 chat-v3 for compare tab"
git push
```

In the Space's **Settings → Variables**, add:

```
GHOSTLM_COMPARE_CHECKPOINT = checkpoints/phase5_chat_v3/best_model.pt
```

The Space restarts automatically. The Compare tab will now be visible.

### 4. Updates

Push to the Space repo whenever the demo changes; the Space rebuilds.
For a checkpoint update push the new `.pt` file (LFS handles it).

## What it looks like

The **Generate** tab gives you a prompt textbox, three sampling sliders
(max tokens, temperature, top-k), and a continuation panel. Below that,
collapsible accordions group the preset prompts by register (CVE / MITRE
/ CTF / CAPEC / free-form) so visitors can immediately see what kind of
prose the model knows. A history panel keeps the last five generations
visible.

The **Compare** tab, only shown when a second checkpoint is loaded,
sends the same prompt + sampling settings to both models in turn so
the trajectory between two checkpoints is visible in real text
rather than just accuracy numbers.

## Why this exists

The demo isn't there to impress visitors with fluency. At 81M
parameters trained on 273M tokens, v0.9 is a register-matching
"cybersec parrot" (per the v0.9.2 fact-recall benchmark, free-form
factual recall is at floor across the whole ghost-small line). The
demo lets visitors poke at the canonical model, see exactly what it
knows and doesn't, and if a second checkpoint is loaded, see the
trajectory across versions in real prose. Once ghost-base v1.0
ships from the rented-GPU run, this README's default checkpoint
will switch to it.
