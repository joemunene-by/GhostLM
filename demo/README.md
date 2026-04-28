---
title: GhostLM
emoji: 🔐
colorFrom: purple
colorTo: gray
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: apache-2.0
short_description: From-scratch cybersecurity LM — interactive demo
---

# GhostLM Demo

Interactive Gradio UI for the canonical Phase 3.5 ghost-tiny model. Two
tabs: a single-checkpoint **Generate** view with curated prompt presets
and a generation history, and an optional **Compare** tab that runs the
same prompt through two checkpoints side-by-side (the canonical v0.3.5
vs. the v0.3.7 attempt that regressed).

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

Open `http://localhost:7860`. The demo defaults to
`checkpoints/phase3.5_balanced/best_model.pt` — pass `--checkpoint` to
load a different one:

```bash
PYTHONPATH=. python3 demo/app.py --checkpoint checkpoints/phase3.6_exploitdb/best_model.pt
```

To enable the Compare tab, add a second checkpoint:

```bash
PYTHONPATH=. python3 demo/app.py \
  --checkpoint checkpoints/phase3.5_balanced/best_model.pt \
  --compare-checkpoint checkpoints/phase3.6_exploitdb/best_model.pt
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

# Copy the demo + the ghostlm package + the canonical checkpoint
cp ../demo/app.py .
cp ../demo/requirements.txt .
cp ../demo/README.md .
cp -r ../ghostlm .
mkdir -p checkpoints/phase3.5_balanced
cp ../checkpoints/phase3.5_balanced/best_model.pt checkpoints/phase3.5_balanced/

git add .
git commit -m "Initial GhostLM Space deploy"
git push
```

The Space will start building automatically; first build takes ~3–5
minutes (gradio + torch wheel install + checkpoint LFS pull). The
README's frontmatter tells HF this is a Gradio Space, sets the colors,
and pins `app_file: app.py`.

### 3. Optional — include the Phase 3.6 checkpoint for the Compare tab

If you want the Compare tab live in the Space, also copy the Phase 3.6
checkpoint (~177 MB more) and set the env var in the Space's Settings
page:

```bash
mkdir -p checkpoints/phase3.6_exploitdb
cp ../checkpoints/phase3.6_exploitdb/best_model.pt checkpoints/phase3.6_exploitdb/
git add checkpoints/phase3.6_exploitdb
git commit -m "Add Phase 3.6 for compare tab"
git push
```

In the Space's **Settings → Variables**, add:

```
GHOSTLM_COMPARE_CHECKPOINT = checkpoints/phase3.6_exploitdb/best_model.pt
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

The **Compare** tab — only shown when a second checkpoint is loaded —
sends the same prompt + sampling settings to both models in turn so the
Phase 3.5 → 3.6 trajectory is visible in real text rather than just
accuracy numbers.

## Why this exists

The point of the demo isn't to impress visitors with fluency — at 14.7M
parameters trained on 8.8M tokens, the model produces register-shaped
fiction, not knowledge. The point is to make the project's
trajectory-over-absolute-quality framing concrete: visitors can poke at
the canonical model, see exactly what it knows and doesn't, and if both
checkpoints are loaded, see the empirical capacity-ceiling finding for
themselves.
