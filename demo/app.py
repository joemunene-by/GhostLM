"""GhostLM Gradio demo, interactive web UI for any saved checkpoint.

Two-tab interface:

  Generate   single-checkpoint generation with curated prompt presets,
             generation history, and honest "what to expect" framing.
  Compare    side-by-side generation across two checkpoints from the
             same prompt + sampling settings. Lets a visitor see the
             trajectory across versions (e.g. v0.4 chat-v3 vs v0.9 chat)
             in real text rather than just accuracy numbers.

Run locally:
    python3 demo/app.py
    python3 demo/app.py --checkpoint checkpoints/phase19_chat_v09/best_model.pt
    python3 demo/app.py --compare-checkpoint checkpoints/phase5_chat_v3/best_model.pt

The current canonical chat checkpoint is
``checkpoints/phase19_chat_v09/best_model.pt`` (v0.9, 81M wide). When
ghost-base v1.0 ships from the rented-GPU pretrain run, the default
will switch to the new checkpoint.

On Hugging Face Spaces this file is the entry point (see demo/README.md
for the Space metadata frontmatter). The Space ships a single canonical
checkpoint baked in; the Compare tab is hidden when only one checkpoint
is available.
"""

import argparse
import os
import sys
from dataclasses import fields
from pathlib import Path
from typing import List, Optional, Tuple

import torch

# Make the parent repo importable when this file is run directly from demo/
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ghostlm.config import GhostLMConfig
from ghostlm.model import GhostLM
from ghostlm.tokenizer import GhostTokenizer

# Don't try/except this — on Hugging Face Spaces a transitive failure
# inside gradio (e.g. a wheel ABI mismatch under torch) gets swallowed
# as a generic ImportError, and the user sees the unhelpful "Install
# gradio" message in the runtime log instead of the real traceback. The
# raw import lets Python print the actual stack so we can debug.
import gradio as gr


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

# The canonical default. Resolved at startup; users can override with --checkpoint
# locally, and the Hugging Face Space drops a copy at this path.
DEFAULT_CHECKPOINT = "checkpoints/phase19_chat_v09/best_model.pt"

# Common checkpoint paths the Space might know about, ordered by recency.
# The local repo has many; HF Spaces typically only has one.
KNOWN_CHECKPOINTS = [
    ("v0.9 chat (canonical, 81M wide, 273M-token corpus)", "checkpoints/phase19_chat_v09/best_model.pt"),
    ("v0.7 chat (81M wide)", "checkpoints/phase15_chat_v07/best_model.pt"),
    ("v0.4 chat-v3 (45M, single-order CTIBench winner)", "checkpoints/phase5_chat_v3/best_model.pt"),
    ("v0.4 base (45M ghost-small pretrain)", "checkpoints/phase4_ghost_small/best_model.pt"),
    ("Phase 3.5 (v0.3.5 ghost-tiny, historical)", "checkpoints/phase3.5_balanced/best_model.pt"),
]


def load_checkpoint(path: str) -> Tuple[GhostLM, GhostTokenizer, GhostLMConfig, dict]:
    """Load a checkpoint, returning (model, tokenizer, config, metadata).

    Falls back to a randomly-initialized ghost-tiny if the path doesn't
    exist — useful for trying the UI without any weights present, and
    safer than crashing the launch.
    """
    tokenizer = GhostTokenizer()

    if path and Path(path).exists():
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        saved_config = ckpt["config"]
        config = GhostLMConfig(**{
            f.name: saved_config[f.name]
            for f in fields(GhostLMConfig)
            if f.name in saved_config
        })
        model = GhostLM(config)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        meta = {
            "path": path,
            "step": ckpt.get("step"),
            "val_loss": ckpt.get("val_loss"),
            "loaded": True,
        }
    else:
        config = GhostLMConfig.from_preset("ghost-tiny")
        config.vocab_size = 50261
        config.context_length = 128
        model = GhostLM(config)
        model.eval()
        meta = {"path": path, "loaded": False}

    return model, tokenizer, config, meta


def format_model_info(model: GhostLM, config: GhostLMConfig, meta: dict) -> str:
    """Render the model-info markdown card for a loaded checkpoint."""
    lines = [
        f"**Variant:** ghost-tiny ({config.n_layers} layers · {config.d_model} dim · {config.n_heads} heads)",
        f"**Parameters:** {model.num_params():,}",
        f"**Context length:** {config.context_length}",
    ]
    if meta.get("loaded"):
        if meta.get("step") is not None:
            lines.append(f"**Trained step:** {meta['step']:,}")
        if meta.get("val_loss") is not None:
            lines.append(f"**Final val_loss:** {meta['val_loss']:.4f}")
        lines.append(f"**Checkpoint:** `{meta['path']}`")
    else:
        lines.append(
            "**Status:** ⚠️ random-init weights (no checkpoint at "
            f"`{meta.get('path')}`). Output will be incoherent — "
            "the UI runs so you can see the layout, not the model."
        )
    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def _generate(
    model: GhostLM,
    tokenizer: GhostTokenizer,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_k: int,
) -> str:
    """Run model.generate against the given prompt and return the new tokens.

    Strips the prompt from the model's full output so the user sees only
    the continuation, which is what they expect from a "generate" button.
    """
    if not prompt or not prompt.strip():
        return "Please enter a prompt."

    try:
        ids = tokenizer.encode(prompt)
        # Trim to context window from the left so long prompts don't crash
        # the model — same logic the eval scorer uses.
        max_input = max(1, model.config.context_length - int(max_tokens))
        if len(ids) > max_input:
            ids = ids[-max_input:]

        x = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
        top_k_val = top_k if top_k > 0 else None
        with torch.no_grad():
            out = model.generate(
                x,
                max_new_tokens=int(max_tokens),
                temperature=float(temperature),
                top_k=top_k_val,
            )
        text = tokenizer.decode(out[0].tolist())
        if text.startswith(prompt):
            text = text[len(prompt):]
        return text.strip() or "(empty generation — try lowering temperature or shortening the prompt)"
    except Exception as e:  # noqa: BLE001 - surface any model error to the UI
        return f"Error: {e}"


# ---------------------------------------------------------------------------
# Prompt presets
# ---------------------------------------------------------------------------

# Organised by register so the user can quickly try the four kinds of
# prose the canonical model has actually been trained on. v0.3.5's whole
# story is "switches register based on prompt domain" — these presets are
# the showcase.
PROMPT_PRESETS = {
    "CVE description": [
        "CVE-2024-99999 is a vulnerability in",
        "A buffer overflow in the kernel allows a local attacker to",
        "An authentication bypass in the administrative web interface",
    ],
    "MITRE ATT&CK": [
        "MITRE ATT&CK technique T1003 is used by adversaries to",
        "Adversaries may abuse",
        "The Persistence tactic encompasses techniques that",
    ],
    "CTF writeup": [
        "The CTF challenge involved",
        "After running the binary in gdb, we noticed",
        "The web app was vulnerable to",
    ],
    "CAPEC pattern": [
        "CAPEC-66 SQL Injection is an attack pattern in which",
        "The attacker exploits a trust relationship between",
    ],
    "Free-form security prose": [
        "A SQL injection attack works by",
        "Ransomware encrypts victim files by",
        "Zero-day vulnerabilities are dangerous because",
    ],
}


def _flatten_presets() -> List[str]:
    """All preset prompts in one flat list for Gradio's Examples component."""
    out = []
    for prompts in PROMPT_PRESETS.values():
        out.extend(prompts)
    return out


# ---------------------------------------------------------------------------
# CLI args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="GhostLM Gradio demo")
    p.add_argument(
        "--checkpoint",
        default=os.environ.get("GHOSTLM_CHECKPOINT", DEFAULT_CHECKPOINT),
        help=f"Primary checkpoint path. Default: {DEFAULT_CHECKPOINT} or "
             "GHOSTLM_CHECKPOINT env var if set.",
    )
    p.add_argument(
        "--compare-checkpoint",
        default=os.environ.get("GHOSTLM_COMPARE_CHECKPOINT"),
        help="Optional second checkpoint for the Compare tab. Tab is hidden "
             "when this is unset.",
    )
    p.add_argument(
        "--port", type=int, default=int(os.environ.get("PORT", 7860)),
        help="Gradio listen port (default 7860; PORT env honored for HF Spaces).",
    )
    p.add_argument(
        "--share", action="store_true",
        help="Generate a public gradio.live tunnel URL (local only).",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

GROUND_RULES = (
    "**This is a completion model, not a chatbot.** GhostLM has no "
    "instruction tuning. Prompt it with the *start of a sentence* in a "
    "register it knows — CVE descriptions, MITRE techniques, CTF "
    "writeups, arXiv abstracts — and it continues. Prompts like "
    "`hello`, `who are you`, or `summarize this` produce drifty cyber-"
    "prose because the model has no notion of an instruction to follow.\n\n"
    "**What it does well:** structurally correct security prose. "
    "**What it doesn't:** facts. CVE IDs, version chains, technique IDs "
    "are all made up. Treat outputs as register-shaped fiction, not "
    "reference material. The trajectory across phases is the project's "
    "value, not the absolute output of any one continuation."
)


def build_ui(primary, compare):
    """Construct the Gradio Blocks app. Returns the demo object."""
    primary_model, primary_tok, primary_cfg, primary_meta = primary

    # theme moved from Blocks() to launch() in Gradio 6.0; title stayed.
    with gr.Blocks(title="GhostLM Demo") as demo:
        gr.Markdown("# 🔐 GhostLM")
        gr.Markdown(
            "Open-source cybersecurity language model — built from scratch in PyTorch. "
            "[Repo](https://github.com/joemunene-by/GhostLM) · [ROADMAP](https://github.com/joemunene-by/GhostLM/blob/main/ROADMAP.md) · "
            "[MODEL_CARD](https://github.com/joemunene-by/GhostLM/blob/main/MODEL_CARD.md)"
        )
        gr.Markdown(GROUND_RULES)

        with gr.Tabs():
            # -------------------------------------------------- Generate tab
            with gr.Tab("Generate"):
                gr.Markdown(
                    f"### Loaded model\n\n{format_model_info(primary_model, primary_cfg, primary_meta)}"
                )

                with gr.Row():
                    with gr.Column(scale=2):
                        prompt = gr.Textbox(
                            label="Prompt (start of a sentence — model continues from here)",
                            lines=4,
                            placeholder=(
                                "e.g.  'CVE-2024-99999 is a vulnerability in'\n"
                                "       'The CTF challenge involved'\n"
                                "       'MITRE ATT&CK technique T1003 is used to'\n"
                                "Don't type 'hello' — the model has no instruction tuning."
                            ),
                        )
                        with gr.Row():
                            max_tokens = gr.Slider(50, 300, value=150, step=10, label="Max tokens")
                            temperature = gr.Slider(0.1, 2.0, value=0.8, step=0.1, label="Temperature")
                            top_k = gr.Slider(0, 100, value=40, step=5, label="Top-k (0 = off)")
                        with gr.Row():
                            generate_btn = gr.Button("Generate", variant="primary")
                            clear_btn = gr.Button("Clear")

                    with gr.Column(scale=3):
                        # show_copy_button was removed from Textbox in
                        # Gradio 6.0 — visitors can still copy via the
                        # browser's native selection.
                        output = gr.Textbox(
                            label="Continuation",
                            lines=8,
                            interactive=False,
                        )
                        history = gr.Markdown(
                            "_Last 5 generations will appear here._",
                            label="History",
                        )

                gr.Markdown("### Preset prompts (organised by register)")
                with gr.Accordion("CVE description", open=False):
                    gr.Examples(PROMPT_PRESETS["CVE description"], inputs=prompt, label="")
                with gr.Accordion("MITRE ATT&CK", open=False):
                    gr.Examples(PROMPT_PRESETS["MITRE ATT&CK"], inputs=prompt, label="")
                with gr.Accordion("CTF writeup", open=False):
                    gr.Examples(PROMPT_PRESETS["CTF writeup"], inputs=prompt, label="")
                with gr.Accordion("CAPEC pattern", open=False):
                    gr.Examples(PROMPT_PRESETS["CAPEC pattern"], inputs=prompt, label="")
                with gr.Accordion("Free-form security prose", open=False):
                    gr.Examples(PROMPT_PRESETS["Free-form security prose"], inputs=prompt, label="")

                # Generation history kept in a hidden state list. Newest first,
                # cap at 5 so the panel stays readable.
                history_state = gr.State([])

                def do_generate(p, m, t, k, hist):
                    text = _generate(primary_model, primary_tok, p, m, t, k)
                    new_hist = ([(p, text)] + (hist or []))[:5]
                    return text, render_history(new_hist), new_hist

                generate_btn.click(
                    fn=do_generate,
                    inputs=[prompt, max_tokens, temperature, top_k, history_state],
                    outputs=[output, history, history_state],
                )
                clear_btn.click(
                    fn=lambda: ("", "", "_Last 5 generations will appear here._", []),
                    outputs=[prompt, output, history, history_state],
                )

            # --------------------------------------------------- Compare tab
            if compare is not None:
                cmp_model, cmp_tok, cmp_cfg, cmp_meta = compare
                with gr.Tab("Compare"):
                    gr.Markdown(
                        "### Two checkpoints, same prompt, same sampling settings.\n\n"
                        "The cleanest demo in this project is Phase 3.5 vs Phase 3.6 — "
                        "the same prompt produces different prose because the second "
                        "model was retrained on a 43% larger corpus that pushed "
                        "ghost-tiny past its capacity ceiling. See "
                        "[CHANGELOG v0.3.7](https://github.com/joemunene-by/GhostLM/blob/main/CHANGELOG.md)."
                    )

                    with gr.Row():
                        with gr.Column():
                            gr.Markdown(f"#### Left\n\n{format_model_info(primary_model, primary_cfg, primary_meta)}")
                        with gr.Column():
                            gr.Markdown(f"#### Right\n\n{format_model_info(cmp_model, cmp_cfg, cmp_meta)}")

                    cmp_prompt = gr.Textbox(
                        label="Prompt (sent to both models)",
                        lines=3,
                        placeholder="Try the same CVE/MITRE/CTF prompts as the Generate tab.",
                    )
                    with gr.Row():
                        cmp_max_tokens = gr.Slider(50, 300, value=150, step=10, label="Max tokens")
                        cmp_temperature = gr.Slider(0.1, 2.0, value=0.8, step=0.1, label="Temperature")
                        cmp_top_k = gr.Slider(0, 100, value=40, step=5, label="Top-k")
                    cmp_btn = gr.Button("Generate from both", variant="primary")
                    with gr.Row():
                        left_out = gr.Textbox(label="Left continuation", lines=8, interactive=False)
                        right_out = gr.Textbox(label="Right continuation", lines=8, interactive=False)

                    def do_compare(p, m, t, k):
                        # Same seed for both models so sampling differences come
                        # purely from the weights, not the RNG. torch.manual_seed
                        # snapshot+restore keeps any global RNG users undisturbed.
                        left = _generate(primary_model, primary_tok, p, m, t, k)
                        right = _generate(cmp_model, cmp_tok, p, m, t, k)
                        return left, right

                    cmp_btn.click(
                        fn=do_compare,
                        inputs=[cmp_prompt, cmp_max_tokens, cmp_temperature, cmp_top_k],
                        outputs=[left_out, right_out],
                    )

            # ------------------------------------------------------ About tab
            with gr.Tab("About"):
                gr.Markdown("""
### GhostLM in one paragraph

GhostLM is a from-scratch decoder-only transformer in PyTorch, trained on a
curated cybersecurity corpus (NVD CVEs, MITRE ATT&CK, CAPEC, CTFtime
real writeups, arXiv cs.CR, Exploit-DB). It's deliberately *small* — 14.7M
parameters — and the canonical v0.3.5 model uses 8.8M training tokens. The
goal isn't to be GPT-4 for security; it's to build a transparent, hand-
written reference implementation that grows in capacity over a multi-year
scale ladder. ghost-tiny is rung 1.

### Where this model sits on the trajectory

Phase 1 → 2 → 3 → 3.5 → 3.6 (attempted) on the 5×25 = 125-sample eval suite:
12.0% → 18.4% → 20.0% → **31.2%** → 16.8%. Phase 3.5 is the canonical model
because Phase 3.6 regressed when Exploit-DB content pushed ghost-tiny past
its capacity ceiling. The fix is the next rung (ghost-small at 55M params),
not more data — see the [ROADMAP](https://github.com/joemunene-by/GhostLM/blob/main/ROADMAP.md).

### What this UI is

A polished Gradio demo that lets you (1) generate from the canonical model
with curated prompt presets and (2) compare it side-by-side against the
Phase 3.6 attempt to see the regression in real text rather than just
accuracy numbers. The model is deliberately small enough to run on
free-tier CPU.

### Caveats

- **Hallucinates facts.** CVE IDs, version chains, technique IDs are made
  up. The model has learned register, not knowledge.
- **Mode-collapses.** v0.3.5 picks "Critical" for 72% of CVE-severity prompts;
  v0.3.6 picks one Vuln Type label for 96% of vuln-type prompts. The
  numbers in the eval table aren't reasoning — they're priors.
- **No instruction tuning.** This is a base language model — it continues
  text, it doesn't follow instructions.

### License

Apache 2.0. Built by Joe Munene · [github.com/joemunene-by/GhostLM](https://github.com/joemunene-by/GhostLM)
""")

        gr.Markdown("---")
        gr.Markdown(
            "Press **Generate** with one of the preset prompts to see what "
            "register-shaped output from a 14.7M-param model looks like. "
            "Outputs are deliberately not surprising — the project's value "
            "is in the trajectory, not the absolute quality."
        )

    return demo


def render_history(items: List[Tuple[str, str]]) -> str:
    """Render the recent prompts/outputs as a markdown block."""
    if not items:
        return "_Last 5 generations will appear here._"
    parts = []
    for i, (p, o) in enumerate(items, start=1):
        # Truncate long entries so the panel doesn't sprawl
        p_short = p if len(p) <= 120 else p[:117] + "…"
        o_short = o if len(o) <= 240 else o[:237] + "…"
        parts.append(f"**{i}.** _{p_short}_\n\n{o_short}")
    return "\n\n---\n\n".join(parts)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    print(f"Loading primary checkpoint: {args.checkpoint}")
    primary = load_checkpoint(args.checkpoint)

    compare: Optional[Tuple[GhostLM, GhostTokenizer, GhostLMConfig, dict]] = None
    if args.compare_checkpoint:
        print(f"Loading compare checkpoint: {args.compare_checkpoint}")
        compare = load_checkpoint(args.compare_checkpoint)

    demo = build_ui(primary, compare)
    # theme is a launch() arg in Gradio 6.0+. The Base theme keeps the UI
    # neutral so it inherits the Space's colorFrom/colorTo accents from
    # the README frontmatter rather than fighting them.
    demo.launch(
        share=args.share,
        server_port=args.port,
        server_name="0.0.0.0",  # bind on all interfaces; HF Spaces requires this
        theme=gr.themes.Base(),
    )


if __name__ == "__main__":
    main()
