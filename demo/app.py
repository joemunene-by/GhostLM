"""GhostLM Gradio demo — web interface for interactive text generation."""

import sys
from dataclasses import fields
from pathlib import Path

import torch

from ghostlm.config import GhostLMConfig
from ghostlm.model import GhostLM
from ghostlm.tokenizer import GhostTokenizer

try:
    import gradio as gr
except ImportError:
    print("Install gradio: pip install gradio")
    sys.exit(1)


# Global model state
model = None
tokenizer = None
config = None


def load_model(checkpoint_path=None):
    """Load GhostLM model from checkpoint or initialize randomly.

    Args:
        checkpoint_path: Path to .pt checkpoint file, or None for random init.

    Returns:
        Tuple of (model, tokenizer, config).
    """
    global model, tokenizer, config

    tokenizer = GhostTokenizer()

    if checkpoint_path and Path(checkpoint_path).exists():
        print(f"Loading GhostLM from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        saved_config = checkpoint["config"]
        config = GhostLMConfig(**{
            f.name: saved_config[f.name]
            for f in fields(GhostLMConfig)
            if f.name in saved_config
        })

        model = GhostLM(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        step = checkpoint.get("step", 0)
        val_loss = checkpoint.get("val_loss", None)
        print(f"  Loaded: step={step}, val_loss={val_loss}")
    else:
        print("No checkpoint found — using random ghost-tiny weights.")
        config = GhostLMConfig.from_preset("ghost-tiny")
        config.vocab_size = 50261
        config.context_length = 128
        model = GhostLM(config)
        model.eval()

    return model, tokenizer, config


def generate_text(prompt, max_tokens, temperature, top_k):
    """Generate text from a user prompt using the loaded GhostLM model.

    Args:
        prompt: Input text string from the user.
        max_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature for generation.
        top_k: Top-k filtering value (0 to disable).

    Returns:
        Generated text string with the prompt stripped from the beginning.
    """
    global model, tokenizer, config

    if not prompt or not prompt.strip():
        return "Please enter a prompt."

    try:
        ids = tokenizer.encode(prompt)
        input_tensor = torch.tensor(ids, dtype=torch.long).unsqueeze(0)

        top_k_val = top_k if top_k > 0 else None

        with torch.no_grad():
            output = model.generate(
                input_tensor,
                max_new_tokens=int(max_tokens),
                temperature=temperature,
                top_k=top_k_val,
            )

        generated_text = tokenizer.decode(output[0].tolist())

        # Strip prompt from output
        if generated_text.startswith(prompt):
            return generated_text[len(prompt):].strip()
        return generated_text.strip()

    except Exception as e:
        return f"Error: {str(e)}"


# Load model at module import time
checkpoint = "checkpoints/best_model.pt"
model, tokenizer, config = load_model(checkpoint)

# Build model info string
model_info = (
    f"**Model:** ghost-tiny ({config.n_layers} layers, {config.d_model} dim)  \n"
    f"**Parameters:** {model.num_params():,}  \n"
    f"**Vocab Size:** {config.vocab_size:,}  \n"
    f"**Context Length:** {config.context_length}"
)

if Path(checkpoint).exists():
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model_info += f"  \n**Val Loss:** {ckpt.get('val_loss', 'N/A')}"

# Create Gradio interface
with gr.Blocks(theme=gr.themes.Base()) as demo:
    gr.Markdown("# 🔐 GhostLM — Cybersecurity Language Model")
    gr.Markdown("Open-source transformer built from scratch in PyTorch")

    gr.Markdown(model_info)

    with gr.Row():
        prompt_box = gr.Textbox(
            label="Your Prompt",
            lines=4,
            placeholder="A SQL injection attack works by...",
        )
        output_box = gr.Textbox(
            label="GhostLM Output",
            lines=4,
            interactive=False,
        )

    with gr.Row():
        max_tokens = gr.Slider(
            minimum=50, maximum=300, value=150, step=10, label="Max Tokens"
        )
        temperature = gr.Slider(
            minimum=0.1, maximum=2.0, value=0.8, step=0.1, label="Temperature"
        )
        top_k = gr.Slider(
            minimum=0, maximum=100, value=50, step=5, label="Top-K"
        )

    with gr.Row():
        generate_btn = gr.Button("Generate", variant="primary")
        clear_btn = gr.Button("Clear")

    gr.Examples(
        examples=[
            "A SQL injection attack works by",
            "The CVE scoring system rates vulnerabilities",
            "To perform a buffer overflow exploit",
            "Ransomware encrypts victim files by",
            "In penetration testing, reconnaissance involves",
            "Zero-day vulnerabilities are dangerous because",
        ],
        inputs=prompt_box,
    )

    gr.Markdown("---")
    gr.Markdown("Built by Joe Munene | [github.com/joemunene-by/GhostLM](https://github.com/joemunene-by/GhostLM) | Apache 2.0")

    generate_btn.click(
        fn=generate_text,
        inputs=[prompt_box, max_tokens, temperature, top_k],
        outputs=output_box,
    )

    clear_btn.click(
        fn=lambda: ("", ""),
        inputs=None,
        outputs=[prompt_box, output_box],
    )

if __name__ == "__main__":
    demo.launch(share=False, server_port=7860)
