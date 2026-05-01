#!/usr/bin/env python3
"""GhostLM MCP server — exposes the cybersecurity model as a Claude tool.

Speaks the Model Context Protocol over stdio. Claude Desktop / Claude Code
users register the server with::

    claude mcp add ghostlm -- python3 /path/to/GhostLM/scripts/mcp_server.py \\
        --checkpoint /path/to/checkpoints/phase5_chat/best_model.pt

After that, three tools become available inside any Claude conversation:

- ``ghostlm_query(question)``      — free-form security Q&A.
- ``ghostlm_explain_cve(cve_id)``  — explain a specific CVE.
- ``ghostlm_map_to_attack(text)``  — map a description to MITRE ATT&CK techniques.

Requires Python ≥ 3.10 and ``pip install mcp`` (the official Anthropic SDK).
The model itself runs on whatever device PyTorch picks (MPS on Apple Silicon,
CUDA if available, CPU otherwise).
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import fields
from pathlib import Path

import torch

# Allow running from any cwd by adding the repo root to sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ghostlm.config import GhostLMConfig
from ghostlm.model import GhostLM
from ghostlm.tokenizer import GhostTokenizer

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as e:  # pragma: no cover
    print(
        "ghostlm-mcp requires the 'mcp' package (Python ≥ 3.10):\n"
        "    pip install mcp\n",
        file=sys.stderr,
    )
    raise


# ---------------------------------------------------------------------------
# Model state — loaded once at startup, shared across all tool calls.
# ---------------------------------------------------------------------------


class GhostLMRuntime:
    """Loads a chat-tuned GhostLM checkpoint and exposes a single chat method."""

    def __init__(self, checkpoint_path: str, device: str = "auto") -> None:
        """Load model + tokenizer, resolve device."""
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device

        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        cfg_raw = ckpt["config"]
        if isinstance(cfg_raw, dict):
            cfg = GhostLMConfig(**{
                f.name: cfg_raw[f.name]
                for f in fields(GhostLMConfig)
                if f.name in cfg_raw
            })
        else:
            cfg = cfg_raw
        self.config = cfg

        self.model = GhostLM(cfg)
        state = ckpt.get("model_state_dict", ckpt.get("model"))
        self.model.load_state_dict(state, strict=False)
        self.model.eval()
        self.model = self.model.to(device)

        self.tokenizer = GhostTokenizer()
        self.end_id = self.tokenizer._special_tokens[self.tokenizer.END]

    def chat(
        self,
        prompt: str,
        *,
        max_tokens: int = 300,
        temperature: float = 0.7,
        top_k: int = 40,
        top_p: float = 0.95,
    ) -> str:
        """Run a single user turn through the chat-tuned model."""
        # Late-import to avoid a top-level cycle when this module is imported by tests.
        from scripts.chat import generate_until_end  # noqa: WPS433

        ids = self.tokenizer.format_chat_prompt([{"role": "user", "content": prompt}])
        new_ids = generate_until_end(
            self.model,
            ids,
            end_id=self.end_id,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            device=self.device,
        )
        return self.tokenizer.decode(new_ids).strip()


# ---------------------------------------------------------------------------
# MCP server definition
# ---------------------------------------------------------------------------

mcp = FastMCP("ghostlm")
_runtime: GhostLMRuntime | None = None


def runtime() -> GhostLMRuntime:
    """Return the lazily-initialized runtime singleton.

    The runtime is set in ``main()`` after parsing CLI args and is required
    for every tool call.
    """
    if _runtime is None:
        raise RuntimeError(
            "GhostLM runtime not initialized — start the server via main()"
        )
    return _runtime


@mcp.tool()
def ghostlm_query(question: str) -> str:
    """Ask GhostLM a free-form cybersecurity question.

    Args:
        question: A natural-language security question — vulnerability classes,
            CTF approaches, defensive controls, attack technique walkthroughs.

    Returns:
        The model's answer. Note: GhostLM is a small (45M-param) specialist
        model trained on cybersecurity text — verify any specific facts (CVE
        numbers, exact CVSS scores, dates) against authoritative sources.
    """
    return runtime().chat(question)


@mcp.tool()
def ghostlm_explain_cve(cve_id: str) -> str:
    """Explain a CVE by ID, formatted as a security analyst would summarize it.

    Args:
        cve_id: A CVE identifier in the format CVE-YYYY-NNNNN
            (e.g. "CVE-2021-44228" for Log4Shell).

    Returns:
        Description of the affected product, the vulnerability class, the
        impact, and known mitigations to the extent the model has them.
    """
    cve_id = cve_id.strip()
    return runtime().chat(f"Explain {cve_id}.")


@mcp.tool()
def ghostlm_map_to_attack(description: str) -> str:
    """Suggest MITRE ATT&CK techniques that match an attack description.

    Args:
        description: A free-text description of an observed attack, intrusion,
            or capability — for example incident-report excerpts, CTI fragments,
            or hypothetical attacker workflows.

    Returns:
        A short list of likely MITRE ATT&CK technique IDs and names with brief
        justification per match. Lower-confidence matches are noted.
    """
    prompt = (
        "Given the following description of an attack or capability, list the "
        "most likely MITRE ATT&CK techniques (give technique IDs and names, "
        "and a one-line justification per match). If you're not confident, "
        "say so.\n\n"
        f"Description:\n{description}"
    )
    return runtime().chat(prompt, max_tokens=400)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse CLI args. The MCP server itself takes no flags from Claude."""
    p = argparse.ArgumentParser(description="GhostLM MCP server")
    p.add_argument("--checkpoint", required=True,
                   help="Chat-tuned GhostLM checkpoint .pt file")
    p.add_argument("--device", default="auto")
    return p.parse_args()


def main() -> None:
    """Initialize the runtime and start the MCP stdio server."""
    global _runtime
    args = parse_args()
    print(f"[ghostlm-mcp] loading {args.checkpoint} ({args.device})", file=sys.stderr)
    _runtime = GhostLMRuntime(args.checkpoint, args.device)
    print(f"[ghostlm-mcp] ready on device={_runtime.device}", file=sys.stderr)
    mcp.run()


if __name__ == "__main__":
    main()
