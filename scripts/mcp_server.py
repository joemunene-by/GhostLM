#!/usr/bin/env python3
"""GhostLM MCP server, exposing the cybersecurity model as a Claude tool.

Speaks the Model Context Protocol over stdio. Claude Desktop / Claude Code
users register the server with::

    claude mcp add ghostlm -- python3 /path/to/GhostLM/scripts/mcp_server.py \\
        --checkpoint /path/to/checkpoints/phase5_chat/best_model.pt

After that, seven tools become available inside any Claude conversation:

Agent loop (full GhostAgent runtime with bet-1 tool dispatch + bet-9 cite tags):

- ``ghostlm_agent(query, max_iters)``   run the full agent loop. The model
  sees the cybersec system prompt, may emit <|tool_call|> blocks that the
  runtime dispatches against the canonical CVE / MITRE / CWE / RAG tools,
  and produces a cite-tagged final answer. Use this when you want the
  model to do tool-grounded reasoning, not just direct chat.

Direct model invocation (subject to 81M-scale hallucination):

- ``ghostlm_query(question)``      free-form security Q&A.
- ``ghostlm_explain_cve(cve_id)``  explain a specific CVE.
- ``ghostlm_map_to_attack(text)``  map a description to MITRE ATT&CK techniques.
- ``ghostlm_rag_query(question)``  retrieval-augmented chat (top-K passages from the corpus).

Deterministic / fact-grounded (no model invocation, prefer these for factual lookups):

- ``ghostlm_search_cve_nvd(cve_id)``       canonical CVE data via NVD REST API.
- ``ghostlm_lookup_mitre_technique(tid)``  exact technique text from the local MITRE shard.

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

from ghostlm.agent import GhostAgent, RuntimeConfig
from ghostlm.agent.runner import make_generator_from_loaded
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
# Model state. Loaded once at startup, shared across all tool calls.
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
        self._agent: GhostAgent | None = None

    def agent(self, max_iters: int = 6) -> GhostAgent:
        """Lazily-built GhostAgent that reuses this runtime's loaded
        model + tokenizer. Cached after first call; subsequent calls
        with a different ``max_iters`` build a fresh agent on the same
        underlying model so the cap is honoured per-request."""
        if self._agent is not None and self._agent.config.max_iters == max_iters:
            return self._agent
        gen = make_generator_from_loaded(
            self.model, self.config, self.tokenizer, self.device,
            max_new_tokens=384, temperature=0.6,
            top_p=0.9, top_k=0, repetition_penalty=1.15,
        )
        cfg = RuntimeConfig(max_iters=max_iters, max_new_tokens=384,
                              temperature=0.6, top_p=0.9)
        self._agent = GhostAgent(gen, cfg)
        return self._agent

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
            "GhostLM runtime not initialized; start the server via main()"
        )
    return _runtime


@mcp.tool()
def ghostlm_agent(query: str, max_iters: int = 6,
                    include_trace: bool = False) -> str:
    """Run the full GhostAgent loop: tool-using agent over the cybersec model.

    The model sees the GhostAgent system prompt and may emit
    <|tool_call|>{json}<|/tool_call|> blocks. The runtime parses those,
    dispatches them against the canonical tool registry (search_cve_nvd,
    lookup_mitre_technique, lookup_cwe, rag_retrieve), feeds the responses
    back, and lets the model produce a cite-tagged final answer.

    Use this when you want the model to do tool-grounded reasoning over
    structured cybersec data, not just direct chat. v0.9 chat may not
    produce reliable tool calls (it predates the bet-1 SFT corpus); the
    agent loop terminates safely on max_iters in that case.

    Args:
        query: A natural-language security question.
        max_iters: Cap on model -> tool round-trips. Default 6.
        include_trace: If True, prepend a JSON-serialised trace
            (every message, every tool call, every cite tag) before
            the final answer. Useful for debugging the loop.

    Returns:
        The cite-tagged final answer, optionally preceded by a JSON
        trace block.
    """
    import json as _json
    agent = runtime().agent(max_iters=max(1, max_iters))
    trace = agent.run(query)
    if include_trace:
        trace_block = ("```json\n"
                        + _json.dumps(trace.to_dict(), ensure_ascii=False,
                                       indent=2)
                        + "\n```\n\n")
        return trace_block + (trace.final_answer or "")
    return trace.final_answer or ""


@mcp.tool()
def ghostlm_query(question: str) -> str:
    """Ask GhostLM a free-form cybersecurity question.

    Args:
        question: A natural-language security question (vulnerability classes,
            CTF approaches, defensive controls, attack technique walkthroughs).

    Returns:
        The model's answer. Note: GhostLM is a small (81M-param) specialist
        model trained on cybersecurity text; verify any specific facts (CVE
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
            or capability (incident-report excerpts, CTI fragments, hypothetical
            attacker workflows).

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
# Deterministic-lookup tools (no model invocation)
#
# The chat-based tools above route through the GhostLM model, which
# means hallucination on factual lookups is possible at the 81M scale.
# These three tools give Claude fact-grounded sources to consult before
# (or instead of) the chat tools: live NVD for canonical CVE data,
# the local MITRE corpus shard for technique definitions, and a
# retrieval-augmented query path that grounds chat answers in the
# corpus rather than the model's compressed memory of it.
# ---------------------------------------------------------------------------


_MITRE_INDEX: dict[str, str] | None = None


def _load_mitre_index() -> dict[str, str]:
    """Lazy-load mitre_attack.jsonl into a dict keyed by technique ID
    (uppercased). Cached after first call. Returns {} if the corpus
    shard isn't present (e.g. the user installed the MCP server alone
    without cloning data/)."""
    global _MITRE_INDEX
    if _MITRE_INDEX is not None:
        return _MITRE_INDEX
    import json as _json
    repo_root = Path(__file__).resolve().parent.parent
    candidates = [
        repo_root / "data" / "raw" / "mitre_attack.jsonl",
        repo_root / "data" / "raw" / "mitre_full.jsonl",
    ]
    out: dict[str, str] = {}
    for path in candidates:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = _json.loads(line)
                except _json.JSONDecodeError:
                    continue
                if not isinstance(rec, dict):
                    continue
                tid = (rec.get("technique_id") or rec.get("attack_id")
                       or rec.get("id") or rec.get("ref") or "").strip().upper()
                txt = rec.get("text") or rec.get("description") or rec.get("content") or ""
                if tid and txt and tid not in out:
                    out[tid] = str(txt)
    _MITRE_INDEX = out
    print(f"[ghostlm-mcp] mitre index: {len(out)} techniques loaded", file=sys.stderr)
    return out


_RAG_STATE: dict | None = None


def _load_rag_state() -> dict:
    """Lazy-load the RAG index + BGE embedder. Cached. Returns {} on
    any failure (caller treats RAG as optional and falls back to
    bare chat). The same Models repo (Ghostgim/GhostLM-v0.9-experimental)
    used by the Space hosts the index files."""
    global _RAG_STATE
    if _RAG_STATE is not None:
        return _RAG_STATE
    try:
        import json as _json
        import numpy as np
        from huggingface_hub import hf_hub_download
        repo = "Ghostgim/GhostLM-v0.9-experimental"
        index_path = hf_hub_download(repo_id=repo, filename="rag/index.npy", repo_type="model")
        chunks_path = hf_hub_download(repo_id=repo, filename="rag/chunks.jsonl", repo_type="model")
        idx = np.load(index_path)
        if idx.dtype != np.float32:
            idx = idx.astype(np.float32)
        chunks = []
        with open(chunks_path) as f:
            for line in f:
                chunks.append(_json.loads(line))
        from transformers import AutoModel, AutoTokenizer
        e_tok = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
        e_model = AutoModel.from_pretrained("BAAI/bge-small-en-v1.5").eval()
        _RAG_STATE = {"index": idx, "chunks": chunks,
                      "embed_tok": e_tok, "embed_model": e_model}
        print(f"[ghostlm-mcp] rag: {len(chunks)} chunks loaded", file=sys.stderr)
    except Exception as e:  # noqa: BLE001 - RAG is optional
        print(f"[ghostlm-mcp] rag disabled: {type(e).__name__}: {e}", file=sys.stderr)
        _RAG_STATE = {}
    return _RAG_STATE


@mcp.tool()
def ghostlm_search_cve_nvd(cve_id: str) -> str:
    """Look up canonical CVE data via the NVD REST API. Authoritative
    and not subject to model hallucination; use this for any factual
    CVE question before falling back to ghostlm_query.

    Args:
        cve_id: CVE identifier (e.g. CVE-2021-44228).

    Returns:
        Description, CVSS v3 + v2 scores, CWE references, and
        publication dates pulled live from NIST's National Vulnerability
        Database. The text is the canonical NVD content; the model is
        not invoked.
    """
    import json as _json
    import urllib.request
    cid = cve_id.strip().upper()
    url = f"https://services.nvd.nist.gov/rest/json/cves/2.0?cveId={cid}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "ghostlm-mcp/0.9.2"})
        with urllib.request.urlopen(req, timeout=12) as resp:
            data = _json.loads(resp.read())
    except Exception as e:  # noqa: BLE001 - NVD downtime / rate limit is normal
        return f"NVD lookup failed: {type(e).__name__}: {e}"
    items = (data or {}).get("vulnerabilities", [])
    if not items:
        return f"No NVD record for {cid}."
    cve = items[0].get("cve", {})
    desc = next(
        (d.get("value") for d in cve.get("descriptions", []) if d.get("lang") == "en"),
        "(no English description)",
    )
    metrics = cve.get("metrics", {}) or {}
    v3 = next(iter(metrics.get("cvssMetricV31", []) or metrics.get("cvssMetricV30", []) or []), None)
    v2 = next(iter(metrics.get("cvssMetricV2", []) or []), None)
    s3 = (v3 or {}).get("cvssData", {}).get("baseScore") if v3 else None
    s2 = (v2 or {}).get("cvssData", {}).get("baseScore") if v2 else None
    cwes: list[str] = []
    for w in cve.get("weaknesses", []) or []:
        for d in (w.get("description") or []):
            if d.get("lang") == "en":
                val = d.get("value")
                if val and val not in cwes:
                    cwes.append(val)
    lines = [
        f"{cid}",
        f"Description: {desc}",
        f"CVSS v3: {s3 if s3 is not None else '(unscored)'}",
        f"CVSS v2: {s2 if s2 is not None else '(unscored)'}",
        f"CWEs: {', '.join(cwes) if cwes else '(none listed)'}",
        f"Published: {cve.get('published', '?')}",
        f"Modified:  {cve.get('lastModified', '?')}",
    ]
    return "\n".join(lines)


@mcp.tool()
def ghostlm_lookup_mitre_technique(technique_id: str) -> str:
    """Look up a MITRE ATT&CK technique from the local corpus mirror.

    Args:
        technique_id: MITRE technique ID (e.g. T1059, T1059.001, TA0001).

    Returns:
        The technique's description as captured in GhostLM's corpus.
        Deterministic lookup against the bundled mitre_attack /
        mitre_full shards; the model is not invoked. Use this when
        you want canonical text rather than the model's compressed
        memory of it.
    """
    tid = technique_id.strip().upper()
    idx = _load_mitre_index()
    if not idx:
        return ("MITRE index not available "
                "(data/raw/mitre_attack.jsonl + mitre_full.jsonl missing).")
    if tid in idx:
        return f"{tid}\n\n{idx[tid]}"
    return f"No MITRE record for {tid} in the local corpus."


@mcp.tool()
def ghostlm_rag_query(question: str, top_k: int = 4) -> str:
    """Ask GhostLM with retrieval augmentation.

    Args:
        question: A natural-language security question.
        top_k: How many corpus passages to inject as context (default 4).

    Returns:
        The model's answer, conditioned on the top-K most-similar
        passages retrieved from the GhostLM corpus index (BGE-small
        embeddings, ~83K cybersec chunks). Substantially reduces the
        hallucination floor of bare ghostlm_query. Falls back to bare
        chat if the RAG index isn't available (e.g. offline, or the
        Models repo download failed).
    """
    rag = _load_rag_state()
    if not rag:
        return runtime().chat(question)
    import numpy as np
    import torch.nn.functional as F
    text = "Represent this sentence for searching relevant passages: " + question
    enc = rag["embed_tok"](
        text, padding=True, truncation=True, max_length=512, return_tensors="pt",
    )
    with torch.no_grad():
        out = rag["embed_model"](**enc)
    emb = out.last_hidden_state[:, 0]
    emb = F.normalize(emb, p=2, dim=-1)
    q_vec = emb.cpu().to(torch.float32).numpy().reshape(-1)
    scores = rag["index"] @ q_vec
    idxs = np.argsort(-scores)[: max(1, top_k)]
    refs: list[str] = []
    for i, j in enumerate(idxs):
        ch = rag["chunks"][int(j)]
        snippet = (ch.get("text") or "")[:400]
        if len(ch.get("text") or "") > 400:
            snippet = snippet.rsplit(" ", 1)[0] + "..."
        refs.append(f"[{i+1}] ({ch.get('source', '?')} {ch.get('ref', '')}) {snippet}")
    prompt = (
        "Reference passages from the cybersecurity corpus:\n\n"
        + "\n\n".join(refs)
        + "\n\nUse the reference passages above to answer the question. If the "
        "passages don't contain the answer, say so rather than guessing.\n\n"
        f"Question: {question}"
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
