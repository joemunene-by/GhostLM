"""Output parsing for GhostAgent.

The bet 1 SFT format puts tool calls inside literal tag strings:

    <|tool_call|>{"name": "<TOOL>", "args": {...}}<|/tool_call|>

The bet 9 SFT format adds inline citations:

    The CVE is X <|cite|>nvd:CVE-2017-0144#description<|/cite|>.

This module parses the model's raw output text into:

  - ``ToolCall`` objects: zero, one, or many tool calls extracted
    from ``<|tool_call|>{json}<|/tool_call|>`` blocks. Each carries
    the parsed ``name`` and ``args`` dict.

  - Cite tags: a list of ``{source_type, source_id, field}`` dicts
    extracted from ``<|cite|>{type}:{id}[#field]<|/cite|>`` tags.

  - Plain text: everything that wasn't a tool-call or cite tag,
    concatenated. This is what the runtime treats as the
    'spoken' answer when no tool calls are present.

The parser is deliberately lenient on whitespace, code-fence
wrapping, and minor deviations (e.g. ``<|tool call|>`` with a space
gets normalised before parsing so traces from less-trained
checkpoints don't crash the runtime). Strict-mode parsing for
training-data validation happens in scripts/distill_tool_use.py.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# Regexes for the bet-1 tool-call wire format. The lazy quantifier
# (.+?) is essential to handle multiple tool calls in one message.
_TOOL_CALL_RE = re.compile(
    r"<\|tool_call\|>(.+?)<\|/tool_call\|>",
    re.DOTALL,
)
_TOOL_RESPONSE_RE = re.compile(
    r"<\|tool_response\|>(.+?)<\|/tool_response\|>",
    re.DOTALL,
)
_CITE_RE = re.compile(r"<\|cite\|>([^<]+)<\|/cite\|>")


@dataclass
class ToolCall:
    """One parsed tool call from an assistant message."""
    name: str
    args: Dict[str, Any]
    raw_json: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "args": self.args}


@dataclass
class Cite:
    """One parsed citation tag."""
    source_type: str
    source_id: str
    field: Optional[str] = None
    raw: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = {"source_type": self.source_type, "source_id": self.source_id}
        if self.field:
            d["field"] = self.field
        return d


@dataclass
class ParsedOutput:
    """Result of parsing a raw assistant output string."""
    plain_text: str
    tool_calls: List[ToolCall] = field(default_factory=list)
    cites: List[Cite] = field(default_factory=list)
    parse_warnings: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Tag normalisation
# ---------------------------------------------------------------------------


_NORMALISATIONS = [
    # Common deviations a not-yet-trained model produces; tolerate
    # them so the runtime doesn't choke on noisy v0.9 chat output.
    ("<|tool call|>", "<|tool_call|>"),
    ("<|/tool call|>", "<|/tool_call|>"),
    ("<|tool response|>", "<|tool_response|>"),
    ("<|/tool response|>", "<|/tool_response|>"),
    ("<TOOL_CALL>", "<|tool_call|>"),
    ("</TOOL_CALL>", "<|/tool_call|>"),
]


def normalise_tags(text: str) -> Tuple[str, List[str]]:
    """Apply lenient tag normalisations. Returns the normalised text
    plus a list of warnings describing what was changed."""
    warnings: List[str] = []
    out = text
    for old, new in _NORMALISATIONS:
        if old in out:
            warnings.append(f"normalised {old!r} -> {new!r}")
            out = out.replace(old, new)
    return out, warnings


# ---------------------------------------------------------------------------
# Tool calls
# ---------------------------------------------------------------------------


def parse_tool_calls(text: str) -> Tuple[List[ToolCall], List[str]]:
    """Extract ``<|tool_call|>{json}<|/tool_call|>`` blocks. Returns
    the list of ToolCall objects plus warnings for anything that
    looked tool-call-ish but didn't parse."""
    calls: List[ToolCall] = []
    warnings: List[str] = []
    for m in _TOOL_CALL_RE.finditer(text):
        body = m.group(1).strip()
        # Strip an optional ```json fence the model sometimes wraps
        # the body in.
        body = re.sub(r"^```(?:json)?\s*", "", body)
        body = re.sub(r"```\s*$", "", body)
        try:
            obj = json.loads(body)
        except json.JSONDecodeError as e:
            warnings.append(f"tool_call body did not parse as JSON: {e}")
            continue
        if not isinstance(obj, dict):
            warnings.append(f"tool_call body parsed but is not a dict: {obj!r}")
            continue
        name = obj.get("name")
        args = obj.get("args", {})
        if not isinstance(name, str) or not name:
            warnings.append(f"tool_call missing valid 'name': {obj!r}")
            continue
        if not isinstance(args, dict):
            warnings.append(f"tool_call 'args' is not a dict: {args!r}")
            continue
        calls.append(ToolCall(name=name, args=args, raw_json=body))
    return calls, warnings


# ---------------------------------------------------------------------------
# Cite tags
# ---------------------------------------------------------------------------


def parse_cite_tags(text: str) -> Tuple[List[Cite], List[str]]:
    """Extract ``<|cite|>{type}:{id}[#field]<|/cite|>`` tags."""
    cites: List[Cite] = []
    warnings: List[str] = []
    for m in _CITE_RE.finditer(text):
        body = m.group(1).strip()
        if not body:
            warnings.append("empty cite tag")
            continue
        if ":" not in body:
            warnings.append(f"cite missing 'type:id' shape: {body!r}")
            continue
        st, _, rest = body.partition(":")
        if "#" in rest:
            sid, _, field = rest.partition("#")
        else:
            sid, field = rest, None
        if not st or not sid:
            warnings.append(f"cite has empty source_type or source_id: {body!r}")
            continue
        cites.append(Cite(
            source_type=st.strip(), source_id=sid.strip(),
            field=field.strip() if field else None, raw=body,
        ))
    return cites, warnings


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------


def strip_tool_call_blocks(text: str) -> str:
    """Remove every ``<|tool_call|>...<|/tool_call|>`` block from
    text, leaving the surrounding plain text. Used to derive the
    spoken answer."""
    return _TOOL_CALL_RE.sub("", text).strip()


def parse_agent_output(raw_text: str) -> ParsedOutput:
    """Parse a raw assistant output into structured form.

    The runtime calls this on every model output. If
    ``parsed.tool_calls`` is non-empty, the runtime executes those
    calls and continues the loop. If empty, ``parsed.plain_text``
    is taken as the final answer and the loop terminates.
    """
    if not isinstance(raw_text, str):
        return ParsedOutput(
            plain_text="",
            parse_warnings=[f"non-string input: {type(raw_text).__name__}"],
        )

    normalised, norm_warnings = normalise_tags(raw_text)
    calls, call_warnings = parse_tool_calls(normalised)
    cites, cite_warnings = parse_cite_tags(normalised)
    plain = strip_tool_call_blocks(normalised)

    return ParsedOutput(
        plain_text=plain,
        tool_calls=calls,
        cites=cites,
        parse_warnings=norm_warnings + call_warnings + cite_warnings,
    )
