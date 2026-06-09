"""Shared infrastructure for the data-collection scripts.

Every ``collect_*.py`` script reimplements the same four things: an
HTTP GET with a User-Agent and a timeout, polite request pacing, JSONL
record writing in the standard ``{"id", "source", "text", ...}``
schema, and skip/truncate accounting. This module implements them once.

Reference usage (see ``collect_owasp_top10.py`` for a full example):

    from collect_common import http_get_text, http_get_json, JsonlWriter

    files = http_get_json(api_url)
    with JsonlWriter("data/raw/foo.jsonl", source="foo",
                     min_chars=200, max_chars=12000,
                     request_delay=0.5) as out:
        for fname, url in files:
            try:
                md = http_get_text(url)
            except Exception as e:
                out.count_failure(f"{fname}: {e}")
                continue
            out.write(rec_id=f"FOO-{fname}", text=md, title=fname)
    # summary line + skip/truncate/failure accounting printed on exit

The remaining collectors can migrate mechanically; new collectors
should start here.
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

USER_AGENT = "GhostLM-collector/0.9 (+https://github.com/joemunene-by/GhostLM)"


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


def http_get(
    url: str,
    *,
    headers: Optional[Dict[str, str]] = None,
    timeout: float = 30.0,
    retries: int = 3,
    backoff: float = 2.0,
) -> bytes:
    """GET a URL with a User-Agent, timeout, and exponential-backoff retries.

    Raises the last exception if every attempt fails.
    """
    merged = {"User-Agent": USER_AGENT}
    if headers:
        merged.update(headers)
    last_exc: Optional[Exception] = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=merged)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except Exception as e:  # noqa: BLE001 - network surface is wide
            last_exc = e
            if attempt < retries - 1:
                time.sleep(backoff * (2 ** attempt))
    raise last_exc  # type: ignore[misc]


def http_get_text(url: str, *, encoding: str = "utf-8", **kwargs) -> str:
    """GET a URL and decode the body as text (errors ignored)."""
    return http_get(url, **kwargs).decode(encoding, errors="ignore")


def http_get_json(url: str, *, headers: Optional[Dict[str, str]] = None, **kwargs) -> Any:
    """GET a URL and parse the body as JSON."""
    merged = {"Accept": "application/json"}
    if headers:
        merged.update(headers)
    return json.loads(http_get(url, headers=merged, **kwargs))


# ---------------------------------------------------------------------------
# JSONL output
# ---------------------------------------------------------------------------


class JsonlWriter:
    """Standard-schema JSONL writer with dedup, length policy, pacing,
    and skip/truncate/failure accounting.

    Records carry the corpus-wide ``{"id", "source", "text", ...}``
    shape. Exact-duplicate texts (sha256) are dropped. Texts shorter
    than ``min_chars`` are skipped; texts longer than ``max_chars`` are
    truncated at the last paragraph boundary that fits. After every
    accepted or failed record the writer sleeps ``request_delay``
    seconds so collectors stay polite to upstreams by default.

    Use as a context manager; a summary is printed on exit.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        source: str,
        min_chars: int = 0,
        max_chars: int = 0,
        request_delay: float = 0.0,
        append: bool = False,
    ):
        self.path = Path(path)
        self.source = source
        self.min_chars = min_chars
        self.max_chars = max_chars
        self.request_delay = request_delay
        self.written = 0
        self.skipped_short = 0
        self.skipped_dupe = 0
        self.truncated = 0
        self.failed = 0
        self._hashes: set = set()

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a" if append else "w", encoding="utf-8")

    def write(self, *, rec_id: str, text: str, **extra: Any) -> bool:
        """Write one record. Returns True if it was accepted."""
        text = text.strip()
        if self.min_chars and len(text) < self.min_chars:
            self.skipped_short += 1
            self._pace()
            return False
        if self.max_chars and len(text) > self.max_chars:
            clipped = text[: self.max_chars].rsplit("\n\n", 1)[0]
            text = clipped if clipped else text[: self.max_chars]
            self.truncated += 1

        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if digest in self._hashes:
            self.skipped_dupe += 1
            self._pace()
            return False
        self._hashes.add(digest)

        rec = {"id": rec_id, "source": self.source, "text": text, **extra}
        self._fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.written += 1
        self._pace()
        return True

    def count_failure(self, message: str = "") -> None:
        """Record a fetch/parse failure (and keep pacing)."""
        self.failed += 1
        if message:
            print(f"  {message}")
        self._pace()

    def _pace(self) -> None:
        if self.request_delay > 0:
            time.sleep(self.request_delay)

    def close(self) -> None:
        self._fh.close()

    def summary(self) -> str:
        parts = [f"Wrote {self.written} {self.source} records to {self.path}"]
        if self.skipped_short:
            parts.append(f"  Skipped {self.skipped_short} too-short")
        if self.skipped_dupe:
            parts.append(f"  Skipped {self.skipped_dupe} exact duplicates")
        if self.truncated:
            parts.append(f"  Truncated {self.truncated} to {self.max_chars} chars")
        if self.failed:
            parts.append(f"  Failed {self.failed}")
        return "\n".join(parts)

    def __enter__(self) -> "JsonlWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
        print(self.summary())


def iter_jsonl(path: str | Path) -> Iterator[dict]:
    """Yield records from a JSONL file, skipping blank lines."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)
