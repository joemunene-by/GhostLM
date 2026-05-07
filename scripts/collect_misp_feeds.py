#!/usr/bin/env python3
"""Collect threat-intelligence events from open MISP OSINT feeds.

MISP (Malware Information Sharing Platform) feeds are JSON-formatted
threat-intel events with structured attributes: CVE references, IPs,
domains, file hashes, mutex names, MITRE ATT&CK mappings, and
free-text descriptions of attacks. Two well-known free OSINT feeds:

  CIRCL OSINT      circl.lu/doc/misp/feed-osint/
  BotvrijEU OSINT  botvrij.eu/data/feed-osint/

Both publish a `manifest.json` listing event UUIDs with metadata,
plus per-event `<uuid>.json` files containing the full event tree.
This collector reads the manifest, fetches each event, and renders
the structured event tree as a prose paragraph the LM can train on.

Why this matters for GhostLM: MISP events are the operational
ground-truth of how threat intel looks in practice. The corpus
already has descriptive prose (CVE descriptions, MITRE technique
text, vendor blog writeups); MISP feeds add the *structured-IOC*
register the model otherwise never sees. Helps when a downstream
user asks GhostLM something like "what does a typical Cobalt
Strike beacon C2 look like" and the answer needs to surface real
IoC patterns rather than register-only fiction.

Output: ``data/raw/misp_feeds.jsonl`` with one record per MISP
event. Source field is ``misp_feeds``. Per-record event UUID,
feed name, info string preserved for traceability.

License posture: CIRCL and BotvrijEU OSINT feeds are released for
research use; CIRCL specifies CC0, BotvrijEU specifies "free to
use". Per-event source_url + feed gives attribution.

Run:

    PYTHONPATH=. python3 scripts/collect_misp_feeds.py \\
        --max-events-per-feed 200 --request-delay 0.5
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple


FEEDS: List[Tuple[str, str]] = [
    ("circl-osint",     "https://www.circl.lu/doc/misp/feed-osint/"),
    ("botvrij-osint",   "https://www.botvrij.eu/data/feed-osint/"),
]

USER_AGENT = "GhostLM-corpus-collector/0.9.3"


def http_get_text(url: str, timeout: int = 30) -> Optional[str]:
    """GET returning decoded text or None."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            ctype = resp.headers.get("Content-Type", "")
            charset = "utf-8"
            m = re.search(r"charset=([\w-]+)", ctype, re.I)
            if m:
                charset = m.group(1)
            return resp.read().decode(charset, errors="replace")
    except Exception as e:
        print(f"    fetch failed: {url[:90]}: {type(e).__name__}: {e}")
        return None


def http_get_json(url: str, timeout: int = 30) -> Optional[dict]:
    """GET parsing JSON or None."""
    text = http_get_text(url, timeout)
    if text is None:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        print(f"    json parse failed: {url[:90]}: {e}")
        return None


def render_misp_event(feed_name: str, event_uuid: str, payload: dict) -> Optional[Dict]:
    """Convert a MISP event JSON tree into a single training record.

    The event JSON shape (per MISP spec): top-level "Event" with
    fields info, date, threat_level_id, analysis, and a list of
    Attribute (and possibly Object) entries. We render as prose:

        Event <UUID>: <info string>
        Date: <YYYY-MM-DD>
        Threat level: <1-4>
        Analysis: <0-2>
        Attributes:
          - <type> | <category> | <value>  (count: N)
        Tags: <list>

    Filters out null/empty events. Returns None if the event has no
    info string or zero attributes (those are stub events worth nothing
    to a training corpus)."""
    event = payload.get("Event")
    if not isinstance(event, dict):
        return None
    info = (event.get("info") or "").strip()
    if not info:
        return None
    attrs = event.get("Attribute") or []
    objects = event.get("Object") or []
    # Count attributes including object-attributes.
    flat_attrs: List[Dict] = list(attrs)
    for o in objects:
        for a in (o.get("Attribute") or []):
            flat_attrs.append(a)
    if not flat_attrs:
        return None

    lines: List[str] = []
    lines.append(f"Event {event_uuid}: {info}")
    if event.get("date"):
        lines.append(f"Date: {event['date']}")
    if event.get("threat_level_id"):
        lines.append(f"Threat level: {event['threat_level_id']}")
    if event.get("analysis"):
        lines.append(f"Analysis: {event['analysis']}")
    tags = []
    for tag in (event.get("Tag") or []):
        if isinstance(tag, dict) and tag.get("name"):
            tags.append(tag["name"])
    if tags:
        lines.append(f"Tags: {', '.join(tags[:20])}")
    lines.append("")
    lines.append(f"Attributes ({len(flat_attrs)} total):")
    # Group attributes by (type, category) so we don't render 500 raw rows.
    grouped: Dict[Tuple[str, str], int] = {}
    samples: Dict[Tuple[str, str], List[str]] = {}
    for a in flat_attrs:
        t = a.get("type", "?")
        c = a.get("category", "?")
        v = a.get("value", "")
        key = (t, c)
        grouped[key] = grouped.get(key, 0) + 1
        if len(samples.setdefault(key, [])) < 3 and v:
            samples[key].append(v)
    for (t, c), count in sorted(grouped.items(), key=lambda kv: -kv[1])[:30]:
        sample_str = ", ".join(samples.get((t, c), [])[:3])
        if len(sample_str) > 240:
            sample_str = sample_str[:240] + "..."
        lines.append(f"  {t} | {c} | count={count} | examples: {sample_str}")

    text = "\n".join(lines)
    return {
        "id": f"misp_feeds#{feed_name}#{event_uuid}",
        "source": "misp_feeds",
        "feed": feed_name,
        "event_uuid": event_uuid,
        "title": info,
        "date": event.get("date"),
        "n_attributes": len(flat_attrs),
        "text": text,
    }


def load_seen_uuids(path: Path) -> Set[str]:
    out: Set[str] = set()
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                u = rec.get("event_uuid")
                if u:
                    out.add(u)
            except json.JSONDecodeError:
                continue
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="data/raw/misp_feeds.jsonl")
    p.add_argument("--max-events-per-feed", type=int, default=200)
    p.add_argument("--max-events-total", type=int, default=0,
                   help="Total cap; 0 = no cap")
    p.add_argument("--request-delay", type=float, default=0.5)
    args = p.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    seen = load_seen_uuids(out_path)
    print(f"Output:    {out_path}")
    print(f"Resuming:  {len(seen)} events already in output")
    print(f"Feeds:     {len(FEEDS)}")
    print()

    total_written = 0
    fh = out_path.open("a", encoding="utf-8")
    try:
        for feed_name, base_url in FEEDS:
            if args.max_events_total and total_written >= args.max_events_total:
                break
            print(f"=== {feed_name} ({base_url}) ===")
            manifest = http_get_json(base_url.rstrip("/") + "/manifest.json")
            if not manifest:
                print("    manifest fetch failed; skipping")
                continue
            event_uuids = list(manifest.keys())
            if not event_uuids:
                print("    manifest empty (unexpected schema?)")
                continue
            print(f"    {len(event_uuids)} events in manifest, sampling first {args.max_events_per_feed}")
            per_feed = 0
            for uuid in event_uuids[: args.max_events_per_feed]:
                if args.max_events_total and total_written >= args.max_events_total:
                    break
                if uuid in seen:
                    continue
                time.sleep(args.request_delay)
                event_url = f"{base_url.rstrip('/')}/{uuid}.json"
                payload = http_get_json(event_url)
                if not payload:
                    continue
                rec = render_misp_event(feed_name, uuid, payload)
                if not rec:
                    continue
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                seen.add(uuid)
                per_feed += 1
                total_written += 1
            print(f"    wrote {per_feed} new events")
    finally:
        fh.close()
    print(f"\nDone. {total_written} new MISP events written to {out_path}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
