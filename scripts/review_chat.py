"""Review gate for model-written chat data: nothing from data/chat_gen/ or teacher_sft.jsonl reaches training unread.

``next`` writes a batch of unreviewed conversations for a reviewer to read line by line.
``apply`` takes the reviewer's verdicts (keep, fix with corrected turns, or drop with a reason)
and files the survivors under data/chat_reviewed/, which build_chat_mix.py reads.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_seed import problems  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/chat_gen"
REVIEWED = ROOT / "data/chat_reviewed"
DONE = REVIEWED / "reviewed_ids.txt"
DROPS = REVIEWED / "dropped.jsonl"
TEACHER = ROOT / "data/processed/teacher_sft.jsonl"


def rid(rec: dict) -> str:
    return hashlib.sha1(json.dumps(rec["turns"], ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:16]


def raw_records() -> list[dict]:
    out = []
    for f in sorted(RAW.glob("*.jsonl")) + [TEACHER]:
        if not f.exists():
            continue
        for line in f.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                out.append({**rec, "id": rid(rec), "area": "teacher_rag" if f == TEACHER else f.stem})
    return out


def done_ids() -> set[str]:
    return set(DONE.read_text().split()) if DONE.exists() else set()


def cmd_next(n: int, out: Path, skip: int) -> None:
    done = done_ids()
    todo = [r for r in raw_records() if r["id"] not in done][skip:skip + n]
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in todo))
    print(f"wrote {len(todo)} conversations to {out}")


def cmd_apply(path: Path) -> None:
    REVIEWED.mkdir(parents=True, exist_ok=True)
    pending = {r["id"]: r for r in raw_records()}
    done = done_ids()
    kept = fixed = dropped = 0
    for n, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        v = json.loads(line)
        src = pending.get(v["id"])
        if src is None or v["id"] in done:
            continue
        if v["verdict"] == "drop":
            with DROPS.open("a") as f:
                f.write(json.dumps({"id": v["id"], "reason": v.get("reason", ""), "turns": src["turns"]},
                                   ensure_ascii=False) + "\n")
            dropped += 1
        else:
            turns = v["turns"] if v["verdict"] == "fix" else src["turns"]
            rec = {"turns": turns, "source": src["source"], "teacher": src.get("teacher", "opencode"),
                   "review": v["verdict"]}
            # Reviewed records must still pass the same mechanical bar as hand-written seeds.
            errs = [e for e in problems({**rec, "source": "seed_check"}) if "source" not in e]
            if errs:
                print(f"{path.name}:{n}: {v['id']} rejected after review: {'; '.join(errs)}")
                continue
            with (REVIEWED / f"{src['area']}.jsonl").open("a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            kept += v["verdict"] == "keep"
            fixed += v["verdict"] == "fix"
        with DONE.open("a") as f:
            f.write(v["id"] + "\n")
        done.add(v["id"])
    print(f"kept {kept}, fixed {fixed}, dropped {dropped}")


def cmd_status() -> None:
    done = done_ids()
    raw = raw_records()
    counts = {f.stem: sum(1 for line in f.read_text().splitlines() if line.strip()) for f in sorted(REVIEWED.glob("*.jsonl"))
              if f.name != DROPS.name} if REVIEWED.exists() else {}
    drops = sum(1 for _ in DROPS.open()) if DROPS.exists() else 0
    print(f"raw {len(raw)}, reviewed {len(done & {r['id'] for r in raw})}, "
          f"approved {sum(counts.values())} {counts}, dropped {drops}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    nx = sub.add_parser("next")
    nx.add_argument("--n", type=int, default=60)
    nx.add_argument("--skip", type=int, default=0, help="offset, so parallel reviewers get different batches")
    nx.add_argument("--out", type=Path, required=True)
    ap = sub.add_parser("apply")
    ap.add_argument("verdicts", type=Path)
    sub.add_parser("status")
    a = p.parse_args()
    if a.cmd == "next":
        cmd_next(a.n, a.out, a.skip)
    elif a.cmd == "apply":
        cmd_apply(a.verdicts)
    else:
        cmd_status()


if __name__ == "__main__":
    main()
