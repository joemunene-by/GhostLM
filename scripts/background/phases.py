"""Phase plan for the background run: dense pretrain, then optional BTX expert branching.

    pretrain  ->  branch_<domain> x N  ->  merge  ->  router

Each branch resumes the pretrain's ``pre_decay.pt`` (the last flat-LR state of the
WSD schedule) and keeps training on one domain only; ``merge`` turns the branches
into a Mixture-of-Experts (scripts/btx_merge.py); ``router`` trains the routers,
attention and norms of that MoE on mixed data with the experts frozen. Enabled by
a ``"btx"`` block in .bg/train_config.json; without it the plan is just pretrain.
"""

from __future__ import annotations

import re
from pathlib import Path

PRETRAIN_RUN = "ghost_base_mac"
MOE_RUN = "ghost_base_moe"

DEFAULT_BTX = {
    "branches": {
        "cybersec": {"cybersec": 1.0},
        "code": {"code": 1.0},
        "math": {"math": 1.0},
        "general": {"general_web": 1.0, "knowledge": 1.0, "instruction": 0.2},
    },
    "branch_steps": 2300,
    "branch_decay_frac": 0.1,
    "top_k": 2,
    "router_steps": 1500,
    "router_learning_rate": 3e-4,
    "router_warmup_steps": 100,
}


def btx_config(cfg: dict) -> dict | None:
    if not cfg.get("btx"):
        return None
    out = dict(DEFAULT_BTX)
    out.update(cfg["btx"] if isinstance(cfg["btx"], dict) else {})
    return out


def plan(cfg: dict) -> list[dict]:
    phases = [{"name": "pretrain", "run": PRETRAIN_RUN, "kind": "train"}]
    btx = btx_config(cfg)
    if btx:
        for domain in btx["branches"]:
            phases.append({"name": f"branch_{domain}", "run": f"btx_{domain}", "kind": "train", "domain": domain})
        phases.append({"name": "merge", "run": MOE_RUN, "kind": "merge"})
        phases.append({"name": "router", "run": MOE_RUN, "kind": "train"})
    return phases


def _pretrain_max(cfg: dict) -> int:
    # The supervisor overwrites max_steps with the current phase's; this keeps the original.
    return cfg.get("pretrain_max_steps", cfg["max_steps"])


def pretrain_decay_start(cfg: dict) -> int:
    return int(_pretrain_max(cfg) * (1 - cfg.get("wsd_decay_frac", 0.2)))


def phase_max_steps(cfg: dict, phase: dict) -> int:
    btx = btx_config(cfg)
    if phase["name"].startswith("branch_"):
        return pretrain_decay_start(cfg) + btx["branch_steps"]
    if phase["name"] == "router":
        return btx["router_steps"]
    return _pretrain_max(cfg)


def phase_overrides(cfg: dict, phase: dict, root: Path) -> list:
    """Trainer arguments that differ from the pretrain for this phase."""
    btx = btx_config(cfg)
    if phase["name"].startswith("branch_"):
        max_steps = phase_max_steps(cfg, phase)
        weights = btx["branches"][phase["domain"]]
        spec = "1.0:" + ",".join(f"{d}={w}" for d, w in weights.items())
        decay_frac = btx["branch_decay_frac"] * btx["branch_steps"] / max_steps
        return ["--curriculum-spec", spec, "--wsd-decay-frac", f"{decay_frac:.6f}"]
    if phase["name"] == "router":
        return ["--moe-from", str(root / "checkpoints" / MOE_RUN / "merged.pt"), "--freeze-experts",
                "--learning-rate", str(btx["router_learning_rate"]),
                "--warmup-steps", str(btx["router_warmup_steps"])]
    return []


def initial_resume(phase: dict, root: Path) -> Path | None:
    """Checkpoint a phase starts from before it has any of its own."""
    if phase["name"].startswith("branch_"):
        return root / "checkpoints" / PRETRAIN_RUN / "pre_decay.pt"
    return None


def merge_command(cfg: dict, root: Path, py: Path) -> list:
    btx = btx_config(cfg)
    cmd = [str(py), "scripts/btx_merge.py", "--top-k", str(btx["top_k"]),
           "--out", str(root / "checkpoints" / MOE_RUN / "merged.pt")]
    for domain in btx["branches"]:
        cmd += ["--branch", f"{domain}={root / 'checkpoints' / f'btx_{domain}' / 'final.pt'}"]
    return cmd


def finalize_run(ckpt_dir: Path, keep: tuple = ()) -> Path | None:
    """Write the newest checkpoint as weights-only final.pt and delete rolling checkpoints.

    Anything named in ``keep`` (e.g. pre_decay.pt, averaged.pt) survives; best_model.pt
    is removed unless kept, since final.pt now holds the end-of-phase weights.
    """
    import torch  # lazy: keeps the long-running supervisor from holding torch in memory

    rolling = sorted(ckpt_dir.glob("checkpoint_step_*.pt"),
                     key=lambda p: int(re.search(r"(\d+)", p.stem).group(1)))
    if not rolling:
        return None
    ck = torch.load(rolling[-1], map_location="cpu", weights_only=False, mmap=True)
    final = ckpt_dir / "final.pt"
    tmp = final.with_name("final.pt.tmp")
    torch.save({k: ck[k] for k in ("step", "val_loss", "best_val_loss", "model_state_dict", "config") if k in ck},
               tmp)
    tmp.replace(final)
    del ck
    for p in rolling:
        p.unlink(missing_ok=True)
    if "best_model.pt" not in keep:
        (ckpt_dir / "best_model.pt").unlink(missing_ok=True)
    return final
