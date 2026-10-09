"""Tests for the BTX phase plan and the supervisor's phase handling."""

import importlib.util
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts/background"))
import phases  # noqa: E402

spec = importlib.util.spec_from_file_location("supervisor", ROOT / "scripts/background/supervisor.py")
sup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sup)

CFG = dict(sup.DEFAULT_CONFIG, max_steps=15000, wsd_decay_frac=0.2, btx=True)


def test_plan_without_btx_is_just_pretrain():
    assert [p["name"] for p in phases.plan(dict(sup.DEFAULT_CONFIG))] == ["pretrain"]


def test_plan_with_btx_branches_then_merge_then_router():
    names = [p["name"] for p in phases.plan(CFG)]
    assert names == ["pretrain", "branch_cybersec", "branch_code", "branch_math", "branch_general",
                     "merge", "router"]


def test_branch_resumes_pre_decay_and_trains_one_domain(tmp_path, monkeypatch):
    monkeypatch.setattr(sup, "ROOT", tmp_path)
    branch = phases.plan(CFG)[2]
    sup.set_run(branch["run"])
    # Mirror the supervisor, which overwrites max_steps with the phase's own.
    cfg = dict(CFG, pretrain_max_steps=15000, max_steps=phases.phase_max_steps(CFG, branch))
    cmd = sup.train_command(cfg, branch)
    assert cfg["max_steps"] == 12000 + 2300
    assert cmd[cmd.index("--resume") + 1].endswith("checkpoints/ghost_base_mac/pre_decay.pt")
    assert cmd[cmd.index("--curriculum-spec") + 1] == "1.0:code=1.0"
    # Decay covers the last 10% of the branch's own 2,300 steps.
    frac = float(cmd[cmd.index("--wsd-decay-frac") + 1])
    assert round(cfg["max_steps"] * (1 - frac)) == 12000 + 2070

    (sup.CKPT_DIR / "checkpoint_step_12075.pt").touch()
    assert sup.train_command(cfg, branch)[-1].endswith("btx_code/checkpoint_step_12075.pt")


def test_router_phase_uses_merged_moe_with_frozen_experts(tmp_path, monkeypatch):
    monkeypatch.setattr(sup, "ROOT", tmp_path)
    router = phases.plan(CFG)[-1]
    sup.set_run(router["run"])
    cmd = sup.train_command(dict(CFG, max_steps=phases.phase_max_steps(CFG, router)), router)
    assert "--freeze-experts" in cmd and "--resume" not in cmd
    assert cmd[cmd.index("--moe-from") + 1].endswith("ghost_base_moe/merged.pt")
    assert cmd[cmd.index("--max-steps") + 1] == "1500"


def test_merge_command_uses_each_branch_final():
    cmd = phases.merge_command(CFG, Path("/r"), Path("/py"))
    assert "--branch" in cmd and "cybersec=/r/checkpoints/btx_cybersec/final.pt" in cmd
    assert cmd[cmd.index("--top-k") + 1] == "2"


def test_finalize_keeps_weights_only_final_and_requested_files(tmp_path):
    for step in (100, 175):
        torch.save({"step": step, "val_loss": 1.0, "best_val_loss": 1.0, "config": {},
                    "model_state_dict": {"w": torch.ones(2) * step}, "mlx_optimizer_state": {"m": 1}},
                   tmp_path / f"checkpoint_step_{step}.pt")
    (tmp_path / "best_model.pt").touch()
    (tmp_path / "pre_decay.pt").touch()
    final = phases.finalize_run(tmp_path, keep=("pre_decay.pt",))
    ck = torch.load(final, weights_only=False)
    assert ck["step"] == 175 and "mlx_optimizer_state" not in ck
    assert sorted(p.name for p in tmp_path.iterdir()) == ["final.pt", "pre_decay.pt"]


def test_advance_phase_resets_counters_and_finishes_at_the_end(monkeypatch):
    monkeypatch.setattr(sup, "load_config", lambda: dict(CFG))
    monkeypatch.setattr(sup, "notify", lambda msg: None)
    state = {"phase": 0, "crashes": 4, "finished": False}
    sup.advance_phase(CFG, state)
    assert state["phase"] == 1 and state["crashes"] == 0 and state["last_eval_step"] == 12000
    state["phase"] = len(phases.plan(CFG)) - 1
    sup.advance_phase(CFG, state)
    assert state["finished"]
