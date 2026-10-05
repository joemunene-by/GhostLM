"""Tests for scripts/background/supervisor.py: command building, ETA and the dashboard."""

import importlib.util
import json
import os
import sys
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("supervisor", ROOT / "scripts/background/supervisor.py")
sup = importlib.util.module_from_spec(spec)
sys.modules["supervisor"] = sup
spec.loader.exec_module(sup)


@pytest.fixture
def bg(tmp_path, monkeypatch):
    bg = tmp_path / ".bg"
    (bg / "logs").mkdir(parents=True)
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    for name, value in {"BG": bg, "CKPT_DIR": ckpt, "LOG_DIR": tmp_path / "logs", "STATUS": bg / "STATUS.txt",
                        "PAUSE_FILE": bg / "PAUSE", "TOKEN_FILE": bg / "dashboard_token",
                        "CONFIG": bg / "train_config.json", "MANIFEST": tmp_path / "none.json"}.items():
        monkeypatch.setattr(sup, name, value)
    return bg


def test_train_command_resumes_from_latest_and_maps_flags(bg):
    for step in (75, 150):
        (sup.CKPT_DIR / f"checkpoint_step_{step}.pt").touch()
    cfg = dict(sup.DEFAULT_CONFIG, attn_gate=True, value_residual=False, learning_rate=1e-3)
    cmd = sup.train_command(cfg)
    assert cmd[cmd.index("--resume") + 1].endswith("checkpoint_step_150.pt")
    assert "--attn-gate" in cmd and "--value-residual" not in cmd
    assert cmd[cmd.index("--learning-rate") + 1] == "0.001"
    assert "--best-weights-only" in cmd


def test_prune_keeps_only_newest(bg):
    for step in (75, 150, 225):
        (sup.CKPT_DIR / f"checkpoint_step_{step}.pt").touch()
    sup.prune()
    assert [p.name for p in sup.checkpoints()] == ["checkpoint_step_225.pt"]


def test_eta_from_rate_samples(bg):
    state = {"rate_samples": [[1000.0, 100], [1000.0 + 160 * 10, 110]]}
    assert sup.seconds_per_step(state) == 160
    (sup.CKPT_DIR / "checkpoint_step_110.pt").touch()
    sup.refresh_status("training", dict(sup.DEFAULT_CONFIG, max_steps=200), state)
    assert sup.LIVE["eta_hours"] == pytest.approx(90 * 160 / 3600)
    assert "eta:" in sup.STATUS.read_text()


def test_dashboard_status_and_token_gated_pause(bg):
    sup.LIVE.update({"state": "training", "step": 5})
    server = ThreadingHTTPServer(("127.0.0.1", 0), sup.DashboardHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        status = json.loads(urllib.request.urlopen(base + "/api/status").read())
        assert status["state"] == "training"
        assert b"GhostLM" in urllib.request.urlopen(base + "/").read()

        def post(path, token):
            req = urllib.request.Request(base + path, method="POST", headers={"X-Token": token})
            return urllib.request.urlopen(req).status

        with pytest.raises(urllib.error.HTTPError) as err:
            post("/api/pause", "wrong")
        assert err.value.code == 403 and not sup.PAUSE_FILE.exists()

        token = sup.dashboard_token()
        assert post("/api/pause", token) == 200 and sup.PAUSE_FILE.exists()
        assert sup.pause_reason() == "paused by you"
        assert post("/api/resume", token) == 200 and not sup.PAUSE_FILE.exists()
    finally:
        server.shutdown()


def test_backup_uploads_only_changed_files_once_a_day(bg, monkeypatch):
    sup.LOG_DIR.mkdir()
    (sup.CKPT_DIR / "best_model.pt").write_text("w1")
    (sup.LOG_DIR / "training_log.json").write_text("[]")
    launched = []

    class FakeProc:
        returncode = 0

        def poll(self):
            return 0

    def fake_popen(cmd, **kwargs):
        launched.append(cmd)
        return FakeProc()

    monkeypatch.setattr(sup.subprocess, "Popen", fake_popen)
    monkeypatch.setitem(sup.BACKUP, "proc", None)
    cfg = dict(sup.DEFAULT_CONFIG, backup_repo="user/backup")
    state = {}

    sup.maybe_backup(cfg, state)
    assert launched and launched[0][6] == "user/backup"
    assert {a.split("=")[1] for a in launched[0][7:]} == {"best_model.pt", "training_log.json"}

    sup.maybe_backup(cfg, state)  # upload finished: mtimes recorded, clock reset
    assert state["last_backup"] and len(launched) == 1

    state["last_backup"] = 0
    sup.maybe_backup(cfg, state)  # nothing changed since the upload
    assert len(launched) == 1

    state["last_backup"] = 0
    (sup.CKPT_DIR / "best_model.pt").write_text("w2")

    os.utime(sup.CKPT_DIR / "best_model.pt", (1, 1))
    sup.maybe_backup(cfg, state)
    assert [a.split("=")[1] for a in launched[1][7:]] == ["best_model.pt"]


def test_backup_disabled_without_repo(bg, monkeypatch):
    monkeypatch.setattr(sup.subprocess, "Popen", lambda *a, **k: pytest.fail("should not upload"))
    monkeypatch.setitem(sup.BACKUP, "proc", None)
    sup.maybe_backup(dict(sup.DEFAULT_CONFIG), {})


def test_memory_pause_hysteresis(bg, monkeypatch):
    monkeypatch.setattr(sup, "MEM", {"low_since": None, "ok_since": None, "paused": False, "free": None})
    monkeypatch.setattr(sup, "game_running", lambda: False)
    sup.update_memory_state(5, 0)
    sup.update_memory_state(5, 20)
    assert not sup.MEM["paused"]  # a brief dip does not pause
    sup.update_memory_state(5, 31)
    assert sup.MEM["paused"] and "low memory" in sup.pause_reason()
    sup.update_memory_state(15, 100)  # recovered a bit, but under the resume threshold
    sup.update_memory_state(30, 200)
    sup.update_memory_state(30, 400)
    assert sup.MEM["paused"]  # needs 5 minutes above 20%
    sup.update_memory_state(30, 501)
    assert not sup.MEM["paused"] and sup.pause_reason() == ""


def test_backend_selects_trainer_and_flags(bg):
    mlx = sup.train_command(dict(sup.DEFAULT_CONFIG, backend="mlx"))
    assert "scripts/train_ghost_base_mlx.py" in mlx and "--device" not in mlx
    assert mlx[mlx.index("--dtype") + 1] == "bfloat16"
    torch_cmd = sup.train_command(dict(sup.DEFAULT_CONFIG, backend="torch"))
    assert "scripts/train_ghost_base.py" in torch_cmd and "--dropout" in torch_cmd
