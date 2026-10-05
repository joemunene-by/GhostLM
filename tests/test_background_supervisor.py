"""Tests for scripts/background/supervisor.py: command building, ETA and the dashboard."""

import importlib.util
import json
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
