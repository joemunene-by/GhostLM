#!/usr/bin/env python3
"""Supervisor for a long-running ghost-base pretrain on a Mac, run by launchd.

Starts training once the pretokenized corpus exists, stops it cleanly
(SIGTERM, which makes the trainer checkpoint and exit) while a game is running
or .bg/PAUSE exists, resumes from the newest checkpoint afterwards, pauses
every ``eval_every_steps`` to score the latest checkpoint, keeps weights-only
snapshots through the WSD decay phase and averages them at the end. Serves a
status dashboard on ``dashboard_port``. Runtime state lives in ``.bg/``.

See docs/mac_background_training.md.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import shutil
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BG = ROOT / ".bg"
PY = ROOT / ".venv/bin/python"
sys.path.insert(0, str(Path(__file__).resolve().parent))
import phases  # noqa: E402

RUN = phases.PRETRAIN_RUN
CKPT_DIR = ROOT / "checkpoints" / RUN
LOG_DIR = ROOT / "logs" / RUN
TRAIN_LOG = BG / "logs" / "train.log"
STATUS = BG / "STATUS.txt"
STATE = BG / "state.json"
SNAP_DIR = BG / "snapshots"
PAUSE_FILE = BG / "PAUSE"
CONFIG = BG / "train_config.json"
TOKEN_FILE = BG / "dashboard_token"
DASHBOARD_HTML = Path(__file__).with_name("dashboard.html")

MANIFEST = ROOT / "data/processed/train.domains.json"
TRAIN_BIN = ROOT / "data/processed/train.bin"
VAL_BIN = ROOT / "data/processed/val.bin"

POLL_S = 60
RESUME_COOLDOWN_S = 5 * 60
STOP_TIMEOUT_S = 15 * 60
KEEP_CHECKPOINTS = 1

# wine/cellar games show up as wineserver or *.exe processes.
GAME_PATTERN = re.compile(r"wineserver|wine64-preloader|wine-preloader|\.exe(\s|$)", re.I)

DEFAULT_CONFIG = {
    "max_steps": 15_000,
    # GhostTrainer splits batch_size sequences into grad_accum_steps micro-batches,
    # so this is 64 x 1024 = 65,536 tokens per step in micro-batches of 2.
    "batch_size": 64,
    "grad_accum_steps": 32,
    "context_length": 1024,
    "eval_interval": 75,
    "save_interval": 75,
    # "mlx" trains about 2x faster than "torch" (PyTorch MPS) on an M4 at a third of the memory.
    "backend": "mlx",
    "dtype": "float32",
    "dropout": 0.0,
    "grad_checkpoint": True,
    "optimizer": "adamw",
    "lr_schedule": "wsd",
    "learning_rate": None,
    "eval_every_steps": 1500,
    "eval_limit_per_bench": 200,
    "dashboard_port": 8090,
    # Private Hugging Face repo (e.g. "user/ghost-base-backup") that receives the
    # weights-only best model and other small artifacts; None disables backups.
    "backup_repo": None,
    "backup_every_hours": 24,
}

LIVE: dict = {"state": "starting"}
BACKUP: dict = {"proc": None}

UPLOAD_SNIPPET = (
    "import sys; from huggingface_hub import HfApi; api = HfApi(); repo = sys.argv[1]\n"
    "for pair in sys.argv[2:]:\n"
    "    local, remote = pair.split('=', 1)\n"
    "    api.upload_file(path_or_fileobj=local, path_in_repo=remote, repo_id=repo,\n"
    "                    commit_message=f'backup {remote}')\n"
)


def set_run(run: str) -> None:
    """Point checkpoint/log lookups at the current phase's run."""
    global RUN, CKPT_DIR, LOG_DIR
    RUN, CKPT_DIR, LOG_DIR = run, ROOT / "checkpoints" / run, ROOT / "logs" / run
    CKPT_DIR.mkdir(parents=True, exist_ok=True)


def current_phase(cfg: dict, state: dict) -> dict:
    plan = phases.plan(cfg)
    return plan[min(state.get("phase", 0), len(plan) - 1)]


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def load_config() -> dict:
    cfg = dict(DEFAULT_CONFIG)
    if CONFIG.exists():
        cfg.update(json.loads(CONFIG.read_text()))
    return cfg


def load_state() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {"crashes": 0, "finished": False, "last_daily_note": ""}


def save_state(state: dict) -> None:
    STATE.write_text(json.dumps(state, indent=2))


def notify(msg: str) -> None:
    subprocess.run(
        ["osascript", "-e", f'display notification "{msg}" with title "GhostLM training"'],
        capture_output=True,
    )


def game_running() -> bool:
    out = subprocess.run(["ps", "-axo", "command"], capture_output=True, text=True).stdout
    return any(GAME_PATTERN.search(line) for line in out.splitlines())


MEM = {"low_since": None, "ok_since": None, "paused": False, "free": None}
# Training alone sits around 15-20% free on a 16GB Mac; pause only on a real squeeze.
MEM_LOW, MEM_OK, MEM_LOW_S, MEM_OK_S = 10, 20, 30, 300
WAKE = threading.Event()


def free_memory_percent() -> int | None:
    out = subprocess.run(["memory_pressure"], capture_output=True, text=True).stdout
    m = re.search(r"free percentage:\s*(\d+)%", out)
    return int(m.group(1)) if m else None


def update_memory_state(free: int | None, now_s: float) -> None:
    """Pause after free memory stays under MEM_LOW for MEM_LOW_S; lift after MEM_OK for MEM_OK_S."""
    MEM["free"] = free
    if free is None:
        return
    if free < MEM_LOW:
        MEM["low_since"] = now_s if MEM["low_since"] is None else MEM["low_since"]
    else:
        MEM["low_since"] = None
    if free >= MEM_OK:
        MEM["ok_since"] = now_s if MEM["ok_since"] is None else MEM["ok_since"]
    else:
        MEM["ok_since"] = None
    if not MEM["paused"] and MEM["low_since"] is not None and now_s - MEM["low_since"] >= MEM_LOW_S:
        MEM["paused"] = True
        WAKE.set()
    elif MEM["paused"] and MEM["ok_since"] is not None and now_s - MEM["ok_since"] >= MEM_OK_S:
        MEM["paused"] = False


def watch_memory() -> None:
    while True:
        update_memory_state(free_memory_percent(), time.time())
        time.sleep(5)


def pause_reason() -> str:
    if PAUSE_FILE.exists():
        return "paused by you"
    if game_running():
        return "paused while a game is running"
    if MEM["paused"]:
        return f"paused: low memory ({MEM['free']}% free)"
    return ""


def checkpoints() -> list:
    found = []
    for p in CKPT_DIR.glob("checkpoint_step_*.pt"):
        m = re.search(r"checkpoint_step_(\d+)\.pt$", p.name)
        if m:
            found.append((int(m.group(1)), p))
    return [p for _, p in sorted(found, reverse=True)]


def latest_step() -> int:
    ckpts = checkpoints()
    return int(re.search(r"(\d+)", ckpts[0].name).group(1)) if ckpts else 0


def log_event(msg: str) -> None:
    print(f"[{now()}] {msg}", flush=True)


def live_step() -> int:
    """Newest step the trainer has printed (MLX "step N ..." or tqdm "N/M ["), else 0."""
    try:
        with TRAIN_LOG.open("rb") as f:
            f.seek(max(0, TRAIN_LOG.stat().st_size - 8192))
            tail = f.read().decode(errors="ignore")
    except OSError:
        return 0
    found = re.findall(r"^step (\d+) ", tail, re.M) or re.findall(r"(\d+)/\d+ \[", tail)
    return int(found[-1]) if found else 0


def free_disk_gb() -> float:
    return shutil.disk_usage(ROOT).free / 1e9


def prune() -> None:
    for p in checkpoints()[KEEP_CHECKPOINTS:]:
        p.unlink(missing_ok=True)
    for p in CKPT_DIR.glob("*.tmp"):
        if time.time() - p.stat().st_mtime > 3600:
            p.unlink(missing_ok=True)


def training_log() -> list:
    try:
        return [e for e in json.loads((LOG_DIR / "training_log.json").read_text())
                if "val_loss" in e and e["val_loss"] != float("inf")]
    except (OSError, ValueError):
        return []


def train_command(cfg: dict, phase: dict | None = None) -> list:
    phase = phase or phases.plan(cfg)[0]
    mlx = cfg.get("backend") == "mlx"
    script = "scripts/train_ghost_base_mlx.py" if mlx else "scripts/train_ghost_base.py"
    cmd = [
        "nice", "-n", "10", str(PY), script,
        "--run-name", RUN,
        "--train-data", str(TRAIN_BIN), "--val-data", str(VAL_BIN),
        "--max-steps", str(cfg["max_steps"]),
        "--batch-size", str(cfg["batch_size"]),
        "--grad-accum-steps", str(cfg["grad_accum_steps"]),
        "--context-length", str(cfg["context_length"]),
        "--eval-interval", str(cfg["eval_interval"]),
        "--save-interval", str(cfg["save_interval"]),
        "--best-weights-only",
        "--optimizer", cfg["optimizer"],
        "--lr-schedule", cfg["lr_schedule"],
    ]
    if mlx:
        cmd += ["--dtype", "bfloat16" if cfg["dtype"] == "float32" else cfg["dtype"]]
    else:
        cmd += ["--device", "mps", "--dtype", cfg["dtype"], "--dropout", str(cfg["dropout"])]
    if cfg.get("learning_rate"):
        cmd += ["--learning-rate", str(cfg["learning_rate"])]
    if MANIFEST.exists():
        cmd += ["--curriculum-manifest", str(MANIFEST)]
    for flag in ("grad_checkpoint", "cautious_wd", "attn_gate", "value_residual"):
        if cfg.get(flag):
            cmd.append("--" + flag.replace("_", "-"))
    cmd += phases.phase_overrides(cfg, phase, ROOT)
    latest = checkpoints()
    start = latest[0] if latest else phases.initial_resume(phase, ROOT)
    if start:
        cmd += ["--resume", str(start)]
    return cmd


def seconds_per_step(state: dict) -> float | None:
    samples = state.get("rate_samples", [])
    if len(samples) < 2:
        return None
    (t0, s0), (t1, s1) = samples[0], samples[-1]
    return (t1 - t0) / (s1 - s0) if s1 > s0 else None


def record_rate(state: dict, training: bool) -> None:
    """Track (time, step) while training so the ETA reflects real wall-clock speed."""
    if not training:
        state["rate_samples"] = []
        return
    step = max(live_step(), latest_step())
    samples = state.setdefault("rate_samples", [])
    if not samples or samples[-1][1] != step:
        samples.append([time.time(), step])
    del samples[:-20]


def refresh_status(state_line: str, cfg: dict, state: dict) -> None:
    log = training_log()
    last = log[-1] if log else {}
    step = max(last.get("step", 0), latest_step(), live_step())
    sps = seconds_per_step(state)
    eta_h = max(0, cfg["max_steps"] - step) * sps / 3600 if sps else None
    evals = []
    if (BG / "evals.jsonl").exists():
        evals = [json.loads(line) for line in (BG / "evals.jsonl").read_text().splitlines() if line.strip()]
    LIVE.clear()
    LIVE.update({
        "state": state_line, "updated": now(), "step": step, "max_steps": cfg["max_steps"],
        "val_loss": last.get("val_loss"), "train_loss": last.get("train_loss"),
        "seconds_per_step": sps, "eta_hours": eta_h,
        "tokens_per_step": cfg["batch_size"] * cfg["context_length"],
        "paused": PAUSE_FILE.exists(), "finished": state.get("finished", False),
        "free_memory": MEM["free"],
        "free_disk_gb": round(free_disk_gb(), 1),
        "optimizer": cfg["optimizer"], "lr_schedule": cfg["lr_schedule"], "run": RUN,
        "val_curve": [[e["step"], e["val_loss"]] for e in log][-400:],
        "evals": [{"step": e["step"], "results": {k: round(v["acc"], 1) for k, v in e["results"].items()}}
                  for e in evals],
    })
    lines = [
        f"GhostLM ghost-base background pretrain ({now()})",
        f"state:     {state_line}",
        f"run:       {RUN}",
        f"progress:  step {step:,} / {cfg['max_steps']:,} ({100 * step / cfg['max_steps']:.1f}%)",
    ]
    if last:
        lines.append(f"val_loss:  {last['val_loss']:.4f} (train {last.get('train_loss', 0):.4f})")
    if eta_h is not None:
        lines.append(f"eta:       {eta_h / 24:.1f} days of training at the current pace")
    lines += [
        "",
        f"dashboard: http://<this mac>:{cfg['dashboard_port']}",
        "pause:  touch .bg/PAUSE   resume: rm .bg/PAUSE",
        "logs:   .bg/logs/train.log",
    ]
    STATUS.write_text("\n".join(lines) + "\n")


def eval_due(cfg: dict, state: dict) -> bool:
    return latest_step() >= state.get("last_eval_step", 0) + cfg["eval_every_steps"]


def start_eval(cfg: dict) -> subprocess.Popen:
    ckpt = checkpoints()[0]
    step = latest_step()
    log = open(BG / "logs" / "eval.log", "a")
    log.write(f"\n===== {now()} eval step {step} =====\n")
    log.flush()
    cmd = (
        f"nice -n 10 {PY} scripts/scorecard.py --checkpoint {ckpt} --label {RUN}"
        f" --device mps --limit-per-bench {cfg['eval_limit_per_bench']} --n-permutations 2"
        f" --out .bg/evals/step_{step}.md --json-out .bg/evals.jsonl"
        f" && {PY} scripts/plot_eval_progress.py --evals .bg/evals.jsonl"
        f" --train-log {LOG_DIR / 'training_log.json'} --out .bg/progress"
    )
    return subprocess.Popen(["/bin/zsh", "-c", cmd], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)


def snapshot_tail(cfg: dict) -> None:
    """Keep bf16 weights-only snapshots across the WSD decay phase for checkpoint averaging."""
    step = latest_step()
    if step < int(cfg["max_steps"] * 0.8) or step % 500 >= cfg["save_interval"]:
        return
    out = SNAP_DIR / f"step_{step - step % 500}.pt"
    if out.exists():
        return
    SNAP_DIR.mkdir(exist_ok=True)
    subprocess.run([str(PY), "-c", (
        "import sys, torch; c = torch.load(sys.argv[1], map_location='cpu', weights_only=False); "
        "torch.save({'step': c['step'], 'config': c['config'], "
        "'model_state_dict': {k: v.to(torch.bfloat16) if v.is_floating_point() else v "
        "for k, v in c['model_state_dict'].items()}}, sys.argv[2])"
    ), str(checkpoints()[0]), str(out)], cwd=ROOT)


def average_tail() -> None:
    snaps = sorted(SNAP_DIR.glob("step_*.pt"), key=lambda p: int(re.search(r"(\d+)", p.stem).group(1)))
    if len(snaps) < 2:
        return
    avg = CKPT_DIR / "averaged.pt"
    subprocess.run([str(PY), "scripts/average_checkpoints.py", *map(str, snaps), "--out", str(avg)], cwd=ROOT)
    with open(BG / "logs" / "eval.log", "a") as log:
        subprocess.run([str(PY), "scripts/scorecard.py", "--checkpoint", str(avg), "--label", f"{RUN}_averaged",
                        "--device", "mps", "--out", str(BG / "evals" / "averaged.md")],
                       cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)


def backup_files(state: dict) -> list:
    """Artifacts that changed since their last upload, as local=remote pairs."""
    candidates = {
        CKPT_DIR / "best_model.pt": "best_model.pt",
        CKPT_DIR / "pre_decay.pt": "pre_decay.pt",
        CKPT_DIR / "averaged.pt": "averaged.pt",
        LOG_DIR / "training_log.json": "training_log.json",
        BG / "evals.jsonl": "evals.jsonl",
        CONFIG: "train_config.json",
    }
    uploaded = state.setdefault("backed_up_mtimes", {})
    pairs = []
    for local, remote in candidates.items():
        if local.exists() and uploaded.get(remote) != local.stat().st_mtime:
            pairs.append((local, remote))
    return pairs


def maybe_backup(cfg: dict, state: dict) -> None:
    proc = BACKUP["proc"]
    if proc is not None:
        if proc.poll() is None:
            return
        if proc.returncode == 0:
            state["backed_up_mtimes"].update(BACKUP["pending"])
            state["last_backup"] = time.time()
        BACKUP["proc"] = None
    if not cfg.get("backup_repo"):
        return
    if time.time() - state.get("last_backup", 0) < cfg["backup_every_hours"] * 3600:
        return
    pairs = backup_files(state)
    if not pairs:
        state["last_backup"] = time.time()
        return
    BACKUP["pending"] = {remote: local.stat().st_mtime for local, remote in pairs}
    log = open(BG / "logs" / "backup.log", "a")
    log.write(f"\n===== {now()} uploading {', '.join(r for _, r in pairs)} =====\n")
    log.flush()
    BACKUP["proc"] = subprocess.Popen(
        ["nice", "-n", "15", str(PY), "-c", UPLOAD_SNIPPET, cfg["backup_repo"],
         *[f"{local}={remote}" for local, remote in pairs]],
        cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
    )


def advance_phase(cfg: dict, state: dict) -> None:
    """Move to the next phase, or mark the whole plan finished."""
    plan = phases.plan(cfg)
    nxt = state.get("phase", 0) + 1
    if nxt >= len(plan):
        state["finished"] = True
        notify("All training phases finished.")
        return
    state["phase"] = nxt
    state["crashes"] = 0
    state["rate_samples"] = []
    state["last_eval_step"] = (phases.pretrain_decay_start(load_config())
                               if plan[nxt]["name"].startswith("branch_") else 0)
    log_event(f"starting phase {plan[nxt]['name']}")


def corpus_ready() -> bool:
    return TRAIN_BIN.exists() and VAL_BIN.exists()


def dashboard_token() -> str:
    if not TOKEN_FILE.exists():
        TOKEN_FILE.write_text(secrets.token_urlsafe(16))
        TOKEN_FILE.chmod(0o600)
    return TOKEN_FILE.read_text().strip()


class DashboardHandler(BaseHTTPRequestHandler):
    """Read-only status for anyone who can reach the port; pause/resume need the token."""

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, DASHBOARD_HTML.read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/api/status":
            self._send(200, json.dumps(LIVE).encode(), "application/json")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if not secrets.compare_digest(self.headers.get("X-Token", ""), dashboard_token()):
            self._send(403, b'{"error": "bad token"}', "application/json")
            return
        if self.path == "/api/pause":
            PAUSE_FILE.touch()
        elif self.path == "/api/resume":
            PAUSE_FILE.unlink(missing_ok=True)
        else:
            self._send(404, b"not found", "text/plain")
            return
        self._send(200, b'{"ok": true}', "application/json")

    def log_message(self, *args):
        pass


def start_dashboard(port: int) -> None:
    dashboard_token()
    try:
        server = ThreadingHTTPServer(("0.0.0.0", port), DashboardHandler)
    except OSError as e:
        print(f"[warn] dashboard disabled: {e}", flush=True)
        return
    threading.Thread(target=server.serve_forever, daemon=True).start()


def main() -> None:
    os.chdir(ROOT)
    (BG / "logs").mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    state = load_state()
    start_dashboard(load_config()["dashboard_port"])
    threading.Thread(target=watch_memory, daemon=True).start()
    child = None
    eval_child = None
    stopping_since = None
    clear_since = None

    def shutdown(signum, frame):
        if child and child.poll() is None:
            child.send_signal(signal.SIGTERM)
            try:
                child.wait(timeout=STOP_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                child.kill()
        sys.exit(0)

    signal.signal(signal.SIGTERM, shutdown)

    while True:
        cfg = load_config()
        phase = current_phase(cfg, state)
        set_run(phase["run"])
        cfg["pretrain_max_steps"] = cfg["max_steps"]
        cfg["max_steps"] = phases.phase_max_steps(cfg, phase)

        if phase["kind"] == "merge" and not state["finished"]:
            refresh_status("merging branches into the MoE", cfg, state)
            with open(BG / "logs" / "merge.log", "a") as log:
                code = subprocess.run(phases.merge_command(cfg, ROOT, PY), cwd=ROOT,
                                      stdout=log, stderr=subprocess.STDOUT).returncode
            log_event(f"merge exited with code {code}")
            if code == 0:
                advance_phase(cfg, state)
            else:
                notify("BTX merge failed; see .bg/logs/merge.log")
                state["finished"] = True
            save_state(state)
            continue

        if state["finished"]:
            maybe_backup(cfg, state)
            save_state(state)
            refresh_status("finished", cfg, state)
            time.sleep(POLL_S * 10)
            continue

        if not corpus_ready():
            refresh_status("waiting for the corpus", cfg, state)
            time.sleep(POLL_S)
            continue

        if not state.get("post_prep_started"):
            # CPU-only data prep for fine-tuning, so it can run alongside training.
            subprocess.Popen(["/bin/zsh", str(ROOT / "scripts/background/prepare_post_training.sh")], cwd=ROOT)
            state["post_prep_started"] = now()
            save_state(state)

        reason = pause_reason()
        if reason != state.get("last_reason", ""):
            log_event(f"pause: {reason}" if reason else "pause lifted")
            state["last_reason"] = reason
        if free_disk_gb() < 8 and time.time() - state.get("disk_warned", 0) > 6 * 3600:
            log_event(f"low disk: {free_disk_gb():.1f}GB free")
            notify(f"Only {free_disk_gb():.1f}GB disk free; checkpoints need about 4GB. Free some space.")
            state["disk_warned"] = time.time()

        if eval_child is not None:
            if eval_child.poll() is None:
                if reason:
                    eval_child.terminate()
                else:
                    refresh_status(f"evaluating step {latest_step():,}", cfg, state)
                    time.sleep(POLL_S)
                    continue
            else:
                if eval_child.returncode == 0:
                    state["last_eval_step"] = latest_step()
                    state["eval_pending"] = False
                    # Resume straight away; the cooldown is only for game pauses.
                    clear_since = time.time() - RESUME_COOLDOWN_S
                save_state(state)
            eval_child = None

        if child is not None and child.poll() is not None:
            code = child.returncode
            log_event(f"trainer exited with code {code} at step {max(latest_step(), live_step())}")
            child = None
            stopping_since = None
            step = max(latest_step(), training_log()[-1]["step"] if training_log() else 0)
            if step > state.get("launch_step", 0):
                state["crashes"] = 0
            if code == 0 and not reason and step >= cfg["max_steps"]:
                refresh_status(f"{phase['name']} finished, wrapping up", cfg, state)
                if phase["name"] == "pretrain":
                    average_tail()
                    phases.finalize_run(CKPT_DIR, keep=("best_model.pt", "pre_decay.pt", "averaged.pt"))
                else:
                    phases.finalize_run(CKPT_DIR, keep=("best_model.pt",) if phase["name"] == "router" else ())
                log_event(f"phase {phase['name']} finished at step {step}")
                notify(f"{phase['name']} finished at step {step:,}.")
                advance_phase(cfg, state)
            elif code != 0 and not reason:
                # Restart the cooldown clock so the crash backoff below is actually waited out.
                clear_since = time.time()
                state["crashes"] += 1
                if state["crashes"] in (3, 10):
                    notify(f"Training crashed {state['crashes']} times. Check .bg/logs/train.log")
            save_state(state)

        if child is not None:
            if reason and stopping_since is None:
                child.send_signal(signal.SIGTERM)
                stopping_since = time.time()
            elif not reason and stopping_since is None and eval_due(cfg, state):
                state["eval_pending"] = True
                save_state(state)
                child.send_signal(signal.SIGTERM)
                stopping_since = time.time()
            elif stopping_since and time.time() - stopping_since > STOP_TIMEOUT_S:
                child.kill()
            record_rate(state, stopping_since is None)
            refresh_status("stopping: " + reason if reason else
                           ("stopping for eval" if state.get("eval_pending") else "training"), cfg, state)
        else:
            record_rate(state, False)
            if not reason and state.get("eval_pending") and checkpoints():
                eval_child = start_eval(cfg)
                refresh_status(f"evaluating step {latest_step():,}", cfg, state)
                time.sleep(POLL_S)
                continue
            if reason:
                clear_since = None
                refresh_status(reason, cfg, state)
            else:
                clear_since = clear_since or time.time()
                backoff = min(3600, 600 * state["crashes"])
                wait = max(RESUME_COOLDOWN_S if checkpoints() else 0, backoff)
                if time.time() - clear_since >= wait:
                    prune()
                    with open(TRAIN_LOG, "a") as log:
                        log.write(f"\n===== {now()} launching =====\n")
                        log.flush()
                        child = subprocess.Popen(
                            train_command(cfg, phase), cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                        )
                    # caffeinate does not forward signals to a wrapped command, so it
                    # watches the trainer's pid instead of wrapping it.
                    subprocess.Popen(["caffeinate", "-i", "-w", str(child.pid)])
                    log_event(f"launched trainer (pid {child.pid}, resume {checkpoints()[0].name if checkpoints() else 'none'})")
                    state["launch_step"] = max(latest_step(), live_step())
                    refresh_status("training", cfg, state)
                else:
                    refresh_status(f"resuming in {int(wait - (time.time() - clear_since))}s", cfg, state)

        today = datetime.now().strftime("%Y-%m-%d")
        if child is not None and datetime.now().hour >= 20 and state["last_daily_note"] != today:
            if LIVE.get("val_loss") is not None:
                notify(f"Step {LIVE['step']:,}/{cfg['max_steps']:,}, val_loss {LIVE['val_loss']:.3f}")
            state["last_daily_note"] = today
            save_state(state)

        if child is not None and child.poll() is None:
            if phase["name"] == "pretrain":
                snapshot_tail(cfg)
            prune()
        maybe_backup(cfg, state)
        save_state(state)
        # Woken early by the memory watcher so a squeeze is acted on within seconds.
        WAKE.wait(POLL_S)
        WAKE.clear()


if __name__ == "__main__":
    main()
