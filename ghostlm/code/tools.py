"""The tools Ghost Code gives the model: read, search, edit, run, and inspect the machine's security."""

from __future__ import annotations

import difflib
import fnmatch
import json
import os
import platform
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

MAX_OUTPUT = 20_000
SKIP_DIRS = {".git", "node_modules", ".venv", "venv", "__pycache__", "dist", "build", ".mypy_cache", ".ruff_cache"}
READ_ONLY = {"read_file", "grep", "glob", "list_dir", "security_check", "secret_scan", "lookup_cve"}
EDITS = {"write_file", "edit_file"}


# Read-only commands per area. Each is safe to run without asking: none of them changes state.
CHECKS = {
    "firewall": {
        "Darwin": ["/usr/libexec/ApplicationFirewall/socketfilterfw --getglobalstate --getstealthmode --getblockall"],
        "Linux": ["ufw status verbose 2>/dev/null || firewall-cmd --state 2>/dev/null || echo 'no ufw or firewalld found'"],
    },
    "disk_encryption": {
        "Darwin": ["fdesetup status"],
        "Linux": ["lsblk -o NAME,TYPE,FSTYPE,MOUNTPOINT | grep -iE 'crypt|NAME' || echo 'no encrypted volumes found'"],
    },
    "system_protection": {
        "Darwin": ["csrutil status", "spctl --status",
                   "defaults read /Library/Preferences/com.apple.loginwindow GuestEnabled 2>/dev/null || echo 'guest: not set'"],
        "Linux": ["sysctl kernel.randomize_va_space kernel.kptr_restrict 2>/dev/null",
                  "(aa-status 2>/dev/null | head -3) || (getenforce 2>/dev/null) || echo 'no AppArmor or SELinux found'"],
    },
    "listening_ports": {
        "Darwin": ["lsof -nP -iTCP -sTCP:LISTEN | awk 'NR==1 || !seen[$1,$9]++ {print $1, $3, $9}'"],
        "Linux": ["ss -tlnpH 2>/dev/null | awk '{print $4, $6}'"],
    },
    "remote_access": {
        "Darwin": ["launchctl print-disabled system 2>/dev/null | grep -iE 'ssh|screensharing|smbd|remote|ftp|telnet' "
                   "|| echo 'could not read service state'",
                   "grep -iE '^(PasswordAuthentication|PermitRootLogin|PubkeyAuthentication)' /etc/ssh/sshd_config "
                   "2>/dev/null || echo 'sshd_config: defaults in use'"],
        "Linux": ["systemctl is-active ssh sshd 2>/dev/null",
                  "grep -iE '^(PasswordAuthentication|PermitRootLogin|PubkeyAuthentication)' /etc/ssh/sshd_config "
                  "2>/dev/null || echo 'sshd_config: defaults in use'"],
    },
    "persistence": {
        "Darwin": ["ls ~/Library/LaunchAgents /Library/LaunchAgents /Library/LaunchDaemons 2>/dev/null",
                   "crontab -l 2>/dev/null || echo 'no crontab'"],
        "Linux": ["systemctl list-unit-files --state=enabled --type=service --no-legend 2>/dev/null | awk '{print $1}'",
                  "crontab -l 2>/dev/null || echo 'no crontab'", "ls /etc/cron.d 2>/dev/null"],
    },
    "accounts": {
        "Darwin": ["dscl . list /Users UniqueID | awk '$2 >= 500'", "dscl . -read /Groups/admin GroupMembership"],
        "Linux": ["awk -F: '$3 == 0 || $3 >= 1000 {print $1, $3, $7}' /etc/passwd", "getent group sudo wheel 2>/dev/null"],
    },
    "ssh_keys": {
        "Darwin": ["ls -la ~/.ssh 2>/dev/null || echo 'no ~/.ssh'",
                   "wc -l ~/.ssh/authorized_keys 2>/dev/null || echo 'no authorized_keys'"],
        "Linux": ["ls -la ~/.ssh 2>/dev/null || echo 'no ~/.ssh'",
                  "wc -l ~/.ssh/authorized_keys 2>/dev/null || echo 'no authorized_keys'"],
    },
    "updates": {
        "Darwin": ["sw_vers", "defaults read /Library/Preferences/com.apple.SoftwareUpdate 2>/dev/null "
                   "| grep -iE 'Automatic|LastSuccessful|RecommendedUpdates' || echo 'no update preferences set'"],
        "Linux": ["(. /etc/os-release && echo $PRETTY_NAME); uname -r",
                  "(apt list --upgradable 2>/dev/null | head -30) || (dnf -q check-update 2>/dev/null | head -30) "
                  "|| echo 'no apt or dnf found'"],
    },
}
SECRET_PATTERNS = [
    ("private key", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA |PGP )?PRIVATE KEY")),
    ("AWS access key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("GitHub token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b")),
    ("API key", re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b")),
    ("Slack token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b")),
    ("hardcoded credential", re.compile(
        r"(?i)(?:password|passwd|secret|api_?key|token)\w*\s*[:=]\s*[\"']([^\"'\s]{6,})[\"']")),
    ("credentials in URL", re.compile(r"[a-z][a-z0-9+.-]*://[^/\s:@\"']+:([^@\s\"']{3,})@")),
]


@dataclass
class Result:
    output: str
    summary: str
    diff: list[str] = field(default_factory=list)
    ok: bool = True


def _spec(name: str, description: str, props: dict, required: list[str]) -> dict:
    return {"type": "function", "function": {"name": name, "description": description, "parameters": {
        "type": "object", "properties": props, "required": required}}}


SPECS = [
    _spec("read_file", "Read a text file. Returns numbered lines.", {
        "path": {"type": "string"},
        "offset": {"type": "integer", "description": "First line to read, 1-based."},
        "limit": {"type": "integer", "description": "Maximum number of lines."}}, ["path"]),
    _spec("write_file", "Create a file or overwrite it with new content.", {
        "path": {"type": "string"}, "content": {"type": "string"}}, ["path", "content"]),
    _spec("edit_file", "Replace an exact string in a file. old_string must match exactly and be unique "
          "unless replace_all is true.", {
        "path": {"type": "string"}, "old_string": {"type": "string"}, "new_string": {"type": "string"},
        "replace_all": {"type": "boolean"}}, ["path", "old_string", "new_string"]),
    _spec("bash", "Run a shell command in the working directory and return its output.", {
        "command": {"type": "string"},
        "timeout": {"type": "integer", "description": "Seconds, default 120."}}, ["command"]),
    _spec("grep", "Search file contents with a regular expression.", {
        "pattern": {"type": "string"},
        "path": {"type": "string", "description": "File or directory, default the working directory."},
        "glob": {"type": "string", "description": "Only search files matching this glob, e.g. *.py"}}, ["pattern"]),
    _spec("glob", "Find files by glob pattern, e.g. **/*.py", {"pattern": {"type": "string"}}, ["pattern"]),
    _spec("list_dir", "List a directory.", {"path": {"type": "string"}}, []),
    _spec("security_check", "Run a read-only security check of this machine and return the raw evidence. "
          "Changes nothing. Areas: " + ", ".join(CHECKS) + ", or all.", {
        "area": {"type": "string", "enum": [*CHECKS, "all"]}}, ["area"]),
    _spec("secret_scan", "Scan files for hardcoded secrets (keys, tokens, passwords, private keys). "
          "Reports file and line with the value masked.", {
        "path": {"type": "string", "description": "File or directory, default the working directory."}}, []),
    _spec("lookup_cve", "Look up a CVE in NVD and check whether it is on CISA's Known Exploited "
          "Vulnerabilities list.", {"cve_id": {"type": "string", "description": "e.g. CVE-2024-3094"}}, ["cve_id"]),
]


def resolve(path: str, cwd: Path) -> Path:
    p = Path(os.path.expanduser(path))
    return (p if p.is_absolute() else cwd / p).resolve()


def inside(path: Path, cwd: Path) -> bool:
    return path == cwd or cwd in path.parents


def _clip(text: str) -> str:
    if len(text) <= MAX_OUTPUT:
        return text
    return text[:MAX_OUTPUT] + f"\n[truncated, {len(text) - MAX_OUTPUT:,} more characters]"


def _walk(root: Path):
    for base, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
        for name in sorted(files):
            yield Path(base) / name


def _diff(before: str, after: str, limit: int = 40) -> tuple[list[str], int, int]:
    lines, hunks = [], 0
    for ln in difflib.unified_diff(before.splitlines(), after.splitlines(), lineterm="", n=2):
        if ln.startswith(("---", "+++")):
            continue
        if ln.startswith("@@"):
            hunks += 1
            if hunks > 1:
                lines.append("⋯")
            continue
        lines.append(ln)
    added = sum(1 for ln in lines if ln.startswith("+"))
    removed = sum(1 for ln in lines if ln.startswith("-"))
    if len(lines) > limit:
        lines = lines[:limit] + [f"… {len(lines) - limit} more diff lines"]
    return lines, added, removed


def _plural(n: int, word: str, plural: str | None = None) -> str:
    return f"{n} {word if n == 1 else plural or word + 's'}"


def read_file(args: dict, cwd: Path) -> Result:
    path = resolve(args["path"], cwd)
    if not path.is_file():
        return Result(f"No such file: {path}", "File not found", ok=False)
    try:
        lines = path.read_text().splitlines()
    except UnicodeDecodeError:
        return Result(f"{path} is not a text file.", "Not a text file", ok=False)
    start = max(1, int(args.get("offset") or 1))
    chunk = lines[start - 1:start - 1 + int(args.get("limit") or 2000)]
    body = "\n".join(f"{start + i:>6}\t{ln}" for i, ln in enumerate(chunk))
    return Result(_clip(body) or "(empty file)", f"Read {_plural(len(chunk), 'line')}")


def write_file(args: dict, cwd: Path) -> Result:
    path = resolve(args["path"], cwd)
    before = path.read_text() if path.is_file() else ""
    existed = path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(args["content"])
    diff, added, removed = _diff(before, args["content"])
    verb = "Updated" if existed else "Created"
    return Result(f"{verb} {path}", f"{verb} {path.name} with {_plural(added, 'addition')} and "
                  f"{_plural(removed, 'removal')}", diff)


def edit_file(args: dict, cwd: Path) -> Result:
    path = resolve(args["path"], cwd)
    if not path.is_file():
        return Result(f"No such file: {path}", "File not found", ok=False)
    before, old, new = path.read_text(), args["old_string"], args["new_string"]
    count = before.count(old)
    if count == 0:
        return Result("old_string was not found in the file. Read the file and match it exactly.",
                      "String not found", ok=False)
    if count > 1 and not args.get("replace_all"):
        return Result(f"old_string matches {count} places. Add more context or set replace_all.",
                      f"{count} matches, expected 1", ok=False)
    after = before.replace(old, new) if args.get("replace_all") else before.replace(old, new, 1)
    path.write_text(after)
    diff, added, removed = _diff(before, after)
    return Result(f"Edited {path}", f"Updated {path.name} with {_plural(added, 'addition')} and "
                  f"{_plural(removed, 'removal')}", diff)


def bash(args: dict, cwd: Path) -> Result:
    timeout = int(args.get("timeout") or 120)
    try:
        proc = subprocess.run(args["command"], shell=True, cwd=cwd, capture_output=True, text=True,
                              timeout=timeout, stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        return Result(f"Command timed out after {timeout}s.", f"Timed out after {timeout}s", ok=False)
    out = (proc.stdout + proc.stderr).strip()
    body = out or "(no output)"
    if proc.returncode:
        body += f"\n[exit code {proc.returncode}]"
    return Result(_clip(body), out or "(no output)", ok=proc.returncode == 0)


def grep(args: dict, cwd: Path) -> Result:
    try:
        rx = re.compile(args["pattern"])
    except re.error as e:
        return Result(f"Bad regular expression: {e}", "Bad pattern", ok=False)
    root = resolve(args.get("path") or ".", cwd)
    files = [root] if root.is_file() else _walk(root)
    hits = []
    for f in files:
        if args.get("glob") and not fnmatch.fnmatch(f.name, args["glob"]):
            continue
        try:
            for n, line in enumerate(f.read_text().splitlines(), 1):
                if rx.search(line):
                    hits.append(f"{os.path.relpath(f, cwd)}:{n}: {line.strip()[:200]}")
        except (UnicodeDecodeError, OSError):
            continue
        if len(hits) >= 200:
            hits.append("[stopped at 200 matches]")
            break
    return Result("\n".join(hits) or "No matches.", f"Found {_plural(len(hits), 'match', 'matches')}")


def glob(args: dict, cwd: Path) -> Result:
    patterns = [part.strip() for part in args["pattern"].split(",") if part.strip()]
    found = sorted({os.path.relpath(p, cwd) for pattern in patterns for p in cwd.glob(pattern)
                    if p.is_file() and not SKIP_DIRS & set(p.relative_to(cwd).parts)})[:300]
    return Result("\n".join(found) or "No files matched.", f"Found {_plural(len(found), 'file')}")


def list_dir(args: dict, cwd: Path) -> Result:
    path = resolve(args.get("path") or ".", cwd)
    if not path.is_dir():
        return Result(f"No such directory: {path}", "Directory not found", ok=False)
    names = sorted(p.name + ("/" if p.is_dir() else "") for p in path.iterdir())
    return Result("\n".join(names) or "(empty)", f"Listed {_plural(len(names), 'entry', 'entries')}")


def security_check(args: dict, cwd: Path) -> Result:
    system, area = platform.system(), args["area"]
    if area != "all" and area not in CHECKS:
        return Result(f"Unknown area. Choose from: {', '.join(CHECKS)}, all", "Unknown area", ok=False)
    sections = []
    for name in (CHECKS if area == "all" else [area]):
        commands = CHECKS[name].get(system)
        if not commands:
            sections.append(f"## {name}\nNot supported on {system}.")
            continue
        parts = [f"## {name}"]
        for command in commands:
            try:
                proc = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=30,
                                      stdin=subprocess.DEVNULL)
                out = (proc.stdout + proc.stderr).strip() or "(no output)"
            except subprocess.TimeoutExpired:
                out = "(timed out)"
            lines = out.splitlines()
            if len(lines) > 60:
                lines = lines[:60] + [f"[{len(lines) - 60} more lines]"]
            parts.append("\n".join(lines))
        sections.append("\n".join(parts))
    count = len(CHECKS) if area == "all" else 1
    return Result(_clip("\n\n".join(sections)), f"Checked {_plural(count, 'area')}")


def mask(value: str) -> str:
    return value[:4] + "…" if len(value) > 8 else "…"


def secret_scan(args: dict, cwd: Path) -> Result:
    root = resolve(args.get("path") or ".", cwd)
    files = [root] if root.is_file() else _walk(root)
    hits = []
    for f in files:
        try:
            if f.stat().st_size > 1_000_000:
                continue
            text = f.read_text()
        except (UnicodeDecodeError, OSError):
            continue
        for n, line in enumerate(text.splitlines(), 1):
            for kind, rx in SECRET_PATTERNS:
                m = rx.search(line)
                if m:
                    hits.append(f"{os.path.relpath(f, cwd)}:{n}: {kind} ({mask(m.group(m.lastindex or 0))})")
                    break
        if len(hits) >= 100:
            hits.append("[stopped at 100 findings]")
            break
    return Result("\n".join(hits) or "No secrets found.", f"Found {_plural(len(hits), 'possible secret')}")


def lookup_cve(args: dict, cwd: Path) -> Result:
    from ghostlm.agent.tools import dispatch

    cve_id = args["cve_id"].strip().upper()
    nvd = dispatch("search_cve_nvd", {"q": cve_id}).response
    kev = dispatch("lookup_cisa_kev", {"cve_id": cve_id}).response
    found = bool(nvd and nvd.get("description"))
    return Result(json.dumps({"nvd": nvd, "cisa_kev": kev}, indent=1)[:6000],
                  f"{cve_id}: {'found' if found else 'not found'}", ok=found)


def precheck(name: str, args: dict, cwd: Path) -> Result | None:
    """Reject an edit that cannot succeed before the user is asked to approve it."""
    if name != "edit_file" or "path" not in args or "old_string" not in args:
        return None
    path = resolve(args["path"], cwd)
    if not path.is_file():
        return Result(f"No such file: {path}", "File not found", ok=False)
    count = path.read_text().count(args["old_string"])
    if count == 0:
        return Result("old_string was not found in the file. Read the file and match it exactly.",
                      "String not found", ok=False)
    if count > 1 and not args.get("replace_all"):
        return Result(f"old_string matches {count} places. Add more context or set replace_all.",
                      f"{count} matches, expected 1", ok=False)
    return None


def preview(name: str, args: dict, cwd: Path) -> str:
    """What an edit would change, for the permission prompt. Touches nothing."""
    path = resolve(args.get("path", ""), cwd)
    before = path.read_text() if path.is_file() else ""
    if name == "write_file":
        after = args.get("content", "")
    else:
        old, new = args.get("old_string", ""), args.get("new_string", "")
        after = before.replace(old, new) if args.get("replace_all") else before.replace(old, new, 1)
    where = os.path.relpath(path, cwd) if inside(path, cwd) else str(path)
    return "\n".join([where, "", *_diff(before, after, limit=24)[0]])


HANDLERS = {"read_file": read_file, "write_file": write_file, "edit_file": edit_file, "bash": bash,
            "grep": grep, "glob": glob, "list_dir": list_dir, "security_check": security_check,
            "secret_scan": secret_scan, "lookup_cve": lookup_cve}
LABELS = {"read_file": "Read", "write_file": "Write", "edit_file": "Edit", "bash": "Bash", "grep": "Search",
          "glob": "Glob", "list_dir": "List", "security_check": "Check", "secret_scan": "SecretScan",
          "lookup_cve": "CVE"}


def label(name: str, args: dict, cwd: Path) -> str:
    """How a call reads in the transcript, e.g. Read(src/app.py)."""
    if name == "bash":
        detail = args.get("command", "")
    elif name in ("grep", "glob"):
        detail = args.get("pattern", "")
    elif name == "security_check":
        detail = args.get("area", "")
    elif name == "lookup_cve":
        detail = args.get("cve_id", "")
    else:
        raw = args.get("path") or "."
        full = resolve(raw, cwd)
        detail = os.path.relpath(full, cwd) if inside(full, cwd) else str(full)
    return f"{LABELS.get(name, name)}({detail})"


def run(name: str, args: dict, cwd: Path) -> Result:
    handler = HANDLERS.get(name)
    if handler is None:
        return Result(f"Unknown tool: {name}", "Unknown tool", ok=False)
    try:
        return handler(args, cwd)
    except KeyError as e:
        return Result(f"Missing required argument: {e.args[0]}", "Missing argument", ok=False)
    except OSError as e:
        return Result(f"{type(e).__name__}: {e}", type(e).__name__, ok=False)
