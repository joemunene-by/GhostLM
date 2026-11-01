#!/bin/zsh
# Installs (or reinstalls) the launchd agent that keeps the Groq chat SFT generator running.
# A power cut takes a nohup process with it; launchd starts this one again at login and
# restarts it after a crash, but lets a clean exit (job list finished) stand.
# Usage: scripts/background/install_gen.sh [--uninstall]
set -e
ROOT="${0:A:h:h:h}"
LABEL=com.ghostlm.gen
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"

launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
if [[ "$1" == "--uninstall" ]]; then
  rm -f "$PLIST"
  echo "uninstalled $LABEL"
  exit 0
fi

[[ -x "$ROOT/.venv/bin/python" ]] || { echo "missing $ROOT/.venv"; exit 1; }
grep -q '^GROQ_API_KEY=' "$ROOT/.env" 2>/dev/null || { echo "put GROQ_API_KEY=... in $ROOT/.env first"; exit 1; }
mkdir -p "$ROOT/.bg/logs" "$HOME/Library/LaunchAgents"
# The sandbox only lets launchd open a Desktop file it created itself (com.apple.macl), so a
# log left behind by a nohup run makes the spawn fail with EX_CONFIG; move it aside.
LOG="$ROOT/.bg/logs/gen_chat_groq.log"
if [[ -e "$LOG" ]] && ! xattr -p com.apple.macl "$LOG" >/dev/null 2>&1; then
  mv "$LOG" "$LOG.$(date +%Y%m%d%H%M%S).old"
fi

cat > "$PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
	<key>Label</key><string>$LABEL</string>
	<key>ProgramArguments</key>
	<array>
		<string>$ROOT/.venv/bin/python</string>
		<string>$ROOT/scripts/gen_chat_sft.py</string>
		<string>run</string>
		<string>--backend</string><string>groq</string>
		<string>--workers</string><string>1</string>
		<string>--rounds</string><string>4</string>
	</array>
	<key>WorkingDirectory</key><string>$ROOT</string>
	<key>RunAtLoad</key><true/>
	<key>KeepAlive</key>
	<dict>
		<key>SuccessfulExit</key><false/>
	</dict>
	<key>ThrottleInterval</key><integer>60</integer>
	<key>ProcessType</key><string>Standard</string>
	<key>StandardOutPath</key><string>$ROOT/.bg/logs/gen_chat_groq.log</string>
	<key>StandardErrorPath</key><string>$ROOT/.bg/logs/gen_chat_groq.log</string>
	<key>EnvironmentVariables</key>
	<dict>
		<key>PATH</key><string>/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
		<key>PYTHONUNBUFFERED</key><string>1</string>
	</dict>
</dict>
</plist>
PLIST

plutil -lint "$PLIST" >/dev/null
launchctl bootstrap "gui/$(id -u)" "$PLIST"
echo "installed $LABEL; log in $ROOT/.bg/logs/gen_chat_groq.log"
