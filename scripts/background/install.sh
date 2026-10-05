#!/bin/zsh
# Installs (or reinstalls) the launchd agent that runs the training supervisor at login.
# Usage: scripts/background/install.sh [--uninstall]
set -e
ROOT="${0:A:h:h:h}"
LABEL=com.ghostlm.bgtrain
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"

launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
if [[ "$1" == "--uninstall" ]]; then
  rm -f "$PLIST"
  echo "uninstalled $LABEL"
  exit 0
fi

[[ -x "$ROOT/.venv/bin/python" ]] || { echo "missing $ROOT/.venv; run: python3 -m venv .venv && .venv/bin/pip install -e '.[train,data,dev]'"; exit 1; }
mkdir -p "$ROOT/.bg/logs" "$HOME/Library/LaunchAgents"

cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
	<key>Label</key><string>$LABEL</string>
	<key>ProgramArguments</key>
	<array>
		<string>$ROOT/.venv/bin/python</string>
		<string>$ROOT/scripts/background/supervisor.py</string>
	</array>
	<key>WorkingDirectory</key><string>$ROOT</string>
	<key>RunAtLoad</key><true/>
	<key>KeepAlive</key><true/>
	<key>ThrottleInterval</key><integer>60</integer>
	<!-- Background would throttle the trainer's GPU work; the trainer runs under nice instead. -->
	<key>ProcessType</key><string>Standard</string>
	<key>StandardOutPath</key><string>$ROOT/.bg/logs/supervisor.log</string>
	<key>StandardErrorPath</key><string>$ROOT/.bg/logs/supervisor.log</string>
	<key>EnvironmentVariables</key>
	<dict>
		<key>PATH</key><string>/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
		<key>PYTHONUNBUFFERED</key><string>1</string>
	</dict>
</dict>
</plist>
EOF

plutil -lint "$PLIST" >/dev/null
launchctl bootstrap "gui/$(id -u)" "$PLIST"
echo "installed $LABEL; status in $ROOT/.bg/STATUS.txt"
