#!/bin/zsh
# Installs (or reinstalls) the launchd agent that serves the GhostLM web app on port 8091.
# Usage: scripts/background/install_app.sh [--uninstall]
set -e
ROOT="${0:A:h:h:h}"
LABEL=com.ghostlm.app
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"

launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
if [[ "$1" == "--uninstall" ]]; then
  rm -f "$PLIST"
  echo "uninstalled $LABEL"
  exit 0
fi
mkdir -p "$ROOT/.bg/logs"

cat > "$PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
	<key>Label</key><string>$LABEL</string>
	<key>ProgramArguments</key>
	<array>
		<string>$ROOT/.venv/bin/python</string>
		<string>-m</string>
		<string>ghostlm.app</string>
		<string>--port</string>
		<string>8091</string>
	</array>
	<key>WorkingDirectory</key><string>$ROOT</string>
	<key>RunAtLoad</key><true/>
	<key>KeepAlive</key><true/>
	<key>ThrottleInterval</key><integer>30</integer>
	<key>StandardOutPath</key><string>$ROOT/.bg/logs/app.log</string>
	<key>StandardErrorPath</key><string>$ROOT/.bg/logs/app.log</string>
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
echo "installed $LABEL; open http://localhost:8091"
