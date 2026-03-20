#!/usr/bin/env bash
# Quick update script — pulls latest code and restarts the bot.
# Run as root: bash /opt/weather-bot/deploy/update.sh
set -euo pipefail

APP_DIR="/opt/weather-bot"
BRANCH="claude/polymarket-weather-bot-4wNMw"

echo "Pulling latest from $BRANCH..."
cd "$APP_DIR"
sudo -u weatherbot git fetch origin "$BRANCH"
sudo -u weatherbot git reset --hard "origin/$BRANCH"

echo "Updating Python packages..."
sudo -u weatherbot ./venv/bin/pip install -e . -q

echo "Restarting service..."
systemctl restart weather-bot

echo "Done! Status:"
systemctl status weather-bot --no-pager -l
