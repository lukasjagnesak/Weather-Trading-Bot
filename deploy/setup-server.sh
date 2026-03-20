#!/usr/bin/env bash
# Setup script for deploying Weather Trading Bot on a fresh Hetzner Ubuntu server.
# Run as root: bash setup-server.sh
set -euo pipefail

APP_DIR="/opt/weather-bot"
APP_USER="weatherbot"
REPO_URL="https://github.com/lukasjagnesak/Weather-Trading-Bot.git"
BRANCH="claude/polymarket-weather-bot-4wNMw"

echo "=== Weather Trading Bot — Hetzner Deployment ==="

# 1. System packages
echo "[1/7] Installing system packages..."
apt-get update -qq
apt-get install -y -qq python3 python3-venv python3-pip git

# 2. Create service user
echo "[2/7] Creating service user..."
if ! id "$APP_USER" &>/dev/null; then
    useradd --system --shell /usr/sbin/nologin --home-dir "$APP_DIR" "$APP_USER"
fi

# 3. Clone / update repository
echo "[3/7] Setting up application..."
if [ -d "$APP_DIR/.git" ]; then
    echo "  Repository exists, pulling latest..."
    cd "$APP_DIR"
    git fetch origin "$BRANCH"
    git reset --hard "origin/$BRANCH"
else
    git clone --branch "$BRANCH" "$REPO_URL" "$APP_DIR"
fi

# 4. Python virtual environment
echo "[4/7] Setting up Python environment..."
cd "$APP_DIR"
python3 -m venv venv
./venv/bin/pip install --upgrade pip -q
./venv/bin/pip install -e . -q

# 5. Create .env if it doesn't exist
echo "[5/7] Checking .env configuration..."
if [ ! -f "$APP_DIR/.env" ]; then
    cp "$APP_DIR/.env.example" "$APP_DIR/.env"
    echo "  Created .env from template — EDIT IT before starting the bot!"
    echo "  >>> nano $APP_DIR/.env"
fi

# 6. Fix permissions
echo "[6/7] Setting permissions..."
chown -R "$APP_USER:$APP_USER" "$APP_DIR"
chmod 600 "$APP_DIR/.env"

# 7. Install and enable systemd service
echo "[7/7] Installing systemd service..."
cp "$APP_DIR/deploy/weather-bot.service" /etc/systemd/system/
systemctl daemon-reload
systemctl enable weather-bot.service

echo ""
echo "=== Setup complete! ==="
echo ""
echo "Next steps:"
echo "  1. Edit config:    nano $APP_DIR/.env"
echo "  2. Start bot:      systemctl start weather-bot"
echo "  3. Check status:   systemctl status weather-bot"
echo "  4. View logs:      journalctl -u weather-bot -f"
echo ""
echo "Telegram setup:"
echo "  1. Message @BotFather on Telegram, create a bot, get the token"
echo "  2. Get your chat_id: message @userinfobot"
echo "  3. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env"
echo ""
