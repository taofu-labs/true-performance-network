#!/usr/bin/env bash
set -e
if ! command -v uv &> /dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
echo "Syncing workspace..."
uv sync
echo "Installing TPN CLI..."
uv tool install src/cli --force --reinstall   # installs as 'tpn'
echo "✅ Done. Try: tpn --help"
