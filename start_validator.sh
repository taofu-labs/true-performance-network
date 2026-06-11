#!/usr/bin/env bash
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

: "${TPN_DOTENV_PATH:=$ROOT_DIR/.env}"

cd "$ROOT_DIR"
uv sync
TPN_DOTENV_PATH="$TPN_DOTENV_PATH" uv run --package validator python src/validator/main.py "$@"
