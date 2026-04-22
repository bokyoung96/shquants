#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
STATE_DIR="$ROOT/data/state"
PYTHON_BIN="${PYTHON_BIN:-../.venv/bin/python}"
TELEGRAM_LOG="$STATE_DIR/telegram.log"
GMAIL_LOG="$STATE_DIR/gmail.log"
TELEGRAM_UNTIL="${TELEGRAM_UNTIL:-2099-12-31T23:59:59+09:00}"

mkdir -p "$STATE_DIR"

usage() {
  cat <<'EOF'
Usage:
  ./run.sh telegram   Start the Telegram realtime watcher
  ./run.sh gmail      Run Gmail sync once and summarize latest
  ./run.sh help       Show this help

Logs:
  data/state/telegram.log
  data/state/gmail.log
EOF
}

channel() {
  PYTHONPATH=src "$PYTHON_BIN" - <<'PY'
from pathlib import Path
from analysts.config import build_config

config = build_config(Path("."))
print(config.telethon.channel if config.telethon else "")
PY
}

run_telegram() {
  local name
  name="$(channel)"
  if [ -z "$name" ]; then
    echo "Missing Telegram channel in config.local.json" >&2
    exit 1
  fi

  echo "== Telegram watcher =="
  echo "channel: $name"
  echo "log: $TELEGRAM_LOG"
  echo "stop: Ctrl+C"
  exec env PYTHONPATH=src "$PYTHON_BIN" -m analysts.cli watch-until \
    --base-dir . \
    --channel "$name" \
    --until "$TELEGRAM_UNTIL" \
    2>>"$TELEGRAM_LOG"
}

run_gmail() {
  {
    echo "==== gmail run started $(date '+%Y-%m-%d %H:%M:%S %z') ===="
    echo
    echo "== Gmail sync =="
    PYTHONPATH=src "$PYTHON_BIN" -m analysts.cli gmail-sync-once --base-dir . --limit 20
    echo
    echo "== Gmail summarize latest =="
    PYTHONPATH=src "$PYTHON_BIN" -m analysts.cli gmail-summarize-latest --base-dir .
    echo
    echo "==== gmail run finished $(date '+%Y-%m-%d %H:%M:%S %z') ===="
  } 2>&1 | tee -a "$GMAIL_LOG"
}

cd "$ROOT"

case "${1:-}" in
  "")
    usage
    ;;
  help|-h|--help)
    usage
    ;;
  telegram)
    run_telegram
    ;;
  gmail)
    run_gmail
    ;;
  *)
    usage >&2
    exit 1
    ;;
esac
