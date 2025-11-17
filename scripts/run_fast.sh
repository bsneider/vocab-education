#!/usr/bin/env zsh
set -euo pipefail

# Quick runner that uses the working settings for Apple Silicon / CPU
# Usage:
#   ./scripts/run_fast.sh                # process all videos with fast settings
#   ./scripts/run_fast.sh <basename>     # process only the specified basename (no extension)

ROOT_DIR="${0:A:h}/.."
cd "$ROOT_DIR"

# Optional arg: basename to filter
if [[ $# -ge 1 ]]; then
  export VOCAB_ONLY_BASENAME="$1"
fi

# Fast/compatible settings that worked in this repo
export VOCAB_DISABLE_DIARIZATION=1
export VOCAB_WHISPER_MODEL=small
export VOCAB_COMPUTE_TYPE=int8
# Do not auto-restrict to smallest here; caller can export VOCAB_ONLY_SMALLEST=1 if desired

# Suppress version mismatch warnings (models work fine despite version differences)
export PYTHONWARNINGS="ignore"

PY="$ROOT_DIR/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "Warning: .venv Python not found; falling back to python3 on PATH" >&2
  PY="python3"
fi

exec "$PY" transcription_pipeline.py
