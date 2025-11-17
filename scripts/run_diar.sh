#!/usr/bin/env zsh
set -euo pipefail

# Quick runner that enables simple speaker diarization (no HuggingFace required)
# 
# Uses turn-taking patterns to detect speaker changes (pauses > 1 second)
# For better accuracy with HuggingFace models, set VOCAB_HF_TOKEN
#
# Usage:
#   ./scripts/run_diar.sh                # process all videos
#   ./scripts/run_diar.sh <basename>     # process only specified file
#
# Optional: For advanced diarization with pyannote models:
#   VOCAB_HF_TOKEN=hf_xxx ./scripts/run_diar.sh

ROOT_DIR="${0:A:h}/.."
cd "$ROOT_DIR"

if [[ $# -ge 1 ]]; then
  export VOCAB_ONLY_BASENAME="$1"
fi

# Diarization ON; simple turn-taking detection; small/int8 for Apple Silicon
export VOCAB_DISABLE_DIARIZATION=0
export VOCAB_WHISPER_MODEL=small
export VOCAB_COMPUTE_TYPE=int8

# Suppress version mismatch warnings
export PYTHONWARNINGS="ignore"

# Optional HF token for advanced models (not required)
if [[ -n "${VOCAB_HF_TOKEN:-}" ]]; then
  echo "Using HuggingFace token for advanced diarization models"
else
  echo "Using simple turn-taking speaker detection (no HF token)"
  echo "Tip: Set VOCAB_HF_TOKEN for better accuracy with pyannote models"
fi

PY="$ROOT_DIR/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "Warning: .venv Python not found; falling back to python3 on PATH" >&2
  PY="python3"
fi

exec "$PY" transcription_pipeline.py
