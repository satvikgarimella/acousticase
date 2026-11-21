#!/usr/bin/env bash
set -euo pipefail
ACOUSTICASE_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ACOUSTICASE_ROOT"
if [[ -x .venv/bin/python ]]; then
  ACOUSTICASE_PYTHON=.venv/bin/python
else
  echo 'Create .venv and install the project first; see README.md.' >&2
  exit 1
fi
export PYTHONPATH="$ACOUSTICASE_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="$ACOUSTICASE_ROOT/data/matplotlib"
export TF_CPP_MIN_LOG_LEVEL=2
if [[ -f "$ACOUSTICASE_ROOT/models/yamnet/yamnet.h5" || -f "$ACOUSTICASE_ROOT/models/yamnet/saved_model.pb" ]]; then
  export ACOUSTICASE_YAMNET_DIR="${ACOUSTICASE_YAMNET_DIR:-$ACOUSTICASE_ROOT/models/yamnet}"
fi
exec "$ACOUSTICASE_PYTHON" -m uvicorn acousticase.api:app --host 127.0.0.1 --port "${ACOUSTICASE_PORT:-8000}"
