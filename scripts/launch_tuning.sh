#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
exec /venv/main/bin/python -B -m medical.tuning_experiments "$@"
