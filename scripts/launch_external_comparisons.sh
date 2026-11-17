#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
exec /venv/main/bin/python -m medical.comparison_experiments --methods convexadam fireants nir demons niftyreg "$@"
