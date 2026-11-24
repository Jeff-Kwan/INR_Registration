#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
exec /venv/main/bin/python -m medical.sirenpath_ablations \
    --runs 13 11 8 9 10 16 12 19 --gpus 0 1 2 3 \
    --output results/sirenpath_l2r_oasis_supplement "$@"
