#!/usr/bin/env bash
# Run every step in order. All arguments are passed to each script, e.g.
#   bash scripts/run_all.sh --data-root ~/ddi_data --work-root ~/ddi_runs --test
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
for step in 01_prepare_data 02_train_warmstart 03_diagnostics 04_episodic 05_coldstart; do
    echo "=== $step"
    python "$DIR/$step.py" "$@"
done
echo "all steps finished"
