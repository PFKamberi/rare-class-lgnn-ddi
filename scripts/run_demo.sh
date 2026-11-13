#!/usr/bin/env bash
# Demo on public data: build the sample graph, then train and evaluate the warm-start
# model and run the diagnostics on it. Needs only --data-root and --work-root (or
# DDI_DATA_ROOT / DDI_WORK_ROOT); the META-DDIE data and ChemBERTa are downloaded.
#   bash scripts/run_demo.sh --data-root ~/ddi_data --work-root ~/ddi_runs
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
echo "=== build_sample_graph";  python "$DIR/build_sample_graph.py" "$@"
echo "=== 02_train_warmstart";  python "$DIR/02_train_warmstart.py" --sample --epochs 30 "$@"
echo "=== 03_diagnostics";      python "$DIR/03_diagnostics.py" --sample "$@"

WORK="${DDI_WORK_ROOT:-}"
prev=""
for a in "$@"; do
    [[ "$prev" == "--work-root" ]] && WORK="$a"
    [[ "$a" == --work-root=* ]] && WORK="${a#--work-root=}"
    prev="$a"
done
WORK="$(cd "$WORK" 2>/dev/null && pwd || echo "$WORK")"
echo "demo finished. Results:"
echo "  $WORK/warmstart/table_7_8_sample.json"
echo "  $WORK/diagnostics/diagnostics_sample.json"
