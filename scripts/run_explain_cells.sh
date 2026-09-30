#!/usr/bin/env bash
# Doze células com --save-explain, em série, com registo datado.
# Uso:  nohup bash scripts/run_explain_cells.sh > /dev/null 2>&1 &
set -u
cd "$(dirname "$0")/.."
LOG="local/experiments/explain/run_$(date +%Y%m%d_%H%M%S).log"
mkdir -p local/experiments/explain
exec > >(tee -a "$LOG") 2>&1
echo "início: $(date -Is) | host: $(hostname) | GPU: ${CUDA_VISIBLE_DEVICES:-0}"
"${JAUNDICE_PYTHON:-python}" scripts/run_explain_cells.py --gpu "${GPU:-0}" "$@"
echo "fim: $(date -Is)"
