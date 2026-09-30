#!/usr/bin/env bash
# Wrapper do run_hpo_reseed.py, para nohup. Ver a docstring do .py.
set -u
cd "$(dirname "$0")/.."
mkdir -p evidence/v7_reseed
LOG="evidence/v7_reseed/reseed_$(date +%Y%m%d_%H%M%S).log"
echo "log: $LOG"
exec "${JAUNDICE_PYTHON:-python}" scripts/run_hpo_reseed.py "$@" > "$LOG" 2>&1
