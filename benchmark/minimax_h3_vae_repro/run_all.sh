#!/bin/bash
# Runs inside the container: sequential measurement of the given config ids (default: full order).
set -u
EXP=/workspace/experiment/minimax-h3/2026-09-22-minimax-h3-vae-pr-repro
export PATH=/opt/sglang/bin:/usr/local/cuda/bin:$PATH
cd $EXP
ORDER=${*:-"baseline-lossless fast-path-extra-high-fp8 fast-path-high-fp8 baseline-lossless-2"}
for run in $ORDER; do
  cfg=${run%-2}; cfg=${cfg%-3}
  echo "===== $(date +%T) start $run (config $cfg)"
  python3 scripts/run_repro.py --config configs/$cfg.json --output $EXP/$run 2>&1 | tee -a logs/run_all.log | grep -E "ready|warmup|timed|summary|Error|error|Traceback" 
  echo "===== $(date +%T) end $run rc=${PIPESTATUS[0]}"
done
