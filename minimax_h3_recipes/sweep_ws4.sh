#!/usr/bin/env bash
# The 4-GPU comparison, on an idle machine: A/B in both orders, then an
# nsys capture with fast-ulysses on to sit beside the baseline capture.
#
# GPU utilisation is sampled around every run. This box is shared, and a
# foreign job at 80-95% util flattens both arms and hides a 3% signal -- so a
# number is only worth quoting alongside what else was running.
set -u

REPO=/workspace/sglang-worktrees/fast-ulysses-a2a
CFG=minimax_h3_recipes/t2va_5s.json
cd "${REPO}"

stamp() { date +%H:%M:%S; }
gpus()  { nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader | tr '\n' '|'; }

echo "[$(stamp)] GPU before: $(gpus)"

echo "[$(stamp)] === round 1: off then on ==="
H3_GPU_IDS=4,5,6,7 H3_AB_ORDER="off on" ./minimax_h3_recipes/run_ab.sh "${CFG}"
echo "[$(stamp)] GPU after round 1: $(gpus)"

echo "[$(stamp)] === round 2: on then off ==="
H3_GPU_IDS=4,5,6,7 H3_AB_ORDER="on off" ./minimax_h3_recipes/run_ab.sh "${CFG}"
echo "[$(stamp)] GPU after round 2: $(gpus)"

echo "[$(stamp)] === nsys capture with fast-ulysses ON ==="
SGLANG_DIFFUSION_FAST_ULYSSES=1 H3_GPU_IDS=4,5,6,7 \
  ./minimax_h3_recipes/run_nsys.sh "${CFG}" h3-ws4-fastulysses
echo "[$(stamp)] GPU after nsys: $(gpus)"

echo "[$(stamp)] SWEEP_DONE"
