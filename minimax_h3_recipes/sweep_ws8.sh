#!/usr/bin/env bash
# The 8-GPU comparison. Same shape as sweep_ws4.sh.
#
# Needs all eight GPUs, which on a shared box is a window rather than a given:
# H3 admits Ulysses only at degree 2, 4 or 8, so seven free GPUs are worth
# nothing here. GPU utilisation is sampled around every run so a contended
# number can be thrown away rather than quoted.
set -u

REPO=/workspace/sglang-worktrees/fast-ulysses-a2a
CFG=minimax_h3_recipes/t2va_5s.json
GPUS=0,1,2,3,4,5,6,7
cd "${REPO}"

stamp() { date +%H:%M:%S; }
gpus()  { nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader | tr '\n' '|'; }

echo "[$(stamp)] GPU before: $(gpus)"

echo "[$(stamp)] === round 1: off then on ==="
H3_GPU_IDS=${GPUS} H3_AB_ORDER="off on" ./minimax_h3_recipes/run_ab.sh "${CFG}"
echo "[$(stamp)] GPU after round 1: $(gpus)"

echo "[$(stamp)] === round 2: on then off ==="
H3_GPU_IDS=${GPUS} H3_AB_ORDER="on off" ./minimax_h3_recipes/run_ab.sh "${CFG}"
echo "[$(stamp)] GPU after round 2: $(gpus)"

echo "[$(stamp)] === nsys baseline ==="
H3_GPU_IDS=${GPUS} ./minimax_h3_recipes/run_nsys.sh "${CFG}" h3-ws8-baseline
echo "[$(stamp)] === nsys fast-ulysses ==="
SGLANG_DIFFUSION_FAST_ULYSSES=1 H3_GPU_IDS=${GPUS} \
  ./minimax_h3_recipes/run_nsys.sh "${CFG}" h3-ws8-fastulysses
echo "[$(stamp)] GPU after nsys: $(gpus)"

echo "[$(stamp)] SWEEP_DONE"
