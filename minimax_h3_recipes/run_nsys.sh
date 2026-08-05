#!/usr/bin/env bash
# Capture one MiniMax-H3 request with Nsight Systems.
#
# The capture range is driven by cudaProfilerStart/Stop, which
# --enable-cuda-profiler-range issues on world rank 0 around the real request's
# pipeline execution, so weight loading stays out of the report. Warmup is off
# in the recipes and is excluded by the flag itself either way.
#
# Usage: run_nsys.sh <config.json> <report-name>
set -Eeuo pipefail

if (( $# != 2 )); then
  echo "usage: $(basename "$0") <config.json> <report-name>" >&2
  exit 2
fi

H3_CONFIG="$1"
H3_REPORT_NAME="$2"
H3_GPU_IDS="${H3_GPU_IDS:-4,5,6,7}"
H3_SGLANG_BIN="${H3_SGLANG_BIN:-/workspace/.sglang/bin/sglang}"
H3_NSYS_BIN="${H3_NSYS_BIN:-/usr/local/cuda/bin/nsys}"
H3_REPORT_DIR="/workspace/sglang/nsight"
H3_LOG_DIR="/workspace/sglang/logs/minimax_h3_recipes"
H3_RUN_STAMP="$(date +%Y%m%d-%H%M%S)"
H3_LOG="${H3_LOG_DIR}/nsys-${H3_REPORT_NAME}-${H3_RUN_STAMP}.log"

for bin in "${H3_SGLANG_BIN}" "${H3_NSYS_BIN}"; do
  if [[ ! -x "${bin}" ]]; then
    echo "executable not found: ${bin}" >&2
    exit 2
  fi
done
if [[ ! -s "${H3_CONFIG}" ]]; then
  echo "config not found: ${H3_CONFIG}" >&2
  exit 2
fi

IFS=',' read -r -a H3_GPU_LIST <<< "${H3_GPU_IDS}"
if (( ${#H3_GPU_LIST[@]} != 4 )); then
  echo "This recipe requires exactly four GPU ids; got: ${H3_GPU_IDS}" >&2
  exit 2
fi

mkdir -p "${H3_REPORT_DIR}" "${H3_LOG_DIR}"

export CUDA_VISIBLE_DEVICES="${H3_GPU_IDS}"
echo "GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "Config: ${H3_CONFIG}"
echo "Report: ${H3_REPORT_DIR}/${H3_REPORT_NAME}.nsys-rep"
echo "Log: ${H3_LOG}"

cd /workspace/sglang

# --capture-range-end=stop keeps the app alive after cudaProfilerStop so the
# mp4 is still written and the run can be checked for correctness.
# --trace-fork-before-exec covers the spawned GPU worker processes.
"${H3_NSYS_BIN}" profile \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --trace=cuda,nvtx,osrt \
  --cuda-graph-trace=node \
  --trace-fork-before-exec=true \
  --sample=none \
  --cpuctxsw=none \
  --force-overwrite=true \
  --output "${H3_REPORT_DIR}/${H3_REPORT_NAME}" \
  "${H3_SGLANG_BIN}" generate \
    --model-path MiniMaxAI/MiniMax-H3 \
    --config "${H3_CONFIG}" \
    --warmup-mode off \
    --enable-layerwise-nvtx-marker \
    --enable-cuda-profiler-range \
  2>&1 | tee "${H3_LOG}"

test -s "${H3_REPORT_DIR}/${H3_REPORT_NAME}.nsys-rep"
echo "Report: ${H3_REPORT_DIR}/${H3_REPORT_NAME}.nsys-rep"
