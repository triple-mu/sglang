#!/usr/bin/env bash
set -Eeuo pipefail

H3_RECIPE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
H3_GPU_IDS="${H3_GPU_IDS:-4,5,6,7}"
H3_SGLANG_BIN="${H3_SGLANG_BIN:-/workspace/.sglang/bin/sglang}"
H3_CONFIG="${H3_RECIPE_DIR}/fl2va_first_frame_5s.json"
H3_INPUT="/workspace/sglang/inputs/minimax_h3_recipes/t2va_first_frame.png"
H3_OUTPUT_DIR="/workspace/sglang/outputs/minimax_h3_recipes"
H3_OUTPUT="${H3_OUTPUT_DIR}/fl2va_first_frame_5s.mp4"
H3_LOG_DIR="/workspace/sglang/logs/minimax_h3_recipes"
H3_RUN_STAMP="$(date +%Y%m%d-%H%M%S)"
H3_LOG="${H3_LOG_DIR}/fl2va-${H3_RUN_STAMP}.log"

if [[ ! -x "${H3_SGLANG_BIN}" ]]; then
  echo "sglang executable not found: ${H3_SGLANG_BIN}" >&2
  exit 2
fi
if [[ ! -s "${H3_INPUT}" ]]; then
  echo "Missing FL2VA first frame: ${H3_INPUT}" >&2
  echo "Run ./prepare_fl2va_input.sh first." >&2
  exit 2
fi

IFS=',' read -r -a H3_GPU_LIST <<< "${H3_GPU_IDS}"
if (( ${#H3_GPU_LIST[@]} != 4 )); then
  echo "This recipe requires exactly four GPU ids; got: ${H3_GPU_IDS}" >&2
  exit 2
fi

mkdir -p "${H3_OUTPUT_DIR}" "${H3_LOG_DIR}"
if [[ -e "${H3_OUTPUT}" ]]; then
  H3_BACKUP="${H3_OUTPUT%.mp4}.before-${H3_RUN_STAMP}.mp4"
  cp -p -- "${H3_OUTPUT}" "${H3_BACKUP}"
  echo "Previous output preserved as: ${H3_BACKUP}"
fi

export CUDA_VISIBLE_DEVICES="${H3_GPU_IDS}"
echo "Task: fl2va (first frame)"
echo "GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "Input: ${H3_INPUT}"
echo "Config: ${H3_CONFIG}"
echo "Log: ${H3_LOG}"

cd /workspace/sglang
"${H3_SGLANG_BIN}" generate \
  --model-path MiniMaxAI/MiniMax-H3 \
  --config "${H3_CONFIG}" \
  --warmup-mode off \
  2>&1 | tee "${H3_LOG}"

test -s "${H3_OUTPUT}"
echo "FL2VA output: ${H3_OUTPUT}"
