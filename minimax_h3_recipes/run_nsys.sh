#!/usr/bin/env bash
# Capture one MiniMax-H3 request with Nsight Systems, at 2, 4 or 8 GPUs.
#
# The capture range is driven by cudaProfilerStart/Stop, which
# --enable-cuda-profiler-range issues around the real request's pipeline
# execution, so weight loading stays out of the report. Every rank issues the
# pair: nsys tracks a capture range per process, and the GPU workers are
# separate spawned processes, so a rank that never calls cudaProfilerStart
# contributes nothing to the report.
#
# Warmup is off in the recipes and is excluded by the flag itself either way.
#
# Usage: H3_GPU_IDS=4,5,6,7 run_nsys.sh <config.json> <report-name>
set -Eeuo pipefail

if (( $# != 2 )); then
  echo "usage: $(basename "$0") <config.json> <report-name>" >&2
  exit 2
fi

H3_CONFIG="$1"
H3_REPORT_NAME="$2"
H3_GPU_IDS="${H3_GPU_IDS:-4,5,6,7}"
# Warmup must be on to exercise breakable CUDA graph capture, which is what
# makes --cuda-graph-trace=node show anything. The capture range still excludes
# warmup, so the report stays clean either way.
H3_WARMUP_MODE="${H3_WARMUP_MODE:-off}"
H3_EXTRA_ARGS="${H3_EXTRA_ARGS:-}"
H3_REPO="${H3_REPO:-/workspace/sglang-worktrees/fast-ulysses-a2a}"
H3_SGLANG_BIN="${H3_SGLANG_BIN:-/workspace/.sglang/bin/sglang}"
H3_NSYS_BIN="${H3_NSYS_BIN:-/usr/local/cuda/bin/nsys}"
H3_REPORT_DIR="${H3_REPO}/nsight"
H3_LOG_DIR="${H3_REPO}/logs/minimax_h3_recipes"
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
H3_NUM_GPUS="${#H3_GPU_LIST[@]}"
# H3 admits Ulysses only, and _validate_sequence_parallel_config requires both
# num_heads (56) and 64 to divide the degree -- which leaves 2, 4 and 8.
case "${H3_NUM_GPUS}" in
  2|4|8) ;;
  *)
    echo "ulysses degree must be 2, 4 or 8; got ${H3_NUM_GPUS} from ${H3_GPU_IDS}" >&2
    exit 2
    ;;
esac

mkdir -p "${H3_REPORT_DIR}" "${H3_LOG_DIR}"

# Rewrite the parallel degrees into a scratch config rather than relying on CLI
# overrides winning against the config file.
H3_RESOLVED_CONFIG="${H3_LOG_DIR}/${H3_REPORT_NAME}-ws${H3_NUM_GPUS}.json"
python3 - "${H3_CONFIG}" "${H3_RESOLVED_CONFIG}" "${H3_NUM_GPUS}" "${H3_REPORT_NAME}" \
         "${H3_WARMUP_MODE}" <<'PY'
import json, sys
src, dst, ws, report, warmup = (
    sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5]
)
cfg = json.load(open(src))
cfg["num_gpus"] = ws
cfg["sp_degree"] = ws
cfg["ulysses_degree"] = ws
cfg["ring_degree"] = 1
cfg["tp_size"] = 1
# The config file wins over the CLI flag, so the mode has to be written here.
cfg["warmup_mode"] = warmup
stem = cfg.get("output_file_name", "out.mp4").rsplit(".", 1)[0]
cfg["output_file_name"] = f"{stem}-{report}-ws{ws}.mp4"
json.dump(cfg, open(dst, "w"), indent=2)
print(f"resolved config -> {dst} (ulysses_degree={ws})")
PY

export CUDA_VISIBLE_DEVICES="${H3_GPU_IDS}"
export PYTHONPATH="${H3_REPO}/python${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
# This container cannot bind NVLink SHARP multicast (CUDA 401 from the Fabric
# Manager), which fails NCCL init rather than degrading.
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"

echo "GPUs: ${CUDA_VISIBLE_DEVICES} (ulysses_degree=${H3_NUM_GPUS})"
echo "Repo: ${H3_REPO}"
echo "Config: ${H3_RESOLVED_CONFIG}"
echo "Report: ${H3_REPORT_DIR}/${H3_REPORT_NAME}.nsys-rep"
echo "Log: ${H3_LOG}"

cd "${H3_REPO}"

# --capture-range-end=stop keeps the app alive after cudaProfilerStop so the
# mp4 is still written and the run can be checked for correctness.
# --trace-fork-before-exec covers the spawned GPU worker processes.
# --cuda-graph-trace=node expands captured graphs into their nodes, which is
# what makes the BCG segments readable.
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
    --config "${H3_RESOLVED_CONFIG}" \
    --warmup-mode "${H3_WARMUP_MODE}" \
    --enable-layerwise-nvtx-marker \
    --enable-cuda-profiler-range \
    ${H3_EXTRA_ARGS} \
  2>&1 | tee "${H3_LOG}"

test -s "${H3_REPORT_DIR}/${H3_REPORT_NAME}.nsys-rep"
echo "Report: ${H3_REPORT_DIR}/${H3_REPORT_NAME}.nsys-rep"
