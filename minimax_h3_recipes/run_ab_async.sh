#!/usr/bin/env bash
# A/B the split-async q/k/v exchange against the packed synchronous one, with
# fast-ulysses on in BOTH arms -- the question here is the split, not the
# transport. No profiler: nsys changes kernel timing, and the whole point is
# whether a transfer hides behind a kernel.
#
# Usage: H3_GPU_IDS=0,1,3,6 run_ab_async.sh <config.json>
set -u

if (( $# != 1 )); then
  echo "usage: $(basename "$0") <config.json>" >&2
  exit 2
fi

H3_CONFIG="$1"
H3_GPU_IDS="${H3_GPU_IDS:-4,5,6,7}"
H3_REPO="${H3_REPO:-/workspace/sglang-worktrees/fast-ulysses-a2a}"
H3_SGLANG_BIN="${H3_SGLANG_BIN:-/workspace/.sglang/bin/sglang}"
H3_LOG_DIR="${H3_REPO}/logs/minimax_h3_recipes"
H3_RUN_STAMP="$(date +%Y%m%d-%H%M%S)"
H3_AB_ORDER="${H3_AB_ORDER:-sync async}"

IFS=',' read -r -a H3_GPU_LIST <<< "${H3_GPU_IDS}"
H3_NUM_GPUS="${#H3_GPU_LIST[@]}"
case "${H3_NUM_GPUS}" in
  2|4|8) ;;
  *) echo "ulysses degree must be 2, 4 or 8; got ${H3_NUM_GPUS}" >&2; exit 2 ;;
esac

mkdir -p "${H3_LOG_DIR}"
export CUDA_VISIBLE_DEVICES="${H3_GPU_IDS}"
export PYTHONPATH="${H3_REPO}/python${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
cd "${H3_REPO}"

for arm in ${H3_AB_ORDER}; do
  cfg="${H3_LOG_DIR}/async-ws${H3_NUM_GPUS}-${arm}.json"
  python3 - "${H3_CONFIG}" "${cfg}" "${H3_NUM_GPUS}" "${arm}" <<'PY'
import json, sys
src, dst, ws, arm = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
cfg = json.load(open(src))
cfg.update(num_gpus=ws, sp_degree=ws, ulysses_degree=ws, ring_degree=1, tp_size=1)
stem = cfg.get("output_file_name", "out.mp4").rsplit(".", 1)[0]
cfg["output_file_name"] = f"{stem}-async-ws{ws}-{arm}.mp4"
json.dump(cfg, open(dst, "w"), indent=2)
PY

  log="${H3_LOG_DIR}/async-ws${H3_NUM_GPUS}-${arm}-${H3_RUN_STAMP}.log"
  echo "=== ${arm} (ws=${H3_NUM_GPUS}) gpu=$(nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader | tr '\n' ' ') ==="
  SGLANG_DIFFUSION_FAST_ULYSSES=1 \
  SGLANG_DIFFUSION_FAST_ULYSSES_ASYNC_QKV=$([[ "${arm}" == async ]] && echo 1 || echo 0) \
    "${H3_SGLANG_BIN}" generate \
      --model-path MiniMaxAI/MiniMax-H3 \
      --config "${cfg}" \
      --warmup-mode off \
    > "${log}" 2>&1 || echo "  (exit $? -- ffprobe validation is expected to fail here)"
  grep -oE "\[MiniMaxH3DenoisingStage\] finished in [0-9.]+ seconds" "${log}" | tail -1 || true
  grep -oE "49/49 \[[0-9:]+<[0-9:]+, +[0-9.]+(s/it|it/s)\]" "${log}" | tail -1 || true
done

echo "outputs:"
ls -la "${H3_REPO}/outputs" 2>/dev/null | grep async | tail -4
