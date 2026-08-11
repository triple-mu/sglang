#!/usr/bin/env bash
# A/B one MiniMax-H3 request with and without fast-ulysses, no profiler.
#
# nsys -- especially --cuda-graph-trace=node -- changes kernel timing, so the
# end-to-end claim has to come from an unprofiled run. Same seed both arms, so
# the outputs are comparable byte for byte: the exchange is bit-exact, and any
# difference in the mp4 is a bug rather than a tolerance.
#
# Usage: H3_GPU_IDS=4,5,6,7 run_ab.sh <config.json>
set -Eeuo pipefail

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
# Run the arms in either order. A one-directional sweep cannot tell a real
# speedup from "whatever runs second is faster" (warm caches, clocks), so the
# claim only holds if reversing the order reproduces it.
H3_AB_ORDER="${H3_AB_ORDER:-off on}"

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
  cfg="${H3_LOG_DIR}/ab-ws${H3_NUM_GPUS}-${arm}.json"
  python3 - "${H3_CONFIG}" "${cfg}" "${H3_NUM_GPUS}" "${arm}" <<'PY'
import json, sys
src, dst, ws, arm = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
cfg = json.load(open(src))
cfg.update(num_gpus=ws, sp_degree=ws, ulysses_degree=ws, ring_degree=1, tp_size=1)
stem = cfg.get("output_file_name", "out.mp4").rsplit(".", 1)[0]
cfg["output_file_name"] = f"{stem}-ab-ws{ws}-{arm}.mp4"
json.dump(cfg, open(dst, "w"), indent=2)
PY

  log="${H3_LOG_DIR}/ab-ws${H3_NUM_GPUS}-${arm}-${H3_RUN_STAMP}.log"
  echo "=== fast-ulysses ${arm} (ws=${H3_NUM_GPUS}) -> ${log} ==="
  # The final-output validation shells out to ffprobe, which this image lacks.
  # That failure is after the pixels exist, so it must not abort the sweep --
  # the denoise timings below are read from the log either way.
  SGLANG_DIFFUSION_FAST_ULYSSES=$([[ "${arm}" == on ]] && echo 1 || echo 0) \
    "${H3_SGLANG_BIN}" generate \
      --model-path MiniMaxAI/MiniMax-H3 \
      --config "${cfg}" \
      --warmup-mode off \
    > "${log}" 2>&1 || echo "  (exit $? -- check the log; ffprobe validation is expected to fail here)"
  # Every one of these is a report line, not a check: a miss must not abort the
  # sweep under `pipefail`. tqdm flips to "it/s" above one iteration per
  # second, which is exactly what the faster arm does, so match both spellings.
  grep -oE "\[MiniMaxH3DenoisingStage\] finished in [0-9.]+ seconds" "${log}" | tail -1 || true
  grep -oE "49/49 \[[0-9:]+<[0-9:]+, +[0-9.]+(s/it|it/s)\]" "${log}" | tail -1 || true
  grep -oE "fast-ulysses \(world_size=[0-9]+\)" "${log}" | tail -1 || true
done

echo
echo "outputs:"
ls -la "$(python3 -c "import json,sys;print(json.load(open('${H3_CONFIG}'))['output_path'])")" 2>/dev/null | tail -4
