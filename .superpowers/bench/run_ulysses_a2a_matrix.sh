#!/usr/bin/env bash
# H200x8 benchmark matrix for the custom Ulysses a2a op.
# 2 models x {U8R1, U4R2} x {baseline, custom op TMA+tune}.
# Run inside docker `sglang-diffusion-ulysess` after `source /data/.torch/bin/activate`.
set -euo pipefail

OUT_DIR="${OUT_DIR:-/data/ulysses_a2a_bench}"
PROMPT="A cat and a dog baking a cake together in a kitchen."
mkdir -p "$OUT_DIR/torch"

run_cell() {
  local model="$1" tag="$2" uly="$3" ring="$4" use_custom="$5"
  local label="${tag}_U${uly}R${ring}_$([ "$use_custom" = 1 ] && echo custom || echo baseline)"
  echo "=== $label ==="
  export SGLANG_DIFFUSION_USE_CUSTOM_ULYSSES_A2A="$use_custom"
  export SGLANG_DIFFUSION_CUSTOM_ULYSSES_TMA=1

  # 1) clean latency (no --profile); read the "(with warmup excluded)" line
  sglang generate --backend=sglang --model-path="$model" --prompt="$PROMPT" \
    --width=1280 --height=720 --num-frames=81 --seed=42 \
    --num-gpus=8 --ulysses-degree="$uly" --ring-degree="$ring" \
    --warmup --perf-dump-path="$OUT_DIR/${label}.perf.json" \
    2>&1 | tee "$OUT_DIR/${label}.latency.log"

  # 2) kernel trace
  SGLANG_DIFFUSION_TORCH_PROFILER_DIR="$OUT_DIR/torch/${label}" \
  sglang generate --backend=sglang --model-path="$model" --prompt="$PROMPT" \
    --width=1280 --height=720 --num-frames=81 --seed=42 \
    --num-gpus=8 --ulysses-degree="$uly" --ring-degree="$ring" \
    --warmup --profile --num-profiled-timesteps=5 \
    2>&1 | tee "$OUT_DIR/${label}.profile.log"
}

for MODEL in "Wan-AI/Wan2.1-T2V-14B-Diffusers" "Wan-AI/Wan2.2-T2V-A14B-Diffusers"; do
  TAG="$(echo "$MODEL" | sed 's#.*/##; s/-Diffusers//')"
  for USE_CUSTOM in 0 1; do
    run_cell "$MODEL" "$TAG" 8 1 "$USE_CUSTOM"   # 单独 usp
    run_cell "$MODEL" "$TAG" 4 2 "$USE_CUSTOM"   # cp+usp
  done
done
echo "Done. Artifacts in $OUT_DIR"
