#!/bin/bash
# Runs inside the container. Environment + import + unit-test smoke for the PR tree.
set -u
EXP=/workspace/experiment/minimax-h3/2026-09-22-minimax-h3-vae-pr-repro
export PYTHONPATH=/workspace/source/worktrees/h3-vae-pr/python PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 FLASHINFER_DISABLE_VERSION_CHECK=1
export SGLANG_JIT_CACHE_DIR=$EXP/cache/jit TORCH_EXTENSIONS_DIR=$EXP/cache/torch TRITON_CACHE_DIR=$EXP/cache/triton
export TORCHINDUCTOR_CACHE_DIR=$EXP/cache/inductor CUDA_CACHE_PATH=$EXP/cache/cuda XDG_CACHE_HOME=$EXP/cache/xdg FLASHINFER_WORKSPACE_BASE=$EXP/cache/flashinfer
export PATH=/opt/sglang/bin:/usr/local/cuda/bin:$PATH
cd /workspace/source/worktrees/h3-vae-pr
echo "== host $(hostname) job ${SLURM_JOB_ID:-?} cpus $(nproc) affinity $(taskset -cp $$ 2>/dev/null | cut -d: -f2)"
nvidia-smi --query-gpu=index,name,pci.bus_id,temperature.gpu,clocks.sm,clocks.max.sm,clocks_throttle_reasons.active --format=csv
echo "== git"; git rev-parse --short=10 HEAD; git status --short | head -3
python3 - <<'PY'
import importlib, torch, sglang, flashinfer, sgl_kernel
print("torch", torch.__version__, "cuda", torch.version.cuda, "cudnn", torch.backends.cudnn.version())
print("flashinfer", getattr(flashinfer, "__version__", "?"), flashinfer.__file__)
print("sglang", sglang.__file__)
print("sgl_kernel", getattr(sgl_kernel, "__version__", "?"), sgl_kernel.__file__)
from sgl_kernel import fp8_scaled_mm
from flashinfer import mm_fp4, fp4_quantize
from sglang.kernels.ops.quantization.fp8_kernel import sglang_per_token_quant_fp8
import sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.fp8 as m
print("PR fp8 module", m.__file__)
for mod in ["skimage", "numpy", "imageio"]:
    try:
        importlib.import_module(mod); print(mod, "ok")
    except Exception as e:  # noqa: BLE001
        print(mod, "MISSING", e)
PY
echo "== tools"; which ffmpeg ffprobe numactl nsys | sed 's/^/  /'
echo "== kernel tests"
python3 -m pytest test/registered/kernels/ops/diffusion/test_minimax_h3_vae_fp8_decoder.py test/registered/kernels/ops/diffusion/test_minimax_h3_vae_fp8_producers.py test/registered/kernels/ops/diffusion/test_minimax_h3_vae_output.py test/registered/kernels/ops/diffusion/test_group_norm_silu_ncthw.py -q -x -p no:cacheprovider 2>&1 | tail -6
echo "== unit tests"
python3 -m pytest python/sglang/multimodal_gen/test/unit/test_minimax_h3_vae_fast_path.py -q -x -p no:cacheprovider 2>&1 | tail -4
echo "== smoke done"
