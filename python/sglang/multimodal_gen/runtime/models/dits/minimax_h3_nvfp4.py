# SPDX-License-Identifier: Apache-2.0
"""Online W4A4 NVFP4 overlay for MiniMax-H3 DiT linears loaded through online FP8.

Experiment surface, env-gated by ``SGLANG_DIFFUSION_MINIMAX_H3_DIT_NVFP4_*`` (off by
default). The selected linears (``mlp`` = ``fc1``/``fc2``, ``all`` adds ``qkv_proj``/
``out_proj``) keep their per-channel FP8 weights and gain an NVFP4 copy: E2M1 codes,
the 128x4-swizzled E4M3 block-16 scales ``flashinfer.fp4_quantize`` emits and one
fp32 tensor scale (``448 * 6 / amax``). Activations are quantised per call with that
call's own tensor amax, so nothing clips and no calibration pass is needed; the GEMM
is ``flashinfer.mm_fp4`` (cutlass or cudnn). Steps below ``FROM_STEP`` and any rows a
producer already quantised to fp8 run the original FP8 method. The fused adaLN norms
stop emitting fp8 rows for overlaid linears on their own: ``_per_token_fp8_blockers``
no longer sees an ``Fp8LinearMethod`` and hands over bf16 rows instead.
"""

from __future__ import annotations

from functools import lru_cache

import torch

from sglang.multimodal_gen import envs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

NVFP4_BACKENDS = ("cutlass", "cudnn")
_FP4_GLOBAL = 448.0 * 6.0  # E4M3 max times E2M1 max
_LAYER_SETS = {
    "mlp": ("mlp.fc1", "mlp.fc2"),
    "all": ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2"),
}


@lru_cache(None)
def _primitives():
    from flashinfer import fp4_quantize, mm_fp4

    return fp4_quantize, mm_fp4


def _current_step() -> int | None:
    """Denoise step index of the running forward, None outside a forward."""
    from sglang.multimodal_gen.runtime.managers.forward_context import (
        get_forward_context,
    )

    try:
        return int(get_forward_context().current_timestep)
    except Exception:
        return None


class MiniMaxH3NVFP4Overlay:
    """Quant-method stand-in: NVFP4 GEMM on bf16 rows, FP8 for early steps and fp8 rows."""

    def __init__(self, fp8_method, *, backend: str, from_step: int):
        if backend not in NVFP4_BACKENDS:
            raise ValueError(
                f"Unsupported MiniMax-H3 DiT NVFP4 backend {backend!r}; expected one of {NVFP4_BACKENDS}"
            )
        self.fp8 = fp8_method
        self.backend = backend
        self.from_step = int(from_step)

    def accepts_mxfp8_input(self, layer) -> bool:  # probed by the DiT's MXFP8 path
        return False

    @torch.no_grad()
    def prepare(self, layer) -> None:
        """Build the NVFP4 weight copy from the FP8 method's ``[K, N]`` weight and ``[N]`` scales."""
        if not layer.weight.is_cuda:
            raise ValueError("MiniMax-H3 DiT NVFP4 needs resident CUDA weights (no offload)")
        fp4_quantize, _ = _primitives()
        weight = (
            layer.weight.t().to(torch.float32)
            * layer.weight_scale.to(torch.float32).reshape(-1, 1)
        ).to(torch.bfloat16).contiguous()
        n, k = weight.shape
        if k % 64 or n % 128:
            raise ValueError(f"MiniMax-H3 DiT NVFP4 needs K % 64 == 0 and N % 128 == 0, got {(n, k)}")
        amax = weight.abs().amax().to(torch.float32).clamp_min(torch.finfo(torch.float32).tiny)
        global_scale = _FP4_GLOBAL / amax
        codes, scales = fp4_quantize(weight, global_scale)
        layer.register_buffer("nvfp4_weight", codes.contiguous(), persistent=False)
        layer.register_buffer("nvfp4_weight_scale", scales.contiguous(), persistent=False)
        layer.register_buffer("nvfp4_weight_global_scale", global_scale, persistent=False)

    def apply(self, layer, x, bias=None):
        if isinstance(x, tuple):
            return self.fp8.apply(layer, x, bias)
        if self.from_step > 0:
            step = _current_step()
            if step is not None and step < self.from_step:
                return self.fp8.apply(layer, x, bias)
        fp4_quantize, mm_fp4 = _primitives()
        shape = x.shape
        rows = x.reshape(-1, shape[-1])
        if not rows.is_contiguous():
            rows = rows.contiguous()
        amax = torch.linalg.vector_norm(rows, float("inf")).to(torch.float32)
        global_scale = _FP4_GLOBAL / amax.clamp_min(torch.finfo(torch.float32).tiny)
        codes, scales = fp4_quantize(rows, global_scale)
        alpha = 1.0 / (global_scale * layer.nvfp4_weight_global_scale)
        out = mm_fp4(
            codes,
            layer.nvfp4_weight.t(),
            scales,
            layer.nvfp4_weight_scale.t(),
            alpha,
            torch.bfloat16,
            backend=self.backend,
        )
        if bias is not None:
            out = out + bias
        return out.reshape(*shape[:-1], out.shape[-1])


def install_minimax_h3_dit_nvfp4(model) -> int:
    """Overlay the configured DiT linears; returns how many were overlaid (0 when off)."""
    layers = (envs.SGLANG_DIFFUSION_MINIMAX_H3_DIT_NVFP4_LAYERS or "").strip().lower()
    if not layers:
        return 0
    if layers not in _LAYER_SETS:
        raise ValueError(
            f"SGLANG_DIFFUSION_MINIMAX_H3_DIT_NVFP4_LAYERS={layers!r}; expected one of {sorted(_LAYER_SETS)}"
        )
    from sglang.multimodal_gen.runtime.layers.quantization.fp8 import Fp8LinearMethod
    from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import (
        _per_token_fp8_blockers,
    )

    backend = envs.SGLANG_DIFFUSION_MINIMAX_H3_DIT_NVFP4_BACKEND
    from_step = int(envs.SGLANG_DIFFUSION_MINIMAX_H3_DIT_NVFP4_FROM_STEP or 0)
    _primitives()  # a missing flashinfer fails the load, not the first step
    count = 0
    for block in model.blocks:
        for path in _LAYER_SETS[layers]:
            linear = block.get_submodule(path)
            method = linear.quant_method
            if isinstance(method, MiniMaxH3NVFP4Overlay):
                continue
            if not isinstance(method, Fp8LinearMethod):
                raise ValueError(
                    f"MiniMax-H3 DiT NVFP4 overlay needs --quantization fp8 (online per-channel FP8); "
                    f"{path} has {type(method).__name__}"
                )
            blockers = _per_token_fp8_blockers(linear)
            if blockers:
                raise ValueError(f"MiniMax-H3 DiT NVFP4 overlay cannot wrap {path}: {'; '.join(blockers)}")
            overlay = MiniMaxH3NVFP4Overlay(method, backend=backend, from_step=from_step)
            overlay.prepare(linear)
            linear.quant_method = overlay
            count += 1
    logger.info(
        "MiniMax-H3 DiT NVFP4 overlay: %d linears (%s) run W4A4 NVFP4 via flashinfer mm_fp4[%s]%s",
        count,
        layers,
        backend,
        f"; steps < {from_step} keep FP8" if from_step else "",
    )
    return count
