# SPDX-License-Identifier: Apache-2.0
"""Online NVFP4 for the MiniMax-H3 decoder FFN linears (``ff.w1``, ``ff.w2``).

Weights are quantized once at load: E2M1 codes, 128x4-swizzled E4M3 block-16
scales and one fp32 tensor scale (``448 * 6 / amax``). Activations come from
the fused producers in ``fused_decoder`` with a per-linear input scale that is
calibrated on the first decode after install; until then the linears run an
eager path (``flashinfer.fp4_quantize`` with that call's own amax) and record
the largest activation. ``flashinfer.mm_fp4`` has no bias epilogue, so a
linear's bias is folded into the kernel that consumes its output. The attention
linears stay in online FP8 (``fp8.py``).
"""

from functools import lru_cache
from math import prod

import torch
from torch import nn

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

from .fp8 import MiniMaxH3FP8Linear, inspect_fp8_scope
from .fp8 import _primitives as _fp8_primitives

logger = init_logger(__name__)

NVFP4_LINEARS = frozenset({"ff.w1", "ff.w2"})
NVFP4_BACKENDS = ("cudnn", "cutlass")
FP8_E4M3_MAX = 448.0
E2M1_MAX = 6.0
SCALE_ROW_TILE = 128
# Headroom over the largest activation seen while calibrating; a later activation
# above amax * margin saturates its block scale and clips. Arbitrary.
CALIBRATION_MARGIN = 1.5


@lru_cache(None)
def _primitives():
    from flashinfer import fp4_quantize, mm_fp4

    return fp4_quantize, mm_fp4


def _global_scale(amax: float, device: torch.device) -> torch.Tensor:
    amax = max(float(amax), torch.finfo(torch.float32).tiny)
    return torch.tensor(
        [FP8_E4M3_MAX * E2M1_MAX / amax], dtype=torch.float32, device=device
    )


def _scale_rows(rows: int) -> int:
    return (rows + SCALE_ROW_TILE - 1) // SCALE_ROW_TILE * SCALE_ROW_TILE


class MiniMaxH3NVFP4Linear(nn.Module):
    """Static NVFP4 weights, NVFP4 activations under a calibrated tensor scale, FP16 output."""

    def __init__(self, source: nn.Linear, *, backend: str):
        super().__init__()
        if (
            source.weight.dtype != torch.float16
            or source.bias is None
            or source.bias.dtype != torch.float16
            or not source.weight.is_cuda
        ):
            raise ValueError(
                "Online NVFP4 requires resident, prepared FP16 decoder weights"
            )
        if backend not in NVFP4_BACKENDS:
            raise ValueError(
                f"Unsupported MiniMax-H3 NVFP4 GEMM backend {backend!r}; "
                f"expected one of {NVFP4_BACKENDS}"
            )
        if source.in_features % 64 != 0 or source.out_features % 128 != 0:
            raise ValueError(
                "Online NVFP4 needs in_features % 64 == 0 and out_features % 128 == 0"
            )
        self.backend = backend
        self.in_features = source.in_features
        self.out_features = source.out_features
        self.train(source.training)
        fp4_quantize, _ = _primitives()
        with torch.no_grad(), torch.autocast("cuda", enabled=False):
            weight = source.weight.detach()
            weight_global_scale = _global_scale(
                weight.float().abs().amax().item(), weight.device
            )
            codes, scales = fp4_quantize(weight, weight_global_scale)
            self.weight = nn.Parameter(codes.contiguous(), requires_grad=False)
            self.register_buffer(
                "weight_scale", scales.view(torch.uint8).reshape(-1, self.in_features // 16).contiguous()
            )
            self.register_buffer("weight_global_scale", weight_global_scale)
            self.bias = nn.Parameter(source.bias.detach(), requires_grad=False)
            # fp32 copy for the consumer kernels that fold the bias in.
            self.register_buffer("bias_fp32", source.bias.detach().float().contiguous())
        # Largest |activation| seen by the eager path; frozen into input_global_scale.
        self.input_amax = 0.0
        self.input_global_scale: torch.Tensor | None = None
        self.input_global_scale_value = 0.0
        self.alpha: torch.Tensor | None = None

    @property
    def calibrated(self) -> bool:
        return self.input_global_scale is not None

    def freeze_input_scale(self) -> bool:
        """Fix the activation scale from the calibration amax; False when nothing was seen."""
        if self.calibrated or self.input_amax <= 0.0:
            return False
        scale = _global_scale(self.input_amax * CALIBRATION_MARGIN, self.weight.device)
        self.input_global_scale = scale
        self.input_global_scale_value = scale.item()
        self.alpha = (1.0 / (scale * self.weight_global_scale)).contiguous()
        return True

    def _apply(self, fn, recurse=True):
        # Whole-VAE dtype scopes must not widen the codes or recast the scales.
        device = fn(torch.empty(0, dtype=torch.uint8, device=self.weight.device)).device
        if device.type not in ("cuda", "cpu"):
            raise ValueError("MiniMax-H3 NVFP4 storage supports CPU/CUDA moves only")
        self.weight = nn.Parameter(
            self.weight.detach().to(device=device, dtype=torch.uint8), requires_grad=False
        )
        self.weight_scale = self.weight_scale.to(device=device, dtype=torch.uint8)
        self.weight_global_scale = self.weight_global_scale.to(device=device, dtype=torch.float32)
        self.bias = nn.Parameter(
            self.bias.detach().to(device=device, dtype=torch.float16), requires_grad=False
        )
        self.bias_fp32 = self.bias_fp32.to(device=device, dtype=torch.float32)
        if self.input_global_scale is not None:
            self.input_global_scale = self.input_global_scale.to(device=device)
            self.alpha = self.alpha.to(device=device)
        return self

    def _gemm(self, codes, scales, alpha):
        _, mm_fp4 = _primitives()
        return mm_fp4(
            codes,
            self.weight.t(),
            scales.view(torch.float8_e4m3fn),
            self.weight_scale.view(torch.float8_e4m3fn).t(),
            alpha,
            torch.float16,
            backend=self.backend,
        )

    def forward_quantized(self, codes, scales, shape):
        """GEMM on producer output (E2M1 codes, swizzled E4M3 scales); no bias is added."""
        if self.training or torch.is_grad_enabled():
            raise RuntimeError("MiniMax-H3 NVFP4 linears are inference-only")
        if not self.calibrated:
            raise RuntimeError("MiniMax-H3 NVFP4 linear used before its input scale was calibrated")
        rows = prod(shape[:-1])
        if (
            len(shape) < 2
            or shape[-1] != self.in_features
            or codes.shape != (rows, self.in_features // 2)
            or codes.dtype != torch.uint8
            or not codes.is_contiguous()
            or scales.dtype != torch.uint8
            or scales.dim() != 2
            or scales.shape[0] < _scale_rows(rows)
            or scales.shape[1] != self.in_features // 16
            or not scales.is_contiguous()
            or not codes.is_cuda
            or codes.device != self.weight.device
            or scales.device != codes.device
        ):
            raise ValueError("Unsupported prequantized MiniMax-H3 NVFP4 activation")
        with torch.autocast("cuda", enabled=False):
            out = self._gemm(codes, scales, self.alpha)
        return out.reshape(*shape[:-1], self.out_features)

    def forward(self, value):
        """Eager path: quantize with this call's amax (recorded for calibration), add the bias."""
        if self.training or torch.is_grad_enabled():
            raise RuntimeError("MiniMax-H3 NVFP4 linears are inference-only")
        if (
            not value.is_cuda
            or value.device != self.weight.device
            or value.shape[-1] != self.in_features
            or value.dtype not in (torch.float16, torch.float32)
        ):
            raise ValueError("Unsupported MiniMax-H3 NVFP4 activation")
        if (
            not torch.is_autocast_enabled("cuda")
            or torch.get_autocast_dtype("cuda") != torch.float16
        ):
            raise ValueError("MiniMax-H3 NVFP4 requires FP16 decode autocast")
        fp4_quantize, _ = _primitives()
        with torch.autocast("cuda", enabled=False):
            rounded = value.to(torch.float16).reshape(-1, self.in_features).contiguous()
            amax = rounded.float().abs().amax().item()
            self.input_amax = max(self.input_amax, amax)
            if self.calibrated:
                scale, alpha = self.input_global_scale, self.alpha
            else:
                scale = _global_scale(amax, rounded.device)
                alpha = (1.0 / (scale * self.weight_global_scale)).contiguous()
            codes, scales = fp4_quantize(rounded, scale)
            out = self._gemm(
                codes, scales.view(torch.uint8).reshape(-1, self.in_features // 16), alpha
            )
            out = out + self.bias
        return out.reshape(*value.shape[:-1], self.out_features)


def install_mixed_block_linears(decoder, *, backend: str) -> int:
    """FFN linears to NVFP4, attention linears to FP8; returns how many were replaced."""
    if decoder.fp8_installed:
        return 0
    paths = inspect_fp8_scope(decoder)
    decoder.prepare_autocast_linear_weights(torch.float16)
    # Import the quantizers and GEMMs now so a missing flashinfer / sgl_kernel fails the load.
    _primitives()
    _fp8_primitives()
    for path in paths:
        parent, name = path.rsplit(".", 1)
        suffix = ".".join(path.split(".")[2:])
        source = decoder.get_submodule(path)
        replacement = (
            MiniMaxH3NVFP4Linear(source, backend=backend)
            if suffix in NVFP4_LINEARS
            else MiniMaxH3FP8Linear(source)
        )
        decoder.get_submodule(parent)._modules[name] = replacement
    decoder.fp8_installed = True
    decoder.nvfp4_installed = True
    decoder._autocast_linear_dtype = torch.float16
    return len(paths)


def nvfp4_linears(decoder):
    for block in decoder.transformer_blocks:
        for module in (block.ff.w1, block.ff.w2):
            if isinstance(module, MiniMaxH3NVFP4Linear):
                yield module


def freeze_nvfp4_calibration(decoder) -> int:
    """Freeze every NVFP4 linear that saw activations; returns how many were frozen."""
    frozen = sum(module.freeze_input_scale() for module in nvfp4_linears(decoder))
    if frozen:
        logger.info(
            "[H3 VAE] NVFP4 input scales calibrated for %d decoder linears "
            "(margin %.2f); later decodes run the fused NVFP4 producers",
            frozen,
            CALIBRATION_MARGIN,
        )
    return frozen
