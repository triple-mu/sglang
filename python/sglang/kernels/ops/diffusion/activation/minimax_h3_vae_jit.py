# SPDX-License-Identifier: Apache-2.0
"""Close-contract H3 SiLU/multiply and dynamic per-token FP8 quantization.

FP32 intermediates remove eager FP16 rounding boundaries. Mount only behind
request quality policy; tested prototype FFN width is 8192 on SM100.
"""

import torch

from ..common.minimax_h3_vae_jit import load_vae_kernel, supported_cuda_tensor


def can_use_minimax_h3_vae_silu_quant(value) -> bool:
    return (
        supported_cuda_tensor(value)
        and value.ndim >= 2
        and value.is_contiguous()
        and value.data_ptr() % 16 == 0
        and value.dtype in (torch.float16, torch.bfloat16)
        and value.shape[-1] == 16384
    )


def minimax_h3_vae_silu_quant(value):
    if not can_use_minimax_h3_vae_silu_quant(value):
        raise ValueError("MiniMax-H3 gated FFN fusion requires contiguous width 2*8192")
    flat = value.view(-1, 16384)
    out = torch.empty(
        (flat.shape[0], 8192), device=value.device, dtype=torch.float8_e4m3fn
    )
    scale = torch.empty((flat.shape[0], 1), device=value.device, dtype=torch.float32)
    load_vae_kernel("silu", value.dtype, width=8192).run(flat, out, scale)
    return out, scale
