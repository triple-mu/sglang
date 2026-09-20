# SPDX-License-Identifier: Apache-2.0
"""Lazy CUDA/C++ specialization shared by MiniMax-H3 VAE operators."""

from functools import lru_cache

import torch


def supported_cuda_tensor(value: torch.Tensor) -> bool:
    return (
        value.is_cuda
        and torch.cuda.is_available()
        and torch.version.hip is None
        and value.numel() > 0
        and not torch.is_grad_enabled()
        and not torch.compiler.is_compiling()
        and torch.cuda.get_device_capability(value.device) == (10, 0)
        and not torch.cuda.is_current_stream_capturing()
    )


@lru_cache(None)
def load_vae_kernel(kind, dtype, width=2048, mode=0, projected_dtype=torch.float16):
    from sglang.kernels.jit.utils import load_jit, make_cpp_args

    if kind == "qk":
        args = make_cpp_args(dtype, False)
        function = f"minimax_h3_vae::joint_qk<{args}>"
    elif kind == "norm":
        args = make_cpp_args(dtype, projected_dtype, width, mode)
        function = f"minimax_h3_vae::norm_quant<{args}>"
    elif kind == "silu":
        args = make_cpp_args(dtype, width)
        function = f"minimax_h3_vae::silu_quant_vectorized<{args}>"
    else:
        raise ValueError(f"Unknown MiniMax-H3 VAE kernel {kind!r}")
    return load_jit(
        "diffusion_minimax_h3_vae_" + kind,
        *args,
        cuda_files=["diffusion/minimax_h3_vae.cuh"],
        cuda_wrappers=[("run", function)],
    )


@lru_cache(None)
def load_output_kernels():
    from sglang.kernels.jit.utils import load_jit

    return load_jit(
        "diffusion_minimax_h3_vae_output",
        cuda_files=["diffusion/minimax_h3_vae_output.cuh"],
        cuda_wrappers=[
            ("assemble", "minimax_h3_vae_output::assemble"),
            ("temporal", "minimax_h3_vae_output::temporal"),
            ("denorm", "minimax_h3_vae_output::denorm"),
        ],
        extra_cuda_cflags=["--fmad=false"],
    )


@lru_cache(None)
def load_group_norm_kernel():
    from sglang.kernels.jit.utils import load_jit

    return load_jit(
        "diffusion_minimax_h3_vae_groupnorm",
        cuda_files=["diffusion/minimax_h3_vae_groupnorm.cuh"],
        cuda_wrappers=[("run", "minimax_h3_vae::group_norm_silu")],
        extra_cuda_cflags=["--fmad=false"],
    )
