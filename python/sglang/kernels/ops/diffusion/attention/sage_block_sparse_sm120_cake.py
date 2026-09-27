"""SM120 Sage (QK-INT8 / PV-FP8) block-sparse attention, vendored Cake kernel.

One generated specialization of flashinfer PR #4951 compiled through the SGLang
JIT (see ``kernels/jit/csrc/diffusion/cake_sage_bsa_sm120/README.md``). The
operands follow the SageAttention SM120 contract that
``sglang.kernels.ops.diffusion.ulysses_lowp_unpack_for_sage`` produces:

* ``q_int8`` / ``k_int8``: INT8 ``[B, H, S, 128]``; ``q_scale`` fp32
  ``[B, H, ceil(seqlen_q / 128) * 4]`` (one per 32-query group), ``k_scale``
  fp32 ``[B, H, seqlen_k / 64]`` (one per key block, K channel-mean centred).
* ``v_fp8``: FP8 E4M3 ``[B, H, 128, S_k]`` with the Sage 16-token permutation
  baked in; ``v_scale`` fp32 ``[B, H, 128]``.
* ``q2k_block_index``: int32 ``[B, H, ceil(seqlen_q / 64), capacity]`` key
  blocks per query block in any order; ``q2k_block_nums`` int32
  ``[B, H, ceil(seqlen_q / 64)]`` live entries per row (``0`` allowed).

``seqlen_q`` / ``seqlen_k`` default to the allocated rows and may be smaller, so
a resident staging buffer can be passed in and only its live prefix computed;
``seqlen_k`` must be a multiple of 64 (pad K/V with zero rows). Rows of ``out``
at or beyond ``seqlen_q`` are not written. MHA only, head dimension 128,
non-causal, no LSE, BF16 output. Numerics: P is quantised to FP8 inside the
kernel, so results match a dequantised fp32 reference to about 1e-2 relative L2
(measured 0.037 against unquantised bf16 SDPA on MiniMax-H3 activations).

Compute capability 12.0 only; ``can_use_sage_block_sparse_attn_sm120`` is the
cheap gate, everything about the tensors is checked in the launcher.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module

#: sha256 of the vendored generated kernel; a re-sync must update it together
#: with vendor/MANIFEST.json so nobody edits the generated source by accident.
_VENDOR_KERNEL = "diffusion/cake_sage_bsa_sm120/vendor/cake_sage_block_sparse_attention_939d22c4b83f8f4c938f_kernel.cu"
_VENDOR_KERNEL_SHA256 = (
    "faacbf9c99619f01d553cf17a9a9a0fd6d1a259c2f887fd9d0c6f5f04f55b59b"
)
_DESCRIPTOR_WORKSPACE_BYTES = 512
_BLOCK = 64
_HEAD_DIM = 128
_REQUIRED_CAPABILITY = (12, 0)


def vendored_kernel_path() -> Path:
    import sglang.kernels.jit as jit_root

    return Path(jit_root.__file__).resolve().parent / "csrc" / _VENDOR_KERNEL


def verify_vendored_kernel() -> None:
    """Refuse to build from a generated source that no longer matches the manifest."""
    digest = hashlib.sha256(vendored_kernel_path().read_bytes()).hexdigest()
    if digest != _VENDOR_KERNEL_SHA256:
        raise RuntimeError(
            "cake_sage_bsa_sm120: vendored kernel sha256 mismatch "
            f"({digest[:16]}... != {_VENDOR_KERNEL_SHA256[:16]}...); the generated "
            "source must not be edited, re-sync it from upstream and update the hash"
        )


@cache_once
def _jit_module() -> Module:
    if torch.version.hip is not None:
        raise RuntimeError("cake_sage_bsa_sm120 is a CUDA SM120 kernel")
    capability = tuple(torch.cuda.get_device_capability())
    if capability != _REQUIRED_CAPABILITY:
        raise RuntimeError(
            f"cake_sage_bsa_sm120 requires compute capability {_REQUIRED_CAPABILITY}, "
            f"got {capability}"
        )
    verify_vendored_kernel()
    return load_jit(
        "diffusion_cake_sage_bsa_sm120",
        cuda_files=["diffusion/cake_sage_bsa_sm120/entry.cuh"],
        cuda_wrappers=[("run", "cake_sage_bsa_sm120::run")],
        # Part of the generated kernel's contract (upstream manifest compile_flags).
        extra_cuda_cflags=["--use_fast_math"],
    )


@cache_once
def _descriptor_workspace(device_index: int) -> torch.Tensor:
    """One 512-byte, 128-byte-aligned TMA descriptor slot set per device."""
    raw = torch.empty(
        _DESCRIPTOR_WORKSPACE_BYTES + 128,
        dtype=torch.uint8,
        device=f"cuda:{device_index}",
    )
    offset = (-raw.data_ptr()) % 128
    return raw[offset : offset + _DESCRIPTOR_WORKSPACE_BYTES]


def can_use_sage_block_sparse_attn_sm120() -> bool:
    return (
        torch.cuda.is_available()
        and torch.version.hip is None
        and tuple(torch.cuda.get_device_capability()) == _REQUIRED_CAPABILITY
    )


def sage_block_sparse_attn_sm120(
    q_int8: torch.Tensor,
    k_int8: torch.Tensor,
    v_fp8: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    q2k_block_index: torch.Tensor,
    q2k_block_nums: torch.Tensor,
    *,
    out: torch.Tensor,
    seqlen_q: int | None = None,
    seqlen_k: int | None = None,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """Run the kernel into ``out`` (BF16 ``[B, H, SQ_ALLOC, 128]``) and return it.

    Shapes, dtypes, alignment and the ``seqlen_k % 64 == 0`` rule are enforced
    by the C++ launcher; this entry point only resolves the defaults.
    """
    if seqlen_q is None:
        seqlen_q = int(q_int8.shape[2])
    if seqlen_k is None:
        seqlen_k = int(k_int8.shape[2])
    if softmax_scale is None:
        softmax_scale = _HEAD_DIM**-0.5
    module = _jit_module()
    module.run(
        q_int8,
        k_int8,
        v_fp8,
        out,
        q_scale,
        k_scale,
        v_scale,
        q2k_block_index,
        q2k_block_nums,
        _descriptor_workspace(q_int8.device.index),
        int(seqlen_q),
        int(seqlen_k),
        float(softmax_scale),
    )
    return out


def sage_block_sparse_dense_block_index(
    batch: int, heads: int, seqlen_q: int, seqlen_k: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Full-density routing tables: every query block keeps every key block.

    Used for the dense warmup steps of a sparse schedule so that one kernel
    serves both regimes.
    """
    q_blocks = -(-seqlen_q // _BLOCK)
    k_blocks = seqlen_k // _BLOCK
    index = (
        torch.arange(k_blocks, device=device, dtype=torch.int32)
        .view(1, 1, 1, k_blocks)
        .expand(batch, heads, q_blocks, k_blocks)
        .contiguous()
    )
    nums = torch.full(
        (batch, heads, q_blocks), k_blocks, device=device, dtype=torch.int32
    )
    return index, nums


__all__ = [
    "can_use_sage_block_sparse_attn_sm120",
    "sage_block_sparse_dense_block_index",
    "sage_block_sparse_attn_sm120",
    "verify_vendored_kernel",
]
