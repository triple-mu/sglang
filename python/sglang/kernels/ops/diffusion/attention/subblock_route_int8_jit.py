# SPDX-License-Identifier: Apache-2.0
"""Sub-block routing inputs straight from the Sage INT8 operands.

``subblock_pool_int8`` mean-pools the INT8 Q/K operands into the bf16 cells the
SubBlock router scores, bit for bit what pooling BF16 copies of them produced,
without materialising those copies. ``subblock_block_tables`` writes the Cake
``(q2k_block_index, q2k_block_nums)`` pair for a query-block mask in one launch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, is_hip_runtime, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module

HEAD_DIM = 128


@cache_once
def _module() -> Module:
    return load_jit(
        "diffusion_subblock_route_int8",
        cuda_files=["diffusion/subblock_route_int8.cuh"],
        cuda_wrappers=[
            ("pool", "subblock_route_int8::Kernels::pool"),
            ("block_tables", "subblock_route_int8::Kernels::block_tables"),
        ],
    )


def can_use_subblock_route_int8() -> bool:
    return torch.cuda.is_available() and not is_hip_runtime()


def subblock_pool_int8(
    q: torch.Tensor,
    k: torch.Tensor,
    k_scale: torch.Tensor,
    *,
    used: int,
    sub_q: int,
    sub_k: int,
    cells_q: int,
    cells_k: int,
    q_factor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Routing cells `([B*H, cells_q, 128], [B*H, cells_k, 128])` bf16 from `[B, H, S, 128]` int8 Q/K.

    K tokens carry their per-64-token scale, Q cells are multiplied by `q_factor`;
    rows at or past `used` are left out of the means.
    """
    batch, heads = q.shape[:2]
    pooled_q = torch.empty(
        (batch * heads, cells_q, HEAD_DIM), dtype=torch.bfloat16, device=q.device
    )
    pooled_k = torch.empty(
        (batch * heads, cells_k, HEAD_DIM), dtype=torch.bfloat16, device=q.device
    )
    _module().pool(
        q,
        k,
        k_scale,
        pooled_q,
        pooled_k,
        int(used),
        int(sub_q),
        int(sub_k),
        float(q_factor),
    )
    return pooled_q, pooled_k


def subblock_block_tables(
    index: torch.Tensor, mask: torch.Tensor, num_blocks: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cake `(q2k_block_index, q2k_block_nums)`; query blocks with a false `mask` keep every key block."""
    batch, heads, q_blocks, _ = index.shape
    tables = torch.empty(
        (batch, heads, q_blocks, num_blocks), dtype=torch.int32, device=index.device
    )
    nums = torch.empty((batch, heads, q_blocks), dtype=torch.int32, device=index.device)
    _module().block_tables(index, mask.contiguous().view(torch.uint8), tables, nums)
    return tables, nums


__all__ = [
    "can_use_subblock_route_int8",
    "subblock_block_tables",
    "subblock_pool_int8",
]
