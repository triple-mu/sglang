"""The INT8 routing-cell kernels against the BF16 route they replace."""

import sys

import pytest
import torch

from sglang.kernels.ops.diffusion import subblock_block_tables, subblock_pool_int8
from sglang.multimodal_gen.runtime.layers.attention.backends.subblock_sparse.kernels import (
    fused_pool,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA kernels under test"
)

HEAD_DIM = 128
LOG2E = 1.4426950408889634


def _bf16_route_pooled(
    q, k, k_scale, *, used, sub_q, sub_k, cells_q, cells_k, q_factor
):
    """Pool BF16 copies the way the backend did before pooling INT8 directly."""
    heads = q.shape[1]
    per_token = k_scale.to(torch.bfloat16).repeat_interleave(64, dim=-1)[:, :, :used]
    q_bf16 = q[:, :, :used].to(torch.bfloat16).transpose(1, 2)
    k_bf16 = (k[:, :, :used].to(torch.bfloat16) * per_token.unsqueeze(-1)).transpose(
        1, 2
    )
    pooled_q = torch.empty(
        heads, cells_q, HEAD_DIM, dtype=torch.bfloat16, device=q.device
    )
    pooled_k = torch.empty(
        heads, cells_k, HEAD_DIM, dtype=torch.bfloat16, device=q.device
    )
    fused_pool(q_bf16, cells_q, sub_q, pooled_q, scale=q_factor)
    fused_pool(k_bf16, cells_k, sub_k, pooled_k)
    return pooled_q, pooled_k


@requires_cuda
@pytest.mark.parametrize(
    "used,sub", [(1024, 16), (1000, 16), (1000, 32), (777, 64), (64, 16), (1, 16)]
)
def test_int8_pooling_matches_the_bf16_route_bit_for_bit(used, sub):
    torch.manual_seed(used * 7 + sub)
    heads, rows = 3, 1024
    q = torch.randint(
        -127, 128, (1, heads, rows, HEAD_DIM), dtype=torch.int8, device="cuda"
    )
    k = torch.randint(
        -127, 128, (1, heads, rows, HEAD_DIM), dtype=torch.int8, device="cuda"
    )
    q[:, :, used:] = 0
    k[:, :, used:] = 0
    k_scale = torch.rand(1, heads, rows // 64, device="cuda") * 0.05 + 1e-3
    cells = -(-used // 64) * (64 // sub)
    q_factor = HEAD_DIM**-0.5 * LOG2E
    kwargs = dict(
        used=used, sub_q=sub, sub_k=sub, cells_q=cells, cells_k=cells, q_factor=q_factor
    )
    got_q, got_k = subblock_pool_int8(q, k, k_scale, **kwargs)
    want_q, want_k = _bf16_route_pooled(q, k, k_scale, **kwargs)
    assert torch.equal(got_q, want_q)
    assert torch.equal(got_k, want_k)


@requires_cuda
def test_block_tables_keep_every_key_block_where_the_mask_is_off():
    index = torch.tensor(
        [[[[3, 0], [1, 2], [2, 3]], [[0, 1], [3, 2], [1, 0]]]],
        dtype=torch.int32,
        device="cuda",
    )
    mask = torch.tensor([True, False, True], device="cuda")
    tables, nums = subblock_block_tables(index, mask, 4)
    assert nums.tolist() == [[[2, 4, 2], [2, 4, 2]]]
    for head in range(2):
        assert tables[0, head, 1].tolist() == [0, 1, 2, 3]
        for q_block in (0, 2):
            assert (
                tables[0, head, q_block, :2].tolist()
                == index[0, head, q_block].tolist()
            )
            assert tables[0, head, q_block, 2:].tolist() == [2, 3]


@requires_cuda
def test_block_tables_match_the_torch_construction_at_scale():
    torch.manual_seed(3)
    batch, heads, q_blocks, num_blocks, topk = 1, 4, 37, 96, 24
    index = torch.randint(
        0, num_blocks, (batch, heads, q_blocks, topk), dtype=torch.int32
    )
    mask = torch.rand(q_blocks) < 0.7
    tables = (
        torch.arange(num_blocks, dtype=torch.int32)
        .view(1, 1, 1, -1)
        .expand(batch, heads, q_blocks, -1)
        .clone()
    )
    tables[..., :topk] = torch.where(mask.view(1, 1, -1, 1), index, tables[..., :topk])
    nums = (
        torch.where(mask.view(1, 1, -1), topk, num_blocks)
        .expand(batch, heads, q_blocks)
        .to(torch.int32)
    )
    got_tables, got_nums = subblock_block_tables(index.cuda(), mask.cuda(), num_blocks)
    assert torch.equal(got_tables.cpu(), tables)
    assert torch.equal(got_nums.cpu(), nums)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
