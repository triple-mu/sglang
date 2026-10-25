"""``_layout`` of the symmetric-memory all-to-all, checked on CPU by emulating
the pitched copies it describes with byte slices."""

import pytest
import torch

from sglang.multimodal_gen.runtime.distributed.device_communicators.symm_a2a import (
    _layout,
)


def _emulate(xs, split_dim):
    """Run the exchange that ``xs[rank]`` (one tensor per rank) describes."""
    world = len(xs)
    shapes = [_layout(x, world, r, split_dim)[0] for r, x in enumerate(xs)]
    outs = [
        torch.zeros(shape, dtype=x.dtype).flatten().view(torch.uint8)
        for shape, x in zip(shapes, xs)
    ]
    for rank, x in enumerate(xs):
        _, src, dst, width, height = _layout(x, world, rank, split_dim)
        raw = torch.empty(0, dtype=torch.uint8).set_(x.untyped_storage())
        base = x.storage_offset() * x.element_size()
        for peer in range(world):
            for bb in range(x.shape[0]):
                for i in range(height):
                    s0 = base + peer * src[0] + bb * src[2] + i * src[1]
                    d0 = dst[0] + bb * dst[2] + i * dst[1]
                    outs[peer][d0 : d0 + width] = raw[s0 : s0 + width]
    return [o.view(x.dtype).view(shape) for o, x, shape in zip(outs, xs, shapes)]


def _inputs(world, b, s, n, d, dtype, strided):
    xs = []
    for rank in range(world):
        if strided:
            # MiniMax-H3's fused projection: [.., 3, n, d] split into q/k/v views
            full = torch.arange(b * s * 3 * n * d, dtype=torch.float32) + rank * 1e6
            x = full.to(dtype).reshape(b, s, 3, n, d)[:, :, 1]
            assert not x.is_contiguous()
        else:
            x = (torch.arange(b * s * n * d, dtype=torch.float32) + rank * 1e6).to(dtype)
            x = x.reshape(b, s, n, d)
        xs.append(x)
    return xs


@pytest.mark.parametrize("world", [2, 4])
@pytest.mark.parametrize("b", [1, 2])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("strided", [False, True])
def test_seq_to_head_then_head_to_seq(world, b, dtype, strided):
    s, n, d = 6, 8, 4
    xs = _inputs(world, b, s, n, d, dtype, strided)
    gathered = _emulate(xs, 2)
    full = torch.cat(xs, dim=1)
    for rank, got in enumerate(gathered):
        assert got.shape == (b, s * world, n // world, d)
        torch.testing.assert_close(
            got, full[:, :, rank * n // world : (rank + 1) * n // world], rtol=0, atol=0
        )
    back = _emulate(gathered, 1)
    for rank, got in enumerate(back):
        assert got.shape == (b, s, n, d)
        torch.testing.assert_close(got, xs[rank], rtol=0, atol=0)


@pytest.mark.parametrize("world", [2, 4])
def test_head_to_seq_gathers_heads_in_rank_order(world):
    b, s, n, d = 2, 8, 3, 4
    xs = _inputs(world, b, s, n, d, torch.float32, strided=True)
    outs = _emulate(xs, 1)
    s_local = s // world
    for rank, got in enumerate(outs):
        expect = torch.cat(
            [x[:, rank * s_local : (rank + 1) * s_local] for x in xs], dim=2
        )
        torch.testing.assert_close(got, expect, rtol=0, atol=0)


def test_rejects_layouts_the_copy_engine_cannot_express():
    x = torch.zeros(2, 6, 8, 4)
    with pytest.raises(ValueError):
        _layout(x, 3, 0, 2)  # heads not divisible
    with pytest.raises(ValueError):
        _layout(x, 4, 0, 1)  # sequence not divisible
    with pytest.raises(ValueError):
        _layout(x.transpose(2, 3), 2, 0, 2)  # dim not unit-stride
    with pytest.raises(ValueError):
        _layout(x[:, :1].expand(2, 6, 8, 4), 2, 0, 2)  # broadcast rows
    with pytest.raises(ValueError):
        _layout(x[0], 2, 0, 2)  # 3D
    with pytest.raises(ValueError):
        _layout(x[:, :0], 2, 0, 2)  # empty
    # a sequence-offset slice and a token-strided view are fine
    _layout(x[:, 2:], 2, 0, 1)
    _layout(torch.zeros(1, 6, 3, 8, 4)[:, :, 1], 2, 1, 2)
