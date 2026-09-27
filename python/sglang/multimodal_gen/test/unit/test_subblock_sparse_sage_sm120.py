# SPDX-License-Identifier: Apache-2.0
"""The SM120 SubBlock Sage impl on one GPU: local quantisation, Cake attention, gating.

Needs an SM120 GPU; skipped elsewhere. The first call compiles the JIT kernels.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="subblock_sparse_sage_sm120 needs an SM120 GPU",
)

HEADS = 8
HEAD_DIM = 128
SEQ = 1024
USED = 1000
SCALE = HEAD_DIM**-0.5
DEFAULT_CONFIG = {"sparsity": 0.5, "skip_first_steps": 10, "min_seq_len": 512}


class _FakeServerArgs:
    def __init__(self, config):
        self.attention_backend_config = config


class _DenseStub:
    def __init__(self):
        self.calls = 0

    def forward_varlen(self, query, key, value, **kwargs):
        self.calls += 1
        return torch.zeros_like(query)


def _impl(prefix="blocks.3.attn", **config):
    from sglang.multimodal_gen.runtime.layers.attention.backends.subblock_sparse_sage_sm120 import (
        SubBlockSparseSageSM120Impl,
    )

    merged = {**DEFAULT_CONFIG, **config}
    stub = _DenseStub()
    with (
        patch(
            "sglang.multimodal_gen.runtime.server_args.get_global_server_args",
            return_value=_FakeServerArgs(merged),
        ),
        patch.object(
            SubBlockSparseSageSM120Impl, "_build_dense_impl", return_value=stub
        ),
    ):
        impl = SubBlockSparseSageSM120Impl(
            num_heads=HEADS,
            head_size=HEAD_DIM,
            causal=False,
            softmax_scale=SCALE,
            prefix=prefix,
        )
    return impl, stub


def _step(step: int):
    class _Ctx:
        current_timestep = step

    return patch(
        "sglang.multimodal_gen.runtime.layers.attention.backends.subblock_sparse_attn.get_forward_context",
        return_value=_Ctx(),
    )


def _qkv(seq=SEQ, used=USED, seed=0):
    """Q/K/V as the DiT hands them over: views into one `[S, 3, H, D]` projection, zero tail."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    qkv = torch.randn(
        seq, 3, HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16, generator=g
    )
    qkv[used:] = 0
    return qkv[:, 0], qkv[:, 1], qkv[:, 2]


def _varlen(impl, q, k, v, used=USED, mask=None):
    total = q.shape[0]
    return impl.forward_varlen(
        q,
        k,
        v,
        cu_seqlens=torch.tensor([0, used, total], dtype=torch.int32, device="cuda"),
        max_seqlen=used,
        cu_seqlens_host=(0, used, total),
        first_segment_sparse_query_block_mask=mask,
    )


def _reference(q, k, v, used=USED):
    qf, kf, vf = (t[:used].transpose(0, 1).float() for t in (q, k, v))
    return (torch.softmax(qf @ kf.transpose(-1, -2) * SCALE, dim=-1) @ vf).transpose(
        0, 1
    )


def test_dense_step_matches_fp32_attention():
    impl, stub = _impl()
    q, k, v = _qkv()
    with _step(0):
        out = _varlen(impl, q, k, v)
    assert stub.calls == 0
    assert tuple(out.shape) == (SEQ, HEADS, HEAD_DIM) and out.dtype is torch.bfloat16
    want = _reference(q, k, v)
    got = out[:USED].float()
    rel = (got - want).norm() / want.norm()
    cos = torch.nn.functional.cosine_similarity(got.flatten(), want.flatten(), dim=0)
    assert rel < 6e-2 and cos > 0.999, (rel.item(), cos.item())
    assert not out[USED:].float().any()


def test_full_budget_step_stays_within_kernel_noise_of_dense():
    impl, _ = _impl(sparsity=1e-6)  # every key block survives the top-k
    q, k, v = _qkv(seed=1)
    with _step(0):
        dense = _varlen(impl, q, k, v).float()
    with _step(10):
        routed = _varlen(impl, q, k, v).float()
    rel = (routed - dense).norm() / dense.norm()
    assert rel < 1e-2, rel.item()


def test_mask_keeps_unselected_query_blocks_bitwise_dense():
    impl, _ = _impl(sparsity=0.5)
    q, k, v = _qkv(seed=2)
    q_blocks = -(-USED // 64)
    mask = torch.arange(q_blocks, device="cuda") % 2 == 0
    with _step(0):
        dense = _varlen(impl, q, k, v)
    with _step(10):
        routed = _varlen(impl, q, k, v, mask=mask)
    rows = torch.arange(USED, device="cuda")
    dense_rows = ~mask[rows // 64]
    assert torch.equal(routed[:USED][dense_rows], dense[:USED][dense_rows])
    assert not torch.equal(routed[:USED][~dense_rows], dense[:USED][~dense_rows])
    assert torch.isfinite(routed.float()).all()


def test_layers_outside_the_dit_and_short_documents_go_dense():
    refiner, stub = _impl(prefix="refiner.0.attn")
    q, k, v = _qkv()
    with _step(10):
        _varlen(refiner, q, k, v)
    assert stub.calls == 1
    impl, stub = _impl(min_seq_len=4096)
    with _step(10):
        _varlen(impl, q, k, v)
    assert stub.calls == 1
    impl, stub = _impl()
    with _step(10):
        impl.forward_varlen(
            q,
            k,
            v,
            cu_seqlens=torch.tensor(
                [0, 500, 1000, SEQ], dtype=torch.int32, device="cuda"
            ),
            max_seqlen=500,
            cu_seqlens_host=(0, 500, 1000, SEQ),
        )
    assert stub.calls == 1


def test_unaligned_sequence_is_rejected():
    impl, _ = _impl()
    q, k, v = _qkv(seq=1000, used=900)
    with _step(0), pytest.raises(ValueError, match="multiple of 128"):
        _varlen(impl, q, k, v, used=900)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
