# SPDX-License-Identifier: Apache-2.0
"""CPU tests of the Veda attention backend's plumbing.

The sparse path needs a GPU, Miowtion and a predictor bundle; here the
runtime is faked so the dispatch, the dense fallbacks, the layout checks and
the head mapping can be exercised without them. The Miowtion-facing glue is
covered by test_veda_runtime.py.
"""

from types import SimpleNamespace

import pytest
import torch

from sglang.multimodal_gen.runtime.layers.attention.backends import (
    veda_attn_h3,
    veda_runtime,
)
from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionMetadata,
)
from sglang.multimodal_gen.runtime.layers.attention.backends.sdpa import SDPAImpl
from sglang.multimodal_gen.runtime.layers.attention.backends.veda_attn_h3 import (
    VedaAttentionBackend,
    VedaAttentionImpl,
    VedaAttentionMetadata,
    _int_list,
)
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum

HEAD_DIM = 128
SCALE = HEAD_DIM**-0.5


def _impl(prefix="blocks.7.attn", head_size=HEAD_DIM, **kwargs):
    return VedaAttentionImpl(
        num_heads=28,
        head_size=head_size,
        softmax_scale=head_size**-0.5,
        prefix=prefix,
        **kwargs,
    )


def _packed(seq_len=192, used=170, heads=2, seed=0):
    gen = torch.Generator().manual_seed(seed)
    q, k, v = (torch.randn(seq_len, heads, HEAD_DIM, generator=gen) for _ in range(3))
    cu = torch.tensor([0, used, seq_len], dtype=torch.int32)
    return q, k, v, cu, used


def test_backend_registration():
    assert VedaAttentionBackend.get_enum() is AttentionBackendEnum.VEDA_ATTN
    assert AttentionBackendEnum.VEDA_ATTN.is_sparse
    assert AttentionBackendEnum["VEDA_ATTN"] is AttentionBackendEnum.VEDA_ATTN
    assert VedaAttentionBackend.supports_packed_varlen()
    assert VedaAttentionBackend.get_impl_cls() is VedaAttentionImpl
    assert VedaAttentionBackend.get_builder_cls()().build(
        current_timestep=0,
        video_start=3,
        grid=(2, 4, 4),
        used=40,
        seq_len=64,
        num_steps=8,
    ) == VedaAttentionMetadata(0, 3, (2, 4, 4), 40, 64, 8)


def test_layer_parsing_and_sparse_capability():
    assert _impl("blocks.7.attn").layer_index == 7
    assert _impl("blocks.7.attn")._sparse_capable
    refiner = _impl("token_refiner.blocks.0.attn")
    assert refiner.layer_index is None and not refiner._sparse_capable
    # Non-DiT attention (a VAE under a global backend selection) is dense-only.
    assert not _impl("vae.attn", head_size=256)._sparse_capable
    with pytest.raises(ValueError):
        _impl("blocks.3.attn", head_size=256)
    with pytest.raises(ValueError):
        _impl("blocks.3.attn", causal=True)


def test_int_list_config_forms():
    assert _int_list(None) == [] and _int_list("") == []
    assert _int_list("0, 1,7") == [0, 1, 7]
    assert _int_list([3, "4"]) == [3, 4]
    assert _int_list(5) == [5]


def test_dense_fallback_matches_sdpa_without_forward_context():
    q, k, v, cu, used = _packed()
    impl = _impl()
    dense = SDPAImpl(
        num_heads=28, head_size=HEAD_DIM, causal=False, softmax_scale=SCALE
    )
    out = impl.forward_varlen(
        q, k, v, cu_seqlens=cu, max_seqlen=used, cu_seqlens_host=tuple(cu.tolist())
    )
    ref = dense.forward_varlen(
        q, k, v, cu_seqlens=cu, max_seqlen=used, cu_seqlens_host=tuple(cu.tolist())
    )
    assert torch.equal(out, ref)
    assert impl.calls == {"sparse": 0, "dense": 1}


def test_dense_fallback_for_foreign_metadata_and_refiner():
    q, k, v, cu, used = _packed()
    metadata = AttentionMetadata(current_timestep=3)
    with set_forward_context(current_timestep=3, attn_metadata=metadata):
        out = _impl().forward_varlen(q, k, v, cu_seqlens=cu, max_seqlen=used)
        refiner_out = _impl("token_refiner.blocks.1.attn").forward_varlen(
            q, k, v, cu_seqlens=cu, max_seqlen=used
        )
    dense = SDPAImpl(
        num_heads=28, head_size=HEAD_DIM, causal=False, softmax_scale=SCALE
    )
    ref = dense.forward_varlen(q, k, v, cu_seqlens=cu, max_seqlen=used)
    assert torch.equal(out, ref) and torch.equal(refiner_out, ref)


class _FakeStudent:
    """Stands in for Miowtion's SparseStudent: records the call, returns ones."""

    def __init__(self):
        self.calls = []

    def __call__(self, q, k, v, layer_index):
        self.calls.append((q.data_ptr(), q.shape, layer_index))
        return torch.ones_like(q)


def _fake_runtime(monkeypatch, dense_first_n_steps=0, dense_layers=(), num_layers=50):
    student = _FakeStudent()
    runtime = SimpleNamespace(
        dense_first_n_steps=dense_first_n_steps,
        dense_layers=frozenset(dense_layers),
        num_layers=num_layers,
        local=SimpleNamespace(heads=veda_runtime.HeadRange(42, 49)),
        _announced_sparse=True,
        _announced_extra_layer=False,
        student=lambda metadata: student,
        get_calls=[],
    )

    def get(cls, device, *, heads_per_tp, local_heads):
        runtime.get_calls.append((heads_per_tp, local_heads))
        return runtime

    monkeypatch.setattr(veda_attn_h3._VedaRuntime, "get", classmethod(get))
    return student, runtime


def _veda_metadata(seq_len=192, used=170, grid=(2, 4, 4)):
    return VedaAttentionMetadata(
        current_timestep=0,
        video_start=used - grid[0] * grid[1] * grid[2],
        grid=grid,
        used=used,
        seq_len=seq_len,
        num_steps=8,
    )


def test_local_head_range_follows_tp_then_ulysses(monkeypatch):
    from sglang.multimodal_gen.runtime.distributed import parallel_state

    monkeypatch.setattr(parallel_state, "get_tp_rank", lambda: 1)
    monkeypatch.setattr(parallel_state, "get_ulysses_parallel_rank", lambda: 2)
    # tp_rank * 28 + ulysses_rank * 7 + j
    assert veda_attn_h3._local_head_range(28, 7) == veda_runtime.HeadRange(42, 49)


def test_sparse_dispatch_calls_the_student_on_the_local_heads(monkeypatch):
    student, runtime = _fake_runtime(monkeypatch)
    q, k, v, cu, used = _packed(heads=7)
    impl = _impl("blocks.5.attn")  # 28 heads per TP rank
    with set_forward_context(current_timestep=4, attn_metadata=_veda_metadata()):
        out = impl.forward_varlen(
            q, k, v, cu_seqlens=cu, max_seqlen=used, cu_seqlens_host=tuple(cu.tolist())
        )
    assert impl.calls == {"sparse": 1, "dense": 0}
    assert runtime.get_calls == [(28, 7)]
    ((ptr, shape, layer),) = student.calls
    assert layer == 5 and shape == q.shape
    # Unit stride along head_dim is all the tile gather needs: no copy.
    assert ptr == q.data_ptr()
    assert out.shape == q.shape


def test_strided_head_views_are_passed_without_copy(monkeypatch):
    student, _ = _fake_runtime(monkeypatch)
    q8, k8, v8, cu, used = _packed(heads=8)
    q, k, v = q8[:, 2:9], k8[:, 2:9], v8[:, 2:9]
    assert q.stride(-1) == 1 and not q.is_contiguous()
    with set_forward_context(current_timestep=4, attn_metadata=_veda_metadata()):
        _impl("blocks.5.attn").forward_varlen(q, k, v, cu_seqlens=cu, max_seqlen=used)
    ((ptr, shape, _),) = student.calls
    assert ptr == q.data_ptr() and shape == q.shape


def test_layers_beyond_the_bundle_run_dense(monkeypatch):
    student, _ = _fake_runtime(monkeypatch, num_layers=4)
    q, k, v, cu, used = _packed()
    dense = SDPAImpl(
        num_heads=28, head_size=HEAD_DIM, causal=False, softmax_scale=SCALE
    )
    ref = dense.forward_varlen(q, k, v, cu_seqlens=cu, max_seqlen=used)
    with set_forward_context(current_timestep=4, attn_metadata=_veda_metadata()):
        out = _impl("blocks.4.attn").forward_varlen(
            q, k, v, cu_seqlens=cu, max_seqlen=used
        )
    assert torch.equal(out, ref) and student.calls == []


def test_dense_steps_and_dense_layers_take_the_sdpa_path(monkeypatch):
    student, _ = _fake_runtime(monkeypatch, dense_first_n_steps=2, dense_layers=(9,))
    q, k, v, cu, used = _packed()
    metadata = _veda_metadata()
    dense = SDPAImpl(
        num_heads=28, head_size=HEAD_DIM, causal=False, softmax_scale=SCALE
    )
    ref = dense.forward_varlen(q, k, v, cu_seqlens=cu, max_seqlen=used)
    with set_forward_context(current_timestep=1, attn_metadata=metadata):
        early = _impl("blocks.3.attn").forward_varlen(
            q, k, v, cu_seqlens=cu, max_seqlen=used
        )
    with set_forward_context(current_timestep=5, attn_metadata=metadata):
        dense_layer = _impl("blocks.9.attn").forward_varlen(
            q, k, v, cu_seqlens=cu, max_seqlen=used
        )
    assert torch.equal(early, ref) and torch.equal(dense_layer, ref)
    assert student.calls == []


def test_layout_mismatch_is_rejected(monkeypatch):
    _fake_runtime(monkeypatch)
    q, k, v, cu, used = _packed()
    metadata = _veda_metadata(used=used - 1)
    with set_forward_context(current_timestep=3, attn_metadata=metadata):
        with pytest.raises(ValueError, match="packed H3 layout"):
            _impl().forward_varlen(q, k, v, cu_seqlens=cu, max_seqlen=used)


def test_stage_metadata_from_packed_layout():
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.stages.denoising import (
        _build_veda_attn_metadata,
    )

    grid = (37, 24, 42)
    used = 164 + 414 + 37 * 24 * 42
    seq_len = (used + 63) // 64 * 64
    packed = {
        "stream_layout": {"target_shape": grid},
        "cu_seqlens": torch.tensor([0, used, seq_len], dtype=torch.int32),
    }
    pipeline_config = SimpleNamespace(
        resolve_transformer_attention_backend=lambda server_args: (
            AttentionBackendEnum.VEDA_ATTN
        )
    )
    server_args = SimpleNamespace(pipeline_config=pipeline_config)
    metadata = _build_veda_attn_metadata(
        server_args, packed=packed, ctx=SimpleNamespace(is_ref2va=False), num_steps=8
    )
    assert metadata == VedaAttentionMetadata(0, 578, grid, used, seq_len, 8)
    with pytest.raises(NotImplementedError):
        _build_veda_attn_metadata(
            server_args, packed=packed, ctx=SimpleNamespace(is_ref2va=True), num_steps=8
        )
    other = SimpleNamespace(
        pipeline_config=SimpleNamespace(
            resolve_transformer_attention_backend=lambda server_args: (
                AttentionBackendEnum.FA
            )
        )
    )
    assert (
        _build_veda_attn_metadata(
            other, packed=packed, ctx=SimpleNamespace(is_ref2va=False), num_steps=8
        )
        is None
    )
