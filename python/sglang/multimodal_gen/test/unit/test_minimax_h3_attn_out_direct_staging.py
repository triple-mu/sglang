# SPDX-License-Identifier: Apache-2.0
"""MINIMAX_H3_ATTN_OUT_DIRECT_STAGING: FA3 ``out=`` writes the ulysses output
IPC staging directly, eliminating the local merge copy.

The merged output bytes (out_proj's input) and the bytes delivered to the
peer must be torch.equal with the legacy path (FA3 -> local copy + NVLink
copy via ``_ipc_varlen_fast`` "output"), the strided FA3 store must land
in place without touching the peer's head columns, and ineligible
configurations (flag off, non-FA backend, FA4, odd split, world != 2)
must fall back to the copy path.
"""

from unittest.mock import patch

import pytest
import torch

from sglang.multimodal_gen.configs.models.dits.minimax_h3 import (
    MiniMaxH3DiTArchConfig,
)
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    maybe_init_distributed_environment_and_model_parallel,
    model_parallel_is_initialized,
)
from sglang.multimodal_gen.runtime.layers import usp
from sglang.multimodal_gen.runtime.layers.attention.backends import flash_attn
from sglang.multimodal_gen.runtime.models.dits import minimax_h3
from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import (
    MiniMaxH3Attention,
    _attn_out_direct_staging_enabled,
    _minimax_h3_attention_core_impl,
)
from sglang.multimodal_gen.test.single_test_file.component_accuracy.utils import (
    ensure_distributed_env_defaults,
)

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)

_ARCH_KWARGS = dict(
    hidden_size=512,
    num_attention_heads=8,
    attention_head_dim=128,
    rope_inv_freq_len=16,
)
_WORLD = 2
_SENTINEL_GROUP = object()


class _FakeIpcA2A:
    """Single-process stand-in for IPC_A2A: the 'peer' buffer is a local
    allocation, signal/wait are recorded no-ops."""

    def __init__(self, rank: int = 0):
        self.rank = rank
        self.calls = 0
        self.staging = {}
        self.events = []

    def get_staging(self, n_local, n_peer, dtype, group):
        key = (n_local, n_peer, dtype)
        if key not in self.staging:
            self.staging[key] = (
                torch.zeros(2, n_local, dtype=dtype, device="cuda"),
                torch.zeros(2, n_peer, dtype=dtype, device="cuda"),
            )
        return self.staging[key]

    def next_slot(self):
        slot = self.calls % 2
        self.calls += 1
        return slot

    def signal(self):
        self.events.append("signal")

    def wait(self):
        self.events.append("wait")


def _fake_ipc(monkeypatch, rank: int) -> _FakeIpcA2A:
    from sglang.multimodal_gen.runtime.distributed.device_communicators import (
        ipc_a2a,
    )

    fake = _FakeIpcA2A(rank)
    monkeypatch.setattr(ipc_a2a, "IPC_A2A", fake)
    monkeypatch.setattr(usp, "_ipc_ready_group", lambda: _SENTINEL_GROUP)
    monkeypatch.setattr(
        usp, "get_ulysses_parallel_world_size", lambda: _WORLD
    )
    return fake


def _ensure_single_process_parallel_runtime() -> None:
    if model_parallel_is_initialized():
        return
    ensure_distributed_env_defaults()
    maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=1)


@requires_cuda
@pytest.mark.parametrize("rank", [0, 1])
def test_exchange_semantics_match_legacy_output_branch(monkeypatch, rank):
    """finish() must deliver the same merged bytes and the same peer payload
    as ``_ipc_varlen_fast(x, ..., "output")`` for the same attention output."""
    torch.manual_seed(0)
    s_global, h_local, d = 512, 4, 128
    half = s_global // 2
    x = torch.randn(s_global, h_local, d, dtype=torch.bfloat16, device="cuda")

    fake_legacy = _fake_ipc(monkeypatch, rank)
    legacy = usp._ipc_varlen_fast(x[None], [half, half], 2, "output")
    assert legacy is not None
    legacy_peer_key = next(iter(fake_legacy.staging))
    legacy_peer_bytes = (
        fake_legacy.staging[legacy_peer_key][1][0]
        .narrow(0, 0, half * 2 * h_local * d)
        .view(half, 2 * h_local, d)[:, rank * h_local : (rank + 1) * h_local]
        .clone()
    )

    fake_direct = _fake_ipc(monkeypatch, rank)
    staging = usp._usp_attn_output_direct_begin(
        s_global=s_global, h_local=h_local, head_dim=d, dtype=x.dtype
    )
    assert staging is not None
    # Pass a foreign tensor: finish() must take the copy fallback and still
    # produce identical bytes (the direct-store case is covered below).
    merged = usp._usp_attn_output_direct_finish(staging, x)

    assert torch.equal(merged, legacy[0])
    assert torch.equal(staging.nvlink_dst, legacy_peer_bytes)
    assert fake_direct.events == ["signal", "wait"]
    assert fake_direct.calls == 1


@requires_cuda
@pytest.mark.parametrize("rank", [0, 1])
def test_fa3_strided_store_lands_in_staging_bitwise(monkeypatch, rank):
    """FA3 with ``out=`` must store in place into the strided staging view,
    torch.equal with its default-out run, without touching peer columns."""
    from sglang.kernels.ops.attention.flash_attention import (
        flash_attn_varlen_func,
    )

    torch.manual_seed(1)
    s_global, h_local, d = 1024, 4, 128
    half = s_global // 2
    q = torch.randn(s_global, h_local, d, dtype=torch.bfloat16, device="cuda")
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    cu = torch.tensor([0, s_global], dtype=torch.int32, device="cuda")

    def fa3(out=None):
        return flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cu,
            cu_seqlens_k=cu,
            max_seqlen_q=s_global,
            max_seqlen_k=s_global,
            softmax_scale=d**-0.5,
            causal=False,
            ver=3,
            out=out,
        )

    ref = fa3()
    fake = _fake_ipc(monkeypatch, rank)
    staging = usp._usp_attn_output_direct_begin(
        s_global=s_global, h_local=h_local, head_dim=d, dtype=q.dtype
    )
    peer_columns = slice((1 - rank) * h_local, (2 - rank) * h_local)
    out = fa3(out=staging.attn_out)
    assert out.data_ptr() == staging.attn_out.data_ptr()
    merged = usp._usp_attn_output_direct_finish(staging, out)

    local_columns = slice(rank * h_local, (rank + 1) * h_local)
    my_rows = slice(rank * half, (rank + 1) * half)
    assert torch.equal(merged[:, local_columns], ref[my_rows])
    assert torch.equal(staging.nvlink_src, ref[(1 - rank) * half :][:half])
    assert (merged[:, peer_columns] == 0).all(), "peer columns must stay unwritten"
    key = next(iter(fake.staging))
    assert (
        merged.untyped_storage().data_ptr()
        == fake.staging[key][0].untyped_storage().data_ptr()
    ), "merged result must be a staging view, not a copy"


def _build_attention(monkeypatch) -> MiniMaxH3Attention:
    _ensure_single_process_parallel_runtime()
    monkeypatch.setenv("MINIMAX_H3_QKV_NATIVE_ORDER", "0")
    arch = MiniMaxH3DiTArchConfig(**_ARCH_KWARGS)
    with torch.device("cuda"):
        return MiniMaxH3Attention(
            arch, None, prefix="blocks.0.attn", bcg_breakpoint=False
        )


def _run_core(attention, q, k, v, s_global):
    cu = torch.tensor([0, s_global], dtype=torch.int32, device="cuda")
    with torch.inference_mode():
        return _minimax_h3_attention_core_impl(
            attention,
            q,
            k,
            v,
            cu_seqlens=cu,
            cu_seqlens_host=(0, s_global),
            max_seqlen=s_global,
            ulysses_active=True,
        )


@requires_cuda
def test_core_direct_staging_matches_legacy_ipc_path(monkeypatch):
    """Full core path old vs new: FA3 -> staging -> merged view bytes (the
    out_proj input) must be torch.equal, flag ON vs OFF."""
    torch.manual_seed(2)
    attention = _build_attention(monkeypatch)
    heads = _ARCH_KWARGS["num_attention_heads"]
    h_local = heads // _WORLD
    d = _ARCH_KWARGS["attention_head_dim"]
    s_global = 768
    post_a2a = tuple(
        torch.randn(s_global, h_local, d, dtype=torch.bfloat16, device="cuda")
        for _ in range(3)
    )

    outs = {}
    for flag in ("0", "1"):
        monkeypatch.setenv("MINIMAX_H3_ATTN_OUT_DIRECT_STAGING", flag)
        fake = _fake_ipc(monkeypatch, rank=0)
        with patch.object(
            usp, "_usp_input_all_to_all_packed_qkv", lambda q, k, v: post_a2a
        ):
            outs[flag] = _run_core(attention, *post_a2a, s_global).clone()
        assert fake.calls == 1, "both paths must consume exactly one slot"

    assert outs["1"].shape == (s_global // _WORLD, heads, d)
    assert torch.equal(outs["1"], outs["0"])


@requires_cuda
def test_gate_fails_closed(monkeypatch):
    monkeypatch.setenv("MINIMAX_H3_ATTN_OUT_DIRECT_STAGING", "0")
    attention = _build_attention(monkeypatch)
    attention._attention_backend_enum = minimax_h3.AttentionBackendEnum.FA
    assert not _attn_out_direct_staging_enabled(attention)

    monkeypatch.setenv("MINIMAX_H3_ATTN_OUT_DIRECT_STAGING", "1")
    assert _attn_out_direct_staging_enabled(attention)

    attention._attention_backend_enum = minimax_h3.AttentionBackendEnum.TORCH_SDPA
    assert not _attn_out_direct_staging_enabled(attention)

    attention._attention_backend_enum = minimax_h3.AttentionBackendEnum.FA
    monkeypatch.setattr(flash_attn, "fa_ver", 4)
    assert not _attn_out_direct_staging_enabled(attention)


@requires_cuda
def test_begin_rejects_ineligible_splits(monkeypatch):
    _fake_ipc(monkeypatch, rank=0)
    assert (
        usp._usp_attn_output_direct_begin(
            s_global=511, h_local=4, head_dim=128, dtype=torch.bfloat16
        )
        is None
    ), "odd sequence split must fall back"
    monkeypatch.setattr(usp, "get_ulysses_parallel_world_size", lambda: 4)
    assert (
        usp._usp_attn_output_direct_begin(
            s_global=512, h_local=4, head_dim=128, dtype=torch.bfloat16
        )
        is None
    ), "world size != 2 must fall back"
