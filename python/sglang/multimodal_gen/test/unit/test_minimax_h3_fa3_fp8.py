# SPDX-License-Identifier: Apache-2.0
"""MINIMAX_H3_FA3_FP8: FA3 fp8 e4m3 attention with per-head dynamic descales.

Quality-gated and default OFF: with the flag off (or a non-FA3 backend, or
the ring branch) the attention core must stay bitwise identical to plain
bf16 FA3. With the flag on, q/k/v are quantized per head after qknorm+rope,
FA3 consumes the fp8 payloads with [segments, heads] descales, the output
stays bf16, and the accuracy delta against bf16 attention is bounded and
recorded. Composes with the direct-staging output path bitwise.
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
    _fa3_fp8_attention_enabled,
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
    """Single-process stand-in for IPC_A2A (see the direct-staging test)."""

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
    monkeypatch.setattr(usp, "get_ulysses_parallel_world_size", lambda: _WORLD)
    return fake


def _ensure_single_process_parallel_runtime() -> None:
    if model_parallel_is_initialized():
        return
    ensure_distributed_env_defaults()
    maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=1)


def _build_attention(monkeypatch) -> MiniMaxH3Attention:
    _ensure_single_process_parallel_runtime()
    monkeypatch.setenv("MINIMAX_H3_QKV_NATIVE_ORDER", "0")
    arch = MiniMaxH3DiTArchConfig(**_ARCH_KWARGS)
    with torch.device("cuda"):
        return MiniMaxH3Attention(
            arch, None, prefix="blocks.0.attn", bcg_breakpoint=False
        )


def _run_core(attention, q, k, v, cu_host, *, ulysses_active=False):
    cu = torch.tensor(cu_host, dtype=torch.int32, device="cuda")
    max_seqlen = max(b - a for a, b in zip(cu_host[:-1], cu_host[1:]))
    with torch.inference_mode():
        return _minimax_h3_attention_core_impl(
            attention,
            q,
            k,
            v,
            cu_seqlens=cu,
            cu_seqlens_host=cu_host,
            max_seqlen=max_seqlen,
            ulysses_active=ulysses_active,
        )


@requires_cuda
def test_gate_fails_closed(monkeypatch):
    attention = _build_attention(monkeypatch)
    attention._attention_backend_enum = minimax_h3.AttentionBackendEnum.FA

    monkeypatch.setenv("MINIMAX_H3_FA3_FP8", "0")
    assert not _fa3_fp8_attention_enabled(attention)

    monkeypatch.setenv("MINIMAX_H3_FA3_FP8", "1")
    assert _fa3_fp8_attention_enabled(attention)

    for backend in (
        minimax_h3.AttentionBackendEnum.TORCH_SDPA,
        minimax_h3.AttentionBackendEnum.SUBBLOCK_SPARSE_ATTN,
    ):
        attention._attention_backend_enum = backend
        assert not _fa3_fp8_attention_enabled(attention)

    attention._attention_backend_enum = minimax_h3.AttentionBackendEnum.FA
    monkeypatch.setattr(flash_attn, "fa_ver", 4)
    assert not _fa3_fp8_attention_enabled(attention)


@requires_cuda
def test_gate_off_is_bitwise_plain_bf16_fa3(monkeypatch):
    """Flag off must leave the core exactly on the bf16 FA3 kernel."""
    from sglang.kernels.ops.attention.flash_attention import flash_attn_varlen_func

    torch.manual_seed(0)
    attention = _build_attention(monkeypatch)
    monkeypatch.setenv("MINIMAX_H3_FA3_FP8", "0")
    s, h, d = 1024, 8, 128
    q = torch.randn(s, h, d, dtype=torch.bfloat16, device="cuda")
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    out = _run_core(attention, q, k, v, (0, s))
    cu = torch.tensor([0, s], dtype=torch.int32, device="cuda")
    ref = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=cu,
        cu_seqlens_k=cu,
        max_seqlen_q=s,
        max_seqlen_k=s,
        softmax_scale=d**-0.5,
        causal=False,
        ver=3,
    )
    assert torch.equal(out, ref)


@requires_cuda
@pytest.mark.parametrize(
    "cu_host",
    [
        (0, 41984),
        (0, 37296, 41392, 41984),  # H3 packed video+audio+text segment mix
    ],
)
def test_fp8_accuracy_bound_at_production_shape(monkeypatch, cu_host):
    """fp8 vs bf16 attention at the fl2va production shape [41984, 28, 128]:
    output must stay bf16 and the delta within the documented envelope
    (probe on H200: max diff ~3e-4 at sigma=0.3, cosine ~0.9993)."""
    torch.manual_seed(1)
    _ensure_single_process_parallel_runtime()
    monkeypatch.setenv("MINIMAX_H3_QKV_NATIVE_ORDER", "0")
    arch = MiniMaxH3DiTArchConfig(
        hidden_size=512,
        num_attention_heads=28,
        attention_head_dim=128,
        rope_inv_freq_len=16,
    )
    with torch.device("cuda"):
        attention = MiniMaxH3Attention(
            arch, None, prefix="blocks.0.attn", bcg_breakpoint=False
        )
    s = cu_host[-1]
    q = torch.randn(s, 28, 128, dtype=torch.bfloat16, device="cuda") * 0.3
    k = torch.randn_like(q) * 0.3
    v = torch.randn_like(q)

    monkeypatch.setenv("MINIMAX_H3_FA3_FP8", "0")
    ref = _run_core(attention, q, k, v, cu_host)
    monkeypatch.setenv("MINIMAX_H3_FA3_FP8", "1")
    out = _run_core(attention, q, k, v, cu_host)

    assert out.dtype == torch.bfloat16
    diff = (out.float() - ref.float()).abs()
    cosine = torch.nn.functional.cosine_similarity(
        out.float().flatten(), ref.float().flatten(), dim=0
    )
    print(
        f"\nfa3 fp8 vs bf16 @ cu={cu_host}: max {diff.max().item():.6f} "
        f"mean {diff.mean().item():.8f} cosine {cosine.item():.6f}"
    )
    assert not torch.equal(out, ref), "flag on must actually change the kernel"
    assert diff.max().item() < 5e-2
    assert cosine.item() > 0.999


@requires_cuda
def test_fp8_composes_with_direct_staging_bitwise(monkeypatch):
    """fp8 + direct staging: FA3's fp8 kernel writes the bf16 staging via
    ``out=``; merged bytes must equal the copy-path fp8 run bitwise."""
    torch.manual_seed(2)
    attention = _build_attention(monkeypatch)
    monkeypatch.setenv("MINIMAX_H3_FA3_FP8", "1")
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
            outs[flag] = _run_core(
                attention, *post_a2a, (0, s_global), ulysses_active=True
            ).clone()
        assert fake.calls == 1

    assert outs["1"].shape == (s_global // _WORLD, heads, d)
    assert torch.equal(outs["1"], outs["0"])


@requires_cuda
def test_ring_branch_stays_bf16(monkeypatch):
    """The ring branch never quantizes: with the flag on, the ring path must
    still receive bf16 q/k/v (fp8 fail-closed for ring)."""
    torch.manual_seed(3)
    attention = _build_attention(monkeypatch)
    monkeypatch.setenv("MINIMAX_H3_FA3_FP8", "1")
    s, h, d = 256, 8, 128
    q = torch.randn(s, h, d, dtype=torch.bfloat16, device="cuda")
    seen = {}

    def fake_ring(q, k, v, *, attn_impl, real_seq_len, ring_ws):
        seen["dtypes"] = (q.dtype, k.dtype, v.dtype)
        return torch.zeros_like(q)

    cu = torch.tensor([0, s], dtype=torch.int32, device="cuda")
    with (
        patch.object(minimax_h3, "_ring_attention_varlen", fake_ring),
        patch.object(minimax_h3, "get_ring_ctx", return_value=(2, 0)),
        torch.inference_mode(),
    ):
        _minimax_h3_attention_core_impl(
            attention,
            q,
            q.clone(),
            q.clone(),
            cu_seqlens=cu,
            cu_seqlens_host=(0, s),
            max_seqlen=s,
            ulysses_active=False,
            ring_active=True,
        )
    assert seen["dtypes"] == (torch.bfloat16,) * 3


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
