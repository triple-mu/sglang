# SPDX-License-Identifier: Apache-2.0
"""fp8 serving-path wiring for the producer-fused per-token quant kernels.

Under ``--quantization fp8``, fc2 and out_proj must accept a producer-
quantized ``(fp8_payload, per_token_scale)`` tuple whose payload/scale are
bitwise equal to the separated chain (eager producer + standalone
``sgl_per_token_quant_fp8``), and whose GEMM output is bitwise equal to
feeding the bf16 producer output into the layer. The kill-switch envs and
the bf16 path must stay untouched.
"""

from unittest.mock import patch

import pytest
import torch

from sglang.kernels.ops.quantization.fp8_kernel import sglang_per_token_quant_fp8
from sglang.multimodal_gen.configs.models.dits.minimax_h3 import (
    MiniMaxH3DiTArchConfig,
)
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    maybe_init_distributed_environment_and_model_parallel,
    model_parallel_is_initialized,
)
from sglang.multimodal_gen.runtime.layers import usp
from sglang.multimodal_gen.runtime.layers.linear import RowParallelLinear
from sglang.multimodal_gen.runtime.layers.quantization.fp8 import Fp8Config
from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import (
    MiniMaxH3Attention,
    MiniMaxH3MLP,
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
    ffn_hidden_size=1024,
    rope_inv_freq_len=16,
)
_WORLD = 2
_SENTINEL_GROUP = object()


def _ensure_single_process_parallel_runtime() -> None:
    if model_parallel_is_initialized():
        return
    ensure_distributed_env_defaults()
    maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=1)


def _process_fp8_weights(*layers) -> None:
    for layer in layers:
        layer.quant_method.process_weights_after_loading(layer)


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


# ---------------------------------------------------------------------------
# Fp8LinearMethod: per-token pre-quantized tuple input
# ---------------------------------------------------------------------------


@requires_cuda
def test_fp8_apply_tuple_matches_separated_chain():
    """apply((payload, scale)) must be bitwise equal to apply(bf16 x): srt's
    prequant hook asserts per-tensor scales, so the per-row tuple form
    dispatches through the diffusion-side branch and must reproduce the
    same GEMM dispatch."""
    _ensure_single_process_parallel_runtime()
    torch.manual_seed(0)
    with torch.device("cuda"):
        layer = RowParallelLinear(
            1024,
            512,
            bias=False,
            input_is_parallel=True,
            params_dtype=torch.bfloat16,
            quant_config=Fp8Config(),
            prefix="blocks.0.mlp.fc2",
        )
    layer.weight.data.normal_()
    _process_fp8_weights(layer)
    method = layer.quant_method
    assert method.supports_per_token_prequant_input(layer)

    x = torch.randn(1797, 1024, dtype=torch.bfloat16, device="cuda")
    expected = method.apply(layer, x)
    actual = method.apply(layer, sglang_per_token_quant_fp8(x))
    assert actual.dtype == expected.dtype == torch.bfloat16
    assert torch.equal(actual, expected)


@requires_cuda
def test_fp8_apply_tuple_rejects_mismatched_scale():
    _ensure_single_process_parallel_runtime()
    with torch.device("cuda"):
        layer = RowParallelLinear(
            256,
            256,
            bias=False,
            input_is_parallel=True,
            params_dtype=torch.bfloat16,
            quant_config=Fp8Config(),
            prefix="blocks.0.attn.out_proj",
        )
    layer.weight.data.normal_()
    _process_fp8_weights(layer)
    payload, _ = sglang_per_token_quant_fp8(
        torch.randn(8, 256, dtype=torch.bfloat16, device="cuda")
    )
    wrong_scale = torch.ones(3, 1, dtype=torch.float32, device="cuda")
    with pytest.raises(RuntimeError, match="per-token pre-quantized"):
        layer.quant_method.apply(layer, (payload, wrong_scale))


# ---------------------------------------------------------------------------
# Q3b: fc1 activation -> fused silu_mul + quant -> fc2
# ---------------------------------------------------------------------------


def _build_mlp(quant_config):
    _ensure_single_process_parallel_runtime()
    arch = MiniMaxH3DiTArchConfig(**_ARCH_KWARGS)
    with torch.device("cuda"):
        mlp = MiniMaxH3MLP(arch, quant_config, prefix="blocks.0.mlp")
    for layer in (mlp.fc1, mlp.fc2):
        layer.weight.data.normal_()
    return mlp


@requires_cuda
def test_mlp_fused_silu_quant_matches_separated_fp8_chain(monkeypatch):
    """Flag ON (fused kernel -> tuple -> fc2) vs flag OFF (eager silu chain +
    fc2's standalone quant) must produce bitwise-identical MLP output."""
    torch.manual_seed(1)
    mlp = _build_mlp(Fp8Config())
    _process_fp8_weights(mlp.fc1, mlp.fc2)
    assert not mlp.reuse_fc1_activation
    assert mlp.fc2_accepts_prequant_fp8

    x = torch.randn(
        1797, _ARCH_KWARGS["hidden_size"], dtype=torch.bfloat16, device="cuda"
    )
    with torch.inference_mode():
        monkeypatch.setenv("MINIMAX_H3_FUSED_SILU_QUANT", "0")
        separated = mlp(x)
        monkeypatch.setenv("MINIMAX_H3_FUSED_SILU_QUANT", "1")
        fused = mlp(x)
    assert torch.equal(fused, separated)


@requires_cuda
def test_mlp_bf16_path_untouched():
    mlp = _build_mlp(None)
    assert mlp.reuse_fc1_activation
    assert not mlp.fc2_accepts_prequant_fp8


# ---------------------------------------------------------------------------
# Q3c: Ulysses output merge -> fused quant -> out_proj
# ---------------------------------------------------------------------------


@requires_cuda
@pytest.mark.parametrize("rank", [0, 1])
def test_usp_ipc_merge_quant_matches_legacy_merge_plus_quant(monkeypatch, rank):
    """IPC form: fused (payload, scale) must be bitwise equal to quantizing
    the legacy ``_ipc_varlen_fast`` merged rows, for both rank column
    orders, with the same slot/signal sequence and the same peer payload."""
    torch.manual_seed(2)
    s_global, h_local, d = 512, 4, 128
    half = s_global // 2
    x = torch.randn(s_global, h_local, d, dtype=torch.bfloat16, device="cuda")

    fake_legacy = _fake_ipc(monkeypatch, rank)
    legacy = usp._ipc_varlen_fast(x[None], [half, half], 2, "output")
    assert legacy is not None
    expected = sglang_per_token_quant_fp8(legacy[0].reshape(half, 2 * h_local * d))
    legacy_key = next(iter(fake_legacy.staging))
    legacy_peer_bytes = (
        fake_legacy.staging[legacy_key][1][0]
        .narrow(0, 0, half * 2 * h_local * d)
        .view(half, 2 * h_local, d)[:, rank * h_local : (rank + 1) * h_local]
        .clone()
    )

    fake = _fake_ipc(monkeypatch, rank)
    fused = usp._usp_output_all_to_all_quant_fp8(x)
    assert fused is not None
    assert torch.equal(fused[0].view(torch.uint8), expected[0].view(torch.uint8))
    assert torch.equal(fused[1], expected[1])
    assert fake.events == ["signal", "wait"]
    assert fake.calls == 1
    key = next(iter(fake.staging))
    fused_peer_bytes = (
        fake.staging[key][1][0]
        .narrow(0, 0, half * 2 * h_local * d)
        .view(half, 2 * h_local, d)[:, rank * h_local : (rank + 1) * h_local]
    )
    assert torch.equal(fused_peer_bytes, legacy_peer_bytes)


@requires_cuda
def test_usp_nccl_merge_quant_matches_legacy_merge_plus_quant(monkeypatch):
    """NCCL form (IPC unavailable): fused result must be bitwise equal to the
    unfused exchange + standalone quant through the same (mocked) collective."""
    torch.manual_seed(3)
    s_global, h_local, d = 512, 4, 128
    x = torch.randn(s_global, h_local, d, dtype=torch.bfloat16, device="cuda")
    monkeypatch.setattr(usp, "get_ulysses_parallel_world_size", lambda: _WORLD)
    monkeypatch.setattr(usp, "_ipc_ready_group", lambda: None)

    with patch.object(usp, "_usp_all_to_all_single", lambda t, role=None: t):
        legacy = usp._usp_output_all_to_all(x[None], head_dim=2)[0]
        expected = sglang_per_token_quant_fp8(
            legacy.reshape(s_global // _WORLD, _WORLD * h_local * d)
        )
        fused = usp._usp_output_all_to_all_quant_fp8(x)
    assert fused is not None
    assert torch.equal(fused[0].view(torch.uint8), expected[0].view(torch.uint8))
    assert torch.equal(fused[1], expected[1])


@requires_cuda
def test_usp_merge_quant_ineligible_inputs_fall_back(monkeypatch):
    monkeypatch.setattr(usp, "get_ulysses_parallel_world_size", lambda: 1)
    x = torch.randn(64, 4, 128, dtype=torch.bfloat16, device="cuda")
    assert usp._usp_output_all_to_all_quant_fp8(x) is None
    monkeypatch.setattr(usp, "get_ulysses_parallel_world_size", lambda: _WORLD)
    assert usp._usp_output_all_to_all_quant_fp8(x.half()) is None
    odd = torch.randn(63, 4, 128, dtype=torch.bfloat16, device="cuda")
    assert usp._usp_output_all_to_all_quant_fp8(odd) is None


# ---------------------------------------------------------------------------
# Attention-core composition under fp8 out_proj
# ---------------------------------------------------------------------------


def _build_fp8_attention(monkeypatch) -> MiniMaxH3Attention:
    _ensure_single_process_parallel_runtime()
    monkeypatch.setenv("MINIMAX_H3_QKV_NATIVE_ORDER", "0")
    arch = MiniMaxH3DiTArchConfig(**_ARCH_KWARGS)
    with torch.device("cuda"):
        attention = MiniMaxH3Attention(
            arch, Fp8Config(), prefix="blocks.0.attn", bcg_breakpoint=False
        )
    heads = _ARCH_KWARGS["num_attention_heads"]
    head_dim = _ARCH_KWARGS["attention_head_dim"]
    hidden = _ARCH_KWARGS["hidden_size"]
    checkpoint = torch.randn(
        3 * heads * head_dim, hidden, dtype=torch.bfloat16, device="cuda"
    )
    attention.qkv_proj.weight.weight_loader(attention.qkv_proj.weight, checkpoint)
    attention.out_proj.weight.data.normal_()
    _process_fp8_weights(attention.qkv_proj, attention.out_proj)
    return attention


def _run_attention(attention, x, s_global):
    total = x.shape[0]
    rope_dim = 6 * _ARCH_KWARGS["rope_inv_freq_len"]
    cos_sin_cache = torch.randn(total, rope_dim, dtype=torch.bfloat16, device="cuda")
    positions = torch.arange(total, dtype=torch.int32, device="cuda")
    cu = torch.tensor([0, s_global], dtype=torch.int32, device="cuda")
    with torch.inference_mode():
        return attention(
            x,
            rope_cache=(cos_sin_cache, positions),
            cu_seqlens=cu,
            cu_seqlens_host=(0, s_global),
            max_seqlen=s_global,
            ulysses_active=True,
        )


@requires_cuda
def test_attention_fp8_merge_quant_matches_unfused_and_direct_staging(monkeypatch):
    """Full fp8 attention path: (a) fused merge+quant vs unfused merge with
    out_proj's internal quant must be bitwise equal (direct staging off), and
    (b) direct staging on (which bypasses the merge, so the fused form never
    engages) must also be bitwise equal -- the D2 + Q3c combination verdict."""
    torch.manual_seed(4)
    attention = _build_fp8_attention(monkeypatch)
    assert attention.out_proj_accepts_prequant_fp8
    heads = _ARCH_KWARGS["num_attention_heads"]
    h_local = heads // _WORLD
    d = _ARCH_KWARGS["attention_head_dim"]
    s_global = 768
    total = s_global // _WORLD
    x = torch.randn(
        total, _ARCH_KWARGS["hidden_size"], dtype=torch.bfloat16, device="cuda"
    )
    post_a2a = tuple(
        torch.randn(s_global, h_local, d, dtype=torch.bfloat16, device="cuda")
        for _ in range(3)
    )

    outs = {}
    cases = {
        "unfused": ("0", "0"),
        "fused": ("1", "0"),
        "direct_staging": ("0", "1"),
    }
    for name, (merge_quant, direct) in cases.items():
        monkeypatch.setenv("MINIMAX_H3_FUSED_MERGE_QUANT", merge_quant)
        monkeypatch.setenv("MINIMAX_H3_ATTN_OUT_DIRECT_STAGING", direct)
        _fake_ipc(monkeypatch, rank=0)
        with patch.object(
            usp, "_usp_input_all_to_all_packed_qkv", lambda q, k, v: post_a2a
        ):
            outs[name] = _run_attention(attention, x, s_global).clone()

    assert outs["fused"].shape == (total, _ARCH_KWARGS["hidden_size"])
    assert torch.equal(outs["fused"], outs["unfused"])
    assert torch.equal(outs["direct_staging"], outs["unfused"])


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
