# SPDX-License-Identifier: Apache-2.0
"""MINIMAX_H3_QKV_NATIVE_ORDER: native qkv row order + dual-slab GEMM contract.

The flag keeps the checkpoint's per-head [q|k|v] qkv row interleave resident
and writes the Ulysses destination-major A2A send buffer directly from one
GEMM per destination rank, making the pack kernel a no-op. The send-buffer
bytes must match the legacy path (reorder-on-load + single GEMM + qknorm+rope
+ pack) bit-for-bit, the ulysses=1 path must produce identical attention
output, and unsupported combinations (quantized checkpoints, TP > 1,
already-reordered checkpoints, LoRA) must fail closed.
"""

from types import SimpleNamespace
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
from sglang.multimodal_gen.runtime.models.dits import minimax_h3
from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import MiniMaxH3Attention
from sglang.multimodal_gen.runtime.pipelines_core.lora.pipeline import LoRAPipeline
from sglang.multimodal_gen.test.single_test_file.component_accuracy.utils import (
    ensure_distributed_env_defaults,
)

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)

# Production head geometry with a narrow hidden width: head_dim/rope_dim drive
# the fused qknorm+rope kernel variant, num_heads drives the slab split.
_ARCH_KWARGS = dict(
    hidden_size=512,
    num_attention_heads=8,
    attention_head_dim=128,
    rope_inv_freq_len=16,
)
_WORLD = 2


def _ensure_single_process_parallel_runtime() -> None:
    if model_parallel_is_initialized():
        return
    ensure_distributed_env_defaults()
    maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=1)


def _build_attention(monkeypatch, *, native: bool) -> MiniMaxH3Attention:
    _ensure_single_process_parallel_runtime()
    monkeypatch.setenv("MINIMAX_H3_QKV_NATIVE_ORDER", "1" if native else "0")
    arch = MiniMaxH3DiTArchConfig(**_ARCH_KWARGS)
    with torch.device("cuda"):
        attention = MiniMaxH3Attention(
            arch, None, prefix="blocks.0.attn", bcg_breakpoint=False
        )
    return attention


def _load_pair(monkeypatch, checkpoint_qkv, out_proj_weight):
    legacy = _build_attention(monkeypatch, native=False)
    native = _build_attention(monkeypatch, native=True)
    for attention in (legacy, native):
        weight = attention.qkv_proj.weight
        weight.weight_loader(weight, checkpoint_qkv)
        attention.out_proj.weight.data.copy_(out_proj_weight)
    return legacy, native


def _rope_inputs(arch_heads_tokens):
    total = arch_heads_tokens
    rope_dim = 6 * _ARCH_KWARGS["rope_inv_freq_len"]
    cos_sin_cache = torch.randn(total, rope_dim, dtype=torch.bfloat16, device="cuda")
    positions = torch.arange(total, dtype=torch.int32, device="cuda")
    return cos_sin_cache, positions


class _CaptureCore:
    """Stands in for the attention core; records pre-exchange q/k/v/prepacked."""

    def __init__(self):
        self.q = self.k = self.v = self.prepacked = None

    def __call__(self, attention, q, k, v, *, prepacked_qkv=None, **kwargs):
        self.q, self.k, self.v, self.prepacked = q, k, v, prepacked_qkv
        rows = prepacked_qkv.shape[1] if q is None else q.shape[0]
        return torch.zeros(
            rows,
            attention.num_heads,
            attention.head_dim,
            dtype=torch.bfloat16,
            device="cuda",
        )


@requires_cuda
def test_native_order_loader_keeps_checkpoint_rows(monkeypatch):
    torch.manual_seed(0)
    hidden = _ARCH_KWARGS["hidden_size"]
    rows = 3 * _ARCH_KWARGS["num_attention_heads"] * _ARCH_KWARGS["attention_head_dim"]
    checkpoint = torch.randn(rows, hidden, dtype=torch.bfloat16, device="cuda")
    native = _build_attention(monkeypatch, native=True)
    weight = native.qkv_proj.weight
    weight.weight_loader(weight, checkpoint)
    assert native._qkv_rows_interleaved
    assert weight.qkv_rows_interleaved
    assert torch.equal(weight.data, checkpoint)


@requires_cuda
def test_dual_slab_send_buffer_matches_legacy_pack_bytes(monkeypatch):
    """Flag-ON staging bytes == flag-OFF qkv GEMM -> qknorm+rope -> pack."""
    from sglang.kernels.ops.diffusion import pack_qkv_destination_major

    torch.manual_seed(1)
    hidden = _ARCH_KWARGS["hidden_size"]
    heads = _ARCH_KWARGS["num_attention_heads"]
    head_dim = _ARCH_KWARGS["attention_head_dim"]
    rows = 3 * heads * head_dim
    total = 1536
    checkpoint = torch.randn(rows, hidden, dtype=torch.bfloat16, device="cuda")
    out_w = torch.randn(hidden, heads * head_dim, dtype=torch.bfloat16, device="cuda")
    legacy, native = _load_pair(monkeypatch, checkpoint, out_w)

    x = torch.randn(total, hidden, dtype=torch.bfloat16, device="cuda")
    cos_sin_cache, positions = _rope_inputs(total)
    cu = torch.tensor([0, total], dtype=torch.int32, device="cuda")
    forward_kwargs = dict(
        rope_cache=(cos_sin_cache, positions),
        cu_seqlens=cu,
        cu_seqlens_host=(0, total),
        max_seqlen=total,
        ulysses_active=True,
    )

    core = _CaptureCore()
    with (
        torch.inference_mode(),
        patch.object(minimax_h3, "_minimax_h3_attention_core_impl", core),
        patch.object(minimax_h3, "get_ulysses_ctx", return_value=(_WORLD, 0)),
    ):
        native(x, **forward_kwargs)
        assert core.prepacked is not None, "dual-slab path did not engage"
        assert core.q is None and core.k is None and core.v is None
        native_staging = core.prepacked.clone()
        legacy(x, **forward_kwargs)
        assert core.prepacked is None
        legacy_staging = pack_qkv_destination_major(
            core.q.contiguous(), core.k.contiguous(), core.v.contiguous(), _WORLD
        )

    assert native_staging.shape == (_WORLD, total, heads // _WORLD, 3 * head_dim)
    assert torch.equal(native_staging, legacy_staging)


@requires_cuda
def test_ulysses1_forward_matches_legacy_bitwise(monkeypatch):
    """Single-GPU path: interleaved strided views through FA must be bit-equal."""
    torch.manual_seed(2)
    hidden = _ARCH_KWARGS["hidden_size"]
    heads = _ARCH_KWARGS["num_attention_heads"]
    head_dim = _ARCH_KWARGS["attention_head_dim"]
    rows = 3 * heads * head_dim
    total = 640
    checkpoint = torch.randn(rows, hidden, dtype=torch.bfloat16, device="cuda")
    out_w = torch.randn(hidden, heads * head_dim, dtype=torch.bfloat16, device="cuda")
    legacy, native = _load_pair(monkeypatch, checkpoint, out_w)

    x = torch.randn(total, hidden, dtype=torch.bfloat16, device="cuda")
    cos_sin_cache, positions = _rope_inputs(total)
    cu = torch.tensor([0, total], dtype=torch.int32, device="cuda")
    forward_kwargs = dict(
        rope_cache=(cos_sin_cache, positions),
        cu_seqlens=cu,
        cu_seqlens_host=(0, total),
        max_seqlen=total,
        ulysses_active=False,
    )
    with torch.inference_mode():
        out_legacy = legacy(x, **forward_kwargs)
        out_native = native(x, **forward_kwargs)
    assert torch.equal(out_native, out_legacy)


@requires_cuda
def test_default_auto_mode_enables_native_order(monkeypatch):
    """With the env var unset, a supported config gets native order by default."""
    _ensure_single_process_parallel_runtime()
    monkeypatch.delenv("MINIMAX_H3_QKV_NATIVE_ORDER", raising=False)
    arch = MiniMaxH3DiTArchConfig(**_ARCH_KWARGS)
    with torch.device("cuda"):
        attention = MiniMaxH3Attention(
            arch, None, prefix="blocks.0.attn", bcg_breakpoint=False
        )
    assert attention._qkv_rows_interleaved
    assert attention.qkv_proj.weight.qkv_rows_interleaved


@requires_cuda
def test_default_auto_mode_falls_back_on_quantized_checkpoints(monkeypatch):
    from sglang.multimodal_gen.runtime.layers.quantization.fp8 import Fp8Config

    _ensure_single_process_parallel_runtime()
    monkeypatch.delenv("MINIMAX_H3_QKV_NATIVE_ORDER", raising=False)
    arch = MiniMaxH3DiTArchConfig(**_ARCH_KWARGS)
    with torch.device("meta"):
        attention = MiniMaxH3Attention(arch, Fp8Config(), prefix="blocks.0.attn")
    assert not attention._qkv_rows_interleaved


@requires_cuda
def test_default_auto_mode_falls_back_on_already_reordered_checkpoints(monkeypatch):
    _ensure_single_process_parallel_runtime()
    monkeypatch.delenv("MINIMAX_H3_QKV_NATIVE_ORDER", raising=False)
    arch = MiniMaxH3DiTArchConfig(checkpoint_uses_diffusers_layout=True, **_ARCH_KWARGS)
    with torch.device("meta"):
        attention = MiniMaxH3Attention(arch, None, prefix="blocks.0.attn")
    assert not attention._qkv_rows_interleaved


def test_default_auto_mode_falls_back_on_startup_lora(monkeypatch):
    monkeypatch.delenv("MINIMAX_H3_QKV_NATIVE_ORDER", raising=False)
    attention = MiniMaxH3Attention.__new__(MiniMaxH3Attention)
    torch.nn.Module.__init__(attention)
    attention.tp_size = 1
    with (
        patch.object(minimax_h3.current_platform, "is_cuda", return_value=True),
        patch.object(minimax_h3, "_startup_lora_configured", return_value=True),
    ):
        reason = attention._qkv_native_order_unsupported_reason(None)
    assert reason is not None and "LoRA" in reason


def test_native_order_mode_resolution(monkeypatch):
    monkeypatch.delenv("MINIMAX_H3_QKV_NATIVE_ORDER", raising=False)
    assert minimax_h3._qkv_native_order_mode() == "auto"
    monkeypatch.setenv("MINIMAX_H3_QKV_NATIVE_ORDER", "1")
    assert minimax_h3._qkv_native_order_mode() == "strict"
    monkeypatch.setenv("MINIMAX_H3_QKV_NATIVE_ORDER", "0")
    assert minimax_h3._qkv_native_order_mode() == "off"


@requires_cuda
def test_flag_rejects_quantized_checkpoints(monkeypatch):
    from sglang.multimodal_gen.runtime.layers.quantization.fp8 import Fp8Config

    _ensure_single_process_parallel_runtime()
    monkeypatch.setenv("MINIMAX_H3_QKV_NATIVE_ORDER", "1")
    arch = MiniMaxH3DiTArchConfig(**_ARCH_KWARGS)
    with torch.device("meta"), pytest.raises(ValueError, match="quantized"):
        MiniMaxH3Attention(arch, Fp8Config(), prefix="blocks.0.attn")


@requires_cuda
def test_flag_rejects_already_reordered_checkpoints(monkeypatch):
    _ensure_single_process_parallel_runtime()
    monkeypatch.setenv("MINIMAX_H3_QKV_NATIVE_ORDER", "1")
    arch = MiniMaxH3DiTArchConfig(checkpoint_uses_diffusers_layout=True, **_ARCH_KWARGS)
    with torch.device("meta"), pytest.raises(ValueError, match="interleaved"):
        MiniMaxH3Attention(arch, None, prefix="blocks.0.attn")


def test_flag_rejects_tp_sharding():
    attention = MiniMaxH3Attention.__new__(MiniMaxH3Attention)
    torch.nn.Module.__init__(attention)
    attention.tp_size = 2
    attention._qkv_rows_interleaved = False
    with (
        patch.object(minimax_h3.current_platform, "is_cuda", return_value=True),
        pytest.raises(ValueError, match="tp_size"),
    ):
        attention._install_qkv_native_order(None)


def test_lora_pipeline_rejects_interleaved_qkv_weight():
    layer = torch.nn.Linear(4, 4, bias=False)
    layer.weight.qkv_rows_interleaved = True
    module = torch.nn.Module()
    module.qkv_proj = layer
    fake = SimpleNamespace(
        modules={"transformer": module},
        is_target_layer=lambda name: bool(name),
    )
    with pytest.raises(ValueError, match="interleaved"):
        LoRAPipeline._reject_lora_on_packed_weights(fake)


@patch(
    "sglang.multimodal_gen.runtime.layers.usp.get_ulysses_parallel_world_size",
    return_value=2,
)
def test_prepacked_exchange_matches_packed_exchange_semantics(_):
    """The prepacked exchange must consume the exact send layout the packed
    exchange produces and emit identical receive-side q/k/v."""
    from sglang.multimodal_gen.runtime.layers.usp import (
        _usp_input_all_to_all_packed_qkv,
        _usp_input_all_to_all_prepacked_qkv,
    )

    torch.manual_seed(3)
    seq, heads, dim = 3, 4, 2
    q = torch.randn(seq, heads, dim)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    sent = []

    def fake_all_to_all(x, role=None):
        sent.append(x.clone())
        return x

    with patch(
        "sglang.multimodal_gen.runtime.layers.usp._usp_all_to_all_single",
        side_effect=fake_all_to_all,
    ):
        packed_out = _usp_input_all_to_all_packed_qkv(q, k, v)
        prepacked_out = _usp_input_all_to_all_prepacked_qkv(sent[0])

    assert torch.equal(sent[0], sent[1])
    for reference, actual in zip(packed_out, prepacked_out, strict=True):
        assert torch.equal(actual, reference)
