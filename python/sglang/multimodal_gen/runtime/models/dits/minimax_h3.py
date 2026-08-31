# SPDX-License-Identifier: Apache-2.0
"""MiniMax H3 packed-token DiT.

Native SGLang implementation of the MiniMax H3 audio-video DiT. The forward
contract accepts packed inference keyword arguments and returns packed logits.
"""

from __future__ import annotations

import math
import os
from collections import defaultdict
from collections.abc import Iterable, Iterator
from contextlib import ExitStack
from typing import Any, Callable

import torch
import torch.nn as nn
from safetensors.torch import safe_open
from torch.distributed.tensor import DTensor

from sglang.kernels.ops.activation.activation import (
    silu_and_mul_with_activation_rounding_,
)
from sglang.kernels.ops.diffusion import (
    can_use_fused_inplace_qknorm_rope,
    can_use_fused_silu_mul_per_token_quant_fp8,
    can_use_gate_residual_rmsnorm_indexed_scale_shift_bitexact_cuda,
    can_use_per_head_quant_fp8,
    can_use_rmsnorm_indexed_scale_shift_bitexact_cuda,
    fused_inplace_qknorm_rope,
    fused_silu_mul_per_token_quant_fp8,
    gate_residual_rmsnorm_indexed_scale_shift_,
    gate_residual_rmsnorm_indexed_scale_shift_bitexact_,
    indexed_gate_bf16,
    indexed_gate_bf16_,
    indexed_scale_shift_bf16_,
    indexed_scale_shift_bf16_to_fp32,
    per_head_quant_fp8,
    rmsnorm_indexed_scale_shift,
    rmsnorm_indexed_scale_shift_bitexact,
)
from sglang.kernels.ops.layernorm.norm import fused_inplace_qknorm
from sglang.multimodal_gen import envs
from sglang.multimodal_gen.configs.models.dits.minimax_h3 import (
    MINIMAX_H3_ADALN_MODALITY_NUM,
    MINIMAX_H3_PACKED_SEQUENCE_ALIGNMENT,
    MiniMaxH3DiTArchConfig,
    MiniMaxH3DiTConfig,
)
from sglang.multimodal_gen.configs.models.fsdp import is_block
from sglang.multimodal_gen.runtime.distributed import (
    get_tp_world_size,
    tensor_model_parallel_all_gather,
)
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_ring_ctx,
    get_tp_rank,
    get_ulysses_ctx,
    get_world_rank,
)
from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionRequirements,
)
from sglang.multimodal_gen.runtime.layers.attention.selector import (
    claim_deferred_component_attn_backend,
    get_attn_backend,
    get_global_forced_attn_backend,
)
from sglang.multimodal_gen.runtime.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
    UnquantizedLinearMethod,
)
from sglang.multimodal_gen.runtime.layers.quantization.configs.base_config import (
    QuantizationConfig,
)
from sglang.multimodal_gen.runtime.layers.usp import _ring_attention_varlen
from sglang.multimodal_gen.runtime.loader.utils import get_param_names_mapping
from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload import (
    LayerwiseOffloadableModuleMixin,
    is_layerwise_offloaded_module,
)
from sglang.multimodal_gen.runtime.models.dits.base import BaseDiT
from sglang.multimodal_gen.runtime.models.dits.minimax_h3_adaln_persist import (
    adaln_plan_key,
    load_adaln_plans,
    make_adaln_persist_spec,
    save_adaln_plans,
)
from sglang.multimodal_gen.runtime.platforms import (
    AttentionBackendEnum,
    current_platform,
)
from sglang.multimodal_gen.runtime.server_args import get_global_server_args
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    eager_on_graph,
)

logger = init_logger(__name__)

_ARCH_DEFAULTS = MiniMaxH3DiTArchConfig()


def _diffusers_h3_checkpoint(
    iterator: Iterable[tuple[str, torch.Tensor]],
) -> Iterator[tuple[str, torch.Tensor]]:
    """Map Diffusers H3 names/layout to the fused native checkpoint layout."""
    mapping = get_param_names_mapping(_ARCH_DEFAULTS.param_names_mapping)
    pending: dict[str, dict[int, torch.Tensor]] = defaultdict(dict)

    for source_name, tensor in iterator:
        target_name, merge_index, merge_count = mapping(source_name)

        # Diffusers SwiGLU stores [value, gate]; the native fused MLP consumes
        # [gate, value]. Packed GPTQ tensors carry output channels on dim 1.
        if ".ff.net.0.proj." in source_name:
            output_dim = (
                1 if source_name.endswith((".qweight", ".qzeros", ".scales")) else 0
            )
            value, gate = tensor.chunk(2, dim=output_dim)
            tensor = torch.cat((gate, value), dim=output_dim)

        if merge_index is None:
            yield target_name, tensor
            continue

        assert merge_count is not None
        pending[target_name][merge_index] = tensor
        if len(pending[target_name]) != merge_count:
            continue

        merge_dim = 1 if target_name.endswith((".qweight", ".qzeros", ".scales")) else 0
        yield target_name, torch.cat(
            [pending[target_name][index] for index in range(merge_count)],
            dim=merge_dim,
        )
        del pending[target_name]

    if pending:
        incomplete = ", ".join(sorted(pending))
        raise ValueError(f"Incomplete Diffusers H3 fused parameters: {incomplete}")


_BF16_DTYPE = torch.bfloat16
_FP32_DTYPE = torch.float32
_MPS_MLP_TOKEN_CHUNK_SIZE = 128
# keep MPS activation chunks below the allocator high-watermark; CUDA keeps
# its fused full-sequence projection
_MPS_QKV_PROJECTION_TOKEN_CHUNK_SIZE = 128
_MPS_ATTENTION_QUERY_TOKEN_CHUNK_SIZE = 128

_MPS_EMBED_WEIGHT_PREFIXES = (
    "condition_proj",
    "video_patch_proj",
    "audio_patch_proj",
    "time_embedder",
    "token_refiner.final_norm",
)

_MINIMAX_H3_FP32_PARAM_NAMES_IN_MODEL_ORDER = (
    "video_patch_proj.weight",
    "video_patch_proj.bias",
    "audio_patch_proj.weight",
    "audio_patch_proj.bias",
    "time_embedder.proj_in.weight",
    "time_embedder.proj_in.bias",
    "time_embedder.proj_out.weight",
    "time_embedder.proj_out.bias",
    "final_layer.video_out.weight",
    "final_layer.video_out.bias",
    "final_layer.audio_out.weight",
    "final_layer.audio_out.bias",
)
MINIMAX_H3_FP32_PARAM_NAMES = frozenset(_MINIMAX_H3_FP32_PARAM_NAMES_IN_MODEL_ORDER)
MINIMAX_H3_FP32_BUFFER_NAMES = frozenset({"rope.inv_freq"})


def _required_kwarg(kwargs: dict[str, Any], key: str) -> Any:
    if key not in kwargs or kwargs[key] is None:
        raise ValueError(f"MiniMaxH3DiTModel.forward requires kwarg {key!r}")
    return kwargs[key]


# The exhaustive keyword contract of MiniMaxH3DiTModel.forward. Anything not
# listed here is rejected with a TypeError before any tensor work starts.
_FORWARD_SUPPORTED_KWARGS = frozenset(
    {
        "x",
        "audio_x",
        "img_position_ids",
        "rope_cache",
        "unique_timesteps",
        "inverse_indices",
        "update_mask",
        "update_audio_mask",
        "token_tags",
        "block_token_tags",
        "block_combined_indices",
        "subblock_sparse_query_block_mask",
        "skip_mask_out_condition",
        "prompt_embeds",
        "refined_prompt_embeds_length",
        "img_pos_info",
        "audio_pos_info",
        "text_pos_info",
        "img_pos_for_infer_output_info",
        "local_embedding_layout",
        "packed_seq_params",
        "refiner_packed_seq_params",
    }
)


def _reorder_grouped_qkv_to_qkv(
    weight: torch.Tensor,
    *,
    num_query_groups: int,
    heads_per_group: int,
    head_dim: int,
) -> torch.Tensor:
    per_group = (heads_per_group + 2) * head_dim
    expected_out = num_query_groups * per_group
    if weight.shape[0] != expected_out:
        raise ValueError(
            "qkv weight has incompatible output dim for grouped checkpoint layout: "
            f"got {tuple(weight.shape)}, expected first dim {expected_out}."
        )

    rest_shape = weight.shape[1:]
    grouped = weight.reshape(num_query_groups, per_group, *rest_shape)
    q, k, v = torch.split(
        grouped,
        [heads_per_group * head_dim, head_dim, head_dim],
        dim=1,
    )
    return torch.cat(
        [
            q.reshape(num_query_groups * heads_per_group * head_dim, *rest_shape),
            k.reshape(num_query_groups * head_dim, *rest_shape),
            v.reshape(num_query_groups * head_dim, *rest_shape),
        ],
        dim=0,
    )


def _install_qkv_row_reorder(
    param: torch.Tensor,
    reorder: Callable[[torch.Tensor], torch.Tensor],
    qkv_rows: int,
) -> None:
    """Reorder a per-output-row qkv parameter the same way its rows are reordered.

    Applied to quantization metadata rather than the weight itself. Anything whose
    leading dim is not the checkpoint's qkv row count is passed through: per-tensor
    scales are scalars, and a swizzled block-scale layout is not row-indexed.
    """

    def _maybe_reorder(loaded_weight: torch.Tensor) -> torch.Tensor:
        if loaded_weight.dim() >= 2 and loaded_weight.shape[0] == qkv_rows:
            return reorder(loaded_weight)
        return loaded_weight

    base_loader = (
        param._weight_loader
        if hasattr(param, "_weight_loader")
        else param.weight_loader
    )

    def _weight_loader(p: torch.Tensor, loaded_weight: torch.Tensor) -> None:
        base_loader(p, _maybe_reorder(loaded_weight))

    if hasattr(param, "_weight_loader"):
        param._weight_loader = _weight_loader
    else:
        param.weight_loader = _weight_loader
    param.rank_local_weight_transform = _maybe_reorder


def _copy_grouped_qkv_tp_shard(
    param: torch.Tensor,
    loaded_weight: torch.Tensor,
    *,
    num_query_groups: int,
    head_dim: int,
    tp_rank: int,
    tp_size: int,
) -> bool:
    """Copy a dense MHA checkpoint directly into its TP-local Q/K/V rows."""
    if (
        tp_size <= 0
        or not 0 <= tp_rank < tp_size
        or num_query_groups % tp_size
        or getattr(param, "output_dim", None) != 0
        or getattr(param, "is_sharded_weight", False)
        or getattr(param, "packed_dim", None) is not None
        or param.dtype != loaded_weight.dtype
        or param.dtype not in (_BF16_DTYPE, torch.float8_e4m3fn)
        or not param.is_contiguous()
        or not loaded_weight.is_contiguous()
    ):
        return False

    expected_rows = num_query_groups * 3 * head_dim
    local_groups = num_query_groups // tp_size
    rest_shape = loaded_weight.shape[1:]
    if loaded_weight.shape[0] != expected_rows or tuple(param.shape) != (
        3 * local_groups * head_dim,
        *rest_shape,
    ):
        return False

    grouped = loaded_weight.view(num_query_groups, 3, head_dim, *rest_shape)
    grouped = grouped.narrow(0, tp_rank * local_groups, local_groups)
    target = param.data.view(3, local_groups, head_dim, *rest_shape)
    for index in range(3):
        target[index].copy_(grouped[:, index])
    return True


def _norm(size: int, *, eps: float, dtype: torch.dtype = _BF16_DTYPE) -> nn.RMSNorm:
    # RMSNorm uses fp32 accumulation with bf16 inputs and outputs.
    # torch.nn.RMSNorm upcasts reduced-precision inputs for the variance
    # reduction, matching that accumulation semantic.
    return nn.RMSNorm(size, eps=eps, dtype=dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = torch.chunk(x, 2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _modulate_scale_shift(
    x: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    indices: torch.Tensor,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Apply indexed affine modulation, reusing disposable CUDA BF16 input."""
    # Apply per-index affine modulation: x * (1 + scale[idx]) + shift[idx].
    if (
        x.is_cuda
        and x.dtype == _BF16_DTYPE
        and dtype == _BF16_DTYPE
        and shift.dtype == _BF16_DTYPE
        and scale.dtype == _BF16_DTYPE
        and x.is_contiguous()
    ):
        return indexed_scale_shift_bf16_(x, shift, scale, indices)
    return (
        x * (1.0 + scale.index_select(0, indices)) + shift.index_select(0, indices)
    ).to(dtype)


def _modulate_scale_shift_to_fp32(
    x: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    """Indexed affine modulation fused with the final-layer fp32 widening.

    One kernel reads the bf16 activation once and stores fp32; bitwise equal
    to ``_modulate_scale_shift(..., dtype=bf16)`` followed by ``.to(fp32)``
    (the fused kernel keeps every bf16 rounding boundary and the bf16 -> fp32
    widening is exact).
    """
    if (
        x.is_cuda
        and x.dtype == _BF16_DTYPE
        and shift.dtype == _BF16_DTYPE
        and scale.dtype == _BF16_DTYPE
        and x.is_contiguous()
        and not torch.compiler.is_compiling()
    ):
        return indexed_scale_shift_bf16_to_fp32(x, shift, scale, indices)
    return _modulate_scale_shift(x, shift, scale, indices, dtype=_BF16_DTYPE).to(
        _FP32_DTYPE
    )


def _modulate_gate(
    x: torch.Tensor,
    gate: torch.Tensor,
    other: torch.Tensor,
    indices: torch.Tensor,
    *,
    dtype: torch.dtype,
    allow_inplace: bool = True,
) -> torch.Tensor:
    """Apply an indexed gated residual, optionally reusing the input buffer."""
    # Apply the per-index gated residual: x + gate[idx] * other.
    if (
        x.is_cuda
        and x.dtype == _BF16_DTYPE
        and dtype == _BF16_DTYPE
        and gate.dtype == _BF16_DTYPE
        and other.dtype == _BF16_DTYPE
        and x.is_contiguous()
        and other.is_contiguous()
    ):
        if allow_inplace:
            return indexed_gate_bf16_(x, gate, other, indices)
        return indexed_gate_bf16(x, gate, other, indices)
    return (x + gate.index_select(0, indices) * other).to(dtype)


def _fused_norm_modulate(
    x: torch.Tensor,
    weight: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    shift: torch.Tensor,
    indices: torch.Tensor,
    *,
    eps: float,
) -> torch.Tensor:
    """Fused-adaLN Plan A site: a ``(gamma, scale)`` pair runs the bitexact
    kernel (falling back to the equivalent eager chain per call), a merged
    fp32 ``w_eff`` tensor the near-lossless one."""
    if isinstance(weight, tuple):
        gamma, scale = weight
        if can_use_rmsnorm_indexed_scale_shift_bitexact_cuda(
            x, gamma, scale, shift, indices
        ):
            return rmsnorm_indexed_scale_shift_bitexact(
                x, gamma, scale, shift, indices, eps=eps
            )
        h = nn.functional.rms_norm(x, gamma.shape, gamma, eps)
        return _modulate_scale_shift(h, shift, scale, indices, dtype=_BF16_DTYPE)
    return rmsnorm_indexed_scale_shift(x, weight, shift, indices, eps=eps)


def _fused_gate_norm_modulate_(
    residual: torch.Tensor,
    update: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    shift: torch.Tensor,
    indices: torch.Tensor,
    *,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused-adaLN Plan B site (see ``_fused_norm_modulate``); ``residual`` is
    updated in place on every branch and returned as ``(out, residual)``."""
    if isinstance(weight, tuple):
        gamma, scale = weight
        if can_use_gate_residual_rmsnorm_indexed_scale_shift_bitexact_cuda(
            residual, update, gate, gamma, scale, shift, indices
        ):
            return gate_residual_rmsnorm_indexed_scale_shift_bitexact_(
                residual, update, gate, gamma, scale, shift, indices, eps=eps
            )
        residual = _modulate_gate(residual, gate, update, indices, dtype=_BF16_DTYPE)
        h = nn.functional.rms_norm(residual, gamma.shape, gamma, eps)
        return (
            _modulate_scale_shift(h, shift, scale, indices, dtype=_BF16_DTYPE),
            residual,
        )
    return gate_residual_rmsnorm_indexed_scale_shift_(
        residual, update, gate, weight, shift, indices, eps=eps
    )


def _linear_accepts_per_token_prequant_fp8(linear) -> bool:
    """Whether this linear's quant method takes a producer-quantized
    ``(fp8_payload, per_token_scale)`` tuple (the diffusion fp8 W8A8 path)."""
    if linear.quant_config is None:
        return False
    from sglang.multimodal_gen.runtime.layers.quantization.fp8 import (
        Fp8LinearMethod,
    )

    method = linear.quant_method
    return isinstance(
        method, Fp8LinearMethod
    ) and method.supports_per_token_prequant_input(linear)


def _silu_mul(hidden: torch.Tensor, *, reuse_input: bool) -> torch.Tensor:
    if (
        reuse_input
        and hidden.is_cuda
        and hidden.dtype == _BF16_DTYPE
        and hidden.is_contiguous()
        and hidden.shape[-1] % 16 == 0
    ):
        return silu_and_mul_with_activation_rounding_(hidden)
    gate, up = hidden.chunk(2, dim=-1)
    return nn.functional.silu(gate) * up


def _apply_qk_norm(
    q: torch.Tensor,
    k: torch.Tensor,
    q_norm: nn.RMSNorm,
    k_norm: nn.RMSNorm,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if (
        q.is_cuda
        and q.dtype == _BF16_DTYPE
        and q.dtype == k.dtype == q_norm.weight.dtype == k_norm.weight.dtype
        and q.stride(-1) == k.stride(-1) == 1
        and q.stride(-2) == k.stride(-2) == head_dim
        and q_norm.eps == k_norm.eps
        and not torch.compiler.is_compiling()
    ):
        fused_inplace_qknorm(
            q,
            k,
            q_norm.weight,
            k_norm.weight,
            eps=q_norm.eps,
            head_dim=head_dim,
        )
        return q, k
    return q_norm(q), k_norm(k)


class MiniMaxH3Rope(nn.Module):
    """3D rope over (t, h, w); rotates 96 of 128 head dims (rotary_percent 0.75).

    Frequency layout concatenates temporal, height, and width embeddings twice,
    with 16 frequencies per axis (inv_freq = base^-(arange(0,32,2)/32)).
    """

    def __init__(self, inv_freq_len: int) -> None:
        super().__init__()
        self.register_buffer(
            "inv_freq",
            torch.empty(inv_freq_len, dtype=_FP32_DTYPE),
            persistent=True,
        )

    def forward(self, img_position_ids: torch.Tensor) -> torch.Tensor:
        """img_position_ids: [1, S, 3] (t, h, w) -> freqs [S, rot_dim=96]."""
        if img_position_ids.dim() != 3 or img_position_ids.shape[0] != 1:
            raise ValueError(
                "img_position_ids must be [1, S, 3], got "
                f"{list(img_position_ids.shape)}"
            )
        pos = img_position_ids[0].to(_FP32_DTYPE)  # [S, 3]
        per_axis = pos.unsqueeze(-1) * self.inv_freq.view(1, 1, -1)  # [S, 3, 16]
        t_f, h_f, w_f = per_axis.unbind(dim=1)  # each [S, 16]
        half = torch.cat((t_f, h_f, w_f), dim=-1)  # [S, 48]
        return torch.cat((half, half), dim=-1)  # [S, 96]


def _rope_cos_sin_cache(freqs: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
    """Build the activation-dtype cos|sin cache for fused Q/K RoPE."""
    half = freqs.shape[-1] // 2
    return (
        torch.cat(
            (torch.cos(freqs[:, :half]), torch.sin(freqs[:, :half])),
            dim=-1,
        )
        .to(dtype=dtype, copy=False)
        .contiguous()
    )


def _apply_rope_cos_sin(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Rotate the first cached RoPE dims; pass the remaining head dims through."""
    rot_dim = cos.shape[-1]
    x_rot, x_pass = x[..., :rot_dim], x[..., rot_dim:]
    x_rot = (x_rot * cos) + (_rotate_half(x_rot) * sin)
    return torch.cat((x_rot, x_pass), dim=-1)


def _apply_rope_qk(
    q: torch.Tensor,
    k: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not q.is_cuda:
        half = cos_sin_cache.shape[-1] // 2
        cos_half, sin_half = cos_sin_cache.split(half, dim=-1)
        cos = torch.cat((cos_half, cos_half), dim=-1).unsqueeze(1)
        sin = torch.cat((sin_half, sin_half), dim=-1).unsqueeze(1)
        return (
            _apply_rope_cos_sin(q, cos, sin),
            _apply_rope_cos_sin(k, cos, sin),
        )

    from sgl_kernel import rotary_embedding as apply_sgl_kernel_rotary_embedding

    apply_sgl_kernel_rotary_embedding(
        positions,
        q.view(q.shape[0], -1),
        k.view(k.shape[0], -1),
        q.shape[-1],
        cos_sin_cache,
        True,
    )
    return q, k


def _apply_rope(
    x: torch.Tensor,
    cos_sin_cache: torch.Tensor,
) -> torch.Tensor:
    """Apply the eager (non-CUDA) H3 RoPE path to one Q or K tensor."""
    half = cos_sin_cache.shape[-1] // 2
    cos_half, sin_half = cos_sin_cache.split(half, dim=-1)
    cos = torch.cat((cos_half, cos_half), dim=-1).unsqueeze(1)
    sin = torch.cat((sin_half, sin_half), dim=-1).unsqueeze(1)
    return _apply_rope_cos_sin(x, cos, sin)


class MiniMaxH3TimeEmbedder(nn.Module):
    def __init__(
        self,
        arch: MiniMaxH3DiTArchConfig,
        *,
        prefix: str,
    ) -> None:
        super().__init__()
        self.frequency_embedding_size = arch.timestep_input_dim
        self.proj_in = ColumnParallelLinear(
            arch.timestep_input_dim,
            arch.time_embed_hidden_size,
            bias=True,
            gather_output=False,
            params_dtype=_FP32_DTYPE,
            quant_config=None,
            prefix=f"{prefix}.proj_in",
        )
        self.proj_out = RowParallelLinear(
            arch.time_embed_hidden_size,
            arch.time_embed_dim,
            bias=True,
            input_is_parallel=True,
            params_dtype=_FP32_DTYPE,
            quant_config=None,
            prefix=f"{prefix}.proj_out",
        )
        self.register_buffer("_frequency_cache", None, persistent=False)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: [M] -> [M, time_embed_dim] fp32.

        The sinusoidal embedding stays fp32 throughout and concatenates cosine
        values before sine values.
        """
        half = self.frequency_embedding_size // 2
        freqs = self._frequency_cache
        if freqs is None or freqs.device != t.device:
            # Construct this on the execution device once so the values keep
            # the established CUDA numerics without repeating arange/exp on
            # every denoise step.
            freqs = torch.exp(
                -math.log(10000.0)
                * torch.arange(half, dtype=_FP32_DTYPE, device=t.device)
                / half
            )
            self._frequency_cache = freqs
        args = t.to(_FP32_DTYPE)[:, None] * freqs[None]
        t_freq = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        hidden, _ = self.proj_in(t_freq)
        hidden = nn.functional.silu(hidden)
        out, _ = self.proj_out(hidden)
        return out


def _qkv_native_order_mode() -> str:
    """Resolve MINIMAX_H3_QKV_NATIVE_ORDER into one of three modes.

    'off': explicitly disabled. 'auto': default-on -- unsupported configs fall
    back to the legacy reordered rows with one log line. 'strict': explicitly
    enabled -- unsupported configs raise instead of falling back (debugging).
    """
    if not envs.MINIMAX_H3_QKV_NATIVE_ORDER:
        return "off"
    if "MINIMAX_H3_QKV_NATIVE_ORDER" in os.environ:
        return "strict"
    return "auto"


def _startup_lora_configured() -> bool:
    """Whether serving was launched with a startup LoRA adapter (--lora-path).

    Adapter rows assume the reordered [q_all, k_all, v_all] qkv layout, so
    native order must fall back before weights load. Dynamic ``set_lora`` on a
    server launched without --lora-path is still rejected at conversion time
    (see LoRAPipeline._reject_lora_on_packed_weights).
    """
    try:
        server_args = get_global_server_args()
    except ValueError:
        # Direct model construction without serving config (unit tests,
        # library callers): no startup adapter exists.
        return False
    return server_args.lora_path is not None


_qkv_native_order_fallback_logged = False


def _log_qkv_native_order_fallback(reason: str) -> None:
    global _qkv_native_order_fallback_logged
    if _qkv_native_order_fallback_logged:
        return
    _qkv_native_order_fallback_logged = True
    logger.info(
        "MiniMax H3 qkv native order stays on the legacy reordered rows for "
        "this run: %s (set MINIMAX_H3_QKV_NATIVE_ORDER=1 to make this an "
        "error instead)",
        reason,
    )


def _attn_out_direct_staging_enabled(attention: MiniMaxH3Attention) -> bool:
    """Whether FA3 may write the ulysses output IPC staging via ``out=``.

    Only the FA v3 path is validated for a strided ``out=`` destination; FA4
    and other backends fall back to the copy-based output all-to-all.
    """
    if not envs.MINIMAX_H3_ATTN_OUT_DIRECT_STAGING:
        return False
    if attention._attention_backend_enum is not AttentionBackendEnum.FA:
        return False
    from sglang.multimodal_gen.runtime.layers.attention.backends import (
        flash_attn as _flash_attn_backend,
    )

    return _flash_attn_backend.fa_ver == 3


def _fa3_fp8_attention_enabled(attention: MiniMaxH3Attention) -> bool:
    """Whether FA3 may run its fp8 e4m3 path for this attention call.

    Quality-gated and default OFF (MINIMAX_H3_FA3_FP8). Only the FA v3
    backend takes fp8 payloads with per-head descales; ring and sparse
    branches never reach the quantization site, FA4 and other backends fail
    closed here.
    """
    if not envs.MINIMAX_H3_FA3_FP8:
        return False
    if torch.compiler.is_compiling():
        return False
    if attention._attention_backend_enum is not AttentionBackendEnum.FA:
        return False
    from sglang.multimodal_gen.runtime.layers.attention.backends import (
        flash_attn as _flash_attn_backend,
    )

    return _flash_attn_backend.fa_ver == 3


def _quantize_qkv_per_head_fp8(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    num_segments: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]] | None:
    """Per-head fp8 e4m3 payloads + FA3 ``[segments, heads]`` descales.

    Runs after qknorm+rope (post all-to-all under Ulysses, so each rank owns
    its heads' full sequence and the per-head amax needs no cross-rank
    coordination). None when any input is ineligible; callers then keep bf16.
    """
    if not (
        can_use_per_head_quant_fp8(q)
        and can_use_per_head_quant_fp8(k)
        and can_use_per_head_quant_fp8(v)
    ):
        return None
    payloads = []
    descales = []
    for tensor in (q, k, v):
        payload, scale = per_head_quant_fp8(tensor)
        payloads.append(payload)
        descales.append(scale.view(1, -1).repeat(num_segments, 1))
    return payloads[0], payloads[1], payloads[2], tuple(descales)


def _usp_merge_quant_fp8_enabled(attention: MiniMaxH3Attention) -> bool:
    """Whether the Ulysses output merge may emit out_proj's input pre-quantized."""
    return attention.out_proj_accepts_prequant_fp8 and envs.MINIMAX_H3_FUSED_MERGE_QUANT


def _minimax_h3_attention_core_impl(
    attention: MiniMaxH3Attention,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    cu_seqlens: torch.Tensor,
    cu_seqlens_host: tuple[int, ...] | None,
    max_seqlen: int,
    ulysses_active: bool,
    subblock_sparse_query_block_mask: torch.Tensor | None = None,
    ring_active: bool = False,
    prepacked_qkv: torch.Tensor | None = None,
) -> torch.Tensor:
    """Dynamic varlen attention and Ulysses/Ring collectives.

    This is the narrow BCG break point: projections, normalization, RoPE,
    residuals, and MLPs remain captured while the dynamic packed attention
    kernel and sequence-parallel collectives execute eagerly.

    ``prepacked_qkv`` is the destination-major send buffer written directly by
    the native-row-order qkv projection; q/k/v are None in that case and the
    pack step is skipped.
    """

    if ulysses_active:
        from sglang.multimodal_gen.runtime.layers.usp import (
            _usp_attn_output_direct_begin,
            _usp_attn_output_direct_finish,
            _usp_input_all_to_all_packed_qkv,
            _usp_input_all_to_all_prepacked_qkv,
            _usp_output_all_to_all,
            _usp_output_all_to_all_quant_fp8,
        )

        if prepacked_qkv is not None:
            q, k, v = _usp_input_all_to_all_prepacked_qkv(prepacked_qkv)
        else:
            q, k, v = _usp_input_all_to_all_packed_qkv(q, k, v)

    if attention._attention_impl is None:
        attention._set_attention_backend(
            get_attn_backend(
                attention.head_dim,
                q.dtype,
                attention_requirements=AttentionRequirements(packed_varlen=True),
            )
        )

    if ring_active:
        ring_ws, _ = get_ring_ctx()
        if attention._attention_backend_enum is not AttentionBackendEnum.FA:
            raise NotImplementedError(
                "MiniMax H3 ring parallelism requires the FlashAttention "
                "backend (matches --ring-degree's general restriction)."
            )
        # max_seqlen is cu_seqlens[1] (`used`) by construction -- the real,
        # non-padding row count ring needs, already a host int here.
        out = _ring_attention_varlen(
            q,
            k,
            v,
            attn_impl=attention._attention_impl,
            real_seq_len=max_seqlen,
            ring_ws=ring_ws,
        )
    else:
        if (
            attention._attention_backend_enum
            is AttentionBackendEnum.SUBBLOCK_SPARSE_ATTN
        ):
            impl = attention._attention_impl
            sparse_will_run = (
                cu_seqlens_host is not None
                and impl._sparse_ready(q, k)
                and any(
                    stop - start >= impl.schedule.min_seq_len
                    for start, stop in zip(
                        cu_seqlens_host[:-1],
                        cu_seqlens_host[1:],
                    )
                )
            )
            if sparse_will_run and subblock_sparse_query_block_mask is None:
                raise ValueError(
                    "MiniMax H3 requires subblock_sparse_query_block_mask "
                    "when SubBlock sparse attention is active"
                )
            out = attention._attention_impl.forward_varlen(
                q,
                k,
                v,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                cu_seqlens_host=cu_seqlens_host,
                first_segment_sparse_query_block_mask=(
                    subblock_sparse_query_block_mask
                ),
            )
        else:
            # Staging geometry/dtype come from the bf16 q; the fp8 quant below
            # only changes payload dtype, and FA3's fp8 output stays bf16.
            direct = (
                _usp_attn_output_direct_begin(
                    s_global=q.shape[0],
                    h_local=q.shape[1],
                    head_dim=q.shape[2],
                    dtype=q.dtype,
                )
                if ulysses_active and _attn_out_direct_staging_enabled(attention)
                else None
            )
            descale_kwargs = {}
            if _fa3_fp8_attention_enabled(attention):
                fp8_qkv = _quantize_qkv_per_head_fp8(
                    q, k, v, num_segments=cu_seqlens.shape[0] - 1
                )
                if fp8_qkv is not None:
                    q, k, v, (q_descale, k_descale, v_descale) = fp8_qkv
                    descale_kwargs = dict(
                        q_descale=q_descale,
                        k_descale=k_descale,
                        v_descale=v_descale,
                    )
            if direct is not None:
                out = attention._attention_impl.forward_varlen(
                    q,
                    k,
                    v,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                    cu_seqlens_host=cu_seqlens_host,
                    out=direct.attn_out,
                    **descale_kwargs,
                )
                return _usp_attn_output_direct_finish(direct, out)
            out = attention._attention_impl.forward_varlen(
                q,
                k,
                v,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                cu_seqlens_host=cu_seqlens_host,
                **descale_kwargs,
            )
    if ulysses_active:
        if _usp_merge_quant_fp8_enabled(attention):
            fused = _usp_output_all_to_all_quant_fp8(out)
            if fused is not None:
                return fused
        out = _usp_output_all_to_all(out[None], head_dim=2)[0]
    return out


_minimax_h3_attention_core_bcg = eager_on_graph(True)(_minimax_h3_attention_core_impl)


class MiniMaxH3Attention(nn.Module):
    def __init__(
        self,
        arch: MiniMaxH3DiTArchConfig,
        quant_config: QuantizationConfig | None,
        *,
        prefix: str,
        bcg_breakpoint: bool = True,
    ) -> None:
        super().__init__()
        self.bcg_breakpoint = bcg_breakpoint
        self.tp_size = get_tp_world_size()
        if arch.num_attention_heads % self.tp_size:
            raise ValueError(
                "MiniMax H3 attention heads must be divisible by TP size: "
                f"{arch.num_attention_heads} % {self.tp_size} != 0"
            )
        self.total_num_heads = arch.num_attention_heads
        self.num_heads = self.total_num_heads // self.tp_size
        self.head_dim = arch.attention_head_dim
        self.inner_dim = self.total_num_heads * self.head_dim
        self.local_inner_dim = self.num_heads * self.head_dim
        self.softmax_scale = self.head_dim**-0.5
        self.prefix = prefix
        self._attention_impl = None
        self._attention_backend_enum: AttentionBackendEnum | None = None
        # The checkpoint stores one fused qkv tensor. Each logical Q/K/V
        # matrix must be sharded independently; a plain ColumnParallelLinear
        # would instead slice across the concatenated tensor and is incorrect
        # for TP > 1.
        self.qkv_proj = MergedColumnParallelLinear(
            arch.hidden_size,
            [self.inner_dim] * 3,
            bias=False,
            gather_output=False,
            params_dtype=_BF16_DTYPE,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        # Official safetensors interleave Q/K/V by head. Comfy and GGUF
        # checkpoints already store [q_all, k_all, v_all].
        checkpoint_qkv_is_native = quant_config is not None and (
            quant_config.get_name() == "gguf"
            or quant_config.checkpoint_uses_native_qkv_layout
        )
        checkpoint_qkv_is_native = (
            checkpoint_qkv_is_native or arch.checkpoint_uses_diffusers_layout
        )
        self._qkv_rows_interleaved = False
        native_order_mode = _qkv_native_order_mode()
        if native_order_mode != "off":
            if checkpoint_qkv_is_native:
                reason = (
                    "MINIMAX_H3_QKV_NATIVE_ORDER applies to the official "
                    "interleaved qkv checkpoints only; this checkpoint already "
                    "stores [q_all, k_all, v_all] rows."
                )
            else:
                reason = self._qkv_native_order_unsupported_reason(quant_config)
            if reason is None:
                self._install_qkv_native_order(quant_config)
            elif native_order_mode == "strict":
                raise ValueError(reason)
            else:
                _log_qkv_native_order_fallback(reason)
        if not self._qkv_rows_interleaved and not checkpoint_qkv_is_native:
            self._install_qkv_weight_loader(arch)
        self.q_norm = _norm(arch.attention_head_dim, eps=arch.qk_norm_eps)
        self.k_norm = _norm(arch.attention_head_dim, eps=arch.qk_norm_eps)
        # cache width covers cos/sin for temporal, height, and width frequencies
        rope_dim = 6 * arch.rope_inv_freq_len
        self._use_fused_qknorm_rope = (
            current_platform.is_cuda()
            and can_use_fused_inplace_qknorm_rope(
                arch.attention_head_dim,
                rope_dim,
                True,
                _BF16_DTYPE,
                cache_dtype=_BF16_DTYPE,
                round_norm_before_rope=True,
            )
        )
        # Opt-in wide variant (two heads per warp): near-lossless, not
        # bitwise vs the split-chain contract; see MINIMAX_H3_QKNORM_ROPE_WIDE.
        self._qknorm_rope_wide = (
            self._use_fused_qknorm_rope
            and envs.MINIMAX_H3_QKNORM_ROPE_WIDE
            and can_use_fused_inplace_qknorm_rope(
                arch.attention_head_dim,
                rope_dim,
                True,
                _BF16_DTYPE,
                cache_dtype=_BF16_DTYPE,
                round_norm_before_rope=True,
                wide_head=True,
            )
        )
        self.out_proj = RowParallelLinear(
            self.inner_dim,
            arch.hidden_size,
            bias=False,
            input_is_parallel=True,
            params_dtype=_BF16_DTYPE,
            quant_config=quant_config,
            prefix=f"{prefix}.out_proj",
        )
        self.out_proj_accepts_prequant_fp8 = _linear_accepts_per_token_prequant_fp8(
            self.out_proj
        )

    def _set_attention_backend(self, backend) -> None:
        impl_cls = backend.get_impl_cls()
        self._attention_impl = impl_cls(
            num_heads=self.num_heads,
            head_size=self.head_dim,
            causal=False,
            softmax_scale=self.softmax_scale,
            num_kv_heads=self.num_heads,
            prefix=self.prefix,
        )
        # Ring only supports FA (see _minimax_h3_attention_core_impl); keep
        # the resolved enum alongside the impl instance instead of a second
        # get_attn_backend() call at the ring gate.
        self._attention_backend_enum = backend.get_enum()

    def _install_qkv_weight_loader(self, arch: MiniMaxH3DiTArchConfig) -> None:
        weight = self.qkv_proj.weight
        # h3 checkpoints interleave each attention head's Q, K, and V rows
        # this parameter needs reordering before the native QKV projection
        weight.checkpoint_mapping_unsafe = True
        base_loader = weight.weight_loader

        def _reorder_checkpoint_weight(loaded_weight: torch.Tensor) -> torch.Tensor:
            return _reorder_grouped_qkv_to_qkv(
                loaded_weight,
                num_query_groups=arch.num_attention_heads,
                heads_per_group=1,
                head_dim=arch.attention_head_dim,
            )

        def _weight_loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
            # The grouped checkpoint layout is
            # [num_query_groups, q_per_group + k + v] before splitting.
            # MiniMax H3 uses MHA, so checkpoint rows are per-head [q, k, v],
            # while SGLang stores [q_all, k_all, v_all].
            if _copy_grouped_qkv_tp_shard(
                param,
                loaded_weight,
                num_query_groups=arch.num_attention_heads,
                head_dim=arch.attention_head_dim,
                tp_rank=self.qkv_proj.tp_rank,
                tp_size=self.tp_size,
            ):
                return
            base_loader(param, _reorder_checkpoint_weight(loaded_weight))

        if hasattr(weight, "_weight_loader"):
            weight._weight_loader = _weight_loader
        else:
            weight.weight_loader = _weight_loader
        # rank-local FSDP must reorder grouped QKV before selecting each shard
        weight.rank_local_weight_transform = _reorder_checkpoint_weight

        # A quantized checkpoint stores metadata indexed by output row next to the
        # rows themselves (NVFP4 block scales, fp8 per-channel scales). Those rows
        # are permuted above, so the per-row metadata has to be permuted the same
        # way. Row count is the gate: a swizzled scale layout is not row-indexed,
        # and per-tensor scales are scalars, so both are passed through untouched.
        qkv_rows = 3 * arch.num_attention_heads * arch.attention_head_dim
        for name, param in self.qkv_proj.named_parameters(recurse=False):
            if name == "weight":
                continue
            _install_qkv_row_reorder(param, _reorder_checkpoint_weight, qkv_rows)

    def _qkv_native_order_unsupported_reason(
        self, quant_config: QuantizationConfig | None
    ) -> str | None:
        """First reason the native qkv row order cannot serve this config.

        Everything that assumes the reordered [q_all, k_all, v_all] layout is
        unsupported: quantized checkpoints (row-indexed scale metadata moves
        with the reorder), TP sharding (rows are sliced per logical Q/K/V
        matrix), non-CUDA platforms (the MPS streamed path splits contiguous
        Q/K/V segments), and startup LoRA adapters (adapter rows are sliced
        assuming the reordered layout).
        """
        if not current_platform.is_cuda():
            return "MINIMAX_H3_QKV_NATIVE_ORDER requires a CUDA platform."
        if quant_config is not None:
            return (
                "MINIMAX_H3_QKV_NATIVE_ORDER does not support quantized "
                f"checkpoints (quantization={quant_config.get_name()!r}); "
                "row-indexed scale metadata assumes the reordered layout."
            )
        if self.tp_size > 1:
            return (
                "MINIMAX_H3_QKV_NATIVE_ORDER requires DiT tp_size == 1, got "
                f"{self.tp_size}."
            )
        if _startup_lora_configured():
            return (
                "MINIMAX_H3_QKV_NATIVE_ORDER does not support LoRA adapters "
                "(--lora-path is configured); adapter rows assume the "
                "reordered [q_all, k_all, v_all] layout."
            )
        return None

    def _install_qkv_native_order(
        self, quant_config: QuantizationConfig | None
    ) -> None:
        """Keep the checkpoint's per-head [q|k|v] qkv row interleave resident.

        The interleaved rows are already the destination-major Ulysses send
        layout: destination w's [T, h_local, 3*head_dim] slab is the w-th
        contiguous row block of the weight, so the projection can write the
        A2A send buffer directly and the pack kernel becomes a no-op.
        Unsupported configs raise here; under the default-on auto mode the
        constructor checks the same preconditions first and falls back to the
        legacy reordered rows instead. Dynamic LoRA attached after startup is
        rejected at conversion time via the weight marker read by the LoRA
        pipeline.
        """
        reason = self._qkv_native_order_unsupported_reason(quant_config)
        if reason is not None:
            raise ValueError(reason)
        self._qkv_rows_interleaved = True
        self.qkv_proj.weight.qkv_rows_interleaved = True

    def _fused_qknorm_rope_inplace(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        positions: torch.Tensor,
    ) -> None:
        fused_inplace_qknorm_rope(
            q,
            k,
            self.q_norm.weight,
            self.k_norm.weight,
            cos_sin_cache,
            positions,
            is_neox=True,
            eps=self.q_norm.eps,
            head_dim=self.head_dim,
            rope_dim=cos_sin_cache.shape[-1],
            round_norm_before_rope=True,
            wide_head=self._qknorm_rope_wide,
        )

    def _project_qkv_interleaved(
        self,
        x: torch.Tensor,
        *,
        rope_cache: tuple[torch.Tensor, torch.Tensor] | None,
        ulysses_active: bool,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        """Project qkv with the checkpoint's per-head [q|k|v] rows resident.

        Returns ``(q, k, v, prepacked)``. On the Ulysses dual-slab path the
        projection writes the destination-major A2A send buffer directly and
        norm+RoPE run in place on its interleaved views; q/k/v are None and
        ``prepacked`` is that buffer. Otherwise q/k/v are [T, heads, head_dim]
        views with head stride 3*head_dim -- the same stride pattern every
        attention backend already receives after the Ulysses exchange.
        """
        total = x.shape[0]
        world, _ = get_ulysses_ctx() if ulysses_active else (1, 0)
        fused_rope_ready = (
            rope_cache is not None
            and self._use_fused_qknorm_rope
            and not torch.compiler.is_compiling()
        )
        if (
            world > 1
            and fused_rope_ready
            and x.is_cuda
            and x.dtype is _BF16_DTYPE
            and self.num_heads % world == 0
            and not torch.is_grad_enabled()
            # The captured GEMM would bake the grow-only staging buffer's
            # address into the graph; capture keeps the pack-kernel path.
            and not torch.cuda.is_current_stream_capturing()
        ):
            cos_sin_cache, positions = rope_cache
            packed = self._project_qkv_dual_slab(
                x, cos_sin_cache=cos_sin_cache, positions=positions, world=world
            )
            return None, None, None, packed

        qkv, _ = self.qkv_proj(x)
        heads = qkv.view(total, self.num_heads, 3 * self.head_dim)
        q = heads[..., : self.head_dim]
        k = heads[..., self.head_dim : 2 * self.head_dim]
        v = heads[..., 2 * self.head_dim :]
        if rope_cache is None:
            q, k = _apply_qk_norm(q, k, self.q_norm, self.k_norm, self.head_dim)
        elif fused_rope_ready:
            cos_sin_cache, positions = rope_cache
            self._fused_qknorm_rope_inplace(q, k, cos_sin_cache, positions)
        else:
            cos_sin_cache, positions = rope_cache
            q, k = _apply_qk_norm(q, k, self.q_norm, self.k_norm, self.head_dim)
            q, k = _apply_rope_qk(q, k, cos_sin_cache, positions)
        return q, k, v, None

    def _project_qkv_dual_slab(
        self,
        x: torch.Tensor,
        *,
        cos_sin_cache: torch.Tensor,
        positions: torch.Tensor,
        world: int,
    ) -> torch.Tensor:
        """qkv projection written directly as the Ulysses send buffer.

        One [T, qkv_rows/world] GEMM per destination rank into the contiguous
        [world, T, h_local, 3*head_dim] staging buffer; splitting the GEMM N
        dimension keeps each output element's K reduction unchanged, so the
        buffer is bit-equal to pack_qkv_destination_major over the single-GEMM
        projection (verified on H200). Norm+RoPE then run in place per slab on
        the interleaved head views.
        """
        from sglang.multimodal_gen.runtime.layers.usp import _a2a_staging_buffer

        total = x.shape[0]
        head_dim = self.head_dim
        h_local = self.num_heads // world
        slab_rows = 3 * h_local * head_dim
        weight = self.qkv_proj.weight
        packed = _a2a_staging_buffer(
            "usp_packed_qkv_src",
            (world, total, h_local, 3 * head_dim),
            x.dtype,
            x.device,
        )
        slabs = packed.view(world, total, slab_rows)
        for dest in range(world):
            torch.mm(
                x,
                weight[dest * slab_rows : (dest + 1) * slab_rows].t(),
                out=slabs[dest],
            )
        for dest in range(world):
            slab = packed[dest]
            self._fused_qknorm_rope_inplace(
                slab[..., :head_dim],
                slab[..., head_dim : 2 * head_dim],
                cos_sin_cache,
                positions,
            )
        return packed

    def _forward_mps_streamed_attention(
        self,
        x: torch.Tensor,
        *,
        rope_cache: tuple[torch.Tensor, torch.Tensor] | None,
        cu_seqlens: torch.Tensor,
        cu_seqlens_host: tuple[int, ...] | None,
        max_seqlen: int,
    ) -> torch.Tensor:
        """Run MPS attention without materializing the full QKV activation.

        H3's fused QKV output alone is roughly 1.45 GiB at 768px.  MPS shares
        unified memory with the host, so retaining it alongside the packed
        residual and SDPA workspace can evict the OS.  Build normalized K/V
        once, then project Q a small chunk at a time and immediately consume it
        through attention and the output projection.  The formula, weights,
        and complete K/V context are unchanged; this is intentionally limited
        to single-device MPS where Ulysses collectives are not active.
        """
        total = x.shape[0]
        key = torch.empty(
            (total, self.num_heads, self.head_dim), dtype=x.dtype, device=x.device
        )
        value = torch.empty_like(key)
        cos_sin_cache = None if rope_cache is None else rope_cache[0]

        # Do not retain Q while producing the K/V cache.  Dropping the chunk
        # before the next transfer keeps only two full-width attention tensors.
        for start in range(0, total, _MPS_QKV_PROJECTION_TOKEN_CHUNK_SIZE):
            stop = min(start + _MPS_QKV_PROJECTION_TOKEN_CHUNK_SIZE, total)
            qkv, _ = self.qkv_proj(x[start:stop])
            q_chunk, k_chunk, v_chunk = qkv.split(self.local_inner_dim, dim=-1)
            del q_chunk
            k_chunk = self.k_norm(k_chunk.view(-1, self.num_heads, self.head_dim))
            if cos_sin_cache is not None:
                k_chunk = _apply_rope(k_chunk, cos_sin_cache[start:stop])
            key[start:stop].copy_(k_chunk)
            value[start:stop].copy_(v_chunk.view(-1, self.num_heads, self.head_dim))
            del qkv, k_chunk, v_chunk
            torch.mps.synchronize()
            torch.mps.empty_cache()

        if self._attention_impl is None:
            self._set_attention_backend(
                get_attn_backend(
                    self.head_dim,
                    x.dtype,
                    attention_requirements=AttentionRequirements(packed_varlen=True),
                )
            )
        bounds = (
            cu_seqlens_host
            if cu_seqlens_host is not None
            else tuple(int(item) for item in cu_seqlens.tolist())
        )
        out = torch.empty_like(x)
        for sequence_start, sequence_stop in zip(bounds[:-1], bounds[1:]):
            if sequence_start == sequence_stop:
                continue
            keys = key[sequence_start:sequence_stop].unsqueeze(0)
            values = value[sequence_start:sequence_stop].unsqueeze(0)
            for start in range(
                sequence_start,
                sequence_stop,
                _MPS_QKV_PROJECTION_TOKEN_CHUNK_SIZE,
            ):
                stop = min(start + _MPS_QKV_PROJECTION_TOKEN_CHUNK_SIZE, sequence_stop)
                qkv, _ = self.qkv_proj(x[start:stop])
                q_chunk, k_chunk, v_chunk = qkv.split(self.local_inner_dim, dim=-1)
                del k_chunk, v_chunk
                for query_start in range(
                    start, stop, _MPS_ATTENTION_QUERY_TOKEN_CHUNK_SIZE
                ):
                    query_stop = min(
                        query_start + _MPS_ATTENTION_QUERY_TOKEN_CHUNK_SIZE, stop
                    )
                    q = self.q_norm(
                        q_chunk[query_start - start : query_stop - start].view(
                            -1, self.num_heads, self.head_dim
                        )
                    )
                    if cos_sin_cache is not None:
                        q = _apply_rope(q, cos_sin_cache[query_start:query_stop])
                    attention_out = self._attention_impl.forward(
                        q.unsqueeze(0), keys, values, None
                    )[0]
                    projected, _ = self.out_proj(
                        attention_out.reshape(
                            query_stop - query_start, self.local_inner_dim
                        )
                    )
                    out[query_start:query_stop].copy_(projected)
                    del q, attention_out, projected
                del qkv, q_chunk
                torch.mps.synchronize()
                torch.mps.empty_cache()

        del key, value
        torch.mps.empty_cache()
        return out

    def forward(
        self,
        x: torch.Tensor,
        *,
        rope_cache: tuple[torch.Tensor, torch.Tensor] | None,
        cu_seqlens: torch.Tensor,
        cu_seqlens_host: tuple[int, ...] | None = None,
        max_seqlen: int,
        subblock_sparse_query_block_mask: torch.Tensor | None = None,
        ulysses_active: bool = False,
        ring_active: bool = False,
    ) -> torch.Tensor:
        """x: [T, hidden] packed thd rows -> [T, hidden].

        Operation order: fused qkv projection -> per-head q/k RMSNorm -> RoPE
        on q/k -> variable-length non-causal flash attention -> output projection.

        With Ulysses sequence parallelism, x holds this rank's row shard;
        qkv/norm/RoPE run locally, an all-to-all trades sequence for heads.
        Each rank attends the full sequence with heads/world_size local heads,
        so cu_seqlens retains global packed-document semantics. The inverse
        all-to-all restores the row shard before the output projection.
        """
        if x.device.type == "mps" and not ulysses_active:
            return self._forward_mps_streamed_attention(
                x,
                rope_cache=rope_cache,
                cu_seqlens=cu_seqlens,
                cu_seqlens_host=cu_seqlens_host,
                max_seqlen=max_seqlen,
            )

        total = x.shape[0]
        prepacked_qkv = None
        if self._qkv_rows_interleaved:
            q, k, v, prepacked_qkv = self._project_qkv_interleaved(
                x, rope_cache=rope_cache, ulysses_active=ulysses_active
            )
        else:
            qkv, _ = self.qkv_proj(x)
            q, k, v = qkv.split(self.local_inner_dim, dim=-1)
            q = q.view(total, self.num_heads, self.head_dim)
            k = k.view(total, self.num_heads, self.head_dim)
            v = v.view(total, self.num_heads, self.head_dim)
            if rope_cache is None:
                q, k = _apply_qk_norm(
                    q,
                    k,
                    self.q_norm,
                    self.k_norm,
                    self.head_dim,
                )
            else:
                cos_sin_cache, positions = rope_cache
                if self._use_fused_qknorm_rope and not torch.compiler.is_compiling():
                    self._fused_qknorm_rope_inplace(q, k, cos_sin_cache, positions)
                else:
                    q, k = _apply_qk_norm(
                        q,
                        k,
                        self.q_norm,
                        self.k_norm,
                        self.head_dim,
                    )
                    q, k = _apply_rope_qk(q, k, cos_sin_cache, positions)

        attention_core = (
            _minimax_h3_attention_core_bcg
            if self.bcg_breakpoint
            else _minimax_h3_attention_core_impl
        )
        out = attention_core(
            self,
            q,
            k,
            v,
            cu_seqlens=cu_seqlens,
            cu_seqlens_host=cu_seqlens_host,
            max_seqlen=max_seqlen,
            subblock_sparse_query_block_mask=subblock_sparse_query_block_mask,
            ulysses_active=ulysses_active,
            ring_active=ring_active,
            prepacked_qkv=prepacked_qkv,
        )
        if isinstance(out, tuple):
            # The Ulysses merge pre-quantized out_proj's input:
            # (fp8 payload [T, heads*head_dim], per-token fp32 scale).
            out, _ = self.out_proj(out)
            return out
        out = out.reshape(total, self.num_heads * self.head_dim)
        out, _ = self.out_proj(out)
        return out


class MiniMaxH3MLP(nn.Module):
    def __init__(
        self,
        arch: MiniMaxH3DiTArchConfig,
        quant_config: QuantizationConfig | None,
        *,
        prefix: str,
    ) -> None:
        super().__init__()
        # As with qkv, gate and up are two independently sharded logical
        # matrices even though the checkpoint stores them fused.
        self.fc1 = MergedColumnParallelLinear(
            arch.hidden_size,
            [arch.ffn_hidden_size] * 2,
            bias=False,
            gather_output=False,
            params_dtype=_BF16_DTYPE,
            quant_config=quant_config,
            prefix=f"{prefix}.fc1",
        )
        # Chunk the fused fc1 output as [gate, up], then compute
        # silu(gate) * up.
        self.fc2 = RowParallelLinear(
            arch.ffn_hidden_size,
            arch.hidden_size,
            bias=False,
            input_is_parallel=True,
            params_dtype=_BF16_DTYPE,
            quant_config=quant_config,
            prefix=f"{prefix}.fc2",
        )
        self.reuse_fc1_activation = quant_config is None or (
            quant_config.get_name() == "gguf"
        )
        # Under fp8 W8A8 the eager silu chain + fc2's standalone quant are
        # replaced by one fused kernel emitting fc2's pre-quantized input;
        # bitwise equal to that separated chain (MINIMAX_H3_FUSED_SILU_QUANT
        # is the kill switch, read per forward).
        self.fc2_accepts_prequant_fp8 = (
            not self.reuse_fc1_activation
            and _linear_accepts_per_token_prequant_fp8(self.fc2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.device.type == "mps":
            out = torch.empty_like(x)
            for start in range(0, x.shape[0], _MPS_MLP_TOKEN_CHUNK_SIZE):
                stop = min(start + _MPS_MLP_TOKEN_CHUNK_SIZE, x.shape[0])
                hidden, _ = self.fc1(x[start:stop])
                hidden = _silu_mul(hidden, reuse_input=self.reuse_fc1_activation)
                chunk, _ = self.fc2(hidden)
                out[start:stop].copy_(chunk)
                del hidden, chunk
                torch.mps.synchronize()
                torch.mps.empty_cache()
            return out
        hidden, _ = self.fc1(x)
        if (
            self.fc2_accepts_prequant_fp8
            and envs.MINIMAX_H3_FUSED_SILU_QUANT
            and not torch.compiler.is_compiling()
            and can_use_fused_silu_mul_per_token_quant_fp8(hidden)
        ):
            out, _ = self.fc2(fused_silu_mul_per_token_quant_fp8(hidden))
            return out
        hidden = _silu_mul(hidden, reuse_input=self.reuse_fc1_activation)
        out, _ = self.fc2(hidden)
        return out


class MiniMaxH3AdalnProj(nn.Module):
    """SiLU + zero-init linear over unique condition embeddings.

    Per block, three modalities each produce six H-wide vectors:
    [M, t_dim] -> [M, 3*6H] -> view(M*3, 6H) -> chunk(6).
    The final layer uses one modality and produces two H-wide vectors:
    [M, t_dim] -> [M, 2H] -> chunk(2).
    """

    def __init__(
        self,
        arch: MiniMaxH3DiTArchConfig,
        out_features: int,
        quant_config: QuantizationConfig | None,
        *,
        prefix: str,
        expand_ratio: int,
        modality_num: int,
    ) -> None:
        super().__init__()
        if out_features != expand_ratio * arch.hidden_size * modality_num:
            raise ValueError(
                "adaln out_features mismatch: "
                f"{out_features} != {expand_ratio}*{arch.hidden_size}*{modality_num}"
            )
        self.expand_ratio = expand_ratio
        self.modality_num = modality_num
        self.hidden_size = arch.hidden_size
        # Curve checkpoints store both the sampled curve and their reduced
        # AdaLN projections in FP32. Preserve that precision island to match
        # the published pruned implementation; these outputs intentionally do
        # not enter the BF16-only fused modulation kernels.
        params_dtype = _FP32_DTYPE if arch.adaln_curve_grid is not None else _BF16_DTYPE
        self.linear = ColumnParallelLinear(
            arch.time_embed_dim,
            out_features,
            bias=True,
            gather_output=False,
            params_dtype=params_dtype,
            quant_config=quant_config,
            prefix=f"{prefix}.linear",
        )

    def project_local(self, adaln_input: torch.Tensor) -> torch.Tensor:
        x, _ = self.linear(adaln_input)
        return x

    def split_output(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        m = x.shape[0]
        x = x.view(m * self.modality_num, self.expand_ratio * self.hidden_size)
        return tuple(x.chunk(self.expand_ratio, dim=-1))

    def forward(self, adaln_input: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Project the post-SiLU embedding in its checkpoint-defined dtype."""
        x = self.project_local(adaln_input)
        if get_tp_world_size() > 1:
            x = tensor_model_parallel_all_gather(x)
        return self.split_output(x)


# A ref2va request carrying both a visual and an audio reference reaches four
# distinct timesteps in one step: video, audio, the imgvid condition and the
# audio reference. That is the widest case, so it is the default; a deployment
# serving only narrower tasks (t2va reaches 2, fl2va 3) can shrink the slab
# proportionally via --minimax-h3-adaln-plan-width.
MINIMAX_H3_ADALN_MAX_PLAN_WIDTH = 4


# Canonical plan-key definition lives with the persist store so the on-disk
# format and the in-memory slab can never disagree on key semantics.
_plan_key = adaln_plan_key


class MiniMaxH3AdalnCache(nn.Module):
    """Precomputed AdaLN outputs for fixed FP32 timestep plans."""

    _FORMAT_VERSION = "2"
    plan_timesteps: torch.Tensor
    plan_lengths: torch.Tensor
    block_params: torch.Tensor
    final_params: torch.Tensor

    def __init__(
        self,
        arch: MiniMaxH3DiTArchConfig,
        *,
        path: str | None = None,
        model_variant: str | None = None,
        weight_files: list[str] | None = None,
        max_plans: int = 64,
        max_plan_width: int = MINIMAX_H3_ADALN_MAX_PLAN_WIDTH,
    ) -> None:
        super().__init__()
        if (path is None) == (weight_files is None):
            raise ValueError(
                "MiniMax H3 AdaLN cache takes exactly one of path (prebuilt "
                "sidecar) or weight_files (rebuild from the checkpoint)"
            )
        if max_plans < 1:
            raise ValueError("MiniMax H3 AdaLN cache max_plans must be positive")
        if max_plan_width < 1:
            raise ValueError(
                "MiniMax H3 AdaLN cache max_plan_width must be positive; "
                "set --minimax-h3-adaln-plan-width to at least 1"
            )
        self.path = path
        self.model_variant = model_variant
        self.weight_files = weight_files
        self.max_plans = max_plans
        self.max_plan_width = max_plan_width
        self.num_layers = arch.num_layers
        self.hidden_size = arch.hidden_size
        self.block_width = 6 * MINIMAX_H3_ADALN_MODALITY_NUM * arch.hidden_size
        self.final_width = 2 * arch.hidden_size
        # Rebuild path only: plan bit pattern -> slot, tracked on the host.
        self._slots: dict[tuple[int, ...], int] = {}
        self.rebuilds = 0
        self.persist_loads = 0
        persist_dir = envs.MINIMAX_H3_ADALN_PERSIST_DIR
        self._persist_spec = (
            make_adaln_persist_spec(
                directory=persist_dir,
                weight_files=weight_files,
                tp_size=get_tp_world_size(),
                num_layers=self.num_layers,
                block_width=self.block_width,
                final_width=self.final_width,
            )
            if weight_files is not None and persist_dir
            else None
        )

    def load(self, device: torch.device) -> None:
        if self.path is None:
            self._allocate(device)
            return
        if not os.path.isfile(self.path):
            raise ValueError(f"MiniMax H3 AdaLN cache does not exist: {self.path}")

        with safe_open(self.path, framework="pt", device="cpu") as cache_file:
            metadata = cache_file.metadata() or {}
            if metadata.get("format_version") != self._FORMAT_VERSION:
                raise ValueError(
                    "MiniMax H3 AdaLN cache has an unsupported or missing format_version"
                )
            cache_variant = metadata.get("model_variant")
            if self.model_variant is not None and cache_variant != self.model_variant:
                raise ValueError(
                    "MiniMax H3 AdaLN cache model_variant does not match the loaded "
                    f"variant ({cache_variant!r} != {self.model_variant!r})"
                )
            plan_timesteps = cache_file.get_tensor("plan_timesteps")
            plan_lengths = cache_file.get_tensor("plan_lengths")
            block_params = cache_file.get_tensor("block_params")
            final_params = cache_file.get_tensor("final_params")

        expected_block_width = 6 * MINIMAX_H3_ADALN_MODALITY_NUM * self.hidden_size
        expected_final_width = 2 * self.hidden_size
        if (
            plan_timesteps.dtype != _FP32_DTYPE
            or plan_timesteps.ndim != 2
            or plan_lengths.dtype != torch.int64
            or plan_lengths.shape != (plan_timesteps.shape[0],)
            or (plan_lengths < 1).any()
            or (plan_lengths > plan_timesteps.shape[1]).any()
        ):
            raise ValueError("MiniMax H3 AdaLN cache has invalid timestep plans")
        if block_params.dtype != _BF16_DTYPE or block_params.shape != (
            plan_timesteps.shape[0],
            plan_timesteps.shape[1],
            self.num_layers,
            expected_block_width,
        ):
            raise ValueError("MiniMax H3 AdaLN cache has invalid block_params")
        if final_params.dtype != _BF16_DTYPE or final_params.shape != (
            plan_timesteps.shape[0],
            plan_timesteps.shape[1],
            expected_final_width,
        ):
            raise ValueError("MiniMax H3 AdaLN cache has invalid final_params")

        self.register_buffer("plan_timesteps", plan_timesteps.to(device))
        self.register_buffer("plan_lengths", plan_lengths.to(device))
        self.register_buffer("block_params", block_params.to(device))
        self.register_buffer("final_params", final_params.to(device))

    def _allocate(self, device: torch.device) -> None:
        """Empty slab for the rebuild path; its pointers must never move.

        ``plan_lengths`` starts at zero and that is what keeps unused slots out
        of ``lookup``: a real plan always has at least one timestep, so a zero
        length can never match. Breakable CUDA graph keys its replay signature
        on tensor pointers, so this is allocated once and only written in place.
        """
        width = self.max_plan_width
        self.register_buffer(
            "plan_timesteps",
            torch.zeros((self.max_plans, width), dtype=_FP32_DTYPE, device=device),
        )
        self.register_buffer(
            "plan_lengths",
            torch.zeros((self.max_plans,), dtype=torch.int64, device=device),
        )
        self.register_buffer(
            "block_params",
            torch.zeros(
                (self.max_plans, width, self.num_layers, self.block_width),
                dtype=_BF16_DTYPE,
                device=device,
            ),
        )
        self.register_buffer(
            "final_params",
            torch.zeros(
                (self.max_plans, width, self.final_width),
                dtype=_BF16_DTYPE,
                device=device,
            ),
        )
        logger.info(
            "MiniMax H3 AdaLN rebuild slab: %d plans x %d timesteps = %.2f GiB",
            self.max_plans,
            width,
            self.block_params.numel() * 2 / 2**30,
        )

    def build(
        self,
        step_timesteps: list[torch.Tensor],
        *,
        embed: Callable[[torch.Tensor], torch.Tensor],
    ) -> None:
        """Fill every plan this request will look up, in one streaming pass.

        Each plan keeps its own timestep count as the GEMM batch size, because
        cuBLAS selects kernels by shape and the selection is not monotonic in M:
        against the runtime's M == 2, results at M == 4/8/16/64/96 are
        bit-identical while M == 32 differs in 11760 of 96768 elements and
        M == 1 (the GEMV path the first denoise step takes) differs in 69.
        Rebuilding a plan at any other batch size silently perturbs the output.

        The pass reads all 50 adaln_proj layers regardless of how many plans are
        missing, so a request builds everything it needs before denoising rather
        than filling in step by step. When a persist store holds every missing
        plan (MINIMAX_H3_ADALN_PERSIST_DIR), the checkpoint pass is skipped and
        the previously built slabs are read back instead.
        """
        wanted: dict[tuple[int, ...], torch.Tensor] = {}
        for timesteps in step_timesteps:
            wanted.setdefault(_plan_key(timesteps), timesteps)
        missing = {k: v for k, v in wanted.items() if k not in self._slots}
        if not missing:
            return
        if len(wanted) > self.max_plans:
            raise ValueError(
                f"MiniMax H3 AdaLN rebuild needs {len(wanted)} plans but "
                f"max_plans is {self.max_plans}"
            )
        widest = max(timesteps.numel() for timesteps in wanted.values())
        if widest > self.max_plan_width:
            raise ValueError(
                f"MiniMax H3 AdaLN rebuild hit a {widest}-timestep plan but the "
                f"slab was allocated for {self.max_plan_width}; raise "
                "--minimax-h3-adaln-plan-width (t2va needs 2, fl2va 3, ref2va 4)"
            )

        reset = len(self._slots) + len(missing) > self.max_plans
        # A reset also evicts this request's cache hits, so rebuild its complete
        # plan set rather than only the plans that were initially missing.
        plans_to_build = wanted if reset else missing
        if reset:
            self._slots.clear()
            self.plan_lengths.zero_()

        device = self.block_params.device
        if self._load_plans_from_persist_store(plans_to_build, device=device):
            return

        slots = []
        pending_slots: dict[tuple[int, ...], int] = {}
        for offset, (key, timesteps) in enumerate(plans_to_build.items()):
            slot = len(self._slots) + offset
            pending_slots[key] = slot
            slots.append((slot, timesteps.numel(), embed(timesteps.to(device))))
            self.plan_timesteps[slot, : timesteps.numel()] = timesteps.to(device)

        # adaln_proj is a ColumnParallelLinear: each rank owns a slice of the
        # output features and all-gathers afterwards. The rebuild has to do the
        # same rather than read the full width in one go -- a sharded GEMM has a
        # different N, so cuBLAS picks a different kernel and the outputs stop
        # matching. It also cuts per-rank checkpoint reads to 1/tp.
        tp_size = get_tp_world_size()
        tp_rank = get_tp_rank() if tp_size > 1 else 0

        with ExitStack() as stack:
            handles = [
                stack.enter_context(safe_open(f, framework="pt", device=str(device)))
                for f in self.weight_files
            ]
            index = {name: h for h in handles for name in h.keys()}

            def read_shard(name: str, out_features: int) -> torch.Tensor:
                if tp_size == 1:
                    return index[name].get_tensor(name)
                shard = out_features // tp_size
                start = tp_rank * shard
                return index[name].get_slice(name)[start : start + shard]

            def project(adaln_input: torch.Tensor, weight, bias) -> torch.Tensor:
                out = nn.functional.linear(adaln_input, weight, bias)
                return tensor_model_parallel_all_gather(out) if tp_size > 1 else out

            for layer in range(self.num_layers):
                prefix = f"blocks.{layer}.adaln_proj.linear"
                weight = read_shard(f"{prefix}.weight", self.block_width)
                bias = read_shard(f"{prefix}.bias", self.block_width)
                for slot, length, adaln_input in slots:
                    self.block_params[slot, :length, layer] = project(
                        adaln_input, weight, bias
                    )
                del weight, bias
            prefix = "final_layer.adaln_proj.linear"
            weight = read_shard(f"{prefix}.weight", self.final_width)
            bias = read_shard(f"{prefix}.bias", self.final_width)
            for slot, length, adaln_input in slots:
                self.final_params[slot, :length] = project(adaln_input, weight, bias)
            del weight, bias

        for slot, length, _ in slots:
            self.plan_lengths[slot] = length
        # Commit host metadata only after every layer has been written. If a
        # checkpoint read or projection raises, the zero-length slots remain
        # invisible and a later request can retry the rebuild.
        self._slots.update(pending_slots)
        self.rebuilds += 1
        logger.info(
            "MiniMax H3 AdaLN: rebuilt %d plan(s), %d/%d resident, pass #%d",
            len(plans_to_build),
            len(self._slots),
            self.max_plans,
            self.rebuilds,
        )
        self._save_plans_to_persist_store(plans_to_build)

    def _load_plans_from_persist_store(
        self,
        plans_to_build: dict[tuple[int, ...], torch.Tensor],
        *,
        device: torch.device,
    ) -> bool:
        """Fill the requested slots from disk instead of the checkpoint pass."""
        if self._persist_spec is None:
            return False
        loaded = load_adaln_plans(
            self._persist_spec, list(plans_to_build), device=device
        )
        if loaded is None:
            return False
        pending_slots: dict[tuple[int, ...], int] = {}
        lengths: list[tuple[int, int]] = []
        for offset, (key, entry) in enumerate(zip(plans_to_build, loaded)):
            plan_timesteps, block_params, final_params = entry
            slot = len(self._slots) + offset
            length = plan_timesteps.numel()
            pending_slots[key] = slot
            self.plan_timesteps[slot, :length] = plan_timesteps
            self.block_params[slot, :length] = block_params
            self.final_params[slot, :length] = final_params
            lengths.append((slot, length))
        for slot, length in lengths:
            self.plan_lengths[slot] = length
        self._slots.update(pending_slots)
        self.persist_loads += 1
        logger.info(
            "MiniMax H3 AdaLN: loaded %d plan(s) from the persist store, "
            "%d/%d resident",
            len(plans_to_build),
            len(self._slots),
            self.max_plans,
        )
        return True

    def _save_plans_to_persist_store(
        self,
        plans_built: dict[tuple[int, ...], torch.Tensor],
    ) -> None:
        """Best-effort persist of freshly built plans, first world rank only."""
        if self._persist_spec is None or get_world_rank() != 0:
            return
        try:
            saved = save_adaln_plans(
                self._persist_spec,
                (
                    (
                        key,
                        self.plan_timesteps[self._slots[key], : timesteps.numel()],
                        self.block_params[self._slots[key], : timesteps.numel()],
                        self.final_params[self._slots[key], : timesteps.numel()],
                    )
                    for key, timesteps in plans_built.items()
                ),
            )
        except Exception:
            logger.warning(
                "MiniMax H3 AdaLN: persisting built plans failed; continuing",
                exc_info=True,
            )
            return
        if saved:
            logger.info(
                "MiniMax H3 AdaLN: persisted %d plan(s) to %s",
                saved,
                self._persist_spec.directory,
            )

    def lookup(self, unique_timesteps: torch.Tensor) -> torch.Tensor:
        num_timesteps = unique_timesteps.shape[0]
        matches = self.plan_lengths.eq(num_timesteps) & self.plan_timesteps[
            :, :num_timesteps
        ].eq(unique_timesteps).all(dim=-1)
        if not bool(matches.any()):
            raise ValueError(
                "MiniMax H3 AdaLN cache does not cover the request timestep plan"
            )
        return matches.to(torch.int64).argmax()

    def block(
        self,
        index: int,
        cache_plan_index: torch.Tensor,
        num_timesteps: int,
    ) -> tuple[torch.Tensor, ...]:
        params = self.block_params[cache_plan_index, :num_timesteps, index]
        params = params.reshape(-1, 6, self.hidden_size)
        return tuple(params.unbind(dim=1))

    def block_all(
        self,
        cache_plan_index: torch.Tensor,
        num_timesteps: int,
    ) -> tuple[tuple[torch.Tensor, ...], ...]:
        """Every block's parameters through one batched slab gather.

        Per-block ``block()`` calls run one advanced-index gather each -- with
        a device plan index that is num_layers small latency-bound kernels per
        step. One gather over the plan's whole [num_layers, width] slab keeps
        the same elements and hands each block the same views ``block()``
        returns; the remaining per-block consumption is view-only.
        """
        # The permute-first gather materializes [num_layers, num_timesteps,
        # block_width] contiguous in a single indexing kernel.
        params = self.block_params.permute(2, 0, 1, 3)[
            :, cache_plan_index, :num_timesteps
        ]
        params = params.reshape(self.num_layers, -1, 6, self.hidden_size)
        # One transpose-copy makes every per-param [groups, hidden] row
        # contiguous, which the C++ modulation kernels require; without it
        # each unbound view keeps the interleaved (6*hidden, 1) stride and
        # every fused site would fall back or re-copy per block.
        params = params.transpose(1, 2).contiguous()
        return tuple(tuple(layer.unbind(dim=0)) for layer in params)

    def final(
        self,
        cache_plan_index: torch.Tensor,
        num_timesteps: int,
    ) -> tuple[torch.Tensor, ...]:
        params = self.final_params[cache_plan_index, :num_timesteps]
        return tuple(params.reshape(-1, 2, self.hidden_size).unbind(dim=1))


class MiniMaxH3TokenRefinerBlock(nn.Module):
    """Standard pre-norm transformer block without AdaLN or RoPE."""

    def __init__(
        self,
        arch: MiniMaxH3DiTArchConfig,
        quant_config: QuantizationConfig | None,
        *,
        prefix: str,
    ) -> None:
        super().__init__()
        self.norm1 = _norm(arch.hidden_size, eps=arch.norm_eps)
        self.norm2 = _norm(arch.hidden_size, eps=arch.norm_eps)
        # The whole dynamic text-refiner/scatter phase is one BCG break point;
        # do not nest per-attention graph breaks inside it.
        self.attn = MiniMaxH3Attention(
            arch,
            quant_config,
            prefix=f"{prefix}.attn",
            bcg_breakpoint=False,
        )
        self.mlp = MiniMaxH3MLP(arch, quant_config, prefix=f"{prefix}.mlp")

    def forward(
        self,
        x: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor,
        cu_seqlens_host: tuple[int, ...] | None = None,
        max_seqlen: int,
    ) -> torch.Tensor:
        x = x + self.attn(
            self.norm1(x),
            rope_cache=None,
            cu_seqlens=cu_seqlens,
            cu_seqlens_host=cu_seqlens_host,
            max_seqlen=max_seqlen,
        )
        x = x + self.mlp(self.norm2(x))
        return x


class MiniMaxH3TokenRefiner(nn.Module):
    def __init__(
        self,
        arch: MiniMaxH3DiTArchConfig,
        quant_config: QuantizationConfig | None,
        *,
        prefix: str,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                MiniMaxH3TokenRefinerBlock(
                    arch,
                    quant_config,
                    prefix=f"{prefix}.blocks.{index}",
                )
                for index in range(arch.token_refiner_num_layers)
            ]
        )
        self.final_norm = _norm(arch.hidden_size, eps=arch.final_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor,
        cu_seqlens_host: tuple[int, ...] | None = None,
        max_seqlen: int,
    ) -> torch.Tensor:
        for block in self.blocks:
            x = block(
                x,
                cu_seqlens=cu_seqlens,
                cu_seqlens_host=cu_seqlens_host,
                max_seqlen=max_seqlen,
            )
        return self.final_norm(x)


class MiniMaxH3DiTBlock(nn.Module):
    def __init__(
        self,
        arch: MiniMaxH3DiTArchConfig,
        quant_config: QuantizationConfig | None,
        *,
        prefix: str,
        use_adaln_cache: bool = False,
    ) -> None:
        super().__init__()
        self.norm1 = _norm(arch.hidden_size, eps=arch.norm_eps)
        self.norm2 = _norm(arch.hidden_size, eps=arch.norm_eps)
        self.attn = MiniMaxH3Attention(
            arch,
            quant_config,
            prefix=f"{prefix}.attn",
        )
        self.mlp = MiniMaxH3MLP(arch, quant_config, prefix=f"{prefix}.mlp")
        self.adaln_proj = (
            None
            if use_adaln_cache
            else MiniMaxH3AdalnProj(
                arch,
                arch.adaln_out_features,
                quant_config,
                prefix=f"{prefix}.adaln_proj",
                expand_ratio=6,
                modality_num=MINIMAX_H3_ADALN_MODALITY_NUM,
            )
        )
        self.preserve_input_for_cache_dit = False

    def forward(
        self,
        x: torch.Tensor,
        *,
        adaln_input: torch.Tensor | None,
        combined_indices: torch.Tensor,
        rope_cache: tuple[torch.Tensor, torch.Tensor],
        cu_seqlens: torch.Tensor,
        cu_seqlens_host: tuple[int, ...] | None = None,
        max_seqlen: int,
        subblock_sparse_query_block_mask: torch.Tensor | None = None,
        ulysses_active: bool = False,
        ring_active: bool = False,
        adaln_params: tuple[torch.Tensor, ...] | None = None,
    ) -> torch.Tensor:
        """x: [T, H]; adaln_input: [M, t_dim]; combined_indices: [T]
        (= inverse_indices * modality_num + token_tags.clamp(min=0)).

        Each block computes AdaLN parameters once, then applies
        norm1 -> scale/shift -> attention -> gated residual, followed by
        norm2 -> scale/shift -> MLP -> gated residual.
        """
        if adaln_params is None:
            if self.adaln_proj is None:
                raise ValueError("MiniMax H3 AdaLN cache parameters are required")
            adaln_params = self.adaln_proj(adaln_input)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = adaln_params
        # Cache-DiT retains the inputs to its Fn and Mn block ranges. Only the
        # first gated residual writes to that tensor; the second one operates on
        # a block-local buffer.
        residual = x
        h = self.norm1(x)
        h = _modulate_scale_shift(
            h, shift_msa, scale_msa, combined_indices, dtype=_BF16_DTYPE
        )
        h = self.attn(
            h,
            rope_cache=rope_cache,
            cu_seqlens=cu_seqlens,
            cu_seqlens_host=cu_seqlens_host,
            max_seqlen=max_seqlen,
            subblock_sparse_query_block_mask=subblock_sparse_query_block_mask,
            ulysses_active=ulysses_active,
            ring_active=ring_active,
        )
        x = _modulate_gate(
            residual,
            gate_msa,
            h,
            combined_indices,
            dtype=_BF16_DTYPE,
            allow_inplace=not self.preserve_input_for_cache_dit,
        )

        residual = x
        h = self.norm2(x)
        h = _modulate_scale_shift(
            h, shift_mlp, scale_mlp, combined_indices, dtype=_BF16_DTYPE
        )
        h = self.mlp(h)
        # `residual` is block-local here (see above), so this stays in-place
        # even while Cache-DiT is attached.
        return _modulate_gate(
            residual,
            gate_mlp,
            h,
            combined_indices,
            dtype=_BF16_DTYPE,
        )

    def forward_fused(
        self,
        x: torch.Tensor,
        *,
        fused_adaln: tuple[torch.Tensor, ...],
        pending_gate: tuple[torch.Tensor, torch.Tensor] | None,
        combined_indices: torch.Tensor,
        rope_cache: tuple[torch.Tensor, torch.Tensor],
        cu_seqlens: torch.Tensor,
        cu_seqlens_host: tuple[int, ...] | None = None,
        max_seqlen: int,
        subblock_sparse_query_block_mask: torch.Tensor | None = None,
        ulysses_active: bool = False,
        ring_active: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Fused-adaLN ``forward``: norm + indexed scale/shift run as one
        kernel per site, absorbing the preceding indexed gated-residual add
        where one exists (see the fused block loop in
        ``MiniMaxH3DiTModel.forward``).

        ``fused_adaln`` is ``(w_msa, shift_msa, gate_msa, w_mlp, shift_mlp,
        gate_mlp)``, where ``w_*`` is either a bitexact ``(gamma, scale_*)``
        pair or a merged ``norm_weight * (1 + scale_*)`` fp32 tensor (see
        ``_fused_adaln_block_params``). ``pending_gate`` carries the previous
        block's ``(gate, update)`` whose gated-residual add lands in this
        block's first fused norm; the return value ``(residual, (gate_mlp,
        update))`` defers this block's trailing gate to the caller the same
        way.
        """
        w_msa, shift_msa, gate_msa, w_mlp, shift_mlp, gate_mlp = fused_adaln
        if pending_gate is None:
            residual = x
            h = _fused_norm_modulate(
                x, w_msa, shift_msa, combined_indices, eps=self.norm1.eps
            )
        else:
            prev_gate, prev_update = pending_gate
            h, residual = _fused_gate_norm_modulate_(
                x,
                prev_update,
                prev_gate,
                w_msa,
                shift_msa,
                combined_indices,
                eps=self.norm1.eps,
            )
        h = self.attn(
            h,
            rope_cache=rope_cache,
            cu_seqlens=cu_seqlens,
            cu_seqlens_host=cu_seqlens_host,
            max_seqlen=max_seqlen,
            subblock_sparse_query_block_mask=subblock_sparse_query_block_mask,
            ulysses_active=ulysses_active,
            ring_active=ring_active,
        )
        h, residual = _fused_gate_norm_modulate_(
            residual,
            h,
            gate_msa,
            w_mlp,
            shift_mlp,
            combined_indices,
            eps=self.norm2.eps,
        )
        h = self.mlp(h)
        return residual, (gate_mlp, h)


class MiniMaxH3FinalLayer(nn.Module):
    def __init__(
        self,
        arch: MiniMaxH3DiTArchConfig,
        quant_config: QuantizationConfig | None,
        *,
        prefix: str,
        use_adaln_cache: bool = False,
    ) -> None:
        super().__init__()
        video_patch_dim = (
            arch.latents_dim
            * arch.patch_size[0]
            * arch.patch_size[1]
            * arch.patch_size[2]
        )
        self.norm = _norm(arch.hidden_size, eps=arch.final_norm_eps)
        self.adaln_proj = (
            None
            if use_adaln_cache
            else MiniMaxH3AdalnProj(
                arch,
                arch.final_adaln_out_features,
                quant_config,
                prefix=f"{prefix}.adaln_proj",
                expand_ratio=2,
                modality_num=1,
            )
        )
        self.video_out = ColumnParallelLinear(
            arch.hidden_size,
            video_patch_dim,
            bias=True,
            gather_output=False,
            params_dtype=_FP32_DTYPE,
            quant_config=None,
            prefix=f"{prefix}.video_out",
        )
        self.audio_out = ColumnParallelLinear(
            arch.hidden_size,
            arch.audio_latents_dim,
            bias=True,
            gather_output=False,
            params_dtype=_FP32_DTYPE,
            quant_config=None,
            prefix=f"{prefix}.audio_out",
        )
        self._merged_out_key: tuple[int, ...] | None = None
        self._merged_out_weight: torch.Tensor | None = None
        self._merged_out_bias: torch.Tensor | None = None

    def _merged_out_params(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Concatenated [video|audio] head params for the single-GEMM path.

        Only plain unquantized heads qualify: a LoRA (or any other) wrapper
        owns the projection math, so the two-GEMM path stays authoritative
        there. The cache key tracks each tensor's storage pointer and in-place
        version, so weight reloads, device moves, and in-place merges rebuild
        the concat instead of leaving it stale.
        """
        video_out = self.video_out
        audio_out = self.audio_out
        if (
            type(video_out) is not ColumnParallelLinear
            or type(audio_out) is not ColumnParallelLinear
        ):
            return None
        if not isinstance(
            video_out.quant_method, UnquantizedLinearMethod
        ) or not isinstance(audio_out.quant_method, UnquantizedLinearMethod):
            return None
        params = (video_out.weight, video_out.bias, audio_out.weight, audio_out.bias)
        if any(isinstance(param, DTensor) for param in params):
            return None
        key = tuple(
            value for param in params for value in (param.data_ptr(), param._version)
        )
        if key != self._merged_out_key:
            self._merged_out_weight = torch.cat(
                (video_out.weight.detach(), audio_out.weight.detach()), dim=0
            )
            self._merged_out_bias = torch.cat(
                (video_out.bias.detach(), audio_out.bias.detach()), dim=0
            )
            self._merged_out_key = key
        return self._merged_out_weight, self._merged_out_bias

    def forward(
        self,
        x: torch.Tensor,
        *,
        adaln_input: torch.Tensor | None,
        inverse_indices: torch.Tensor,
        adaln_params: tuple[torch.Tensor, ...] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Project all rows into TP-local video/audio output shards.

        Apply single-modality shift/scale AdaLN to the final normalized
        activations, cast to fp32, then apply both output heads to all rows.
        The model gathers output columns only after selecting live media rows,
        preserving the GEMM shape while reducing collective payload.

        Returns (video, audio, merged): with plain unwrapped heads both
        projections run as one concatenated GEMM, video/audio are column views
        of ``merged`` (already the ``cat((video, audio), -1)`` layout), and
        callers needing the concatenation reuse ``merged``; otherwise merged
        is None and the two-GEMM path is authoritative.
        """
        if adaln_params is None:
            if self.adaln_proj is None:
                raise ValueError("MiniMax H3 AdaLN cache parameters are required")
            adaln_params = self.adaln_proj(adaln_input)
        shift, scale = adaln_params
        if x.device.type == "mps":
            video = audio = None
            for start in range(0, x.shape[0], _MPS_MLP_TOKEN_CHUNK_SIZE):
                stop = min(start + _MPS_MLP_TOKEN_CHUNK_SIZE, x.shape[0])
                h = self.norm(x[start:stop])
                h = _modulate_scale_shift(
                    h,
                    shift,
                    scale,
                    inverse_indices[start:stop],
                    dtype=_BF16_DTYPE,
                ).to(_FP32_DTYPE)
                video_chunk, _ = self.video_out(h)
                audio_chunk, _ = self.audio_out(h)
                if video is None:
                    video = torch.empty(
                        (x.shape[0], video_chunk.shape[-1]),
                        dtype=video_chunk.dtype,
                        device=x.device,
                    )
                    audio = torch.empty(
                        (x.shape[0], audio_chunk.shape[-1]),
                        dtype=audio_chunk.dtype,
                        device=x.device,
                    )
                video[start:stop].copy_(video_chunk)
                audio[start:stop].copy_(audio_chunk)
                del h, video_chunk, audio_chunk
                torch.mps.synchronize()
                torch.mps.empty_cache()
            assert video is not None and audio is not None
            return video, audio, None
        h = self.norm(x)
        # Preserve full precision through both final output projections; the
        # fused kernel widens the modulated bf16 values to fp32 in one pass.
        h = _modulate_scale_shift_to_fp32(h, shift, scale, inverse_indices)
        merged_params = self._merged_out_params()
        if merged_params is None:
            video, _ = self.video_out(h)
            audio, _ = self.audio_out(h)
            return video, audio, None
        merged_weight, merged_bias = merged_params
        merged = nn.functional.linear(h, merged_weight, merged_bias)
        video, audio = merged.split(
            (
                self.video_out.output_size_per_partition,
                self.audio_out.output_size_per_partition,
            ),
            dim=-1,
        )
        return video, audio, merged


class MiniMaxH3DiTModel(BaseDiT, LayerwiseOffloadableModuleMixin):
    _aliases = [
        "MiniMaxH3Transformer3DModel",
        "MiniMaxH3PrunedTransformer3DModel",
    ]
    _fsdp_shard_conditions = [is_block]
    # refine_prompt_embeds drives a forward pass outside __call__.
    _fsdp_forward_methods = ("refine_prompt_embeds",)
    # parameters mix fp32 (patch projections, timestep embedder, and output
    # heads) with bf16 blocks; FSDP must gather in each parameter's own dtype
    _fsdp_mixed_dtype_params = True
    mps_stream_non_layer_weights = True
    _compile_conditions = [is_block]
    param_names_mapping = _ARCH_DEFAULTS.param_names_mapping
    reverse_param_names_mapping = _ARCH_DEFAULTS.reverse_param_names_mapping
    lora_param_names_mapping = _ARCH_DEFAULTS.lora_param_names_mapping

    def prepare_lora_adapter(
        self, adapter: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Project released-checkpoint AdaLN LoRAs onto pruned coordinates."""
        full_width = self.arch.adaln_affine_input_dim
        if full_width is None:
            return adapter

        suffix = ".adaln_proj.linear.lora_A"
        a_keys = sorted(key for key in adapter if key.endswith(suffix))
        if not a_keys:
            return adapter
        widths = {int(adapter[key].shape[-1]) for key in a_keys}
        if widths == {self.arch.time_embed_dim}:
            return adapter
        if widths != {full_width}:
            raise ValueError(
                "MiniMax H3 pruned AdaLN LoRA inputs must be uniformly "
                f"{self.arch.time_embed_dim} or {full_width} wide, got "
                f"{sorted(widths)}."
            )

        basis = self.adaln_basis
        mean = self.adaln_mean
        assert basis is not None and mean is not None
        if isinstance(basis, DTensor):
            basis = basis.full_tensor()
            mean = mean.full_tensor()
        if torch.count_nonzero(basis).item() == 0:
            raise ValueError(
                "MiniMax H3 pruned LoRA projection requires adaln_basis and "
                "adaln_mean from the component checkpoint."
            )

        projected = dict(adapter)
        work_device = adapter[a_keys[0]].device
        work_basis = basis.to(device=work_device, dtype=torch.float64)
        work_mean = mean.to(device=work_device, dtype=torch.float64)
        for a_key in a_keys:
            b_key = a_key[: -len("lora_A")] + "lora_B"
            if b_key not in adapter:
                raise ValueError(f"MiniMax H3 AdaLN LoRA is missing {b_key!r}.")
            a = adapter[a_key]
            b = adapter[b_key]
            a64 = a.to(torch.float64)
            b64 = b.to(device=work_device, dtype=torch.float64)
            projected[a_key] = (a64 @ work_basis.T).to(torch.float32)
            projected[a_key[: -len("lora_A")] + "lora_output_offset"] = (
                b64 @ (a64 @ work_mean)
            ).to(torch.float32)

        logger.info(
            "Projected %d MiniMax H3 AdaLN LoRA modules from width %d to %d",
            len(a_keys),
            full_width,
            self.arch.time_embed_dim,
        )
        return projected

    def prepare_adaln_plans(self, step_timesteps: list[torch.Tensor]) -> None:
        """Fill the AdaLN cache for this request before denoising starts.

        No-op for a prebuilt sidecar; the rebuild path needs the model's own
        timestep embedding so a filled plan is bit-identical to what resident
        adaln_proj weights would have produced.
        """
        if self.adaln_cache is None or self.adaln_cache.weight_files is None:
            return

        def embed(timesteps: torch.Tensor) -> torch.Tensor:
            return nn.functional.silu(self.time_embedder(timesteps)).to(_BF16_DTYPE)

        self.adaln_cache.build(step_timesteps, embed=embed)

    def _can_batch_block_adaln(self) -> bool:
        return (
            self.adaln_cache is None
            and get_tp_world_size() > 1
            and not torch.compiler.is_compiling()
            and not envs.SGLANG_CACHE_DIT_ENABLED
            and not hasattr(self, "_sglang_cache_dit_adapter")
            and not is_layerwise_offloaded_module(self)
            and all(type(block) is MiniMaxH3DiTBlock for block in self.blocks)
        )

    def _stacked_norm_weights(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        weights = [
            weight
            for block in self.blocks
            for weight in (block.norm1.weight, block.norm2.weight)
        ]
        if any(
            weight.dtype is not _BF16_DTYPE or isinstance(weight, DTensor)
            for weight in weights
        ):
            return None
        # Keyed on storage pointer and in-place version so weight reloads and
        # device moves rebuild the stacks instead of leaving them stale.
        key = tuple(
            value
            for weight in weights
            for value in (weight.data_ptr(), weight._version)
        )
        if key != self._fused_adaln_gamma_key:
            self._fused_adaln_gamma1 = torch.stack(
                [block.norm1.weight.detach() for block in self.blocks]
            ).to(_FP32_DTYPE)
            self._fused_adaln_gamma2 = torch.stack(
                [block.norm2.weight.detach() for block in self.blocks]
            ).to(_FP32_DTYPE)
            self._fused_adaln_gamma_key = key
        return self._fused_adaln_gamma1, self._fused_adaln_gamma2

    def _fused_adaln_block_params(
        self,
        block_adaln_params: tuple[tuple[torch.Tensor, ...], ...] | None,
        hidden: torch.Tensor,
        indices: torch.Tensor,
    ) -> tuple[tuple[torch.Tensor, ...], ...] | None:
        """Per-step AdaLN parameters for the fused block loop.

        Default (``MINIMAX_H3_FUSED_ADALN_BITEXACT``): per-site bitexact
        ``(gamma, scale)`` pairs, unmerged -- the aten-order kernel reproduces
        every eager rounding boundary, which forbids the ``w_eff`` premerge
        (the gamma multiply must round to bf16 inside the norm). Opting out
        folds each block's RMSNorm weight into the step's ``(1 + scale)``
        rows -- ``w_* = gamma * (1 + scale_*)`` in fp32, ~50 us over the
        50-block slab -- for the faster near-lossless kernels. Returns None
        whenever the fused chain cannot engage (flag off, fp32 curve-AdaLN
        island, Cache-DiT / offload wrappers, non-CUDA input, compile,
        missing bitexact backend), which keeps the eager loop fully
        authoritative.
        """
        if (
            block_adaln_params is None
            or not envs.MINIMAX_H3_FUSED_ADALN
            or not hidden.is_cuda
            or hidden.dtype is not _BF16_DTYPE
            or not hidden.is_contiguous()
            or torch.compiler.is_compiling()
            or envs.SGLANG_CACHE_DIT_ENABLED
            or hasattr(self, "_sglang_cache_dit_adapter")
            or is_layerwise_offloaded_module(self)
            or any(
                type(block) is not MiniMaxH3DiTBlock
                or block.preserve_input_for_cache_dit
                for block in self.blocks
            )
            or any(param.dtype is not _BF16_DTYPE for param in block_adaln_params[0])
        ):
            return None
        if envs.MINIMAX_H3_FUSED_ADALN_BITEXACT:
            return self._log_fused_adaln_engaged(
                self._bitexact_adaln_block_params(block_adaln_params, hidden, indices),
                mode="bitexact",
            )
        gammas = self._stacked_norm_weights()
        if gammas is None:
            return None
        gamma1, gamma2 = gammas
        scale_msa = torch.stack([params[1] for params in block_adaln_params])
        scale_mlp = torch.stack([params[4] for params in block_adaln_params])
        w_msa = (scale_msa.to(_FP32_DTYPE) + 1.0).mul_(gamma1.unsqueeze(1))
        w_mlp = (scale_mlp.to(_FP32_DTYPE) + 1.0).mul_(gamma2.unsqueeze(1))
        return self._log_fused_adaln_engaged(
            tuple(
                (w_msa[index], params[0], params[2], w_mlp[index], params[3], params[5])
                for index, params in enumerate(block_adaln_params)
            ),
            mode="merged w_eff",
        )

    def _log_fused_adaln_engaged(
        self,
        fused: tuple[tuple[torch.Tensor, ...], ...] | None,
        *,
        mode: str,
    ) -> tuple[tuple[torch.Tensor, ...], ...] | None:
        """One-shot serving-visibility marker: without it a silent gate
        failure (fused loop falling back to eager) is indistinguishable from
        the fused chain running."""
        if fused is not None and not self._fused_adaln_mode_logged:
            self._fused_adaln_mode_logged = True
            logger.info("MiniMax H3 fused adaLN chain engaged (%s)", mode)
        return fused

    def _bitexact_adaln_block_params(
        self,
        block_adaln_params: tuple[tuple[torch.Tensor, ...], ...],
        hidden: torch.Tensor,
        indices: torch.Tensor,
    ) -> tuple[tuple[torch.Tensor, ...], ...] | None:
        """Bitexact fused params: ``((gamma, scale), shift, gate, ...)`` per
        block, gated on the C++ aten-order backend accepting the serving
        shape (probed once on the first block's rows)."""
        if any(
            norm.weight.dtype is not _BF16_DTYPE or isinstance(norm.weight, DTensor)
            for block in self.blocks
            for norm in (block.norm1, block.norm2)
        ):
            return None
        rows = tuple(
            tuple(row.contiguous() for row in params) for params in block_adaln_params
        )
        first = rows[0]
        if not can_use_rmsnorm_indexed_scale_shift_bitexact_cuda(
            hidden, self.blocks[0].norm1.weight, first[1], first[0], indices
        ):
            return None
        return tuple(
            (
                (block.norm1.weight, params[1]),
                params[0],
                params[2],
                (block.norm2.weight, params[4]),
                params[3],
                params[5],
            )
            for block, params in zip(self.blocks, rows)
        )

    def _validate_tp_config(
        self, *, arch: MiniMaxH3DiTArchConfig, tp_size: int
    ) -> None:
        if tp_size <= 0:
            raise ValueError("TP size must be positive.")
        if arch.num_attention_heads <= 0:
            raise ValueError("num_attention_heads must be positive.")
        if arch.hidden_size <= 0:
            raise ValueError("hidden_size must be positive.")
        if arch.attention_head_dim <= 0:
            raise ValueError("attention_head_dim must be positive.")
        if arch.ffn_hidden_size <= 0:
            raise ValueError("ffn_hidden_size must be positive.")
        for name, value in (
            ("num_attention_heads", arch.num_attention_heads),
            ("hidden_size", arch.hidden_size),
            ("ffn_hidden_size", arch.ffn_hidden_size),
            ("time_embed_hidden_size", arch.time_embed_hidden_size),
            ("adaln_out_features", arch.adaln_out_features),
            ("final_adaln_out_features", arch.final_adaln_out_features),
            ("video_patch_output_dim", arch.latents_dim * math.prod(arch.patch_size)),
            ("audio_patch_output_dim", arch.audio_latents_dim),
        ):
            if value % tp_size:
                raise ValueError(
                    f"MiniMax H3 {name}={value} must be divisible by "
                    f"TP size {tp_size}."
                )

    @staticmethod
    def _validate_sequence_parallel_config(
        *,
        arch: MiniMaxH3DiTArchConfig,
        tp_size: int,
        ulysses_size: int,
        ring_size: int,
    ) -> None:
        if ulysses_size <= 0:
            raise ValueError("MiniMax H3 Ulysses size must be positive.")
        if ring_size <= 0:
            raise ValueError("MiniMax H3 ring size must be positive.")
        local_heads = arch.num_attention_heads // tp_size
        if local_heads % ulysses_size:
            raise ValueError(
                f"MiniMax H3 TP-local heads {local_heads} must be divisible by "
                f"Ulysses size {ulysses_size} (total heads="
                f"{arch.num_attention_heads}, TP={tp_size})."
            )
        # ring never shards heads (only rows), so it has no head-divisibility
        # constraint; the packed sequence alignment constant must still
        # divide the *combined* sequence-parallel size, since ring adds an
        # outer row split on top of Ulysses's inner one (see forward()).
        sp_size = ulysses_size * ring_size
        if MINIMAX_H3_PACKED_SEQUENCE_ALIGNMENT % sp_size:
            raise ValueError(
                "MiniMax H3 packed sequence alignment "
                f"{MINIMAX_H3_PACKED_SEQUENCE_ALIGNMENT} must be divisible by "
                f"the combined sequence-parallel size {sp_size} "
                f"(ulysses={ulysses_size} x ring={ring_size}). Choose degrees "
                "whose product divides both the TP-local attention heads and "
                "the packed sequence alignment."
            )

    def __init__(
        self,
        config: MiniMaxH3DiTConfig,
        hf_config: dict[str, Any],
        quant_config: QuantizationConfig | None = None,
        adaln_cache_path: str | None = None,
        adaln_cache_model_variant: str | None = None,
        adaln_weight_files: list[str] | None = None,
        adaln_plan_width: int = MINIMAX_H3_ADALN_MAX_PLAN_WIDTH,
    ) -> None:
        super().__init__(config=config, hf_config=hf_config)
        arch = self.config
        if (
            adaln_cache_path is not None or adaln_weight_files is not None
        ) and quant_config is not None:
            raise ValueError(
                "MiniMax H3 AdaLN cache is only compatible with unquantized weights"
            )
        if arch.adaln_curve_grid is not None and (
            adaln_cache_path is not None or adaln_weight_files is not None
        ):
            raise ValueError(
                "MiniMax H3 pruned curve checkpoints cannot use a separate "
                "AdaLN cache"
            )
        self._adaln_precomputed = (
            adaln_cache_path is not None or adaln_weight_files is not None
        )
        self.arch = arch
        if arch.checkpoint_uses_diffusers_layout:
            self.preprocess_loaded_state_dict = _diffusers_h3_checkpoint
        self.hidden_size = arch.hidden_size
        self.num_attention_heads = arch.num_attention_heads
        self.num_channels_latents = arch.latents_dim
        tp_size = get_tp_world_size()
        ulysses_size, _ = get_ulysses_ctx()
        self._validate_tp_config(arch=arch, tp_size=tp_size)
        self._validate_sequence_parallel_config(
            arch=arch,
            tp_size=tp_size,
            ulysses_size=ulysses_size,
            ring_size=get_ring_ctx()[0],
        )

        self.video_patch_proj = ColumnParallelLinear(
            arch.latents_dim
            * arch.patch_size[0]
            * arch.patch_size[1]
            * arch.patch_size[2],
            arch.hidden_size,
            bias=True,
            gather_output=True,
            params_dtype=_FP32_DTYPE,
            quant_config=None,
            prefix="video_patch_proj",
        )
        self.audio_patch_proj = ColumnParallelLinear(
            arch.audio_latents_dim,
            arch.hidden_size,
            bias=True,
            gather_output=True,
            params_dtype=_FP32_DTYPE,
            quant_config=None,
            prefix="audio_patch_proj",
        )
        self.condition_proj = ColumnParallelLinear(
            arch.text_dim,
            arch.hidden_size,
            bias=True,
            gather_output=True,
            params_dtype=_BF16_DTYPE,
            quant_config=quant_config,
            prefix="condition_proj",
        )
        if arch.adaln_curve_grid is None:
            self.time_embedder = MiniMaxH3TimeEmbedder(
                arch,
                prefix="time_embedder",
            )
            self.register_parameter("adaln_t_table", None)
        else:
            self.time_embedder = None
            self.adaln_t_table = nn.Parameter(
                torch.empty(
                    arch.adaln_curve_grid,
                    arch.time_embed_dim,
                    dtype=_FP32_DTYPE,
                ),
                requires_grad=False,
            )
        if arch.adaln_affine_input_dim is None:
            self.register_parameter("adaln_basis", None)
            self.register_parameter("adaln_mean", None)
        else:
            self.register_parameter(
                "adaln_basis",
                nn.Parameter(
                    torch.empty(
                        arch.time_embed_dim,
                        arch.adaln_affine_input_dim,
                        dtype=_FP32_DTYPE,
                    ),
                    requires_grad=False,
                ),
            )
            self.register_parameter(
                "adaln_mean",
                nn.Parameter(
                    torch.empty(arch.adaln_affine_input_dim, dtype=_FP32_DTYPE),
                    requires_grad=False,
                ),
            )
        self.rope = MiniMaxH3Rope(arch.rope_inv_freq_len)
        self.token_refiner = MiniMaxH3TokenRefiner(
            arch,
            quant_config,
            prefix="token_refiner",
        )
        self.blocks = nn.ModuleList(
            [
                MiniMaxH3DiTBlock(
                    arch,
                    quant_config,
                    prefix=f"blocks.{index}",
                    use_adaln_cache=self._adaln_precomputed,
                )
                for index in range(arch.num_layers)
            ]
        )
        self.layer_names = ["token_refiner.blocks", "blocks"]
        self.final_layer = MiniMaxH3FinalLayer(
            arch,
            quant_config,
            prefix="final_layer",
            use_adaln_cache=self._adaln_precomputed,
        )
        self.adaln_cache = (
            MiniMaxH3AdalnCache(
                arch,
                path=adaln_cache_path,
                model_variant=adaln_cache_model_variant,
                weight_files=adaln_weight_files,
                max_plan_width=adaln_plan_width,
            )
            if self._adaln_precomputed
            else None
        )
        # Fused-adaLN loop state: fp32 [num_layers, H] stacks of the block
        # norm weights, keyed like MiniMaxH3FinalLayer._merged_out_params.
        self._fused_adaln_gamma_key: tuple[int, ...] | None = None
        self._fused_adaln_gamma1: torch.Tensor | None = None
        self._fused_adaln_gamma2: torch.Tensor | None = None
        self._fused_adaln_mode_logged = False
        # Component overrides disappear when the loader context exits. Preserve
        # only that selection; process-wide overrides are resolved at first use.
        self._component_attention_backend_override = (
            claim_deferred_component_attn_backend()
        )
        self._resolved_attention_backend: AttentionBackendEnum | None = None
        self._mark_missing_params_required()

    def set_cache_dit_input_preservation(self, enabled: bool) -> None:
        """Stop the blocks from overwriting the input Cache-DiT holds by reference.

        Cache-DiT snapshots the block-stack input to measure its residuals, so a
        block that rewrites its own input in place makes that residual read as
        zero. Only the first gated residual of a block writes the block input;
        the second one operates on a buffer this block just allocated, so it is
        left on the in-place fused path either way.

        The caller owns the lifecycle. It has to be on before Cache-DiT mounts,
        because mounting replaces `blocks` with a wrapper and the real blocks
        stop being reachable by iterating it.
        """
        for block in self.blocks:
            block.preserve_input_for_cache_dit = enabled

    def _resolve_attention_backend_once(self) -> None:
        if self._resolved_attention_backend is not None:
            return
        selected_backend = (
            get_global_forced_attn_backend()
            or self._component_attention_backend_override
        )
        backend = get_attn_backend(
            self.arch.attention_head_dim,
            _BF16_DTYPE,
            selected_attention_backend=selected_backend,
            attention_requirements=AttentionRequirements(packed_varlen=True),
        )
        for module in self.modules():
            if isinstance(module, MiniMaxH3Attention):
                module._set_attention_backend(backend)
        self._resolved_attention_backend = backend.get_enum()

    def _mark_missing_params_required(self) -> None:
        for _, param in self.named_parameters():
            # A quant method's create_weights() declares its own policy for scales
            # it can synthesize (weight-only NVFP4 has no input_scale and marks it
            # "ones"); claiming only undeclared params keeps that intact.
            if getattr(param, "missing_param_init", None) is None:
                param.missing_param_init = "error"

    def post_load_weights(self) -> None:
        fp32_param_names = list(_MINIMAX_H3_FP32_PARAM_NAMES_IN_MODEL_ORDER)
        if self.adaln_t_table is not None:
            fp32_param_names = [
                name
                for name in fp32_param_names
                if not name.startswith("time_embedder.")
            ]
            fp32_param_names.append("adaln_t_table")
            if self.adaln_basis is not None:
                fp32_param_names.extend(("adaln_basis", "adaln_mean"))
        for name in fp32_param_names:
            param = self.get_parameter(name)
            if param.dtype != _FP32_DTYPE:
                raise ValueError(
                    f"{name} must stay fp32 after load, got {param.dtype}."
                )
        if self.adaln_t_table is not None:
            for name, param in self.named_parameters():
                if ".adaln_proj.linear." in name and param.dtype != _FP32_DTYPE:
                    raise ValueError(
                        f"{name} must stay fp32 with curve AdaLN, got {param.dtype}."
                    )
        # assign=True loading may re-register this persistent buffer as a parameter
        rope_inv_freq = self.rope.inv_freq
        if rope_inv_freq.dtype != _FP32_DTYPE:
            raise ValueError(
                f"rope.inv_freq must stay fp32 after load, got {rope_inv_freq.dtype}."
            )
        if self.adaln_cache is not None:
            self.adaln_cache.load(self.video_patch_proj.weight.device)

    def _time_embedding(self, timesteps: torch.Tensor) -> torch.Tensor:
        if self.adaln_t_table is None:
            assert self.time_embedder is not None
            return self.time_embedder(timesteps)

        grid = self.adaln_t_table.shape[0]
        position = timesteps.to(_FP32_DTYPE).clamp(0, 1) * (grid - 1)
        lower = position.floor().clamp(max=grid - 2).to(torch.long)
        fraction = (position - lower).unsqueeze(-1)
        lower_value = self.adaln_t_table.index_select(0, lower)
        upper_value = self.adaln_t_table.index_select(0, lower + 1)
        return torch.lerp(lower_value, upper_value, fraction)

    @staticmethod
    def _pos_ids(pos_info: Any, key: str) -> torch.Tensor:
        if isinstance(pos_info, dict):
            ids = pos_info.get("position_ids")
        else:
            ids = getattr(pos_info, "position_ids", None)
        if ids is None:
            raise ValueError(f"{key}.position_ids is required")
        return ids.view(-1).to(torch.long)

    @staticmethod
    def _psp_field(psp: Any, key: str, field: str) -> Any:
        if isinstance(psp, dict):
            value = psp.get(field)
        else:
            value = getattr(psp, field, None)
        if value is None:
            raise ValueError(f"{key}.{field} is required")
        return value

    @staticmethod
    def _psp_optional_field(psp: Any, field: str) -> Any:
        if isinstance(psp, dict):
            return psp.get(field)
        return getattr(psp, field, None)

    def refine_prompt_embeds(
        self,
        prompt_embeds: torch.Tensor,
        refiner_cu_seqlens: torch.Tensor,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        """Project and refine request-static text conditioning once."""
        self.materialize_mps_non_layer_weights(
            "condition_proj", "token_refiner.final_norm"
        )
        text_len = int(refiner_cu_seqlens[1].item())
        if text_len <= 0 or text_len > int(prompt_embeds.shape[0]):
            raise ValueError(
                "refiner cu_seqlens live text length must be in "
                f"[1, {int(prompt_embeds.shape[0])}], got {text_len}"
            )
        text_rows = prompt_embeds[:text_len].to(device=device, dtype=_BF16_DTYPE)
        true_refiner_cu = torch.stack(
            (
                refiner_cu_seqlens[0],
                refiner_cu_seqlens[1],
                refiner_cu_seqlens[1],
            )
        )
        text_embed, _ = self.condition_proj(text_rows)
        refined = self.token_refiner(
            text_embed,
            cu_seqlens=true_refiner_cu,
            cu_seqlens_host=(0, text_len, text_len),
            max_seqlen=text_len,
        )
        self.release_mps_non_layer_weights("condition_proj", "token_refiner.final_norm")
        return refined

    def build_rope_cache(
        self,
        img_position_ids: torch.Tensor,
        *,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build request-static RoPE inputs for this rank's row shard.

        Same 2D row split as forward(): ring first (outer, contiguous
        ring_chunk_len slice), Ulysses second (inner slice within that
        chunk) -- see forward()'s row_start derivation for the identity
        this must stay in sync with.
        """
        self.materialize_mps_non_layer_weights("rope")
        if img_position_ids.dim() != 3 or img_position_ids.shape[0] != 1:
            raise ValueError(
                "img_position_ids must be [1, S, 3], got "
                f"{list(img_position_ids.shape)}"
            )
        seq_len = int(img_position_ids.shape[1])
        ulysses_ws, ulysses_rank = get_ulysses_ctx()
        ring_ws, ring_rank = get_ring_ctx()
        sp_ws = ulysses_ws * ring_ws
        if seq_len % sp_ws:
            raise ValueError(
                f"packed seq_len {seq_len} not divisible by the combined "
                f"sequence-parallel world size {sp_ws} "
                f"(ulysses={ulysses_ws} x ring={ring_ws})"
            )
        local_seq_len = seq_len // sp_ws
        ring_chunk_len = local_seq_len * ulysses_ws
        row_start = ring_rank * ring_chunk_len + ulysses_rank * local_seq_len
        rope_freqs = self.rope(
            img_position_ids[:, row_start : row_start + local_seq_len]
        ).to(device)
        result = (
            _rope_cos_sin_cache(rope_freqs, dtype=_BF16_DTYPE),
            torch.arange(
                local_seq_len,
                device=device,
                dtype=torch.long,
            ),
        )
        self.release_mps_non_layer_weights("rope")
        return result

    def _write_projected_latent_rows(
        self,
        embeddings: torch.Tensor,
        *,
        latents: torch.Tensor,
        proj: nn.Module,
        global_ids: torch.Tensor,
        row_ids: torch.Tensor,
        accumulate: bool = False,
    ) -> None:
        """Project selected latent rows and scatter them into the shard buffer."""
        if row_ids.numel() == 0:
            return
        # latent embedders stay fp32; only rows owned by this SP rank are
        # projected, then cast during scattering into the bf16 sequence
        rows = (
            latents.view(-1, latents.shape[-1])
            .index_select(0, global_ids)
            .to(_FP32_DTYPE)
        )
        projected, _ = proj(rows)
        write_rows = embeddings.index_add_ if accumulate else embeddings.index_copy_
        write_rows(0, row_ids, projected.to(_BF16_DTYPE))

    @eager_on_graph(True)
    def _embed(
        self,
        *,
        x: torch.Tensor,
        audio_x: torch.Tensor,
        text_embeddings_selected: torch.Tensor,
        unique_timesteps: torch.Tensor,
        img_pos: torch.Tensor,
        audio_pos: torch.Tensor,
        text_pos: torch.Tensor,
        refiner_cu_seqlens: torch.Tensor,
        refiner_max_seqlen: int,
        row_start: int,
        row_stop: int,
        device: torch.device,
        refined_prompt_embeds_length: int | torch.Tensor | None = None,
        local_embedding_layout: dict[str, Any] | None = None,
        skip_time_embedding: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Build embeddings for one contiguous block-stack row shard.

        Returns (decoder_input [S_local, H] bf16, t_emb [M, t_dim] fp32).
        t_emb is None with ``skip_time_embedding`` (resident AdaLN cache: every
        consumer reads cached params, so the embedding would be dead compute).
        """
        t_emb = None if skip_time_embedding else self._time_embedding(unique_timesteps)
        embed_cache = (
            None
            if local_embedding_layout is None
            else local_embedding_layout.get("embedding_cache")
        )
        embeddings = None if embed_cache is None else embed_cache.get("embeddings")
        if embeddings is not None:
            # Post-priming steps: text/condition/reference rows and the zero
            # padding tail persist from the priming forward; only target rows
            # of x/audio_x change per step (the denoise loop's x_buffer
            # priming contract), so only their projections are refreshed.
            self._write_projected_latent_rows(
                embeddings,
                latents=x,
                proj=self.video_patch_proj,
                global_ids=local_embedding_layout["img_target_global_ids"],
                row_ids=local_embedding_layout["img_target_row_ids"],
            )
            self._write_projected_latent_rows(
                embeddings,
                latents=audio_x,
                proj=self.audio_patch_proj,
                global_ids=local_embedding_layout["audio_target_global_ids"],
                row_ids=local_embedding_layout["audio_target_row_ids"],
            )
            return embeddings, t_emb
        # BCG pads the prompt tensor only to stabilize its input signature.
        # Raw-input callers recover the live length from refiner metadata;
        # request-static refined inputs carry it as a host integer and avoid a
        # per-step device scalar read. Running the refiner at the bucketed M
        # dimension changes GEMM selection and is not bitwise equivalent.
        if refined_prompt_embeds_length is None:
            text_len = int(refiner_cu_seqlens[1].item())
        elif torch.is_tensor(refined_prompt_embeds_length):
            # BCG turns this request-varying host constant into a scalar input
            # so different live lengths can replay one padded-text signature.
            # _embed is an eager graph break, so this value is read outside
            # captured CUDA graphs.
            text_len = int(refined_prompt_embeds_length.item())
        else:
            text_len = int(refined_prompt_embeds_length)
        if text_len <= 0 or text_len > int(text_embeddings_selected.shape[0]):
            raise ValueError(
                "refiner cu_seqlens live text length must be in "
                f"[1, {int(text_embeddings_selected.shape[0])}], got {text_len}"
            )
        text_pos = text_pos[:text_len]
        if refined_prompt_embeds_length is not None:
            text_embed = text_embeddings_selected[:text_len].to(
                device=device, dtype=_BF16_DTYPE
            )
            if int(text_embed.shape[-1]) != self.hidden_size:
                raise ValueError(
                    "refined prompt embeddings must have hidden width "
                    f"{self.hidden_size}, got {int(text_embed.shape[-1])}"
                )
        else:
            text_embed = self.refine_prompt_embeds(
                text_embeddings_selected,
                refiner_cu_seqlens,
                device=device,
            )

        local_seq_len = row_stop - row_start
        trusted_layout = local_embedding_layout is not None
        if trusted_layout:
            used_len = text_len + int(img_pos.numel()) + int(audio_pos.numel())
            local_live_rows = min(max(used_len - row_start, 0), local_seq_len)
            embeddings = torch.empty(
                (local_seq_len, self.hidden_size), device=device, dtype=_BF16_DTYPE
            )
            if local_live_rows < local_seq_len:
                embeddings[local_live_rows:].zero_()
        else:
            # Direct model callers do not provide the serving-time partition
            # contract. Preserve their historical zero-fill/add semantics for
            # sparse or overlapping row maps.
            embeddings = torch.zeros(
                (local_seq_len, self.hidden_size), device=device, dtype=_BF16_DTYPE
            )

        if local_embedding_layout is None:
            text_source_ids = torch.nonzero(
                (text_pos >= row_start) & (text_pos < row_stop),
                as_tuple=False,
            ).view(-1)
            text_row_ids = text_pos.index_select(0, text_source_ids) - row_start
            img_global_ids = img_pos.index_select(
                0,
                torch.nonzero(
                    (img_pos >= row_start) & (img_pos < row_stop),
                    as_tuple=False,
                ).view(-1),
            )
            img_row_ids = img_global_ids - row_start
            audio_global_ids = audio_pos.index_select(
                0,
                torch.nonzero(
                    (audio_pos >= row_start) & (audio_pos < row_stop),
                    as_tuple=False,
                ).view(-1),
            )
            audio_row_ids = audio_global_ids - row_start
        else:
            text_source_start = int(local_embedding_layout["text_source_start"])
            text_source_stop = int(local_embedding_layout["text_source_stop"])
            img_global_ids = local_embedding_layout["img_global_ids"]
            img_row_ids = local_embedding_layout["img_row_ids"]
            audio_global_ids = local_embedding_layout["audio_global_ids"]
            audio_row_ids = local_embedding_layout["audio_row_ids"]

        write_rows = embeddings.index_copy_ if trusted_layout else embeddings.index_add_
        if trusted_layout:
            text_rows = text_source_stop - text_source_start
            if text_rows:
                embeddings[:text_rows].copy_(
                    text_embed[text_source_start:text_source_stop]
                )
        elif text_row_ids.numel():
            write_rows(
                0,
                text_row_ids,
                text_embed.index_select(0, text_source_ids).to(_BF16_DTYPE),
            )

        self._write_projected_latent_rows(
            embeddings,
            latents=x,
            proj=self.video_patch_proj,
            global_ids=img_global_ids,
            row_ids=img_row_ids,
            accumulate=not trusted_layout,
        )
        self._write_projected_latent_rows(
            embeddings,
            latents=audio_x,
            proj=self.audio_patch_proj,
            global_ids=audio_global_ids,
            row_ids=audio_row_ids,
            accumulate=not trusted_layout,
        )

        if embed_cache is not None:
            embed_cache["embeddings"] = embeddings
        return embeddings, t_emb

    def forward(self, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Packed inference forward.

        Keyword names follow the checkpoint's serving contract.
        Returns `(video_logits, audio_logits)` from rows selected by
        `img_pos_for_infer_output_info` and `audio_pos_info`, with condition
        rows zeroed by update masks.
        """
        # Strict keyword contract: refuse any kwarg forward does not consume.
        unexpected = sorted(set(kwargs) - _FORWARD_SUPPORTED_KWARGS)
        if unexpected:
            raise TypeError(
                "MiniMaxH3DiTModel.forward received unexpected kwargs: "
                f"{unexpected}; supported kwargs: "
                f"{sorted(_FORWARD_SUPPORTED_KWARGS)}"
            )

        x = _required_kwarg(kwargs, "x")
        audio_x = _required_kwarg(kwargs, "audio_x")
        img_position_ids = _required_kwarg(kwargs, "img_position_ids")
        unique_timesteps = _required_kwarg(kwargs, "unique_timesteps")
        inverse_indices = (
            _required_kwarg(kwargs, "inverse_indices").view(-1).to(torch.long)
        )
        update_mask = _required_kwarg(kwargs, "update_mask")
        subblock_sparse_query_block_mask = kwargs.get(
            "subblock_sparse_query_block_mask"
        )
        block_token_tags = kwargs.get("block_token_tags")
        token_tags = kwargs.get("token_tags")
        if block_token_tags is None:
            token_tags = _required_kwarg(kwargs, "token_tags").view(-1).to(torch.long)
        else:
            block_token_tags = block_token_tags.view(-1).to(torch.long)
            token_tags = None
        skip_mask_out_condition = bool(kwargs.get("skip_mask_out_condition", False))

        text_selected = _required_kwarg(kwargs, "prompt_embeds")

        img_pos = self._pos_ids(_required_kwarg(kwargs, "img_pos_info"), "img_pos_info")
        audio_pos = self._pos_ids(
            _required_kwarg(kwargs, "audio_pos_info"), "audio_pos_info"
        )
        text_pos = self._pos_ids(
            _required_kwarg(kwargs, "text_pos_info"),
            "text_pos_info",
        )
        infer_out_pos = self._pos_ids(
            _required_kwarg(kwargs, "img_pos_for_infer_output_info"),
            "img_pos_for_infer_output_info",
        )

        psp = _required_kwarg(kwargs, "packed_seq_params")
        cu_seqlens = self._psp_field(psp, "packed_seq_params", "cu_seqlens_q").to(
            torch.int32
        )
        raw_cu_seqlens_host = self._psp_optional_field(psp, "cu_seqlens_q_host")
        cu_seqlens_host = tuple(
            int(value)
            for value in (
                cu_seqlens.tolist()
                if raw_cu_seqlens_host is None
                else raw_cu_seqlens_host
            )
        )
        # max_seqlen_q is set to cu_seqlens[1] (`used`, the real/non-padding
        # row count) by construction -- already a plain host int here, so
        # ring can reuse it as real_seq_len below with no new device sync.
        max_seqlen = int(self._psp_field(psp, "packed_seq_params", "max_seqlen_q"))
        refiner_psp = _required_kwarg(kwargs, "refiner_packed_seq_params")
        refiner_cu = self._psp_field(
            refiner_psp, "refiner_packed_seq_params", "cu_seqlens_q"
        ).to(torch.int32)
        refiner_max = int(
            self._psp_field(refiner_psp, "refiner_packed_seq_params", "max_seqlen_q")
        )

        if x.dim() != 3 or x.shape[0] != 1:
            raise ValueError(f"x must be [1, S, C], got {list(x.shape)}")
        seq_len = int(x.shape[1])
        if token_tags is not None and token_tags.shape[0] != seq_len:
            raise ValueError(
                "token_tags must cover the full packed sequence "
                f"({seq_len}), got {token_tags.shape[0]}."
            )
        if inverse_indices.shape[0] != seq_len:
            raise ValueError(
                f"inverse_indices must be [{seq_len}], got {list(inverse_indices.shape)}"
            )
        device = x.device
        if subblock_sparse_query_block_mask is not None and not isinstance(
            subblock_sparse_query_block_mask, torch.Tensor
        ):
            raise ValueError("subblock_sparse_query_block_mask must be a tensor")
        self._resolve_attention_backend_once()

        # Row split is 2D: ring first (an outer, contiguous ring_chunk_len
        # slice of the packed sequence), Ulysses second (an inner slice
        # within this rank's ring chunk). Only Ulysses shards heads inside
        # attention -- ring instead ring-rotates each rank's local KV chunk
        # and online-softmax merges partial outputs (see
        # _minimax_h3_attention_core_impl), so it has no head constraint.
        ulysses_ws, ulysses_rank = get_ulysses_ctx()
        ring_ws, ring_rank = get_ring_ctx()
        sp_ws = ulysses_ws * ring_ws
        local_seq_len = seq_len
        if sp_ws > 1:
            if seq_len % sp_ws:
                raise ValueError(
                    f"packed seq_len {seq_len} not divisible by the combined "
                    f"sequence-parallel world size {sp_ws} "
                    f"(ulysses={ulysses_ws} x ring={ring_ws})"
                )
            local_heads = self.num_attention_heads // get_tp_world_size()
            if local_heads % ulysses_ws:
                raise ValueError(
                    f"TP-local heads {local_heads} not divisible by Ulysses "
                    f"world size {ulysses_ws} (total heads="
                    f"{self.num_attention_heads}, TP={get_tp_world_size()})"
                )
            local_seq_len = seq_len // sp_ws
        ring_chunk_len = local_seq_len * ulysses_ws
        row_start = ring_rank * ring_chunk_len + ulysses_rank * local_seq_len
        row_stop = row_start + local_seq_len

        # RoPE and latent projections are row-local before Ulysses exchanges
        # sequence for heads inside attention. Serving normally prepares the
        # request-static cache once; direct model callers use this fallback.
        rope_cache = kwargs.get("rope_cache")
        if rope_cache is None:
            self.materialize_mps_non_layer_weights("rope")
            rope_freqs = self.rope(img_position_ids[:, row_start:row_stop]).to(device)
            rope_cache = (
                _rope_cos_sin_cache(rope_freqs, dtype=_BF16_DTYPE),
                torch.arange(
                    local_seq_len,
                    device=device,
                    dtype=torch.long,
                ),
            )
            self.release_mps_non_layer_weights("rope")
        self.materialize_mps_non_layer_weights(*_MPS_EMBED_WEIGHT_PREFIXES)
        img_pos = img_pos.to(device)
        audio_pos = audio_pos.to(device)
        text_pos = text_pos.to(device)

        decoder_input, t_emb = self._embed(
            x=x,
            audio_x=audio_x,
            text_embeddings_selected=text_selected,
            unique_timesteps=unique_timesteps.view(-1).to(device),
            img_pos=img_pos,
            audio_pos=audio_pos,
            text_pos=text_pos,
            refiner_cu_seqlens=refiner_cu.to(device),
            refiner_max_seqlen=refiner_max,
            row_start=row_start,
            row_stop=row_stop,
            device=device,
            refined_prompt_embeds_length=kwargs.get("refined_prompt_embeds_length"),
            local_embedding_layout=kwargs.get("local_embedding_layout"),
            skip_time_embedding=self.adaln_cache is not None,
        )
        self.release_mps_non_layer_weights(*_MPS_EMBED_WEIGHT_PREFIXES)
        # request-step AdaLN input shared by all blocks; None with a resident
        # AdaLN cache, whose consumers only need the unique-timestep count
        if t_emb is None:
            adaln_input = None
        else:
            adaln_input = (
                t_emb
                if self.adaln_t_table is not None
                else nn.functional.silu(t_emb).to(_BF16_DTYPE)
            )
        num_unique_timesteps = int(unique_timesteps.view(-1).shape[0])
        inverse_indices = inverse_indices.to(device)
        block_inverse = inverse_indices[row_start:row_stop]
        if block_token_tags is None:
            assert token_tags is not None
            token_tags = token_tags.to(device)
            block_token_tags = token_tags[row_start:row_stop].clamp(min=0)
        else:
            block_token_tags = block_token_tags.to(device)
            if block_token_tags.shape[0] != local_seq_len:
                raise ValueError(
                    "block_token_tags must cover the rank-local packed sequence "
                    f"({local_seq_len}), got {block_token_tags.shape[0]}."
                )
        block_combined = kwargs.get("block_combined_indices")
        if block_combined is None:
            block_combined = torch.add(
                block_token_tags,
                block_inverse,
                alpha=MINIMAX_H3_ADALN_MODALITY_NUM,
            )

        hidden = decoder_input
        cu_seqlens = cu_seqlens.to(device)
        block_adaln_params = None
        adaln_cache_plan_index = None
        if self.adaln_cache is not None:
            adaln_cache_plan_index = self.adaln_cache.lookup(
                unique_timesteps.view(-1).to(device)
            )
            block_adaln_params = self.adaln_cache.block_all(
                adaln_cache_plan_index,
                num_unique_timesteps,
            )
        elif self._can_batch_block_adaln():
            local_adaln = torch.stack(
                [block.adaln_proj.project_local(adaln_input) for block in self.blocks]
            )
            gathered_adaln = tensor_model_parallel_all_gather(local_adaln)
            block_adaln_params = tuple(
                block.adaln_proj.split_output(output)
                for block, output in zip(self.blocks, gathered_adaln)
            )
        # With sequence parallelism, shard rows across the group for the
        # block stack. Attention trades sequence for heads internally
        # (Ulysses) and/or ring-rotates KV across ring ranks; everything
        # else, including the final layer, is row-local. Only the narrow
        # video/audio logits are gathered after the final layer.
        fused_block_adaln = self._fused_adaln_block_params(
            block_adaln_params, hidden, block_combined
        )
        if fused_block_adaln is not None:
            # Each block defers its trailing gated-residual add into the next
            # block's fused norm (Plan B); block 0's first norm has no
            # preceding gate and the last block's trailing gate runs eagerly
            # because the final layer keeps its own norm path.
            pending_gate = None
            for index, block in enumerate(self.blocks):
                hidden, pending_gate = block.forward_fused(
                    hidden,
                    fused_adaln=fused_block_adaln[index],
                    pending_gate=pending_gate,
                    combined_indices=block_combined,
                    rope_cache=rope_cache,
                    cu_seqlens=cu_seqlens,
                    cu_seqlens_host=cu_seqlens_host,
                    max_seqlen=max_seqlen,
                    subblock_sparse_query_block_mask=subblock_sparse_query_block_mask,
                    ulysses_active=ulysses_ws > 1,
                    ring_active=ring_ws > 1,
                )
            last_gate, last_update = pending_gate
            hidden = _modulate_gate(
                hidden, last_gate, last_update, block_combined, dtype=_BF16_DTYPE
            )
        else:
            for index, block in enumerate(self.blocks):
                hidden = block(
                    hidden,
                    adaln_input=adaln_input,
                    combined_indices=block_combined,
                    rope_cache=rope_cache,
                    cu_seqlens=cu_seqlens,
                    cu_seqlens_host=cu_seqlens_host,
                    max_seqlen=max_seqlen,
                    subblock_sparse_query_block_mask=subblock_sparse_query_block_mask,
                    ulysses_active=ulysses_ws > 1,
                    ring_active=ring_ws > 1,
                    adaln_params=(
                        None
                        if block_adaln_params is None
                        else block_adaln_params[index]
                    ),
                )
        self.materialize_mps_non_layer_weights("final_layer")
        video_logits, audio_logits, merged_logits = self.final_layer(
            hidden,
            adaln_input=adaln_input,
            inverse_indices=block_inverse,
            adaln_params=(
                None
                if adaln_cache_plan_index is None
                else self.adaln_cache.final(
                    adaln_cache_plan_index,
                    num_unique_timesteps,
                )
            ),
        )
        self.release_mps_non_layer_weights("final_layer")
        if sp_ws > 1:
            from sglang.multimodal_gen.runtime.distributed.parallel_state import (
                get_sp_group,
            )

            video_width = video_logits.shape[-1]
            if merged_logits is None:
                merged_logits = torch.cat((video_logits, audio_logits), dim=-1)
            logits = get_sp_group().all_gather(merged_logits, dim=0)
            video_logits, audio_logits = logits.split(
                (video_width, logits.shape[-1] - video_width), dim=-1
            )

        # Preserve the full-row output GEMM (and therefore its numerical
        # contract), but defer TP column gathers until after dead text/padding
        # rows have been removed. For hybrid TP+Ulysses, the preceding SP row
        # gather also carries only the TP-local output width.
        video_logits = video_logits.index_select(0, infer_out_pos.to(device))
        audio_logits = audio_logits.index_select(0, audio_pos.to(device))
        if get_tp_world_size() > 1:
            video_logits = tensor_model_parallel_all_gather(video_logits)
            audio_logits = tensor_model_parallel_all_gather(audio_logits)
        if not skip_mask_out_condition:
            update_mask = update_mask.view(-1).to(device)
            if update_mask.shape[0] != video_logits.shape[0]:
                raise ValueError(
                    "update_mask length mismatch: "
                    f"{update_mask.shape[0]} != {video_logits.shape[0]}"
                )
            video_logits = video_logits * update_mask.unsqueeze(-1)
            # Audio has no condition rows in the supported tasks, so its
            # derived update mask is all ones. Honor an explicit mask when
            # provided.
            update_audio_mask = kwargs.get("update_audio_mask")
            if update_audio_mask is not None:
                audio_logits = audio_logits * update_audio_mask.view(-1).unsqueeze(-1)
        return video_logits, audio_logits


EntryClass = MiniMaxH3DiTModel

__all__ = [
    "MINIMAX_H3_FP32_BUFFER_NAMES",
    "MINIMAX_H3_FP32_PARAM_NAMES",
    "MiniMaxH3DiTModel",
    "_reorder_grouped_qkv_to_qkv",
]
