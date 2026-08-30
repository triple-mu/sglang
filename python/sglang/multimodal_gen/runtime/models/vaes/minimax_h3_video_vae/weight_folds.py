# SPDX-License-Identifier: Apache-2.0
"""Load-time algebraic weight folds for the MiniMax H3 visual VAE.

``AutoencoderKLLegacy._load_from_state_dict`` applies these folds to the
incoming checkpoint mapping before parameters are assigned, so forward never
executes the absorbed ops:

- LayerScale ``scale1``/``scale2`` fold into the rows of ``attn.to_out`` /
  ``ff.w2``; the ``TransformerBlock`` residual adds are plain adds.
- ``post_quant_conv`` (1x1x1) folds into ``decoder.x_embedder`` and
  ``quant_conv`` (1x1x1) folds into ``encoder.conv_out``; ``encode`` /
  ``decode`` skip both convs.
- The pixel-domain Normalize folds into ``encoder.conv_in`` and the pixel
  denormalize plus the ``norm_out`` affine fold into ``decoder.proj_out``;
  the VAE consumes and produces raw [0, 1] pixels and the processor
  transforms are identities.

All folds compute in FP32 and cast back to the incoming tensor's dtype.
Each fold is a pure function of the incoming mapping and is idempotent:
consumed tensors are replaced by their neutral element (ones / identity
kernel) or their keys are removed, and the two pixel folds are guarded by
marker keys that ride the state dict. Loading an original checkpoint,
re-loading it, or loading an already-folded dump all converge to the same
parameters. Partial mappings (for example the fp16 decode-dtype store, which
carries neither the scale vectors nor any folded-source key) pass through
untouched.
"""

import torch

CONV_IN_PIXEL_NORM_FOLDED_KEY = "conv_in_pixel_norm_folded"
PROJ_OUT_PIXEL_DENORM_FOLDED_KEY = "proj_out_pixel_denorm_folded"

_LAYER_SCALE_TARGETS = {"scale1": "attn.to_out.", "scale2": "ff.w2."}


def _put(state_dict: dict, key: str, value_fp32: torch.Tensor, like: torch.Tensor):
    state_dict[key] = value_fp32.to(dtype=like.dtype)


def fold_layer_scales_(state_dict: dict, *, prefix: str) -> int:
    """Fold ``x + scale * f(x)`` into ``x + f'(x)`` by scaling f's output rows.

    Checkpoint LayerScale spans ~9e-9..0.10 per channel; rows whose folded
    fp16 weights land below the subnormal range flush toward zero, matching
    the old path where those contributions vanish below the fp32 residual ulp.
    """
    blocks_prefix = prefix + "decoder.transformer_blocks."
    folded = 0
    for key in sorted(state_dict):
        if not key.startswith(blocks_prefix):
            continue
        leaf = key.rsplit(".", 1)[-1]
        target = _LAYER_SCALE_TARGETS.get(leaf)
        if target is None:
            continue
        target_prefix = key[: -len(leaf)] + target
        weight_key = target_prefix + "weight"
        if weight_key not in state_dict:
            continue
        scale = state_dict[key].float()
        weight = state_dict[weight_key]
        _put(state_dict, weight_key, weight.float() * scale[:, None], weight)
        bias_key = target_prefix + "bias"
        if bias_key in state_dict:
            bias = state_dict[bias_key]
            _put(state_dict, bias_key, bias.float() * scale, bias)
        state_dict[key] = torch.ones_like(state_dict[key])
        folded += 1
    return folded


def _pointwise_conv_matrix(conv_weight: torch.Tensor, key: str) -> torch.Tensor:
    if conv_weight.ndim != 5 or any(k != 1 for k in conv_weight.shape[2:]):
        raise ValueError(
            f"{key} must be a 1x1x1 conv kernel, got shape {tuple(conv_weight.shape)}"
        )
    return conv_weight.float().reshape(conv_weight.shape[0], conv_weight.shape[1])


def _neutralize_pointwise_conv_(state_dict: dict, weight_key: str, bias_key: str):
    weight = state_dict[weight_key]
    channels = weight.shape[0]
    identity = torch.eye(channels, dtype=torch.float32).view(
        channels, channels, 1, 1, 1
    )
    _put(state_dict, weight_key, identity, weight)
    bias = state_dict[bias_key]
    state_dict[bias_key] = torch.zeros_like(bias)


def fold_post_quant_conv_into_x_embedder_(state_dict: dict, *, prefix: str) -> bool:
    """Compose ``x_embedder(pack(post_quant_conv(z)))`` into one Linear.

    ``pack(1, 1)`` is a pure permute, so the 1x1x1 conv is a per-voxel linear
    map on the Linear's input features: W' = W_emb @ W_pq, b' = W_emb @ b_pq
    + b_emb.
    """
    conv_weight_key = prefix + "post_quant_conv.weight"
    linear_weight_key = prefix + "decoder.x_embedder.weight"
    if conv_weight_key not in state_dict or linear_weight_key not in state_dict:
        return False
    conv_matrix = _pointwise_conv_matrix(state_dict[conv_weight_key], conv_weight_key)
    conv_bias = state_dict[prefix + "post_quant_conv.bias"].float()
    linear_weight = state_dict[linear_weight_key]
    linear_bias_key = prefix + "decoder.x_embedder.bias"
    linear_bias = state_dict[linear_bias_key]
    weight_fp32 = linear_weight.float()
    _put(state_dict, linear_weight_key, weight_fp32 @ conv_matrix, linear_weight)
    _put(
        state_dict,
        linear_bias_key,
        weight_fp32 @ conv_bias + linear_bias.float(),
        linear_bias,
    )
    _neutralize_pointwise_conv_(
        state_dict, conv_weight_key, prefix + "post_quant_conv.bias"
    )
    return True


def fold_quant_conv_into_conv_out_(state_dict: dict, *, prefix: str) -> bool:
    """Compose ``quant_conv(conv_out(h))`` into one k^3 conv.

    W'[o, i, k] = sum_m W_q[o, m] * W_c[m, i, k]; b' = W_q @ b_c + b_q.
    """
    quant_weight_key = prefix + "quant_conv.weight"
    conv_weight_key = prefix + "encoder.conv_out.weight"
    if quant_weight_key not in state_dict or conv_weight_key not in state_dict:
        return False
    quant_matrix = _pointwise_conv_matrix(
        state_dict[quant_weight_key], quant_weight_key
    )
    quant_bias = state_dict[prefix + "quant_conv.bias"].float()
    conv_weight = state_dict[conv_weight_key]
    if quant_matrix.shape[1] != conv_weight.shape[0]:
        raise ValueError(
            f"quant_conv input channels {quant_matrix.shape[1]} do not match "
            f"conv_out output channels {conv_weight.shape[0]}"
        )
    conv_bias_key = prefix + "encoder.conv_out.bias"
    conv_bias = state_dict[conv_bias_key]
    composed = quant_matrix @ conv_weight.float().reshape(conv_weight.shape[0], -1)
    _put(
        state_dict,
        conv_weight_key,
        composed.reshape(quant_matrix.shape[0], *conv_weight.shape[1:]),
        conv_weight,
    )
    _put(
        state_dict,
        conv_bias_key,
        quant_matrix @ conv_bias.float() + quant_bias,
        conv_bias,
    )
    _neutralize_pointwise_conv_(
        state_dict, quant_weight_key, prefix + "quant_conv.bias"
    )
    return True


def _marker_already_set(state_dict: dict, marker_key: str) -> bool:
    already = marker_key in state_dict and bool(state_dict[marker_key])
    state_dict[marker_key] = torch.tensor(True)
    return already


def fold_pixel_norm_into_conv_in_(
    state_dict: dict, *, prefix: str, mean: tuple, std: tuple
) -> bool:
    """Fold ``conv_in((x - mean) / std)`` into ``conv_in'(x)``.

    W' = W / std per input channel; b' = b - sum_{c,k} W[o,c,k] * mean_c /
    std_c. The conv's constant temporal padding must then inject ``mean``
    instead of zero (see ``BaseConv3d.temporal_pad_values``); zero-padding
    the normalized domain is mean-padding the raw domain.
    """
    weight_key = prefix + "encoder.conv_in.weight"
    if weight_key not in state_dict:
        return False
    if _marker_already_set(state_dict, prefix + CONV_IN_PIXEL_NORM_FOLDED_KEY):
        return False
    weight = state_dict[weight_key]
    bias_key = prefix + "encoder.conv_in.bias"
    bias = state_dict[bias_key]
    mean_t = torch.tensor(mean, dtype=torch.float32).view(1, -1, 1, 1, 1)
    std_t = torch.tensor(std, dtype=torch.float32).view(1, -1, 1, 1, 1)
    weight_fp32 = weight.float()
    _put(state_dict, weight_key, weight_fp32 / std_t, weight)
    shift = (weight_fp32 * (mean_t / std_t)).sum(dim=(1, 2, 3, 4))
    _put(state_dict, bias_key, bias.float() - shift, bias)
    return True


def fold_norm_out_and_pixel_denorm_into_proj_out_(
    state_dict: dict,
    *,
    prefix: str,
    mean: tuple,
    std: tuple,
    rows_per_channel: int,
) -> bool:
    """Fold the ``norm_out`` affine and the pixel denormalize into proj_out.

    proj_out(LN_affine(h)) = n(h) @ (W diag(gamma))^T + (W @ beta + b), and
    unpack maps proj_out row r to pixel channel r // rows_per_channel, so the
    denormalize ``y * std_c + mean_c`` scales rows and shifts the bias.
    """
    weight_key = prefix + "decoder.proj_out.weight"
    if weight_key not in state_dict:
        return False
    weight = state_dict[weight_key]
    bias_key = prefix + "decoder.proj_out.bias"
    bias = state_dict[bias_key]
    weight_fp32 = weight.float()
    bias_fp32 = bias.float()
    changed = False

    gamma_key = prefix + "decoder.norm_out.weight"
    if gamma_key in state_dict:
        gamma = state_dict.pop(gamma_key).float()
        beta_key = prefix + "decoder.norm_out.bias"
        if beta_key in state_dict:
            beta = state_dict.pop(beta_key).float()
            bias_fp32 = weight_fp32 @ beta + bias_fp32
        weight_fp32 = weight_fp32 * gamma
        changed = True

    if not _marker_already_set(state_dict, prefix + PROJ_OUT_PIXEL_DENORM_FOLDED_KEY):
        rows = weight_fp32.shape[0]
        if rows != len(mean) * rows_per_channel:
            raise ValueError(
                f"proj_out rows {rows} do not match {len(mean)} pixel channels "
                f"x {rows_per_channel} rows per channel"
            )
        row_channel = torch.arange(rows) // rows_per_channel
        row_scale = torch.tensor(std, dtype=torch.float32)[row_channel]
        row_shift = torch.tensor(mean, dtype=torch.float32)[row_channel]
        weight_fp32 = weight_fp32 * row_scale[:, None]
        bias_fp32 = bias_fp32 * row_scale + row_shift
        changed = True

    if changed:
        _put(state_dict, weight_key, weight_fp32, weight)
        _put(state_dict, bias_key, bias_fp32, bias)
    return changed


def apply_minimax_h3_vae_weight_folds_(
    state_dict: dict,
    *,
    prefix: str,
    pixel_mean: tuple,
    pixel_std: tuple,
    proj_rows_per_channel: int,
) -> None:
    fold_layer_scales_(state_dict, prefix=prefix)
    fold_post_quant_conv_into_x_embedder_(state_dict, prefix=prefix)
    fold_quant_conv_into_conv_out_(state_dict, prefix=prefix)
    fold_pixel_norm_into_conv_in_(
        state_dict, prefix=prefix, mean=pixel_mean, std=pixel_std
    )
    fold_norm_out_and_pixel_denorm_into_proj_out_(
        state_dict,
        prefix=prefix,
        mean=pixel_mean,
        std=pixel_std,
        rows_per_channel=proj_rows_per_channel,
    )
