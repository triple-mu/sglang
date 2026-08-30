# SPDX-License-Identifier: Apache-2.0
# 3D convolution for the MiniMax H3 visual VAE.
import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseConv3d(nn.Conv3d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        bias=True,
        padding_mode="zeros",
        padding_mode_t=None,
        causal=True,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
            padding_mode=padding_mode,
        )
        padding_mode = "constant" if padding_mode == "zeros" else padding_mode
        padding_mode_t = "constant" if padding_mode_t == "zeros" else padding_mode_t
        self.pad_mode = padding_mode
        self.pad_mode_t = padding_mode_t or ("constant" if causal else "replicate")
        self.causal = causal
        # Per-channel value for constant temporal padding, instead of zero.
        # AutoencoderKLLegacy sets this on conv_in: with the pixel Normalize
        # folded into its weights (weight_folds), zero-padding the normalized
        # domain corresponds to mean-padding the raw pixel domain.
        self.register_buffer("temporal_pad_values", None, persistent=False)

    def _pad_temporal_constant(self, x, front, back):
        if self.temporal_pad_values is None:
            return F.pad(x, (0, 0, 0, 0, front, back), mode="constant")
        B, C, _, H, W = x.shape
        values = self.temporal_pad_values.to(dtype=x.dtype).view(1, C, 1, 1, 1)
        parts = []
        if front > 0:
            parts.append(values.expand(B, C, front, H, W))
        parts.append(x)
        if back > 0:
            parts.append(values.expand(B, C, back, H, W))
        return torch.cat(parts, dim=2) if len(parts) > 1 else x

    def _apply_temporal_padding(self, x):
        B, C, D, H, W = x.shape
        if D > 1:
            front = self.padding[0] * 2 if self.causal else self.padding[0]
            back = 0 if self.causal else self.padding[0]
            if self.pad_mode_t == "constant":
                return self._pad_temporal_constant(x, front, back)
            return F.pad(x, (0, 0, 0, 0, front, back), mode=self.pad_mode_t)
        else:
            if self.pad_mode_t == "constant":
                assert self.causal, "Zeros padding is only supported for causal mode"
                return self._pad_temporal_constant(x, self.kernel_size[0] - 1, 0)
            else:
                return x.expand(-1, -1, self.kernel_size[0], -1, -1)

    def _apply_padding(self, x):
        if sum(self.padding) == 0:
            return x

        x = F.pad(
            x,
            (self.padding[2], self.padding[2], self.padding[1], self.padding[1], 0, 0),
            mode=self.pad_mode,
        )

        x = self._apply_temporal_padding(x)
        return x

    def forward(self, x):
        if sum(self.padding) == 0:
            return super().forward(x)

        x = self._apply_padding(x)
        return F.conv3d(
            x,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=0,
            dilation=self.dilation,
        )
