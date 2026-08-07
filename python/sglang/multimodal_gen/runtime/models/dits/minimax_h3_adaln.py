# SPDX-License-Identifier: Apache-2.0
"""Residency helpers for the hoisted MiniMax-H3 AdaLN projection.

Once the projection is hoisted out of the denoise loop it runs exactly once
per request, so its 26GB of weights spend the rest of the request idle. These
helpers let the model memoize the projected plan by schedule and drop the
weights to host memory in between.
"""

from __future__ import annotations

import hashlib

import torch


def adaln_plan_key(step_unique_timesteps: list[torch.Tensor]) -> bytes:
    """Identity of the AdaLN plan a schedule produces.

    The plan is a pure function of the timestep rows fed to the projection, so
    hashing their raw fp32 bytes covers everything that reaches it: the sigma
    schedule, the request's condition noise-aug anchors, and which modality
    rows the layout actually has.
    """
    values = torch.cat([t.reshape(-1) for t in step_unique_timesteps]).to(
        "cpu", torch.float32
    )
    digest = hashlib.sha256(values.numpy().tobytes())
    digest.update(repr([int(t.numel()) for t in step_unique_timesteps]).encode())
    return digest.digest()


class AdalnWeightStash:
    """Host-side custody of AdaLN weights while the plan stands in for them."""

    def __init__(self, linears: list[torch.nn.Module]) -> None:
        self._linears = linears
        # `.data` assignment is rejected on inference tensors, and this runs
        # under the inference_mode of build_step_adaln_params.
        with torch.inference_mode(False), torch.no_grad():
            self._host: list[list[tuple[torch.nn.Parameter, torch.Tensor]]] = []
            for linear in linears:
                saved = []
                for param in linear.parameters():
                    saved.append((param, param.data.to("cpu", copy=True)))
                    param.data = torch.empty(0, dtype=param.dtype, device=param.device)
                self._host.append(saved)

    def restore(self) -> None:
        """Move the weights back; the stash is spent afterwards."""
        with torch.inference_mode(False), torch.no_grad():
            for saved in self._host:
                for param, host_tensor in saved:
                    param.data = host_tensor.to(param.device, copy=True)
        self._host = []
