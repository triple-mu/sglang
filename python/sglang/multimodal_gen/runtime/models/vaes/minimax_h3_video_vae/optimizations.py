# SPDX-License-Identifier: Apache-2.0
"""Model-owned configuration and request-scoped H3 VAE optimization policy."""

import torch

from sglang.multimodal_gen.runtime.models.vaes.fast_path_gate import (
    VaeFastPathGate,
    register_vae_fast_path_gate,
)


class MiniMaxH3VAEOptimizationState:
    def __init__(self, config):
        self.config = config
        self.gate = VaeFastPathGate()

    @staticmethod
    def supported(value: torch.Tensor) -> bool:
        return (
            value.is_cuda
            and torch.cuda.is_available()
            and torch.version.hip is None
            and not torch.is_grad_enabled()
            and not torch.compiler.is_compiling()
            and torch.cuda.get_device_capability(value.device) == (10, 0)
            and not torch.cuda.is_current_stream_capturing()
        )

    def enabled(self, value: torch.Tensor) -> bool:
        return (
            self.config.enable_optimizations
            and self.gate.enabled
            and self.supported(value)
        )

    def output_projection_enabled(self, value: torch.Tensor) -> bool:
        # This changes input precision, so it is an explicit deployment choice,
        # independent of request-scoped, mathematically equivalent fusions.
        return (
            self.config.decoder_output_projection_precision == "fp16"
            and self.supported(value)
        )


def install_optimization_state(vae, config):
    state = MiniMaxH3VAEOptimizationState(config)
    vae._sgl_h3_vae_optimization = state
    vae.processor._sgl_h3_vae_optimization = state
    for module in [*vae.encoder.modules(), *vae.decoder.modules()]:
        module._sgl_h3_vae_optimization = state
    register_vae_fast_path_gate(vae, state.gate)


def optimization_active(module, value) -> bool:
    state = getattr(module, "_sgl_h3_vae_optimization", None)
    return (
        state is not None
        and not getattr(module, "training", False)
        and state.enabled(value)
    )


def output_projection_active(module, value) -> bool:
    state = getattr(module, "_sgl_h3_vae_optimization", None)
    return (
        state is not None
        and not module.training
        and state.output_projection_enabled(value)
    )
