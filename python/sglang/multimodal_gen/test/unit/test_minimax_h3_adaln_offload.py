# SPDX-License-Identifier: Apache-2.0
"""Offloading the hoisted AdaLN weights must not change what the plan holds."""

import os
from unittest.mock import patch

import torch

from sglang.multimodal_gen.configs.models.dits.minimax_h3 import (
    MiniMaxH3DiTArchConfig,
    MiniMaxH3DiTConfig,
)
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    maybe_init_distributed_environment_and_model_parallel,
    model_parallel_is_initialized,
)
from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import (
    MiniMaxH3AdalnProj,
    MiniMaxH3DiTModel,
)
from sglang.multimodal_gen.runtime.models.dits.minimax_h3_adaln import (
    AdalnWeightStash,
    adaln_plan_key,
)
from sglang.multimodal_gen.test.single_test_file.component_accuracy.utils import (
    ensure_distributed_env_defaults,
)


def _ensure_single_process_parallel_runtime() -> None:
    if model_parallel_is_initialized():
        return
    ensure_distributed_env_defaults()
    maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=1)


def _tiny_model(device: torch.device) -> MiniMaxH3DiTModel:
    arch = MiniMaxH3DiTArchConfig(
        num_layers=3,
        token_refiner_num_layers=1,
        hidden_size=128,
        num_attention_heads=2,
        attention_head_dim=64,
        ffn_hidden_size=256,
        timestep_input_dim=16,
        time_embed_hidden_size=128,
        time_embed_dim=48,
        adaln_out_features=18 * 128,
        final_adaln_out_features=2 * 128,
    )
    _ensure_single_process_parallel_runtime()
    with device:
        model = MiniMaxH3DiTModel(
            config=MiniMaxH3DiTConfig(arch_config=arch),
            hf_config={},
            quant_config=None,
        )
    for param in model.parameters():
        param.detach().normal_(std=0.05)
    return model.eval()


def _steps(device: torch.device, values: list[list[float]]) -> list[torch.Tensor]:
    return [torch.tensor(v, dtype=torch.float32, device=device) for v in values]


def _hoist_and_offload_enabled(model: MiniMaxH3DiTModel):
    """Both env flags on; each fresh model carries its own unruled gate."""
    return patch.dict(
        os.environ,
        {
            "SGLANG_DIFFUSION_MINIMAX_H3_ENABLE_ADALN_HOIST": "1",
            "SGLANG_DIFFUSION_MINIMAX_H3_ENABLE_ADALN_OFFLOAD": "1",
        },
    )


def test_stash_round_trips_weights_bitwise():
    """Capture must free the device storage and restore must give it back."""
    model = _tiny_model(torch.device("cpu"))
    linears = model._adaln_linears()
    originals = [
        [param.detach().clone() for param in linear.parameters()] for linear in linears
    ]

    stash = AdalnWeightStash(linears)
    for linear in linears:
        for param in linear.parameters():
            assert param.numel() == 0

    stash.restore()
    for linear, saved in zip(linears, originals):
        for param, original in zip(linear.parameters(), saved):
            assert torch.equal(param, original)


def test_plan_memo_skips_the_second_projection():
    """A repeated schedule must reuse the plan instead of re-reading weights."""
    device = torch.device("cpu")
    model = _tiny_model(device)
    steps = _steps(device, [[0.9, 0.5], [0.7, 0.3]])

    calls = []
    original = MiniMaxH3AdalnProj.project

    def counting_project(self, adaln_input):
        calls.append(self)
        return original(self, adaln_input)

    env = _hoist_and_offload_enabled(model)
    with env, patch.object(MiniMaxH3AdalnProj, "project", counting_project):
        first = model.build_step_adaln_params(steps)
        projections = len(calls)
        second = model.build_step_adaln_params(steps)
        assert len(calls) == projections, "memo hit must not project again"

    for step_first, step_second in zip(first, second):
        for block_first, block_second in zip(step_first[0], step_second[0]):
            for a, b in zip(block_first, block_second):
                assert torch.equal(a, b)
        for a, b in zip(step_first[1], step_second[1]):
            assert torch.equal(a, b)


def test_new_schedule_restores_offloaded_weights():
    """A schedule change must bring the weights back and match the eager plan."""
    device = torch.device("cpu")
    model = _tiny_model(device)
    first_steps = _steps(device, [[0.9, 0.5]])
    second_steps = _steps(device, [[0.8, 0.4], [0.2]])

    # Baseline first: the weights are still resident, so this is the plan a
    # run that never offloaded would build.
    with patch.dict(
        os.environ, {"SGLANG_DIFFUSION_MINIMAX_H3_ENABLE_ADALN_HOIST": "1"}
    ):
        assert not model.can_offload_step_adaln()
        baseline = [
            [[t.clone() for t in block] for block in step[0]]
            for step in model.build_step_adaln_params(second_steps)
        ]

    model._adaln_plan_memo = None
    env = _hoist_and_offload_enabled(model)
    with env:
        model.build_step_adaln_params(first_steps)
        assert model._adaln_stash is not None
        offloaded = model.build_step_adaln_params(second_steps)
        assert model._adaln_stash is not None

    for step_offloaded, step_baseline in zip(offloaded, baseline):
        for block_a, block_b in zip(step_offloaded[0], step_baseline):
            for a, b in zip(block_a, block_b):
                assert torch.equal(a, b)


def test_offload_requires_hoist():
    """The offload flag alone must not drop weights the loop still needs."""
    model = _tiny_model(torch.device("cpu"))
    with patch.dict(
        os.environ, {"SGLANG_DIFFUSION_MINIMAX_H3_ENABLE_ADALN_OFFLOAD": "1"}
    ):
        assert not model.can_offload_step_adaln()
    env = _hoist_and_offload_enabled(model)
    with env:
        assert model.can_offload_step_adaln()


def test_plan_key_tracks_the_timestep_bits():
    """Anything that changes the projected rows must change the key."""
    device = torch.device("cpu")
    base = _steps(device, [[0.9, 0.5], [0.7]])
    assert adaln_plan_key(base) == adaln_plan_key(_steps(device, [[0.9, 0.5], [0.7]]))
    # a different noise-aug anchor
    assert adaln_plan_key(base) != adaln_plan_key(
        _steps(device, [[0.9, 0.5], [0.70001]])
    )
    # the same values split across a different number of steps
    assert adaln_plan_key(base) != adaln_plan_key(_steps(device, [[0.9], [0.5, 0.7]]))


if __name__ == "__main__":
    test_stash_round_trips_weights_bitwise()
    print("PASS test_stash_round_trips_weights_bitwise")
    test_plan_memo_skips_the_second_projection()
    print("PASS test_plan_memo_skips_the_second_projection")
    test_new_schedule_restores_offloaded_weights()
    print("PASS test_new_schedule_restores_offloaded_weights")
    test_offload_requires_hoist()
    print("PASS test_offload_requires_hoist")
    test_plan_key_tracks_the_timestep_bits()
    print("PASS test_plan_key_tracks_the_timestep_bits")
