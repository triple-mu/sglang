# SPDX-License-Identifier: Apache-2.0
"""Cross-step AdaLN hoisting must reproduce the per-step projection exactly."""

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
from sglang.multimodal_gen.runtime.layers.attention.selector import (
    global_force_attn_backend_context_manager,
)
from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import (
    MiniMaxH3AdalnProj,
    MiniMaxH3DiTModel,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.denoise_loop import (
    MiniMaxH3DenoiseBranch,
    minimax_h3_denoise_loop,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.packed_sequence import (
    minimax_h3_packed_sequence,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.test.single_test_file.component_accuracy.utils import (
    ensure_distributed_env_defaults,
)

# production adaln_proj shape: [M, 2688] x [2688, 18 * 5376]
_PROD_TIME_EMBED_DIM = 2688
_PROD_HIDDEN_SIZE = 5376


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


def _max_diff(pairs: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[float, float]:
    """(max abs, max rel) error over (actual, expected) tensor pairs."""
    max_abs = 0.0
    max_rel = 0.0
    for actual, expected in pairs:
        assert actual.shape == expected.shape
        diff = (actual.float() - expected.float()).abs()
        rel = diff / expected.float().abs().clamp(min=1e-12)
        max_abs = max(max_abs, float(diff.max()))
        max_rel = max(max_rel, float(rel.max()))
    return max_abs, max_rel


def test_build_step_adaln_params_matches_per_step_projection():
    """Batched projection must reproduce every step's per-step AdaLN tensors."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    model = _tiny_model(device)
    step_unique_timesteps = [
        torch.tensor(values, dtype=torch.float32, device=device)
        for values in ([1.0], [0.9, 0.5], [0.8, 0.4, 0.999], [0.1, 0.05])
    ]

    plan = model.build_step_adaln_params(step_unique_timesteps)
    assert len(plan) == len(step_unique_timesteps)

    pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
    with torch.inference_mode():
        for step, unique_timesteps in enumerate(step_unique_timesteps):
            adaln_input = torch.nn.functional.silu(
                model.time_embedder(unique_timesteps)
            ).to(torch.bfloat16)
            block_params, final_params = plan[step]
            for index, block in enumerate(model.blocks):
                expected = block.adaln_proj(adaln_input)
                assert len(block_params[index]) == len(expected) == 6
                pairs.extend(zip(block_params[index], expected))
            expected_final = model.final_layer.adaln_proj(adaln_input)
            assert len(final_params) == len(expected_final) == 2
            pairs.extend(zip(final_params, expected_final))

    max_abs, max_rel = _max_diff(pairs)
    print(f"tiny model: pairs={len(pairs)} max_abs={max_abs:.3e} max_rel={max_rel:.3e}")
    for actual, expected in pairs:
        assert torch.equal(actual, expected)


def test_production_shape_adaln_gemm_is_bit_exact_across_batched_steps():
    """The real [M, 2688] x [2688, 96768] projection at M=2 vs batched M=98."""
    if not torch.cuda.is_available():
        return
    device = torch.device("cuda")
    torch.manual_seed(0)
    _ensure_single_process_parallel_runtime()
    arch = MiniMaxH3DiTArchConfig()
    assert arch.time_embed_dim == _PROD_TIME_EMBED_DIM
    assert arch.hidden_size == _PROD_HIDDEN_SIZE
    with device:
        proj = MiniMaxH3AdalnProj(
            arch,
            arch.adaln_out_features,
            None,
            prefix="blocks.0.adaln_proj",
            expand_ratio=6,
            modality_num=3,
        )
    for param in proj.parameters():
        param.detach().normal_(std=0.02)

    # t2va runs M=2 every step; fl2va/ref2va mix in M=1 and M=3 steps
    step_rows = [1 + step % 3 for step in range(49)]
    with torch.inference_mode():
        per_step = [
            torch.randn(rows, arch.time_embed_dim, device=device, dtype=torch.bfloat16)
            for rows in step_rows
        ]
        expected = torch.cat([proj.project(rows) for rows in per_step])
        batched = proj.project(torch.cat(per_step))

    max_abs, max_rel = _max_diff([(batched, expected)])
    print(
        f"prod adaln_proj M={sorted(set(step_rows))} vs M={sum(step_rows)}: "
        f"max_abs={max_abs:.3e} max_rel={max_rel:.3e} "
        f"mismatched={int((batched != expected).sum())}/{batched.numel()}"
    )
    assert torch.equal(batched, expected)


def _run_tiny_denoise_loop(model: MiniMaxH3DiTModel, *, hoist: bool):
    device = torch.device("cuda")
    packed = minimax_h3_packed_sequence(
        text_len=3,
        latent_t=2,
        latent_h=4,
        latent_w=4,
        audio_t=3,
        include_keyframe_cond=False,
    )
    torch.manual_seed(1234)
    branch = MiniMaxH3DenoiseBranch(
        packed=packed,
        text_embeddings=torch.randn(3, model.arch.text_dim),
        token_tags=packed["token_tags"],
        device=device,
    )
    sigmas = [1.0, 0.7, 0.4, 0.2, 0.0]
    return minimax_h3_denoise_loop(
        model=model,
        positive=branch,
        initial_video_rows=torch.randn(int(branch.img_pos.shape[0]), 96),
        initial_audio_rows=torch.randn(int(branch.audio_pos.shape[0]), 32),
        keyframe_cond_rows=None,
        sigmas_video=sigmas,
        sigmas_audio=[1.0, 0.6, 0.35, 0.1, 0.0],
        device=device,
        hoist_step_adaln=hoist,
    )


def test_hoisted_denoise_loop_matches_per_step_loop_bitwise():
    """Whole-loop contract: hoisting must not move a single output bit."""
    if not torch.cuda.is_available():
        return
    torch.manual_seed(0)
    model = _tiny_model(torch.device("cuda"))
    num_projections = len(model.blocks) + 1  # blocks + final layer
    projected = []
    original_project = MiniMaxH3AdalnProj.project

    def counting_project(self, adaln_input):
        projected.append(int(adaln_input.shape[0]))
        return original_project(self, adaln_input)

    with (
        global_force_attn_backend_context_manager(AttentionBackendEnum.TORCH_SDPA),
        patch.object(MiniMaxH3AdalnProj, "project", counting_project),
    ):
        base_video, base_audio = _run_tiny_denoise_loop(model, hoist=False)
        base_projections = list(projected)
        projected.clear()
        hoisted_video, hoisted_audio = _run_tiny_denoise_loop(model, hoist=True)
        hoisted_projections = list(projected)

    num_steps = 4
    print(
        f"adaln_proj calls: per-step={len(base_projections)} "
        f"hoisted={len(hoisted_projections)} rows={set(hoisted_projections)}"
    )
    assert len(base_projections) == num_steps * num_projections
    assert len(hoisted_projections) == num_projections
    assert set(hoisted_projections) == {sum(base_projections) // num_projections}

    max_abs, max_rel = _max_diff(
        [(hoisted_video, base_video), (hoisted_audio, base_audio)]
    )
    print(f"tiny denoise loop: max_abs={max_abs:.3e} max_rel={max_rel:.3e}")
    assert torch.equal(hoisted_video, base_video)
    assert torch.equal(hoisted_audio, base_audio)


def test_adaln_hoist_is_off_unless_the_env_flag_is_set():
    """The GEMM shape change is opt-in; the default must stay on the old path."""
    model = _tiny_model(torch.device("cpu"))
    assert not model.can_hoist_step_adaln()
    with patch.dict(
        os.environ, {"SGLANG_DIFFUSION_MINIMAX_H3_ENABLE_ADALN_HOIST": "1"}
    ):
        assert model.can_hoist_step_adaln()


if __name__ == "__main__":
    test_build_step_adaln_params_matches_per_step_projection()
    print("PASS test_build_step_adaln_params_matches_per_step_projection")
    test_production_shape_adaln_gemm_is_bit_exact_across_batched_steps()
    print("PASS test_production_shape_adaln_gemm_is_bit_exact_across_batched_steps")
    test_hoisted_denoise_loop_matches_per_step_loop_bitwise()
    print("PASS test_hoisted_denoise_loop_matches_per_step_loop_bitwise")
    test_adaln_hoist_is_off_unless_the_env_flag_is_set()
    print("PASS test_adaln_hoist_is_off_unless_the_env_flag_is_set")
