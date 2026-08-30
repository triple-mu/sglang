# SPDX-License-Identifier: Apache-2.0
"""Numerical contract for request-static H3 denoise metadata."""

from unittest.mock import patch

import torch

from sglang.multimodal_gen.configs.models.dits.minimax_h3 import (
    MINIMAX_H3_ADALN_MODALITY_NUM,
)
from sglang.multimodal_gen.runtime.models.schedulers.scheduling_minimax_h3_euler_ancestral import (
    _minimax_h3_euler_eta0_step,
    _minimax_h3_rf_v_to_x0,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.denoise_loop import (
    MiniMaxH3DenoiseBranch,
    _build_local_embedding_layout,
    _minimax_h3_update_target_rows_,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.packed_sequence import (
    minimax_h3_packed_sequence,
    minimax_h3_packed_sequence_ref2va_blocks,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.stages.denoising import (
    _precompute_refined_prompt_embeds,
)


def _branch(
    mode: str, token_tags: torch.Tensor | None = None
) -> MiniMaxH3DenoiseBranch:
    common = dict(text_len=3, latent_t=2, latent_h=4, latent_w=4, audio_t=3)
    if mode == "t2va":
        packed = minimax_h3_packed_sequence(
            **common,
            include_keyframe_cond=False,
        )
    elif mode == "fl2va":
        packed = minimax_h3_packed_sequence(
            **common,
            include_keyframe_cond=True,
            keyframe_frame_indices=[0, -1],
            frame_count=5,
        )
    else:
        packed = minimax_h3_packed_sequence_ref2va_blocks(
            **common,
            ref_blocks=[
                {"kind": "image", "latent_h": 4, "latent_w": 4},
                {"kind": "audio", "ref_audio_t": 2},
            ],
        )
    return MiniMaxH3DenoiseBranch(
        packed=packed,
        text_embeddings=torch.zeros(3, 5120),
        token_tags=packed["token_tags"] if token_tags is None else token_tags,
        device=torch.device("cpu"),
    )


def test_precomputed_timestep_plan_matches_full_unique_reference():
    """Preplanning must preserve fp32 collisions and every packed row class."""

    for mode in ("t2va", "fl2va", "ref2va"):
        branch = _branch(mode)
        assert branch.static_kwargs["skip_mask_out_condition"]
        assert "token_tags" not in branch.static_kwargs
        assert not bool((branch.static_kwargs["block_token_tags"] < 0).any())
        torch.testing.assert_close(
            branch.static_kwargs["img_pos_for_infer_output_info"]["position_ids"],
            branch.img_pos_dev[branch.update_mask_dev],
            rtol=0,
            atol=0,
        )
        video_steps = [0.75, 0.1]
        audio_steps = [0.625, 0.2]
        plan = branch.prepare_timestep_plan(
            video_timesteps=video_steps,
            audio_timesteps=audio_steps,
            imgvid_cond_noise_aug=0.6,
            audio_ref_cond_noise_aug=0.4,
        )

        assert branch.static_kwargs["packed_seq_params"]["cu_seqlens_q_host"] == tuple(
            int(value)
            for value in branch.static_kwargs["packed_seq_params"][
                "cu_seqlens_q"
            ].tolist()
        )
        assert branch.static_kwargs["refiner_packed_seq_params"][
            "cu_seqlens_q_host"
        ] == (0, 3, 3)

        for step, (video_t, audio_t) in enumerate(
            zip(video_steps, audio_steps, strict=True)
        ):
            reference = torch.full((branch.seq_len,), video_t, dtype=torch.float32)
            reference[branch.img_cond_seq_idx] = max(video_t, 0.6)
            reference[branch.audio_target_seq_idx] = audio_t
            reference[branch.audio_ref_seq_idx] = max(audio_t, 0.4)
            expected = torch.unique(reference, sorted=True, return_inverse=True)
            torch.testing.assert_close(plan[step][0], expected[0], rtol=0, atol=0)
            torch.testing.assert_close(plan[step][1], expected[1], rtol=0, atol=0)
            torch.testing.assert_close(
                plan[step][2],
                branch.static_kwargs["block_token_tags"]
                + expected[1] * MINIMAX_H3_ADALN_MODALITY_NUM,
                rtol=0,
                atol=0,
            )

        repeated_plan = branch.prepare_timestep_plan(
            video_timesteps=[0.0, 0.1, 0.2],
            audio_timesteps=[0.0, 0.2, 0.4],
            imgvid_cond_noise_aug=0.999,
            audio_ref_cond_noise_aug=1.0,
        )
        assert repeated_plan[1][1] is repeated_plan[2][1]
        assert repeated_plan[1][2] is repeated_plan[2][2]


def test_inplace_target_update_matches_scheduler_math():
    generator = torch.Generator().manual_seed(7)
    for sigma_curr, sigma_next in ((1.0, 0.7), (0.2, 0.0), (0.0, 0.0)):
        state = torch.randn(11, 32, generator=generator)
        velocity = torch.randn(11, 32, generator=generator)
        timestep = torch.tensor(1.0 - sigma_curr)
        ratio = torch.tensor(0.0 if sigma_curr == 0.0 else sigma_next / sigma_curr)
        denoised = _minimax_h3_rf_v_to_x0(state, velocity, timestep)
        expected = _minimax_h3_euler_eta0_step(
            state,
            denoised,
            sigma_curr=sigma_curr,
            sigma_next=sigma_next,
            sigma_ratio=ratio,
        )

        actual = state.clone()
        _minimax_h3_update_target_rows_(
            actual,
            velocity.clone(),
            sigma_curr=sigma_curr,
            sigma_next=sigma_next,
        )
        if sigma_curr == 0.0:
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        else:
            # The collapsed axpy and the scheduler's expanded r*x + (1-r)*x0
            # compute the same real-valued update with different fp32 rounding;
            # test_collapsed_update_matches_legacy_kernel_chain derives the
            # few-ulp bound this tolerance covers for unit-scale inputs.
            torch.testing.assert_close(actual, expected, rtol=0, atol=1e-5)


def _legacy_update_target_rows_(
    state: torch.Tensor,
    velocity: torch.Tensor,
    *,
    sigma_curr: float,
    sigma_next: float,
) -> None:
    """Pre-collapse five-kernel device chain, kept as the A2 exactness anchor."""
    device = state.device
    step_t = torch.tensor([1.0 - sigma_curr], dtype=torch.float32, device=device)
    sigma_t = 1.0 - step_t
    sigmas = torch.tensor([sigma_curr, sigma_next], dtype=torch.float32, device=device)
    sigma_ratio = sigmas[1:] / sigmas[:-1]
    one_minus_sigma_ratio = 1.0 - sigma_ratio
    denoised_scratch = torch.empty_like(state)
    torch.mul(sigma_t, velocity, out=denoised_scratch)
    torch.add(state, denoised_scratch, out=denoised_scratch)
    if sigma_curr == 0.0:
        return
    torch.mul(one_minus_sigma_ratio, denoised_scratch, out=velocity)
    torch.mul(sigma_ratio, state, out=state)
    torch.add(state, velocity, out=state)


def test_collapsed_update_matches_legacy_kernel_chain():
    """x + (sigma_c - sigma_n)*v must track the old r*x + (1-r)*(x + sigma_c*v)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = torch.Generator().manual_seed(11)
    pairs = (
        (1.0, 0.6604),
        (0.6604, 0.3396),
        (0.3396, 0.0),
        # near-equal sigmas maximize the fp32 cancellation in the legacy
        # (1 - sigma_n/sigma_c) coefficient the float64 collapse avoids
        (0.51, 0.5),
        (0.02, 0.0),
        (0.0, 0.0),
    )
    for sigma_curr, sigma_next in pairs:
        state = torch.randn(97, 96, generator=generator).to(device)
        velocity = torch.randn(97, 96, generator=generator).to(device)
        legacy_state = state.clone()
        _legacy_update_target_rows_(
            legacy_state,
            velocity.clone(),
            sigma_curr=sigma_curr,
            sigma_next=sigma_next,
        )
        actual = state.clone()
        _minimax_h3_update_target_rows_(
            actual,
            velocity.clone(),
            sigma_curr=sigma_curr,
            sigma_next=sigma_next,
        )
        if sigma_curr == 0.0:
            assert torch.equal(actual, state)
            assert torch.equal(legacy_state, state)
            continue
        # Not bit-exact by design: the legacy chain rounds sigma_t, the fp32
        # ratio, and every intermediate per element, while the axpy applies one
        # float64-computed coefficient with a single fused rounding. Each side
        # sits within a few fp32 ulps of the exact update, so bound their
        # difference by 8 eps of the operand magnitudes.
        eps = torch.finfo(torch.float32).eps
        magnitude = float(
            state.abs().max() + velocity.abs().max() + legacy_state.abs().max()
        )
        max_diff = float((actual - legacy_state).abs().max())
        assert max_diff <= 8 * eps * magnitude, (sigma_curr, sigma_next, max_diff)


def test_local_layout_target_rows_match_update_masks():
    """Target-row subsets drive the per-step embed rewrite; condition and
    reference rows must never be in them."""
    for mode in ("t2va", "fl2va", "ref2va"):
        branch = _branch(mode)
        text_len = int(branch.static_kwargs["prompt_embeds"].shape[0])
        branch_layout = branch.static_kwargs["local_embedding_layout"]
        assert branch_layout["embedding_cache"] == {}
        for world_size in (1, 2, 4):
            for rank in range(world_size):
                layout = _build_local_embedding_layout(
                    seq_len=branch.seq_len,
                    text_pos=torch.arange(text_len),
                    img_pos=branch.img_pos,
                    audio_pos=branch.audio_pos,
                    world_size=world_size,
                    rank=rank,
                    device=torch.device("cpu"),
                    img_update_mask=branch.update_mask,
                    audio_update_mask=branch.audio_update_mask,
                )
                row_start = rank * (branch.seq_len // world_size)
                row_stop = row_start + branch.seq_len // world_size
                img_in_shard = (branch.img_pos >= row_start) & (
                    branch.img_pos < row_stop
                )
                expected_img = branch.img_pos[img_in_shard & branch.update_mask]
                assert torch.equal(layout["img_target_global_ids"], expected_img)
                assert torch.equal(
                    layout["img_target_row_ids"], expected_img - row_start
                )
                audio_in_shard = (branch.audio_pos >= row_start) & (
                    branch.audio_pos < row_stop
                )
                expected_audio = branch.audio_pos[
                    audio_in_shard & branch.audio_update_mask
                ]
                assert torch.equal(layout["audio_target_global_ids"], expected_audio)
                assert torch.equal(
                    layout["audio_target_row_ids"], expected_audio - row_start
                )
                assert layout["embedding_cache"] == {}
                assert torch.equal(
                    layout["img_global_ids"], branch.img_pos[img_in_shard]
                )


def test_local_text_layout_is_a_contiguous_prefix_per_ulysses_rank():
    for mode in ("t2va", "fl2va", "ref2va"):
        branch = _branch(mode)
        text_len = int(branch.static_kwargs["prompt_embeds"].shape[0])
        for world_size in (1, 2, 4, 8):
            for rank in range(world_size):
                layout = _build_local_embedding_layout(
                    seq_len=branch.seq_len,
                    text_pos=torch.arange(text_len),
                    img_pos=branch.img_pos,
                    audio_pos=branch.audio_pos,
                    world_size=world_size,
                    rank=rank,
                    device=torch.device("cpu"),
                )
                start = int(layout["text_source_start"])
                stop = int(layout["text_source_stop"])
                row_start = rank * (branch.seq_len // world_size)
                expected = torch.nonzero(
                    (torch.arange(text_len) >= row_start)
                    & (
                        torch.arange(text_len)
                        < row_start + branch.seq_len // world_size
                    )
                ).view(-1)
                assert expected.tolist() == list(range(start, stop))


def test_rank_local_token_tags_match_reference_slice():
    for mode in ("t2va", "fl2va", "ref2va"):
        seq_len = _branch(mode).seq_len
        token_tags = torch.arange(seq_len, dtype=torch.long) - seq_len // 2
        for world_size in (1, 2, 4, 8):
            for rank in range(world_size):
                with patch(
                    "sglang.multimodal_gen.runtime.pipelines_core.stages."
                    "model_specific_stages.minimax_h3.denoise_loop.get_ulysses_ctx",
                    return_value=(world_size, rank),
                ):
                    branch = _branch(mode, token_tags=token_tags)
                local_rows = branch.seq_len // world_size
                expected = token_tags[
                    rank * local_rows : (rank + 1) * local_rows
                ].clamp(min=0)
                torch.testing.assert_close(
                    branch.static_kwargs["block_token_tags"], expected, rtol=0, atol=0
                )


def test_grouped_outputs_share_prompt_refinement():
    class Refiner:
        calls = 0

        def refine_prompt_embeds(self, prompt_embeds, refiner_cu, *, device):
            del refiner_cu
            self.calls += 1
            return torch.ones(
                prompt_embeds.shape[0], 5376, dtype=prompt_embeds.dtype, device=device
            )

    model = Refiner()
    conditioning = {}
    first, second = _branch("t2va"), _branch("t2va")

    for branch in (first, second):
        assert _precompute_refined_prompt_embeds(
            model,
            branch,
            device=torch.device("cpu"),
            shared_conditioning=conditioning,
        )

    assert model.calls == 1
    assert first.static_kwargs["prompt_embeds"] is second.static_kwargs["prompt_embeds"]
