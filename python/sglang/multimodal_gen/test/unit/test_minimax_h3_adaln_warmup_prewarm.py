# SPDX-License-Identifier: Apache-2.0
"""Warmup/serving AdaLN plan-key alignment: a warmup request trimmed to
--warmup-steps must co-build exactly the plans a real serving request looks
up, for every task shape, and must never overflow the rebuild slab."""

from types import SimpleNamespace

import pytest
import torch

from sglang.multimodal_gen.runtime.models.dits.minimax_h3_adaln_persist import (
    adaln_plan_key,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.constants import (
    MINIMAX_H3_ADALN_PREWARM_SIGMAS_EXTRA_KEY,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.denoise_loop import (
    MINIMAX_H3_AUDIO_REF_COND_TIMESTEP,
    MINIMAX_H3_IMGVID_COND_TIMESTEP,
    MiniMaxH3DenoiseBranch,
    _adaln_prewarm_step_timesteps,
    minimax_h3_denoise_loop,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.packed_sequence import (
    minimax_h3_packed_sequence,
    minimax_h3_packed_sequence_ref2va_blocks,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.stages.timestep_preparation import (
    MiniMaxH3TimestepPreparationStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.time_request import (
    minimax_h3_time_shift_sigmas,
)

_VIDEO_SHIFT = 12.0
_AUDIO_SHIFT = 3.0
_SERVING_STEPS = 10
_WARMUP_STEPS = 2


def _branch(mode: str) -> MiniMaxH3DenoiseBranch:
    common = dict(text_len=3, latent_t=2, latent_h=4, latent_w=4, audio_t=3)
    if mode == "t2va":
        packed = minimax_h3_packed_sequence(**common, include_keyframe_cond=False)
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
        token_tags=packed["token_tags"],
        device=torch.device("cpu"),
    )


def _sigmas(num_steps: int) -> dict[str, list[float]]:
    return {
        "video": minimax_h3_time_shift_sigmas(
            num_steps=num_steps, shift_scale=_VIDEO_SHIFT
        ),
        "audio": minimax_h3_time_shift_sigmas(
            num_steps=num_steps, shift_scale=_AUDIO_SHIFT
        ),
    }


def _plan_keys(branch: MiniMaxH3DenoiseBranch, sigmas: dict[str, list[float]]):
    plan = branch.prepare_timestep_plan(
        video_timesteps=[1.0 - sigma for sigma in sigmas["video"][:-1]],
        audio_timesteps=[1.0 - sigma for sigma in sigmas["audio"][:-1]],
        imgvid_cond_noise_aug=MINIMAX_H3_IMGVID_COND_TIMESTEP,
        audio_ref_cond_noise_aug=MINIMAX_H3_AUDIO_REF_COND_TIMESTEP,
    )
    return [adaln_plan_key(entry[0]) for entry in plan]


def _online_model(*, max_plans: int = 64, max_plan_width: int = 4):
    return SimpleNamespace(
        adaln_cache=SimpleNamespace(
            weight_files=["model.safetensors"],
            max_plans=max_plans,
            max_plan_width=max_plan_width,
        )
    )


@pytest.mark.parametrize("mode", ("t2va", "fl2va", "ref2va"))
def test_prewarm_step_timesteps_match_serving_plan_keys(mode):
    """The co-built plans must be key-identical to a real serving request's."""
    branch = _branch(mode)
    warmup_plan = branch.prepare_timestep_plan(
        video_timesteps=[1.0 - sigma for sigma in _sigmas(_WARMUP_STEPS)["video"][:-1]],
        audio_timesteps=[1.0 - sigma for sigma in _sigmas(_WARMUP_STEPS)["audio"][:-1]],
        imgvid_cond_noise_aug=MINIMAX_H3_IMGVID_COND_TIMESTEP,
        audio_ref_cond_noise_aug=MINIMAX_H3_AUDIO_REF_COND_TIMESTEP,
    )

    prewarm = _adaln_prewarm_step_timesteps(
        _online_model(),
        branch,
        base_step_timesteps=[entry[0] for entry in warmup_plan],
        adaln_prewarm_sigmas=_sigmas(_SERVING_STEPS),
        imgvid_cond_noise_aug=MINIMAX_H3_IMGVID_COND_TIMESTEP,
        audio_ref_cond_noise_aug=MINIMAX_H3_AUDIO_REF_COND_TIMESTEP,
    )

    serving_keys = _plan_keys(_branch(mode), _sigmas(_SERVING_STEPS))
    assert [adaln_plan_key(timesteps) for timesteps in prewarm] == serving_keys


@pytest.mark.parametrize(
    ("mode", "width"),
    (("t2va", 2), ("fl2va", 3), ("ref2va", 4)),
)
def test_prewarm_plan_width_matches_task_shape(mode, width):
    """Widest per-step plan per task: t2va 2, fl2va 3, ref2va 4."""
    keys = _plan_keys(_branch(mode), _sigmas(_SERVING_STEPS))
    assert max(len(key) for key in keys) == width


def test_prewarm_skipped_when_union_exceeds_max_plans():
    branch = _branch("t2va")
    warmup_plan = _plan_keys(branch, _sigmas(_WARMUP_STEPS))
    prewarm = _adaln_prewarm_step_timesteps(
        _online_model(max_plans=len(warmup_plan) + 1),
        branch,
        base_step_timesteps=[
            entry[0]
            for entry in branch.prepare_timestep_plan(
                video_timesteps=[
                    1.0 - sigma for sigma in _sigmas(_WARMUP_STEPS)["video"][:-1]
                ],
                audio_timesteps=[
                    1.0 - sigma for sigma in _sigmas(_WARMUP_STEPS)["audio"][:-1]
                ],
                imgvid_cond_noise_aug=MINIMAX_H3_IMGVID_COND_TIMESTEP,
                audio_ref_cond_noise_aug=MINIMAX_H3_AUDIO_REF_COND_TIMESTEP,
            )
        ],
        adaln_prewarm_sigmas=_sigmas(_SERVING_STEPS),
        imgvid_cond_noise_aug=MINIMAX_H3_IMGVID_COND_TIMESTEP,
        audio_ref_cond_noise_aug=MINIMAX_H3_AUDIO_REF_COND_TIMESTEP,
    )
    assert prewarm == []


def test_prewarm_skipped_when_plan_width_exceeds_slab():
    branch = _branch("ref2va")
    prewarm = _adaln_prewarm_step_timesteps(
        _online_model(max_plan_width=2),
        branch,
        base_step_timesteps=[],
        adaln_prewarm_sigmas=_sigmas(_SERVING_STEPS),
        imgvid_cond_noise_aug=MINIMAX_H3_IMGVID_COND_TIMESTEP,
        audio_ref_cond_noise_aug=MINIMAX_H3_AUDIO_REF_COND_TIMESTEP,
    )
    assert prewarm == []


def test_prewarm_noop_without_sigmas_or_online_cache():
    branch = _branch("t2va")
    kwargs = dict(
        base_step_timesteps=[],
        imgvid_cond_noise_aug=MINIMAX_H3_IMGVID_COND_TIMESTEP,
        audio_ref_cond_noise_aug=MINIMAX_H3_AUDIO_REF_COND_TIMESTEP,
    )
    assert (
        _adaln_prewarm_step_timesteps(
            _online_model(), branch, adaln_prewarm_sigmas=None, **kwargs
        )
        == []
    )
    for cache in (None, SimpleNamespace(weight_files=None)):
        assert (
            _adaln_prewarm_step_timesteps(
                SimpleNamespace(adaln_cache=cache),
                branch,
                adaln_prewarm_sigmas=_sigmas(_SERVING_STEPS),
                **kwargs,
            )
            == []
        )


def test_denoise_loop_prepares_warmup_and_serving_plan_union():
    """The single rebuild pass must cover both step schedules end to end."""
    branch = _branch("t2va")
    recorded: list[list[torch.Tensor]] = []

    class _Model:
        adaln_cache = SimpleNamespace(
            weight_files=["model.safetensors"], max_plans=64, max_plan_width=4
        )

        @staticmethod
        def prepare_adaln_plans(step_timesteps):
            recorded.append(step_timesteps)

    n_video_targets = int(branch.update_mask.sum())
    n_audio_rows = int(branch.audio_pos.shape[0])
    minimax_h3_denoise_loop(
        model=_Model(),
        model_forward=lambda _model, _kwargs, _step: (
            torch.zeros(n_video_targets, 96),
            torch.zeros(n_audio_rows, 32),
        ),
        positive=branch,
        initial_video_rows=torch.zeros(int(branch.img_pos.shape[0]), 96),
        initial_audio_rows=torch.zeros(n_audio_rows, 32),
        keyframe_cond_rows=None,
        sigmas_video=_sigmas(_WARMUP_STEPS)["video"],
        sigmas_audio=_sigmas(_WARMUP_STEPS)["audio"],
        adaln_prewarm_sigmas=_sigmas(_SERVING_STEPS),
        device=torch.device("cpu"),
    )

    assert len(recorded) == 1
    got = [adaln_plan_key(timesteps) for timesteps in recorded[0]]
    warmup_keys = _plan_keys(_branch("t2va"), _sigmas(_WARMUP_STEPS))
    serving_keys = _plan_keys(_branch("t2va"), _sigmas(_SERVING_STEPS))
    assert got == warmup_keys + serving_keys


# ---- timestep preparation stage: staging the serving schedules ----


def _warmup_batch(*, serving_steps=_SERVING_STEPS, warmup_steps=_WARMUP_STEPS):
    return SimpleNamespace(
        is_warmup=True,
        num_inference_steps=warmup_steps,
        extra={"cache_dit_num_inference_steps": serving_steps},
    )


_PLAN = SimpleNamespace(
    flow_shift=_VIDEO_SHIFT,
    audio_flow_shift=_AUDIO_SHIFT,
    default_flow_shift=_VIDEO_SHIFT,
    default_audio_flow_shift=_AUDIO_SHIFT,
)


def test_stage_stages_serving_sigmas_on_warmup_request(monkeypatch):
    monkeypatch.setenv("MINIMAX_H3_ADALN_WARMUP_MATCH_STEPS", "1")
    stage = MiniMaxH3TimestepPreparationStage()
    batch = _warmup_batch()
    stage._maybe_generate_adaln_prewarm_sigmas(
        batch, _PLAN, SimpleNamespace(minimax_h3_adaln_online=True)
    )
    assert batch.extra[MINIMAX_H3_ADALN_PREWARM_SIGMAS_EXTRA_KEY] == _sigmas(
        _SERVING_STEPS
    )


@pytest.mark.parametrize(
    ("batch", "server_args", "env_value"),
    (
        # env kill-switch off
        (_warmup_batch(), SimpleNamespace(minimax_h3_adaln_online=True), "0"),
        # adaln-online not active
        (_warmup_batch(), SimpleNamespace(minimax_h3_adaln_online=False), "1"),
        # not a warmup request
        (
            SimpleNamespace(is_warmup=False, num_inference_steps=2, extra={}),
            SimpleNamespace(minimax_h3_adaln_online=True),
            "1",
        ),
        # warmup already runs the serving step count
        (
            _warmup_batch(serving_steps=_WARMUP_STEPS),
            SimpleNamespace(minimax_h3_adaln_online=True),
            "1",
        ),
        # pre-trim step count not recorded
        (
            SimpleNamespace(is_warmup=True, num_inference_steps=2, extra={}),
            SimpleNamespace(minimax_h3_adaln_online=True),
            "1",
        ),
    ),
)
def test_stage_skips_prewarm_when_not_applicable(
    batch, server_args, env_value, monkeypatch
):
    monkeypatch.setenv("MINIMAX_H3_ADALN_WARMUP_MATCH_STEPS", env_value)
    stage = MiniMaxH3TimestepPreparationStage()
    stage._maybe_generate_adaln_prewarm_sigmas(batch, _PLAN, server_args)
    assert MINIMAX_H3_ADALN_PREWARM_SIGMAS_EXTRA_KEY not in batch.extra
