# SPDX-License-Identifier: Apache-2.0
"""Warmup/serving AdaLN plan-key alignment: a warmup request trimmed to
--warmup-steps must co-build every plan a real serving request of the same
partition can look up -- across all packed layout classes, not just the
warmup's own -- and must never overflow the rebuild slab.

Regression for the 2026-08-31 e2e gate miss: a t2va warmup covered 49 plans
but the 9-step real request still rebuilt 7 (steps 9, warmup 2, shifts 12/3).
"""

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

# e2e gate serving parameters: --num-inference-steps 9 --warmup-steps 2,
# model sigma_shift_scales video 12 / audio 3.
_VIDEO_SHIFT = 12.0
_AUDIO_SHIFT = 3.0
_SERVING_STEPS = 9
_WARMUP_STEPS = 2

_MODES = ("t2va", "fl2va", "ref2va")


def _branch(mode: str, *, geometry: str = "small") -> MiniMaxH3DenoiseBranch:
    if geometry == "small":
        common = dict(text_len=3, latent_t=2, latent_h=4, latent_w=4, audio_t=3)
    else:
        common = dict(text_len=7, latent_t=7, latent_h=8, latent_w=12, audio_t=9)
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
        text_embeddings=torch.zeros(common["text_len"], 5120),
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


def _step_timesteps(branch: MiniMaxH3DenoiseBranch, sigmas: dict[str, list[float]]):
    plan = branch.prepare_timestep_plan(
        video_timesteps=[1.0 - sigma for sigma in sigmas["video"][:-1]],
        audio_timesteps=[1.0 - sigma for sigma in sigmas["audio"][:-1]],
        imgvid_cond_noise_aug=MINIMAX_H3_IMGVID_COND_TIMESTEP,
        audio_ref_cond_noise_aug=MINIMAX_H3_AUDIO_REF_COND_TIMESTEP,
    )
    return [entry[0] for entry in plan]


def _plan_keys(branch: MiniMaxH3DenoiseBranch, sigmas: dict[str, list[float]]):
    return [adaln_plan_key(t) for t in _step_timesteps(branch, sigmas)]


def _online_model(*, max_plans: int = 64, max_plan_width: int = 4):
    return SimpleNamespace(
        adaln_cache=SimpleNamespace(
            weight_files=["model.safetensors"],
            max_plans=max_plans,
            max_plan_width=max_plan_width,
        )
    )


def _prewarm(warmup_mode: str, **model_kwargs) -> tuple[set, list]:
    """Run the prewarm as a trimmed warmup request of ``warmup_mode`` would."""
    branch = _branch(warmup_mode)
    base = _step_timesteps(branch, _sigmas(_WARMUP_STEPS))
    prewarm = _adaln_prewarm_step_timesteps(
        _online_model(**model_kwargs),
        branch,
        base_step_timesteps=base,
        adaln_prewarm_sigmas=_sigmas(_SERVING_STEPS),
        imgvid_cond_noise_aug=MINIMAX_H3_IMGVID_COND_TIMESTEP,
        audio_ref_cond_noise_aug=MINIMAX_H3_AUDIO_REF_COND_TIMESTEP,
    )
    covered = {adaln_plan_key(t) for t in base}
    covered |= {adaln_plan_key(t) for t in prewarm}
    return covered, prewarm


@pytest.mark.parametrize("warmup_mode", _MODES)
@pytest.mark.parametrize("request_mode", _MODES)
def test_prewarm_covers_every_task_shape(warmup_mode, request_mode):
    """One warmup request must prebuild the plans of every layout class.

    The partition serves multiple tasks (fl2va weights serve t2va and fl2va),
    so a t2va-shaped warmup must still cover fl2va's cond-row keys, and a
    ref2va warmup with only an image reference must cover audio-ref keys.
    """
    covered, _ = _prewarm(warmup_mode)
    serving_keys = _plan_keys(_branch(request_mode), _sigmas(_SERVING_STEPS))
    assert set(serving_keys) <= covered


def test_prewarm_returns_no_duplicate_keys():
    branch = _branch("t2va")
    base = _step_timesteps(branch, _sigmas(_WARMUP_STEPS))
    _, prewarm = _prewarm("t2va")
    keys = [adaln_plan_key(t) for t in base] + [adaln_plan_key(t) for t in prewarm]
    assert len(keys) == len(set(keys))


def test_plan_keys_are_geometry_independent():
    """Warmup resolution/frame count never affects which plans serving needs."""
    for mode in _MODES:
        assert _plan_keys(_branch(mode, geometry="small"), _sigmas(_SERVING_STEPS)) == (
            _plan_keys(_branch(mode, geometry="large"), _sigmas(_SERVING_STEPS))
        )


@pytest.mark.parametrize(
    ("mode", "width"),
    (("t2va", 2), ("fl2va", 3), ("ref2va", 4)),
)
def test_prewarm_plan_width_matches_task_shape(mode, width):
    """Widest per-step plan per task: t2va 2, fl2va 3, ref2va 4."""
    keys = _plan_keys(_branch(mode), _sigmas(_SERVING_STEPS))
    assert max(len(key) for key in keys) == width


def test_prewarm_skipped_when_no_class_fits_max_plans():
    _, prewarm = _prewarm("t2va", max_plans=2)
    assert prewarm == []


def test_prewarm_keeps_own_class_when_slab_fits_only_one():
    """Whole classes are dropped, never partial schedules; own class first."""
    # base (1 key, subset of own class) + own class (_SERVING_STEPS - 1 keys)
    covered, _ = _prewarm("t2va", max_plans=_SERVING_STEPS - 1)
    serving_keys = set(_plan_keys(_branch("t2va"), _sigmas(_SERVING_STEPS)))
    assert serving_keys <= covered
    assert len(covered) == _SERVING_STEPS - 1


def test_prewarm_drops_only_classes_wider_than_slab():
    """A 2-wide slab still prewarms t2va while fl2va/ref2va classes drop."""
    _, prewarm = _prewarm("ref2va", max_plan_width=2)
    assert all(t.numel() <= 2 for t in prewarm)
    t2va_keys = set(_plan_keys(_branch("t2va"), _sigmas(_SERVING_STEPS)))
    assert t2va_keys <= {adaln_plan_key(t) for t in prewarm}


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
    assert got[: len(warmup_keys)] == warmup_keys
    assert len(got) == len(set(got))
    for mode in _MODES:
        serving_keys = set(_plan_keys(_branch(mode), _sigmas(_SERVING_STEPS)))
        assert serving_keys <= set(got)


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


def test_stage_stages_sigmas_even_when_warmup_runs_untrimmed(monkeypatch):
    """An untrimmed warmup still needs the other layout classes prebuilt."""
    monkeypatch.setenv("MINIMAX_H3_ADALN_WARMUP_MATCH_STEPS", "1")
    stage = MiniMaxH3TimestepPreparationStage()
    batch = _warmup_batch(serving_steps=_WARMUP_STEPS)
    stage._maybe_generate_adaln_prewarm_sigmas(
        batch, _PLAN, SimpleNamespace(minimax_h3_adaln_online=True)
    )
    assert batch.extra[MINIMAX_H3_ADALN_PREWARM_SIGMAS_EXTRA_KEY] == _sigmas(
        _WARMUP_STEPS
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
