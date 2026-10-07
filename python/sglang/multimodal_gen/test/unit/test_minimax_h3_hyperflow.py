"""HyperFlow sampling contract: sigma shift, (t, r) pair dedup, config loading."""

import json

import pytest
import torch

from sglang.multimodal_gen.runtime.models.dits.minimax_h3_hyperflow import (
    CONFIG_NAME,
    dedup_timestep_pairs,
    hyperflow_shift_sigmas,
    load_minimax_h3_hyperflow_config,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.time_request import (
    minimax_h3_time_shift_sigmas,
)

GRID = (1.0, 0.931506, 0.839236, 0.703462, 0.5, 0.296538, 0.160764, 0.068494, 0.0)


def test_shift_matches_the_scheduler_on_a_linspace_grid():
    linspace = tuple(torch.linspace(1.0, 0.0, 9).tolist())
    for shift in (12.0, 3.0):
        ours = hyperflow_shift_sigmas(linspace, shift)
        theirs = minimax_h3_time_shift_sigmas(num_steps=9, shift_scale=shift)
        assert ours == pytest.approx(theirs, abs=1e-6)


def test_shift_keeps_the_grid_valid_and_monotonic():
    shifted = hyperflow_shift_sigmas(GRID, 12.0)
    assert shifted[0] == 1.0 and shifted[-1] == 0.0
    assert all(b < a for a, b in zip(shifted, shifted[1:]))
    # The endpoint of step i is 1 - sigma[i + 1]; the last step lands on t = 1.
    assert 1.0 - shifted[-1] == 1.0


def test_pair_dedup_keeps_a_pinned_row_apart_from_generated_rows_at_the_same_t():
    # video rows at t=0.3 heading for r=0.5, a keyframe pinned at t=r=0.3, audio at (0.2, 0.4)
    t, r, slots = dedup_timestep_pairs([0.3, 0.3, 0.2], [0.5, 0.3, 0.4])
    assert t.tolist() == pytest.approx([0.2, 0.3, 0.3])
    assert r.tolist() == pytest.approx([0.4, 0.3, 0.5])
    assert slots.tolist() == [2, 1, 0]
    # identical pairs collapse like the single-timestep dedup does
    t2, r2, slots2 = dedup_timestep_pairs([0.3, 0.3], [0.5, 0.5])
    assert t2.tolist() == [pytest.approx(0.3)] and r2.tolist() == [pytest.approx(0.5)]
    assert slots2.tolist() == [0, 0]


def test_config_loads_from_a_directory_and_validates_the_grid(tmp_path):
    cfg = {"hyperflow_version": "1.0", "sigmas": list(GRID), "video_shift": 12.0, "audio_shift": 3.0, "gate": 0.25}
    (tmp_path / CONFIG_NAME).write_text(json.dumps(cfg))
    loaded = load_minimax_h3_hyperflow_config(tmp_path)
    assert loaded.num_steps == 8 and loaded.gate == 0.25 and loaded.sigmas == GRID
    assert load_minimax_h3_hyperflow_config(tmp_path / CONFIG_NAME) == loaded
    bad = dict(cfg, sigmas=[1.0, 0.5, 0.6, 0.0])
    (tmp_path / CONFIG_NAME).write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="strictly decreasing"):
        load_minimax_h3_hyperflow_config(tmp_path)


def test_bundle_keeps_the_block_lora_unmerged_and_merges_only_the_fp32_embedders(tmp_path):
    """The block deltas sit below bf16 resolution; a bundle that merged them (or left the
    embedder LoRA in the served file, double-applying it) would silently lose HyperFlow."""
    from safetensors.torch import load_file, save_file

    from sglang.multimodal_gen.runtime.models.dits.minimax_h3_hyperflow import (
        ENDPOINT_EMBEDDER_NAME,
        LORA_NAME,
    )
    from sglang.multimodal_gen.tools.build_minimax_h3_hyperflow_weights import main

    torch.manual_seed(0)
    base = tmp_path / "base"
    base.mkdir()
    embedder = {
        "time_embedder.proj_in.weight": torch.randn(8, 4),
        "time_embedder.proj_in.bias": torch.randn(8),
        "time_embedder.proj_out.weight": torch.randn(6, 8),
        "time_embedder.proj_out.bias": torch.randn(6),
    }
    block = {"blocks.0.mlp.fc1.weight": torch.randn(16, 4).to(torch.bfloat16)}
    save_file(embedder, str(base / "model-00001-of-00002.safetensors"))
    save_file(block, str(base / "model-00002-of-00002.safetensors"))
    (base / "config.json").write_text("{}")

    rank, alpha = 2, 4.0
    lora = {}
    for module in (
        "time_embedder.linear_1",
        "time_embedder.linear_2",
        "endpoint_time_embedder.linear_1",
        "endpoint_time_embedder.linear_2",
        "transformer_blocks.0.ff.net.0.proj",
    ):
        out_dim, in_dim = {
            "linear_1": (8, 4),
            "linear_2": (6, 8),
            "proj": (16, 4),
        }[module.rsplit(".", 1)[-1]]
        lora[f"transformer.{module}.lora_A.weight"] = torch.randn(rank, in_dim)
        lora[f"transformer.{module}.lora_B.weight"] = torch.randn(out_dim, rank)
    meta = {
        "hyperflow": "true",
        "hyperflow_version": "1.0",
        "hyperflow_sigmas": json.dumps(list(GRID)),
        "hyperflow_video_shift": "12.0",
        "hyperflow_audio_shift": "3.0",
        "hyperflow_gate": "0.25",
        "lora_rank": str(rank),
        "lora_alpha": str(alpha),
    }
    lora_file = tmp_path / "hf.safetensors"
    save_file(lora, str(lora_file), metadata=meta)
    out = tmp_path / "bundle"
    assert main([str(base), str(lora_file), str(out)]) == 0

    served = load_file(str(out / LORA_NAME))
    assert set(served) == {k for k in lora if ".transformer_blocks." in k}
    assert (out / "transformer" / "model-00002-of-00002.safetensors").is_symlink()
    merged = load_file(str(out / "transformer" / "model-00001-of-00002.safetensors"))
    scale = alpha / rank

    def delta(module):
        return (
            lora[f"transformer.{module}.lora_B.weight"]
            @ lora[f"transformer.{module}.lora_A.weight"]
        ) * scale

    torch.testing.assert_close(
        merged["time_embedder.proj_in.weight"],
        embedder["time_embedder.proj_in.weight"] + delta("time_embedder.linear_1"),
    )
    endpoint = load_file(str(out / ENDPOINT_EMBEDDER_NAME))
    torch.testing.assert_close(
        endpoint["proj_out.weight"],
        embedder["time_embedder.proj_out.weight"] + delta("endpoint_time_embedder.linear_2"),
    )
    torch.testing.assert_close(endpoint["proj_in.bias"], embedder["time_embedder.proj_in.bias"])
    config = load_minimax_h3_hyperflow_config(out)
    assert config.gate == 0.25 and config.num_steps == 8
