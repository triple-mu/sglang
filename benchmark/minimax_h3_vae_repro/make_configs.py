#!/usr/bin/env python3
"""Emit the three reproduction configs (container paths). Old U0 argv/request are the base."""
import json
import sys
from pathlib import Path

EXP = "/workspace/experiment/minimax-h3/2026-09-22-minimax-h3-vae-pr-repro"
SRC = "/workspace/source/worktrees/h3-vae-pr/python"
MODEL = "/workspace/models/pdd-8step"
SSA = {"sparsity": 0.75, "skip_first_steps": 1, "skip_first_layers": 0, "n_k": 4, "n_q": 4, "min_seq_len": 4096}


def argv(video_vae_attn, extra):
    return [
        "/opt/sglang/bin/sglang", "serve", "--model-type", "diffusion", "--model-path", MODEL,
        "--num-gpus", "4", "--tp-size", "1", "--ulysses-degree", "4", "--ring-degree", "1",
        "--performance-mode", "speed", "--quantization", "fp8", "--use-fsdp-inference", "false",
        "--component-attention-backends",
        f"transformer=subblock_sparse_attn,video_vae={video_vae_attn},audio_vae=torch_sdpa",
        "--attention-backend-config", json.dumps(SSA, separators=(",", ":")),
        "--enable-torch-compile", "false", "--enable-breakable-cuda-graph", "false",
        "--warmup-mode", "off", "--enable-layerwise-nvtx-marker", "false",
        "--host", "127.0.0.1", "--port", "50092", "--master-port", "50100",
        "--output-path", "{config_dir}/outputs", "--input-save-path", "{config_dir}/inputs",
        "--attention-backend", "subblock_sparse_attn", *extra,
    ]


BASE_ENV = {
    "PATH": "/opt/sglang/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin",
    "PYTHONPATH": SRC, "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
    "CUDA_VISIBLE_DEVICES": "0,1,2,3", "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
    "HF_HUB_OFFLINE": "1", "FLASHINFER_DISABLE_VERSION_CHECK": "1",
    "SGLANG_DIFFUSION_MINIMAX_H3_PDD_HEADS": f"{MODEL}/pdd_extra/pdd_fused_heads.safetensors",
    "SGLANG_DIFFUSION_IPC_A2A": "0",
    "SGLANG_DIFFUSION_SYNC_STAGE_PROFILING": "1",
    "SGLANG_DIFFUSION_STAGE_LOGGING": "1",
    "SGLANG_JIT_CACHE_DIR": f"{EXP}/cache/jit", "TORCH_EXTENSIONS_DIR": f"{EXP}/cache/torch",
    "TRITON_CACHE_DIR": f"{EXP}/cache/triton", "TORCHINDUCTOR_CACHE_DIR": f"{EXP}/cache/inductor",
    "CUDA_CACHE_PATH": f"{EXP}/cache/cuda", "XDG_CACHE_HOME": f"{EXP}/cache/xdg",
    "FLASHINFER_WORKSPACE_BASE": f"{EXP}/cache/flashinfer",
    "TMPDIR": "{config_dir}/tmp", "TMP": "{config_dir}/tmp", "TEMP": "{config_dir}/tmp",
}

REQUEST = {
    "model": MODEL,
    "prompt": "A young woman in a red wool coat walks briskly across a rain-slicked city crosswalk at dusk, neon signs reflecting in the puddles, her hair and coat hem moving with each step, pedestrians blurring past in the background.",
    "task": "fl2va", "seed": 4404, "num_outputs_per_prompt": 1,
    "target": {"short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 5.0},
    "num_inference_steps": 9, "flow_shift": 12.0, "audio_flow_shift": 3.0,
    "conditions": [
        {"type": "image", "uri": f"file://{EXP}/inputs/p1_portrait_first.png", "role": "keyframe", "frame_index": 0},
        {"type": "image", "uri": f"file://{EXP}/inputs/p1_portrait_last.png", "role": "keyframe", "frame_index": -1},
    ],
    "enable_cache_dit": True,
    "cache_dit_params": {"Fn_compute_blocks": 1, "Bn_compute_blocks": 0, "max_warmup_steps": 4,
                         "residual_diff_threshold": 0.24, "max_continuous_cached_steps": 3,
                         "enable_taylorseer": False, "scm_preset": "none", "scm_policy": "dynamic"},
}

FAST_ENV = {"SGLANG_DIFFUSION_MINIMAX_H3_VAE_OUTPUT_PROJECTION_FP16": "1",
            "SGLANG_DIFFUSION_MINIMAX_H3_VAE_WINDOW_BATCH": "0",
            "SGLANG_DIFFUSION_MINIMAX_H3_VAE_ENCODER_TILE_BATCH": "8",
            "SGLANG_DIFFUSION_MINIMAX_H3_VAE_DECODER_TILE_BATCH": "64"}
FAST_ARGS = ["--component-quantizations.video_vae", "fp8", "--component-residency", "video_vae=resident"]

def nvtx_on(a):
    a = list(a); i = a.index("--enable-layerwise-nvtx-marker"); a[i + 1] = "true"; return a


CONFIGS = {
    "baseline-lossless": dict(quality=None, env={}, argv=argv("torch_sdpa", [])),
    "fast-path-extra-high-fp8": dict(quality="extra-high", env=FAST_ENV, argv=argv("torch_cudnn_sdpa", FAST_ARGS)),
    "fast-path-high-fp8": dict(quality="high", env=FAST_ENV, argv=argv("torch_cudnn_sdpa", FAST_ARGS)),
    # nsys capture of the second request only; layerwise NVTX ranges needed for stage attribution
    "profile-fast-path-high-fp8": dict(quality="high", env={**FAST_ENV, "SGLANG_DIFFUSION_SYNC_STAGE_PROFILING": "0"},
                                      argv=nvtx_on(argv("torch_cudnn_sdpa", FAST_ARGS)), profile=True, timed=1),
}

out = Path(sys.argv[1] if len(sys.argv) > 1 else "configs")
out.mkdir(exist_ok=True)
for cid, c in CONFIGS.items():
    cfg = {"id": cid, "port": 50092, "quality": c["quality"], "warmup": 1, "timed": c.get("timed", 7),
           "profile": c.get("profile", False),
           "ready_timeout_s": 1800, "request_timeout_s": 900, "poll_interval_s": 0.05,
           "server_argv": c["argv"], "env": {**BASE_ENV, **c["env"]}, "request": REQUEST}
    (out / f"{cid}.json").write_text(json.dumps(cfg, indent=1) + "\n")
    print(out / f"{cid}.json")
