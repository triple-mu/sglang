"""
Benchmark and parity harness for the MiniMax-H3 video VAE encoder conv path.

Loads only the encoder + quant_conv weights from the FL2VA video VAE
checkpoint (the ViT decoder stays uninitialized on CPU) and drives the real
tiled encode entry points (`_adaptive_encode`, `encode_images`,
`encode_videos`) so eager vs cudnn_conv-fused measurements exercise the exact
production code path.

Workloads (all at the released 1344x768 canvas):
  image  - one keyframe canvas [1,3,1,768,1344] -> 28 tiles of [1,3,1,256,256]
  clip   - one temporal clip   [1,3,17,768,1344] -> 28 tiles of [1,3,17,256,256]
  video  - full 5s reference video (124 frames) through encode_videos

Modes:
  eager  - unmodified BaseConv3d path
  fused  - cudnn_conv fast path installed via minimax_h3_vae_cuda_opt
  both   - parity gate (fused vs eager vs TF32-off truth) then timings

The parity gate runs before any timing is printed; a gate failure aborts.

Usage:
    python bench_minimax_h3_vae_encoder.py --workload image --mode eager
    python bench_minimax_h3_vae_encoder.py --workload all --mode both
"""

import argparse
import json
import pathlib
import statistics
import time

import numpy as np
import torch

from sglang.multimodal_gen.configs.models.vaes.minimax_h3_video import (
    MiniMaxH3VideoVAEArchConfig,
    MiniMaxH3VideoVAEConfig,
)
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3 import MiniMaxH3VideoVAE
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.keyframe_encoding import (
    minimax_h3_scoped_encode_rng,
)

ENCODE_SEED = 42
CANVAS_H, CANVAS_W = 768, 1344
CLIP_FRAMES = 17
VIDEO_FRAMES = 124


def resolve_video_vae_dir(model_path: str, variant: str) -> pathlib.Path:
    path = pathlib.Path(model_path)
    if path.is_dir():
        root = path
    else:
        from huggingface_hub import snapshot_download

        root = pathlib.Path(
            snapshot_download(
                model_path,
                allow_patterns=[f"{variant.upper()}/video_vae/**"],
            )
        )
    vae_dir = root / variant.upper() / "video_vae"
    if not vae_dir.is_dir():
        raise FileNotFoundError(f"video_vae dir not found: expected {vae_dir}")
    return vae_dir


def load_encoder_only_vae(vae_dir: pathlib.Path, device: torch.device):
    with open(vae_dir / "config.json", encoding="utf-8") as f:
        hf_config = json.load(f)

    arch = MiniMaxH3VideoVAEArchConfig(
        latents_mean=hf_config["latents_mean"],
        latents_std=hf_config["latents_std"],
    )
    config = MiniMaxH3VideoVAEConfig(arch_config=arch)
    config.post_init()
    # The ViT decoder needs global server args (attention backend selection)
    # and is never exercised by this encoder-only bench; stub it out. The
    # DecoderTileCudaGraphRunner reads _graph_epoch off the decoder at init
    # and the load-time weight folds read decoder.config, so the stub carries
    # both (fold values are unused: the filtered state dict has no proj_out).
    from types import SimpleNamespace
    from unittest import mock

    from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae import klvae

    class _DecoderStub(torch.nn.Identity):
        _graph_epoch = 0
        config = SimpleNamespace(patch_size=16, patch_size_t=4)

    with mock.patch.object(klvae, "ViT3DDecoder", lambda **kwargs: _DecoderStub()):
        vae = MiniMaxH3VideoVAE(config)

    from safetensors.torch import load_file

    weights_path = (
        vae_dir / hf_config["source_path"] / hf_config["source_safetensors_path"]
    )
    state = load_file(str(weights_path))
    wanted = {
        k: v
        for k, v in state.items()
        if k.startswith("encoder.") or k.startswith("quant_conv.")
    }
    expected = {
        k
        for k in vae.state_dict()
        if k.startswith("encoder.") or k.startswith("quant_conv.")
    }
    missing = expected - wanted.keys()
    if missing:
        raise KeyError(
            f"checkpoint is missing {len(missing)} encoder keys, "
            f"e.g. {sorted(missing)[:3]}"
        )
    vae.load_state_dict(wanted, strict=False)
    # The filtered dict carries no decoder.proj_out, so its fold never runs
    # and _require_folded_weights would reject encode; the decoder is a stub
    # here, so mark its fold done by hand.
    vae.proj_out_pixel_denorm_folded.fill_(True)

    vae.eval()
    vae.encoder.to(device=device, dtype=torch.float32)
    vae.quant_conv.to(device=device, dtype=torch.float32)
    return vae


def make_image_input(device: torch.device) -> np.ndarray:
    rng = np.random.default_rng(ENCODE_SEED)
    return rng.integers(0, 256, size=(CANVAS_H, CANVAS_W, 3), dtype=np.uint8)


def make_video_input(num_frames: int) -> np.ndarray:
    rng = np.random.default_rng(ENCODE_SEED)
    return rng.integers(
        0, 256, size=(num_frames, CANVAS_H, CANVAS_W, 3), dtype=np.uint8
    )


def make_clip_tensor(vae, device: torch.device) -> torch.Tensor:
    video = make_video_input(CLIP_FRAMES)
    x = vae.processor.convert_numpy_to_tensor(video, device)
    x = vae.processor.transform_tensor(x, runtime_owned=True)
    return x.transpose(0, 1).unsqueeze(0).contiguous()


def cuda_time_ms(fn, warmup: int, iters: int):
    """Returns (first_call_ms, [per-iteration ms])."""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    first_ms = (time.perf_counter() - t0) * 1000.0
    for _ in range(max(warmup - 1, 0)):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return first_ms, times


def error_stats(test: torch.Tensor, ref: torch.Tensor):
    test = test.float()
    ref = ref.float()
    diff = (test - ref).abs()
    peak = ref.abs().max().clamp_min(1e-12)
    cos = torch.nn.functional.cosine_similarity(
        test.flatten(), ref.flatten(), dim=0
    ).item()
    return {
        "max_abs": diff.max().item(),
        "max_rel_vs_peak": (diff.max() / peak).item(),
        "cosine": cos,
    }


@torch.inference_mode()
def compute_moments(vae, x: torch.Tensor, allow_tf32: bool) -> torch.Tensor:
    prev_conv = torch.backends.cudnn.allow_tf32
    prev_mm = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    try:
        return vae._adaptive_encode(x).float().cpu()
    finally:
        torch.backends.cudnn.allow_tf32 = prev_conv
        torch.backends.cuda.matmul.allow_tf32 = prev_mm


@torch.inference_mode()
def encode_image_latent(
    vae, image: np.ndarray, device: torch.device, allow_tf32: bool = True
):
    prev_conv = torch.backends.cudnn.allow_tf32
    prev_mm = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    try:
        with minimax_h3_scoped_encode_rng(ENCODE_SEED, device):
            return vae.encode_images([image], use_fp16_latent=True)[0].cpu()
    finally:
        torch.backends.cudnn.allow_tf32 = prev_conv
        torch.backends.cuda.matmul.allow_tf32 = prev_mm


def parity_gate(vae, install_fn, device: torch.device) -> None:
    """Gate: fused must sit no farther from the TF32-off fp32 truth than 2x the
    eager-TF32 path does. Both paths run TF32 engines, so fused-vs-eager is
    engine-selection noise and is reported but not gated; a semantic bug (a
    misplaced pad, a wrong residual) shows up orders of magnitude above the
    eager-vs-truth floor.
    """
    inputs = {
        "image": make_clip_tensor(vae, device)[:, :, :1].contiguous(),
        "clip": make_clip_tensor(vae, device),
    }
    truth = {k: compute_moments(vae, x, allow_tf32=False) for k, x in inputs.items()}
    eager = {k: compute_moments(vae, x, allow_tf32=True) for k, x in inputs.items()}
    image = make_image_input(device)
    truth_latent = encode_image_latent(vae, image, device, allow_tf32=False).float()
    eager_latent = encode_image_latent(vae, image, device).float()

    install_fn(vae)

    print("== parity gate (moments, fp32) ==")
    failures = []
    for key, x in inputs.items():
        fused = compute_moments(vae, x, allow_tf32=True)
        e = error_stats(eager[key], truth[key])
        f = error_stats(fused, truth[key])
        d = error_stats(fused, eager[key])
        print(
            f"[{key}] eager-vs-truth max_rel={e['max_rel_vs_peak']:.3e} | "
            f"fused-vs-truth max_rel={f['max_rel_vs_peak']:.3e} | "
            f"fused-vs-eager max_rel={d['max_rel_vs_peak']:.3e} "
            f"cos={d['cosine']:.7f}"
        )
        if f["max_rel_vs_peak"] > max(2.0 * e["max_rel_vs_peak"], 1e-6):
            failures.append(
                f"{key}: fused-vs-truth {f['max_rel_vs_peak']:.3e} > "
                f"2x eager-vs-truth {e['max_rel_vs_peak']:.3e}"
            )
        if d["cosine"] < 0.99999:
            failures.append(f"{key}: fused-vs-eager cosine {d['cosine']:.7f}")

    fused_latent = encode_image_latent(vae, image, device).float()
    e_lat = error_stats(eager_latent, truth_latent)
    f_lat = error_stats(fused_latent, truth_latent)
    mismatch = (fused_latent != eager_latent).float().mean().item()
    print(
        f"[latent-fp16] eager-vs-truth max_rel={e_lat['max_rel_vs_peak']:.3e} | "
        f"fused-vs-truth max_rel={f_lat['max_rel_vs_peak']:.3e} | "
        f"fused-vs-eager mismatch fraction={mismatch:.4f} (1-ulp TF32 noise)"
    )
    if f_lat["max_rel_vs_peak"] > max(2.0 * e_lat["max_rel_vs_peak"], 1e-6):
        failures.append(
            f"latent: fused-vs-truth {f_lat['max_rel_vs_peak']:.3e} > "
            f"2x eager-vs-truth {e_lat['max_rel_vs_peak']:.3e}"
        )

    if failures:
        raise SystemExit("parity gate FAILED:\n  " + "\n  ".join(failures))
    print("parity gate PASSED")


def run_workload(vae, workload: str, device: torch.device, warmup: int, iters: int):
    if workload == "image":
        image = make_image_input(device)
        fn = lambda: encode_image_latent(vae, image, device)  # noqa: E731
    elif workload == "clip":
        x = make_clip_tensor(vae, device)
        fn = lambda: compute_moments(vae, x, allow_tf32=True)  # noqa: E731
    elif workload == "video":
        video = make_video_input(VIDEO_FRAMES)

        @torch.inference_mode()
        def fn():
            with minimax_h3_scoped_encode_rng(ENCODE_SEED, device):
                vae.encode_videos([video], use_fp16_latent=True)

    else:
        raise ValueError(f"unknown workload {workload!r}")

    first_ms, times = cuda_time_ms(fn, warmup=warmup, iters=iters)
    med = statistics.median(times)
    print(
        f"[{workload}] first={first_ms:.1f} ms  "
        f"median={med:.2f} ms  min={min(times):.2f}  max={max(times):.2f}  "
        f"(n={iters})"
    )
    return med


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="MiniMaxAI/MiniMax-H3")
    parser.add_argument("--variant", default="fl2va")
    parser.add_argument(
        "--workload", default="all", choices=["image", "clip", "video", "all"]
    )
    parser.add_argument("--mode", default="eager", choices=["eager", "fused", "both"])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--video-iters", type=int, default=3)
    parser.add_argument(
        "--contiguous",
        action="store_true",
        help="install with channels_last=False (fusion-only ablation)",
    )
    args = parser.parse_args()

    device = torch.device("cuda")
    print(
        f"torch={torch.__version__} cudnn={torch.backends.cudnn.version()} "
        f"gpu={torch.cuda.get_device_name(device)} "
        f"conv_tf32={torch.backends.cudnn.allow_tf32} "
        f"matmul_tf32={torch.backends.cuda.matmul.allow_tf32}"
    )

    vae_dir = resolve_video_vae_dir(args.model_path, args.variant)
    print(f"loading encoder weights from {vae_dir}")
    vae = load_encoder_only_vae(vae_dir, device)

    def install(v):
        from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_vae_cuda_opt import (
            install_minimax_h3_vae_encoder_cudnn_conv,
        )

        install_minimax_h3_vae_encoder_cudnn_conv(v, channels_last=not args.contiguous)
        import cudnn_conv

        print(
            f"cudnn_conv installed, autotune={cudnn_conv.get_autotune()}, "
            f"channels_last={not args.contiguous}"
        )

    if args.mode == "both":
        parity_gate(vae, install, device)
    elif args.mode == "fused":
        install(vae)

    workloads = (
        ["image", "clip", "video"] if args.workload == "all" else [args.workload]
    )
    for workload in workloads:
        iters = args.video_iters if workload == "video" else args.iters
        run_workload(vae, workload, device, warmup=args.warmup, iters=iters)

    if args.mode in ("fused", "both"):
        import cudnn_conv

        print(f"plan_cache_size={cudnn_conv.plan_cache_size()}")


if __name__ == "__main__":
    main()
