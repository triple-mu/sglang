#!/usr/bin/env python3
"""VAE-only round trip on one GPU.

Encode an MP4 once with the fp32 encoder, then decode the same latent with
several video-VAE decoder variants (lossless fp16, extra-high fast path, online
FP8, online NVFP4 FFN). Report per-variant decode time (CUDA events, median of
timed repeats) and per-frame PSNR/SSIM against the lossless decode and against
the input frames. Runs inside the container with the worktree on PYTHONPATH.
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

VARIANTS = {
    # name: (quantization override, request quality)
    "lossless-fp16": (None, "exact"),
    "extra-high-fp16": (None, "lossless"),
    "extra-high-fp8": ("fp8", "lossless"),
    "extra-high-nvfp4": ("nvfp4", "lossless"),
}


def read_frames(path, limit):
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height", "-of", "json", path],
        capture_output=True, text=True, check=True)
    stream = json.loads(probe.stdout)["streams"][0]
    width, height = int(stream["width"]), int(stream["height"])
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
        capture_output=True, check=True).stdout
    frames = np.frombuffer(raw, np.uint8).reshape(-1, height, width, 3)
    return frames[:limit] if limit else frames


class FakeServerArgs:
    """The fields VAELoader.load_customized reads (mirrors the loader unit tests)."""

    def __init__(self, pipeline_config, quantization, model_root):
        self.pipeline_config = pipeline_config
        self.model_path = model_root
        self.num_gpus = 1
        self.disable_autocast = False
        self.pin_cpu_memory = False
        self.hsdp_replicate_dim = 1
        self.hsdp_shard_dim = 1
        self.model_paths = {}
        self.revision = None
        self.trust_remote_code = False
        self.layerwise_components = set()
        self.component_weights_paths = {}
        self.component_quantizations = {} if quantization is None else {"video_vae": quantization}
        self.component_precisions = {}
        self.component_direct_gpu_weight_loading = {}

    def resolve_component_attention_backend(self, _name):
        return None, None

    def requested_component_attention_backend(self, _name):
        return None

    def should_start_component_on_cpu(self, _name):
        return False

    def should_configure_layerwise_offload_for_lazy_component(self, _name):
        return False

    def should_direct_gpu_weight_load_component(self, _name):
        return False

    def should_use_fsdp_for_component(self, _name):
        return False

    def disable_fsdp_for_component(self, _name):
        return None

    def explicit_residency_mode(self, _name):
        return None

    def require_component_resident(self, *_args, **_kwargs):
        return None


def stub_global_server_args():
    from sglang.multimodal_gen.runtime.server_args import server_args as module

    module._global_server_args = SimpleNamespace(
        attention_backend=None, attention_backend_config=None,
        enable_attention_backend_autotune=False, enable_breakable_cuda_graph=False,
        enable_layerwise_nvtx_marker=False, kv_gather_degree=1, sp_split_auto=False)


def load_vae(model_dir, quantization, model_root):
    from sglang.multimodal_gen.configs.pipeline_configs.minimax_h3 import MiniMaxH3PipelineConfig
    from sglang.multimodal_gen.runtime.layers.attention.selector import (
        global_force_attn_backend_context_manager,
    )
    from sglang.multimodal_gen.runtime.loader.component_loaders import vae_loader
    from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum

    args = FakeServerArgs(MiniMaxH3PipelineConfig(), quantization, model_root)
    t0 = time.perf_counter()
    with global_force_attn_backend_context_manager(AttentionBackendEnum.TORCH_CUDNN_SDPA):
        vae = vae_loader.VAELoader().load_customized(model_dir, args, "video_vae")
    vae.eval()
    # Single process: no tile-parallel group to consult.
    vae.parallel_tiling = False
    print(f"loaded video_vae quantization={quantization} in {time.perf_counter() - t0:.1f}s "
          f"fast_path={'yes' if vae.fast_path is not None else 'no'} fp8={vae.decoder.fp8_installed} "
          f"nvfp4={vae.decoder.nvfp4_installed}", flush=True)
    return vae


def encode(vae, frames):
    torch.manual_seed(4404)
    with torch.inference_mode():
        latents = vae.encode_videos([frames])
    return latents[0][None].float().contiguous()


def decode(vae, z, frame_num, quality, nvtx_tag):
    from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
    from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.fast_path import (
        minimax_h3_vae_fast_path_scope,
    )

    vae.prepare_decoder_autocast_weights(torch.float16)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    with (torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16),
          minimax_h3_vae_fast_path_scope(vae, quality=quality, stage="decode"),
          set_forward_context(current_timestep=0, attn_metadata=None)):
        torch.cuda.nvtx.range_push(f"decode:{nvtx_tag}")
        start.record()
        recon = vae.decode_base(z, frame_num=frame_num)
        out = vae.processor.revert_tensor(recon)
        end.record()
        torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    frames = (out[0].permute(1, 2, 3, 0).float() * 255).round().clamp(0, 255).to(torch.uint8)
    return frames.cpu().numpy(), start.elapsed_time(end)


def psnr_per_frame(a, b):
    mse = ((a.astype(np.float32) - b.astype(np.float32)) ** 2).mean(axis=(1, 2, 3))
    return 10 * np.log10(255.0 ** 2 / np.maximum(mse, 1e-10))


def ssim_mean(a, b, step):
    from skimage.metrics import structural_similarity

    values = [structural_similarity(a[i], b[i], channel_axis=-1, data_range=255)
              for i in range(0, a.shape[0], step)]
    return float(np.mean(values)), float(np.min(values))


def compare(actual, reference, step):
    psnr = psnr_per_frame(actual, reference)
    ssim, ssim_min = ssim_mean(actual, reference, step)
    return {"psnr_mean_db": float(psnr.mean()), "psnr_min_db": float(psnr.min()),
            "ssim_mean": ssim, "ssim_min": ssim_min}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="video_vae component directory")
    ap.add_argument("--model-root", required=True, help="model root the deployment was loaded from")
    ap.add_argument("--mp4", required=True)
    ap.add_argument("--frames", type=int, default=0, help="0 = all frames")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--timed", type=int, default=3)
    ap.add_argument("--ssim-step", type=int, default=4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    stub_global_server_args()

    frames = read_frames(a.mp4, a.frames)
    print(f"input {frames.shape} from {a.mp4}", flush=True)
    report = {"mp4": a.mp4, "input_shape": list(frames.shape), "variants": {},
              "nvfp4_backend": os.environ.get("SGLANG_DIFFUSION_MINIMAX_H3_VAE_NVFP4_BACKEND", "cudnn")}
    reference = None
    z = None
    frame_num = None
    for name in a.variants.split(","):
        quantization, quality = VARIANTS[name]
        vae = load_vae(a.model, quantization, a.model_root)
        if z is None:
            z = encode(vae, frames)
            frame_num = 1 + 4 * (z.shape[2] - 1)
            print(f"latent {tuple(z.shape)} -> {frame_num} frames", flush=True)
        # First decode: warmup, and for NVFP4 the calibration pass on the eager blocks.
        first, first_ms = decode(vae, z, frame_num, quality, f"{name}:first")
        times = []
        for _ in range(a.timed):
            out, ms = decode(vae, z, frame_num, quality, f"{name}:timed")
            times.append(ms)
        entry = {"decode_ms_median": statistics.median(times), "decode_ms_min": min(times),
                 "decode_ms_max": max(times), "first_decode_ms": first_ms,
                 "output_shape": list(out.shape)}
        if reference is None:
            reference = out
            entry["role"] = "reference"
        else:
            entry["vs_lossless"] = compare(out, reference, a.ssim_step)
            if quantization == "nvfp4":
                entry["calibration_pass_vs_lossless"] = compare(first, reference, a.ssim_step)
        trimmed = frames[: out.shape[0]]
        if trimmed.shape == out.shape:
            entry["vs_input"] = compare(out, trimmed, a.ssim_step)
        report["variants"][name] = entry
        print(name, json.dumps(entry), flush=True)
        del vae
        torch.cuda.empty_cache()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(report, f, indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    sys.exit(main())
