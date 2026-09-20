# SPDX-License-Identifier: Apache-2.0
"""Benchmark opt-in H3 VAE CUDA/C++ JIT operators against eager PyTorch.

Run from the repository root with PYTHONPATH=python. No model weights are
needed. CUDA events bracket repeated public API calls; compilation and warmup
are excluded, while output allocations and CPU launch gaps remain included.
This measures operators, not full-VAE or video-quality acceptance.
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from sglang.kernels.ops.diffusion import (
    minimax_h3_vae_assemble,
    minimax_h3_vae_denorm,
    minimax_h3_vae_group_norm_silu,
    minimax_h3_vae_norm_quant,
    minimax_h3_vae_qk_rope,
    minimax_h3_vae_silu_quant,
    minimax_h3_vae_temporal_write,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=60,
    stage="base-b-kernel-benchmark",
    runner_config="4-gpu-b200",
    disabled="standalone benchmark",
)


def group_norm_reference(x, groups, weight, bias, eps, isolated):
    if isolated:
        b, c, t, h, w = x.shape
        merged = x.permute(0, 2, 1, 3, 4).contiguous().reshape(b * t, c, h, w)
        y = F.group_norm(merged, groups, weight, bias, eps)
        y = y.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4).contiguous()
    else:
        y = F.group_norm(x, groups, weight, bias, eps)
    return F.silu(y, inplace=True)


def quantize(value):
    scale = value.abs().amax(dim=-1, keepdim=True) / 448.0
    quantized = (value / scale.clamp_min(torch.finfo(torch.float32).tiny)).clamp(
        -448, 448
    )
    return quantized.to(torch.float8_e4m3fn), scale


def qk_reference(qkv, cache):
    b, s, _, _ = qkv.shape
    cos, sin = cache.float().reshape(b, s, 1, 48).chunk(2, dim=-1)
    result = []
    for value in qkv[..., :128].chunk(2, dim=-1):
        value = F.rms_norm(value.float(), (64,), eps=1e-5)
        left, right = value[..., :48].chunk(2, dim=-1)
        result.append(
            torch.cat(
                (left * cos - right * sin, right * cos + left * sin, value[..., 48:]),
                dim=-1,
            ).to(qkv.dtype)
        )
    return tuple(result)


def make_blend(extents):
    # The production VAE caches blend weights. Do not charge their creation
    # to the eager reference on every iteration.
    weights = {n: torch.arange(n, device="cuda") / n for n in extents if n}

    def blend(previous, current, extent, dim):
        if extent == 0:
            return current
        shape = [1] * current.ndim
        shape[dim] = extent
        w = weights[extent].view(shape)
        prefix = previous.narrow(dim, previous.shape[dim] - extent, extent) * (1 - w)
        prefix.add_(current.narrow(dim, 0, extent) * w)
        return torch.cat(
            (prefix, current.narrow(dim, extent, current.shape[dim] - extent)), dim=dim
        )

    return blend


def cases(category):
    if category in ("all", "gn"):
        shapes = [
            (1, 128, 1, 256, 256),
            (7, 128, 1, 256, 256),
            (7, 256, 1, 128, 128),
            (7, 512, 1, 16, 16),
            (7, 1024, 1, 16, 16),
            (1, 256, 9, 64, 64),
        ]
        for shape in shapes:
            x = torch.randn(shape, device="cuda")
            weight = torch.randn(shape[1], device="cuda")
            bias = torch.randn_like(weight)
            yield {
                "name": "group_norm_silu_" + "x".join(map(str, shape)),
                "shape": shape,
                "groups": 32,
                "time_isolated": True,
                "jit": lambda: minimax_h3_vae_group_norm_silu(
                    x, 32, weight, bias, 1e-6
                ),
                "torch": lambda: group_norm_reference(x, 32, weight, bias, 1e-6, True),
                "kind": "fp32",
            }
    if category in ("all", "decoder"):
        qkv = torch.randn(1, 1792, 32, 192, dtype=torch.float16, device="cuda")
        angles = torch.randn(1792, 24, device="cuda")
        cache = torch.cat((angles.cos(), angles.sin()), dim=-1).half()
        yield {
            "name": "joint_qk_rmsnorm_rope",
            "shape": qkv.shape,
            "jit": lambda: minimax_h3_vae_qk_rope(qkv, cache, 1e-5, 1e-5),
            "torch": lambda: qk_reference(qkv, cache),
            "kind": "fp16",
        }
        x = torch.randn(1792, 2048, device="cuda")
        projected = torch.randn_like(x, dtype=torch.float16)
        scale = torch.randn(2048, device="cuda") * 0.3
        weight, bias = torch.randn(2, 2048, device="cuda")

        def norm_reference():
            residual = x + projected.float() * scale
            q, scales = quantize(F.rms_norm(residual, (2048,), weight, eps=1e-5))
            return residual, q, scales

        yield {
            "name": "residual_rmsnorm_fp8",
            "shape": x.shape,
            "jit": lambda: minimax_h3_vae_norm_quant(x, weight, 1e-5, projected, scale),
            "torch": norm_reference,
            "kind": "norm_fp8",
        }
        yield {
            "name": "residual_layernorm",
            "shape": x.shape,
            "jit": lambda: minimax_h3_vae_norm_quant(
                x, weight, 1e-5, projected, scale, bias, final=True
            ),
            "torch": lambda: F.layer_norm(
                x + projected.float() * scale, (2048,), weight, bias, 1e-5
            ),
            "kind": "fp32",
        }
        expanded = torch.randn(1792, 16384, device="cuda", dtype=torch.float16)

        def silu_reference():
            gate, up = expanded.float().chunk(2, dim=-1)
            return quantize(F.silu(gate) * up)

        yield {
            "name": "silu_mul_fp8",
            "shape": expanded.shape,
            "jit": lambda: minimax_h3_vae_silu_quant(expanded),
            "torch": silu_reference,
            "kind": "fp8",
        }
    if category in ("all", "output"):
        rows = [
            [torch.randn(1, 3, 17, 256, 256, device="cuda") for _ in range(2)]
            for _ in range(2)
        ]
        blend = make_blend([64, 4])

        def assemble_reference():
            output = torch.empty(1, 3, 17, 448, 448, device="cuda")
            for i, row in enumerate(rows):
                for j, tile in enumerate(row):
                    if i:
                        tile = blend(rows[i - 1][j], tile, 64, -2)
                    if j:
                        tile = blend(row[j - 1], tile, 64, -1)
                    h = 192 if i == 0 else 256
                    w = 192 if j == 0 else 256
                    output[..., i * 192 : i * 192 + h, j * 192 : j * 192 + w].copy_(
                        tile[..., :h, :w]
                    )
            return output

        yield {
            "name": "spatial_blend_crop_write",
            "shape": rows[0][0].shape,
            "grid": [2, 2],
            "overlap": [64, 64],
            "jit": lambda: minimax_h3_vae_assemble(rows, [64], [64]),
            "torch": assemble_reference,
            "kind": "fp32",
        }
        part = torch.randn(1, 3, 17, 448, 448, device="cuda")
        overlap = torch.randn(1, 3, 4, 448, 448, device="cuda")
        out_jit, out_torch = torch.empty_like(part), torch.empty_like(part)

        def temporal_jit():
            minimax_h3_vae_temporal_write(part, overlap, 4, out_jit, 0, 17)
            return out_jit

        def temporal_reference():
            out_torch.copy_(blend(overlap, part, 4, 2))
            return out_torch

        yield {
            "name": "temporal_blend_write",
            "shape": part.shape,
            "jit": temporal_jit,
            "torch": temporal_reference,
            "kind": "fp32",
        }
        mean = (-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225)
        std = (1 / 0.229, 1 / 0.224, 1 / 0.225)
        mean_tensor = part.new_tensor(mean)[None, :, None, None, None]
        std_tensor = part.new_tensor(std)[None, :, None, None, None]
        yield {
            "name": "rgb_denormalize_clamp",
            "shape": part.shape,
            "jit": lambda: minimax_h3_vae_denorm(part, mean, std),
            "torch": lambda: ((part - mean_tensor) / std_tensor).clamp(0, 1),
            "kind": "fp32",
        }


def comparable(value, kind):
    if kind == "fp8":
        return (value[0].float() * value[1],)
    if kind == "norm_fp8":
        return value[0], value[1].float() * value[2]
    return (value,) if isinstance(value, torch.Tensor) else value


def correctness(actual, expected, kind):
    metrics = []
    for a, b in zip(comparable(actual, kind), comparable(expected, kind)):
        difference = a.float() - b.float()
        error = difference.square().mean().sqrt()
        denominator = b.float().square().mean().sqrt().clamp_min(1e-12)
        metrics.append(
            {
                "max_abs": difference.abs().max().item(),
                "relative_rms": (error / denominator).item(),
            }
        )
        assert torch.isfinite(difference).all().item()
        if kind in ("fp8", "norm_fp8"):
            # Quantization midpoint differences are checked more precisely in
            # the companion pytest suite (bin agreement and half-ULP bounds).
            assert metrics[-1]["relative_rms"] < 0.005
        else:
            atol, rtol = (3e-3, 2e-3) if kind == "fp16" else (4e-5, 3e-5)
            torch.testing.assert_close(a, b, atol=atol, rtol=rtol)
    return metrics


def timed(fn, iterations):
    start, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / iterations


def write_json(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(path)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json",
        type=Path,
        required=True,
        help="JSON result file; updated after each case",
    )
    parser.add_argument(
        "--only", choices=("all", "gn", "decoder", "output"), default="all"
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260919)
    args = parser.parse_args()
    if min(args.warmup, args.iterations, args.samples) < 1:
        parser.error("warmup, iterations, and samples must all be positive")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0):
        parser.error("this benchmark requires an SM100 CUDA GPU")
    torch.manual_seed(args.seed)
    report = {
        "schema_version": 1,
        "complete": False,
        "gpu": torch.cuda.get_device_name(),
        "compute_capability": list(torch.cuda.get_device_capability()),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "seed": args.seed,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "samples": args.samples,
        "timing": "CUDA events around repeated eager public calls; compilation excluded; allocation and host launch gaps included; no CUDA graphs",
        "reference": "PyTorch FP32 expressions (GN matches per-frame reshape/contiguous/SiLU recipe); FP8 midpoint rounding may differ",
        "scope": "operator microbenchmarks; no end-to-end or video quality claim",
        "results": [],
    }
    write_json(args.json, report)
    for case in cases(args.only):
        torch.cuda.synchronize()
        started = time.perf_counter()
        actual, expected = case["jit"](), case["torch"]()
        errors = correctness(actual, expected, case["kind"])
        del actual, expected
        for _ in range(args.warmup):
            case["jit"]()
            case["torch"]()
        torch.cuda.synchronize()
        setup_seconds = time.perf_counter() - started
        samples = {"jit": [], "torch": []}
        for sample in range(args.samples):
            for name in ("jit", "torch") if sample % 2 == 0 else ("torch", "jit"):
                samples[name].append(timed(case[name], args.iterations))
        jit_us, torch_us = (
            statistics.median(samples["jit"]),
            statistics.median(samples["torch"]),
        )
        result = {
            key: value for key, value in case.items() if key not in ("jit", "torch")
        }
        result.update(
            jit_us=jit_us,
            torch_us=torch_us,
            speedup=torch_us / jit_us,
            reduction_pct=(1 - jit_us / torch_us) * 100,
            samples_us=samples,
            numerical_error=errors,
            excluded_setup_seconds=setup_seconds,
        )
        report["results"].append(result)
        write_json(args.json, report)
        print(
            f"{case['name']}: torch={torch_us:.2f} us, jit={jit_us:.2f} us, speedup={torch_us / jit_us:.3f}x",
            flush=True,
        )
    report["complete"] = True
    write_json(args.json, report)


if __name__ == "__main__":
    main()
