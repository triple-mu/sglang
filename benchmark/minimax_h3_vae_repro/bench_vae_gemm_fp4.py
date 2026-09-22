#!/usr/bin/env python3
"""GEMM-level prescreen for the 144 MiniMax-H3 VAE decoder linears on one B200.

Shapes are the real ones (M = 49 tiles x 1797 tokens). Candidates:
  fp8_sgl      sgl_kernel.fp8_scaled_mm + sglang_per_token_quant_fp8 (current path)
  fp8_torch    torch._scaled_mm row-wise FP8 (cuBLASLt)
  nvfp4_<be>   flashinfer.mm_fp4 backend=<be> + flashinfer.fp4_quantize for the activation
Timing: CUDA events, eager launches, no CUDA graph; 10 warmup, 5 samples x 30 iterations,
median of per-sample medians. Correctness: rel_l1 against an FP32 matmul of the same
BF16-rounded operands. Operator microbenchmark only; no end-to-end claim.
"""
import argparse
import json
import statistics
import time
import traceback

import torch

SHAPES = {  # name: (K, N)
    "attn.to_qkv": (2048, 6144),
    "attn.to_out": (2048, 2048),
    "ff.w1": (2048, 16384),
    "ff.w2": (8192, 2048),
}
FP8_MAX = torch.finfo(torch.float8_e4m3fn).max  # 448
FP4_MAX = 6.0


def timeit(fn, warmup=10, iters=30, samples=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    meds = []
    for _ in range(samples):
        times = []
        for _ in range(iters):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            fn()
            e.record()
            e.synchronize()
            times.append(s.elapsed_time(e) * 1000.0)
        meds.append(statistics.median(times))
    return {"us": statistics.median(meds), "us_min": min(meds), "us_max": max(meds)}


def rel_l1(y, ref):
    return ((y.float() - ref).abs().sum() / ref.abs().sum()).item()


def fp8_weight(w16):
    w = w16.float()
    amax = w.abs().amax(dim=1, keepdim=True)
    scale = torch.where(amax == 0, torch.ones_like(amax), amax / FP8_MAX)
    return (w / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn), scale.contiguous()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=49 * 1797)
    ap.add_argument("--json", required=True)
    ap.add_argument("--backends", default="cutlass,cudnn,trtllm,cute-dsl")
    ap.add_argument("--skip-correctness", action="store_true")
    a = ap.parse_args()
    torch.manual_seed(20260922)
    dev = torch.device("cuda")
    import flashinfer
    from flashinfer import fp4_quantize, mm_fp4
    from flashinfer.autotuner import autotune
    from sgl_kernel import fp8_scaled_mm
    from sglang.kernels.ops.quantization.fp8_kernel import sglang_per_token_quant_fp8

    report = {
        "gpu": torch.cuda.get_device_name(dev), "capability": list(torch.cuda.get_device_capability(dev)),
        "torch": torch.__version__, "flashinfer": getattr(flashinfer, "__version__", "?"),
        "m": a.m, "timing": "CUDA events, eager, 10 warmup, 5x30 iters, median of medians",
        "peak_dense_pflops_assumed": {"fp16": 2.25, "fp8": 4.5, "fp4": 9.0}, "shapes": {},
    }
    backends = [b for b in a.backends.split(",") if b]
    for name, (K, N) in SHAPES.items():
        M = a.m
        flops = 2.0 * M * N * K
        x16 = (torch.randn(M, K, device=dev) * 0.5).to(torch.float16)
        w16 = (torch.randn(N, K, device=dev) * 0.02).to(torch.float16)
        ref = None if a.skip_correctness else (x16.float() @ w16.float().t())
        row = {"K": K, "N": N, "gflop": flops / 1e9, "candidates": {}}

        def add(cand, fn_gemm, fn_quant=None, y_fn=None):
            try:
                g = timeit(fn_gemm)
                q = timeit(fn_quant) if fn_quant else {"us": 0.0, "us_min": 0.0, "us_max": 0.0}
                entry = {"gemm_us": g["us"], "gemm_us_min": g["us_min"], "gemm_us_max": g["us_max"],
                         "quant_us": q["us"], "total_us": g["us"] + q["us"],
                         "tflops": flops / (g["us"] * 1e-6) / 1e12}
                if ref is not None and y_fn is not None:
                    entry["rel_l1"] = rel_l1(y_fn(), ref)
                row["candidates"][cand] = entry
                print(f"{name:12s} {cand:14s} gemm {g['us']:9.1f} us  quant {q['us']:8.1f} us  "
                      f"{entry['tflops']:7.1f} TFLOPS  rel_l1 {entry.get('rel_l1', float('nan')):.4f}", flush=True)
            except Exception as exc:  # noqa: BLE001
                row["candidates"][cand] = {"error": repr(exc)[:400]}
                print(f"{name:12s} {cand:14s} FAILED: {exc!r}", flush=True)
                traceback.print_exc()

        # --- FP8 current path
        wq, ws = fp8_weight(w16)
        xq, xs = sglang_per_token_quant_fp8(x16, dtype=torch.float8_e4m3fn)
        add("fp8_sgl",
            lambda: fp8_scaled_mm(xq, wq.t(), xs, ws, torch.float16, bias=None),
            lambda: sglang_per_token_quant_fp8(x16, dtype=torch.float8_e4m3fn),
            lambda: fp8_scaled_mm(xq, wq.t(), xs, ws, torch.float16, bias=None))
        # --- FP8 via cuBLASLt row-wise
        ws_t = ws.t().contiguous()
        add("fp8_torch",
            lambda: torch._scaled_mm(xq, wq.t(), scale_a=xs, scale_b=ws_t, out_dtype=torch.float16),
            lambda: sglang_per_token_quant_fp8(x16, dtype=torch.float8_e4m3fn),
            lambda: torch._scaled_mm(xq, wq.t(), scale_a=xs, scale_b=ws_t, out_dtype=torch.float16))
        # --- NVFP4 via flashinfer
        x_gs = (FP8_MAX * FP4_MAX / x16.float().abs().amax()).to(torch.float32)
        w_gs = (FP8_MAX * FP4_MAX / w16.float().abs().amax()).to(torch.float32)
        alpha = (1.0 / (x_gs * w_gs)).to(torch.float32)
        x4, x_sf = fp4_quantize(x16, x_gs)
        w4, w_sf = fp4_quantize(w16, w_gs)
        if x_sf.dtype != torch.float8_e4m3fn:
            x_sf = x_sf.view(torch.float8_e4m3fn)
        if w_sf.dtype != torch.float8_e4m3fn:
            w_sf = w_sf.view(torch.float8_e4m3fn)
        w4_t, w_sf_t = w4.t(), w_sf.t()
        out = torch.empty((M, N), dtype=torch.float16, device=dev)
        for be in backends:
            xs_be = x_sf.to(torch.uint8) if be == "trtllm" else x_sf
            ws_be = w_sf_t.to(torch.uint8) if be == "trtllm" else w_sf_t

            def gemm(be=be, xs_be=xs_be, ws_be=ws_be):
                return mm_fp4(x4, w4_t, xs_be, ws_be, alpha, torch.float16, out, backend=be)
            try:
                with autotune():
                    gemm()
            except Exception as exc:  # noqa: BLE001
                row["candidates"][f"nvfp4_{be}"] = {"error": "warmup: " + repr(exc)[:400]}
                print(f"{name:12s} nvfp4_{be:9s} FAILED at warmup: {exc!r}", flush=True)
                continue
            add(f"nvfp4_{be}", gemm, lambda: fp4_quantize(x16, x_gs), gemm)
        report["shapes"][name] = row
        del x16, w16, ref, xq, xs, wq, ws, x4, x_sf, w4, w_sf, out
        torch.cuda.empty_cache()

    # decision rule: sum over the 4 shapes x 36 blocks
    def total(cand, key):
        vals = [s["candidates"].get(cand, {}).get(key) for s in report["shapes"].values()]
        return None if any(v is None for v in vals) else 36 * sum(vals) / 1000.0  # ms

    report["decision"] = {"fp8_sgl_total_ms": total("fp8_sgl", "total_us"), "fp8_sgl_gemm_ms": total("fp8_sgl", "gemm_us")}
    for cand in sorted({c for s in report["shapes"].values() for c in s["candidates"]}):
        t = total(cand, "total_us")
        if t is not None:
            report["decision"][f"{cand}_total_ms"] = t
            report["decision"][f"{cand}_ratio_vs_fp8_sgl_total"] = t / report["decision"]["fp8_sgl_total_ms"]
    report["decision"]["rule"] = "integrate NVFP4 only if best nvfp4 total <= 0.70 x fp8_sgl total; switch FP8 backend if fp8_torch gemm <= 0.85 x fp8_sgl gemm"
    with open(a.json, "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps(report["decision"], indent=1))


if __name__ == "__main__":
    main()
