#!/usr/bin/env python3
"""Render the FP4 prescreen markdown tables from bench.json (numbers traceable to the file)."""
import json
import sys

d = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "fp4-gemm-prescreen/results/bench.json"))
order = ["fp8_sgl", "fp8_torch", "nvfp4_cudnn", "nvfp4_cutlass", "nvfp4_cute-dsl"]
label = {"fp8_sgl": "FP8 现路径（sgl_kernel fp8_scaled_mm + per-token quant）", "fp8_torch": "FP8 torch._scaled_mm（cuBLASLt row-wise）",
         "nvfp4_cudnn": "NVFP4 flashinfer mm_fp4 cudnn", "nvfp4_cutlass": "NVFP4 flashinfer mm_fp4 cutlass", "nvfp4_cute-dsl": "NVFP4 flashinfer mm_fp4 cute-dsl"}
shapes = list(d["shapes"].items())
print("| 候选 | " + " | ".join(f"{n} GEMM (us)" for n, _ in shapes) + " | 36 block 合计 GEMM+量化 (ms) | 相对 FP8 现路径 |")
print("|---|" + "---:|" * (len(shapes) + 2))
base = d["decision"]["fp8_sgl_total_ms"]
for c in order:
    cells, ok = [], True
    for n, s in shapes:
        v = s["candidates"].get(c, {})
        if "error" in v:
            cells.append("—"); ok = False
        else:
            cells.append(f"{v['gemm_us']:.0f} ({v['gemm_us_min']:.0f}–{v['gemm_us_max']:.0f})")
    tot = d["decision"].get(f"{c}_total_ms")
    ratio = "对照" if c == "fp8_sgl" else (f"{tot / base:.2f}" if tot else "—")
    print(f"| {label[c]} | " + " | ".join(cells) + f" | {tot:.1f} | {ratio} |" if tot else f"| {label[c]} | " + " | ".join(cells) + " | — | — |")
print()
print("| 候选 | " + " | ".join(f"{n} 达到 TFLOPS" for n, _ in shapes) + " | 激活量化 (us，四 shape 合计) | rel_l1（随机正态） |")
print("|---|" + "---:|" * (len(shapes) + 2))
for c in order:
    vals = [s["candidates"].get(c, {}) for _, s in shapes]
    if any("error" in v or not v for v in vals):
        continue
    print(f"| {label[c]} | " + " | ".join(f"{v['tflops']:.0f}" for v in vals) + f" | {sum(v['quant_us'] for v in vals):.0f} | {vals[0].get('rel_l1', float('nan')):.4f} |")
errs = {c: s["candidates"][c]["error"][:90] for _, s in shapes for c in s["candidates"] if "error" in s["candidates"][c]}
if errs:
    print()
    for c, e in errs.items():
        print(f"- {c}: {e}")
