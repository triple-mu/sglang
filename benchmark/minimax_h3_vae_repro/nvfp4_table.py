#!/usr/bin/env python3
"""Markdown table for the NVFP4 vs FP8 end-to-end chain (same node, interleaved runs)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = sys.argv[1:] or ["fast-path-extra-high-fp8-2", "fast-path-extra-high-nvfp4",
                        "fast-path-extra-high-nvfp4-cutlass", "fast-path-extra-high-fp8-3",
                        "fast-path-extra-high-nvfp4-2"]
STAGES = [("MiniMaxH3TextEncodingStage", "text encode"), ("MiniMaxH3VisualEncodingStage", "VAE encode"),
          ("MiniMaxH3DenoisingStage", "DiT"), ("MiniMaxH3DecodingStage", "VAE DecodingStage")]


def load(run):
    s = json.loads((ROOT / run / "results" / "summary.json").read_text())
    return s


def fmt(v):
    return f"{v['median']:.1f} ({v['min']:.1f}–{v['max']:.1f})"


rows = {run: load(run) for run in RUNS if (ROOT / run / "results" / "summary.json").exists()}
control = [s for run, s in rows.items() if "fp8" in run]
ref_e2e = sum(s["e2e_ms"]["median"] for s in control) / len(control)
ref_dec = sum(s["stages_ms"]["MiniMaxH3DecodingStage"]["median"] for s in control) / len(control)
print("| 运行（顺序即执行顺序） | E2E (ms) median (min–max) | E2E 相对 FP8 均值 | " + " | ".join(f"{n} (ms)" for _, n in STAGES) + " | DecodingStage 相对 FP8 均值 | server 就绪 (s) |")
print("|---|---:|---:|" + "---:|" * len(STAGES) + "---:|---:|")
for run, s in rows.items():
    e2e = s["e2e_ms"]; dec = s["stages_ms"]["MiniMaxH3DecodingStage"]["median"]
    stages = " | ".join(fmt(s["stages_ms"][k]) if k in s["stages_ms"] else "—" for k, _ in STAGES)
    print(f"| {run} | {fmt(e2e)} | {(e2e['median'] / ref_e2e - 1) * 100:+.2f}% | {stages} | {(dec / ref_dec - 1) * 100:+.2f}% | {s['startup_s']:.0f} |")
print(f"\nFP8 对照均值：E2E {ref_e2e:.1f} ms，DecodingStage {ref_dec:.1f} ms（{len(control)} 次运行）")
nv = [s for run, s in rows.items() if "nvfp4" in run and "cutlass" not in run]
if nv:
    nv_dec = sum(s["stages_ms"]["MiniMaxH3DecodingStage"]["median"] for s in nv) / len(nv)
    nv_e2e = sum(s["e2e_ms"]["median"] for s in nv) / len(nv)
    print(f"NVFP4 cudnn 均值：E2E {nv_e2e:.1f} ms（{(nv_e2e / ref_e2e - 1) * 100:+.2f}%），DecodingStage {nv_dec:.1f} ms（{(nv_dec / ref_dec - 1) * 100:+.2f}%）")
