#!/usr/bin/env python3
"""Aggregate <run>/results/summary.json files into the report tables and judge the
pre-registered criteria against the archived campaign numbers."""
import argparse
import json
from pathlib import Path

ARCHIVED = {  # ms, medians from the archived 2026-09-18 campaign (different timing cohort)
    "baseline": {"e2e": 5484.54, "encode": 87.41, "decode": 867.98, "vae": 955.40, "mp4": 834.36},
    "final": {"e2e": 4501.47, "encode": 51.49, "decode": 391.61, "vae": 442.70, "mp4": 355.17},
}
ALIASES = [("encode", ("VisualEncoding", "visual_encoding", "vae.encode")),
           ("decode", ("Decoding", "decoding", "vae.decode")),
           ("dit", ("Denoising", "denoising", "dit")),
           ("text", ("TextEncoding", "text_encoding")),
           ("audio_encode", ("AudioEncoding",))]


def alias(name):
    for short, keys in ALIASES:
        if any(k.lower() in name.lower() for k in keys):
            return short
    return None


def load(root):
    runs = {}
    for f in sorted(root.glob("*/results/summary.json")):
        s = json.loads(f.read_text())
        row = {"id": f.parent.parent.name, "e2e": s["e2e_ms"], "stages": {}, "raw_stage_names": sorted(s["stages_ms"]),
               "startup_s": s.get("startup_s"), "host": s.get("hostname"), "outside": s.get("outside_pipeline_ms")}
        for name, st in s["stages_ms"].items():
            a = alias(name)
            if a and a not in row["stages"]:
                row["stages"][a] = st
        enc, dec = row["stages"].get("encode"), row["stages"].get("decode")
        if enc and dec:
            vals = [e + d for e, d in zip(enc["values"], dec["values"])]
            import statistics
            med = statistics.median(vals)
            row["stages"]["vae"] = {"median": med, "min": min(vals), "max": max(vals), "values": vals,
                                    "mad": statistics.median([abs(v - med) for v in vals])}
        runs[row["id"]] = row
    return runs


def fmt(st, key="median"):
    return "—" if not st else f"{st[key]:.2f}"


def rng(st):
    return "—" if not st else f"{st['min']:.0f}–{st['max']:.0f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    runs = load(a.root)
    cols = ["e2e", "text", "encode", "dit", "decode", "vae", "outside"]
    print("| 配置 | E2E (ms) | text | vae.encode | DiT | vae.decode | VAE 合计 | E2E 减 pipeline | 样本 |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for rid, r in runs.items():
        cells = [fmt(r["e2e"])] + [fmt(r["stages"].get(c)) for c in cols[1:6]] + [fmt(r["outside"])]
        print(f"| {rid} | " + " | ".join(cells) + f" | {r['e2e']['count']} |")
    print("\nmin–max: " + "; ".join(f"{rid}: e2e {rng(r['e2e'])}, decode {rng(r['stages'].get('decode'))}, vae {rng(r['stages'].get('vae'))}" for rid, r in runs.items()))
    print("\nstage names seen: " + "; ".join(f"{rid}: {r['raw_stage_names']}" for rid, r in runs.items()))
    verdict = {}
    base = runs.get("baseline-lossless")
    xh, hi = runs.get("fast-path-extra-high-fp8"), runs.get("fast-path-high-fp8")

    def within(val, ref, tol=0.05):
        return abs(val - ref) / ref <= tol

    if base and base["stages"].get("vae"):
        v = base["stages"]["vae"]["median"]
        verdict["baseline_vae_within_5pct_of_955.40"] = (within(v, ARCHIVED["baseline"]["vae"]), round(v, 2))
    if xh and xh["stages"].get("decode"):
        d = xh["stages"]["decode"]["median"]
        verdict["extra_high_decode_within_5pct_of_391.61"] = (within(d, ARCHIVED["final"]["decode"]), round(d, 2))
    if hi and hi["stages"].get("vae"):
        v = hi["stages"]["vae"]["median"]
        verdict["high_vae_within_5pct_of_442.70"] = (within(v, ARCHIVED["final"]["vae"]), round(v, 2))
        if base and base["stages"].get("vae"):
            red = 1 - v / base["stages"]["vae"]["median"]
            verdict["high_vae_reduction_vs_baseline_ge_50pct"] = (red >= 0.5, f"{red:.2%}")
    # same numbers against the archived decode + audio decode (DecodingStage covers both)
    adj = {"baseline_decode_plus_audio": 867.98 + 32.62, "final_decode_plus_audio": 391.61 + 40.22,
           "baseline_vae_plus_audio": 955.40 + 32.62, "final_vae_plus_audio": 442.70 + 40.22}
    if base and base["stages"].get("decode"):
        d = base["stages"]["decode"]["median"]
        verdict["info_baseline_decode_stage_vs_archived_decode_plus_audio_900.60"] = (within(d, adj["baseline_decode_plus_audio"]), round(d, 2))
    if xh and xh["stages"].get("decode"):
        d = xh["stages"]["decode"]["median"]
        verdict["info_extra_high_decode_stage_vs_archived_decode_plus_audio_431.83"] = (within(d, adj["final_decode_plus_audio"]), round(d, 2))
        if base and base["stages"].get("decode"):
            red = 1 - d / base["stages"]["decode"]["median"]
            verdict["info_extra_high_decode_stage_reduction_vs_baseline (archived 52.05%)"] = (red >= 0.45, f"{red:.2%}")
    print("\npre-registered criteria:")
    for k, (ok, val) in verdict.items():
        print(f"  [{'x' if ok else ' '}] {k}: {val}")
    if a.json:
        a.json.write_text(json.dumps({"runs": runs, "verdict": verdict}, indent=1, default=str))


if __name__ == "__main__":
    main()
