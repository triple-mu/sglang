#!/usr/bin/env python3
"""Per-frame PSNR/SSIM between two MP4s decoded by ffmpeg to rgb24.

Scope: sanity check of the fast path against the lossless server on x264-encoded
frames, not the uncompressed-RGB acceptance of the archived campaign.
"""
import argparse
import json
import math
import subprocess
import sys

import numpy as np


def probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height,nb_frames", "-of", "json", path],
                         capture_output=True, text=True, check=True).stdout
    s = json.loads(out)["streams"][0]
    return int(s["width"]), int(s["height"])


def frames(path):
    w, h = probe(path)
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.uint8).reshape(-1, h, w, 3)


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return math.inf if mse == 0 else 10 * math.log10(255.0 ** 2 / mse)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reference")
    ap.add_argument("candidate")
    ap.add_argument("--output")
    ap.add_argument("--mean-psnr", type=float, default=35.0)
    ap.add_argument("--min-psnr", type=float, default=30.0)
    ap.add_argument("--mean-ssim", type=float, default=0.98)
    a = ap.parse_args()
    ref, cand = frames(a.reference), frames(a.candidate)
    if ref.shape != cand.shape:
        raise SystemExit(f"shape mismatch {ref.shape} vs {cand.shape}")
    try:
        from skimage.metrics import structural_similarity
    except ImportError:
        structural_similarity = None
    p = [psnr(x, y) for x, y in zip(ref, cand)]
    s = [structural_similarity(x, y, channel_axis=-1, data_range=255) for x, y in zip(ref, cand)] if structural_similarity else None
    finite = [v for v in p if math.isfinite(v)]
    result = {
        "reference": a.reference, "candidate": a.candidate, "frames": int(ref.shape[0]),
        "shape": list(ref.shape[1:]), "exact_frames": len(p) - len(finite),
        "psnr_mean": (sum(finite) / len(finite)) if finite else None, "psnr_min": min(finite) if finite else None,
        "ssim_mean": (sum(s) / len(s)) if s else None, "ssim_min": min(s) if s else None,
        "thresholds": {"mean_psnr": a.mean_psnr, "min_psnr": a.min_psnr, "mean_ssim": a.mean_ssim},
        "ssim_available": s is not None,
    }
    result["passed"] = bool(
        (result["psnr_mean"] is None or result["psnr_mean"] >= a.mean_psnr)
        and (result["psnr_min"] is None or result["psnr_min"] >= a.min_psnr)
        and (s is None or result["ssim_mean"] >= a.mean_ssim)
    )
    text = json.dumps(result, indent=1)
    if a.output:
        with open(a.output, "w") as f:
            f.write(text)
    print(text)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
