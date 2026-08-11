"""Attribute a MiniMax-H3 Nsight report: how much of the denoise is the a2a.

Answers two things off one report:

  1. Coverage -- how many GPUs the capture actually contains. A rank that never
     opened its own capture range contributes nothing, so a multi-GPU run can
     silently produce a single-GPU report.
  2. Attribution -- the share of the denoise loop spent in the Ulysses
     exchange, split into transfer and relayout, using the NVTX ranges
     projected onto GPU time.

Given two reports it prints them side by side, which is how a baseline capture
and a fast-ulysses capture get compared. The relayout rows vanish on the
fast-ulysses side: that path folds the permutation into copy strides, so those
ranges are never entered.

Usage: python analyze_nsys.py <baseline.nsys-rep> [<other.nsys-rep>]
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import sqlite3
import subprocess
import sys

NSYS = os.environ.get("H3_NSYS_BIN", "/usr/local/cuda/bin/nsys")

# The ranges that make up one attention block's collective, in the order they
# run. `usp.*` come from usp.py, the `h3_attn_*` pair from the model.
TRANSFER_RANGES = ("usp.transfer_in", "usp.transfer_out")
RELAYOUT_RANGES = (
    "usp.relayout_pack_qkv",
    "usp.relayout_out_pre",
    "usp.relayout_out_post",
)
A2A_TOTAL_RANGES = ("h3_attn_ulysses_a2a_in", "h3_attn_ulysses_a2a_out")
ATTENTION_RANGE = "h3_attn_varlen"
LOOP_RANGE = "denoising_loop"


def _stats_csv(report: str, kind: str) -> list[dict]:
    proc = subprocess.run(
        [NSYS, "stats", "--report", kind, "--format", "csv", "--force-export=true",
         report],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        print(f"nsys stats {kind} failed:\n{proc.stderr[-2000:]}", file=sys.stderr)
        return []
    # nsys prints progress lines before the CSV; the header starts the table,
    # and which column it starts with depends on the report.
    lines = proc.stdout.splitlines()
    for index, line in enumerate(lines):
        if line.startswith(("Time (%)", "Range,", '"Time (%)')):
            return list(csv.DictReader(io.StringIO("\n".join(lines[index:]))))
    return []


def _device_coverage(report: str) -> dict[int, int]:
    """Kernel count per device id, straight from the exported sqlite."""
    sqlite_path = report.replace(".nsys-rep", ".sqlite")
    if not os.path.exists(sqlite_path):
        subprocess.run(
            [NSYS, "export", "--type", "sqlite", "--force-overwrite=true",
             "--output", sqlite_path, report],
            capture_output=True,
            text=True,
        )
    if not os.path.exists(sqlite_path):
        return {}
    con = sqlite3.connect(sqlite_path)
    try:
        rows = con.execute(
            "SELECT deviceId, COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL "
            "GROUP BY deviceId ORDER BY deviceId"
        ).fetchall()
    except sqlite3.OperationalError:
        return {}
    finally:
        con.close()
    return {int(device): int(count) for device, count in rows}


def _ns(row: dict, *names: str) -> float:
    for name in names:
        if name in row and row[name]:
            return float(row[name])
    return 0.0


def load(report: str) -> dict:
    """Per-rank GPU milliseconds for every NVTX range in one report."""
    proj = _stats_csv(report, "nvtx_gpu_proj_sum")
    totals: dict[str, float] = {}
    instances: dict[str, int] = {}
    for row in proj:
        # nsys prefixes push/pop range names with the domain, which is empty
        # for `torch.cuda.nvtx`, leaving a bare leading colon.
        name = (row.get("Range") or row.get("Name") or "").strip().lstrip(":")
        totals[name] = totals.get(name, 0.0) + _ns(
            row, "Total Proj Time (ns)", "Total Time (ns)"
        )
        instances[name] = instances.get(name, 0) + int(
            float(row.get("Range Instances") or row.get("Instances") or 0)
        )
    # Every range is summed across ranks, so `denoising_loop` runs once per
    # rank and its instance count is the group size.
    ranks = max(instances.get(LOOP_RANGE, 1), 1)
    ms = {name: value / 1e6 / ranks for name, value in totals.items()}
    return {
        "report": report,
        "ranks": ranks,
        "ms": ms,
        "instances": {n: v // ranks for n, v in instances.items()},
        "coverage": _device_coverage(report),
    }


def summarize(data: dict) -> dict:
    ms = data["ms"]
    transfer = sum(ms.get(n, 0.0) for n in TRANSFER_RANGES)
    relayout = sum(ms.get(n, 0.0) for n in RELAYOUT_RANGES)
    a2a = sum(ms.get(n, 0.0) for n in A2A_TOTAL_RANGES) or (transfer + relayout)
    return {
        "loop": ms.get(LOOP_RANGE, 0.0),
        "attention": ms.get(ATTENTION_RANGE, 0.0),
        "a2a": a2a,
        "transfer": transfer,
        "relayout": relayout,
    }


def _print_one(data: dict) -> None:
    print(f"=== {os.path.basename(data['report'])} ===")
    coverage = data["coverage"]
    if coverage:
        counts = "  ".join(f"dev{d}:{c:,}" for d, c in coverage.items())
        print(f"GPU coverage: {len(coverage)} device(s)   {counts}")
        if len(coverage) == 1:
            print("  WARNING: single-GPU report. If this was a multi-GPU run, "
                  "some ranks never opened their capture range.")
    print(f"ranks: {data['ranks']}  (times per rank)\n")

    s = summarize(data)
    rows = list(TRANSFER_RANGES + RELAYOUT_RANGES + A2A_TOTAL_RANGES)
    print(f"{'NVTX range':<30}{'calls':>10}{'GPU ms':>11}{'% loop':>10}")
    for name in rows:
        if name not in data["ms"]:
            continue
        pct = 100 * data["ms"][name] / s["loop"] if s["loop"] else 0.0
        print(f"{name:<30}{data['instances'][name]:>10,}"
              f"{data['ms'][name]:>11.1f}{pct:>9.1f}%")
    if s["loop"]:
        print(f"\n{'denoising_loop':<30}{'':>10}{s['loop']:>11.1f}")
        print(f"{'  attention':<30}{'':>10}{s['attention']:>11.1f}"
              f"{100 * s['attention'] / s['loop']:>9.1f}%")
        print(f"{'  ulysses a2a':<30}{'':>10}{s['a2a']:>11.1f}"
              f"{100 * s['a2a'] / s['loop']:>9.1f}%")
        print(f"{'    transfer':<30}{'':>10}{s['transfer']:>11.1f}"
              f"{100 * s['transfer'] / s['loop']:>9.1f}%")
        print(f"{'    relayout':<30}{'':>10}{s['relayout']:>11.1f}"
              f"{100 * s['relayout'] / s['loop']:>9.1f}%")
    print()


def _print_compare(base: dict, other: dict) -> None:
    b, o = summarize(base), summarize(other)
    print("=== side by side (per rank, GPU ms) ===")
    print(f"  A = {os.path.basename(base['report'])}")
    print(f"  B = {os.path.basename(other['report'])}\n")
    print(f"{'':<22}{'A':>12}{'B':>12}{'B-A':>12}{'B/A':>9}")
    for label, key in (
        ("denoising_loop", "loop"),
        ("  attention", "attention"),
        ("  ulysses a2a", "a2a"),
        ("    transfer", "transfer"),
        ("    relayout", "relayout"),
    ):
        av, ov = b[key], o[key]
        ratio = f"{ov / av:.3f}" if av else "-"
        print(f"{label:<22}{av:>12.1f}{ov:>12.1f}{ov - av:>+12.1f}{ratio:>9}")
    if b["loop"] and o["loop"]:
        print(f"\na2a share of loop:  A {100 * b['a2a'] / b['loop']:.1f}%"
              f"   ->  B {100 * o['a2a'] / o['loop']:.1f}%")
        saved = b["a2a"] - o["a2a"]
        print(f"a2a time saved:     {saved:.1f} ms/rank per request"
              f"  ({100 * saved / b['loop']:.1f}% of the baseline loop)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report")
    parser.add_argument("other", nargs="?")
    args = parser.parse_args()

    base = load(args.report)
    _print_one(base)
    if args.other:
        other = load(args.other)
        _print_one(other)
        _print_compare(base, other)


if __name__ == "__main__":
    main()
