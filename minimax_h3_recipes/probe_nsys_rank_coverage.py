"""Probe: does a rank-0-only cudaProfilerStart capture the other ranks?

sglang's diffusion GPU workers are separate spawned processes, and nsys tracks
a `--capture-range=cudaProfilerApi` range per process. This reproduces that
shape in a few seconds: two spawned workers, each running a distinguishable
kernel on its own GPU, with only one of them issuing the start/stop pair.

Run under:
  nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop \
      --trace=cuda,nvtx --trace-fork-before-exec=true \
      -o <out> python probe_nsys_rank_coverage.py [--all-ranks]

Then count the devices in the report:
  nsys stats --report cuda_gpu_kern_sum <out>.nsys-rep
"""

from __future__ import annotations

import argparse
import multiprocessing as mp

import torch
import torch.cuda.nvtx as nvtx


def worker(rank: int, all_ranks: bool, use_graph: bool) -> None:
    torch.cuda.set_device(rank)
    a = torch.randn(2048, 2048, device="cuda", dtype=torch.float32)

    # Warm the kernel outside the capture range.
    for _ in range(3):
        a @ a
    torch.cuda.synchronize()

    graph = None
    if use_graph:
        # Capture before the profiler range so the report contains replays,
        # which is what --cuda-graph-trace=node has to expand.
        static = torch.randn(2048, 2048, device="cuda", dtype=torch.float32)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                static @ static
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(5):
                static = static @ static
        torch.cuda.synchronize()

    bracket = all_ranks or rank == 0
    if bracket:
        torch.cuda.cudart().cudaProfilerStart()
    nvtx.range_push(f"probe_rank{rank}")
    if graph is not None:
        for _ in range(4):
            graph.replay()
    for _ in range(20):
        a = a @ a
    torch.cuda.synchronize()
    nvtx.range_pop()
    if bracket:
        torch.cuda.cudart().cudaProfilerStop()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all-ranks", action="store_true")
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--world-size", type=int, default=2)
    args = parser.parse_args()

    ctx = mp.get_context("spawn")
    procs = [
        ctx.Process(target=worker, args=(rank, args.all_ranks, args.graph))
        for rank in range(args.world_size)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
        assert p.exitcode == 0, f"worker failed: {p.exitcode}"


if __name__ == "__main__":
    main()
