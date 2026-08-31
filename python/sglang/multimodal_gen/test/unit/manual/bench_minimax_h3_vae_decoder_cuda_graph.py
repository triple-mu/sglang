"""
Microbench: MiniMax-H3 VAE decoder per-tile forward, eager vs CUDA graph (V2).

Builds the production ViT3DDecoder (36 layers, heads=32, dim_head=64, dim
2048) with random weights (kernel timing does not depend on weight values),
feeds the production decode tile [1, 24, 7, 16, 16] (1797 tokens) under
autocast fp16, and times a 98-tile loop -- the per-request per-rank tile count
-- through the eager forward and through DecoderTileCudaGraphRunner replay.

Measurement is A-B-B-A (eager, graph, graph, eager) with per-arm warmup;
each arm reports wall us/tile (host-timed, synced) and CPU us/tile
(process_time inside the loop, the launch-API share). A torch.equal parity
check between eager and replayed outputs runs before any timing is printed.

Usage:
    python bench_minimax_h3_vae_decoder_cuda_graph.py [--tiles 98] [--repeats 3]
"""

import argparse
import json
import os
import socket
import time

import torch

from sglang.multimodal_gen.runtime.managers.forward_context import (
    set_forward_context,
)
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.decoder_cuda_graph import (
    DecoderTileCudaGraphRunner,
)
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.vae_vit import (
    ViT3DDecoder,
)

# vit_decoder_kwargs of MiniMaxH3VideoVAE (minimax_h3.py) at full depth.
PROD_DECODER_KWARGS = dict(
    patch_size=16,
    patch_size_t=4,
    t_causal=False,
    in_channels=24,
    out_channels=3,
    num_layers=36,
    heads=32,
    dim_head=64,
    norm_type="rms_norm",
    norm_affine=True,
    qk_norm_type="rms_norm",
    qk_norm_affine=False,
    ffn_activation_fn="silu",
    ffn_use_gated=True,
    rope_theta=100.0,
    rope_dim_ratio=0.75,
    bias=True,
    eps=1e-5,
    num_register_tokens=4,
)
TILE_SHAPE = (1, 24, 7, 16, 16)


def _ensure_runtime():
    from sglang.multimodal_gen.runtime import server_args as server_args_module
    from sglang.multimodal_gen.runtime.distributed.parallel_state import (
        maybe_init_distributed_environment_and_model_parallel,
        model_parallel_is_initialized,
    )
    from sglang.multimodal_gen.runtime.server_args import set_global_server_args
    from sglang.multimodal_gen.test.single_test_file.component_accuracy.utils import (
        ensure_distributed_env_defaults,
    )
    from sglang.multimodal_gen.test.unit.conftest import _make_unit_server_args

    if server_args_module._global_server_args is None:
        set_global_server_args(_make_unit_server_args())
    if not model_parallel_is_initialized():
        if "MASTER_PORT" not in os.environ:
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                os.environ["MASTER_PORT"] = str(sock.getsockname()[1])
        ensure_distributed_env_defaults()
        maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=1)


def _time_loop(fn, x, tiles):
    torch.cuda.synchronize()
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    for _ in range(tiles):
        fn(x)
    cpu_elapsed = time.process_time() - cpu_start
    torch.cuda.synchronize()
    wall_elapsed = time.perf_counter() - wall_start
    return wall_elapsed / tiles * 1e6, cpu_elapsed / tiles * 1e6


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiles", type=int, default=98)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=8)
    args = parser.parse_args()

    _ensure_runtime()
    torch.manual_seed(0)
    decoder = ViT3DDecoder(**PROD_DECODER_KWARGS).to("cuda")
    decoder.eval()
    decoder.prepare_autocast_linear_weights(torch.float16)
    runner = DecoderTileCudaGraphRunner(decoder=decoder, offload_owner=None)

    # Production tiles are strided views of the clip latent canvas; layout is
    # part of the runner signature, so bench the same layout.
    canvas = torch.randn((1, 24, 7, 48, 112), device="cuda")
    x = canvas[..., 0:16, 0:16]
    x_check = canvas[..., 0:16, 16:32]
    assert x.shape == TILE_SHAPE

    results = {"tiles": args.tiles, "arms": []}
    with torch.no_grad(), set_forward_context(
        current_timestep=0, attn_metadata=None
    ), torch.autocast("cuda", dtype=torch.float16):
        # Parity first: replayed output must equal eager for a fresh input.
        runner.run(x)  # eager first sight
        runner.run(x_check)  # capture + self-check replay
        assert runner._disabled_reason is None, runner._disabled_reason
        ref = decoder(x_check)
        out = runner.run(x_check)
        assert torch.equal(out, ref), "replay is not bit-exact vs eager"
        print("parity: torch.equal(replay, eager) OK")

        for _ in range(args.warmup):
            decoder(x)
            runner.run(x)

        # A-B-B-A: eager, graph, graph, eager; repeated.
        for rep in range(args.repeats):
            for arm, fn in (
                ("eager", decoder),
                ("graph", runner.run),
                ("graph", runner.run),
                ("eager", decoder),
            ):
                wall_us, cpu_us = _time_loop(fn, x, args.tiles)
                results["arms"].append(
                    {
                        "rep": rep,
                        "arm": arm,
                        "wall_us_per_tile": round(wall_us, 1),
                        "cpu_us_per_tile": round(cpu_us, 1),
                    }
                )
                print(
                    f"rep={rep} arm={arm:5s} wall={wall_us:8.1f} us/tile "
                    f"cpu={cpu_us:8.1f} us/tile"
                )

        # Enqueue probe: host time to submit ONE tile with an idle GPU and
        # empty queue. This is the per-tile CPU dispatch/launch-API cost that
        # stays hidden while the GPU is the bottleneck on a quiet host but
        # becomes exposed wall time under production CPU contention.
        for arm, fn in (("eager", decoder), ("graph", runner.run)):
            samples = []
            for _ in range(10):
                torch.cuda.synchronize()
                start = time.perf_counter()
                fn(x)
                samples.append((time.perf_counter() - start) * 1e6)
            samples.sort()
            median = samples[len(samples) // 2]
            results[f"enqueue_{arm}_us_per_tile"] = round(median, 1)
            print(f"enqueue arm={arm:5s} median={median:8.1f} us/tile")
        torch.cuda.synchronize()

    def _best(arm):
        walls = [a["wall_us_per_tile"] for a in results["arms"] if a["arm"] == arm]
        cpus = [a["cpu_us_per_tile"] for a in results["arms"] if a["arm"] == arm]
        return min(walls), min(cpus)

    eager_wall, eager_cpu = _best("eager")
    graph_wall, graph_cpu = _best("graph")
    summary = {
        "eager_wall_us_per_tile": eager_wall,
        "graph_wall_us_per_tile": graph_wall,
        "eager_cpu_us_per_tile": eager_cpu,
        "graph_cpu_us_per_tile": graph_cpu,
        "wall_saving_us_per_tile": round(eager_wall - graph_wall, 1),
        "wall_saving_ms_per_98_tiles": round((eager_wall - graph_wall) * 98 / 1e3, 1),
        "speedup": round(eager_wall / graph_wall, 3),
        "enqueue_eager_us_per_tile": results["enqueue_eager_us_per_tile"],
        "enqueue_graph_us_per_tile": results["enqueue_graph_us_per_tile"],
    }
    results["summary"] = summary
    print(json.dumps(summary, indent=2))
    print(json.dumps(results))


if __name__ == "__main__":
    main()
