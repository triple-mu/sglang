import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.ops.diffusion import (
    ulysses_lowp_finalize_stats,
    ulysses_lowp_k_sum_v_amax,
    ulysses_lowp_payload_spec,
    ulysses_lowp_quant_pack,
    ulysses_lowp_scale_widths,
    ulysses_lowp_unpack_for_sage,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=30, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)

# MiniMax-H3 t2va at 768p / 5 s on eight ranks: 37888 packed tokens, 56 heads.
WORLD_SIZE = 8
HEADS = 56


def _sm120() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0)


@marker.parametrize("local_sequence", [4736, 2368], [2368])
@marker.benchmark("kernel", ["k_sum_v_amax", "quant_pack", "unpack_for_sage"])
def benchmark(local_sequence: int, kernel: str):
    if not _sm120():
        return marker.skip("compute capability 12.0 only")
    device = torch.device("cuda")
    g = torch.Generator(device="cpu").manual_seed(0)
    mother = torch.randn(1, local_sequence, HEADS, 3, 128, generator=g).to(
        device, torch.bfloat16
    )
    q, k, v = (mother[:, :, :, i, :] for i in range(3))
    spec = ulysses_lowp_payload_spec(
        batch=1, local_sequence=local_sequence, num_heads=HEADS, world_size=WORLD_SIZE
    )
    used = spec.global_sequence - 14
    stats = ulysses_lowp_k_sum_v_amax(k, v)
    gathered = stats.unsqueeze(0).expand(WORLD_SIZE, -1, -1, -1, -1).contiguous()
    k_mean, v_scale = ulysses_lowp_finalize_stats(
        gathered, world_size=WORLD_SIZE, used_sequence=used, dtype=torch.bfloat16
    )
    payload = torch.empty(spec.payload_shape, dtype=torch.uint8, device=device)
    ulysses_lowp_quant_pack(
        q,
        k,
        v,
        k_mean,
        v_scale,
        rank=0,
        world_size=WORLD_SIZE,
        used_sequence=used,
        out=payload,
    )
    recv = payload.clone()
    cut = -(-used // 64) * 64
    q_width, k_width = ulysses_lowp_scale_widths(cut)
    h, S = spec.local_heads, spec.global_sequence
    outs = (
        torch.empty(1, h, S, 128, dtype=torch.int8, device=device),
        torch.empty(1, h, S, 128, dtype=torch.int8, device=device),
        torch.empty(1, h, 128, S, dtype=torch.float8_e4m3fn, device=device),
        torch.empty(1, h, q_width, dtype=torch.float32, device=device),
        torch.empty(1, h, k_width, dtype=torch.float32, device=device),
    )

    if kernel == "k_sum_v_amax":
        return marker.do_bench(
            lambda k, v: ulysses_lowp_k_sum_v_amax(k, v),
            input_args=(k, v),
            graph_clone_args=(),
        )
    if kernel == "quant_pack":
        return marker.do_bench(
            lambda q, k, v, km, vs, out: ulysses_lowp_quant_pack(
                q,
                k,
                v,
                km,
                vs,
                rank=0,
                world_size=WORLD_SIZE,
                used_sequence=used,
                out=out,
            ),
            input_args=(q, k, v, k_mean, v_scale, payload),
            graph_clone_args=(),
            memory_output=(payload,),
        )
    return marker.do_bench(
        lambda recv, outs: ulysses_lowp_unpack_for_sage(
            recv, spec=spec, scale_sequence=cut, used_sequence=used, out=outs
        ),
        input_args=(recv, outs),
        graph_clone_args=(),
        memory_output=outs,
    )


if __name__ == "__main__":
    benchmark.run()
