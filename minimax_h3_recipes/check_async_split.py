"""Check the split async Ulysses exchange against the packed synchronous one.

The three async exchanges move the same bytes as one packed exchange, so the
results must be identical -- any difference is a bug, not a tolerance. Also
times both, since the split trades two extra barrier pairs for the chance to
overlap qk-norm+RoPE with a transfer.

Usage:
    NCCL_NVLS_ENABLE=0 SGLANG_DIFFUSION_FAST_ULYSSES=1 \
        torchrun --nproc_per_node=<ws> check_async_split.py
"""

from __future__ import annotations

import os
import statistics

import torch
import torch.distributed as dist

H3_NUM_HEADS = 56
H3_HEAD_DIM = 128
H3_ROPE_DIM = 96
H3_NUM_LAYERS = 50
H3_SEQ_LEN = 38051


def _median_ms(fn, iters=25, warmup=8) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        dist.barrier()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)

    from sglang.multimodal_gen import envs
    from sglang.multimodal_gen.runtime.distributed.parallel_state import (
        maybe_init_distributed_environment_and_model_parallel,
    )

    maybe_init_distributed_environment_and_model_parallel(
        tp_size=1, sp_size=world, ulysses_degree=world
    )
    envs.SGLANG_DIFFUSION_FAST_ULYSSES = True

    from sglang.kernels.ops.diffusion.qknorm_rope import (
        fused_inplace_qknorm_rope,
        fused_inplace_qknorm_rope_single,
    )
    from sglang.multimodal_gen.runtime.layers.usp import (
        _usp_input_all_to_all_packed_qkv,
        fast_ulysses_async_input_exchange,
        fast_ulysses_wait,
    )

    device = torch.device("cuda", local_rank)
    dtype = torch.bfloat16
    seq_global = -(-H3_SEQ_LEN // world) * world
    s_local = seq_global // world

    torch.manual_seed(1234 + dist.get_rank())
    q0, k0, v0 = (
        torch.randn(s_local, H3_NUM_HEADS, H3_HEAD_DIM, dtype=dtype, device=device)
        for _ in range(3)
    )
    qw = torch.randn(H3_HEAD_DIM, dtype=dtype, device=device)
    kw = torch.randn(H3_HEAD_DIM, dtype=dtype, device=device)
    cache = torch.randn(seq_global, H3_ROPE_DIM, dtype=dtype, device=device)
    positions = torch.arange(s_local, device=device)
    KW = dict(
        is_neox=True,
        eps=1e-6,
        head_dim=H3_HEAD_DIM,
        rope_dim=H3_ROPE_DIM,
        round_norm_before_rope=True,
    )

    def reference():
        """Today's path: one fused kernel for q+k, then one packed exchange."""
        q, k, v = q0.clone(), k0.clone(), v0.clone()
        fused_inplace_qknorm_rope(q, k, qw, kw, cache, positions, **KW)
        return _usp_input_all_to_all_packed_qkv(q, k, v)

    def split_async():
        """v leaves first, then q's kernel, then q leaves, then k's kernel."""
        q, k, v = q0.clone(), k0.clone(), v0.clone()
        v_h = fast_ulysses_async_input_exchange(v, "v")
        fused_inplace_qknorm_rope_single(q, qw, cache, positions, **KW)
        q_h = fast_ulysses_async_input_exchange(q, "q")
        fused_inplace_qknorm_rope_single(k, kw, cache, positions, **KW)
        k_h = fast_ulysses_async_input_exchange(k, "k")
        return (
            fast_ulysses_wait(q_h),
            fast_ulysses_wait(k_h),
            fast_ulysses_wait(v_h),
        )

    ref = [t.clone() for t in reference()]
    got = [t.clone() for t in split_async()]
    failures = [
        name
        for name, a, b in zip("qkv", ref, got)
        if a.shape != b.shape or not torch.equal(a, b)
    ]

    t_ref = _median_ms(reference)
    t_split = _median_ms(split_async)

    verdict = torch.tensor([len(failures)], device=device)
    dist.all_reduce(verdict)
    if failures:
        print(f"rank{dist.get_rank()} MISMATCH in {failures}", flush=True)
    if dist.get_rank() == 0:
        print(f"\n=== ws={world}  s_local={s_local}  heads={H3_NUM_HEADS} ===")
        print(f"{'packed sync (norm+rope + 1 a2a)':<38}{t_ref:>9.3f} ms")
        print(f"{'split async (2 kernels + 3 a2a)':<38}{t_split:>9.3f} ms"
              f"   {t_ref / t_split:.3f}x")
        print(f"per denoise step ({H3_NUM_LAYERS} blocks): "
              f"{t_ref * H3_NUM_LAYERS:.1f} ms -> {t_split * H3_NUM_LAYERS:.1f} ms")
        print(f"PARITY {'FAIL' if verdict.item() else 'PASS'}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
