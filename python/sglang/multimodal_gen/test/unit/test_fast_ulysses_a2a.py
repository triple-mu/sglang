"""4-GPU parity tests: fast_ulysses backend vs NCCL Ulysses a2a.

Not registered in CI (experimental). Run on a 4-GPU NVLink host:

    # expected FAIL (fast path off -> counter assertion trips):
    torchrun --nproc_per_node=4 \
        python/sglang/multimodal_gen/test/unit/test_fast_ulysses_a2a.py
    # expected PASS (fast sync a2a; TYPE defaults to none = auto):
    SGLANG_DIFFUSION_ENABLE_FAST_ULYSSES=1 \
    torchrun --nproc_per_node=4 \
        python/sglang/multimodal_gen/test/unit/test_fast_ulysses_a2a.py
    # expected PASS for each transfer type (base / tma / ce):
    SGLANG_DIFFUSION_ENABLE_FAST_ULYSSES=1 \
    SGLANG_DIFFUSION_FAST_ULYSSES_TYPE=ce \
    torchrun --nproc_per_node=4 \
        python/sglang/multimodal_gen/test/unit/test_fast_ulysses_a2a.py
    # expected PASS (pipelined QKV + per-tensor norm+rope; combine with any TYPE):
    SGLANG_DIFFUSION_ENABLE_FAST_ULYSSES=1 \
    SGLANG_DIFFUSION_FAST_ULYSSES_PIPELINE_QKV=1 \
    SGLANG_DIFFUSION_FAST_ULYSSES_TYPE=ce \
    torchrun --nproc_per_node=4 \
        python/sglang/multimodal_gen/test/unit/test_fast_ulysses_a2a.py
"""

import os

import torch

from sglang.jit_kernel.diffusion.norm_rope import (
    can_use_fused_inplace_norm_rope,
    fused_inplace_norm_rope,
)
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    maybe_init_distributed_environment_and_model_parallel,
)
from sglang.multimodal_gen.runtime.layers import fast_ulysses_backend
from sglang.multimodal_gen.runtime.layers.usp import (
    _usp_input_all_to_all,
    _usp_output_all_to_all,
)

B, S_LOCAL, H, D = 1, 18900, 40, 128  # Wan2.1 14B 720p/81f, sp=4


def _rand(shape, seed, device):
    gen = torch.Generator(device=device).manual_seed(seed)
    return torch.randn(shape, generator=gen, device=device, dtype=torch.bfloat16)


def test_a2a_parity(rank: int, device: torch.device, world: int) -> None:
    x = _rand((B, S_LOCAL, H, D), 1234 + rank, device)
    ref = _usp_input_all_to_all(x, head_dim=2)  # no tag -> NCCL
    fast = _usp_input_all_to_all(x, head_dim=2, comm_tag="t_in")
    assert torch.equal(ref, fast), "mode0 mismatch"

    y = _rand((B, S_LOCAL * world, H // world, D), 4321 + rank, device)
    ref = _usp_output_all_to_all(y, head_dim=2)
    fast = _usp_output_all_to_all(y, head_dim=2, comm_tag="t_out")
    assert torch.equal(ref, fast), "mode1 mismatch"


def _ref_norm_rope(x, weight, cos, sin, eps):
    """Cross-head RMSNorm + interleaved (GPT-J) RoPE, all-fp32 reference."""
    b, s, n, d = x.shape
    xf = x.float().reshape(b, s, n * d)
    inv = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    xf = (xf * inv * weight.float()).reshape(b, s, n, d)
    x1, x2 = xf[..., 0::2], xf[..., 1::2]
    c = cos.view(1, s, 1, d // 2)
    si = sin.view(1, s, 1, d // 2)
    out = torch.empty_like(xf)
    out[..., 0::2] = x1 * c - x2 * si
    out[..., 1::2] = x2 * c + x1 * si
    return out.to(x.dtype)


def _make_norm_rope_inputs(device):
    gen = torch.Generator(device=device).manual_seed(9)
    wq = 1.0 + 0.1 * torch.randn(H * D, generator=gen, device=device)
    wk = 1.0 + 0.1 * torch.randn(H * D, generator=gen, device=device)
    ang = torch.rand((S_LOCAL, D // 2), generator=gen, device=device) * 6.28
    cos, sin = ang.cos().contiguous(), ang.sin().contiguous()
    return wq, wk, cos, sin


def test_pipelined_parity(rank: int, device: torch.device, world: int) -> None:
    """Production-shaped pipelined QKV sequence: three async a2as issued
    v-first (v, then q, then k -- v has no norm/rope so it goes out earliest),
    waited together, then the output a2a. Pure data movement (norm/rope stays
    in the model), so every compare is bitwise. Deferred compares keep the
    loop free of host syncs so the iterations pipeline and pressure the
    tag-buffer reuse; the clones right after the waits race a broken wait()
    while the a2a is still in flight."""
    filler = _rand((4096, 4096), 7 + rank, device)
    base = fast_ulysses_backend.fast_call_counts["a2a_async"]
    exact_checks: list[tuple[str, torch.Tensor, torch.Tensor]] = []
    for it in range(3):
        v = _rand((B, S_LOCAL, H, D), 1000 * it + rank, device)
        q = _rand((B, S_LOCAL, H, D), 2000 * it + rank, device)
        k = _rand((B, S_LOCAL, H, D), 3000 * it + rank, device)
        ref_v = _usp_input_all_to_all(v, head_dim=2)  # no tag -> NCCL
        ref_q = _usp_input_all_to_all(q, head_dim=2)
        ref_k = _usp_input_all_to_all(k, head_dim=2)

        hv = fast_ulysses_backend.pipelined_input_a2a(v, tag="usp_v", barrier=False)
        hq = fast_ulysses_backend.pipelined_input_a2a(q, tag="usp_q", barrier=False)
        hk = fast_ulysses_backend.pipelined_input_a2a(k, tag="usp_k", barrier=True)
        _ = filler @ filler  # main-stream compute inside the overlap window
        fv = hv.wait()
        fq = hq.wait()
        fk = hk.wait()
        exact_checks.append((f"v(iter {it})", fv.clone(), ref_v))
        exact_checks.append((f"q(iter {it})", fq.clone(), ref_q))
        exact_checks.append((f"k(iter {it})", fk.clone(), ref_k))
        # Close the barrier chain like production's output a2a; this is also
        # the cross-rank ordering anchor for the next iteration's buffer reuse.
        _usp_output_all_to_all(fv, head_dim=2, comm_tag="usp_out")
    for name, fast, ref in exact_checks:
        assert torch.equal(ref, fast), f"pipelined mismatch: {name}"
    assert fast_ulysses_backend.fast_call_counts["a2a_async"] == base + 9


def test_pipelined_split_norm_rope_parity(
    rank: int, device: torch.device, world: int
) -> None:
    """Production-shaped SPLIT sequence (wanvideo._pipelined_qkv_attention):
    v issued first, then q normed+roped in place (JIT kernel) and issued, then
    a filler matmul standing in for the k GEMM, then k normed+roped and issued
    with the shared barrier. Compares against fp32 norm+rope + NCCL a2a."""
    wq, wk, cos, sin = _make_norm_rope_inputs(device)
    eps = 1e-6
    filler = _rand((4096, 4096), 7 + rank, device)
    base = fast_ulysses_backend.fast_call_counts["a2a_async"]
    checks: list[tuple[str, torch.Tensor, torch.Tensor]] = []
    for it in range(3):
        v = _rand((B, S_LOCAL, H, D), 5000 * it + rank, device)
        q = _rand((B, S_LOCAL, H, D), 6000 * it + rank, device)
        k = _rand((B, S_LOCAL, H, D), 7000 * it + rank, device)
        ref_v = _usp_input_all_to_all(v, head_dim=2)  # no tag -> NCCL
        ref_q = _usp_input_all_to_all(_ref_norm_rope(q, wq, cos, sin, eps), head_dim=2)
        ref_k = _usp_input_all_to_all(_ref_norm_rope(k, wk, cos, sin, eps), head_dim=2)

        hv = fast_ulysses_backend.pipelined_input_a2a(v, tag="usp_v", barrier=False)
        fused_inplace_norm_rope(q, wq, cos, sin, cross_head=True, eps=eps)
        hq = fast_ulysses_backend.pipelined_input_a2a(q, tag="usp_q", barrier=False)
        _ = filler @ filler  # stands in for the k GEMM under q's a2a
        fused_inplace_norm_rope(k, wk, cos, sin, cross_head=True, eps=eps)
        hk = fast_ulysses_backend.pipelined_input_a2a(k, tag="usp_k", barrier=True)
        fv = hv.wait()
        fq = hq.wait()
        fk = hk.wait()
        checks.append((f"v(iter {it})", fv.clone(), ref_v))
        checks.append((f"q(iter {it})", fq.clone(), ref_q))
        checks.append((f"k(iter {it})", fk.clone(), ref_k))
        # Close the barrier chain like production's output a2a.
        _usp_output_all_to_all(fv, head_dim=2, comm_tag="usp_out")
    for name, fast, ref in checks:
        if name.startswith("v"):
            assert torch.equal(ref, fast), f"split pipelined mismatch: {name}"
        else:
            diff = (fast.float() - ref.float()).abs().max().item()
            if rank == 0:
                print(
                    f"split {name}: max abs diff vs fp32 ref = {diff:.5f}", flush=True
                )
            torch.testing.assert_close(fast, ref, atol=2e-2, rtol=2e-2)
    assert fast_ulysses_backend.fast_call_counts["a2a_async"] == base + 9


def main() -> None:
    maybe_init_distributed_environment_and_model_parallel(
        tp_size=1, sp_size=4, ulysses_degree=4
    )
    import torch.distributed as dist

    rank = dist.get_rank()
    world = dist.get_world_size()
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", rank)))

    test_a2a_parity(rank, device, world)
    assert (
        fast_ulysses_backend.fast_call_counts["a2a"] == 2
    ), f"fast a2a path not exercised: {fast_ulysses_backend.fast_call_counts}"

    if fast_ulysses_backend.can_use_pipelined_qkv(
        num_heads=H, head_dim=D, dtype=torch.bfloat16
    ):
        test_pipelined_parity(rank, device, world)
        if can_use_fused_inplace_norm_rope(D, H, True, True, torch.bfloat16):
            test_pipelined_split_norm_rope_parity(rank, device, world)
        elif rank == 0:
            print("norm_rope kernel unavailable; skipped the split test", flush=True)
    elif rank == 0:
        print("pipelined qkv disabled; skipped its parity test", flush=True)

    if rank == 0:
        print("PASS", flush=True)


if __name__ == "__main__":
    main()
