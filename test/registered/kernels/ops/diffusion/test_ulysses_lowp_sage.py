"""Single-GPU simulation of the low-precision Ulysses exchange over W ranks.

Every rank's shard is a strided view of one global tensor, the all-gather and
the all-to-all are done with torch indexing, and the unpacked operands are
compared against the pure-torch SM120 Sage quantiser reference.
"""

import sys

import pytest
import torch

from sglang.kernels.ops.diffusion import (
    can_use_sage_block_sparse_attn_sm120,
    can_use_ulysses_lowp_sage,
    sage_block_sparse_attn_sm120,
    sage_block_sparse_dense_block_index,
    ulysses_lowp_finalize_stats,
    ulysses_lowp_k_sum_v_amax,
    ulysses_lowp_payload_spec,
    ulysses_lowp_quant_pack,
    ulysses_lowp_scale_widths,
    ulysses_lowp_unpack_for_sage,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.sage_sm120_reference import (
    K_BLOCK,
    quantize_sage_kv,
    quantize_sage_q,
    sage_v_physical_rows,
)

register_cuda_ci(est_time=90, stage="base-b-kernel-unit", runner_config="1-gpu-large")

requires_sm120 = pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0)),
    reason="low-precision Ulysses kernels target compute capability 12.0 only",
)


def _global_qkv(batch, seqlen, heads, *, layout, used, seed):
    """bf16 Q/K/V ``[B, S, H, 128]`` as views of one fused-projection buffer."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    if layout == "interleaved":  # [T, H, 3, D]: head stride 3*D
        mother = torch.randn(batch, seqlen, heads, 3, 128, generator=g)
        mother = mother.to("cuda", torch.bfloat16)
        q, k, v = (mother[:, :, :, i, :] for i in range(3))
    else:  # [T, 3, H, D]: what qkv.split() views look like
        mother = torch.randn(batch, seqlen, 3, heads, 128, generator=g)
        mother = mother.to("cuda", torch.bfloat16)
        q, k, v = (mother[:, :, i] for i in range(3))
    # The packed sequence pads with zero rows; the kernels see those zeros.
    mother[:, used:] = 0
    return q, k, v


def _simulate(q, k, v, *, world_size, used):
    """Run every rank's pack and every rank's unpack on one device."""
    batch, seqlen, heads, _ = q.shape
    local = seqlen // world_size
    spec = ulysses_lowp_payload_spec(
        batch=batch, local_sequence=local, num_heads=heads, world_size=world_size
    )
    shards = [
        tuple(t[:, r * local : (r + 1) * local] for t in (q, k, v))
        for r in range(world_size)
    ]
    stats = torch.stack([ulysses_lowp_k_sum_v_amax(ks, vs) for _, ks, vs in shards])
    k_mean, v_scale = ulysses_lowp_finalize_stats(
        stats, world_size=world_size, used_sequence=used, dtype=q.dtype
    )
    payloads = []
    for r, (qs, ks, vs) in enumerate(shards):
        # The kernel never writes the 128-byte alignment tail of each chunk;
        # start from zeros so whole-payload comparisons stay meaningful.
        out = torch.zeros(spec.payload_shape, dtype=torch.uint8, device="cuda")
        ulysses_lowp_quant_pack(
            qs,
            ks,
            vs,
            k_mean,
            v_scale,
            rank=r,
            world_size=world_size,
            used_sequence=used,
            out=out,
        )
        payloads.append(out)
    cut = -(-used // K_BLOCK) * K_BLOCK
    q_width, k_width = ulysses_lowp_scale_widths(cut)
    h = spec.local_heads
    results = []
    for dest in range(world_size):
        recv = torch.stack([payloads[src][dest] for src in range(world_size)])
        outs = (
            torch.empty(batch, h, seqlen, 128, dtype=torch.int8, device="cuda"),
            torch.empty(batch, h, seqlen, 128, dtype=torch.int8, device="cuda"),
            torch.empty(
                batch, h, 128, seqlen, dtype=torch.float8_e4m3fn, device="cuda"
            ),
            torch.empty(batch, h, q_width, dtype=torch.float32, device="cuda"),
            torch.empty(batch, h, k_width, dtype=torch.float32, device="cuda"),
        )
        ulysses_lowp_unpack_for_sage(
            recv, spec=spec, scale_sequence=cut, used_sequence=used, out=outs
        )
        results.append(outs)
    return spec, k_mean, v_scale, payloads, results


def _assert_int8_close(got, want, name):
    diff = (got.int() - want.int()).abs()
    assert diff.max() <= 1, f"{name}: max int8 step {diff.max()}"
    assert (diff == 0).float().mean() > 0.999, f"{name}: too many one-step differences"


def _assert_fp8_close(got, want, name):
    g, w = got.float(), want.float()
    exact = torch.equal(g, w)
    if not exact:
        # one ulp of E4M3 at the value's binade, plus zero-vs-smallest cases
        tol = torch.maximum(w.abs() * 0.125, torch.full_like(w, 2**-9))
        assert ((g - w).abs() <= tol).float().mean() > 0.999, (
            f"{name}: fp8 mismatch beyond one ulp"
        )


@requires_sm120
@pytest.mark.parametrize("layout", ["interleaved", "split"])
@pytest.mark.parametrize(
    "batch,local,heads,world_size,used",
    [
        (1, 256, 16, 8, 2048),
        (2, 128, 8, 4, 500),
        (1, 384, 8, 2, 700),
        (1, 512, 8, 1, 500),
    ],
)
def test_pack_unpack_matches_reference_quantizer(
    layout, batch, local, heads, world_size, used
):
    seqlen = local * world_size
    q, k, v = _global_qkv(batch, seqlen, heads, layout=layout, used=used, seed=1)
    assert can_use_ulysses_lowp_sage(q, k, v, world_size=world_size)
    spec, k_mean, v_scale, _, results = _simulate(
        q, k, v, world_size=world_size, used=used
    )

    # Reference: quantise the whole sequence at once (that is what "global grid" means).
    q_bhsd = q.transpose(1, 2)
    k_bhsd = k.transpose(1, 2)
    v_bhsd = v.transpose(1, 2)
    ref_q8, ref_qs = quantize_sage_q(q_bhsd)
    ref_k8, ref_v8, ref_ks, ref_vs = quantize_sage_kv(k_bhsd, v_bhsd, used_rows=used)
    cut = -(-used // K_BLOCK) * K_BLOCK
    q_width, k_width = ulysses_lowp_scale_widths(cut)
    torch.testing.assert_close(v_scale, ref_vs, rtol=1e-6, atol=0)
    torch.testing.assert_close(
        k_mean.float(),
        (k.float()[:, :used].sum(1) / used).to(q.dtype).float(),
        rtol=0,
        atol=0,
    )

    h = spec.local_heads
    for dest, (q8, k8, v8, qs, ks) in enumerate(results):
        sl = slice(dest * h, (dest + 1) * h)
        _assert_int8_close(q8[:, :, :used], ref_q8[:, sl, :used], f"q dest {dest}")
        _assert_int8_close(k8[:, :, :used], ref_k8[:, sl, :used], f"k dest {dest}")
        rows = sage_v_physical_rows(seqlen, v8.device)[:used]
        _assert_fp8_close(
            v8[:, :, :, rows], ref_v8[:, sl][:, :, :, rows], f"v dest {dest}"
        )
        torch.testing.assert_close(qs, ref_qs[:, sl, :q_width], rtol=1e-6, atol=0)
        torch.testing.assert_close(ks, ref_ks[:, sl, :k_width], rtol=1e-6, atol=0)
        # padding rows come out as zeros in all three operands
        assert not q8[:, :, used:].any() and not k8[:, :, used:].any()
        assert (
            not v8[:, :, :, sage_v_physical_rows(seqlen, v8.device)[used:]]
            .float()
            .any()
        )


@requires_sm120
def test_pack_is_deterministic_and_padding_stays_out_of_the_amax():
    q, k, v = _global_qkv(1, 1024, 8, layout="interleaved", used=1000, seed=2)
    _, _, _, payloads_a, _ = _simulate(q, k, v, world_size=4, used=1000)
    _, _, _, payloads_b, _ = _simulate(q, k, v, world_size=4, used=1000)
    for a, b in zip(payloads_a, payloads_b):
        assert torch.equal(a, b)
    # The last 64-block mixes live and padded rows; its K scale must equal the
    # live-rows-only amax, which the reference computes by construction.
    _, _, _, _, results = _simulate(q, k, v, world_size=4, used=1000)
    ref_k8, _, ref_ks, _ = quantize_sage_kv(
        k.transpose(1, 2), v.transpose(1, 2), used_rows=1000
    )
    k_width = ulysses_lowp_scale_widths(1024)[1]
    torch.testing.assert_close(
        results[3][4], ref_ks[:, 6:8, :k_width], rtol=1e-6, atol=0
    )


@requires_sm120
@pytest.mark.skipif(
    not can_use_sage_block_sparse_attn_sm120(), reason="needs the Cake kernel"
)
def test_unpacked_operands_drive_dense_sage_attention():
    """Pack -> unpack -> Cake (dense) stays within the Sage error band of fp32 attention."""
    batch, local, heads, world_size, used = 1, 256, 16, 8, 2000
    seqlen = local * world_size
    q, k, v = _global_qkv(batch, seqlen, heads, layout="interleaved", used=used, seed=3)
    spec, _, v_scale, _, results = _simulate(q, k, v, world_size=world_size, used=used)
    cut = -(-used // K_BLOCK) * K_BLOCK
    h = spec.local_heads
    for dest, (q8, k8, v8, qs, ks) in enumerate(results):
        sl = slice(dest * h, (dest + 1) * h)
        index, nums = sage_block_sparse_dense_block_index(batch, h, cut, cut, q.device)
        out = torch.zeros(batch, h, seqlen, 128, dtype=torch.bfloat16, device="cuda")
        sage_block_sparse_attn_sm120(
            q8,
            k8,
            v8,
            qs,
            ks,
            v_scale[:, sl].contiguous(),
            index,
            nums,
            out=out,
            seqlen_q=cut,
            seqlen_k=cut,
            softmax_scale=128**-0.5,
        )
        qf = q[:, :used, sl].transpose(1, 2).float()
        kf = k[:, :used, sl].transpose(1, 2).float()
        vf = v[:, :used, sl].transpose(1, 2).float()
        want = torch.softmax(qf @ kf.transpose(-1, -2) * 128**-0.5, dim=-1) @ vf
        got = out[:, :, :used].float()
        rel = (got - want).norm() / want.norm()
        assert rel < 6e-2, f"dest {dest}: rel-L2 {rel:.4f}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))


@requires_sm120
@pytest.mark.parametrize("world_size", [2, 4, 6, 8])
def test_finalize_stats_kernel_matches_the_torch_reduction_bitwise(world_size):
    from sglang.kernels.ops.diffusion import ulysses_lowp_finalize_stats_local

    g = torch.Generator(device="cuda").manual_seed(world_size)
    heads, local_heads, used = 24, 24 // world_size, 5000
    gathered = (
        torch.randn(world_size, 2, 1, heads, 128, generator=g, device="cuda") * 3
    ).abs()
    want_k, want_v = ulysses_lowp_finalize_stats(
        gathered, world_size=world_size, used_sequence=used, dtype=torch.bfloat16
    )
    for rank in (0, world_size - 1):
        k_mean, v_scale, v_local = ulysses_lowp_finalize_stats_local(
            gathered,
            world_size=world_size,
            used_sequence=used,
            rank=rank,
            local_heads=local_heads,
        )
        assert torch.equal(k_mean, want_k)
        assert torch.equal(v_scale, want_v)
        assert torch.equal(
            v_local, want_v[:, rank * local_heads : (rank + 1) * local_heads]
        )


@requires_sm120
def test_stats_kernel_replicates_its_record_and_the_pack_lands_the_own_chunk():
    q, k, v = _global_qkv(1, 2048, 8, layout="interleaved", used=2000, seed=5)
    replicas = torch.empty(4, 2, 1, 8, 128, dtype=torch.float32, device="cuda")
    own = torch.empty(2, 1, 8, 128, dtype=torch.float32, device="cuda")
    record = ulysses_lowp_k_sum_v_amax(k, v, out=replicas, own=own)
    plain = ulysses_lowp_k_sum_v_amax(k, v)
    assert torch.equal(record, plain)
    assert all(torch.equal(replicas[r], plain) for r in range(4))
    assert torch.equal(own, plain)

    world_size, rank = 4, 2
    local = 2048 // world_size
    spec = ulysses_lowp_payload_spec(
        batch=1, local_sequence=local, num_heads=8, world_size=world_size
    )
    k_mean, v_scale = ulysses_lowp_finalize_stats(
        plain.unsqueeze(0).expand(world_size, -1, -1, -1, -1).contiguous(),
        world_size=world_size,
        used_sequence=2000,
        dtype=torch.bfloat16,
    )
    shard = slice(rank * local, (rank + 1) * local)
    want = torch.zeros(spec.payload_shape, dtype=torch.uint8, device="cuda")
    ulysses_lowp_quant_pack(
        q[:, shard],
        k[:, shard],
        v[:, shard],
        k_mean,
        v_scale,
        rank=rank,
        world_size=world_size,
        used_sequence=2000,
        out=want,
    )
    got = torch.zeros(spec.payload_shape, dtype=torch.uint8, device="cuda")
    own_chunk = torch.zeros(spec.chunk_bytes, dtype=torch.uint8, device="cuda")
    ulysses_lowp_quant_pack(
        q[:, shard],
        k[:, shard],
        v[:, shard],
        k_mean,
        v_scale,
        rank=rank,
        world_size=world_size,
        used_sequence=2000,
        out=got,
        own_out=own_chunk,
    )
    assert torch.equal(own_chunk, want[rank])
    assert torch.all(got[rank] == 0)  # the own row of `out` is left untouched
    for peer in range(world_size):
        if peer != rank:
            assert torch.equal(got[peer], want[peer])
