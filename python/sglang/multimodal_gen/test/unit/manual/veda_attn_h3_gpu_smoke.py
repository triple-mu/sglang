# SPDX-License-Identifier: Apache-2.0
"""GPU smoke test of the Veda backend on the real Miowtion kernels.

Needs one CUDA device Miowtion supports (SM8x / SM120 via its FA4 patch,
SM90 / SM100 natively), `miowtion` at the pinned commit and
`flash-attn-4==4.0.0b32`. Builds a random fp8 bundle for a small 16:9
geometry, pretends to be Ulysses rank 1 of 2 on TP rank 0 (local heads 4..7
of 8) and checks:

  (a) the backend output equals upstream's full-head SparseStudent [:, 4:8]
      bit for bit (same tiles, same predictor rows, same kernel);
  (b) with keep ratio 1.0 the real rows match sglang's SDPA within tolerance;
  (c) the sparse path ran (call counters and the "Veda attention active" log);
  (d) padding rows [used, seq_len) come back as zeros;
  (e) veda_dense_first_n_steps routes to the SDPA path bit for bit.

Usage:
    PYTHONPATH=python python python/sglang/multimodal_gen/test/unit/manual/veda_attn_h3_gpu_smoke.py
"""

from __future__ import annotations

import logging
import tempfile
from types import SimpleNamespace

import torch

from sglang.multimodal_gen.runtime.distributed import parallel_state
from sglang.multimodal_gen.runtime.layers.attention.backends import (
    veda_attn_h3,
    veda_runtime,
)
from sglang.multimodal_gen.runtime.layers.attention.backends.sdpa import SDPAImpl
from sglang.multimodal_gen.runtime.layers.attention.backends.veda_attn_h3 import (
    VedaAttentionImpl,
    VedaAttentionMetadata,
)
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.server_args import server_args as server_args_module

HEAD_DIM = 128
LOCAL = veda_runtime.HeadRange(4, 8)


def _configure(config: dict) -> None:
    server_args_module._global_server_args = SimpleNamespace(
        attention_backend_config=config
    )
    veda_attn_h3._VedaRuntime._instances.clear()


def _impl() -> VedaAttentionImpl:
    return VedaAttentionImpl(
        num_heads=8,
        head_size=HEAD_DIM,
        softmax_scale=HEAD_DIM**-0.5,
        prefix="blocks.1.attn",
    )


def main() -> None:
    from miowtion.h3 import geometry as h3_geometry
    from miowtion.h3 import layout as h3_layout
    from miowtion.veda import attention as veda_attention
    from miowtion.veda import bundle as veda_bundle
    from miowtion.veda import mask as veda_mask
    from miowtion.veda import plan as veda_plan
    from miowtion.veda import predictor as veda_predictor
    from miowtion.veda import tiling

    logging.basicConfig(level=logging.INFO)
    device = torch.device("cuda")
    ok, reason = veda_runtime.fa4_status(device)
    print(
        f"fa4 block sparsity on {torch.cuda.get_device_name(device)}: {ok} {reason or ''}"
    )
    assert ok, "Miowtion's FA4 block sparsity is unavailable on this device"
    veda_runtime.check_contract()

    geometry = h3_geometry.Geometry("16:9", 512, 256, 39, 12, 16, 32, 20)
    layout = h3_layout.pack(torch.ones(64, dtype=torch.long), geometry)
    plan = veda_plan.TilePlan(
        geometry.name,
        geometry.video_grid,
        [tiling.TileShape(4, 4, 8), tiling.TileShape(2, 8, 8)],
        [[0, 1, 1, 0, 1, 0, 0, 1], [1, 1, 0, 0, 0, 0, 1, 1]],
    )
    torch.manual_seed(0)
    predictor = veda_predictor.TileScorePredictor(2, 8, HEAD_DIM)
    with torch.no_grad():
        for layer in predictor.layers:
            layer.proj_q.normal_(std=0.1)
            layer.proj_k.normal_(std=0.1)
    tmp = tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False)
    veda_bundle.save(
        tmp.name,
        predictor.state_dict(),
        veda_plan.PlanTable([plan]),
        num_layers=2,
        num_heads=8,
        head_dim=HEAD_DIM,
        keep_ratio=0.2,
        source="smoke",
        source_weights="live",
        step=0,
        dtype=torch.float8_e4m3fn,
    )

    parallel_state.get_tp_rank = lambda: 0
    parallel_state.get_ulysses_parallel_rank = lambda: 1

    gen = torch.Generator(device=device).manual_seed(1)
    q8, k8, v8 = (
        torch.randn(
            layout.seq_len,
            8,
            HEAD_DIM,
            device=device,
            dtype=torch.bfloat16,
            generator=gen,
        )
        for _ in range(3)
    )
    sl = slice(LOCAL.start, LOCAL.stop)
    q, k, v = q8[:, sl], k8[:, sl], v8[:, sl]  # strided views, unit stride on head_dim
    used, seq_len = layout.used, layout.seq_len
    cu = torch.tensor([0, used, seq_len], dtype=torch.int32, device=device)
    metadata = VedaAttentionMetadata(
        current_timestep=0,
        video_start=layout.target.start,
        grid=layout.target.grid,
        used=used,
        seq_len=seq_len,
        num_steps=8,
    )

    # (a) bitwise against the full-head upstream student (fp8-dequantised weights).
    _configure({"veda_bundle": tmp.name})
    impl = _impl()
    with set_forward_context(current_timestep=0, attn_metadata=metadata):
        out = impl.forward_varlen(
            q, k, v, cu_seqlens=cu, max_seqlen=used, cu_seqlens_host=(0, used, seq_len)
        )
    full_bundle = veda_bundle.load(tmp.name, device)
    config = veda_attention.VedaConfig(target_budget=veda_mask.Budget(ratio=0.2))
    full = veda_attention.SparseStudent(
        veda_attention.ClipTiling(layout, config, device),
        full_bundle.plans.plans[geometry.name],
        full_bundle.predictor,
    )
    with torch.no_grad():
        reference = full(q8, k8, v8, 1)[:, sl]
    torch.cuda.synchronize()
    print(
        f"(a) torch.equal(backend, full-head student[:, 4:8]) = {torch.equal(out, reference)}"
    )
    assert torch.equal(out, reference)
    # (c) sparse path ran
    runtime = veda_attn_h3._VedaRuntime._instances[q.device]
    print(
        f"(c) calls={impl.calls} announced={runtime._announced_sparse} heads={runtime.local.heads}"
    )
    assert impl.calls == {"sparse": 1, "dense": 0} and runtime._announced_sparse
    # (d) padding rows are zero
    pad = out[used:]
    print(
        f"(d) padding rows {seq_len - used}, max |out| = {pad.abs().max().item() if pad.numel() else 0.0}"
    )
    assert pad.numel() == 0 or torch.equal(pad, torch.zeros_like(pad))

    # (b) keep-all vs SDPA
    _configure({"veda_bundle": tmp.name, "veda_keep_ratio": 1.0})
    impl_all = _impl()
    with set_forward_context(current_timestep=0, attn_metadata=metadata):
        dense_all = impl_all.forward_varlen(
            q, k, v, cu_seqlens=cu, max_seqlen=used, cu_seqlens_host=(0, used, seq_len)
        )
    sdpa = SDPAImpl(
        num_heads=8, head_size=HEAD_DIM, causal=False, softmax_scale=HEAD_DIM**-0.5
    )
    sdpa_out = sdpa.forward_varlen(
        q, k, v, cu_seqlens=cu, max_seqlen=used, cu_seqlens_host=(0, used, seq_len)
    )
    # Real rows only: SDPA attends the padding segment to itself, Veda zeroes it.
    err = (dense_all[:used].float() - sdpa_out[:used].float()).abs().max().item()
    print(
        f"(b) keep ratio 1.0 vs SDPA on real rows: max abs err {err:.3e}, max |sdpa| {sdpa_out[:used].abs().max().item():.3e}"
    )
    assert err <= 2e-3, err

    # (e) dense-first-N routes to SDPA bit for bit
    _configure({"veda_bundle": tmp.name, "veda_dense_first_n_steps": 8})
    impl_dense = _impl()
    with set_forward_context(current_timestep=0, attn_metadata=metadata):
        routed = impl_dense.forward_varlen(
            q, k, v, cu_seqlens=cu, max_seqlen=used, cu_seqlens_host=(0, used, seq_len)
        )
    print(
        f"(e) dense-first-8 equals SDPA: {torch.equal(routed, sdpa_out)}, calls={impl_dense.calls}"
    )
    assert torch.equal(routed, sdpa_out) and impl_dense.calls == {
        "sparse": 0,
        "dense": 1,
    }
    print("SMOKE OK")


if __name__ == "__main__":
    main()
