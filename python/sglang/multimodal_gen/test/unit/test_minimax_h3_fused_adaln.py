# SPDX-License-Identifier: Apache-2.0
"""Fused adaLN chain (A1) for the MiniMax-H3 DiT.

Kernel level: the bitexact (aten-order) fused RMSNorm + indexed scale/shift
(Plan A) and gate+residual+RMSNorm+scale/shift (Plan B) kernels must be
bitwise equal to the eager chain (nn.RMSNorm -> indexed_scale_shift_bf16_,
with indexed_gate_bf16_ in front for Plan B) at production shapes and on
adversarial values. The opt-out merged-w_eff kernels must stay near-lossless
against the same chain, and every Plan B residual write-back must stay
bit-exact with indexed_gate_bf16_.

Module level: the default (bitexact) fused block chain (forward_fused +
deferred trailing gate) must be bitwise equal to the eager block forward, the
opt-out near-lossless chain must match within tolerance, and with
MINIMAX_H3_FUSED_ADALN=0 the model must not build fused parameters at all,
leaving the untouched eager path authoritative.
"""

import pytest
import torch
import torch.nn as nn

from sglang.kernels.ops.diffusion import (
    can_use_rmsnorm_indexed_scale_shift_bitexact_cuda,
    gate_residual_rmsnorm_indexed_scale_shift_,
    gate_residual_rmsnorm_indexed_scale_shift_bitexact_,
    indexed_gate_bf16_,
    indexed_scale_shift_bf16_,
    rmsnorm_indexed_scale_shift,
    rmsnorm_indexed_scale_shift_bitexact,
)

# Backend-specific imports: the C++-vs-Triton contract tests below exercise
# each backend directly (this file is allowlisted in test_import_surface.py).
from sglang.kernels.ops.diffusion.modulate import (
    indexed_modulation_jit as _indexed_modulation_jit,
)
from sglang.kernels.ops.diffusion.modulate import (
    indexed_modulation_triton as _indexed_modulation_triton,
)
from sglang.kernels.ops.diffusion.norm import (
    rmsnorm_indexed_modulate_jit as _rmsnorm_indexed_modulate_jit,
)
from sglang.kernels.ops.diffusion.norm import (
    rmsnorm_indexed_modulate_triton as _rmsnorm_indexed_modulate_triton,
)
from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import (
    MiniMaxH3DiTBlock,
    MiniMaxH3DiTModel,
    _modulate_gate,
)

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)

_HIDDEN = 5376
_EPS = 1e-5

# One bf16 ulp of a value v in [2^e, 2^(e+1)) is 2^(e-8), i.e. between 2^-9
# and 2^-8 of v. The eager chain rounds to bf16 after the norm, after
# (1 + scale), after the product and after the shift add; the fused kernel
# rounds once, so the two may differ by a few ulp *of the pre-shift product*
# (out = prod + shift cancels, so an out-relative bound is not meaningful).
# Measured on H200 at [20992, 5376]: <= 7.9 with p99 <= 3.8 for Plan A, and
# fused lands ~2x closer to the fp64 truth than eager (max err 3.1e-2 vs
# 8.6e-2). Asserted with ~2x headroom.
_MAX_SCALED_ULP = 16.0


def _packed_indices(rows: int, groups: int, seed: int) -> torch.Tensor:
    """Contiguous per-(timestep, modality) runs, like the packed layout."""
    generator = torch.Generator().manual_seed(seed)
    lengths = torch.rand(groups, generator=generator) + 0.05
    lengths = (lengths / lengths.sum() * rows).long()
    lengths[-1] = rows - lengths[:-1].sum()
    values = torch.randperm(groups, generator=generator)
    return torch.repeat_interleave(values, lengths).cuda()


def _report_and_assert_close(
    fused: torch.Tensor,
    eager: torch.Tensor,
    magnitude: torch.Tensor,
    *,
    label: str,
    max_scaled_ulp: float = _MAX_SCALED_ULP,
) -> None:
    """Assert |fused - eager| elementwise within a few bf16 ulp of the local
    operand magnitude (|product| + |shift|), and report the raw stats."""
    abs_diff = (fused.float() - eager.float()).abs()
    scaled_ulp = abs_diff / (2**-9 * (magnitude.float() + 2**-6))
    rel = abs_diff / eager.float().abs().clamp_min(1.0)
    print(
        f"\n[{label}] max_abs={abs_diff.max().item():.3e} "
        f"max_rel@|e|>=1={rel.max().item():.3e} "
        f"scaled_ulp max={scaled_ulp.max().item():.2f} "
        f"p99={torch.quantile(scaled_ulp.flatten()[:16_000_000], 0.99).item():.2f} "
        f"mismatch_frac={(fused != eager).float().mean().item():.4f}"
    )
    assert (
        scaled_ulp.max().item() <= max_scaled_ulp
    ), f"{label}: {scaled_ulp.max().item():.2f} scaled ulp"


def _assert_fused_at_least_as_accurate(
    fused: torch.Tensor,
    eager: torch.Tensor,
    reference: torch.Tensor,
    *,
    label: str,
) -> None:
    """Both paths approximate the same fp64 chain; the fused kernel rounds
    less and must not sit farther from the truth than the eager chain."""
    err_fused = (fused.double() - reference).abs()
    err_eager = (eager.double() - reference).abs()
    print(
        f"[{label}] err_vs_fp64: fused(max={err_fused.max().item():.3e}, "
        f"mean={err_fused.mean().item():.3e}) eager(max="
        f"{err_eager.max().item():.3e}, mean={err_eager.mean().item():.3e})"
    )
    assert err_fused.max() <= err_eager.max() * 1.1, label
    assert err_fused.mean() <= err_eager.mean() * 1.1, label


def _eager_norm(weight: torch.Tensor) -> nn.RMSNorm:
    norm = nn.RMSNorm(_HIDDEN, eps=_EPS, dtype=torch.bfloat16, device="cuda")
    with torch.no_grad():
        norm.weight.copy_(weight)
    return norm


def _random_case(rows: int, groups: int, seed: int):
    generator = torch.Generator(device="cuda").manual_seed(seed)

    def randn(*shape, scale=1.0):
        return (torch.randn(shape, generator=generator, device="cuda") * scale).to(
            torch.bfloat16
        )

    x = randn(rows, _HIDDEN)
    weight = (1.0 + 0.2 * randn(_HIDDEN).float()).to(torch.bfloat16)
    scale = randn(groups, _HIDDEN, scale=0.3)
    shift = randn(groups, _HIDDEN, scale=0.5)
    gate = randn(groups, _HIDDEN, scale=0.3)
    update = randn(rows, _HIDDEN)
    indices = _packed_indices(rows, groups, seed)
    w_eff = weight.float() * (1.0 + scale.float())
    return x, weight, scale, shift, gate, update, indices, w_eff


def _norm_modulate_fp64(
    y: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """fp64 ground truth of the norm+modulate chain and the per-element
    operand magnitude |product| + |shift| used to scale the ulp bound."""
    y64 = y.double()
    rstd = torch.rsqrt(y64.pow(2).mean(-1, keepdim=True) + _EPS)
    prod = (
        y64 * rstd * weight.double() * (1.0 + scale.double().index_select(0, indices))
    )
    shift_rows = shift.double().index_select(0, indices)
    return prod + shift_rows, prod.abs() + shift_rows.abs()


@requires_cuda
@pytest.mark.parametrize(("rows", "groups"), [(20992, 12), (20992, 6), (4096, 2)])
def test_plan_a_matches_eager_chain(rows: int, groups: int):
    x, weight, scale, shift, _, _, indices, w_eff = _random_case(rows, groups, 0)
    norm = _eager_norm(weight)

    with torch.no_grad():
        eager = indexed_scale_shift_bf16_(norm(x), shift, scale, indices)
    fused = rmsnorm_indexed_scale_shift(x, w_eff, shift, indices, eps=_EPS)
    fused_again = rmsnorm_indexed_scale_shift(x, w_eff, shift, indices, eps=_EPS)

    assert fused.dtype is torch.bfloat16 and fused.shape == x.shape
    assert torch.equal(fused, fused_again), "fused kernel must be deterministic"
    label = f"plan_a[{rows}x{groups}]"
    reference, magnitude = _norm_modulate_fp64(x, weight, scale, shift, indices)
    _report_and_assert_close(fused, eager, magnitude, label=label)
    _assert_fused_at_least_as_accurate(fused, eager, reference, label=label)


@requires_cuda
@pytest.mark.parametrize(("rows", "groups"), [(20992, 12), (20992, 6), (4096, 2)])
def test_plan_b_matches_eager_chain(rows: int, groups: int):
    x, weight, scale, shift, gate, update, indices, w_eff = _random_case(
        rows, groups, 1
    )
    norm = _eager_norm(weight)

    eager_res = indexed_gate_bf16_(x.clone(), gate, update, indices)
    with torch.no_grad():
        eager_out = indexed_scale_shift_bf16_(norm(eager_res), shift, scale, indices)

    fused_res = x.clone()
    fused_out, returned = gate_residual_rmsnorm_indexed_scale_shift_(
        fused_res, update, gate, w_eff, shift, indices, eps=_EPS
    )

    assert returned.data_ptr() == fused_res.data_ptr(), "residual is in-place"
    assert torch.equal(
        fused_res, eager_res
    ), "Plan B residual write-back must be bit-exact vs indexed_gate_bf16_"
    label = f"plan_b[{rows}x{groups}]"
    # both paths norm the identical bf16 residual, so anchor fp64 on it
    reference, magnitude = _norm_modulate_fp64(eager_res, weight, scale, shift, indices)
    _report_and_assert_close(fused_out, eager_out, magnitude, label=label)
    _assert_fused_at_least_as_accurate(fused_out, eager_out, reference, label=label)


def _skip_unless_bitexact_backend(x, weight, scale, shift, indices) -> None:
    if not can_use_rmsnorm_indexed_scale_shift_bitexact_cuda(
        x, weight, scale, shift, indices
    ):
        pytest.skip("bitexact JIT C++ backend unavailable (needs SM90+ CUDA)")


def _eager_plan_a(x, norm, scale, shift, indices):
    with torch.no_grad():
        return indexed_scale_shift_bf16_(norm(x), shift, scale, indices)


def _adversarial_case(groups: int):
    """Rows and modulation built from reduction-order-hostile values: zero
    rows (rms == sqrt(eps)), fp32-overflowing squares, bf16 subnormals, and
    alternating huge/tiny magnitudes that make the summation order visible."""
    rows = 2048
    generator = torch.Generator(device="cuda").manual_seed(99)
    x = (torch.randn(rows, _HIDDEN, generator=generator, device="cuda") * 4.0).to(
        torch.bfloat16
    )
    x[0].zero_()
    x[1].fill_(3.0e38)  # sum of squares overflows fp32 -> rstd 0 on both paths
    x[2].fill_(1.0e-40)  # bf16 subnormal
    x[3, ::2] = 1.0e18
    x[3, 1::2] = 1.0e-18
    x[4, :7] = 6.0e17  # partial-row extremes across thread boundaries
    weight = (1.0 + 2.0 * torch.randn(_HIDDEN, generator=generator, device="cuda")).to(
        torch.bfloat16
    )
    weight[::17] = 0.0
    scale = (torch.randn(groups, _HIDDEN, generator=generator, device="cuda") * 8.0).to(
        torch.bfloat16
    )
    scale[0, ::5] = -1.0  # (1 + scale) rounds to exact zero
    shift = (
        torch.randn(groups, _HIDDEN, generator=generator, device="cuda") * 1.0e4
    ).to(torch.bfloat16)
    gate = (torch.randn(groups, _HIDDEN, generator=generator, device="cuda") * 16.0).to(
        torch.bfloat16
    )
    update = (torch.randn(rows, _HIDDEN, generator=generator, device="cuda") * 4.0).to(
        torch.bfloat16
    )
    update[5].fill_(2.0e19)
    indices = _packed_indices(rows, groups, 99)
    return x, weight, scale, shift, gate, update, indices


@requires_cuda
@pytest.mark.parametrize("index_dtype", [torch.int64, torch.int32])
@pytest.mark.parametrize(("rows", "groups"), [(20992, 12), (20992, 6), (4096, 2)])
def test_plan_a_bitexact_torch_equal(rows: int, groups: int, index_dtype: torch.dtype):
    x, weight, scale, shift, _, _, indices, _ = _random_case(rows, groups, 31)
    indices = indices.to(index_dtype)
    _skip_unless_bitexact_backend(x, weight, scale, shift, indices)
    eager = _eager_plan_a(x, _eager_norm(weight), scale, shift, indices)
    fused = rmsnorm_indexed_scale_shift_bitexact(
        x, weight, scale, shift, indices, eps=_EPS
    )
    assert fused.dtype is torch.bfloat16 and fused.shape == x.shape
    assert torch.equal(fused, eager), "Plan A bitexact must equal the eager chain"


@requires_cuda
@pytest.mark.parametrize("index_dtype", [torch.int64, torch.int32])
@pytest.mark.parametrize(("rows", "groups"), [(20992, 12), (20992, 6), (4096, 2)])
def test_plan_b_bitexact_torch_equal(rows: int, groups: int, index_dtype: torch.dtype):
    x, weight, scale, shift, gate, update, indices, _ = _random_case(rows, groups, 32)
    indices = indices.to(index_dtype)
    _skip_unless_bitexact_backend(x, weight, scale, shift, indices)
    eager_res = indexed_gate_bf16_(x.clone(), gate, update, indices)
    eager_out = _eager_plan_a(eager_res, _eager_norm(weight), scale, shift, indices)

    fused_res = x.clone()
    fused_out, returned = gate_residual_rmsnorm_indexed_scale_shift_bitexact_(
        fused_res, update, gate, weight, scale, shift, indices, eps=_EPS
    )
    assert returned.data_ptr() == fused_res.data_ptr(), "residual is in-place"
    assert torch.equal(fused_res, eager_res), "Plan B residual must be bitwise"
    assert torch.equal(fused_out, eager_out), "Plan B bitexact must equal eager"


@requires_cuda
def test_bitexact_adversarial_values():
    groups = 5
    x, weight, scale, shift, gate, update, indices = _adversarial_case(groups)
    _skip_unless_bitexact_backend(x, weight, scale, shift, indices)
    norm = _eager_norm(weight)

    eager_a = _eager_plan_a(x, norm, scale, shift, indices)
    fused_a = rmsnorm_indexed_scale_shift_bitexact(
        x, weight, scale, shift, indices, eps=_EPS
    )
    assert torch.equal(fused_a, eager_a), "Plan A bitexact on adversarial values"

    eager_res = indexed_gate_bf16_(x.clone(), gate, update, indices)
    eager_b = _eager_plan_a(eager_res, norm, scale, shift, indices)
    fused_res = x.clone()
    fused_b, _ = gate_residual_rmsnorm_indexed_scale_shift_bitexact_(
        fused_res, update, gate, weight, scale, shift, indices, eps=_EPS
    )
    assert torch.equal(fused_res, eager_res), "Plan B residual on adversarial values"
    assert torch.equal(fused_b, eager_b), "Plan B bitexact on adversarial values"


@requires_cuda
def test_bitexact_deterministic_across_runs():
    x, weight, scale, shift, _, _, indices, _ = _random_case(4096, 6, 33)
    _skip_unless_bitexact_backend(x, weight, scale, shift, indices)
    first = rmsnorm_indexed_scale_shift_bitexact(
        x, weight, scale, shift, indices, eps=_EPS
    )
    second = rmsnorm_indexed_scale_shift_bitexact(
        x, weight, scale, shift, indices, eps=_EPS
    )
    assert torch.equal(first, second)


@requires_cuda
def test_fused_kernels_handle_zero_rows():
    _, _, scale, shift, gate, _, _, w_eff = _random_case(4, 2, 2)
    x = torch.empty(0, _HIDDEN, device="cuda", dtype=torch.bfloat16)
    indices = torch.empty(0, device="cuda", dtype=torch.long)
    out = rmsnorm_indexed_scale_shift(x, w_eff, shift, indices, eps=_EPS)
    assert out.shape == x.shape
    out, res = gate_residual_rmsnorm_indexed_scale_shift_(
        x.clone(), x.clone(), gate, w_eff, shift, indices, eps=_EPS
    )
    assert out.shape == res.shape == x.shape


class _KwargLinear(nn.Module):
    """Deterministic bf16 mixer standing in for attention / MLP."""

    def __init__(self, seed: int):
        super().__init__()
        generator = torch.Generator(device="cuda").manual_seed(seed)
        self.weight = nn.Parameter(
            (
                torch.randn(_HIDDEN, _HIDDEN, generator=generator, device="cuda")
                * _HIDDEN**-0.5
            ).to(torch.bfloat16),
            requires_grad=False,
        )

    def forward(self, x: torch.Tensor, **_kwargs) -> torch.Tensor:
        return x @ self.weight


def _stub_block(seed: int) -> MiniMaxH3DiTBlock:
    block = MiniMaxH3DiTBlock.__new__(MiniMaxH3DiTBlock)
    nn.Module.__init__(block)
    generator = torch.Generator(device="cuda").manual_seed(seed)
    block.norm1 = nn.RMSNorm(_HIDDEN, eps=_EPS, dtype=torch.bfloat16, device="cuda")
    block.norm2 = nn.RMSNorm(_HIDDEN, eps=_EPS, dtype=torch.bfloat16, device="cuda")
    with torch.no_grad():
        for norm in (block.norm1, block.norm2):
            norm.weight.copy_(
                1.0 + 0.2 * torch.randn(_HIDDEN, generator=generator, device="cuda")
            )
    block.attn = _KwargLinear(seed + 100)
    block.mlp = _KwargLinear(seed + 200)
    block.adaln_proj = None
    block.preserve_input_for_cache_dit = False
    return block


def _block_adaln_params(num_blocks: int, groups: int, seed: int):
    generator = torch.Generator(device="cuda").manual_seed(seed)

    def rows(scale):
        return (
            torch.randn(groups, _HIDDEN, generator=generator, device="cuda") * scale
        ).to(torch.bfloat16)

    return tuple(
        # (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)
        (rows(0.5), rows(0.3), rows(0.2), rows(0.5), rows(0.3), rows(0.2))
        for _ in range(num_blocks)
    )


def _model_shell(blocks) -> MiniMaxH3DiTModel:
    model = MiniMaxH3DiTModel.__new__(MiniMaxH3DiTModel)
    nn.Module.__init__(model)
    model.blocks = nn.ModuleList(blocks)
    model._fused_adaln_gamma_key = None
    model._fused_adaln_gamma1 = None
    model._fused_adaln_gamma2 = None
    model._fused_adaln_mode_logged = False
    return model


def _block_kwargs(rows: int):
    return dict(
        rope_cache=None,
        cu_seqlens=torch.tensor([0, rows], device="cuda", dtype=torch.int32),
        max_seqlen=rows,
    )


def _block_chain_fp64(
    x: torch.Tensor, blocks, params, indices: torch.Tensor
) -> torch.Tensor:
    """fp64 twin of the stub block chain (same math, no bf16 rounding)."""
    hidden = x.double()
    for block, block_params in zip(blocks, params):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            param.double().index_select(0, indices) for param in block_params
        )
        rstd = torch.rsqrt(hidden.pow(2).mean(-1, keepdim=True) + _EPS)
        h = hidden * rstd * block.norm1.weight.double() * (1 + scale_msa)
        h = (h + shift_msa) @ block.attn.weight.double()
        hidden = hidden + gate_msa * h
        rstd = torch.rsqrt(hidden.pow(2).mean(-1, keepdim=True) + _EPS)
        h = hidden * rstd * block.norm2.weight.double() * (1 + scale_mlp)
        h = (h + shift_mlp) @ block.mlp.weight.double()
        hidden = hidden + gate_mlp * h
    return hidden


def _run_block_chains(blocks, params, indices, x, model, rows):
    """Run the eager block chain and the fused chain (with its deferred
    trailing gate) on clones of ``x``; returns ``(eager, fused)``."""
    eager = x.clone()
    for block, block_params in zip(blocks, params):
        eager = block(
            eager,
            adaln_input=None,
            combined_indices=indices,
            adaln_params=block_params,
            **_block_kwargs(rows),
        )

    fused_params = model._fused_adaln_block_params(params, x, indices)
    assert fused_params is not None
    fused = x.clone()
    pending = None
    for block, block_fused in zip(blocks, fused_params):
        fused, pending = block.forward_fused(
            fused,
            fused_adaln=block_fused,
            pending_gate=pending,
            combined_indices=indices,
            **_block_kwargs(rows),
        )
    last_gate, last_update = pending
    fused = _modulate_gate(fused, last_gate, last_update, indices, dtype=torch.bfloat16)
    return eager, fused


@requires_cuda
def test_fused_block_chain_bitexact_matches_eager_bitwise(monkeypatch):
    """Default mode: the whole fused block loop is bitwise vs eager forward."""
    monkeypatch.setenv("MINIMAX_H3_FUSED_ADALN", "1")
    monkeypatch.setenv("MINIMAX_H3_FUSED_ADALN_BITEXACT", "1")
    rows, groups, num_blocks = 4096, 12, 3
    blocks = [_stub_block(seed) for seed in range(num_blocks)]
    params = _block_adaln_params(num_blocks, groups, 42)
    model = _model_shell(blocks)
    indices = _packed_indices(rows, groups, 3)
    generator = torch.Generator(device="cuda").manual_seed(7)
    x = torch.randn(rows, _HIDDEN, generator=generator, device="cuda").to(
        torch.bfloat16
    )
    _skip_unless_bitexact_backend(
        x, blocks[0].norm1.weight, params[0][1], params[0][0], indices
    )
    eager, fused = _run_block_chains(blocks, params, indices, x, model, rows)
    assert torch.equal(fused, eager), "bitexact fused chain must be bitwise vs eager"


@requires_cuda
def test_fused_block_chain_matches_eager_forward(monkeypatch):
    monkeypatch.setenv("MINIMAX_H3_FUSED_ADALN", "1")
    # Opt-out mode: the merged-w_eff kernels are near-lossless, not bitwise.
    monkeypatch.setenv("MINIMAX_H3_FUSED_ADALN_BITEXACT", "0")
    rows, groups, num_blocks = 4096, 12, 3
    blocks = [_stub_block(seed) for seed in range(num_blocks)]
    params = _block_adaln_params(num_blocks, groups, 42)
    model = _model_shell(blocks)
    indices = _packed_indices(rows, groups, 3)
    generator = torch.Generator(device="cuda").manual_seed(7)
    x = torch.randn(rows, _HIDDEN, generator=generator, device="cuda").to(
        torch.bfloat16
    )

    eager, fused = _run_block_chains(blocks, params, indices, x, model, rows)

    label = f"block_chain[{num_blocks}]"
    reference = _block_chain_fp64(x, blocks, params, indices)
    # Per-block rounding deltas propagate through the stub GEMMs, so the
    # chain is judged against the fp64 twin: the fused chain must track the
    # truth as well as the eager chain does (1.25x covers drift randomness;
    # measured ~2x in the fused path's favor at 3 blocks).
    err_fused = (fused.double() - reference).abs()
    err_eager = (eager.double() - reference).abs()
    diff = (fused.float() - eager.float()).abs()
    print(
        f"\n[{label}] fused_vs_eager max_abs={diff.max().item():.3e} "
        f"out_absmax={eager.float().abs().max().item():.2f} | err_vs_fp64: "
        f"fused(max={err_fused.max().item():.3e}, mean={err_fused.mean().item():.3e}) "
        f"eager(max={err_eager.max().item():.3e}, mean={err_eager.mean().item():.3e})"
    )
    assert err_fused.max() <= err_eager.max() * 1.25, label
    assert err_fused.mean() <= err_eager.mean() * 1.25, label


@requires_cuda
def test_fused_adaln_param_merge_and_gating(monkeypatch):
    rows, groups = 64, 6
    blocks = [_stub_block(seed) for seed in (10, 11)]
    params = _block_adaln_params(2, groups, 5)
    model = _model_shell(blocks)
    hidden = torch.zeros(rows, _HIDDEN, device="cuda", dtype=torch.bfloat16)
    indices = _packed_indices(rows, groups, 4)

    monkeypatch.setenv("MINIMAX_H3_FUSED_ADALN", "0")
    assert model._fused_adaln_block_params(params, hidden, indices) is None

    monkeypatch.setenv("MINIMAX_H3_FUSED_ADALN", "1")
    monkeypatch.setenv("MINIMAX_H3_FUSED_ADALN_BITEXACT", "0")
    fused = model._fused_adaln_block_params(params, hidden, indices)
    assert fused is not None and len(fused) == 2
    for index, (block, block_params) in enumerate(zip(blocks, params)):
        w_msa, shift_msa, gate_msa, w_mlp, shift_mlp, gate_mlp = fused[index]
        assert shift_msa is block_params[0] and gate_msa is block_params[2]
        assert shift_mlp is block_params[3] and gate_mlp is block_params[5]
        for w, norm, scale in (
            (w_msa, block.norm1, block_params[1]),
            (w_mlp, block.norm2, block_params[4]),
        ):
            assert w.dtype is torch.float32
            expected = norm.weight.float() * (1.0 + scale.float())
            torch.testing.assert_close(w, expected, rtol=0, atol=0)

    # gamma stack cache is reused while weights are unchanged, rebuilt on edit
    key = model._fused_adaln_gamma_key
    model._fused_adaln_block_params(params, hidden, indices)
    assert model._fused_adaln_gamma_key == key
    with torch.no_grad():
        blocks[0].norm1.weight.mul_(2.0)
    model._fused_adaln_block_params(params, hidden, indices)
    assert model._fused_adaln_gamma_key != key

    # default bitexact mode: (gamma, scale) pairs, no premerge, raw rows
    monkeypatch.setenv("MINIMAX_H3_FUSED_ADALN_BITEXACT", "1")
    if can_use_rmsnorm_indexed_scale_shift_bitexact_cuda(
        hidden, blocks[0].norm1.weight, params[0][1], params[0][0], indices
    ):
        fused = model._fused_adaln_block_params(params, hidden, indices)
        assert fused is not None and len(fused) == 2
        for index, (block, block_params) in enumerate(zip(blocks, params)):
            w_msa, shift_msa, gate_msa, w_mlp, shift_mlp, gate_mlp = fused[index]
            assert isinstance(w_msa, tuple) and isinstance(w_mlp, tuple)
            assert w_msa[0] is block.norm1.weight and w_msa[1] is block_params[1]
            assert w_mlp[0] is block.norm2.weight and w_mlp[1] is block_params[4]
            assert shift_msa is block_params[0] and gate_msa is block_params[2]
            assert shift_mlp is block_params[3] and gate_mlp is block_params[5]

    # the fused loop must not engage on non-bf16 params, non-contiguous or
    # non-CUDA hidden states, or blocks held by Cache-DiT
    for bitexact in ("0", "1"):
        monkeypatch.setenv("MINIMAX_H3_FUSED_ADALN_BITEXACT", bitexact)
        fp32_params = tuple(tuple(p.float() for p in block) for block in params)
        assert model._fused_adaln_block_params(fp32_params, hidden, indices) is None
        assert model._fused_adaln_block_params(params, hidden.t(), indices) is None
        assert model._fused_adaln_block_params(params, hidden.cpu(), indices) is None
        blocks[0].preserve_input_for_cache_dit = True
        assert model._fused_adaln_block_params(params, hidden, indices) is None
        blocks[0].preserve_input_for_cache_dit = False


# ---------------------------------------------------------------------------
# C++ JIT backend vs the Triton fallback, and the dispatch-order gates.
# The public symbols now dispatch C++ -> Triton; these tests pin the two
# backends against each other directly.
# ---------------------------------------------------------------------------


def _skip_unless_cpp_backend(x, w_eff, shift, indices) -> None:
    if not _rmsnorm_indexed_modulate_jit.can_use_rmsnorm_indexed_scale_shift_cuda(
        x, w_eff, shift, indices
    ):
        pytest.skip("JIT C++ backend unavailable (needs SM90+ CUDA)")


@requires_cuda
@pytest.mark.parametrize("index_dtype", [torch.int64, torch.int32])
@pytest.mark.parametrize(("rows", "groups"), [(20992, 9), (4096, 3)])
def test_cpp_indexed_scale_shift_bitwise_vs_triton(
    rows: int, groups: int, index_dtype: torch.dtype
):
    """The C++ port replicates the Triton bf16 rounding chain bitwise."""
    x, _, scale, shift, _, _, indices, _ = _random_case(rows, groups, 17)
    indices = indices.to(index_dtype)
    x_cpp, x_tri = x.clone(), x.clone()
    if not _indexed_modulation_jit.can_use_indexed_scale_shift_cuda(
        x_cpp, shift, scale, indices
    ):
        pytest.skip("JIT C++ backend unavailable (needs SM90+ CUDA)")
    _indexed_modulation_jit.indexed_scale_shift_bf16_cuda(x_cpp, shift, scale, indices)
    _indexed_modulation_triton.indexed_scale_shift_bf16_(x_tri, shift, scale, indices)
    assert torch.equal(x_cpp, x_tri)


@requires_cuda
@pytest.mark.parametrize(("rows", "groups"), [(20992, 4), (997, 3), (1, 1)])
def test_indexed_scale_shift_fp32_out_matches_modulate_then_cast(
    rows: int, groups: int
):
    """The fused fp32-out modulate (final layer, R10) must be bitwise equal to
    the in-place bf16 modulate followed by the .to(fp32) widening, including
    for the strided shift/scale row views the final layer passes."""
    from sglang.kernels.ops.diffusion import (
        indexed_scale_shift_bf16_to_fp32,
    )

    x, _, scale, shift, _, _, indices, _ = _random_case(rows, groups, 23)
    reference = indexed_scale_shift_bf16_(x.clone(), shift, scale, indices).to(
        torch.float32
    )

    fused = indexed_scale_shift_bf16_to_fp32(x, shift, scale, indices)
    assert fused.dtype is torch.float32
    assert torch.equal(fused, reference)

    # Strided modulation rows, as unbound from the AdaLN cache's [M, 2, H]
    # final() views.
    packed = torch.stack((shift, scale), dim=1)
    strided_shift, strided_scale = packed.unbind(dim=1)
    if groups > 1:
        assert not strided_shift.is_contiguous()
    fused_strided = indexed_scale_shift_bf16_to_fp32(
        x, strided_shift, strided_scale, indices
    )
    assert torch.equal(fused_strided, reference)

    empty = indexed_scale_shift_bf16_to_fp32(x[:0], shift, scale, indices[:0])
    assert empty.shape == (0, x.shape[1]) and empty.dtype is torch.float32


@requires_cuda
@pytest.mark.parametrize(("rows", "groups"), [(20992, 9), (4096, 3)])
def test_cpp_rmsnorm_indexed_contract_vs_triton(rows: int, groups: int):
    """Plan A/B C++ vs Triton: the residual write-back is bitwise, the norm
    output near-lossless (the two backends reduce in different tree orders)."""
    x, weight, scale, shift, gate, update, indices, w_eff = _random_case(
        rows, groups, 18
    )
    _skip_unless_cpp_backend(x, w_eff, shift, indices)

    out_cpp = _rmsnorm_indexed_modulate_jit.rmsnorm_indexed_scale_shift_cuda(
        x, w_eff, shift, indices, eps=_EPS
    )
    out_tri = _rmsnorm_indexed_modulate_triton.rmsnorm_indexed_scale_shift(
        x, w_eff, shift, indices, eps=_EPS
    )
    _, magnitude = _norm_modulate_fp64(x, weight, scale, shift, indices)
    _report_and_assert_close(
        out_cpp,
        out_tri,
        magnitude,
        label=f"cpp_vs_triton_a[{rows}x{groups}]",
        max_scaled_ulp=4.0,
    )

    res_cpp, res_tri, res_eager = x.clone(), x.clone(), x.clone()
    out_cpp, returned = (
        _rmsnorm_indexed_modulate_jit.gate_residual_rmsnorm_indexed_scale_shift_cuda_(
            res_cpp, update, gate, w_eff, shift, indices, eps=_EPS
        )
    )
    out_tri, _ = (
        _rmsnorm_indexed_modulate_triton.gate_residual_rmsnorm_indexed_scale_shift_(
            res_tri, update, gate, w_eff, shift, indices, eps=_EPS
        )
    )
    _indexed_modulation_triton.indexed_gate_bf16_(res_eager, gate, update, indices)
    assert returned.data_ptr() == res_cpp.data_ptr()
    assert torch.equal(res_cpp, res_tri), "Plan B residual must be bitwise vs Triton"
    assert torch.equal(
        res_cpp, res_eager
    ), "Plan B residual must be bitwise vs indexed_gate_bf16_"
    _, magnitude = _norm_modulate_fp64(res_tri, weight, scale, shift, indices)
    _report_and_assert_close(
        out_cpp,
        out_tri,
        magnitude,
        label=f"cpp_vs_triton_b[{rows}x{groups}]",
        max_scaled_ulp=4.0,
    )


@requires_cuda
def test_indexed_scale_shift_dispatch_order(monkeypatch):
    """Supported input runs the C++ backend; unsupported falls back to Triton."""
    x, _, scale, shift, _, _, indices, _ = _random_case(512, 3, 19)
    if not _indexed_modulation_jit.can_use_indexed_scale_shift_cuda(
        x, shift, scale, indices
    ):
        pytest.skip("JIT C++ backend unavailable (needs SM90+ CUDA)")

    fallback_calls: list[str] = []

    def fallback_stub(*args, **kwargs):
        fallback_calls.append("triton")
        return args[0]

    monkeypatch.setattr(
        _indexed_modulation_jit, "triton_indexed_scale_shift_bf16_", fallback_stub
    )
    reference = _indexed_modulation_triton.indexed_scale_shift_bf16_(
        x.clone(), shift, scale, indices
    )
    got = indexed_scale_shift_bf16_(x.clone(), shift, scale, indices)
    assert not fallback_calls, "supported input must not reach the Triton fallback"
    assert torch.equal(got, reference)

    # int16 indices are outside the C++ gate; the call must route to Triton.
    assert not _indexed_modulation_jit.can_use_indexed_scale_shift_cuda(
        x, shift, scale, indices.to(torch.int16)
    )
    indexed_scale_shift_bf16_(x.clone(), shift, scale, indices.to(torch.int16))
    assert fallback_calls == ["triton"]


@requires_cuda
def test_rmsnorm_indexed_dispatch_order(monkeypatch):
    """Both fused-adaLN entry points prefer C++ and fail closed to Triton."""
    x, _, scale, shift, gate, update, indices, w_eff = _random_case(512, 3, 20)
    _skip_unless_cpp_backend(x, w_eff, shift, indices)

    fallback_calls: list[str] = []
    monkeypatch.setattr(
        _rmsnorm_indexed_modulate_jit,
        "triton_rmsnorm_indexed_scale_shift",
        lambda *a, **k: fallback_calls.append("plan_a"),
    )
    monkeypatch.setattr(
        _rmsnorm_indexed_modulate_jit,
        "triton_gate_residual_rmsnorm_indexed_scale_shift_",
        lambda *a, **k: fallback_calls.append("plan_b"),
    )
    rmsnorm_indexed_scale_shift(x, w_eff, shift, indices, eps=_EPS)
    gate_residual_rmsnorm_indexed_scale_shift_(
        x.clone(), update, gate, w_eff, shift, indices, eps=_EPS
    )
    assert not fallback_calls, "supported input must not reach the Triton fallback"

    # A non-bf16 weight_eff is outside the C++ gate on both entry points.
    bad_w = w_eff.to(torch.bfloat16)
    assert not _rmsnorm_indexed_modulate_jit.can_use_rmsnorm_indexed_scale_shift_cuda(
        x, bad_w, shift, indices
    )
    rmsnorm_indexed_scale_shift(x, bad_w, shift, indices, eps=_EPS)
    gate_residual_rmsnorm_indexed_scale_shift_(
        x.clone(), update, gate, bad_w, shift, indices, eps=_EPS
    )
    assert fallback_calls == ["plan_a", "plan_b"]


@requires_cuda
def test_flag_off_keeps_eager_forward_bit_identical(monkeypatch):
    """MINIMAX_H3_FUSED_ADALN=0 leaves block.forward on the eager kernels."""
    monkeypatch.setenv("MINIMAX_H3_FUSED_ADALN", "0")
    rows, groups = 2048, 6
    block = _stub_block(21)
    (params,) = _block_adaln_params(1, groups, 6)
    indices = _packed_indices(rows, groups, 9)
    generator = torch.Generator(device="cuda").manual_seed(13)
    x = torch.randn(rows, _HIDDEN, generator=generator, device="cuda").to(
        torch.bfloat16
    )
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = params

    # reference: the pre-fusion eager chain spelled out with the raw kernels
    reference = x.clone()
    with torch.no_grad():
        h = indexed_scale_shift_bf16_(
            block.norm1(reference), shift_msa, scale_msa, indices
        )
        h = block.attn(h)
        reference = indexed_gate_bf16_(reference, gate_msa, h, indices)
        h = indexed_scale_shift_bf16_(
            block.norm2(reference), shift_mlp, scale_mlp, indices
        )
        h = block.mlp(h)
        reference = indexed_gate_bf16_(reference, gate_mlp, h, indices)

    actual = block(
        x.clone(),
        adaln_input=None,
        combined_indices=indices,
        adaln_params=params,
        **_block_kwargs(rows),
    )
    assert torch.equal(actual, reference)
