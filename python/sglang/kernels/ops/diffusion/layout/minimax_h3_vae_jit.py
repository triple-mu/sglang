# SPDX-License-Identifier: Apache-2.0
"""Close-contract FP32 H3 tile/window assembly and RGB conversion on SM100.

The vector path uses explicit FP32 FMA; the scalar path preserves strided
layouts and original corner ordering. Model callers must quality-gate it.
"""

import torch

from ..common.minimax_h3_vae_jit import load_output_kernels, supported_cuda_tensor


def can_use_minimax_h3_vae_output(value) -> bool:
    return (
        supported_cuda_tensor(value)
        and value.ndim == 5
        and value.dtype == torch.float32
        and all(s >= 0 for s in value.stride())
    )


def can_use_minimax_h3_vae_assemble(rows, y_overlap, x_overlap) -> bool:
    if not rows or not rows[0]:
        return False
    first = rows[0][0]
    nr = len(rows)
    nc = len(rows[0])
    if (
        not can_use_minimax_h3_vae_output(first)
        or nr * nc > 64
        or any(len(row) != nc for row in rows)
    ):
        return False
    if len(y_overlap) != nr - 1 or len(x_overlap) != nc - 1:
        return False
    if any(not 0 < int(n) < first.shape[-2] for n in y_overlap) or any(
        not 0 < int(n) < first.shape[-1] for n in x_overlap
    ):
        return False
    return all(
        can_use_minimax_h3_vae_output(x)
        and x.shape == first.shape
        and x.stride() == first.stride()
        and x.device == first.device
        for row in rows
        for x in row
    )


def can_use_minimax_h3_vae_temporal_write(
    part, overlap, extent, out, start, count
) -> bool:
    if not can_use_minimax_h3_vae_output(part) or not can_use_minimax_h3_vae_output(
        out
    ):
        return False
    if (
        part.device != out.device
        or part.shape[:2] != out.shape[:2]
        or part.shape[-2:] != out.shape[-2:]
    ):
        return False
    if (
        not all(type(v) is int for v in (extent, start, count))
        or extent < 0
        or start < 0
        or count < 0
        or start + count > out.shape[2]
        or count > part.shape[2]
    ):
        return False
    if out.untyped_storage().data_ptr() == part.untyped_storage().data_ptr():
        return False
    if overlap is not None:
        if not (
            can_use_minimax_h3_vae_output(overlap)
            and overlap.device == part.device
            and overlap.shape[:2] == part.shape[:2]
            and overlap.shape[-2:] == part.shape[-2:]
        ):
            return False
        if out.untyped_storage().data_ptr() == overlap.untyped_storage().data_ptr():
            return False
    return True


def can_use_minimax_h3_vae_denorm(value, mean, std) -> bool:
    return (
        can_use_minimax_h3_vae_output(value)
        and value.shape[1] == 3
        and len(mean) == len(std) == 3
        and all(float(s) != 0 for s in std)
    )


def minimax_h3_vae_assemble(rows, y_overlap, x_overlap):
    if not can_use_minimax_h3_vae_assemble(rows, y_overlap, x_overlap):
        raise ValueError("Unsupported MiniMax-H3 spatial assembly geometry")
    first = rows[0][0]
    nr, nc = len(rows), len(rows[0])
    shape = first.shape
    heights = [shape[-2] - (y_overlap[i] if i < nr - 1 else 0) for i in range(nr)]
    widths = [shape[-1] - (x_overlap[j] if j < nc - 1 else 0) for j in range(nc)]
    if min(heights + widths) <= 0:
        raise ValueError("Invalid crop extent")
    out = torch.empty(
        (*shape[:-2], sum(heights), sum(widths)), dtype=first.dtype, device=first.device
    )
    descriptors = []
    oy = 0
    for i, row in enumerate(rows):
        ox = 0
        for j, value in enumerate(row):
            descriptors.append(
                [
                    value.data_ptr(),
                    rows[i - 1][j].data_ptr() if i else 0,
                    row[j - 1].data_ptr() if j else 0,
                    oy,
                    ox,
                    heights[i],
                    widths[j],
                    int(y_overlap[i - 1]) if i else 0,
                    int(x_overlap[j - 1]) if j else 0,
                ]
            )
            ox += widths[j]
        oy += heights[i]
    # CPU metadata is copied into the CUDA kernel parameter block by the C++
    # launcher. No metadata H2D transfer or device synchronization is required.
    meta = torch.tensor(descriptors, dtype=torch.int64, device="cpu")
    load_output_kernels().assemble(first, meta, out)
    return out


def minimax_h3_vae_temporal_write(part, overlap, extent, out, start, count):
    if not can_use_minimax_h3_vae_temporal_write(
        part, overlap, extent, out, start, count
    ):
        raise ValueError("Unsupported MiniMax-H3 temporal write geometry or alias")
    if overlap is not None:
        if overlap.shape[:2] != part.shape[:2] or overlap.shape[-2:] != part.shape[-2:]:
            raise ValueError("Temporal overlap shape mismatch")
        extent = min(int(extent), overlap.shape[2], part.shape[2])
    else:
        extent = 0
    if count < 0 or start < 0 or start + count > out.shape[2] or count > part.shape[2]:
        raise ValueError("Temporal write exceeds destination")
    if count:
        load_output_kernels().temporal(
            part,
            overlap if overlap is not None else part,
            out,
            int(start),
            int(count),
            int(extent),
        )


def minimax_h3_vae_denorm(value, mean, std):
    if not can_use_minimax_h3_vae_denorm(value, mean, std):
        raise ValueError("Unsupported MiniMax-H3 RGB normalization")
    out = torch.empty(value.shape, dtype=torch.float32, device=value.device)
    load_output_kernels().denorm(value, out, *map(float, mean), *map(float, std))
    return out
