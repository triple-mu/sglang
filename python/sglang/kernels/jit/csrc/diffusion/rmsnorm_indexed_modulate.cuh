// CUDA fast path for the MiniMax-H3 fused adaLN chain over indexed
// modulation groups (one merged kernel body for both plans):
//
//   Plan A (kHasGate = false):
//     out[r] = (x[r] / rms(x[r])) * w_eff[g] + shift[g],   g = indices[r]
//   Plan B (kHasGate = true), residual updated in place first:
//     y[r]   = round(x[r] + round(gate[g] * update[r]))     (bf16 RNE rounds)
//     out[r] = (y[r] / rms(y[r])) * w_eff[g] + shift[g]
//
// The Plan B residual write-back replicates the eager `indexed_gate_bf16_`
// rounding chain bitwise (round after gate*update and after the add; the
// paired cvt.rn.bf16x2.f32 keeps the per-element RNE semantics of the eager
// store's cvt.rn.bf16.f32). The norm/modulate output accumulates the sum of
// squares in fp32 and rounds to bf16 once on the store, like the Triton
// `_rmsnorm_indexed_modulate_kernel` it replaces (near-lossless contract,
// not bitwise -- the reduction tree differs across implementations).
//
// Each CTA owns a row and keeps it in registers as packed bf16x2 between the
// reduction and the modulate epilogue; bf16 -> fp32 widenings are integer
// shifts so the quarter-rate F2F pipe only runs the rounding cvts.

#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/math.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <cstdint>
#include <initializer_list>
#include <limits>

namespace sglang {

namespace rmsnorm_indexed_modulate {

constexpr int kVec = 8;  // 16B of bf16 per load
constexpr uintptr_t kAlignment = 16;

struct RowParams {
  void* out;
  void* x;  // Plan B: also the residual output (in place), so not restrict
  const void* update;
  const void* gate;
  const void* w_eff;
  const void* shift;
  const void* indices;
  float eps;
};

/// CTA sum with a single barrier: warp partials land in smem, then every
/// thread folds them in warp order (deterministic; broadcast smem reads).
template <int kThreads>
SGL_DEVICE float cta_reduce_sum(float value, int warp, int lane, float* scratch) {
  constexpr int kWarps = kThreads / device::kWarpThreads;
  static_assert(kWarps >= 1 && kWarps <= device::kWarpThreads);
  value = device::warp::reduce_sum(value);
  if (lane == 0) {
    scratch[warp] = value;
  }
  __syncthreads();
  float total = scratch[0];
#pragma unroll
  for (int i = 1; i < kWarps; ++i) {
    total += scratch[i];
  }
  return total;
}

/// \brief Widen a packed bf16x2 to fp32 (exact) with integer shifts, keeping
/// the quarter-rate F2F pipe free for the rounding cvts.
SGL_DEVICE float2 widen_bf16x2(uint32_t packed) {
  return {__uint_as_float(packed << 16), __uint_as_float(packed & 0xFFFF0000u)};
}

/// \brief RNE-round two fp32 values to a packed bf16x2; per-element semantics
/// match the `cvt.rn.bf16.f32` the eager stores execute.
SGL_DEVICE uint32_t round_bf16x2_rn(float lo, float hi) {
  const __nv_bfloat162 packed = __float22bfloat162_rn({lo, hi});
  return reinterpret_cast<const uint32_t&>(packed);
}

template <int kHidden, int kThreads, bool kHasGate, typename IdxT>
__launch_bounds__(kThreads) __global__ void rmsnorm_indexed_modulate_kernel(const RowParams __grid_constant__ params) {
  using namespace device;
  using Vec = AlignedVector<bf16_t, kVec>;
  using WVec = AlignedVector<fp32_t, 4>;
  static_assert(kHidden % kVec == 0);
  constexpr int kVecs = kHidden / kVec;
  constexpr int kVecsPerThread = (kVecs + kThreads - 1) / kThreads;
  constexpr int kWarps = kThreads / kWarpThreads;

  const int64_t row = blockIdx.x;
  const int tid = threadIdx.x;
  const int lane = tid & (kWarpThreads - 1);
  const int warp = tid >> 5;
  const int64_t group = static_cast<int64_t>(static_cast<const IdxT*>(params.indices)[row]);
  const int64_t row_offset = row * kHidden;
  const int64_t group_offset = group * kHidden;

  __shared__ float scratch[kWarps];

  auto* x = static_cast<bf16_t*>(params.x);
  Vec x_regs[kVecsPerThread];  // this thread's row slice, packed bf16x2
  float sum_sq = 0.0f;
#pragma unroll
  for (int k = 0; k < kVecsPerThread; ++k) {
    const int vec_id = tid + k * kThreads;
    if (kVecs % kThreads != 0 && vec_id >= kVecs) {
      break;
    }
    Vec& x_vec = x_regs[k];
    x_vec.load(x + row_offset, vec_id);
    auto* x2 = reinterpret_cast<uint32_t*>(x_vec.data());
    if constexpr (kHasGate) {
      const auto* __restrict__ update = static_cast<const bf16_t*>(params.update);
      const auto* __restrict__ gate = static_cast<const bf16_t*>(params.gate);
      Vec update_vec, gate_vec;
      update_vec.load(update + row_offset, vec_id);
      gate_vec.load(gate + group_offset, vec_id);
      const auto* u2 = reinterpret_cast<const uint32_t*>(update_vec.data());
      const auto* g2 = reinterpret_cast<const uint32_t*>(gate_vec.data());
#pragma unroll
      for (int p = 0; p < kVec / 2; ++p) {
        // eager pair: bf16 round after gate*update and after the residual add
        const auto [g0, g1] = widen_bf16x2(g2[p]);
        const auto [u0, u1] = widen_bf16x2(u2[p]);
        const auto [gu0, gu1] = widen_bf16x2(round_bf16x2_rn(__fmul_rn(g0, u0), __fmul_rn(g1, u1)));
        const auto [x0, x1] = widen_bf16x2(x2[p]);
        x2[p] = round_bf16x2_rn(__fadd_rn(x0, gu0), __fadd_rn(x1, gu1));
      }
      x_vec.store(x + row_offset, vec_id);
    }
#pragma unroll
    for (int p = 0; p < kVec / 2; ++p) {
      const auto [v0, v1] = widen_bf16x2(x2[p]);
      sum_sq = fmaf(v0, v0, sum_sq);
      sum_sq = fmaf(v1, v1, sum_sq);
    }
  }

  const float total = cta_reduce_sum<kThreads>(sum_sq, warp, lane, scratch);
  const float rstd = math::rsqrt(total / static_cast<float>(kHidden) + params.eps);

  auto* __restrict__ out = static_cast<bf16_t*>(params.out);
  const auto* __restrict__ w_eff = static_cast<const fp32_t*>(params.w_eff) + group_offset;
  const auto* __restrict__ shift = static_cast<const bf16_t*>(params.shift) + group_offset;
#pragma unroll
  for (int k = 0; k < kVecsPerThread; ++k) {
    const int vec_id = tid + k * kThreads;
    if (kVecs % kThreads != 0 && vec_id >= kVecs) {
      break;
    }
    WVec w_lo, w_hi;
    w_lo.load(w_eff, vec_id * 2);
    w_hi.load(w_eff, vec_id * 2 + 1);
    Vec shift_vec, out_vec;
    shift_vec.load(shift, vec_id);
    const auto* x2 = reinterpret_cast<const uint32_t*>(x_regs[k].data());
    const auto* s2 = reinterpret_cast<const uint32_t*>(shift_vec.data());
    auto* o2 = reinterpret_cast<uint32_t*>(out_vec.data());
#pragma unroll
    for (int p = 0; p < kVec / 2; ++p) {
      const auto [x0, x1] = widen_bf16x2(x2[p]);
      const auto [s0, s1] = widen_bf16x2(s2[p]);
      const float w0 = 2 * p < 4 ? w_lo[2 * p] : w_hi[2 * p - 4];
      const float w1 = 2 * p + 1 < 4 ? w_lo[2 * p + 1] : w_hi[2 * p - 3];
      // x * rstd * weight + shift, rounded to bf16 once on the store
      o2[p] = round_bf16x2_rn(fmaf(__fmul_rn(x0, rstd), w0, s0), fmaf(__fmul_rn(x1, rstd), w1, s1));
    }
    out_vec.store(out + row_offset, vec_id);
  }
}

inline void verify_alignment(std::initializer_list<const void*> pointers) {
  for (const void* pointer : pointers) {
    CHECK_HOST(reinterpret_cast<uintptr_t>(pointer) % kAlignment == 0)
        << "rmsnorm_indexed_modulate requires 16-byte aligned tensors";
  }
}

/**
 * \brief Validate and launch the fused RMSNorm + indexed adaLN scale/shift.
 *
 * \tparam kHidden Row width in elements; the block covers it exactly.
 * \tparam kThreads Threads per block (whole warps).
 */
template <int kHidden, int kThreads>
struct RMSNormIndexedModulateKernel {
  static_assert(kHidden % kVec == 0);
  static_assert(kThreads % 32 == 0);
  static_assert(kHidden / kVec >= kThreads, "each thread owns at least one vector");

  template <bool kHasGate>
  static void launch(const RowParams& params, int64_t rows, bool idx_is_int32, DLDevice device) {
    CHECK_HOST(rows <= std::numeric_limits<uint32_t>::max()) << "rows out of range: " << rows;
    const auto launcher = host::LaunchKernel(static_cast<uint32_t>(rows), kThreads, device);
    if (idx_is_int32) {
      launcher(rmsnorm_indexed_modulate_kernel<kHidden, kThreads, kHasGate, int32_t>, params);
    } else {
      launcher(rmsnorm_indexed_modulate_kernel<kHidden, kThreads, kHasGate, int64_t>, params);
    }
  }

  /**
   * \brief Plan A: ``out = (x / rms(x)) * w_eff[indices] + shift[indices]``.
   *
   * \param out Output rows [rows, kHidden], bf16.
   * \param x Input rows [rows, kHidden], bf16 (read-only).
   * \param w_eff Merged ``norm_weight * (1 + scale)`` rows [groups, kHidden], fp32.
   * \param shift Per-group shift rows [groups, kHidden], bf16.
   * \param indices Per-row group index [rows], int32 or int64.
   * \param eps RMSNorm epsilon.
   */
  static void
  run(tvm::ffi::TensorView out,
      tvm::ffi::TensorView x,
      tvm::ffi::TensorView w_eff,
      tvm::ffi::TensorView shift,
      tvm::ffi::TensorView indices,
      double eps) {
    using namespace host;
    auto R = SymbolicSize{"rows"};
    auto G = SymbolicSize{"groups"};
    auto idx_type = SymbolicDType{};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({R, kHidden}).with_dtype<bf16_t>().with_device(device).verify(out).verify(x);
    TensorMatcher({G, kHidden}).with_dtype<fp32_t>().with_device(device).verify(w_eff);
    TensorMatcher({G, kHidden}).with_dtype<bf16_t>().with_device(device).verify(shift);
    TensorMatcher({R}).with_dtype<int32_t, int64_t>(idx_type).with_device(device).verify(indices);

    const int64_t rows = R.unwrap();
    if (rows == 0) {
      return;
    }
    verify_alignment({out.data_ptr(), x.data_ptr(), w_eff.data_ptr(), shift.data_ptr()});
    CHECK_HOST(out.data_ptr() != x.data_ptr()) << "out must not alias x";

    const auto params = RowParams{
        .out = out.data_ptr(),
        .x = x.data_ptr(),
        .update = nullptr,
        .gate = nullptr,
        .w_eff = w_eff.data_ptr(),
        .shift = shift.data_ptr(),
        .indices = indices.data_ptr(),
        .eps = static_cast<float>(eps),
    };
    launch<false>(params, rows, idx_type.is_type<int32_t>(), device.unwrap());
  }

  /**
   * \brief Plan B: gated residual add fused with the following norm/modulate.
   *
   * ``residual`` is updated in place to ``y = residual + gate[idx] * update``
   * (bitwise vs the eager ``indexed_gate_bf16_`` chain), then
   * ``out = (y / rms(y)) * w_eff[idx] + shift[idx]``.
   *
   * \param out Output rows [rows, kHidden], bf16.
   * \param residual Residual rows [rows, kHidden], bf16, updated in place.
   * \param update Update rows [rows, kHidden], bf16.
   * \param gate Per-group gate rows [groups, kHidden], bf16.
   * \param w_eff Merged ``norm_weight * (1 + scale)`` rows [groups, kHidden], fp32.
   * \param shift Per-group shift rows [groups, kHidden], bf16.
   * \param indices Per-row group index [rows], int32 or int64.
   * \param eps RMSNorm epsilon.
   */
  static void run_gated(
      tvm::ffi::TensorView out,
      tvm::ffi::TensorView residual,
      tvm::ffi::TensorView update,
      tvm::ffi::TensorView gate,
      tvm::ffi::TensorView w_eff,
      tvm::ffi::TensorView shift,
      tvm::ffi::TensorView indices,
      double eps) {
    using namespace host;
    auto R = SymbolicSize{"rows"};
    auto G = SymbolicSize{"groups"};
    auto idx_type = SymbolicDType{};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({R, kHidden}).with_dtype<bf16_t>().with_device(device).verify(out).verify(residual).verify(update);
    TensorMatcher({G, kHidden}).with_dtype<bf16_t>().with_device(device).verify(gate).verify(shift);
    TensorMatcher({G, kHidden}).with_dtype<fp32_t>().with_device(device).verify(w_eff);
    TensorMatcher({R}).with_dtype<int32_t, int64_t>(idx_type).with_device(device).verify(indices);

    const int64_t rows = R.unwrap();
    if (rows == 0) {
      return;
    }
    verify_alignment(
        {out.data_ptr(), residual.data_ptr(), update.data_ptr(), gate.data_ptr(), w_eff.data_ptr(), shift.data_ptr()});
    CHECK_HOST(out.data_ptr() != residual.data_ptr() && out.data_ptr() != update.data_ptr())
        << "out must not alias residual/update";
    CHECK_HOST(residual.data_ptr() != update.data_ptr()) << "residual must not alias update";

    const auto params = RowParams{
        .out = out.data_ptr(),
        .x = residual.data_ptr(),
        .update = update.data_ptr(),
        .gate = gate.data_ptr(),
        .w_eff = w_eff.data_ptr(),
        .shift = shift.data_ptr(),
        .indices = indices.data_ptr(),
        .eps = static_cast<float>(eps),
    };
    launch<true>(params, rows, idx_type.is_type<int32_t>(), device.unwrap());
  }
};

}  // namespace rmsnorm_indexed_modulate

}  // namespace sglang
