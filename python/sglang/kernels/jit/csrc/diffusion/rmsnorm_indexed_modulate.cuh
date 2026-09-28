// Fused RMSNorm + indexed adaLN scale/shift for the MiniMax-H3 DiT, bitwise
// equal to the eager chain it replaces:
//
//   Plan A:  out[r] = modulate(rmsnorm(x[r]), scale[g], shift[g]),  g = indices[r]
//   Plan B:  y[r]   = round(x[r] + round(gate[g] * update[r]))       (residual, in place)
//            out[r] = modulate(rmsnorm(y[r]), scale[g], shift[g])
//
// with rmsnorm = bf16(gamma * (rstd * y)) in aten's reduction order (the
// (32, 4) block of vectorized_layer_norm_kernel<BFloat16, float, rms_norm>,
// vec_size 4, per-thread sequential FFMA, shfl_down tree, two-round smem
// combine, sigma2 / N at thread 0, rsqrtf) and modulate the eager
// indexed_scale_shift_bf16_ chain: round(1 + scale), round(product),
// round(+ shift). The Plan B write-back replicates indexed_gate_bf16_. A
// first-call self-check at the model keeps this honest against the torch
// build actually installed.
//
// Ported from origin/minimax-h3-fusion-dit with only the bitexact variant
// kept; the modulation rows take a row stride so the six views of one
// [groups, 6 * hidden] adaLN slab pass without a copy.

#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/cta.cuh>
#include <sgl_kernel/math.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <cstdint>
#include <initializer_list>
#include <limits>

namespace sglang {

namespace rmsnorm_indexed_modulate {

constexpr uintptr_t kAlignment = 16;
// aten pins a (32, 4) block and vec_size 4; both are part of the reduction order.
constexpr int kVec = 4;
constexpr int kThreads = 128;
constexpr int kWarps = kThreads / 32;

constexpr float kFp8Max = 448.0f;

struct RowParams {
  void* out;   // bf16 [rows, hidden], or the fp8 rows when the kernel quantises
  float* s;    // fp32 [rows] per-token scales (fp8 mode only)
  void* x;  // Plan B: also the residual output (in place)
  const void* update;
  const void* gate;
  const void* gamma;  // bf16 [hidden], the RMSNorm weight, unmerged
  const void* scale;  // bf16 [groups, hidden] rows, `group_stride` elements apart
  const void* shift;
  const void* indices;
  int64_t group_stride;
  float eps;
};

/// Widen a packed bf16x2 to fp32 (exact) with integer shifts.
SGL_DEVICE float2 widen_bf16x2(uint32_t packed) {
  return {__uint_as_float(packed << 16), __uint_as_float(packed & 0xFFFF0000u)};
}

/// RNE-round two fp32 values to a packed bf16x2, like the eager stores' cvt.rn.bf16.f32.
SGL_DEVICE uint32_t round_bf16x2_rn(float lo, float hi) {
  const __nv_bfloat162 packed = __float22bfloat162_rn({lo, hi});
  return reinterpret_cast<const uint32_t&>(packed);
}

/// One bf16x2 pair of the eager gated residual: round after gate*update and after the add.
SGL_DEVICE uint32_t gate_residual_pair_bf16x2(uint32_t x2, uint32_t g2, uint32_t u2) {
  const auto [g0, g1] = widen_bf16x2(g2);
  const auto [u0, u1] = widen_bf16x2(u2);
  const auto [gu0, gu1] = widen_bf16x2(round_bf16x2_rn(__fmul_rn(g0, u0), __fmul_rn(g1, u1)));
  const auto [x0, x1] = widen_bf16x2(x2);
  return round_bf16x2_rn(__fadd_rn(x0, gu0), __fadd_rn(x1, gu1));
}

/// Per-token fp8 quantisation of the modulated row, the scheme the fp8 GEMM applies itself:
/// scale = amax / 448, q = sat(x * rcp(scale)).
template <int kHidden, bool kHasGate, bool kFp8Out, typename IdxT>
__launch_bounds__(kThreads) __global__ void rmsnorm_indexed_modulate_kernel(
    const RowParams __grid_constant__ params) {
  using namespace device;
  using Vec = AlignedVector<bf16_t, kVec>;
  static_assert(kHidden % kVec == 0);
  constexpr int kVecs = kHidden / kVec;
  constexpr int kVecsPerThread = (kVecs + kThreads - 1) / kThreads;

  const int64_t row = blockIdx.x;
  const int tid = threadIdx.x;  // aten's thrx
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int64_t group = static_cast<int64_t>(static_cast<const IdxT*>(params.indices)[row]);
  const int64_t row_offset = row * kHidden;
  const int64_t group_offset = group * params.group_stride;

  // Plan B write-back fused into the load, then the sum of squares in aten's
  // element order: thread t owns vectors t, t + 128, ...; FFMA per element.
  auto* x = static_cast<bf16_t*>(params.x);
  Vec x_regs[kVecsPerThread];
  float sigma2 = 0.0f;
#pragma unroll
  for (int k = 0; k < kVecsPerThread; ++k) {
    const int vec_id = tid + k * kThreads;
    if (kVecs % kThreads != 0 && vec_id >= kVecs) break;
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
      for (int p = 0; p < kVec / 2; ++p) x2[p] = gate_residual_pair_bf16x2(x2[p], g2[p], u2[p]);
      x_vec.store(x + row_offset, vec_id);
    }
#pragma unroll
    for (int p = 0; p < kVec / 2; ++p) {
      const auto [v0, v1] = widen_bf16x2(x2[p]);
      sigma2 = __fmaf_rn(v0, v0, sigma2);
      sigma2 = __fmaf_rn(v1, v1, sigma2);
    }
  }

  // aten's intra-warp shfl_down tree (offsets 16..1) ...
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    sigma2 = __fadd_rn(sigma2, __shfl_down_sync(0xffffffffu, sigma2, offset));
  }
  // ... then its two-round inter-warp combine with two barriers per round.
  __shared__ float sigma_buf[kWarps];
#pragma unroll
  for (int offset = kWarps / 2; offset > 0; offset >>= 1) {
    if (lane == 0 && warp >= offset && warp < 2 * offset) sigma_buf[warp - offset] = sigma2;
    __syncthreads();
    if (lane == 0 && warp < offset) sigma2 = __fadd_rn(sigma2, sigma_buf[warp]);
    __syncthreads();
  }
  if (tid == 0) sigma_buf[0] = __fdiv_rn(sigma2, static_cast<float>(kHidden));
  __syncthreads();
  const float rstd = math::rsqrt(__fadd_rn(sigma_buf[0], params.eps));

  const auto* __restrict__ gamma = static_cast<const bf16_t*>(params.gamma);
  const auto* __restrict__ scale = static_cast<const bf16_t*>(params.scale) + group_offset;
  const auto* __restrict__ shift = static_cast<const bf16_t*>(params.shift) + group_offset;
  Vec out_regs[kVecsPerThread];
  float amax = 0.0f;
#pragma unroll
  for (int k = 0; k < kVecsPerThread; ++k) {
    const int vec_id = tid + k * kThreads;
    if (kVecs % kThreads != 0 && vec_id >= kVecs) break;
    Vec gamma_vec, scale_vec, shift_vec;
    gamma_vec.load(gamma, vec_id);
    scale_vec.load(scale, vec_id);
    shift_vec.load(shift, vec_id);
    Vec& out_vec = out_regs[k];
#pragma unroll
    for (int i = 0; i < kVec; ++i) {
      // aten: one bf16 round of gamma * (rstd * x); then the eager modulate chain.
      const bf16_t normed = cast<bf16_t>(
          __fmul_rn(cast<fp32_t>(gamma_vec[i]), __fmul_rn(rstd, cast<fp32_t>(x_regs[k][i]))));
      const bf16_t one_plus_scale = cast<bf16_t>(__fadd_rn(1.0f, cast<fp32_t>(scale_vec[i])));
      const bf16_t product =
          cast<bf16_t>(__fmul_rn(cast<fp32_t>(normed), cast<fp32_t>(one_plus_scale)));
      out_vec[i] = cast<bf16_t>(__fadd_rn(cast<fp32_t>(product), cast<fp32_t>(shift_vec[i])));
      if constexpr (kFp8Out) amax = fmaxf(amax, fabsf(cast<fp32_t>(out_vec[i])));
    }
    if constexpr (!kFp8Out) out_vec.store(static_cast<bf16_t*>(params.out) + row_offset, vec_id);
  }
  if constexpr (kFp8Out) {
    __shared__ float amax_buf[kWarps];
    cta::reduce_max(amax, amax_buf);
    __syncthreads();
    const float row_scale = __fdiv_rn(amax_buf[0], kFp8Max);
    if (tid == 0) params.s[row] = row_scale;
    const float scale_inv = row_scale == 0.0f ? 0.0f : __frcp_rn(row_scale);
    auto* __restrict__ q = static_cast<fp8_e4m3_t*>(params.out) + row_offset;
    using QVec = AlignedVector<fp8_e4m3_t, kVec>;
#pragma unroll
    for (int k = 0; k < kVecsPerThread; ++k) {
      const int vec_id = tid + k * kThreads;
      if (kVecs % kThreads != 0 && vec_id >= kVecs) break;
      QVec q_vec;
      auto* pairs = reinterpret_cast<__nv_fp8x2_storage_t*>(q_vec.data());
#pragma unroll
      for (int p = 0; p < kVec / 2; ++p) {
        const float v0 = __fmul_rn(scale_inv, cast<fp32_t>(out_regs[k][2 * p]));
        const float v1 = __fmul_rn(scale_inv, cast<fp32_t>(out_regs[k][2 * p + 1]));
        const float2 payload{fmaxf(fminf(v0, kFp8Max), -kFp8Max), fmaxf(fminf(v1, kFp8Max), -kFp8Max)};
        pairs[p] = __nv_cvt_float2_to_fp8x2(payload, __NV_SATFINITE, __NV_E4M3);
      }
      q_vec.store(q, vec_id);
    }
  }
}

inline void verify_alignment(std::initializer_list<const void*> pointers) {
  for (const void* pointer : pointers) {
    CHECK_HOST(reinterpret_cast<uintptr_t>(pointer) % kAlignment == 0)
        << "rmsnorm_indexed_modulate requires 16-byte aligned tensors";
  }
}

/*!
 * \brief Validate and launch the bitexact fused RMSNorm + indexed adaLN modulation.
 * \tparam kHidden Row width in elements; the whole row lives in one block's registers.
 */
template <int kHidden>
struct RMSNormIndexedModulateKernel {
  static_assert(kHidden % kVec == 0);

  template <bool kHasGate, bool kFp8Out = false>
  static void launch(const RowParams& params, int64_t rows, bool idx_is_int32, DLDevice device) {
    CHECK_HOST(rows <= std::numeric_limits<uint32_t>::max()) << "rows out of range: " << rows;
    const auto launcher = host::LaunchKernel(static_cast<uint32_t>(rows), kThreads, device);
    if (idx_is_int32) {
      launcher(rmsnorm_indexed_modulate_kernel<kHidden, kHasGate, kFp8Out, int32_t>, params);
    } else {
      launcher(rmsnorm_indexed_modulate_kernel<kHidden, kHasGate, kFp8Out, int64_t>, params);
    }
  }

  /// The fp8 outputs of `run_fp8` / `run_gated_fp8`: q [rows, kHidden] e4m3, s [rows, 1] fp32.
  static void verify_fp8_outputs(tvm::ffi::TensorView q, tvm::ffi::TensorView s, host::SymbolicSize& R,
                                 host::SymbolicDevice& device) {
    using namespace host;
    TensorMatcher({R, kHidden}).with_dtype<fp8_e4m3_t>().with_device(device).verify(q);
    TensorMatcher({R, 1}).with_dtype<fp32_t>().with_device(device).verify(s);
  }

  /*!
   * \brief Plan A: RMSNorm then indexed scale/shift, every eager round kept.
   * \param out     bf16 [rows, kHidden]
   * \param x       bf16 [rows, kHidden], read-only
   * \param gamma   bf16 [kHidden] RMSNorm weight
   * \param scale   bf16 [groups, kHidden] rows with a shared row stride (a `chunk` view is fine)
   * \param shift   bf16 [groups, kHidden], same stride as `scale`
   * \param indices int32/int64 [rows] group per row
   */
  static void run(tvm::ffi::TensorView out, tvm::ffi::TensorView x, tvm::ffi::TensorView gamma,
                  tvm::ffi::TensorView scale, tvm::ffi::TensorView shift,
                  tvm::ffi::TensorView indices, double eps) {
    run_impl<false>(out, nullptr, x, gamma, scale, shift, indices, eps);
  }

  /*!
   * \brief Plan A with the modulated rows quantised per token for the fp8 GEMM.
   * \param q bf16-free: fp8 e4m3 [rows, kHidden]
   * \param s fp32 [rows, 1] per-token scales (amax / 448)
   */
  static void run_fp8(tvm::ffi::TensorView q, tvm::ffi::TensorView s, tvm::ffi::TensorView x,
                      tvm::ffi::TensorView gamma, tvm::ffi::TensorView scale, tvm::ffi::TensorView shift,
                      tvm::ffi::TensorView indices, double eps) {
    run_impl<true>(q, &s, x, gamma, scale, shift, indices, eps);
  }

  template <bool kFp8Out>
  static void run_impl(tvm::ffi::TensorView out, tvm::ffi::TensorView* s, tvm::ffi::TensorView x,
                       tvm::ffi::TensorView gamma, tvm::ffi::TensorView scale, tvm::ffi::TensorView shift,
                       tvm::ffi::TensorView indices, double eps) {
    using namespace host;
    SymbolicSize R{"rows"}, G{"groups"}, GS{"group_stride"};
    SymbolicDType idx_type;
    SymbolicDevice device;
    device.set_options<kDLCUDA>();
    TensorMatcher({R, kHidden}).with_dtype<bf16_t>().with_device(device).verify(x);
    if constexpr (kFp8Out) {
      verify_fp8_outputs(out, *s, R, device);
    } else {
      TensorMatcher({R, kHidden}).with_dtype<bf16_t>().with_device(device).verify(out);
    }
    TensorMatcher({kHidden}).with_dtype<bf16_t>().with_device(device).verify(gamma);
    TensorMatcher({G, kHidden})
        .with_strides({GS, 1})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(scale)
        .verify(shift);
    TensorMatcher({R}).with_dtype<int32_t, int64_t>(idx_type).with_device(device).verify(indices);
    const int64_t rows = R.unwrap();
    if (rows == 0) return;
    CHECK_HOST(GS.unwrap() % kVec == 0) << "modulation row stride must keep 8-byte alignment";
    verify_alignment({out.data_ptr(), x.data_ptr(), gamma.data_ptr(), scale.data_ptr(), shift.data_ptr()});
    CHECK_HOST(out.data_ptr() != x.data_ptr()) << "out must not alias x";
    const RowParams params{
        .out = out.data_ptr(),
        .s = kFp8Out ? static_cast<float*>(s->data_ptr()) : nullptr,
        .x = x.data_ptr(),
        .update = nullptr,
        .gate = nullptr,
        .gamma = gamma.data_ptr(),
        .scale = scale.data_ptr(),
        .shift = shift.data_ptr(),
        .indices = indices.data_ptr(),
        .group_stride = GS.unwrap(),
        .eps = static_cast<float>(eps),
    };
    launch<false, kFp8Out>(params, rows, idx_type.is_type<int32_t>(), device.unwrap());
  }

  /*!
   * \brief Plan B: gated residual add in place, then the Plan A chain on the result.
   * \param residual bf16 [rows, kHidden], becomes residual + gate[idx] * update
   * \param update   bf16 [rows, kHidden]
   * \param gate     bf16 [groups, kHidden], same stride as `scale`
   */
  static void run_gated(tvm::ffi::TensorView out, tvm::ffi::TensorView residual,
                        tvm::ffi::TensorView update, tvm::ffi::TensorView gate,
                        tvm::ffi::TensorView gamma, tvm::ffi::TensorView scale,
                        tvm::ffi::TensorView shift, tvm::ffi::TensorView indices, double eps) {
    run_gated_impl<false>(out, nullptr, residual, update, gate, gamma, scale, shift, indices, eps);
  }

  /// Plan B with the modulated rows quantised per token; see `run_fp8`.
  static void run_gated_fp8(tvm::ffi::TensorView q, tvm::ffi::TensorView s, tvm::ffi::TensorView residual,
                            tvm::ffi::TensorView update, tvm::ffi::TensorView gate,
                            tvm::ffi::TensorView gamma, tvm::ffi::TensorView scale,
                            tvm::ffi::TensorView shift, tvm::ffi::TensorView indices, double eps) {
    run_gated_impl<true>(q, &s, residual, update, gate, gamma, scale, shift, indices, eps);
  }

  template <bool kFp8Out>
  static void run_gated_impl(tvm::ffi::TensorView out, tvm::ffi::TensorView* s, tvm::ffi::TensorView residual,
                             tvm::ffi::TensorView update, tvm::ffi::TensorView gate,
                             tvm::ffi::TensorView gamma, tvm::ffi::TensorView scale,
                             tvm::ffi::TensorView shift, tvm::ffi::TensorView indices, double eps) {
    using namespace host;
    SymbolicSize R{"rows"}, G{"groups"}, GS{"group_stride"};
    SymbolicDType idx_type;
    SymbolicDevice device;
    device.set_options<kDLCUDA>();
    TensorMatcher({R, kHidden}).with_dtype<bf16_t>().with_device(device).verify(residual).verify(update);
    if constexpr (kFp8Out) {
      verify_fp8_outputs(out, *s, R, device);
    } else {
      TensorMatcher({R, kHidden}).with_dtype<bf16_t>().with_device(device).verify(out);
    }
    TensorMatcher({kHidden}).with_dtype<bf16_t>().with_device(device).verify(gamma);
    TensorMatcher({G, kHidden})
        .with_strides({GS, 1})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(scale)
        .verify(shift)
        .verify(gate);
    TensorMatcher({R}).with_dtype<int32_t, int64_t>(idx_type).with_device(device).verify(indices);
    const int64_t rows = R.unwrap();
    if (rows == 0) return;
    CHECK_HOST(GS.unwrap() % kVec == 0) << "modulation row stride must keep 8-byte alignment";
    verify_alignment({out.data_ptr(), residual.data_ptr(), update.data_ptr(), gate.data_ptr(),
                      gamma.data_ptr(), scale.data_ptr(), shift.data_ptr()});
    CHECK_HOST(out.data_ptr() != residual.data_ptr() && out.data_ptr() != update.data_ptr())
        << "out must not alias residual or update";
    CHECK_HOST(residual.data_ptr() != update.data_ptr()) << "residual must not alias update";
    const RowParams params{
        .out = out.data_ptr(),
        .s = kFp8Out ? static_cast<float*>(s->data_ptr()) : nullptr,
        .x = residual.data_ptr(),
        .update = update.data_ptr(),
        .gate = gate.data_ptr(),
        .gamma = gamma.data_ptr(),
        .scale = scale.data_ptr(),
        .shift = shift.data_ptr(),
        .indices = indices.data_ptr(),
        .group_stride = GS.unwrap(),
        .eps = static_cast<float>(eps),
    };
    launch<true, kFp8Out>(params, rows, idx_type.is_type<int32_t>(), device.unwrap());
  }
};

}  // namespace rmsnorm_indexed_modulate

}  // namespace sglang
