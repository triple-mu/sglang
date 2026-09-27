// Fused packed SwiGLU + per-token FP8 quantisation for the MiniMax-H3 fp8 MLP:
// one pass over the [M, 2 * D] fc1 output laid out as [gate | up] computes
//
//   act   = bf16(bf16(silu(gate)) * up)   (the eager activation's two bf16 rounds)
//   scale = amax(|act|) / 448              per row, fp32
//   q     = fp8_e4m3(act / scale)          saturating; a zero row quantises to zeros
//
// so fc2 consumes (q, scale) through the per-row-scaled fp8 GEMM instead of a
// bf16 activation plus a separate per-tensor quantisation pass. One CTA owns a
// row and keeps the bf16 activation in registers between the amax reduction
// and the quantised store. Ported from origin/minimax-h3-fusion-dit; the
// reference-dispatch replication of its zero-scale handling is dropped in
// favour of always guarding the reciprocal.

#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/cta.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <cstdint>
#include <limits>

namespace sglang {

namespace silu_mul_quant {

constexpr uint32_t kThreads = 512;
constexpr int kVec = 8;  // 16 B of bf16 per load, 8 B of fp8 per store
constexpr uintptr_t kAlignment = 16;
constexpr float kFp8Max = 448.0f;

/// `1 / (1 + exp(-g))` with the fast-exp sequence the eager SiLU compiles to.
SGL_DEVICE float sigmoid_replica(float g) {
  const float kLog2E = __uint_as_float(0x3FB8AA3Bu);  // rn(log2 e)
  const float t = __fmul_rn(__fsub_rn(0.0f, g), kLog2E);
  float e;
  asm("ex2.approx.f32 %0, %1;" : "=f"(e) : "f"(t));
  const float d = __fadd_rn(e, 1.0f);
  float r;
  asm("div.full.f32 %0, %1, %2;" : "=f"(r) : "f"(1.0f), "f"(d));
  return r;
}

SGL_DEVICE float2 widen_bf16x2(uint32_t packed) {
  return {__uint_as_float(packed << 16), __uint_as_float(packed & 0xFFFF0000u)};
}

SGL_DEVICE uint32_t round_bf16x2_rn(float lo, float hi) {
  const __nv_bfloat162 packed = __float22bfloat162_rn({lo, hi});
  return reinterpret_cast<const uint32_t&>(packed);
}

/// Both bf16 rounds of the eager `silu(gate) * up` for two adjacent elements.
SGL_DEVICE uint32_t silu_mul_act_pair(uint32_t gate2, uint32_t up2) {
  const auto [g0, g1] = widen_bf16x2(gate2);
  const uint32_t silu2 =
      round_bf16x2_rn(__fmul_rn(sigmoid_replica(g0), g0), __fmul_rn(sigmoid_replica(g1), g1));
  const auto [s0, s1] = widen_bf16x2(silu2);
  const auto [u0, u1] = widen_bf16x2(up2);
  return round_bf16x2_rn(__fmul_rn(s0, u0), __fmul_rn(s1, u1));
}

SGL_DEVICE float quant_payload(float act, float scale_inv) {
  const float v = __fmul_rn(scale_inv, act);
  return fmaxf(fminf(v, kFp8Max), -kFp8Max);
}

/*!
 * \brief One CTA per row: bf16 SwiGLU into registers, CTA amax, fp8 store.
 * \tparam kVecsPerThread ceil(D / (8 * kThreads)).
 */
template <int kVecsPerThread>
__global__ __launch_bounds__(kThreads) void silu_mul_quant_kernel(
    fp8_e4m3_t* __restrict__ q, float* __restrict__ s, const bf16_t* __restrict__ x,
    uint32_t hidden, int64_t x_row_stride) {
  using namespace device;
  using OutVec = AlignedVector<fp8_e4m3_t, kVec>;

  const uint32_t row = blockIdx.x;
  const bf16_t* gate_row = x + static_cast<int64_t>(row) * x_row_stride;
  const bf16_t* up_row = gate_row + hidden;
  const uint32_t num_vecs = hidden / kVec;

  uint32_t acts[kVecsPerThread][kVec / 2];  // bf16x2 pairs
  float amax = 0.0f;
#pragma unroll
  for (int k = 0; k < kVecsPerThread; ++k) {
    const uint32_t vec_id = threadIdx.x + k * kThreads;
    if (vec_id < num_vecs) {
      // Each element is read once: evict-first keeps the row out of L2.
      const uint4 gate_bits = __ldcs(reinterpret_cast<const uint4*>(gate_row) + vec_id);
      const uint4 up_bits = __ldcs(reinterpret_cast<const uint4*>(up_row) + vec_id);
      const auto* gate2 = reinterpret_cast<const uint32_t*>(&gate_bits);
      const auto* up2 = reinterpret_cast<const uint32_t*>(&up_bits);
#pragma unroll
      for (int i = 0; i < kVec / 2; ++i) {
        const uint32_t act2 = silu_mul_act_pair(gate2[i], up2[i]);
        acts[k][i] = act2;
        const auto [a0, a1] = widen_bf16x2(act2);
        amax = fmaxf(amax, fabsf(a0));
        amax = fmaxf(amax, fabsf(a1));
      }
    }
  }

  __shared__ float scratch[kThreads / kWarpThreads];
  cta::reduce_max(amax, scratch);
  __syncthreads();
  const float scale = __fdiv_rn(scratch[0], kFp8Max);
  if (threadIdx.x == 0) s[row] = scale;
  const float scale_inv = scale == 0.0f ? 0.0f : __frcp_rn(scale);

  fp8_e4m3_t* q_row = q + static_cast<int64_t>(row) * hidden;
#pragma unroll
  for (int k = 0; k < kVecsPerThread; ++k) {
    const uint32_t vec_id = threadIdx.x + k * kThreads;
    if (vec_id < num_vecs) {
      OutVec out_vec;
      auto* pairs = reinterpret_cast<__nv_fp8x2_storage_t*>(out_vec.data());
#pragma unroll
      for (int i = 0; i < kVec / 2; ++i) {
        const auto [a0, a1] = widen_bf16x2(acts[k][i]);
        const float2 payload{quant_payload(a0, scale_inv), quant_payload(a1, scale_inv)};
        pairs[i] = __nv_cvt_float2_to_fp8x2(payload, __NV_SATFINITE, __NV_E4M3);
      }
      out_vec.store(q_row, vec_id);
    }
  }
}

/*!
 * \brief Validate and launch the fused SwiGLU + per-token FP8 quantisation.
 * \tparam kVecsPerThread Must equal ceil(D / (8 * 512)) for the input.
 */
template <int kVecsPerThread>
struct SiluMulQuantKernel {
  static_assert(kVecsPerThread >= 1 && kVecsPerThread <= 8);

  /*!
   * \param x [M, 2 * D] bf16 rows as [gate | up]; any row stride with 16-byte alignment
   * \param q [M, D] fp8 e4m3 output
   * \param s [M, 1] fp32 per-row scales
   */
  static void run(tvm::ffi::TensorView x, tvm::ffi::TensorView q, tvm::ffi::TensorView s) {
    using namespace host;
    SymbolicSize M{"num_tokens"}, P{"packed_width"}, D{"hidden"}, S{"x_row_stride"};
    SymbolicDevice device;
    device.set_options<kDLCUDA>();
    TensorMatcher({M, P}).with_strides({S, 1}).with_dtype<bf16_t>().with_device(device).verify(x);
    TensorMatcher({M, D}).with_dtype<fp8_e4m3_t>().with_device(device).verify(q);
    TensorMatcher({M, 1}).with_dtype<fp32_t>().with_device(device).verify(s);

    const int64_t rows = M.unwrap();
    const int64_t hidden = D.unwrap();
    const int64_t x_row_stride = S.unwrap();
    CHECK_HOST(P.unwrap() == 2 * hidden) << "x width must be 2 * hidden, got " << P.unwrap();
    CHECK_HOST(hidden % kVec == 0) << "hidden must be divisible by " << kVec << ", got " << hidden;
    CHECK_HOST(div_ceil(hidden / kVec, static_cast<int64_t>(kThreads)) == kVecsPerThread)
        << "hidden " << hidden << " does not match kVecsPerThread " << kVecsPerThread;
    CHECK_HOST(x_row_stride % kVec == 0) << "x row stride must keep 16-byte alignment";
    CHECK_HOST(reinterpret_cast<uintptr_t>(x.data_ptr()) % kAlignment == 0) << "x must be 16B aligned";
    CHECK_HOST(rows <= std::numeric_limits<uint32_t>::max()) << "rows out of range: " << rows;
    if (rows == 0) return;

    LaunchKernel(static_cast<uint32_t>(rows), kThreads, device.unwrap())(
        silu_mul_quant_kernel<kVecsPerThread>, static_cast<fp8_e4m3_t*>(q.data_ptr()),
        static_cast<fp32_t*>(s.data_ptr()), static_cast<const bf16_t*>(x.data_ptr()),
        static_cast<uint32_t>(hidden), x_row_stride);
  }
};

}  // namespace silu_mul_quant

}  // namespace sglang
