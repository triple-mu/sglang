// CUDA fast path for fused packed SwiGLU + per-token FP8 quantization
// (MiniMax-H3 fp8 W8A8 MLP): one pass over the [M, 2*D] fc1 output laid out
// as [gate | up] computes
//
//   act   = bf16(bf16(silu(gate)) * up)     (both aten rounding boundaries)
//   scale = amax(|act|) / 448               per row, fp32
//   q     = fp8_e4m3(act / scale)           clamped, satfinite
//
// Numerical contract: bitwise equal to the separated eager chain
// `sgl_per_token_quant_fp8(F.silu(gate) * up)` as built on SM90, i.e. the
// --use_fast_math reference documented in
// `ops/diffusion/common/fp8_quant_replica.py`.  Every non-trivially-rounded
// step replicates the exact instruction of the verified Triton replica
// (`activation/silu_mul_quant_triton.py`), whose PTX was diffed against the
// eager chain on H200:
//
//   sigmoid:   ex2.approx.f32(-x * rn(log2 e)); add.rn 1.0; div.full.f32
//   bf16 round: cvt.rn.bf16x2.f32 (RNE; NaN mantissa truncates -- the paired
//              form of the cvt.rn.bf16.f32 the eager stores execute)
//   amax:      max.f32 over |act|; the reference's per-element FTZ is folded
//              into the scale FMUL.FTZ input flush (a denormal can win the
//              max only when every element is denormal or zero, and the
//              input flush then yields the same +0 scale)
//   scale:     mul.rn.ftz.f32(amax, rn(1/448))  (flushed to +-0)
//   scale_inv: rcp.approx.f32(scale); zeroed when scale == 0 only under the
//              reference's warp-per-token dispatch (num_tokens >= sm * 16)
//   payload:   mul.rn.ftz.f32(scale_inv, act) -- the instruction the
//              fast-math reference executes; its input flush is the
//              replica's explicit FTZ(act) (scale_inv is never denormal:
//              scale <= fp32_max/448 < 2^126, so 1/scale > 2^-126, or it is
//              exactly 0/inf), and its output flush only turns denormal
//              payloads into +-0, which cvt rounds to the same +-0 fp8 --
//              then min/max.f32 clamp to +-448; cvt.rn.satfinite.e4m3x2.f32
//
// Unlike the two-pass Triton kernel this kernel reads the fc1 output once:
// one CTA owns a row and keeps the bf16 activation in registers between the
// amax reduction and the quantized store.  bf16 -> fp32 widenings are done
// with integer shifts on the packed halves to keep the quarter-rate F2F
// pipe out of the hot loop.

#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/cta.cuh>
#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <cstdint>
#include <limits>

namespace sglang {

namespace silu_mul_quant {

constexpr uint32_t kThreads = 512;
constexpr int kVec = 8;  // 16B of bf16 per load, 8B of fp8 per store
constexpr uintptr_t kAlignment = 16;

// Reference launcher constants (gemm/per_token_quant_fp8.cuh): the
// warp-per-token variant engages at sm_count * 2 * kTokensPerCTA rows and is
// the only variant that guards a zero scale.
constexpr uint32_t kReferenceTokensPerCTA = 8;

/// \brief `1 / (1 + exp(-g))` with the exact instruction sequence of the
/// verified eager silu chain (fast-exp `ex2.approx` and `div.full.f32`).
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

/// \brief Widen a packed bf16x2 to fp32 (exact) with integer shifts, keeping
/// the F2F conversion pipe free for the rounding cvts.
SGL_DEVICE float2 widen_bf16x2(uint32_t packed) {
  return {__uint_as_float(packed << 16), __uint_as_float(packed & 0xFFFF0000u)};
}

/// \brief RNE-round two fp32 values to a packed bf16x2; per-element semantics
/// (including NaN-mantissa truncation) match the `cvt.rn.bf16.f32` the eager
/// stores execute.
SGL_DEVICE uint32_t round_bf16x2_rn(float lo, float hi) {
  const __nv_bfloat162 packed = __float22bfloat162_rn({lo, hi});
  return reinterpret_cast<const uint32_t&>(packed);
}

/// \brief Both bf16 rounding boundaries of the eager `F.silu(gate) * up`
/// chain for two adjacent elements, packed as bf16x2.
SGL_DEVICE uint32_t silu_mul_act_pair_replica(uint32_t gate2, uint32_t up2) {
  const auto [g0, g1] = widen_bf16x2(gate2);
  const uint32_t silu2 =
      round_bf16x2_rn(__fmul_rn(sigmoid_replica(g0), g0), __fmul_rn(sigmoid_replica(g1), g1));
  const auto [s0, s1] = widen_bf16x2(silu2);
  const auto [u0, u1] = widen_bf16x2(up2);
  return round_bf16x2_rn(__fmul_rn(s0, u0), __fmul_rn(s1, u1));
}

/// \brief Row scale `amax / 448` as the fast-math reference computes it:
/// FMUL.FTZ by the compile-time constant rn(1/448).
SGL_DEVICE float scale_from_amax_replica(float amax) {
  const float kRcp448 = __uint_as_float(0x3B124925u);  // rn(1/448)
  float scale;
  asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(scale) : "f"(amax), "f"(kRcp448));
  return scale;
}

SGL_DEVICE float rcp_approx(float x) {
  float r;
  asm("rcp.approx.f32 %0, %1;" : "=f"(r) : "f"(x));
  return r;
}

/// \brief Clamped fp32 payload of one activation: the reference build's
/// FMUL.FTZ (input flush covers the replica's explicit FTZ of the value,
/// see the header comment), then clamp to +-448 with minNum/maxNum NaN
/// semantics.
SGL_DEVICE float quant_payload(float act, float scale_inv) {
  float v;
  asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(v) : "f"(scale_inv), "f"(act));
  v = fminf(v, 448.0f);
  return fmaxf(v, -448.0f);
}

/**
 * \brief One CTA per row: compute the bf16 SwiGLU activation into registers,
 * CTA-reduce the row amax, then quantize the held activation to fp8.
 *
 * \tparam kVecsPerThread Row 16B-vector count divided by kThreads, rounded up.
 * \param q Output fp8 payload, [num_tokens, hidden] contiguous.
 * \param s Output fp32 scales, [num_tokens, 1] contiguous.
 * \param x Input [num_tokens, 2 * hidden] bf16 rows as [gate | up].
 * \param hidden Activation width D.
 * \param x_row_stride Row stride of `x` in elements.
 * \param zero_guard Nonzero when the reference dispatch zeroes 1/scale at
 *                   scale == 0 (warp-per-token regime).
 */
template <int kVecsPerThread>
__global__ __launch_bounds__(kThreads) void silu_mul_quant_kernel(
    fp8_e4m3_t* __restrict__ q,
    float* __restrict__ s,
    const bf16_t* __restrict__ x,
    uint32_t hidden,
    int64_t x_row_stride,
    uint32_t zero_guard) {
  using namespace device;
  using InVec = AlignedVector<bf16_t, kVec>;
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
      // Evict-first loads: each element is read exactly once, so keeping the
      // row out of L2 leaves the cache to other CTAs' in-flight lines.
      const uint4 gate_bits = __ldcs(reinterpret_cast<const uint4*>(gate_row) + vec_id);
      const uint4 up_bits = __ldcs(reinterpret_cast<const uint4*>(up_row) + vec_id);
      const auto* gate2 = reinterpret_cast<const uint32_t*>(&gate_bits);
      const auto* up2 = reinterpret_cast<const uint32_t*>(&up_bits);
#pragma unroll
      for (int i = 0; i < kVec / 2; ++i) {
        const uint32_t act2 = silu_mul_act_pair_replica(gate2[i], up2[i]);
        acts[k][i] = act2;
        const auto [a0, a1] = widen_bf16x2(act2);
        // |NaN| stays NaN and is dropped by max.f32's minNum semantics.
        amax = fmaxf(amax, fabsf(a0));
        amax = fmaxf(amax, fabsf(a1));
      }
    }
  }

  __shared__ float scratch[kThreads / kWarpThreads];
  cta::reduce_max(amax, scratch);
  __syncthreads();
  const float scale = scale_from_amax_replica(scratch[0]);
  if (threadIdx.x == 0) {
    s[row] = scale;
  }
  float scale_inv = rcp_approx(scale);
  if (zero_guard != 0 && scale == 0.0f) {
    scale_inv = 0.0f;
  }

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

/**
 * \brief Validate and launch the fused SwiGLU + per-token FP8 quantization.
 *
 * \tparam kVecsPerThread Compile-time per-thread vector count; must equal
 *                        `ceil(hidden / (8 * 512))` for the given input.
 */
template <int kVecsPerThread>
struct SiluMulQuantKernel {
  static_assert(kVecsPerThread >= 1 && kVecsPerThread <= 8);

  static void run(tvm::ffi::TensorView x, tvm::ffi::TensorView q, tvm::ffi::TensorView s) {
    using namespace host;
    auto M = SymbolicSize{"num_tokens"};
    auto P = SymbolicSize{"packed_width"};
    auto D = SymbolicSize{"hidden"};
    auto S = SymbolicSize{"x_row_stride"};
    auto device = SymbolicDevice{};
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
    CHECK_HOST(x_row_stride % kVec == 0) << "x row stride must keep 16B alignment, got " << x_row_stride;
    CHECK_HOST(reinterpret_cast<uintptr_t>(x.data_ptr()) % kAlignment == 0) << "x must be 16B aligned";
    CHECK_HOST(rows <= std::numeric_limits<uint32_t>::max()) << "rows out of range: " << rows;
    if (rows == 0) {
      return;
    }

    // Mirror launch_per_token_quant_fp8's dispatch predicate: only the
    // warp-per-token reference variant guards the zero-scale reciprocal.
    const auto dl_device = device.unwrap();
    const uint32_t sm_count = runtime::get_sm_count(dl_device.device_id);
    const uint32_t zero_guard = rows >= static_cast<int64_t>(sm_count) * 2 * kReferenceTokensPerCTA ? 1u : 0u;

    LaunchKernel(static_cast<uint32_t>(rows), kThreads, dl_device)(
        silu_mul_quant_kernel<kVecsPerThread>,
        static_cast<fp8_e4m3_t*>(q.data_ptr()),
        static_cast<fp32_t*>(s.data_ptr()),
        static_cast<const bf16_t*>(x.data_ptr()),
        static_cast<uint32_t>(hidden),
        x_row_stride,
        zero_guard);
  }
};

}  // namespace silu_mul_quant

}  // namespace sglang
