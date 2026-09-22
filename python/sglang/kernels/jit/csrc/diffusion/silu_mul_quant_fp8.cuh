// Gated-FFN activation producer: silu(gate) * up in fp32 with a dynamic
// per-token FP8 E4M3 epilogue, so the down-projection GEMM input skips the
// 16-bit round trip of the eager silu / mul / quantize chain.
#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/cta.cuh>
#include <sgl_kernel/math.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include "minimax_h3_vae_nvfp4.cuh"

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang {

namespace silu_mul_quant_fp8 {

/// One CTA per row; the fp32 products stay in registers until the row amax is known.
constexpr uint32_t kBlockSize = 256;
constexpr uint32_t kNumWarps = kBlockSize / device::kWarpThreads;
constexpr int64_t kAlignment = 16;

/**
 * \brief `q, s = fp8(silu(gate) * up)` for one `[gate | up]` row per CTA.
 *
 * Both halves are widened to fp32 and the product is never rounded to 16
 * bits; `math::exp` is the precise `expf`, the same formula as aten's fp32
 * SiLU. The FP8 epilogue is `per_token_quant_fp8`'s: scale = amax /
 * FP8_E4M3_MAX, codes clamped to +-FP8_E4M3_MAX, and an all-zero row writes
 * scale 0 with zero codes.
 *
 * \tparam T        Input element type: fp16_t | bf16_t
 * \tparam kWidth   Output row width; the input row holds 2 * kWidth elements
 * \tparam kVecSize Elements per vector load; kWidth must be a multiple of kBlockSize * kVecSize
 * \param x      [rows, 2 * kWidth] laid out as [gate | up]
 * \param out_q  [rows, kWidth] fp8 e4m3, written
 * \param out_s  [rows] fp32 per-row scales, written
 */
template <typename T, uint32_t kWidth, uint32_t kVecSize>
__global__ void __launch_bounds__(kBlockSize)
    silu_mul_quant_fp8_kernel(const T* __restrict__ x, fp8_e4m3_t* __restrict__ out_q, float* __restrict__ out_s) {
  using namespace device;
  static_assert(kWidth % (kBlockSize * kVecSize) == 0, "row width must be a multiple of kBlockSize * kVecSize");
  constexpr uint32_t kSteps = kWidth / (kBlockSize * kVecSize);
  using InVec = AlignedVector<T, kVecSize>;
  using OutVec = AlignedVector<fp8_e4m3_t, kVecSize>;

  __shared__ float smem_amax[kNumWarps];

  const int64_t row = blockIdx.x;
  const T* gate_row = x + row * (2 * static_cast<int64_t>(kWidth));
  const T* up_row = gate_row + kWidth;
  fp8_e4m3_t* out_row = out_q + row * static_cast<int64_t>(kWidth);

  float values[kSteps * kVecSize];
  float amax = 0.0f;
#pragma unroll
  for (uint32_t step = 0; step < kSteps; ++step) {
    const uint32_t chunk = step * kBlockSize + threadIdx.x;
    InVec gate_vec;
    InVec up_vec;
    gate_vec.load(gate_row, chunk);
    up_vec.load(up_row, chunk);
#pragma unroll
    for (uint32_t i = 0; i < kVecSize; ++i) {
      const float gate = cast<fp32_t>(gate_vec[i]);
      const float value = (gate / (1.0f + math::exp(-gate))) * cast<fp32_t>(up_vec[i]);
      values[step * kVecSize + i] = value;
      amax = math::max(amax, math::abs(value));
    }
  }

  cta::reduce_max(amax, smem_amax);
  __syncthreads();
  const float scale = smem_amax[0] / math::FP8_E4M3_MAX;
  if (threadIdx.x == 0) {
    out_s[row] = scale;
  }
  const float scale_inv = scale == 0.0f ? 0.0f : 1.0f / scale;
#pragma unroll
  for (uint32_t step = 0; step < kSteps; ++step) {
    OutVec q_vec;
#pragma unroll
    for (uint32_t i = 0; i < kVecSize; ++i) {
      const float value = values[step * kVecSize + i] * scale_inv;
      q_vec[i] = static_cast<fp8_e4m3_t>(math::max(math::min(value, math::FP8_E4M3_MAX), -math::FP8_E4M3_MAX));
    }
    q_vec.store(out_row, step * kBlockSize + threadIdx.x);
  }
}

/**
 * \brief Validate the `[gate | up]` rows and launch `silu_mul_quant_fp8_kernel`.
 *
 * \tparam T      Input element type: fp16_t | bf16_t
 * \tparam kWidth Output row width; a multiple of kBlockSize * (16 / sizeof(T))
 */
template <typename T, uint32_t kWidth>
struct SiluMulQuantFp8Kernel {
  static constexpr uint32_t kVecSize = 16 / sizeof(T);

  /**
   * \param x      [rows, 2 * kWidth] T as [gate | up], contiguous, 16-byte aligned
   * \param out_q  [rows, kWidth] fp8 e4m3, written
   * \param out_s  [rows, 1] fp32 per-row scales, written
   */
  static void run(tvm::ffi::TensorView x, tvm::ffi::TensorView out_q, tvm::ffi::TensorView out_s) {
    using namespace host;
    auto rows = SymbolicSize{"rows"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({rows, 2 * kWidth}).with_dtype<T>().with_device(device).ensure_alignment(kAlignment).verify(x);
    TensorMatcher({rows, kWidth})
        .with_dtype<fp8_e4m3_t>()
        .with_device(device)
        .ensure_alignment(kAlignment)
        .verify(out_q);
    TensorMatcher({rows, 1}).with_dtype<fp32_t>().with_device(device).verify(out_s);

    const int64_t num_rows = rows.unwrap();
    CHECK_HOST(num_rows > 0 && num_rows <= UINT32_MAX)
        << "silu_mul_quant_fp8: rows must be in (0, UINT32_MAX], got " << num_rows;
    LaunchKernel(static_cast<uint32_t>(num_rows), kBlockSize, device.unwrap())(
        silu_mul_quant_fp8_kernel<T, kWidth, kVecSize>,
        static_cast<const T*>(x.data_ptr()),
        static_cast<fp8_e4m3_t*>(out_q.data_ptr()),
        static_cast<float*>(out_s.data_ptr()));
  }
};

/**
 * \brief `q4, sf = nvfp4(silu(gate + gate_bias) * (up + up_bias))` for one `[gate | up]` row per CTA.
 *
 * Same fp32 activation as `silu_mul_quant_fp8_kernel`; the biases of an NVFP4
 * up-projection GEMM (whose epilogue cannot add them) are folded in here. The
 * epilogue writes packed E2M1 codes and 128x4-swizzled E4M3 block scales for
 * `flashinfer.mm_fp4`, using the calibrated `global_scale` of the consuming
 * down-projection's input. Two consecutive lanes hold one 16-element block.
 *
 * \tparam T        Input element type: fp16_t | bf16_t
 * \tparam kWidth   Output row width; the input row holds 2 * kWidth elements
 * \tparam kVecSize Elements per vector load; must be 8 so that two lanes cover a scale block
 */
template <typename T, uint32_t kWidth, uint32_t kVecSize>
__global__ void __launch_bounds__(kBlockSize) silu_mul_quant_fp4_kernel(
    const T* __restrict__ x,
    const float* __restrict__ gate_bias,
    const float* __restrict__ up_bias,
    uint8_t* __restrict__ out_q4,
    uint8_t* __restrict__ out_sf,
    float global_scale) {
  using namespace device;
  namespace nvfp4 = minimax_h3_vae_nvfp4;
  static_assert(kWidth % (kBlockSize * kVecSize) == 0, "row width must be a multiple of kBlockSize * kVecSize");
  static_assert(kVecSize * 2 == nvfp4::kBlock, "two lanes must cover one 16-element scale block");
  constexpr uint32_t kSteps = kWidth / (kBlockSize * kVecSize);
  constexpr uint32_t kNumBlocks = kWidth / nvfp4::kBlock;
  using InVec = AlignedVector<T, kVecSize>;
  using BiasVec = AlignedVector<float, 4>;

  const int64_t row = blockIdx.x;
  const T* gate_row = x + row * (2 * static_cast<int64_t>(kWidth));
  const T* up_row = gate_row + kWidth;
  uint8_t* q_row = out_q4 + row * static_cast<int64_t>(kWidth / 2);

#pragma unroll
  for (uint32_t step = 0; step < kSteps; ++step) {
    const uint32_t chunk = step * kBlockSize + threadIdx.x;
    InVec gate_vec;
    InVec up_vec;
    gate_vec.load(gate_row, chunk);
    up_vec.load(up_row, chunk);
    BiasVec gate_bias_vec[2];
    BiasVec up_bias_vec[2];
#pragma unroll
    for (uint32_t half = 0; half < 2; ++half) {
      gate_bias_vec[half].load(gate_bias, chunk * 2 + half);
      up_bias_vec[half].load(up_bias, chunk * 2 + half);
    }
    float values[kVecSize];
    float block_amax = 0.0f;
#pragma unroll
    for (uint32_t i = 0; i < kVecSize; ++i) {
      const float gate = cast<fp32_t>(gate_vec[i]) + gate_bias_vec[i / 4][i % 4];
      const float up = cast<fp32_t>(up_vec[i]) + up_bias_vec[i / 4][i % 4];
      values[i] = (gate / (1.0f + math::exp(-gate))) * up;
      block_amax = math::max(block_amax, math::abs(values[i]));
    }
    block_amax = math::max(block_amax, __shfl_xor_sync(0xffffffffu, block_amax, 1));
    const fp8_e4m3_t sf = nvfp4::block_scale(block_amax, global_scale);
    const float multiplier = nvfp4::quant_multiplier(sf, global_scale);
    if ((threadIdx.x & 1u) == 0u) {
      out_sf[nvfp4::sf_offset_128x4(row, chunk >> 1, kNumBlocks)] = *reinterpret_cast<const uint8_t*>(&sf);
    }
    uint32_t packed = 0;
#pragma unroll
    for (uint32_t i = 0; i < kVecSize; i += 2) {
      packed |= static_cast<uint32_t>(nvfp4::pack2(values[i], values[i + 1], multiplier)) << (4 * i);
    }
    reinterpret_cast<uint32_t*>(q_row)[chunk] = packed;
  }
}

/**
 * \brief Validate the rows and biases and launch `silu_mul_quant_fp4_kernel`.
 *
 * \tparam T      Input element type: fp16_t | bf16_t
 * \tparam kWidth Output row width; a multiple of kBlockSize * 8
 */
template <typename T, uint32_t kWidth>
struct SiluMulQuantFp4Kernel {
  static constexpr uint32_t kVecSize = 16 / sizeof(T);
  static_assert(kVecSize == 8, "the NVFP4 gated activation is written for 16-bit inputs");

  /**
   * \param x             [rows, 2 * kWidth] T as [gate | up], contiguous, 16-byte aligned
   * \param gate_bias     [kWidth] fp32
   * \param up_bias       [kWidth] fp32
   * \param out_q4        [rows, kWidth / 2] uint8 packed E2M1, written
   * \param out_sf        [>= ceil(rows / 128) * 128, kWidth / 16] uint8 E4M3, swizzled, written (valid rows only)
   * \param global_scale  NVFP4 tensor scale of the produced activation
   */
  static void run(
      tvm::ffi::TensorView x,
      tvm::ffi::TensorView gate_bias,
      tvm::ffi::TensorView up_bias,
      tvm::ffi::TensorView out_q4,
      tvm::ffi::TensorView out_sf,
      double global_scale) {
    using namespace host;
    namespace nvfp4 = minimax_h3_vae_nvfp4;
    auto rows = SymbolicSize{"rows"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({rows, 2 * kWidth}).with_dtype<T>().with_device(device).ensure_alignment(kAlignment).verify(x);
    TensorMatcher({kWidth}).with_dtype<fp32_t>().with_device(device).ensure_alignment(kAlignment).verify(gate_bias);
    TensorMatcher({kWidth}).with_dtype<fp32_t>().with_device(device).ensure_alignment(kAlignment).verify(up_bias);
    TensorMatcher({rows, kWidth / 2}).with_dtype<uint8_t>().with_device(device).ensure_alignment(kAlignment).verify(out_q4);
    auto sf_rows = SymbolicSize{"sf_rows"};
    TensorMatcher({sf_rows, kWidth / nvfp4::kBlock}).with_dtype<uint8_t>().with_device(device).verify(out_sf);

    const int64_t num_rows = rows.unwrap();
    CHECK_HOST(num_rows > 0 && num_rows <= UINT32_MAX)
        << "silu_mul_quant_fp4: rows must be in (0, UINT32_MAX], got " << num_rows;
    const int64_t needed = (num_rows + nvfp4::kScaleRowTile - 1) / nvfp4::kScaleRowTile * nvfp4::kScaleRowTile;
    CHECK_HOST(sf_rows.unwrap() >= needed)
        << "silu_mul_quant_fp4: NVFP4 scale tensor needs " << needed << " rows, got " << sf_rows.unwrap();
    CHECK_HOST(global_scale > 0.0) << "silu_mul_quant_fp4: NVFP4 global scale must be positive";
    LaunchKernel(static_cast<uint32_t>(num_rows), kBlockSize, device.unwrap())(
        silu_mul_quant_fp4_kernel<T, kWidth, kVecSize>,
        static_cast<const T*>(x.data_ptr()),
        static_cast<const float*>(gate_bias.data_ptr()),
        static_cast<const float*>(up_bias.data_ptr()),
        static_cast<uint8_t*>(out_q4.data_ptr()),
        static_cast<uint8_t*>(out_sf.data_ptr()),
        static_cast<float>(global_scale));
  }
};

}  // namespace silu_mul_quant_fp8

}  // namespace sglang
