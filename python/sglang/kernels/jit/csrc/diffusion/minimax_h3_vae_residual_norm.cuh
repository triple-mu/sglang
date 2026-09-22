// MiniMax-H3 VAE decoder row producers: fp32 residual update + RMSNorm /
// LayerNorm, with a dynamic per-token FP8 E4M3 epilogue for the GEMM inputs.
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

namespace minimax_h3_vae_residual_norm {

/// \brief Which epilogue `residual_norm_kernel` computes on each row.
enum class NormMode : int32_t {
  kRmsNormFp8 = 0,          ///< q, s = fp8(RMSNorm(x) * weight)
  kResidualRmsNormFp8 = 1,  ///< r = fma(projected, layer_scale, x); q, s = fp8(RMSNorm(r) * weight)
  kResidualLayerNorm = 2,   ///< out = LayerNorm(fma(projected, layer_scale, x)) * weight + bias
  kResidualRmsNormFp4 = 3,  ///< r = fma(projected, layer_scale, x); q4, sf = nvfp4(RMSNorm(r) * weight)
};

/// One CTA per row; the fp32 row stays in registers between the two passes.
constexpr uint32_t kBlockSize = 256;
constexpr uint32_t kNumWarps = kBlockSize / device::kWarpThreads;
/// Elements per thread per step. Every stream but `x` / `projected` is fp32
/// (weights, residual, LayerNorm output), so 16-byte vectors on those fix the
/// chunk at four elements; a 16-bit `x` / `projected` then loads 8 bytes.
constexpr uint32_t kVecSize = 16 / sizeof(fp32_t);
constexpr int64_t kAlignment = 16;

/// \brief Row pointers for one launch; pointers a mode does not read are null.
template <typename T, typename U>
struct Params {
  const T* __restrict__ x;                ///< [rows, kWidth]
  const U* __restrict__ projected;        ///< [rows, kWidth]; residual modes only
  const float* __restrict__ projected_bias;  ///< [kWidth] added to `projected` first, or nullptr; residual modes
  const float* __restrict__ layer_scale;  ///< [kWidth]; residual modes only
  const float* __restrict__ weight;       ///< [kWidth]
  const float* __restrict__ bias;         ///< [kWidth]; kResidualLayerNorm only
  float* __restrict__ residual_out;       ///< [rows, kWidth]; kResidualRmsNormFp8 only
  fp8_e4m3_t* __restrict__ out_q;         ///< [rows, kWidth]; FP8 modes only
  float* __restrict__ out_s;              ///< [rows]; FP8 modes only
  float* __restrict__ out;                ///< [rows, kWidth]; kResidualLayerNorm only
  uint8_t* __restrict__ out_q4;           ///< [rows, kWidth / 2] packed E2M1; kResidualRmsNormFp4 only
  uint8_t* __restrict__ out_sf;           ///< [rows_pad, kWidth / 16] swizzled E4M3; kResidualRmsNormFp4 only
  float global_scale;                     ///< NVFP4 tensor scale of the produced activation
  float eps;
};

/**
 * \brief Residual update + RMSNorm / LayerNorm (+ dynamic per-token FP8), one row per CTA.
 *
 * Statistics accumulate in fp32 over the fp32 row; LayerNorm uses the two-pass
 * centered variance; both norms scale by `rsqrt(var + eps)`. The FP8 epilogue
 * is `per_token_quant_fp8`'s: scale = amax / FP8_E4M3_MAX, codes clamped to
 * +-FP8_E4M3_MAX, and an all-zero row writes scale 0 with zero codes.
 *
 * \tparam T      Element type of `x`: fp32_t | fp16_t
 * \tparam U      Element type of `projected`: fp32_t | fp16_t (unused for kRmsNormFp8)
 * \tparam kWidth Row width; a multiple of kBlockSize * kVecSize
 * \tparam kMode  Epilogue selection
 * \param params  Row pointers and eps
 */
template <typename T, typename U, uint32_t kWidth, NormMode kMode>
__global__ void __launch_bounds__(kBlockSize) residual_norm_kernel(const Params<T, U> __grid_constant__ params) {
  using namespace device;
  static_assert(kWidth % (kBlockSize * kVecSize) == 0, "row width must be a multiple of kBlockSize * kVecSize");
  constexpr uint32_t kSteps = kWidth / (kBlockSize * kVecSize);
  constexpr uint32_t kPerThread = kSteps * kVecSize;
  constexpr bool kResidual = kMode != NormMode::kRmsNormFp8;
  constexpr bool kLayerNorm = kMode == NormMode::kResidualLayerNorm;
  constexpr bool kFp4 = kMode == NormMode::kResidualRmsNormFp4;
  constexpr bool kResidualOut = kMode == NormMode::kResidualRmsNormFp8 || kFp4;
  static_assert(!kFp4 || kVecSize == 4, "the NVFP4 epilogue groups four lanes into one 16-element block");
  using XVec = AlignedVector<T, kVecSize>;
  using PVec = AlignedVector<U, kVecSize>;
  using FVec = AlignedVector<float, kVecSize>;
  using QVec = AlignedVector<fp8_e4m3_t, kVecSize>;

  // One buffer per reduction, so no reduction waits on the readers of another.
  __shared__ float smem_sum[kNumWarps];
  __shared__ float smem_sum_sq[kNumWarps];
  __shared__ float smem_amax[kNumWarps];

  const int64_t row_offset = static_cast<int64_t>(blockIdx.x) * kWidth;

  float values[kPerThread];
  float sum = 0.0f;
  float sum_sq = 0.0f;
#pragma unroll
  for (uint32_t step = 0; step < kSteps; ++step) {
    const uint32_t chunk = step * kBlockSize + threadIdx.x;
    XVec x_vec;
    x_vec.load(params.x + row_offset, chunk);
    [[maybe_unused]] PVec projected_vec;
    [[maybe_unused]] FVec layer_scale_vec;
    [[maybe_unused]] FVec projected_bias_vec;
    [[maybe_unused]] bool has_bias = false;
    if constexpr (kResidual) {
      projected_vec.load(params.projected + row_offset, chunk);
      layer_scale_vec.load(params.layer_scale, chunk);
      // A GEMM whose epilogue cannot add the bias (NVFP4) leaves it to the consumer.
      has_bias = params.projected_bias != nullptr;
      if (has_bias) {
        projected_bias_vec.load(params.projected_bias, chunk);
      }
    }
    [[maybe_unused]] FVec residual_vec;
#pragma unroll
    for (uint32_t i = 0; i < kVecSize; ++i) {
      float value = cast<fp32_t>(x_vec[i]);
      if constexpr (kResidual) {
        float projected = cast<fp32_t>(projected_vec[i]);
        if (has_bias) {
          projected += projected_bias_vec[i];
        }
        value = fmaf(projected, layer_scale_vec[i], value);
      }
      values[step * kVecSize + i] = value;
      sum += value;
      sum_sq = fmaf(value, value, sum_sq);
      if constexpr (kResidualOut) {
        residual_vec[i] = value;
      }
    }
    if constexpr (kResidualOut) {
      residual_vec.store(params.residual_out + row_offset, chunk);
    }
  }

  float mean = 0.0f;
  if constexpr (kLayerNorm) {
    cta::reduce_sum(sum, smem_sum);
    __syncthreads();
    mean = smem_sum[0] / static_cast<float>(kWidth);
    sum_sq = 0.0f;
#pragma unroll
    for (uint32_t j = 0; j < kPerThread; ++j) {
      const float centered = values[j] - mean;
      sum_sq = fmaf(centered, centered, sum_sq);
    }
  }
  cta::reduce_sum(sum_sq, smem_sum_sq);
  __syncthreads();
  const float variance = smem_sum_sq[0] / static_cast<float>(kWidth);
  const float inv_std = math::rsqrt(variance + params.eps);

  float amax = 0.0f;
#pragma unroll
  for (uint32_t step = 0; step < kSteps; ++step) {
    const uint32_t chunk = step * kBlockSize + threadIdx.x;
    FVec weight_vec;
    weight_vec.load(params.weight, chunk);
    [[maybe_unused]] FVec bias_vec;
    [[maybe_unused]] FVec out_vec;
    if constexpr (kLayerNorm) {
      bias_vec.load(params.bias, chunk);
    }
#pragma unroll
    for (uint32_t i = 0; i < kVecSize; ++i) {
      const uint32_t j = step * kVecSize + i;
      if constexpr (kLayerNorm) {
        out_vec[i] = (values[j] - mean) * inv_std * weight_vec[i] + bias_vec[i];
      } else {
        values[j] = values[j] * inv_std * weight_vec[i];
        amax = math::max(amax, math::abs(values[j]));
      }
    }
    if constexpr (kLayerNorm) {
      out_vec.store(params.out + row_offset, chunk);
    }
  }

  if constexpr (kFp4) {
    // Four consecutive lanes of one step hold the 16 elements of one scale block.
    namespace nvfp4 = minimax_h3_vae_nvfp4;
    constexpr uint32_t kNumBlocks = kWidth / nvfp4::kBlock;
    uint8_t* q_row = params.out_q4 + static_cast<int64_t>(blockIdx.x) * (kWidth / 2);
#pragma unroll
    for (uint32_t step = 0; step < kSteps; ++step) {
      const uint32_t chunk = step * kBlockSize + threadIdx.x;
      float block_amax = 0.0f;
#pragma unroll
      for (uint32_t i = 0; i < kVecSize; ++i) {
        block_amax = math::max(block_amax, math::abs(values[step * kVecSize + i]));
      }
      block_amax = math::max(block_amax, __shfl_xor_sync(0xffffffffu, block_amax, 1));
      block_amax = math::max(block_amax, __shfl_xor_sync(0xffffffffu, block_amax, 2));
      const fp8_e4m3_t sf = nvfp4::block_scale(block_amax, params.global_scale);
      const float multiplier = nvfp4::quant_multiplier(sf, params.global_scale);
      if ((threadIdx.x & 3u) == 0u) {
        params.out_sf[nvfp4::sf_offset_128x4(blockIdx.x, chunk >> 2, kNumBlocks)] =
            *reinterpret_cast<const uint8_t*>(&sf);
      }
      const uint16_t packed =
          static_cast<uint16_t>(nvfp4::pack2(values[step * kVecSize], values[step * kVecSize + 1], multiplier)) |
          (static_cast<uint16_t>(nvfp4::pack2(values[step * kVecSize + 2], values[step * kVecSize + 3], multiplier))
           << 8);
      reinterpret_cast<uint16_t*>(q_row)[chunk] = packed;
    }
  } else if constexpr (!kLayerNorm) {
    cta::reduce_max(amax, smem_amax);
    __syncthreads();
    const float scale = smem_amax[0] / math::FP8_E4M3_MAX;
    if (threadIdx.x == 0) {
      params.out_s[blockIdx.x] = scale;
    }
    const float scale_inv = scale == 0.0f ? 0.0f : 1.0f / scale;
#pragma unroll
    for (uint32_t step = 0; step < kSteps; ++step) {
      QVec q_vec;
#pragma unroll
      for (uint32_t i = 0; i < kVecSize; ++i) {
        const float value = values[step * kVecSize + i] * scale_inv;
        q_vec[i] = static_cast<fp8_e4m3_t>(math::max(math::min(value, math::FP8_E4M3_MAX), -math::FP8_E4M3_MAX));
      }
      q_vec.store(params.out_q + row_offset, step * kBlockSize + threadIdx.x);
    }
  }
}

namespace details {

template <typename E, uint32_t kWidth>
void verify_rows(tvm::ffi::TensorView view, host::SymbolicSize& rows, host::SymbolicDevice& device) {
  host::TensorMatcher({rows, kWidth}).with_dtype<E>().with_device(device).ensure_alignment(kAlignment).verify(view);
}

template <uint32_t kWidth>
void verify_vector(tvm::ffi::TensorView view, host::SymbolicDevice& device) {
  host::TensorMatcher({kWidth}).with_dtype<fp32_t>().with_device(device).ensure_alignment(kAlignment).verify(view);
}

inline void verify_scales(tvm::ffi::TensorView view, host::SymbolicSize& rows, host::SymbolicDevice& device) {
  host::TensorMatcher({rows, 1}).with_dtype<fp32_t>().with_device(device).verify(view);
}

template <uint32_t kWidth>
void verify_nvfp4_outputs(
    tvm::ffi::TensorView out_q4, tvm::ffi::TensorView out_sf, host::SymbolicSize& rows, host::SymbolicDevice& device) {
  namespace nvfp4 = minimax_h3_vae_nvfp4;
  static_assert(kWidth % (nvfp4::kBlock * 4) == 0, "NVFP4 rows need a multiple of four scale blocks");
  host::TensorMatcher({rows, kWidth / 2}).with_dtype<uint8_t>().with_device(device).ensure_alignment(kAlignment).verify(out_q4);
  auto sf_rows = host::SymbolicSize{"sf_rows"};
  host::TensorMatcher({sf_rows, kWidth / nvfp4::kBlock}).with_dtype<uint8_t>().with_device(device).verify(out_sf);
  const int64_t needed = (rows.unwrap() + nvfp4::kScaleRowTile - 1) / nvfp4::kScaleRowTile * nvfp4::kScaleRowTile;
  CHECK_HOST(sf_rows.unwrap() >= needed)
      << "minimax_h3_vae_residual_norm: NVFP4 scale tensor needs " << needed << " rows, got " << sf_rows.unwrap();
}

template <typename T, typename U, uint32_t kWidth, NormMode kMode>
void launch(const Params<T, U>& params, const host::SymbolicSize& rows, const host::SymbolicDevice& device) {
  const int64_t num_rows = rows.unwrap();
  CHECK_HOST(num_rows > 0 && num_rows <= UINT32_MAX)
      << "minimax_h3_vae_residual_norm: rows must be in (0, UINT32_MAX], got " << num_rows;
  host::LaunchKernel(static_cast<uint32_t>(num_rows), kBlockSize, device.unwrap())(
      residual_norm_kernel<T, U, kWidth, kMode>, params);
}

}  // namespace details

/**
 * \brief `q, s = fp8(RMSNorm(x) * weight)` per row.
 *
 * \tparam T      Element type of `x`: fp32_t | fp16_t
 * \tparam kWidth Row width; a multiple of kBlockSize * kVecSize
 */
template <typename T, uint32_t kWidth>
struct RmsNormFp8Kernel {
  /**
   * \param x       [rows, kWidth] T, contiguous, 16-byte aligned
   * \param weight  [kWidth] fp32
   * \param out_q   [rows, kWidth] fp8 e4m3, written
   * \param out_s   [rows, 1] fp32 per-row scales, written
   * \param eps     RMSNorm epsilon
   */
  static void
  run(tvm::ffi::TensorView x,
      tvm::ffi::TensorView weight,
      tvm::ffi::TensorView out_q,
      tvm::ffi::TensorView out_s,
      double eps) {
    using namespace host;
    auto rows = SymbolicSize{"rows"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    details::verify_rows<T, kWidth>(x, rows, device);
    details::verify_vector<kWidth>(weight, device);
    details::verify_rows<fp8_e4m3_t, kWidth>(out_q, rows, device);
    details::verify_scales(out_s, rows, device);

    const Params<T, T> params{
        .x = static_cast<const T*>(x.data_ptr()),
        .projected = nullptr,
        .projected_bias = nullptr,
        .layer_scale = nullptr,
        .weight = static_cast<const float*>(weight.data_ptr()),
        .bias = nullptr,
        .residual_out = nullptr,
        .out_q = static_cast<fp8_e4m3_t*>(out_q.data_ptr()),
        .out_s = static_cast<float*>(out_s.data_ptr()),
        .out = nullptr,
        .out_q4 = nullptr,
        .out_sf = nullptr,
        .global_scale = 0.0f,
        .eps = static_cast<float>(eps),
    };
    details::launch<T, T, kWidth, NormMode::kRmsNormFp8>(params, rows, device);
  }
};

/**
 * \brief `residual = fma(projected, layer_scale, x)` in fp32, then `q, s = fp8(RMSNorm(residual) * weight)`.
 *
 * \tparam T      Element type of `x`: fp32_t | fp16_t
 * \tparam U      Element type of `projected`: fp32_t | fp16_t
 * \tparam kWidth Row width; a multiple of kBlockSize * kVecSize
 */
template <typename T, typename U, uint32_t kWidth>
struct ResidualRmsNormFp8Kernel {
  /**
   * \param x             [rows, kWidth] T, contiguous, 16-byte aligned
   * \param projected     [rows, kWidth] U, contiguous, 16-byte aligned
   * \param layer_scale   [kWidth] fp32
   * \param weight        [kWidth] fp32
   * \param residual_out  [rows, kWidth] fp32, written
   * \param out_q         [rows, kWidth] fp8 e4m3, written
   * \param out_s         [rows, 1] fp32 per-row scales, written
   * \param eps           RMSNorm epsilon
   */
  static void
  run(tvm::ffi::TensorView x,
      tvm::ffi::TensorView projected,
      tvm::ffi::TensorView layer_scale,
      tvm::ffi::TensorView weight,
      tvm::ffi::TensorView residual_out,
      tvm::ffi::TensorView out_q,
      tvm::ffi::TensorView out_s,
      double eps) {
    run_impl(x, projected, nullptr, layer_scale, weight, residual_out, out_q, out_s, eps);
  }

  /// `projected` gets `projected_bias` ([kWidth] fp32) added before the LayerScale fma.
  static void run_bias(
      tvm::ffi::TensorView x,
      tvm::ffi::TensorView projected,
      tvm::ffi::TensorView projected_bias,
      tvm::ffi::TensorView layer_scale,
      tvm::ffi::TensorView weight,
      tvm::ffi::TensorView residual_out,
      tvm::ffi::TensorView out_q,
      tvm::ffi::TensorView out_s,
      double eps) {
    auto device = host::SymbolicDevice{};
    device.set_options<kDLCUDA>();
    details::verify_vector<kWidth>(projected_bias, device);
    run_impl(
        x,
        projected,
        static_cast<const float*>(projected_bias.data_ptr()),
        layer_scale,
        weight,
        residual_out,
        out_q,
        out_s,
        eps);
  }

  static void run_impl(
      tvm::ffi::TensorView x,
      tvm::ffi::TensorView projected,
      const float* projected_bias,
      tvm::ffi::TensorView layer_scale,
      tvm::ffi::TensorView weight,
      tvm::ffi::TensorView residual_out,
      tvm::ffi::TensorView out_q,
      tvm::ffi::TensorView out_s,
      double eps) {
    using namespace host;
    auto rows = SymbolicSize{"rows"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    details::verify_rows<T, kWidth>(x, rows, device);
    details::verify_rows<U, kWidth>(projected, rows, device);
    details::verify_vector<kWidth>(layer_scale, device);
    details::verify_vector<kWidth>(weight, device);
    details::verify_rows<fp32_t, kWidth>(residual_out, rows, device);
    details::verify_rows<fp8_e4m3_t, kWidth>(out_q, rows, device);
    details::verify_scales(out_s, rows, device);

    const Params<T, U> params{
        .x = static_cast<const T*>(x.data_ptr()),
        .projected = static_cast<const U*>(projected.data_ptr()),
        .projected_bias = projected_bias,
        .layer_scale = static_cast<const float*>(layer_scale.data_ptr()),
        .weight = static_cast<const float*>(weight.data_ptr()),
        .bias = nullptr,
        .residual_out = static_cast<float*>(residual_out.data_ptr()),
        .out_q = static_cast<fp8_e4m3_t*>(out_q.data_ptr()),
        .out_s = static_cast<float*>(out_s.data_ptr()),
        .out = nullptr,
        .out_q4 = nullptr,
        .out_sf = nullptr,
        .global_scale = 0.0f,
        .eps = static_cast<float>(eps),
    };
    details::launch<T, U, kWidth, NormMode::kResidualRmsNormFp8>(params, rows, device);
  }
};

/**
 * \brief `out = LayerNorm(fma(projected, layer_scale, x)) * weight + bias` in fp32.
 *
 * \tparam T      Element type of `x`: fp32_t | fp16_t
 * \tparam U      Element type of `projected`: fp32_t | fp16_t
 * \tparam kWidth Row width; a multiple of kBlockSize * kVecSize
 */
template <typename T, typename U, uint32_t kWidth>
struct ResidualLayerNormKernel {
  /**
   * \param x            [rows, kWidth] T, contiguous, 16-byte aligned
   * \param projected    [rows, kWidth] U, contiguous, 16-byte aligned
   * \param layer_scale  [kWidth] fp32
   * \param weight       [kWidth] fp32 LayerNorm gamma
   * \param bias         [kWidth] fp32 LayerNorm beta
   * \param out          [rows, kWidth] fp32, written
   * \param eps          LayerNorm epsilon
   */
  static void
  run(tvm::ffi::TensorView x,
      tvm::ffi::TensorView projected,
      tvm::ffi::TensorView layer_scale,
      tvm::ffi::TensorView weight,
      tvm::ffi::TensorView bias,
      tvm::ffi::TensorView out,
      double eps) {
    run_impl(x, projected, nullptr, layer_scale, weight, bias, out, eps);
  }

  /// `projected` gets `projected_bias` ([kWidth] fp32) added before the LayerScale fma.
  static void run_bias(
      tvm::ffi::TensorView x,
      tvm::ffi::TensorView projected,
      tvm::ffi::TensorView projected_bias,
      tvm::ffi::TensorView layer_scale,
      tvm::ffi::TensorView weight,
      tvm::ffi::TensorView bias,
      tvm::ffi::TensorView out,
      double eps) {
    auto device = host::SymbolicDevice{};
    device.set_options<kDLCUDA>();
    details::verify_vector<kWidth>(projected_bias, device);
    run_impl(x, projected, static_cast<const float*>(projected_bias.data_ptr()), layer_scale, weight, bias, out, eps);
  }

  static void run_impl(
      tvm::ffi::TensorView x,
      tvm::ffi::TensorView projected,
      const float* projected_bias,
      tvm::ffi::TensorView layer_scale,
      tvm::ffi::TensorView weight,
      tvm::ffi::TensorView bias,
      tvm::ffi::TensorView out,
      double eps) {
    using namespace host;
    auto rows = SymbolicSize{"rows"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    details::verify_rows<T, kWidth>(x, rows, device);
    details::verify_rows<U, kWidth>(projected, rows, device);
    details::verify_vector<kWidth>(layer_scale, device);
    details::verify_vector<kWidth>(weight, device);
    details::verify_vector<kWidth>(bias, device);
    details::verify_rows<fp32_t, kWidth>(out, rows, device);

    const Params<T, U> params{
        .x = static_cast<const T*>(x.data_ptr()),
        .projected = static_cast<const U*>(projected.data_ptr()),
        .projected_bias = projected_bias,
        .layer_scale = static_cast<const float*>(layer_scale.data_ptr()),
        .weight = static_cast<const float*>(weight.data_ptr()),
        .bias = static_cast<const float*>(bias.data_ptr()),
        .residual_out = nullptr,
        .out_q = nullptr,
        .out_s = nullptr,
        .out = static_cast<float*>(out.data_ptr()),
        .out_q4 = nullptr,
        .out_sf = nullptr,
        .global_scale = 0.0f,
        .eps = static_cast<float>(eps),
    };
    details::launch<T, U, kWidth, NormMode::kResidualLayerNorm>(params, rows, device);
  }
};

/**
 * \brief `residual = fma(projected, layer_scale, x)` in fp32, then `q4, sf = nvfp4(RMSNorm(residual) * weight)`.
 *
 * The E2M1 codes and the 128x4-swizzled E4M3 block scales are laid out for
 * `flashinfer.mm_fp4`; `global_scale` is the calibrated tensor scale of the
 * consuming linear's input (`448 * 6 / amax`).
 *
 * \tparam T      Element type of `x`: fp32_t | fp16_t
 * \tparam U      Element type of `projected`: fp32_t | fp16_t
 * \tparam kWidth Row width; a multiple of kBlockSize * kVecSize and of 64
 */
template <typename T, typename U, uint32_t kWidth>
struct ResidualRmsNormFp4Kernel {
  /**
   * \param x             [rows, kWidth] T, contiguous, 16-byte aligned
   * \param projected     [rows, kWidth] U, contiguous, 16-byte aligned
   * \param layer_scale   [kWidth] fp32
   * \param weight        [kWidth] fp32
   * \param residual_out  [rows, kWidth] fp32, written
   * \param out_q4        [rows, kWidth / 2] uint8 packed E2M1, written
   * \param out_sf        [>= ceil(rows / 128) * 128, kWidth / 16] uint8 E4M3, swizzled, written (valid rows only)
   * \param global_scale  NVFP4 tensor scale of the produced activation
   * \param eps           RMSNorm epsilon
   */
  static void
  run(tvm::ffi::TensorView x,
      tvm::ffi::TensorView projected,
      tvm::ffi::TensorView layer_scale,
      tvm::ffi::TensorView weight,
      tvm::ffi::TensorView residual_out,
      tvm::ffi::TensorView out_q4,
      tvm::ffi::TensorView out_sf,
      double global_scale,
      double eps) {
    using namespace host;
    auto rows = SymbolicSize{"rows"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    details::verify_rows<T, kWidth>(x, rows, device);
    details::verify_rows<U, kWidth>(projected, rows, device);
    details::verify_vector<kWidth>(layer_scale, device);
    details::verify_vector<kWidth>(weight, device);
    details::verify_rows<fp32_t, kWidth>(residual_out, rows, device);
    details::verify_nvfp4_outputs<kWidth>(out_q4, out_sf, rows, device);
    CHECK_HOST(global_scale > 0.0) << "minimax_h3_vae_residual_norm: NVFP4 global scale must be positive";

    const Params<T, U> params{
        .x = static_cast<const T*>(x.data_ptr()),
        .projected = static_cast<const U*>(projected.data_ptr()),
        .projected_bias = nullptr,
        .layer_scale = static_cast<const float*>(layer_scale.data_ptr()),
        .weight = static_cast<const float*>(weight.data_ptr()),
        .bias = nullptr,
        .residual_out = static_cast<float*>(residual_out.data_ptr()),
        .out_q = nullptr,
        .out_s = nullptr,
        .out = nullptr,
        .out_q4 = static_cast<uint8_t*>(out_q4.data_ptr()),
        .out_sf = static_cast<uint8_t*>(out_sf.data_ptr()),
        .global_scale = static_cast<float>(global_scale),
        .eps = static_cast<float>(eps),
    };
    details::launch<T, U, kWidth, NormMode::kResidualRmsNormFp4>(params, rows, device);
  }
};

}  // namespace minimax_h3_vae_residual_norm

}  // namespace sglang
