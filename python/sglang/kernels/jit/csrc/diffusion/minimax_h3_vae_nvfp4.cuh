// NVFP4 (E2M1 values, E4M3 block-16 scales, fp32 global scale) epilogue helpers
// shared by the MiniMax-H3 VAE decoder producers. The scale layout is the
// 128x4-swizzled one consumed by the cutlass / cuDNN NVFP4 GEMMs
// (TensorRT-LLM `get_sf_out_offset_128x4`), so a producer's output feeds
// `flashinfer.mm_fp4` directly, without a separate `fp4_quantize` pass.
#pragma once

#include <cuda_fp4.h>

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>

#include <cstdint>

namespace sglang {

namespace minimax_h3_vae_nvfp4 {

/// Elements per E4M3 scale factor.
constexpr uint32_t kBlock = 16;
/// Largest finite E2M1 magnitude.
constexpr float kE2M1Max = 6.0f;
/// Rows of the scale tensor are padded to this multiple.
constexpr uint32_t kScaleRowTile = 128;

/// \brief Byte offset of the scale of (row, kb) in the swizzled `[rows_pad, num_kb]` tensor.
///
/// Layout: [rows / 128][num_kb / 4][row % 32][(row / 32) % 4][kb % 4]; `num_kb`
/// must be a multiple of 4 and the tensor must hold `ceil(rows / 128) * 128` rows.
SGL_DEVICE int64_t sf_offset_128x4(int64_t row, uint32_t kb, uint32_t num_kb) {
  const int64_t m_tile = row >> 7;
  const uint32_t m_in = static_cast<uint32_t>(row & 127);
  const uint32_t k_tiles = num_kb >> 2;
  return (m_tile * k_tiles + (kb >> 2)) * 512 + (m_in & 31) * 16 + (m_in >> 5) * 4 + (kb & 3);
}

/// \brief E4M3 scale of a block with absolute maximum `amax`: `amax * global_scale / 6`.
SGL_DEVICE fp8_e4m3_t block_scale(float amax, float global_scale) {
  return static_cast<fp8_e4m3_t>(amax * (global_scale / kE2M1Max));
}

/// \brief Factor that maps an fp32 value of the block onto the E2M1 grid: `global_scale / sf`.
///
/// A zero scale (all-zero block) yields 0, so every code of the block is 0.
SGL_DEVICE float quant_multiplier(fp8_e4m3_t sf, float global_scale) {
  const float scale = static_cast<float>(sf);
  return scale == 0.0f ? 0.0f : global_scale / scale;
}

/// \brief Two fp32 values to one byte of E2M1 codes; `a` lands in the low nibble.
SGL_DEVICE uint8_t pack2(float a, float b, float multiplier) {
  return static_cast<uint8_t>(
      __nv_cvt_float2_to_fp4x2(float2{a * multiplier, b * multiplier}, __NV_E2M1, cudaRoundNearest));
}

}  // namespace minimax_h3_vae_nvfp4

}  // namespace sglang
