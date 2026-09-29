/*
 * Per-token fp8 rows for the Ulysses attention-output gather.
 *
 * Each rank owns H_local heads of every token after attention. Before the
 * gather it quantises each token's H_local * D values to e4m3 with one scale
 * (amax / 448, the scheme the fp8 GEMM applies itself) and ships
 * `[codes | fp32 scale | zero pad]` rows of row_bytes(G) bytes. The
 * destination receives one such row per source rank and rescales every
 * source's codes to the token's largest scale, which yields the (q, s) pair a
 * per-token-scaled fp8 GEMM consumes directly.
 */
#pragma once

#include <sgl_kernel/ffi.h>
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/warp.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <type_traits>

namespace sglang {

namespace ulysses_fp8_gather {

constexpr float kFp8Max = 448.0f;
constexpr uint32_t kRowAlign = 16;
constexpr uint32_t kLaneElems = 4;  // one 4-byte code word or 8-byte source word per lane
constexpr uint32_t kWarpElems = 32 * kLaneElems;
constexpr uint32_t kMaxIters = 32;  // groups up to 4096 columns stay in registers
constexpr uint32_t kWarpsPerBlock = 4;
constexpr uint32_t kMaxWorld = 32;

SGL_DEVICE_HOST constexpr int64_t row_bytes(int64_t group) {
  return (group + 4 + kRowAlign - 1) / kRowAlign * kRowAlign;
}

namespace details {

template <typename T>
SGL_DEVICE float to_float(T v) {
  if constexpr (std::is_same_v<T, half>) {
    return __half2float(v);
  } else {
    return __bfloat162float(v);
  }
}

SGL_DEVICE uint32_t pack_fp8x4(float v0, float v1, float v2, float v3) {
  const __nv_fp8x2_storage_t lo = __nv_cvt_float2_to_fp8x2(make_float2(v0, v1), __NV_SATFINITE, __NV_E4M3);
  const __nv_fp8x2_storage_t hi = __nv_cvt_float2_to_fp8x2(make_float2(v2, v3), __NV_SATFINITE, __NV_E4M3);
  return static_cast<uint32_t>(lo) | (static_cast<uint32_t>(hi) << 16);
}

SGL_DEVICE float4 unpack_fp8x4(uint32_t word) {
  const __half2 lo = __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(word & 0xffffu), __NV_E4M3);
  const __half2 hi = __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(word >> 16), __NV_E4M3);
  const float2 a = __half22float2(lo);
  const float2 b = __half22float2(hi);
  return make_float4(a.x, a.y, b.x, b.y);
}

SGL_DEVICE float saturate(float v) {
  return fmaxf(fminf(v, kFp8Max), -kFp8Max);
}

}  // namespace details

// One warp per token: amax over the row, then codes, scale and pad.
template <typename T>
__global__ void
QuantKernel(const T* __restrict__ x, uint8_t* __restrict__ payload, uint32_t rows, uint32_t group, uint32_t stride) {
  using namespace device;
  const uint32_t warp = threadIdx.x / 32;
  const uint32_t lane = threadIdx.x % 32;
  const uint32_t row = blockIdx.x * kWarpsPerBlock + warp;
  if (row >= rows) return;
  const T* src = x + static_cast<uint64_t>(row) * group;
  const uint32_t iters = group / kWarpElems;
  uint2 raw[kMaxIters];
  float amax = 0.0f;
  for (uint32_t i = 0; i < iters; ++i) {
    raw[i] = *reinterpret_cast<const uint2*>(src + i * kWarpElems + lane * kLaneElems);
    const T* v = reinterpret_cast<const T*>(&raw[i]);
#pragma unroll
    for (uint32_t j = 0; j < kLaneElems; ++j)
      amax = fmaxf(amax, fabsf(details::to_float(v[j])));
  }
  amax = warp::reduce_max(amax);
  const float scale = __fdiv_rn(amax, kFp8Max);
  const float inv = scale == 0.0f ? 0.0f : __frcp_rn(scale);
  uint8_t* dst = payload + static_cast<uint64_t>(row) * stride;
  for (uint32_t i = 0; i < iters; ++i) {
    const T* v = reinterpret_cast<const T*>(&raw[i]);
    *reinterpret_cast<uint32_t*>(dst + i * kWarpElems + lane * kLaneElems) = details::pack_fp8x4(
        details::saturate(__fmul_rn(details::to_float(v[0]), inv)),
        details::saturate(__fmul_rn(details::to_float(v[1]), inv)),
        details::saturate(__fmul_rn(details::to_float(v[2]), inv)),
        details::saturate(__fmul_rn(details::to_float(v[3]), inv)));
  }
  if (lane == 0) {
    *reinterpret_cast<float*>(dst + group) = scale;
    for (uint32_t b = group + 4; b < stride; b += 4)
      *reinterpret_cast<uint32_t*>(dst + b) = 0u;
  }
}

// One warp per token: the token's scale is the largest source scale; a source
// whose scale is that maximum keeps its codes bit for bit, the others are
// rescaled by scale_source / scale_token and rounded once more.
__global__ void RequantKernel(
    const uint8_t* __restrict__ payload,
    uint8_t* __restrict__ q,
    float* __restrict__ s,
    uint32_t rows,
    uint32_t world,
    uint32_t group,
    uint32_t stride) {
  using namespace device;
  const uint32_t warp = threadIdx.x / 32;
  const uint32_t lane = threadIdx.x % 32;
  const uint32_t row = blockIdx.x * kWarpsPerBlock + warp;
  if (row >= rows) return;
  const uint8_t* base = payload + static_cast<uint64_t>(row) * world * stride;
  const float own = lane < world ? *reinterpret_cast<const float*>(base + lane * stride + group) : 0.0f;
  const float token_scale = warp::reduce_max(own);
  uint8_t* dst = q + static_cast<uint64_t>(row) * world * group;
  const uint32_t iters = group / kWarpElems;
  for (uint32_t g = 0; g < world; ++g) {
    const float source_scale = __shfl_sync(0xffffffffu, own, g);
    const float ratio = token_scale == 0.0f ? 0.0f : __fdiv_rn(source_scale, token_scale);
    const uint8_t* src = base + g * stride;
    uint8_t* out = dst + g * group;
    for (uint32_t i = 0; i < iters; ++i) {
      const uint32_t offset = i * kWarpElems + lane * kLaneElems;
      const uint32_t word = *reinterpret_cast<const uint32_t*>(src + offset);
      if (ratio == 1.0f) {
        *reinterpret_cast<uint32_t*>(out + offset) = word;
        continue;
      }
      const float4 v = details::unpack_fp8x4(word);
      *reinterpret_cast<uint32_t*>(out + offset) = details::pack_fp8x4(
          __fmul_rn(v.x, ratio), __fmul_rn(v.y, ratio), __fmul_rn(v.z, ratio), __fmul_rn(v.w, ratio));
    }
  }
  if (lane == 0) s[row] = token_scale;
}

template <typename T>
struct Kernels {
  static_assert(std::is_same_v<T, half> || std::is_same_v<T, nv_bfloat16>);

  /*!
   * \brief Per-token fp8 rows: `payload[t] = fp8(x[t] / s_t) | fp32 s_t | zero pad`, s_t = amax / 448.
   * \param x        `[S, G]` activations; G a multiple of 128, at most 4096
   * \param payload  uint8 `[S, row_bytes(G)]`
   */
  static void quant(tvm::ffi::TensorView x, tvm::ffi::TensorView payload) {
    using namespace host;
    SymbolicSize S{"rows"}, G{"group"}, R{"row_bytes"};
    SymbolicDevice device;
    device.set_options<kDLCUDA>();
    TensorMatcher({S, G}).template with_dtype<T>().template with_device<kDLCUDA>(device).verify(x);
    TensorMatcher({S, R}).template with_dtype<uint8_t>().template with_device<kDLCUDA>(device).verify(payload);
    const int64_t group = G.unwrap();
    CHECK_HOST(group % kWarpElems == 0 && group > 0 && group <= kWarpElems * kMaxIters)
        << "group must be a positive multiple of " << kWarpElems << " up to " << kWarpElems * kMaxIters << ", got "
        << group;
    CHECK_HOST(R.unwrap() == row_bytes(group)) << "payload rows must be " << row_bytes(group) << " bytes";
    const int64_t rows = S.unwrap();
    LaunchKernel(div_ceil(rows, static_cast<int64_t>(kWarpsPerBlock)), 32 * kWarpsPerBlock, device.unwrap())(
        QuantKernel<T>,
        static_cast<const T*>(x.data_ptr()),
        static_cast<uint8_t*>(payload.data_ptr()),
        static_cast<uint32_t>(rows),
        static_cast<uint32_t>(group),
        static_cast<uint32_t>(row_bytes(group)));
  }

  /*!
   * \brief Merge the rows of W sources into one per-token pair.
   * \param payload  uint8 `[S_local, W, row_bytes(G)]`, source-major per token
   * \param q        e4m3 `[S_local, W * G]`, source g's codes in columns [g * G, (g + 1) * G)
   * \param s        fp32 `[S_local, 1]`, the largest source scale of the token
   */
  static void requant(tvm::ffi::TensorView payload, tvm::ffi::TensorView q, tvm::ffi::TensorView s) {
    using namespace host;
    SymbolicSize S{"rows"}, W{"world"}, R{"row_bytes"}, C{"columns"};
    SymbolicDevice device;
    device.set_options<kDLCUDA>();
    TensorMatcher({S, W, R}).template with_dtype<uint8_t>().template with_device<kDLCUDA>(device).verify(payload);
    TensorMatcher({S, C}).template with_dtype<fp8_e4m3_t>().template with_device<kDLCUDA>(device).verify(q);
    TensorMatcher({S, 1}).template with_dtype<fp32_t>().template with_device<kDLCUDA>(device).verify(s);
    const int64_t world = W.unwrap();
    CHECK_HOST(world >= 1 && world <= kMaxWorld) << "world size out of range: " << world;
    CHECK_HOST(C.unwrap() % world == 0) << "columns must split over the sources";
    const int64_t group = C.unwrap() / world;
    CHECK_HOST(group % kWarpElems == 0 && group > 0 && group <= kWarpElems * kMaxIters)
        << "group must be a positive multiple of " << kWarpElems << " up to " << kWarpElems * kMaxIters << ", got "
        << group;
    CHECK_HOST(R.unwrap() == row_bytes(group)) << "payload rows must be " << row_bytes(group) << " bytes";
    const int64_t rows = S.unwrap();
    LaunchKernel(div_ceil(rows, static_cast<int64_t>(kWarpsPerBlock)), 32 * kWarpsPerBlock, device.unwrap())(
        RequantKernel,
        static_cast<const uint8_t*>(payload.data_ptr()),
        static_cast<uint8_t*>(q.data_ptr()),
        static_cast<float*>(s.data_ptr()),
        static_cast<uint32_t>(rows),
        static_cast<uint32_t>(world),
        static_cast<uint32_t>(group),
        static_cast<uint32_t>(row_bytes(group)));
  }
};

}  // namespace ulysses_fp8_gather

}  // namespace sglang
