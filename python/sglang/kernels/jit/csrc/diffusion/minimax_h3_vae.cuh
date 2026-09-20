// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <cmath>
#include <cstdint>
#include <cuda_fp8.h>

namespace sglang::minimax_h3_vae {

template <class T>
inline T* ptr(tvm::ffi::TensorView t) {
  return reinterpret_cast<T*>(static_cast<char*>(t.data_ptr()) + t.byte_offset());
}

__device__ inline float warp_sum(float x) {
  for (int d = 16; d; d >>= 1)
    x += __shfl_xor_sync(0xffffffff, x, d);
  return x;
}
__device__ inline float warp_max(float x) {
  for (int d = 16; d; d >>= 1)
    x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, d));
  return x;
}

template <bool Max>
__device__ float block_reduce(float x) {
  __shared__ float scratch[8];
  int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  x = Max ? warp_max(x) : warp_sum(x);
  if (lane == 0) scratch[warp] = x;
  __syncthreads();
  float total = lane < 8 ? scratch[lane] : 0.f;
  total = Max ? warp_max(total) : warp_sum(total);
  // Every warp reads the same completed scratch before any subsequent reuse.
  __syncthreads();
  return total;
}

// One warp owns BOTH Q and K of a head. Each lane carries two dimensions.
// Q/K statistics remain independent; cos/sin loads are shared by Q and K.
template <class T, bool PackV>
__global__ void minimax_h3_vae_joint_qk_norm_rope_kernel(
    const T* __restrict__ qkv,
    const T* __restrict__ rope,
    T* __restrict__ q,
    T* __restrict__ k,
    T* __restrict__ v,
    int rows,
    int heads,
    float epsq,
    float epsk) {
  int row = blockIdx.x * 8 + (threadIdx.x >> 5);
  int lane = threadIdx.x & 31;
  if (row >= rows) return;
  const T* in = qkv + static_cast<int64_t>(row) * 192;
  float q0 = float(in[lane]), q1 = float(in[lane + 32]);
  float k0 = float(in[lane + 64]), k1 = float(in[lane + 96]);
  float qr = rsqrtf(warp_sum(fmaf(q0, q0, q1 * q1)) / 64.f + epsq);
  float kr = rsqrtf(warp_sum(fmaf(k0, k0, k1 * k1)) / 64.f + epsk);
  q0 *= qr;
  q1 *= qr;
  k0 *= kr;
  k1 *= kr;
  const T* cs = rope + static_cast<int64_t>(row / heads) * 48;
#pragma unroll
  for (int part = 0; part < 2; ++part) {
    int d = lane + part * 32;
    int partner = d < 24 ? d + 24 : d - 24;
    // All lanes participate in both shuffles, including unrotated dimensions.
    float pq0 = __shfl_sync(0xffffffff, q0, partner & 31);
    float pq1 = __shfl_sync(0xffffffff, q1, partner & 31);
    float pk0 = __shfl_sync(0xffffffff, k0, partner & 31);
    float pk1 = __shfl_sync(0xffffffff, k1, partner & 31);
    float qd = part ? q1 : q0, kd = part ? k1 : k0;
    if (d < 48) {
      int offset = d < 24 ? d : d - 24;
      float c = float(cs[offset]), s = float(cs[offset + 24]);
      float sign = d < 24 ? -1.f : 1.f;
      qd = fmaf(sign * s, partner < 32 ? pq0 : pq1, qd * c);
      kd = fmaf(sign * s, partner < 32 ? pk0 : pk1, kd * c);
    }
    int64_t out = static_cast<int64_t>(row) * 64 + d;
    q[out] = T(qd);
    k[out] = T(kd);
    if constexpr (PackV) v[out] = in[128 + d];
  }
}

template <class T, bool PackV>
void joint_qk(
    tvm::ffi::TensorView qkv,
    tvm::ffi::TensorView rope,
    tvm::ffi::TensorView q,
    tvm::ffi::TensorView k,
    tvm::ffi::TensorView v,
    double epsq,
    double epsk) {
  using namespace host;
  auto B = SymbolicSize{"B"}, S = SymbolicSize{"S"}, H = SymbolicSize{"H"};
  auto D = SymbolicDevice{};
  D.set_options<kDLCUDA>();
  TensorMatcher({B, S, H, 192}).with_dtype<T>().with_device(D).verify(qkv);
  TensorMatcher({B.unwrap() * S.unwrap(), 48}).with_dtype<T>().with_device(D).verify(rope);
  TensorMatcher({B, S, H, 64}).with_dtype<T>().with_device(D).verify(q).verify(k);
  if constexpr (PackV) TensorMatcher({B, S, H, 64}).with_dtype<T>().with_device(D).verify(v);
  int64_t rows = B.unwrap() * S.unwrap() * H.unwrap();
  RuntimeCheck(rows > 0 && rows < INT32_MAX, "invalid joint QK row count");
  LaunchKernel((rows + 7) / 8, 256, D.unwrap())(
      minimax_h3_vae_joint_qk_norm_rope_kernel<T, PackV>,
      ptr<T>(qkv),
      ptr<T>(rope),
      ptr<T>(q),
      ptr<T>(k),
      ptr<T>(v),
      int(rows),
      int(H.unwrap()),
      float(epsq),
      float(epsk));
}

// Mode 0: RMSNorm+affine+quant. Mode 1 additionally forms the new FP32
// residual state. Mode 2 forms residual+LayerNorm+affine with FP32 output.
template <class T, class U, int Width, int Mode>
__global__ void minimax_h3_vae_residual_norm_quant_kernel(
    const T* __restrict__ x,
    const U* __restrict__ projected,
    const float* __restrict__ layer_scale,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ residual,
    fp8_e4m3_t* __restrict__ out,
    float* __restrict__ scales,
    float* __restrict__ norm_out,
    float eps) {
  constexpr int Items = Width / 256;
  int64_t offset = static_cast<int64_t>(blockIdx.x) * Width;
  float values[Items], sum = 0.f, sumsq = 0.f;
#pragma unroll
  for (int j = 0; j < Items; ++j) {
    int col = threadIdx.x + j * 256;
    float value = float(x[offset + col]);
    if constexpr (Mode != 0) value = fmaf(float(projected[offset + col]), layer_scale[col], value);
    values[j] = value;
    sum += value;
    sumsq = fmaf(value, value, sumsq);
    if constexpr (Mode == 1) residual[offset + col] = value;
  }
  float mean = 0.f;
  if constexpr (Mode == 2) mean = block_reduce<false>(sum) / Width;
  if constexpr (Mode == 2) {
    sumsq = 0.f;
#pragma unroll
    for (int j = 0; j < Items; ++j) {
      float d = values[j] - mean;
      sumsq = fmaf(d, d, sumsq);
    }
  }
  float variance = block_reduce<false>(sumsq) / Width;
  float inv = rsqrtf(variance + eps), amax = 0.f;
#pragma unroll
  for (int j = 0; j < Items; ++j) {
    int col = threadIdx.x + j * 256;
    float z = (values[j] - mean) * inv * gamma[col];
    if constexpr (Mode == 2) z += beta[col];
    values[j] = z;
    amax = fmaxf(amax, fabsf(z));
    if constexpr (Mode == 2) norm_out[offset + col] = z;
  }
  if constexpr (Mode != 2) {
    float scale = block_reduce<true>(amax) / 448.f;
    if (threadIdx.x == 0) scales[blockIdx.x] = scale;
    float inverse = scale == 0.f ? 0.f : 1.f / scale;
#pragma unroll
    for (int j = 0; j < Items; ++j)
      out[offset + threadIdx.x + j * 256] = fp8_e4m3_t(fminf(448.f, fmaxf(-448.f, values[j] * inverse)));
  }
}

template <class T, class U, int Width, int Mode>
void norm_quant(
    tvm::ffi::TensorView x,
    tvm::ffi::TensorView projected,
    tvm::ffi::TensorView layer_scale,
    tvm::ffi::TensorView gamma,
    tvm::ffi::TensorView beta,
    tvm::ffi::TensorView residual,
    tvm::ffi::TensorView out,
    tvm::ffi::TensorView scales,
    tvm::ffi::TensorView norm_out,
    double eps) {
  using namespace host;
  auto M = SymbolicSize{"M"};
  auto D = SymbolicDevice{};
  D.set_options<kDLCUDA>();
  TensorMatcher({M, Width}).with_dtype<T>().with_device(D).verify(x);
  TensorMatcher({Width}).with_dtype<float>().with_device(D).verify(gamma);
  if constexpr (Mode != 0) {
    TensorMatcher({M, Width}).with_dtype<U>().with_device(D).verify(projected);
    TensorMatcher({Width}).with_dtype<float>().with_device(D).verify(layer_scale);
    TensorMatcher({M, Width}).with_dtype<float>().with_device(D).verify(residual);
  }
  if constexpr (Mode == 2) {
    TensorMatcher({Width}).with_dtype<float>().with_device(D).verify(beta);
    TensorMatcher({M, Width}).with_dtype<float>().with_device(D).verify(norm_out);
  } else {
    TensorMatcher({M, Width}).with_dtype<fp8_e4m3_t>().with_device(D).verify(out);
    TensorMatcher({M, 1}).with_dtype<float>().with_device(D).verify(scales);
  }
  RuntimeCheck(M.unwrap() > 0 && M.unwrap() < INT32_MAX, "invalid norm rows");
  LaunchKernel(M.unwrap(), 256, D.unwrap())(
      minimax_h3_vae_residual_norm_quant_kernel<T, U, Width, Mode>,
      ptr<T>(x),
      ptr<U>(projected),
      ptr<float>(layer_scale),
      ptr<float>(gamma),
      ptr<float>(beta),
      ptr<float>(residual),
      ptr<fp8_e4m3_t>(out),
      ptr<float>(scales),
      ptr<float>(norm_out),
      float(eps));
}

// Vectorized memory accesses and the FP32 CUDA sigmoid intrinsic avoid the
// scalar implementation's instruction-heavy precise expf/byte-store path.
template <class T, int Width>
__global__ void minimax_h3_vae_silu_mul_quant_vector_kernel(
    const T* __restrict__ x, fp8_e4m3_t* __restrict__ out, float* __restrict__ scales) {
  constexpr int Vec = 8, Vectors = Width / Vec, Iterations = Vectors / 256;
  using In = device::AlignedVector<T, Vec>;
  using Out = device::AlignedVector<fp8_e4m3_t, Vec>;
  int64_t base = static_cast<int64_t>(blockIdx.x) * Vectors;
  float values[Iterations][Vec], amax = 0.f;
#pragma unroll
  for (int j = 0; j < Iterations; ++j) {
    int64_t col = threadIdx.x + j * 256;
    In gate, up;
    gate.load(x, base * 2 + col);
    up.load(x, base * 2 + Vectors + col);
#pragma unroll
    for (int k = 0; k < Vec; ++k) {
      float g = float(gate[k]);
      float value = (g / (1.f + __expf(-g))) * float(up[k]);
      values[j][k] = value;
      amax = fmaxf(amax, fabsf(value));
    }
  }
  float scale = block_reduce<true>(amax) / 448.f;
  if (threadIdx.x == 0) scales[blockIdx.x] = scale;
  float inverse = scale == 0.f ? 0.f : 1.f / scale;
#pragma unroll
  for (int j = 0; j < Iterations; ++j) {
    Out result;
#pragma unroll
    for (int k = 0; k < Vec; ++k)
      result[k] = fp8_e4m3_t(fminf(448.f, fmaxf(-448.f, values[j][k] * inverse)));
    result.store(out, base + threadIdx.x + j * 256);
  }
}

template <class T, int Width>
void silu_quant_vectorized(tvm::ffi::TensorView x, tvm::ffi::TensorView out, tvm::ffi::TensorView scales) {
  using namespace host;
  auto M = SymbolicSize{"M"};
  auto D = SymbolicDevice{};
  D.set_options<kDLCUDA>();
  TensorMatcher({M, Width * 2}).with_dtype<T>().with_device(D).verify(x);
  TensorMatcher({M, Width}).with_dtype<fp8_e4m3_t>().with_device(D).verify(out);
  TensorMatcher({M, 1}).with_dtype<float>().with_device(D).verify(scales);
  RuntimeCheck(M.unwrap() > 0 && M.unwrap() < INT32_MAX, "invalid vector activation rows");
  LaunchKernel(M.unwrap(), 256, D.unwrap())(
      minimax_h3_vae_silu_mul_quant_vector_kernel<T, Width>, ptr<T>(x), ptr<fp8_e4m3_t>(out), ptr<float>(scales));
}
}  // namespace sglang::minimax_h3_vae
