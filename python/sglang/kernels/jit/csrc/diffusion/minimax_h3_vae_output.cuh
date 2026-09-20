// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>

namespace sglang::minimax_h3_vae_output {
template <class T>
inline T* ptr(tvm::ffi::TensorView t) {
  return reinterpret_cast<T*>(static_cast<char*>(t.data_ptr()) + t.byte_offset());
}
struct Layout {
  int64_t s[5];
  int64_t n[5];
};
inline Layout layout(tvm::ffi::TensorView t) {
  Layout l{};
  for (int d = 0; d < 5; ++d) {
    l.s[d] = t.stride(d);
    l.n[d] = t.size(d);
  }
  return l;
}
__device__ inline int64_t off(Layout l, int64_t p, int y, int x) {
  int64_t t = p % l.n[2];
  p /= l.n[2];
  int64_t c = p % l.n[1], b = p / l.n[1];
  return b * l.s[0] + c * l.s[1] + t * l.s[2] + int64_t(y) * l.s[3] + int64_t(x) * l.s[4];
}
__device__ inline float blend(float a, float b, int k, int n) {
  float w = __fdiv_rn(float(k), float(n));
  return __fadd_rn(__fmul_rn(a, __fsub_rn(1.f, w)), __fmul_rn(b, w));
}
struct Tile {
  const float *cur, *up, *left;
  int oy, ox, h, w, hy, wx;
  float inv_hy, inv_wx;
};
struct Tiles {
  Tile values[64];
};
static_assert(sizeof(Tiles) + 2 * sizeof(Layout) + 128 < 4096, "Kernel parameters exceed 4 KiB");

struct FastLayout {
  int64_t b4, c4, t4;
  int t, c, w4;
};
inline bool spatial4(tvm::ffi::TensorView t) {
  return t.stride(4) == 1 && t.stride(3) == t.size(4) && t.size(4) % 4 == 0 && t.stride(0) % 4 == 0 &&
         t.stride(1) % 4 == 0 && t.stride(2) % 4 == 0 && reinterpret_cast<uintptr_t>(ptr<float>(t)) % 16 == 0;
}
inline FastLayout fast_layout(tvm::ffi::TensorView t) {
  return {t.stride(0) / 4, t.stride(1) / 4, t.stride(2) / 4, int(t.size(2)), int(t.size(1)), int(t.size(4) / 4)};
}
__device__ inline int64_t prefix4(FastLayout l, int p) {
  return int64_t(p / (l.c * l.t)) * l.b4 + int64_t((p / l.t) % l.c) * l.c4 + int64_t(p % l.t) * l.t4;
}
__device__ inline float4 mix4(float4 a, float4 b, float w) {
  float wa = 1.f - w;
  return {fmaf(b.x, w, a.x * wa), fmaf(b.y, w, a.y * wa), fmaf(b.z, w, a.z * wa), fmaf(b.w, w, a.w * wa)};
}
__global__ void minimax_h3_vae_output_spatial_blend_write_kernel_vec(
    Tiles tiles, FastLayout in, int prefix, int in_h, int out_h, int out_w4, float4* dst) {
  Tile tile = tiles.values[blockIdx.y];
  int width4 = tile.w / 4;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= prefix * tile.h * width4) return;
  int x4 = i % width4;
  i /= width4;
  int y = i % tile.h, p = i / tile.h;
  int64_t base = prefix4(in, p);
  float4 v = reinterpret_cast<const float4*>(tile.cur)[base + int64_t(y) * in.w4 + x4];
  if (tile.up && y < tile.hy) {
    float4 a = reinterpret_cast<const float4*>(tile.up)[base + int64_t(in_h - tile.hy + y) * in.w4 + x4];
    v = mix4(a, v, float(y) * tile.inv_hy);
  }
  if (tile.left && x4 < tile.wx / 4) {
    float4 a = reinterpret_cast<const float4*>(tile.left)[base + int64_t(y) * in.w4 + in.w4 - tile.wx / 4 + x4];
    float w0 = float(x4 * 4) * tile.inv_wx, w1 = float(x4 * 4 + 1) * tile.inv_wx;
    float w2 = float(x4 * 4 + 2) * tile.inv_wx, w3 = float(x4 * 4 + 3) * tile.inv_wx;
    v = {
        fmaf(v.x, w0, a.x * (1.f - w0)),
        fmaf(v.y, w1, a.y * (1.f - w1)),
        fmaf(v.z, w2, a.z * (1.f - w2)),
        fmaf(v.w, w3, a.w * (1.f - w3))};
  }
  dst[(int64_t(p) * out_h + tile.oy + y) * out_w4 + tile.ox / 4 + x4] = v;
}

__global__ void minimax_h3_vae_output_spatial_blend_write_kernel(Tiles tiles, Layout in, Layout out, float* dst) {
  Tile tile = tiles.values[blockIdx.y];
  int64_t idx = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = in.n[0] * in.n[1] * in.n[2] * tile.h * tile.w;
  if (idx >= total) return;
  int x = idx % tile.w;
  idx /= tile.w;
  int y = idx % tile.h;
  int64_t p = idx / tile.h;
  float v = tile.cur[off(in, p, y, x)];
  if (tile.up && y < tile.hy) v = blend(tile.up[off(in, p, int(in.n[3]) - tile.hy + y, x)], v, y, tile.hy);
  // The released implementation reads the raw left tile, not a vertically
  // blended left tile. Preserve that order and corner behavior exactly.
  if (tile.left && x < tile.wx) v = blend(tile.left[off(in, p, y, int(in.n[4]) - tile.wx + x)], v, x, tile.wx);
  dst[off(out, p, tile.oy + y, tile.ox + x)] = v;
}

void assemble(tvm::ffi::TensorView first, tvm::ffi::TensorView descriptors, tvm::ffi::TensorView output) {
  using namespace host;
  auto D = SymbolicDevice{};
  D.set_options<kDLCUDA>();
  TensorMatcher({output.size(0), output.size(1), output.size(2), output.size(3), output.size(4)})
      .with_dtype<float>()
      .with_device(D)
      .verify(output);
  RuntimeCheck(
      descriptors.size(0) > 0 && descriptors.size(0) <= 64 && descriptors.size(1) == 9, "invalid tile descriptors");
  Tiles tiles{};
  const int64_t* meta = ptr<int64_t>(descriptors);
  int64_t max_area = 0;
  bool vectorized = spatial4(first) && spatial4(output);
  for (int i = 0; i < descriptors.size(0); ++i) {
    const int64_t* m = meta + i * 9;
    tiles.values[i] = {
        reinterpret_cast<const float*>(m[0]),
        reinterpret_cast<const float*>(m[1]),
        reinterpret_cast<const float*>(m[2]),
        int(m[3]),
        int(m[4]),
        int(m[5]),
        int(m[6]),
        int(m[7]),
        int(m[8]),
        m[7] ? 1.f / float(m[7]) : 0.f,
        m[8] ? 1.f / float(m[8]) : 0.f};
    max_area = std::max(max_area, m[5] * m[6]);
    vectorized = vectorized && m[0] % 16 == 0 && m[1] % 16 == 0 && m[2] % 16 == 0 && m[4] % 4 == 0 && m[6] % 4 == 0 &&
                 m[8] % 4 == 0;
  }
  int64_t n = first.size(0) * first.size(1) * first.size(2) * max_area;
  if (vectorized && n / 4 < INT_MAX) {
    LaunchKernel(dim3((n / 4 + 255) / 256, descriptors.size(0), 1), 256, D.unwrap())(
        minimax_h3_vae_output_spatial_blend_write_kernel_vec,
        tiles,
        fast_layout(first),
        int(first.size(0) * first.size(1) * first.size(2)),
        int(first.size(3)),
        int(output.size(3)),
        int(output.size(4) / 4),
        reinterpret_cast<float4*>(ptr<float>(output)));
    return;
  }
  LaunchKernel(dim3((n + 255) / 256, descriptors.size(0), 1), 256, D.unwrap())(
      minimax_h3_vae_output_spatial_blend_write_kernel, tiles, layout(first), layout(output), ptr<float>(output));
}

__global__ void minimax_h3_vae_output_temporal_blend_write_kernel(
    const float* src,
    const float* prev,
    float* dst,
    Layout in,
    Layout tail,
    Layout out,
    int start,
    int count,
    int extent) {
  int64_t k = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t n = in.n[0] * in.n[1] * count * in.n[3] * in.n[4];
  if (k >= n) return;
  int x = k % in.n[4];
  k /= in.n[4];
  int y = k % in.n[3];
  k /= in.n[3];
  int t = k % count;
  k /= count;
  int c = k % in.n[1];
  int b = k / in.n[1];
  int64_t ip = (int64_t(b) * in.n[1] + c) * in.n[2] + t;
  float v = src[off(in, ip, y, x)];
  if (t < extent) {
    int64_t tp = (int64_t(b) * tail.n[1] + c) * tail.n[2] + tail.n[2] - extent + t;
    v = blend(prev[off(tail, tp, y, x)], v, t, extent);
  }
  int64_t op = (int64_t(b) * out.n[1] + c) * out.n[2] + start + t;
  dst[off(out, op, y, x)] = v;
}
__global__ void minimax_h3_vae_output_temporal_blend_write_kernel_vec(
    const float4* src,
    const float4* prev,
    float4* dst,
    FastLayout in,
    FastLayout tail,
    FastLayout out,
    int start,
    int count,
    int extent,
    int plane4,
    int n4,
    float inv_extent) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n4) return;
  int xy = i % plane4;
  int p = i / plane4;
  int t = p % count;
  int bc = p / count;
  int b = bc / in.c, c = bc % in.c;
  float4 v = src[int64_t(b) * in.b4 + int64_t(c) * in.c4 + int64_t(t) * in.t4 + xy];
  if (t < extent) {
    float4 a = prev[int64_t(b) * tail.b4 + int64_t(c) * tail.c4 + int64_t(tail.t - extent + t) * tail.t4 + xy];
    v = mix4(a, v, float(t) * inv_extent);
  }
  dst[int64_t(b) * out.b4 + int64_t(c) * out.c4 + int64_t(start + t) * out.t4 + xy] = v;
}
void temporal(
    tvm::ffi::TensorView src,
    tvm::ffi::TensorView prev,
    tvm::ffi::TensorView dst,
    int64_t start,
    int64_t count,
    int64_t extent) {
  auto d = dst.device();
  int64_t n = src.size(0) * src.size(1) * count * src.size(3) * src.size(4);
  if (spatial4(src) && spatial4(prev) && spatial4(dst) && n / 4 < INT_MAX) {
    host::LaunchKernel((n / 4 + 255) / 256, 256, d)(
        minimax_h3_vae_output_temporal_blend_write_kernel_vec,
        reinterpret_cast<const float4*>(ptr<float>(src)),
        reinterpret_cast<const float4*>(ptr<float>(prev)),
        reinterpret_cast<float4*>(ptr<float>(dst)),
        fast_layout(src),
        fast_layout(prev),
        fast_layout(dst),
        int(start),
        int(count),
        int(extent),
        int(src.size(3) * src.size(4) / 4),
        int(n / 4),
        extent ? 1.f / float(extent) : 0.f);
    return;
  }
  host::LaunchKernel((n + 255) / 256, 256, d)(
      minimax_h3_vae_output_temporal_blend_write_kernel,
      ptr<float>(src),
      ptr<float>(prev),
      ptr<float>(dst),
      layout(src),
      layout(prev),
      layout(dst),
      int(start),
      int(count),
      int(extent));
}

__global__ void minimax_h3_vae_output_denorm_clamp_kernel(
    const float* src, float* dst, Layout in, float m0, float m1, float m2, float s0, float s1, float s2, int64_t n) {
  int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  int64_t k = i;
  int x = k % in.n[4];
  k /= in.n[4];
  int y = k % in.n[3];
  int64_t p = k / in.n[3];
  int c = (p / in.n[2]) % 3;
  float mean = c == 0 ? m0 : c == 1 ? m1 : m2, sd = c == 0 ? s0 : c == 1 ? s1 : s2;
  float v = __fdiv_rn(__fsub_rn(src[off(in, p, y, x)], mean), sd);
  dst[i] = v < 0.f ? 0.f : v > 1.f ? 1.f : v;  // Preserve NaN semantics.
}
__device__ inline float affine_clamp(float x, float scale, float bias) {
  float v = fmaf(x, scale, bias);
  return v < 0.f ? 0.f : v > 1.f ? 1.f : v;
}
__global__ void minimax_h3_vae_output_denorm_clamp_kernel_vec(
    const float4* src, float4* dst, int n4, int channel4, float a0, float a1, float a2, float b0, float b1, float b2) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n4) return;
  int c = (i / channel4) % 3;
  float a = c == 0 ? a0 : c == 1 ? a1 : a2, b = c == 0 ? b0 : c == 1 ? b1 : b2;
  float4 v = src[i];
  dst[i] = {affine_clamp(v.x, a, b), affine_clamp(v.y, a, b), affine_clamp(v.z, a, b), affine_clamp(v.w, a, b)};
}
void denorm(
    tvm::ffi::TensorView src,
    tvm::ffi::TensorView dst,
    double m0,
    double m1,
    double m2,
    double s0,
    double s1,
    double s2) {
  int64_t n = 1;
  for (int d = 0; d < 5; ++d)
    n *= src.size(d);
  bool contiguous = true;
  int64_t stride = 1;
  for (int d = 4; d >= 0; --d) {
    if (src.size(d) > 1 && src.stride(d) != stride) contiguous = false;
    stride *= src.size(d);
  }
  int64_t channel = src.size(2) * src.size(3) * src.size(4);
  if (contiguous && n / 4 < INT_MAX && channel % 4 == 0 && reinterpret_cast<uintptr_t>(ptr<float>(src)) % 16 == 0) {
    float a0 = 1.f / float(s0), a1 = 1.f / float(s1), a2 = 1.f / float(s2);
    host::LaunchKernel((n / 4 + 255) / 256, 256, dst.device())(
        minimax_h3_vae_output_denorm_clamp_kernel_vec,
        reinterpret_cast<const float4*>(ptr<float>(src)),
        reinterpret_cast<float4*>(ptr<float>(dst)),
        int(n / 4),
        int(channel / 4),
        a0,
        a1,
        a2,
        -float(m0) * a0,
        -float(m1) * a1,
        -float(m2) * a2);
    return;
  }
  host::LaunchKernel((n + 255) / 256, 256, dst.device())(
      minimax_h3_vae_output_denorm_clamp_kernel,
      ptr<float>(src),
      ptr<float>(dst),
      layout(src),
      float(m0),
      float(m1),
      float(m2),
      float(s0),
      float(s1),
      float(s2),
      n);
}
}  // namespace sglang::minimax_h3_vae_output
