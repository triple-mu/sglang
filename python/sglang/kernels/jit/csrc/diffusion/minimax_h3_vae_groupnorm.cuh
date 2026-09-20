// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "minimax_h3_vae.cuh"
#include <climits>

namespace sglang::minimax_h3_vae {
struct GroupLayout {
  int64_t strides[5];
  int channels, time, height, width, groups;
  bool isolated;
};

__device__ inline int64_t group_offset(GroupLayout l, int row, int index) {
  int group = row % l.groups;
  int batch_time = row / l.groups;
  int x = index % l.width;
  index /= l.width;
  int y = index % l.height;
  index /= l.height;
  int time = l.isolated ? batch_time % l.time : index % l.time;
  int batch = l.isolated ? batch_time / l.time : batch_time;
  int channel = group * (l.channels / l.groups) + (l.isolated ? index : index / l.time);
  return batch * l.strides[0] + channel * l.strides[1] + time * l.strides[2] + y * l.strides[3] + x * l.strides[4];
}

__device__ inline int group_channel(GroupLayout l, int row, int index) {
  return (row % l.groups) * (l.channels / l.groups) + index / (l.height * l.width * (l.isolated ? 1 : l.time));
}

__device__ inline int64_t group_output_offset(GroupLayout l, int row, int index) {
  int group = row % l.groups, batch_time = row / l.groups;
  int x = index % l.width;
  index /= l.width;
  int y = index % l.height;
  index /= l.height;
  int time = l.isolated ? batch_time % l.time : index % l.time;
  int batch = l.isolated ? batch_time / l.time : batch_time;
  int channel = group * (l.channels / l.groups) + (l.isolated ? index : index / l.time);
  return ((((int64_t(batch) * l.channels + channel) * l.time + time) * l.height + y) * l.width + x);
}

__device__ inline float norm_activation(float x, float mean, float inverse, float gamma, float beta) {
  float value = (x - mean) * inverse;
  value = value * gamma + beta;
  return value / (1.f + expf(-value));
}

__global__ void minimax_h3_gn_small(
    const float* x, const float* gamma, const float* beta, float* out, GroupLayout l, int count, float eps) {
  int row = blockIdx.x;
  float sum = 0.f;
  for (int i = threadIdx.x; i < count; i += blockDim.x)
    sum += x[group_offset(l, row, i)];
  float mean = block_reduce<false>(sum) / count;
  float m2 = 0.f;
  for (int i = threadIdx.x; i < count; i += blockDim.x) {
    float delta = x[group_offset(l, row, i)] - mean;
    m2 += delta * delta;
  }
  float inverse = rsqrtf(block_reduce<false>(m2) / count + eps);
  for (int i = threadIdx.x; i < count; i += blockDim.x) {
    int channel = group_channel(l, row, i);
    out[group_output_offset(l, row, i)] =
        norm_activation(x[group_offset(l, row, i)], mean, inverse, gamma[channel], beta[channel]);
  }
}

// Central moments avoid E[x*x]-E[x]^2 cancellation. The tail contributes its
// actual count to both the weighted mean and the merged second moment.
__global__ void minimax_h3_gn_partial(const float* x, float* partial, GroupLayout l, int count, int parts) {
  int row = blockIdx.x, part = blockIdx.y, start = part * 4096;
  int length = min(4096, count - start);
  float sum = 0.f;
  for (int i = threadIdx.x; i < length; i += blockDim.x)
    sum += x[group_offset(l, row, start + i)];
  float mean = block_reduce<false>(sum) / length;
  float m2 = 0.f;
  for (int i = threadIdx.x; i < length; i += blockDim.x) {
    float delta = x[group_offset(l, row, start + i)] - mean;
    m2 += delta * delta;
  }
  m2 = block_reduce<false>(m2);
  if (threadIdx.x == 0) {
    int64_t offset = (int64_t(row) * parts + part) * 2;
    partial[offset] = mean;
    partial[offset + 1] = m2;
  }
}

__global__ void minimax_h3_gn_merge(const float* partial, float* stats, int count, int parts, float eps) {
  int row = blockIdx.x;
  float sum = 0.f;
  for (int part = threadIdx.x; part < parts; part += blockDim.x)
    sum += partial[(int64_t(row) * parts + part) * 2] * min(4096, count - part * 4096);
  float mean = block_reduce<false>(sum) / count;
  float m2 = 0.f;
  for (int part = threadIdx.x; part < parts; part += blockDim.x) {
    int64_t offset = (int64_t(row) * parts + part) * 2;
    float delta = partial[offset] - mean;
    m2 += partial[offset + 1] + delta * delta * min(4096, count - part * 4096);
  }
  float inverse = rsqrtf(block_reduce<false>(m2) / count + eps);
  if (threadIdx.x == 0) {
    stats[row * 2] = mean;
    stats[row * 2 + 1] = inverse;
  }
}

__global__ void minimax_h3_gn_apply(
    const float* x, const float* gamma, const float* beta, const float* stats, float* out, GroupLayout l, int count) {
  int row = blockIdx.x, start = blockIdx.y * 4096;
  float mean = stats[row * 2], inverse = stats[row * 2 + 1];
  for (int i = start + threadIdx.x; i < min(start + 4096, count); i += blockDim.x) {
    int channel = group_channel(l, row, i);
    out[group_output_offset(l, row, i)] =
        norm_activation(x[group_offset(l, row, i)], mean, inverse, gamma[channel], beta[channel]);
  }
}

// The released encoder uses contiguous NCTHW with power-of-two spatial
// planes. Decode the row once, then address four adjacent pixels together.
// For isolated GN, each channel advance skips the other temporal frames.
__device__ inline int64_t gn_fast_offset(int64_t base, int index, int channel_shift, int64_t channel_gap) {
  return base + index + int64_t(index >> channel_shift) * channel_gap;
}

__device__ inline float gn_fast_activation(float x, float mean, float inverse, float gamma, float beta) {
  float value = (x - mean) * inverse;
  value = value * gamma + beta;
  return value / (1.f + __expf(-value));
}

// Retain input vectors through both central-moment reductions and the output
// epilogue. Groups of 8192 also fit in registers, avoiding the three-launch
// partial/merge/apply path used for larger groups.
template <int Vectors>
__global__ void minimax_h3_gn_contiguous_small(
    const float* x,
    const float* gamma,
    const float* beta,
    float* out,
    GroupLayout l,
    int count,
    int channel_shift,
    int64_t channel_gap,
    float eps) {
  int row = blockIdx.x, vectors = count / 4;
  int64_t base = group_offset(l, row, 0);
  int first_channel = (row % l.groups) * (l.channels / l.groups);
  float4 values[Vectors];
  float sum = 0.f;
#pragma unroll
  for (int j = 0; j < Vectors; ++j) {
    int vector_index = threadIdx.x + j * 256;
    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
    if (vector_index < vectors) {
      int64_t offset = gn_fast_offset(base, vector_index * 4, channel_shift, channel_gap);
      v = *reinterpret_cast<const float4*>(x + offset);
      sum += (v.x + v.y) + (v.z + v.w);
    }
    values[j] = v;
  }
  float mean = block_reduce<false>(sum) / count;
  float m2 = 0.f;
#pragma unroll
  for (int j = 0; j < Vectors; ++j) {
    if (threadIdx.x + j * 256 < vectors) {
      float4 v = values[j];
      float a = v.x - mean, b = v.y - mean, c = v.z - mean, d = v.w - mean;
      m2 += (a * a + b * b) + (c * c + d * d);
    }
  }
  float inverse = rsqrtf(block_reduce<false>(m2) / count + eps);
#pragma unroll
  for (int j = 0; j < Vectors; ++j) {
    int index = (threadIdx.x + j * 256) * 4;
    if (index < count) {
      int channel = first_channel + (index >> channel_shift);
      float g = gamma[channel], b = beta[channel];
      float4 v = values[j];
      v = make_float4(
          gn_fast_activation(v.x, mean, inverse, g, b),
          gn_fast_activation(v.y, mean, inverse, g, b),
          gn_fast_activation(v.z, mean, inverse, g, b),
          gn_fast_activation(v.w, mean, inverse, g, b));
      int64_t offset = gn_fast_offset(base, index, channel_shift, channel_gap);
      *reinterpret_cast<float4*>(out + offset) = v;
    }
  }
}

__global__ void minimax_h3_gn_contiguous_partial(
    const float* x, float* partial, GroupLayout l, int count, int parts, int channel_shift, int64_t channel_gap) {
  int row = blockIdx.x, part = blockIdx.y, start = part * 4096;
  int length = min(4096, count - start);
  int64_t base = group_offset(l, row, 0);
  float4 values[4];
  float sum = 0.f;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    int local = (threadIdx.x + j * 256) * 4;
    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
    if (local < length) {
      int64_t offset = gn_fast_offset(base, start + local, channel_shift, channel_gap);
      v = *reinterpret_cast<const float4*>(x + offset);
      sum += (v.x + v.y) + (v.z + v.w);
    }
    values[j] = v;
  }
  float mean = block_reduce<false>(sum) / length;
  float m2 = 0.f;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    if ((threadIdx.x + j * 256) * 4 < length) {
      float4 v = values[j];
      float a = v.x - mean, b = v.y - mean, c = v.z - mean, d = v.w - mean;
      m2 += (a * a + b * b) + (c * c + d * d);
    }
  }
  m2 = block_reduce<false>(m2);
  if (threadIdx.x == 0) {
    int64_t offset = (int64_t(row) * parts + part) * 2;
    partial[offset] = mean;
    partial[offset + 1] = m2;
  }
}

__global__ void minimax_h3_gn_contiguous_apply(
    const float* x,
    const float* gamma,
    const float* beta,
    const float* stats,
    float* out,
    GroupLayout l,
    int count,
    int channel_shift,
    int64_t channel_gap) {
  int row = blockIdx.x, start = blockIdx.y * 4096;
  int64_t base = group_offset(l, row, 0);
  int first_channel = (row % l.groups) * (l.channels / l.groups);
  float mean = stats[row * 2], inverse = stats[row * 2 + 1];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    int index = start + (threadIdx.x + j * 256) * 4;
    if (index < min(start + 4096, count)) {
      int64_t offset = gn_fast_offset(base, index, channel_shift, channel_gap);
      int channel = first_channel + (index >> channel_shift);
      float g = gamma[channel], b = beta[channel];
      float4 v = *reinterpret_cast<const float4*>(x + offset);
      v = make_float4(
          gn_fast_activation(v.x, mean, inverse, g, b),
          gn_fast_activation(v.y, mean, inverse, g, b),
          gn_fast_activation(v.z, mean, inverse, g, b),
          gn_fast_activation(v.w, mean, inverse, g, b));
      *reinterpret_cast<float4*>(out + offset) = v;
    }
  }
}

void group_norm_silu(
    tvm::ffi::TensorView x,
    tvm::ffi::TensorView gamma,
    tvm::ffi::TensorView beta,
    tvm::ffi::TensorView partial,
    tvm::ffi::TensorView stats,
    tvm::ffi::TensorView out,
    int64_t groups,
    bool isolated,
    double eps) {
  using namespace host;
  auto B = SymbolicSize{"B"}, C = SymbolicSize{"C"}, T = SymbolicSize{"T"}, H = SymbolicSize{"H"},
       W = SymbolicSize{"W"};
  auto D = SymbolicDevice{};
  D.set_options<kDLCUDA>();
  TensorMatcher({B, C, T, H, W}).with_strides({-1, -1, -1, -1, -1}).with_dtype<float>().with_device(D).verify(x);
  TensorMatcher({B, C, T, H, W}).with_dtype<float>().with_device(D).verify(out);
  TensorMatcher({C}).with_dtype<float>().with_device(D).verify(gamma);
  TensorMatcher({C}).with_dtype<float>().with_device(D).verify(beta);
  RuntimeCheck(groups > 0 && C.unwrap() % groups == 0, "invalid GroupNorm groups");
  GroupLayout l{};
  for (int i = 0; i < 5; ++i)
    l.strides[i] = x.stride(i);
  l.channels = C.unwrap();
  l.time = T.unwrap();
  l.height = H.unwrap();
  l.width = W.unwrap();
  l.groups = groups;
  l.isolated = isolated;
  int64_t count64 = (C.unwrap() / groups) * H.unwrap() * W.unwrap() * (isolated ? 1 : T.unwrap());
  int64_t rows64 = B.unwrap() * groups * (isolated ? T.unwrap() : 1);
  RuntimeCheck(
      count64 > 0 && count64 < INT_MAX && rows64 > 0 && rows64 < INT_MAX, "GroupNorm dimensions exceed kernel limits");
  int count = count64, rows = rows64, parts = (count + 4095) / 4096;
  int64_t channel_span = H.unwrap() * W.unwrap() * (isolated ? 1 : T.unwrap());
  bool contiguous = true;
  int64_t expected_stride = 1;
  for (int d = 4; d >= 0; --d) {
    if (x.size(d) > 1 && x.stride(d) != expected_stride) contiguous = false;
    expected_stride *= x.size(d);
  }
  bool vectorized = contiguous && channel_span >= 4 && (channel_span & (channel_span - 1)) == 0 &&
                    reinterpret_cast<uintptr_t>(ptr<float>(x)) % 16 == 0;
  if (vectorized) {
    int channel_shift = 0;
    for (int64_t span = channel_span; span > 1; span >>= 1)
      ++channel_shift;
    int64_t channel_gap = isolated ? (T.unwrap() - 1) * channel_span : 0;
    if (count <= 4096) {
      LaunchKernel(rows, 256, D.unwrap())(
          minimax_h3_gn_contiguous_small<4>,
          ptr<float>(x),
          ptr<float>(gamma),
          ptr<float>(beta),
          ptr<float>(out),
          l,
          count,
          channel_shift,
          channel_gap,
          float(eps));
    } else if (count <= 8192) {
      LaunchKernel(rows, 256, D.unwrap())(
          minimax_h3_gn_contiguous_small<8>,
          ptr<float>(x),
          ptr<float>(gamma),
          ptr<float>(beta),
          ptr<float>(out),
          l,
          count,
          channel_shift,
          channel_gap,
          float(eps));
    } else {
      TensorMatcher({rows, parts, 2}).with_dtype<float>().with_device(D).verify(partial);
      TensorMatcher({rows, 2}).with_dtype<float>().with_device(D).verify(stats);
      LaunchKernel(dim3(rows, parts, 1), 256, D.unwrap())(
          minimax_h3_gn_contiguous_partial,
          ptr<float>(x),
          ptr<float>(partial),
          l,
          count,
          parts,
          channel_shift,
          channel_gap);
      LaunchKernel(rows, 256, D.unwrap())(
          minimax_h3_gn_merge, ptr<float>(partial), ptr<float>(stats), count, parts, float(eps));
      LaunchKernel(dim3(rows, parts, 1), 256, D.unwrap())(
          minimax_h3_gn_contiguous_apply,
          ptr<float>(x),
          ptr<float>(gamma),
          ptr<float>(beta),
          ptr<float>(stats),
          ptr<float>(out),
          l,
          count,
          channel_shift,
          channel_gap);
    }
    return;
  }
  if (count <= 4096) {
    LaunchKernel(rows, 256, D.unwrap())(
        minimax_h3_gn_small, ptr<float>(x), ptr<float>(gamma), ptr<float>(beta), ptr<float>(out), l, count, float(eps));
  } else {
    TensorMatcher({rows, parts, 2}).with_dtype<float>().with_device(D).verify(partial);
    TensorMatcher({rows, 2}).with_dtype<float>().with_device(D).verify(stats);
    LaunchKernel(dim3(rows, parts, 1), 256, D.unwrap())(
        minimax_h3_gn_partial, ptr<float>(x), ptr<float>(partial), l, count, parts);
    LaunchKernel(rows, 256, D.unwrap())(
        minimax_h3_gn_merge, ptr<float>(partial), ptr<float>(stats), count, parts, float(eps));
    LaunchKernel(dim3(rows, parts, 1), 256, D.unwrap())(
        minimax_h3_gn_apply,
        ptr<float>(x),
        ptr<float>(gamma),
        ptr<float>(beta),
        ptr<float>(stats),
        ptr<float>(out),
        l,
        count);
  }
}
}  // namespace sglang::minimax_h3_vae
