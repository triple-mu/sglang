#pragma once

// Low-precision Ulysses all-to-all for SageAttention SM120 consumers, local-grid
// (ALIGN-128) form.
//
// Every rank holds a 128-token-aligned shard of the packed sequence, so no
// 32-token Q group and no 64-token K block straddles a rank boundary and the
// per-group scales computed on the shard are final. The sender quantises Q/K
// to INT8 and V to FP8 E4M3 in one pass per operand and writes them straight
// into a destination-major payload `[world_size, chunk_bytes]`; after the
// all-to-all the receiver unpacks the chunks into the operand layout of the
// SM120 Sage block-sparse kernel (Q/K `[B, h, S, 128]`, V `[B, h, 128, S]`
// with the 16-token permutation, per-group scales). Only two per-channel
// statistics need the whole sequence -- the K mean and the V amax -- and they
// travel through one small fp32 all-gather the caller performs.
//
// The arithmetic is a port of flashinfer's `ulysses_lowp` kernels (themselves
// a port of the pinned SageAttention fork): element order, `1e-7` floors,
// `127 / amax` reciprocal multiply, `cvt.rni.sat.s8` and `cvt.rn.satfinite`
// conversions are load-bearing for parity with the Sage quantiser and must be
// compiled with `--use_fast_math` like the original.

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/ffi.h>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/warp.cuh>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>
#include <type_traits>

namespace sglang {

namespace ulysses_lowp_sage {

constexpr uint32_t kHeadDim = 128;
constexpr uint32_t kQGroup = 32;
constexpr uint32_t kKGroup = 64;
/// Fixed sequence-chunk length of the two-stage K-sum / V-amax reduction.
constexpr uint32_t kStatsChunkTokens = 256;
/// The FP8 V scale is `amax / kVScaleMax`, the Sage consumer's convention.
constexpr float kVScaleMax = 2.25f;
constexpr uint32_t kShardAlignment = 128;

/// Byte layout of one destination chunk (all sizes derive from the shard geometry).
struct ChunkSpec {
  int64_t main_bytes;      ///< one INT8 / FP8 operand section: B * L * h * 128
  int64_t q_scale_offset;  ///< 3 * main_bytes
  int64_t k_scale_offset;  ///< q_scale_offset + B * h * (L / 32) * 4
  int64_t chunk_bytes;     ///< k_scale_offset + B * h * (L / 64) * 4, rounded up to 128
};

SGL_DEVICE_HOST constexpr auto chunk_spec(int64_t batch, int64_t local_sequence, int64_t local_heads) -> ChunkSpec {
  ChunkSpec spec{};
  spec.main_bytes = batch * local_sequence * local_heads * kHeadDim;
  spec.q_scale_offset = 3 * spec.main_bytes;
  spec.k_scale_offset = spec.q_scale_offset + batch * local_heads * (local_sequence / kQGroup) * 4;
  const int64_t raw = spec.k_scale_offset + batch * local_heads * (local_sequence / kKGroup) * 4;
  spec.chunk_bytes = (raw + 127) / 128 * 128;
  return spec;
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

template <typename T>
SGL_DEVICE T from_float(float v) {
  if constexpr (std::is_same_v<T, half>) {
    return __float2half_rn(v);
  } else {
    return __float2bfloat16_rn(v);
  }
}

template <typename T>
struct packed2;
template <>
struct packed2<half> {
  using type = half2;
};
template <>
struct packed2<nv_bfloat16> {
  using type = nv_bfloat162;
};

/// Four fp32 -> four E4M3 with round-to-nearest, saturating to the finite range.
SGL_DEVICE void floatx4_to_e4m3x4(uint32_t* dest, const float* source0, const float* source1) {
  asm volatile(
      "{\n"
      ".reg .b16 lo;\n"
      ".reg .b16 hi;\n"
      "cvt.rn.satfinite.e4m3x2.f32   lo, %2, %1;\n"
      "cvt.rn.satfinite.e4m3x2.f32   hi, %4, %3;\n"
      "mov.b32 %0, {lo, hi};\n"
      "}"
      : "=r"(dest[0])
      : "f"(source0[0]), "f"(source0[1]), "f"(source1[0]), "f"(source1[1]));
}

SGL_DEVICE int8_t float_to_int8_rn(float x) {
  uint32_t dst;
  asm volatile("cvt.rni.sat.s8.f32 %0, %1;" : "=r"(dst) : "f"(x));
  return static_cast<int8_t>(dst & 0xFFu);
}

SGL_DEVICE float block_reduce_max(float val) {
  static __shared__ float shared[32];
  const uint32_t lane = threadIdx.x & 31u;
  const uint32_t wid = threadIdx.x >> 5;
  val = device::warp::reduce_max(val);
  if (lane == 0) shared[wid] = val;
  __syncthreads();
  val = (threadIdx.x < (blockDim.x >> 5)) ? shared[lane] : -1e20f;
  return device::warp::reduce_max(val);
}

/// |x| max over 8 values in packed pairs; same fp32 result as the scalar loop (see flashinfer).
template <typename T>
SGL_DEVICE float packed_abs_max8(const T (&x)[8]) {
  using T2 = typename packed2<T>::type;
  const T2* pairs = reinterpret_cast<const T2*>(&x[0]);
  T2 acc = __habs2(pairs[0]);
#pragma unroll
  for (uint32_t j = 1; j < 4; ++j) {
    acc = __hmax2(acc, __habs2(pairs[j]));
  }
  return fmaxf(to_float<T>(acc.x), to_float<T>(acc.y));
}

/// Element offset of (token, batch, local_head, d) inside one operand section.
SGL_DEVICE uint64_t section_offset(
    uint32_t token, uint32_t batch_id, uint32_t batch_size, uint32_t local_head, uint32_t local_heads, uint32_t d) {
  return ((static_cast<uint64_t>(token) * batch_size + batch_id) * local_heads + local_head) * kHeadDim + d;
}

}  // namespace details

// ---------------------------------------------------------------------------
// Statistics: per-channel K sum and V amax over the local shard.
// Stage 1 reduces fixed 256-token chunks (grid z); stage 2 combines the chunk
// partials in ascending chunk order, so the fp32 sum is bit-identical run to
// run and identical on every rank once the partials are all-gathered.
// ---------------------------------------------------------------------------
template <typename T, bool kUsePDL>
__global__ void KSumVAmaxPartialKernel(
    const T* __restrict__ k,
    const T* __restrict__ v,
    float* __restrict__ k_partial,
    float* __restrict__ v_partial,
    uint32_t num_tokens,
    uint32_t num_heads,
    uint32_t num_chunks,
    int64_t k_stride_batch,
    int64_t k_stride_token,
    int64_t k_stride_head,
    int64_t v_stride_batch,
    int64_t v_stride_token,
    int64_t v_stride_head) {
  constexpr uint32_t kTokenLanes = 2;
  constexpr uint32_t kThreads = kTokenLanes * kHeadDim;
  const uint32_t thread_id = threadIdx.x;
  const uint32_t d_id = thread_id % kHeadDim;
  const uint32_t token_lane = thread_id / kHeadDim;
  const uint32_t head_id = blockIdx.x;
  const uint32_t batch_id = blockIdx.y;
  const uint32_t chunk_id = blockIdx.z;
  const uint32_t token_begin = chunk_id * kStatsChunkTokens;
  const uint32_t token_end = min(token_begin + kStatsChunkTokens, num_tokens);

  float local_sum = 0.0f;
  float local_amax = 0.0f;
  device::PDLWaitPrimary<kUsePDL>();
  for (uint32_t token_id = token_begin + token_lane; token_id < token_end; token_id += kTokenLanes) {
    const uint64_t k_offset = static_cast<uint64_t>(batch_id) * k_stride_batch +
                              static_cast<uint64_t>(token_id) * k_stride_token +
                              static_cast<uint64_t>(head_id) * k_stride_head + d_id;
    const uint64_t v_offset = static_cast<uint64_t>(batch_id) * v_stride_batch +
                              static_cast<uint64_t>(token_id) * v_stride_token +
                              static_cast<uint64_t>(head_id) * v_stride_head + d_id;
    local_sum += details::to_float(k[k_offset]);
    local_amax = fmaxf(local_amax, fabsf(details::to_float(v[v_offset])));
  }

  __shared__ float shared_sum[kThreads];
  __shared__ float shared_amax[kThreads];
  shared_sum[thread_id] = local_sum;
  shared_amax[thread_id] = local_amax;
  __syncthreads();

  if (token_lane == 0) {
    const uint64_t out =
        (((static_cast<uint64_t>(batch_id) * num_heads + head_id) * num_chunks) + chunk_id) * kHeadDim + d_id;
    k_partial[out] = shared_sum[d_id] + shared_sum[kHeadDim + d_id];
    v_partial[out] = fmaxf(shared_amax[d_id], shared_amax[kHeadDim + d_id]);
  }
  device::PDLTriggerSecondary<kUsePDL>();
}

/// Writes the `[2, B, H, 128]` record (K sum, then V amax) `replicas` times at `replica_stride`
/// floats apart -- the replicated all-gather payload -- and once more into `own`.
template <bool kUsePDL>
__global__ void KSumVAmaxCombineKernel(
    const float* __restrict__ k_partial,
    const float* __restrict__ v_partial,
    float* __restrict__ stats,
    float* __restrict__ own,
    uint32_t num_heads,
    uint32_t num_chunks,
    uint32_t replicas,
    uint64_t replica_stride,
    uint64_t half) {
  const uint32_t d_id = threadIdx.x;
  const uint32_t head_id = blockIdx.x;
  const uint32_t batch_id = blockIdx.y;
  const uint64_t base = (static_cast<uint64_t>(batch_id) * num_heads + head_id) * num_chunks;
  float s = 0.0f;
  float m = 0.0f;
  device::PDLWaitPrimary<kUsePDL>();
  for (uint32_t c = 0; c < num_chunks; ++c) {
    s += k_partial[(base + c) * kHeadDim + d_id];
    m = fmaxf(m, v_partial[(base + c) * kHeadDim + d_id]);
  }
  const uint64_t out = (static_cast<uint64_t>(batch_id) * num_heads + head_id) * kHeadDim + d_id;
  for (uint32_t r = 0; r < replicas; ++r) {
    stats[r * replica_stride + out] = s;
    stats[r * replica_stride + half + out] = m;
  }
  own[out] = s;
  own[half + out] = m;
  device::PDLTriggerSecondary<kUsePDL>();
}

/// `k_mean = bf16(sum_r k_sum[r] / used)` and `v_scale = max_r v_amax[r] / 2.25` from the gathered
/// `[W, 2, B, H, 128]` records; the sum runs in torch's reduction order (four round-robin
/// accumulators, then combined in order) so it matches `sum(dim=0)` bit for bit. `v_scale_local`
/// receives the heads this rank attends to.
template <typename T>
__global__ void FinalizeStatsKernel(
    const float* __restrict__ gathered,
    T* __restrict__ k_mean,
    float* __restrict__ v_scale,
    float* __restrict__ v_scale_local,
    uint32_t world_size,
    uint32_t num_heads,
    uint32_t local_heads,
    uint32_t rank,
    uint32_t used_sequence,
    uint64_t half) {
  const uint32_t d_id = threadIdx.x;
  const uint32_t head_id = blockIdx.x;
  const uint32_t batch_id = blockIdx.y;
  const uint64_t out = (static_cast<uint64_t>(batch_id) * num_heads + head_id) * kHeadDim + d_id;
  const uint64_t stride = 2 * half;
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
  float m = 0.f;
  for (uint32_t r = 0; r < world_size; ++r) {
    acc[r % 4] += gathered[r * stride + out];
    m = fmaxf(m, gathered[r * stride + half + out]);
  }
  const float sum = ((acc[0] + acc[1]) + acc[2]) + acc[3];
  k_mean[out] = details::from_float<T>(__fdiv_rn(sum, static_cast<float>(used_sequence)));
  const float scale = __fdiv_rn(m, kVScaleMax);
  v_scale[out] = scale;
  const uint32_t first = rank * local_heads;
  if (head_id >= first && head_id < first + local_heads) {
    v_scale_local[(static_cast<uint64_t>(batch_id) * local_heads + head_id - first) * kHeadDim + d_id] = scale;
  }
}

// ---------------------------------------------------------------------------
// Fused per-group amax + INT8 quantise + pack for Q (GROUP 32, raw values) and
// K (GROUP 64, minus the bf16 channel mean). One CTA per (group, head, batch):
// the block amax becomes the group's scale, every token is quantised with it
// and written to the destination chunk of the rank that owns the head.
//
// `used_sequence` names the live rows of the global sequence: the one K group
// that mixes live and zero-padded rows excludes the padded rows from its amax
// (a zero row would contribute |0 - mean|), while still quantising them.
// ---------------------------------------------------------------------------
template <typename T, uint32_t GROUP, bool kSubMean, bool kUsePDL>
__global__ void QuantInt8FusedAmaxPackKernel(
    const T* __restrict__ input,
    const T* __restrict__ mean,
    uint8_t* __restrict__ output,
    uint8_t* __restrict__ own_output,
    uint32_t rank,
    uint32_t local_sequence,
    uint32_t global_offset,
    uint32_t num_heads,
    uint32_t local_heads,
    uint32_t batch_size,
    uint64_t chunk_bytes,
    uint64_t section_offset,
    uint64_t scale_offset,
    uint32_t slots,
    uint32_t used_sequence,
    int64_t stride_batch,
    int64_t stride_token,
    int64_t stride_head) {
  static_assert(GROUP == kQGroup || GROUP == kKGroup);
  constexpr uint32_t kPack = 8;
  constexpr uint32_t kThreadsPerToken = kHeadDim / kPack;
  const uint32_t slot = blockIdx.x;
  const uint32_t head_id = blockIdx.y;
  const uint32_t batch_id = blockIdx.z;
  const uint32_t thread_id = threadIdx.x;
  const uint32_t token_in_group = thread_id / kThreadsPerToken;
  const uint32_t d_base = thread_id % kThreadsPerToken * kPack;
  const uint32_t local_token = slot * GROUP + token_in_group;
  const uint32_t global_token = global_offset + local_token;
  const uint32_t destination = head_id / local_heads;
  const uint32_t local_head = head_id % local_heads;
  uint8_t* chunk = destination == rank ? own_output : output + static_cast<uint64_t>(destination) * chunk_bytes;

  T x_val[kPack];
  float x_val_float[kPack];
  float amax_val = 0.0000001f;
  device::PDLWaitPrimary<kUsePDL>();
  const uint64_t input_offset = static_cast<uint64_t>(batch_id) * stride_batch +
                                static_cast<uint64_t>(local_token) * stride_token +
                                static_cast<uint64_t>(head_id) * stride_head + d_base;
  *reinterpret_cast<float4*>(&x_val[0]) = *reinterpret_cast<const float4*>(input + input_offset);
  if constexpr (kSubMean) {
    T mean_val[kPack];
    const uint64_t mean_offset = (static_cast<uint64_t>(batch_id) * num_heads + head_id) * kHeadDim + d_base;
    *reinterpret_cast<float4*>(&mean_val[0]) = *reinterpret_cast<const float4*>(mean + mean_offset);
#pragma unroll
    for (uint32_t j = 0; j < kPack; ++j) {
      x_val_float[j] = details::to_float(x_val[j]) - details::to_float(mean_val[j]);
    }
  } else {
#pragma unroll
    for (uint32_t j = 0; j < kPack; ++j) {
      x_val_float[j] = details::to_float(x_val[j]);
    }
  }
  // Tail repair: padded rows contribute nothing to the amax.
  if (global_token < used_sequence) {
    if constexpr (kSubMean) {
#pragma unroll
      for (uint32_t j = 0; j < kPack; ++j) {
        amax_val = fmaxf(amax_val, fabsf(x_val_float[j]));
      }
    } else {
      amax_val = fmaxf(amax_val, details::packed_abs_max8<T>(x_val));
    }
  }

  const float block_amax_val = details::block_reduce_max(amax_val);
  __shared__ float shared_group_amax;
  if (thread_id == 0) {
    shared_group_amax = block_amax_val;
    float* scale_output = reinterpret_cast<float*>(chunk + scale_offset);
    scale_output[(static_cast<uint64_t>(batch_id) * local_heads + local_head) * slots + slot] = block_amax_val / 127.0f;
  }
  __syncthreads();

  const float reciprocal_scale = 127.0f / shared_group_amax;
  char4 quantized[2];
#pragma unroll
  for (uint32_t j = 0; j < 2; ++j) {
    quantized[j] = make_char4(
        details::float_to_int8_rn(x_val_float[j * 4 + 0] * reciprocal_scale),
        details::float_to_int8_rn(x_val_float[j * 4 + 1] * reciprocal_scale),
        details::float_to_int8_rn(x_val_float[j * 4 + 2] * reciprocal_scale),
        details::float_to_int8_rn(x_val_float[j * 4 + 3] * reciprocal_scale));
  }
  const uint64_t packed_offset =
      section_offset + details::section_offset(local_token, batch_id, batch_size, local_head, local_heads, d_base);
  *reinterpret_cast<float2*>(chunk + packed_offset) = *reinterpret_cast<float2*>(&quantized[0]);
  device::PDLTriggerSecondary<kUsePDL>();
}

// V: per-channel FP8 with the caller's scale (`amax / 2.25`), written straight
// into the V section of the destination chunk (canonical token order; the Sage
// permutation is applied by the receiver).
template <typename T, bool kUsePDL>
__global__ void QuantVFP8PackKernel(
    const T* __restrict__ input,
    const float* __restrict__ scale,
    uint8_t* __restrict__ output,
    uint8_t* __restrict__ own_output,
    uint32_t rank,
    uint64_t num_packs,
    uint32_t num_heads,
    uint32_t local_heads,
    uint32_t batch_size,
    uint64_t main_bytes,
    uint64_t chunk_bytes,
    int64_t stride_batch,
    int64_t stride_token,
    int64_t stride_head) {
  constexpr uint32_t kPack = 8;
  const uint64_t pack_id = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (pack_id >= num_packs) return;

  constexpr uint32_t kPacksPerHead = kHeadDim / kPack;
  uint64_t logical_id = pack_id;
  const uint32_t d_base = static_cast<uint32_t>(logical_id % kPacksPerHead) * kPack;
  logical_id /= kPacksPerHead;
  const uint32_t local_head = static_cast<uint32_t>(logical_id % local_heads);
  logical_id /= local_heads;
  const uint32_t batch_id = static_cast<uint32_t>(logical_id % batch_size);
  const uint32_t token_id = static_cast<uint32_t>(logical_id / batch_size);
  const uint32_t destination = blockIdx.y;
  const uint32_t head_id = destination * local_heads + local_head;

  const uint64_t input_offset = static_cast<uint64_t>(batch_id) * stride_batch +
                                static_cast<uint64_t>(token_id) * stride_token +
                                static_cast<uint64_t>(head_id) * stride_head + d_base;
  const uint64_t scale_offset = (static_cast<uint64_t>(batch_id) * num_heads + head_id) * kHeadDim + d_base;
  uint8_t* chunk = destination == rank ? own_output : output + static_cast<uint64_t>(destination) * chunk_bytes;
  const uint64_t output_offset =
      2 * main_bytes + details::section_offset(token_id, batch_id, batch_size, local_head, local_heads, d_base);

  T x_val[kPack];
  float scale_val[kPack];
  float x_val_float[kPack];
  uint32_t x_val_fp8[2];

  device::PDLWaitPrimary<kUsePDL>();
  *reinterpret_cast<float4*>(&x_val[0]) = *reinterpret_cast<const float4*>(input + input_offset);
  *reinterpret_cast<float4*>(&scale_val[0]) = *reinterpret_cast<const float4*>(scale + scale_offset);
  *reinterpret_cast<float4*>(&scale_val[4]) = *reinterpret_cast<const float4*>(scale + scale_offset + 4);

#pragma unroll
  for (uint32_t i = 0; i < kPack; ++i) {
    // The amax is rounded through the activation dtype before dividing, like the Sage quantiser.
    const float amax_val = details::to_float(details::from_float<T>(scale_val[i] * kVScaleMax));
    x_val_float[i] = details::to_float(x_val[i]) * __fdividef(kVScaleMax, amax_val);
  }
  details::floatx4_to_e4m3x4(x_val_fp8, x_val_float, x_val_float + 2);
  details::floatx4_to_e4m3x4(x_val_fp8 + 1, x_val_float + 4, x_val_float + 6);
  *reinterpret_cast<uint2*>(chunk + output_offset) = *reinterpret_cast<uint2*>(&x_val_fp8[0]);
  device::PDLTriggerSecondary<kUsePDL>();
}

// ---------------------------------------------------------------------------
// Receiver: rebuild the SM120 Sage operands from the gathered chunks.
// One CTA per (64-token tile, local head, batch). Under ALIGN-128 every tile
// lies inside one source chunk. Q/K come out `[B, h, S, 128]`, V comes out
// `[B, h, 128, S]` with Sage's 16-token permutation, rows >= used_sequence are
// zeroed in all three, and the per-group scales are emitted at the width the
// attention kernel derives from `scale_sequence` rows.
// ---------------------------------------------------------------------------
template <bool kUsePDL>
__global__ void UnpackForSageKernel(
    const uint8_t* __restrict__ input,
    uint8_t* __restrict__ q,
    uint8_t* __restrict__ k,
    uint8_t* __restrict__ v,
    float* __restrict__ q_scale,
    float* __restrict__ k_scale,
    uint64_t main_bytes,
    uint64_t chunk_bytes,
    uint32_t batch_size,
    uint32_t local_sequence,
    uint32_t global_sequence,
    uint32_t used_sequence,
    uint32_t q_scale_alloc,
    uint32_t k_scale_alloc) {
  constexpr uint32_t kTile = 64;
  constexpr uint32_t kVector = 16;
  constexpr uint32_t kVectorsPerToken = kHeadDim / kVector;
  constexpr uint32_t kSequenceVectors = kTile / kVector;
  const uint32_t block_token_base = blockIdx.x * kTile;
  const uint32_t local_head = blockIdx.y;
  const uint32_t batch_id = blockIdx.z;
  const uint32_t thread_id = threadIdx.x;
  const uint32_t local_heads = gridDim.y;

  const uint32_t token_in_block = thread_id / kVectorsPerToken;
  const uint32_t global_token = block_token_base + token_in_block;
  const uint32_t d_base = thread_id % kVectorsPerToken * kVector;

  const uint32_t source = block_token_base / local_sequence;
  const uint32_t local_token = global_token - source * local_sequence;
  const uint64_t source_chunk = static_cast<uint64_t>(source) * chunk_bytes;
  const uint64_t source_element = details::section_offset(local_token, batch_id, batch_size, local_head, local_heads, d_base);
  device::PDLWaitPrimary<kUsePDL>();
  const bool live = global_token < used_sequence;
  const uint4 zero = make_uint4(0, 0, 0, 0);
  const uint4 q_value = live ? *reinterpret_cast<const uint4*>(input + source_chunk + source_element) : zero;
  const uint4 k_value = live ? *reinterpret_cast<const uint4*>(input + source_chunk + main_bytes + source_element) : zero;
  const uint4 v_value = live ? *reinterpret_cast<const uint4*>(input + source_chunk + 2 * main_bytes + source_element) : zero;

  const uint64_t bhsd_element =
      ((static_cast<uint64_t>(batch_id) * local_heads + local_head) * global_sequence + global_token) * kHeadDim + d_base;
  *reinterpret_cast<uint4*>(q + bhsd_element) = q_value;
  *reinterpret_cast<uint4*>(k + bhsd_element) = k_value;

  // Sage's 16-token permutation on the (16-aligned) tile-local token index.
  const uint32_t token_mod_16 = token_in_block & 15;
  const uint32_t packed_row =
      (token_in_block & ~15U) + (token_mod_16 / 8) * 2 + ((token_mod_16 / 2) % 4) * 4 + token_mod_16 % 2;
  __shared__ uint8_t shared_load[kTile][kHeadDim];
  __shared__ uint8_t shared_store[kHeadDim][kTile];
  *reinterpret_cast<uint4*>(&shared_load[packed_row][d_base]) = v_value;
  __syncthreads();
#pragma unroll
  for (uint32_t i = 0; i < kVector; ++i) {
    shared_store[d_base + i][packed_row] = shared_load[packed_row][d_base + i];
  }
  __syncthreads();
  const uint32_t output_d = thread_id / kSequenceVectors;
  const uint32_t output_token_base = thread_id % kSequenceVectors * kVector;
  const uint64_t v_output_offset =
      ((static_cast<uint64_t>(batch_id) * local_heads + local_head) * kHeadDim + output_d) * global_sequence +
      block_token_base + output_token_base;
  *reinterpret_cast<uint4*>(v + v_output_offset) = *reinterpret_cast<uint4*>(&shared_store[output_d][output_token_base]);

  if (blockIdx.x != 0) {
    device::PDLTriggerSecondary<kUsePDL>();
    return;
  }

  // Scales: slot g belongs to source g / groups_per_source. Each (batch, head)
  // has one CTA here, so every output slot has exactly one writer.
  const uint64_t scale_head = static_cast<uint64_t>(batch_id) * local_heads + local_head;
  const uint32_t q_groups_per_source = local_sequence / kQGroup;
  const uint32_t k_groups_per_source = local_sequence / kKGroup;
  const uint64_t q_scale_section = 3 * main_bytes;
  const uint64_t k_scale_section = q_scale_section + static_cast<uint64_t>(batch_size) * local_heads * q_groups_per_source * 4;
  for (uint32_t g = thread_id; g < q_scale_alloc; g += blockDim.x) {
    const uint32_t owner = g / q_groups_per_source;
    const uint32_t owner_slot = g - owner * q_groups_per_source;
    const float* q_scale_input = reinterpret_cast<const float*>(input + static_cast<uint64_t>(owner) * chunk_bytes + q_scale_section);
    q_scale[scale_head * q_scale_alloc + g] = q_scale_input[scale_head * q_groups_per_source + owner_slot];
  }
  for (uint32_t g = thread_id; g < k_scale_alloc; g += blockDim.x) {
    const uint32_t owner = g / k_groups_per_source;
    const uint32_t owner_slot = g - owner * k_groups_per_source;
    const float* k_scale_input = reinterpret_cast<const float*>(input + static_cast<uint64_t>(owner) * chunk_bytes + k_scale_section);
    k_scale[scale_head * k_scale_alloc + g] = k_scale_input[scale_head * k_groups_per_source + owner_slot];
  }
  device::PDLTriggerSecondary<kUsePDL>();
}

// ---------------------------------------------------------------------------
// Host entry points. Q/K/V are `[B, L, H, 128]` views whose head_dim is dense;
// batch/token/head strides are free (16-byte aligned), which admits the fused
// projection's `[T, H, 3, D]` slices without a copy.
// ---------------------------------------------------------------------------
template <typename T, bool kUsePDL>
struct Kernels {
  static_assert(std::is_same_v<T, half> || std::is_same_v<T, nv_bfloat16>);

  struct Shard {
    int64_t batch, local_sequence, num_heads;
    int64_t stride_batch, stride_token, stride_head;
  };

  /// Validate an NHD shard view and return its geometry; `B`, `L`, `H` are shared symbols.
  static auto shard(
      tvm::ffi::TensorView x,
      const char* name,
      host::SymbolicSize& B,
      host::SymbolicSize& L,
      host::SymbolicSize& H,
      host::SymbolicDevice& device) -> Shard {
    using namespace host;
    SymbolicSize SB{"stride_batch"}, ST{"stride_token"}, SH{"stride_head"};
    TensorMatcher({B, L, H, kHeadDim})
        .with_strides({SB, ST, SH, 1})
        .template with_dtype<T>()
        .template with_device<kDLCUDA>(device)
        .ensure_alignment(16)
        .verify(x);
    CHECK_HOST(L.unwrap() % kShardAlignment == 0)
        << name << ": the local sequence must be a whole number of " << kShardAlignment << "-token blocks, got "
        << L.unwrap();
    return Shard{B.unwrap(), L.unwrap(), H.unwrap(), SB.unwrap(), ST.unwrap(), SH.unwrap()};
  }

  /**
   * \brief Per-channel K sum and V amax over this rank's shard.
   * \param k, v   `[B, L, H, 128]` shard views (same shape / dtype / strides class)
   * \param stats  fp32 `[R, 2, B, H, 128]`: R identical records of (K sum, V amax) over the shard's
   *               tokens, the replicated all-gather payload
   * \param own    fp32 `[2, B, H, 128]`, the same record once more (this rank's row of the gathered result)
   */
  static void k_sum_v_amax(tvm::ffi::TensorView k, tvm::ffi::TensorView v, tvm::ffi::TensorView stats, tvm::ffi::TensorView own) {
    using namespace host;
    SymbolicSize B{"batch"}, L{"local_sequence"}, H{"heads"}, R{"replicas"};
    SymbolicDevice device;
    device.set_options<kDLCUDA>();
    const Shard ks = shard(k, "k", B, L, H, device);
    const Shard vs = shard(v, "v", B, L, H, device);
    TensorMatcher({R, 2, B, H, kHeadDim}).template with_dtype<fp32_t>().template with_device<kDLCUDA>(device).verify(stats);
    TensorMatcher({2, B, H, kHeadDim}).template with_dtype<fp32_t>().template with_device<kDLCUDA>(device).verify(own);
    const uint64_t half = static_cast<uint64_t>(ks.batch) * ks.num_heads * kHeadDim;
    const DLDevice dev = device.unwrap();
    const int64_t chunks = div_ceil(ks.local_sequence, static_cast<int64_t>(kStatsChunkTokens));
    const int64_t partial_elems = ks.batch * ks.num_heads * chunks * kHeadDim;
    auto k_partial = ffi::alloc_workspace_tensor(partial_elems * sizeof(float), dev);
    auto v_partial = ffi::alloc_workspace_tensor(partial_elems * sizeof(float), dev);
    LaunchKernel(dim3(ks.num_heads, ks.batch, chunks), 2 * kHeadDim, dev).enable_pdl(kUsePDL)(
        KSumVAmaxPartialKernel<T, kUsePDL>,
        static_cast<const T*>(k.data_ptr()),
        static_cast<const T*>(v.data_ptr()),
        static_cast<float*>(k_partial.data_ptr()),
        static_cast<float*>(v_partial.data_ptr()),
        static_cast<uint32_t>(ks.local_sequence),
        static_cast<uint32_t>(ks.num_heads),
        static_cast<uint32_t>(chunks),
        ks.stride_batch,
        ks.stride_token,
        ks.stride_head,
        vs.stride_batch,
        vs.stride_token,
        vs.stride_head);
    LaunchKernel(dim3(ks.num_heads, ks.batch), kHeadDim, dev).enable_pdl(kUsePDL)(
        KSumVAmaxCombineKernel<kUsePDL>,
        static_cast<const float*>(k_partial.data_ptr()),
        static_cast<const float*>(v_partial.data_ptr()),
        static_cast<float*>(stats.data_ptr()),
        static_cast<float*>(own.data_ptr()),
        static_cast<uint32_t>(ks.num_heads),
        static_cast<uint32_t>(chunks),
        static_cast<uint32_t>(R.unwrap()),
        2 * half,
        half);
  }

  /**
   * \brief `k_mean` and `v_scale` from the gathered statistics, plus this rank's slice of `v_scale`.
   * \param gathered      fp32 `[W, 2, B, H, 128]`, row r from rank r
   * \param k_mean        `[B, H, 128]` in the activation dtype
   * \param v_scale       fp32 `[B, H, 128]`
   * \param v_scale_local fp32 `[B, h, 128]`, heads `[rank * h, (rank + 1) * h)`
   * \param used_sequence live rows of the global sequence
   */
  static void finalize_stats(
      tvm::ffi::TensorView gathered,
      tvm::ffi::TensorView k_mean,
      tvm::ffi::TensorView v_scale,
      tvm::ffi::TensorView v_scale_local,
      int64_t used_sequence,
      int64_t rank) {
    using namespace host;
    SymbolicSize W{"world_size"}, B{"batch"}, H{"heads"}, h{"local_heads"};
    SymbolicDevice device;
    device.set_options<kDLCUDA>();
    TensorMatcher({W, 2, B, H, kHeadDim}).template with_dtype<fp32_t>().template with_device<kDLCUDA>(device).verify(gathered);
    TensorMatcher({B, H, kHeadDim}).template with_dtype<T>().template with_device<kDLCUDA>(device).verify(k_mean);
    TensorMatcher({B, H, kHeadDim}).template with_dtype<fp32_t>().template with_device<kDLCUDA>(device).verify(v_scale);
    TensorMatcher({B, h, kHeadDim}).template with_dtype<fp32_t>().template with_device<kDLCUDA>(device).verify(v_scale_local);
    CHECK_HOST(H.unwrap() % h.unwrap() == 0 && rank >= 0 && (rank + 1) * h.unwrap() <= H.unwrap())
        << "local heads must tile the heads and rank must own a slice";
    CHECK_HOST(used_sequence > 0) << "used_sequence must be positive";
    const DLDevice dev = device.unwrap();
    LaunchKernel(dim3(H.unwrap(), B.unwrap()), kHeadDim, dev)(
        FinalizeStatsKernel<T>,
        static_cast<const float*>(gathered.data_ptr()),
        static_cast<T*>(k_mean.data_ptr()),
        static_cast<float*>(v_scale.data_ptr()),
        static_cast<float*>(v_scale_local.data_ptr()),
        static_cast<uint32_t>(W.unwrap()),
        static_cast<uint32_t>(H.unwrap()),
        static_cast<uint32_t>(h.unwrap()),
        static_cast<uint32_t>(rank),
        static_cast<uint32_t>(used_sequence),
        static_cast<uint64_t>(B.unwrap()) * H.unwrap() * kHeadDim);
  }

  /**
   * \brief Quantise this rank's Q/K/V shard and pack it destination-major.
   * \param k_mean  `[B, H, 128]` in the activation dtype, the global channel mean of K
   * \param v_scale fp32 `[B, H, 128]`, global `amax(|v|) / 2.25`
   * \param out     uint8 `[world_size, chunk_bytes]` (see chunk_spec); may be a transport landing buffer
   * \param used_sequence live rows of the global sequence, in `(0, L * world_size]`
   */
  static void quant_pack(
      tvm::ffi::TensorView q,
      tvm::ffi::TensorView k,
      tvm::ffi::TensorView v,
      tvm::ffi::TensorView k_mean,
      tvm::ffi::TensorView v_scale,
      tvm::ffi::TensorView out,
      tvm::ffi::TensorView own_out,
      int64_t rank,
      int64_t world_size,
      int64_t used_sequence) {
    using namespace host;
    SymbolicSize B{"batch"}, L{"local_sequence"}, H{"heads"};
    SymbolicDevice device;
    device.set_options<kDLCUDA>();
    const Shard qs = shard(q, "q", B, L, H, device);
    const Shard ks = shard(k, "k", B, L, H, device);
    const Shard vs = shard(v, "v", B, L, H, device);
    TensorMatcher({B, H, kHeadDim}).template with_dtype<T>().template with_device<kDLCUDA>(device).verify(k_mean);
    TensorMatcher({B, H, kHeadDim}).template with_dtype<fp32_t>().template with_device<kDLCUDA>(device).verify(v_scale);
    CHECK_HOST(world_size >= 1 && rank >= 0 && rank < world_size) << "need 0 <= rank < world_size";
    CHECK_HOST(qs.num_heads % world_size == 0) << "heads " << qs.num_heads << " must split evenly over " << world_size;
    const int64_t local_heads = qs.num_heads / world_size;
    const int64_t global_sequence = qs.local_sequence * world_size;
    CHECK_HOST(used_sequence > 0 && used_sequence <= global_sequence)
        << "used_sequence must lie in (0, " << global_sequence << "], got " << used_sequence;
    const ChunkSpec spec = chunk_spec(qs.batch, qs.local_sequence, local_heads);
    TensorMatcher({world_size, spec.chunk_bytes}).template with_dtype<uint8_t>().template with_device<kDLCUDA>(device).verify(out);
    TensorMatcher({spec.chunk_bytes}).template with_dtype<uint8_t>().template with_device<kDLCUDA>(device).verify(own_out);
    const DLDevice dev = device.unwrap();
    auto* payload = static_cast<uint8_t*>(out.data_ptr());
    auto* own_payload = static_cast<uint8_t*>(own_out.data_ptr());
    const uint32_t global_offset = static_cast<uint32_t>(rank * qs.local_sequence);

    LaunchKernel(dim3(qs.local_sequence / kQGroup, qs.num_heads, qs.batch), kQGroup * (kHeadDim / 8), dev)
        .enable_pdl(kUsePDL)(
            QuantInt8FusedAmaxPackKernel<T, kQGroup, false, kUsePDL>,
            static_cast<const T*>(q.data_ptr()),
            static_cast<const T*>(nullptr),
            payload,
            own_payload,
            static_cast<uint32_t>(rank),
            static_cast<uint32_t>(qs.local_sequence),
            global_offset,
            static_cast<uint32_t>(qs.num_heads),
            static_cast<uint32_t>(local_heads),
            static_cast<uint32_t>(qs.batch),
            static_cast<uint64_t>(spec.chunk_bytes),
            static_cast<uint64_t>(0),
            static_cast<uint64_t>(spec.q_scale_offset),
            static_cast<uint32_t>(qs.local_sequence / kQGroup),
            static_cast<uint32_t>(used_sequence),
            qs.stride_batch,
            qs.stride_token,
            qs.stride_head);
    LaunchKernel(dim3(ks.local_sequence / kKGroup, ks.num_heads, ks.batch), kKGroup * (kHeadDim / 8), dev)
        .enable_pdl(kUsePDL)(
            QuantInt8FusedAmaxPackKernel<T, kKGroup, true, kUsePDL>,
            static_cast<const T*>(k.data_ptr()),
            static_cast<const T*>(k_mean.data_ptr()),
            payload,
            own_payload,
            static_cast<uint32_t>(rank),
            static_cast<uint32_t>(ks.local_sequence),
            global_offset,
            static_cast<uint32_t>(ks.num_heads),
            static_cast<uint32_t>(local_heads),
            static_cast<uint32_t>(ks.batch),
            static_cast<uint64_t>(spec.chunk_bytes),
            static_cast<uint64_t>(spec.main_bytes),
            static_cast<uint64_t>(spec.k_scale_offset),
            static_cast<uint32_t>(ks.local_sequence / kKGroup),
            static_cast<uint32_t>(used_sequence),
            ks.stride_batch,
            ks.stride_token,
            ks.stride_head);
    constexpr uint32_t kVThreads = 256;
    const uint64_t v_packs = static_cast<uint64_t>(vs.batch) * vs.local_sequence * local_heads * (kHeadDim / 8);
    LaunchKernel(dim3(div_ceil(v_packs, static_cast<uint64_t>(kVThreads)), world_size), kVThreads, dev)
        .enable_pdl(kUsePDL)(
            QuantVFP8PackKernel<T, kUsePDL>,
            static_cast<const T*>(v.data_ptr()),
            static_cast<const float*>(v_scale.data_ptr()),
            payload,
            own_payload,
            static_cast<uint32_t>(rank),
            v_packs,
            static_cast<uint32_t>(vs.num_heads),
            static_cast<uint32_t>(local_heads),
            static_cast<uint32_t>(vs.batch),
            static_cast<uint64_t>(spec.main_bytes),
            static_cast<uint64_t>(spec.chunk_bytes),
            vs.stride_batch,
            vs.stride_token,
            vs.stride_head);
  }

  /**
   * \brief Rebuild the SM120 Sage operands from the received chunks.
   * \param recv    uint8 `[world_size, chunk_bytes]`
   * \param q, k    INT8 `[B, h, S, 128]`, S = L * world_size
   * \param v       FP8 E4M3 `[B, h, 128, S]` (Sage 16-token permutation)
   * \param q_scale fp32 `[B, h, ceil(scale_sequence / 128) * 4]`
   * \param k_scale fp32 `[B, h, ceil(scale_sequence / 64)]`
   * \param used_sequence rows `>= used_sequence` of q, k, v are written as zero
   */
  static void unpack_for_sage(
      tvm::ffi::TensorView recv,
      tvm::ffi::TensorView q,
      tvm::ffi::TensorView k,
      tvm::ffi::TensorView v,
      tvm::ffi::TensorView q_scale,
      tvm::ffi::TensorView k_scale,
      int64_t local_sequence,
      int64_t world_size,
      int64_t scale_sequence,
      int64_t used_sequence) {
    using namespace host;
    CHECK_HOST(world_size >= 1) << "world_size must be positive";
    CHECK_HOST(local_sequence > 0 && local_sequence % kShardAlignment == 0)
        << "local_sequence must be a positive multiple of " << kShardAlignment;
    const int64_t global_sequence = local_sequence * world_size;
    CHECK_HOST(scale_sequence > 0 && scale_sequence <= global_sequence && used_sequence > 0 && used_sequence <= global_sequence)
        << "scale_sequence and used_sequence must lie in (0, " << global_sequence << "]";
    const int64_t q_scale_alloc = div_ceil(scale_sequence, int64_t{128}) * 4;
    const int64_t k_scale_alloc = div_ceil(scale_sequence, static_cast<int64_t>(kKGroup));

    SymbolicSize B{"batch"}, h{"local_heads"};
    SymbolicDevice device;
    device.set_options<kDLCUDA>();
    TensorMatcher({B, h, global_sequence, kHeadDim}).template with_dtype<int8_t>().template with_device<kDLCUDA>(device).verify(q).verify(k);
    TensorMatcher({B, h, kHeadDim, global_sequence}).template with_dtype<fp8_e4m3_t>().template with_device<kDLCUDA>(device).verify(v);
    TensorMatcher({B, h, q_scale_alloc}).template with_dtype<fp32_t>().template with_device<kDLCUDA>(device).verify(q_scale);
    TensorMatcher({B, h, k_scale_alloc}).template with_dtype<fp32_t>().template with_device<kDLCUDA>(device).verify(k_scale);
    const ChunkSpec spec = chunk_spec(B.unwrap(), local_sequence, h.unwrap());
    TensorMatcher({world_size, spec.chunk_bytes}).template with_dtype<uint8_t>().template with_device<kDLCUDA>(device).verify(recv);
    const DLDevice dev = device.unwrap();

    LaunchKernel(dim3(global_sequence / 64, h.unwrap(), B.unwrap()), 64 * kHeadDim / 16, dev).enable_pdl(kUsePDL)(
        UnpackForSageKernel<kUsePDL>,
        static_cast<const uint8_t*>(recv.data_ptr()),
        static_cast<uint8_t*>(q.data_ptr()),
        static_cast<uint8_t*>(k.data_ptr()),
        static_cast<uint8_t*>(v.data_ptr()),
        static_cast<float*>(q_scale.data_ptr()),
        static_cast<float*>(k_scale.data_ptr()),
        static_cast<uint64_t>(spec.main_bytes),
        static_cast<uint64_t>(spec.chunk_bytes),
        static_cast<uint32_t>(B.unwrap()),
        static_cast<uint32_t>(local_sequence),
        static_cast<uint32_t>(global_sequence),
        static_cast<uint32_t>(used_sequence),
        static_cast<uint32_t>(q_scale_alloc),
        static_cast<uint32_t>(k_scale_alloc));
  }
};

}  // namespace ulysses_lowp_sage
}  // namespace sglang
