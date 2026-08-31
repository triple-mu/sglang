#include <sgl_kernel/tensor.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <sgl_kernel/impl/norm.cuh>

#include <dlpack/dlpack.h>

#include <cstdint>
#include <type_traits>

namespace sglang {

struct QKNormRopeParams {
  void* __restrict__ q_ptr;
  void* __restrict__ k_ptr;  // pre-offset by -num_qo_heads * head_stride_bytes
  const void* __restrict__ q_weight_ptr;
  const void* __restrict__ k_weight_ptr;
  const void* __restrict__ cos_sin_cache_ptr;
  const void* __restrict__ positions;
  int64_t q_stride_bytes;
  int64_t k_stride_bytes;
  int64_t head_stride_bytes;
  uint32_t num_qo_heads;
  uint32_t num_kv_heads;
  uint32_t num_tokens;
  float eps;
  // Wide variant only: whether every lane's cos/sin span is one aligned
  // vector load (base, row stride, and cos->sin offset all span-aligned).
  bool cache_vec_aligned;
};

struct QKNormRopePackKVParams : QKNormRopeParams {
  const void* __restrict__ v_ptr;
  const void* __restrict__ k_prefix_ptr;
  const void* __restrict__ v_prefix_ptr;
  void* __restrict__ packed_k_ptr;
  void* __restrict__ packed_v_ptr;
  int64_t v_stride_bytes;
  int64_t k_prefix_stride_bytes;
  int64_t v_prefix_stride_bytes;
  int64_t packed_token_stride_bytes;
  int64_t packed_head_stride_bytes;
  uint32_t batch_size;
  uint32_t prefix_tokens;
  uint32_t suffix_tokens;
};

template <bool kPackKV>
using QKNormRopeParamsT = std::conditional_t<kPackKV, QKNormRopePackKVParams, QKNormRopeParams>;

constexpr uint32_t kThreadsPerBlock = 256;
constexpr uint32_t kWarpsPerBlock = kThreadsPerBlock / device::kWarpThreads;

template <uint32_t kLaneCount>
constexpr uint32_t active_mask() {
  static_assert(kLaneCount <= device::kWarpThreads, "active_mask lane count must not exceed warp size");
  if constexpr (kLaneCount == device::kWarpThreads) {
    return 0xffffffffu;
  } else {
    return (1u << kLaneCount) - 1u;
  }
}

template <typename CacheDType>
SGL_DEVICE CacheDType load_cache_value(const CacheDType* ptr, int64_t idx) {
#ifdef USE_ROCM
  return ptr[idx];
#else
  return __ldg(ptr + idx);
#endif
}

/// \brief Load one lane's contiguous cos/sin span, as one vector when the
/// caller proved alignment (values identical to the per-element loads).
template <uint32_t kSpan, typename CacheDType>
SGL_DEVICE void load_cache_span(const CacheDType* __restrict__ ptr, CacheDType (&dst)[kSpan], bool vec_aligned) {
  constexpr uint32_t kSpanBytes = kSpan * sizeof(CacheDType);
  if constexpr ((kSpan & (kSpan - 1)) == 0 && kSpanBytes <= 16) {
    if (vec_aligned) {
      device::AlignedVector<CacheDType, kSpan> vec;
      vec.load(ptr);
#pragma unroll
      for (uint32_t i = 0; i < kSpan; ++i) {
        dst[i] = vec[i];
      }
      return;
    }
  }
#pragma unroll
  for (uint32_t i = 0; i < kSpan; ++i) {
    dst[i] = load_cache_value(ptr, i);
  }
}

/// Elements each of the 16 lanes of a half-warp head group owns in the wide
/// (two heads per warp) variant.
template <int64_t kHeadDim>
constexpr uint32_t wide_elems_per_lane() {
  return kHeadDim / (device::kWarpThreads / 2);
}

/// Host check for the wide variant's cos/sin span loads: the cache base, the
/// per-position row stride, and the cos->sin offset must all be span-aligned
/// (lane offsets within a row are span multiples already).
template <int64_t kHeadDim, int64_t kRopeDim, typename CacheDType>
inline bool wide_cache_span_aligned(const void* cache_ptr) {
  constexpr int64_t kSpanBytes = wide_elems_per_lane<kHeadDim>() * static_cast<int64_t>(sizeof(CacheDType));
  constexpr int64_t kStrideBytes = kRopeDim * static_cast<int64_t>(sizeof(CacheDType));
  constexpr int64_t kSinOffsetBytes = kRopeDim / 2 * static_cast<int64_t>(sizeof(CacheDType));
  return reinterpret_cast<uintptr_t>(cache_ptr) % kSpanBytes == 0 && kStrideBytes % kSpanBytes == 0 &&
         kSinOffsetBytes % kSpanBytes == 0;
}

template <typename T>
SGL_DEVICE T rotary_mul_rn(T lhs, T rhs) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1030 || __CUDA_ARCH__ >= 1200)
  uint16_t lhs_bits;
  uint16_t rhs_bits;
  if constexpr (std::is_same_v<T, bf16_t>) {
    lhs_bits = __bfloat16_as_ushort(lhs);
    rhs_bits = __bfloat16_as_ushort(rhs);
  } else {
    lhs_bits = __half_as_ushort(lhs);
    rhs_bits = __half_as_ushort(rhs);
  }
  uint16_t out_bits;
  if constexpr (std::is_same_v<T, bf16_t>) {
    asm volatile("mul.rn.bf16 %0, %1, %2;" : "=h"(out_bits) : "h"(lhs_bits), "h"(rhs_bits));
  } else {
    asm volatile("mul.rn.f16 %0, %1, %2;" : "=h"(out_bits) : "h"(lhs_bits), "h"(rhs_bits));
  }
  if constexpr (std::is_same_v<T, bf16_t>) {
    return __ushort_as_bfloat16(out_bits);
  } else {
    return __ushort_as_half(out_bits);
  }
#else
  return lhs * rhs;
#endif
}

template <typename T>
SGL_DEVICE T rotary_add(T x, T cos, T y, T sin) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1030 || __CUDA_ARCH__ >= 1200)
  // nvcc may contract the packed local expression on Blackwell SM103/SM120
  // even though the reference RoPE kernel rounds both products to the
  // activation dtype first.
  const T lhs = rotary_mul_rn(x, cos);
  const T rhs = rotary_mul_rn(y, sin);
  uint16_t lhs_bits;
  uint16_t rhs_bits;
  if constexpr (std::is_same_v<T, bf16_t>) {
    lhs_bits = __bfloat16_as_ushort(lhs);
    rhs_bits = __bfloat16_as_ushort(rhs);
  } else {
    lhs_bits = __half_as_ushort(lhs);
    rhs_bits = __half_as_ushort(rhs);
  }
  uint16_t out_bits;
  if constexpr (std::is_same_v<T, bf16_t>) {
    asm volatile("add.rn.bf16 %0, %1, %2;" : "=h"(out_bits) : "h"(lhs_bits), "h"(rhs_bits));
  } else {
    asm volatile("add.rn.f16 %0, %1, %2;" : "=h"(out_bits) : "h"(lhs_bits), "h"(rhs_bits));
  }
  if constexpr (std::is_same_v<T, bf16_t>) {
    return __ushort_as_bfloat16(out_bits);
  } else {
    return __ushort_as_half(out_bits);
  }
#else
  return x * cos + y * sin;
#endif
}

template <typename T>
SGL_DEVICE T rotary_sub(T x, T cos, T y, T sin) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1030 || __CUDA_ARCH__ >= 1200)
  const T lhs = rotary_mul_rn(x, cos);
  const T rhs = rotary_mul_rn(y, sin);
  uint16_t lhs_bits;
  uint16_t rhs_bits;
  if constexpr (std::is_same_v<T, bf16_t>) {
    lhs_bits = __bfloat16_as_ushort(lhs);
    rhs_bits = __bfloat16_as_ushort(rhs);
  } else {
    lhs_bits = __half_as_ushort(lhs);
    rhs_bits = __half_as_ushort(rhs);
  }
  uint16_t out_bits;
  if constexpr (std::is_same_v<T, bf16_t>) {
    asm volatile("sub.rn.bf16 %0, %1, %2;" : "=h"(out_bits) : "h"(lhs_bits), "h"(rhs_bits));
  } else {
    asm volatile("sub.rn.f16 %0, %1, %2;" : "=h"(out_bits) : "h"(lhs_bits), "h"(rhs_bits));
  }
  if constexpr (std::is_same_v<T, bf16_t>) {
    return __ushort_as_bfloat16(out_bits);
  } else {
    return __ushort_as_half(out_bits);
  }
#else
  return x * cos - y * sin;
#endif
}

template <typename T>
SGL_DEVICE T rotary_add_fp32(T x, float cos, T y, float sin) {
  const float x_fp32 = device::cast<fp32_t>(x);
  const float y_fp32 = device::cast<fp32_t>(y);
#ifdef USE_ROCM
  return device::cast<T>(x_fp32 * cos + y_fp32 * sin);
#else
  const float lhs = __fmul_rn(x_fp32, cos);
  const float rhs = __fmul_rn(y_fp32, sin);
  return device::cast<T>(__fadd_rn(lhs, rhs));
#endif
}

template <typename T>
SGL_DEVICE T rotary_sub_fp32(T x, float cos, T y, float sin) {
  const float x_fp32 = device::cast<fp32_t>(x);
  const float y_fp32 = device::cast<fp32_t>(y);
#ifdef USE_ROCM
  return device::cast<T>(x_fp32 * cos - y_fp32 * sin);
#else
  const float lhs = __fmul_rn(x_fp32, cos);
  const float rhs = __fmul_rn(-y_fp32, sin);
  return device::cast<T>(__fadd_rn(lhs, rhs));
#endif
}

// Replicates the sum-of-squares reduction of aten's
// vectorized_layer_norm_kernel<c10::Half, float, true> (torch nn.RMSNorm on
// half inputs, N == 64): 4 contiguous elements per thread summed sequentially
// in fp32, xor-tree combine, rsqrtf. Verified bit-exact vs torch 2.13 on
// H200; retire if torch changes that kernel's reduction.
template <int64_t kHeadDim, typename Storage>
SGL_DEVICE float aten_order_rms_norm_factor(const Storage& input_vec, float eps, uint32_t lane_id) {
  using namespace device;
  static_assert(kHeadDim == 64, "aten-order norm replication is only mapped for head_dim 64");
#ifdef USE_ROCM
  // aten dispatches a different kernel on ROCm; this replication is CUDA-only.
  return 0.0f;
#else
  // Each lane holds elements [2*lane, 2*lane+1]; aten groups 4 contiguous
  // elements per thread, so even lanes fold in the odd neighbor's squares.
  // fp16 squares are exact in fp32, so only the addition order matters.
  const auto [x0, x1] = cast<fp32x2_t>(input_vec[0]);
  const float sq0 = x0 * x0;
  const float sq1 = x1 * x1;
  const float nsq0 = __shfl_down_sync(warp::kFullMask, sq0, 1);
  const float nsq1 = __shfl_down_sync(warp::kFullMask, sq1, 1);
  float partial = ((sq0 + sq1) + nsq0) + nsq1;
#pragma unroll
  for (uint32_t offset = 16; offset >= 2; offset >>= 1) {
    partial += __shfl_xor_sync(warp::kFullMask, partial, offset);
  }
  const float total = __shfl_sync(warp::kFullMask, partial, lane_id & ~1u);
  return math::rsqrt(total / static_cast<float>(kHeadDim) + eps);
#endif
}

/**
 * \brief Wide-variant work: 16 lanes per head, two heads per warp.
 *
 * Shares every warp instruction between two heads (and doubles the per-lane
 * load width) on this issue-bound kernel, at the cost of the bitwise
 * contract: the sum-of-squares partials regroup (twice the sequential
 * elements per lane, xor tree over 16 lanes), so the norm factor may differ
 * from the contract-exact variant in the last fp32 bit. Everything after the
 * norm factor is the same per-element arithmetic. Near-lossless; opt-in only
 * (see the wide_head flag in qknorm_rope_jit.py).
 *
 * \param group_lane Lane within this head's 16-lane group.
 * \param group_base First lane of this group (0 or 16).
 */
template <int64_t kHeadDim, int64_t kRopeDim, typename DType, typename CacheDType, typename Storage>
SGL_DEVICE void qknorm_rope_process_work_wide(
    Storage& input_vec,
    const Storage& weight_vec,
    const void* __restrict__ cos_sin_cache_ptr,
    int64_t pos,
    void* __restrict__ output,
    uint32_t group_lane,
    uint32_t group_base,
    float eps,
    bool cache_vec_aligned) {
  using namespace device;

  constexpr uint32_t kGroupLanes = kWarpThreads / 2;
  constexpr uint32_t kElemsPerLane = wide_elems_per_lane<kHeadDim>();
  constexpr uint32_t kVecSize = kElemsPerLane / 2;
  constexpr uint32_t kRotaryLanes = kRopeDim / kElemsPerLane;
  constexpr uint32_t kHalfRotaryLanes = kRotaryLanes / 2;
  constexpr int64_t kCosSinStrideBytes = kRopeDim * sizeof(CacheDType);
  using Packed = packed_t<DType>;

  const uint32_t norm_mask = 0xffffu << group_base;
  const uint32_t rope_mask = ((1u << kRotaryLanes) - 1u) << group_base;

  const auto cos_ptr = static_cast<const CacheDType*>(pointer::offset(cos_sin_cache_ptr, pos * kCosSinStrideBytes));
  const auto sin_ptr = cos_ptr + kRopeDim / 2;

  // Same accumulation expression as apply_norm_impl, reduced over the 16-lane
  // group instead of the warp.
  float sum_of_squares = 0.0f;
#pragma unroll
  for (uint32_t j = 0; j < kVecSize; ++j) {
    const auto fp32_input = cast<fp32x2_t>(input_vec[j]);
    sum_of_squares += fp32_input.x * fp32_input.x;
    sum_of_squares += fp32_input.y * fp32_input.y;
  }
  sum_of_squares = warp::reduce_sum<kGroupLanes>(sum_of_squares, norm_mask);
  const float norm_factor = math::rsqrt(sum_of_squares / static_cast<float>(kHeadDim) + eps);

  Storage output_vec;
#pragma unroll
  for (uint32_t j = 0; j < kVecSize; ++j) {
    const auto [x0, x1] = cast<fp32x2_t>(input_vec[j]);
    const auto [w0, w1] = cast<fp32x2_t>(weight_vec[j]);
    output_vec[j] = cast<Packed, fp32x2_t>({x0 * norm_factor * w0, x1 * norm_factor * w1});
  }

  if (group_lane < kRotaryLanes) {
    const auto partner_lane =
        group_base + (group_lane < kHalfRotaryLanes ? group_lane + kHalfRotaryLanes : group_lane - kHalfRotaryLanes);
    const uint32_t lane_base = (group_lane % kHalfRotaryLanes) * kElemsPerLane;
    CacheDType cos_lane[kElemsPerLane];
    CacheDType sin_lane[kElemsPerLane];
    load_cache_span(cos_ptr + lane_base, cos_lane, cache_vec_aligned);
    load_cache_span(sin_ptr + lane_base, sin_lane, cache_vec_aligned);
#pragma unroll
    for (uint32_t j = 0; j < kVecSize; ++j) {
      auto partner_vec = output_vec[j];
      auto partner_bits = reinterpret_cast<const uint32_t&>(partner_vec);
      partner_bits = __shfl_sync(rope_mask, partner_bits, partner_lane);
      reinterpret_cast<uint32_t&>(partner_vec) = partner_bits;
      auto& values = unpack(output_vec[j]);
      const auto& partner_values = unpack(partner_vec);
#pragma unroll
      for (uint32_t i = 0; i < 2; ++i) {
        const auto cos = cos_lane[2 * j + i];
        const auto sin = sin_lane[2 * j + i];
        if constexpr (std::is_same_v<CacheDType, fp32_t>) {
          values[i] = group_lane < kHalfRotaryLanes ? rotary_sub_fp32(values[i], cos, partner_values[i], sin)
                                                    : rotary_add_fp32(values[i], cos, partner_values[i], sin);
        } else {
          values[i] = group_lane < kHalfRotaryLanes ? rotary_sub(values[i], cos, partner_values[i], sin)
                                                    : rotary_add(values[i], cos, partner_values[i], sin);
        }
      }
    }
  }
  store_as<Storage>(output, output_vec, group_lane);
}

template <
    int64_t kHeadDim,
    int64_t kRopeDim,
    bool kIsNeox,
    bool kUsePDL,
    typename DType,
    typename CacheDType,
    bool kRoundNormBeforeRope,
    bool kPackKV,
    bool kCacheHasFullWidth,
    bool kAtenNormOrder,
    bool kWideHead,
    typename IdType>
__global__ void fused_qknorm_rope_warp(const QKNormRopeParamsT<kPackKV> __grid_constant__ params) {
  using namespace device;

  static_assert(std::is_same_v<DType, fp16_t> || std::is_same_v<DType, bf16_t>);
  static_assert(kHeadDim <= 256, "Only warp-level fused qknorm+rope is supported");
  static_assert(kHeadDim % kWarpThreads == 0, "head_dim must be divisible by warp size");

  constexpr uint32_t kElemsPerThread = kHeadDim / kWarpThreads;
  constexpr uint32_t kVecSize = kElemsPerThread / 2;
  constexpr uint32_t kRotaryLanes = kRopeDim / kElemsPerThread;
  constexpr uint32_t kHalfRotaryLanes = kRotaryLanes / 2;
  constexpr uint32_t kActiveMask = active_mask<kRotaryLanes>();
  constexpr int64_t kCacheRotaryDim = kCacheHasFullWidth ? 2 * kRopeDim : kRopeDim;
  constexpr int64_t kCosSinStrideBytes = kCacheRotaryDim * sizeof(CacheDType);

  static_assert(kElemsPerThread % 2 == 0, "Each lane must own an even number of elements");
  static_assert(kRopeDim > 0 && kRopeDim <= kHeadDim, "Invalid rope dimension");
  static_assert(kRopeDim % kElemsPerThread == 0, "rope_dim must align with per-lane vector width");
  static_assert(
      !kIsNeox || (kRotaryLanes >= 2 && kRotaryLanes % 2 == 0),
      "NeoX fused qknorm+rope requires an even rotary lane count");
  static_assert(
      !kRoundNormBeforeRope || std::is_same_v<DType, CacheDType> || std::is_same_v<CacheDType, fp32_t>,
      "Rounded QKNorm+RoPE requires cache and activation dtypes to match or an FP32 cache");
  static_assert(!kAtenNormOrder || kHeadDim == 64, "aten-order norm replication is only mapped for head_dim 64");
  static_assert(
      !kWideHead || (kIsNeox && kRoundNormBeforeRope && !kAtenNormOrder && !kPackKV && !kCacheHasFullWidth),
      "The wide (two heads per warp) variant is mapped for the rounded NeoX compact-cache path only");
  static_assert(
      !kWideHead ||
          (kHeadDim % (kWarpThreads / 2) == 0 && wide_elems_per_lane<kHeadDim>() % 2 == 0 &&
           kRopeDim % wide_elems_per_lane<kHeadDim>() == 0 && (kRopeDim / wide_elems_per_lane<kHeadDim>()) % 2 == 0 &&
           kRopeDim / wide_elems_per_lane<kHeadDim>() <= kWarpThreads / 2),
      "Wide-variant lane mapping does not cover this head/rope geometry");

  using Packed = packed_t<DType>;
  using Storage = AlignedVector<Packed, kVecSize>;

  const auto& [q_ptr, k_ptr, q_weight_ptr, k_weight_ptr, cos_sin_cache_ptr, positions, q_stride_bytes, k_stride_bytes, head_stride_bytes, num_qo_heads, num_kv_heads, num_tokens, eps, cache_vec_aligned] =
      static_cast<const QKNormRopeParams&>(params);

  const uint32_t lane_id = threadIdx.x % kWarpThreads;
  const uint32_t warp_id = threadIdx.x / kWarpThreads;
  const uint32_t start_worker_id = blockIdx.x * kWarpsPerBlock + warp_id;
  const uint32_t num_workers = gridDim.x * kWarpsPerBlock;
  const uint32_t num_qk_heads = num_qo_heads + num_kv_heads;
  const uint32_t num_qk_works = num_qk_heads * num_tokens;
  uint32_t num_prefix_works = 0;
  uint32_t num_works = num_qk_works;
  if constexpr (kPackKV) {
    num_prefix_works = params.batch_size * params.prefix_tokens * num_kv_heads;
    num_works += 2 * num_prefix_works + num_tokens * num_kv_heads;
  }

  PDLWaitPrimary<kUsePDL>();

  if constexpr (kWideHead) {
    // Two heads per warp: the low and high 16-lane groups each own one work,
    // so every warp instruction moves two heads' worth of data.
    using StorageWide = AlignedVector<Packed, wide_elems_per_lane<kHeadDim>() / 2>;
    const uint32_t group_id = lane_id >> 4;
    const uint32_t group_base = group_id << 4;
    const uint32_t group_lane = lane_id & 15u;
    const uint32_t iter_stride = num_workers * 2;
    const uint32_t stride_tokens = iter_stride / num_qk_heads;
    const uint32_t stride_heads = iter_stride % num_qk_heads;
    const uint32_t start_work = start_worker_id * 2;
    uint32_t walk_token = start_work / num_qk_heads;
    uint32_t walk_head = start_work % num_qk_heads;
    for (uint32_t base = start_work; base < num_works; base += iter_stride) {
      uint32_t next_token = walk_token;
      uint32_t next_head = walk_head + 1;
      if (next_head == num_qk_heads) {
        next_head = 0;
        ++next_token;
      }
      const uint32_t token_id = group_id ? next_token : walk_token;
      const uint32_t head_id = group_id ? next_head : walk_head;
      if (base + group_id < num_works) {
        const bool load_q = head_id < num_qo_heads;
        const void* input = load_q ? pointer::offset(q_ptr, token_id * q_stride_bytes, head_id * head_stride_bytes)
                                   : pointer::offset(k_ptr, token_id * k_stride_bytes, head_id * head_stride_bytes);
        auto input_vec = load_as<StorageWide>(input, group_lane);
        const auto weight_vec = load_as<StorageWide>(load_q ? q_weight_ptr : k_weight_ptr, group_lane);
        const auto pos = static_cast<int64_t>(static_cast<const IdType*>(positions)[token_id]);
        qknorm_rope_process_work_wide<kHeadDim, kRopeDim, DType, CacheDType>(
            input_vec,
            weight_vec,
            cos_sin_cache_ptr,
            pos,
            const_cast<void*>(input),
            group_lane,
            group_base,
            eps,
            cache_vec_aligned);
      }
      walk_token += stride_tokens;
      walk_head += stride_heads;
      if (walk_head >= num_qk_heads) {
        walk_head -= num_qk_heads;
        ++walk_token;
      }
    }
    PDLTriggerSecondary<kUsePDL>();
    return;
  }

  // The runtime-divisor div/mod per work costs ~20 issue slots each on this
  // issue-bound kernel; the two divisions run once and the (token, head)
  // cursor advances by the fixed grid stride instead. Measured -3.4% (hd128
  // DiT) and -3.5% (hd64 aten) on H200, direct-call ABAB min-of-20. The
  // packed variant keeps the direct div/mod: its copy works skip the cursor
  // advance, and its qk section is not the packed path's bottleneck.
  constexpr bool kWalkCursor = !kPackKV;
  uint32_t stride_tokens = 0;
  uint32_t stride_heads = 0;
  uint32_t walk_token = 0;
  uint32_t walk_head = 0;
  if constexpr (kWalkCursor) {
    stride_tokens = num_workers / num_qk_heads;
    stride_heads = num_workers % num_qk_heads;
    walk_token = start_worker_id / num_qk_heads;
    walk_head = start_worker_id % num_qk_heads;
  }

  for (uint32_t idx = start_worker_id; idx < num_works; idx += num_workers) {
    if constexpr (kPackKV) {
      if (idx >= num_qk_works) {
        const uint32_t copy_idx = idx - num_qk_works;
        const bool copy_k_prefix = copy_idx < num_prefix_works;
        const bool copy_v_prefix = copy_idx >= num_prefix_works && copy_idx < 2 * num_prefix_works;
        const uint32_t local_idx =
            copy_k_prefix ? copy_idx : (copy_v_prefix ? copy_idx - num_prefix_works : copy_idx - 2 * num_prefix_works);
        const uint32_t token_id = local_idx / num_kv_heads;
        const uint32_t head_id = local_idx % num_kv_heads;
        const bool copy_prefix = copy_k_prefix || copy_v_prefix;
        const uint32_t batch_id = token_id / (copy_prefix ? params.prefix_tokens : params.suffix_tokens);
        const uint32_t sequence_id = token_id % (copy_prefix ? params.prefix_tokens : params.suffix_tokens);
        const uint32_t packed_token_id = batch_id * (params.prefix_tokens + params.suffix_tokens) +
                                         (copy_prefix ? sequence_id : params.prefix_tokens + sequence_id);
        const void* input = nullptr;
        void* output = nullptr;
        if (copy_k_prefix) {
          input = pointer::offset(
              params.k_prefix_ptr, token_id * params.k_prefix_stride_bytes, head_id * head_stride_bytes);
          output = pointer::offset(
              params.packed_k_ptr,
              packed_token_id * params.packed_token_stride_bytes,
              head_id * params.packed_head_stride_bytes);
        } else {
          const void* v_ptr = copy_v_prefix ? params.v_prefix_ptr : params.v_ptr;
          const int64_t v_stride = copy_v_prefix ? params.v_prefix_stride_bytes : params.v_stride_bytes;
          input = pointer::offset(v_ptr, token_id * v_stride, head_id * head_stride_bytes);
          output = pointer::offset(
              params.packed_v_ptr,
              packed_token_id * params.packed_token_stride_bytes,
              head_id * params.packed_head_stride_bytes);
        }
        const auto copy_vec = load_as<Storage>(input, lane_id);
        store_as<Storage>(output, copy_vec, lane_id);
        continue;
      }
    }

    uint32_t token_id;
    uint32_t head_id;
    if constexpr (kWalkCursor) {
      token_id = walk_token;
      head_id = walk_head;
      walk_token += stride_tokens;
      walk_head += stride_heads;
      if (walk_head >= num_qk_heads) {
        walk_head -= num_qk_heads;
        ++walk_token;
      }
    } else {
      token_id = idx / num_qk_heads;
      head_id = idx % num_qk_heads;
    }
    const bool load_q = head_id < num_qo_heads;
    const void* input = load_q ? pointer::offset(q_ptr, token_id * q_stride_bytes, head_id * head_stride_bytes)
                               : pointer::offset(k_ptr, token_id * k_stride_bytes, head_id * head_stride_bytes);
    void* output = const_cast<void*>(input);
    if constexpr (kPackKV) {
      if (!load_q) {
        const uint32_t batch_id = token_id / params.suffix_tokens;
        const uint32_t sequence_id = token_id % params.suffix_tokens;
        const uint32_t kv_head_id = head_id - num_qo_heads;
        const uint32_t packed_token_id =
            batch_id * (params.prefix_tokens + params.suffix_tokens) + params.prefix_tokens + sequence_id;
        output = pointer::offset(
            params.packed_k_ptr,
            packed_token_id * params.packed_token_stride_bytes,
            kv_head_id * params.packed_head_stride_bytes);
      }
    }
    const void* weight_ptr = load_q ? q_weight_ptr : k_weight_ptr;

    auto input_vec = load_as<Storage>(input, lane_id);
    const auto weight_vec = load_as<Storage>(weight_ptr, lane_id);

    if constexpr (kRoundNormBeforeRope) {
      Storage output_vec;
      if constexpr (kAtenNormOrder) {
        const float norm_factor = aten_order_rms_norm_factor<kHeadDim>(input_vec, eps, lane_id);
#pragma unroll
        for (uint32_t j = 0; j < kVecSize; ++j) {
          const auto [x0, x1] = cast<fp32x2_t>(input_vec[j]);
          const auto [w0, w1] = cast<fp32x2_t>(weight_vec[j]);
          output_vec[j] = cast<Packed, fp32x2_t>({x0 * norm_factor * w0, x1 * norm_factor * w1});
        }
      } else {
        output_vec = norm::apply_norm_warp<kHeadDim>(input_vec, weight_vec, eps);
      }
      const auto pos = static_cast<int64_t>(static_cast<const IdType*>(positions)[token_id]);
      const auto cos_ptr = static_cast<const CacheDType*>(pointer::offset(cos_sin_cache_ptr, pos * kCosSinStrideBytes));
      const auto sin_ptr = cos_ptr + (kCacheHasFullWidth ? kRopeDim : kRopeDim / 2);

      if constexpr (kIsNeox) {
        if (lane_id < kRotaryLanes) {
          const auto partner_lane =
              lane_id < kHalfRotaryLanes ? lane_id + kHalfRotaryLanes : lane_id - kHalfRotaryLanes;
#pragma unroll
          for (uint32_t j = 0; j < kVecSize; ++j) {
            auto partner_vec = output_vec[j];
            auto partner_bits = reinterpret_cast<const uint32_t&>(partner_vec);
            partner_bits = __shfl_sync(kActiveMask, partner_bits, partner_lane);
            reinterpret_cast<uint32_t&>(partner_vec) = partner_bits;
            auto& values = unpack(output_vec[j]);
            const auto& partner_values = unpack(partner_vec);
#pragma unroll
            for (uint32_t i = 0; i < 2; ++i) {
              const auto cache_idx =
                  (kCacheHasFullWidth ? lane_id : lane_id % kHalfRotaryLanes) * kElemsPerThread + 2 * j + i;
              const auto cos = load_cache_value(cos_ptr, cache_idx);
              const auto sin = load_cache_value(sin_ptr, cache_idx);
              if constexpr (std::is_same_v<CacheDType, fp32_t>) {
                values[i] = lane_id < kHalfRotaryLanes ? rotary_sub_fp32(values[i], cos, partner_values[i], sin)
                                                       : rotary_add_fp32(values[i], cos, partner_values[i], sin);
              } else {
                values[i] = lane_id < kHalfRotaryLanes ? rotary_sub(values[i], cos, partner_values[i], sin)
                                                       : rotary_add(values[i], cos, partner_values[i], sin);
              }
            }
          }
        }
      } else {
        if (lane_id < kRotaryLanes) {
#pragma unroll
          for (uint32_t j = 0; j < kVecSize; ++j) {
            auto& values = unpack(output_vec[j]);
            const auto cache_idx_0 =
                kCacheHasFullWidth ? lane_id * kElemsPerThread + 2 * j : lane_id * kElemsPerThread / 2 + j;
            const auto cache_idx_1 = kCacheHasFullWidth ? cache_idx_0 + 1 : cache_idx_0;
            const auto cos_0 = load_cache_value(cos_ptr, cache_idx_0);
            const auto sin_0 = load_cache_value(sin_ptr, cache_idx_0);
            const auto cos_1 = load_cache_value(cos_ptr, cache_idx_1);
            const auto sin_1 = load_cache_value(sin_ptr, cache_idx_1);
            const auto x = values[0];
            const auto y = values[1];
            if constexpr (std::is_same_v<CacheDType, fp32_t>) {
              values[0] = rotary_sub_fp32(x, cos_0, y, sin_0);
              values[1] = rotary_add_fp32(y, cos_1, x, sin_1);
            } else {
              values[0] = rotary_sub(x, cos_0, y, sin_0);
              values[1] = rotary_add(y, cos_1, x, sin_1);
            }
          }
        }
      }
      store_as<Storage>(output, output_vec, lane_id);
      continue;
    }

    float elems[kElemsPerThread];
    float sum_of_squares = 0.0f;

#pragma unroll
    for (uint32_t j = 0; j < kVecSize; ++j) {
      const auto [x0, x1] = cast<fp32x2_t>(input_vec[j]);
      elems[2 * j] = x0;
      elems[2 * j + 1] = x1;
      sum_of_squares += x0 * x0 + x1 * x1;
    }

    float norm_factor;
    if constexpr (kAtenNormOrder) {
      norm_factor = aten_order_rms_norm_factor<kHeadDim>(input_vec, eps, lane_id);
    } else {
      sum_of_squares = warp::reduce_sum(sum_of_squares);
      norm_factor = math::rsqrt(sum_of_squares / static_cast<float>(kHeadDim) + eps);
    }

#pragma unroll
    for (uint32_t j = 0; j < kVecSize; ++j) {
      const auto [w0, w1] = cast<fp32x2_t>(weight_vec[j]);
      elems[2 * j] *= norm_factor * w0;
      elems[2 * j + 1] *= norm_factor * w1;
    }

    if constexpr (kIsNeox) {
      if (lane_id < kRotaryLanes) {
        const auto pos = static_cast<int64_t>(static_cast<const IdType*>(positions)[token_id]);
        const auto cos_ptr =
            static_cast<const CacheDType*>(pointer::offset(cos_sin_cache_ptr, pos * kCosSinStrideBytes));
        const auto sin_ptr = cos_ptr + (kCacheHasFullWidth ? kRopeDim : kRopeDim / 2);
        const auto partner_lane = lane_id < kHalfRotaryLanes ? lane_id + kHalfRotaryLanes : lane_id - kHalfRotaryLanes;

#pragma unroll
        for (uint32_t i = 0; i < kElemsPerThread; ++i) {
          float swapped = __shfl_sync(kActiveMask, elems[i], partner_lane);
          if (lane_id < kHalfRotaryLanes) {
            swapped = -swapped;
          }
          const auto cache_idx = (kCacheHasFullWidth ? lane_id : lane_id % kHalfRotaryLanes) * kElemsPerThread + i;
          const float cos = cast<fp32_t>(load_cache_value(cos_ptr, cache_idx));
          const float sin = cast<fp32_t>(load_cache_value(sin_ptr, cache_idx));
          elems[i] = elems[i] * cos + swapped * sin;
        }
      }
    } else {
      if (lane_id < kRotaryLanes) {
        const auto pos = static_cast<int64_t>(static_cast<const IdType*>(positions)[token_id]);
        const auto cos_ptr =
            static_cast<const CacheDType*>(pointer::offset(cos_sin_cache_ptr, pos * kCosSinStrideBytes));
        const auto sin_ptr = cos_ptr + (kCacheHasFullWidth ? kRopeDim : kRopeDim / 2);

#pragma unroll
        for (uint32_t i = 0; i < kElemsPerThread; i += 2) {
          const float x = elems[i];
          const float y = elems[i + 1];
          const auto cache_idx_0 =
              kCacheHasFullWidth ? lane_id * kElemsPerThread + i : (lane_id * kElemsPerThread + i) / 2;
          const auto cache_idx_1 = kCacheHasFullWidth ? cache_idx_0 + 1 : cache_idx_0;
          const float cos_0 = cast<fp32_t>(load_cache_value(cos_ptr, cache_idx_0));
          const float sin_0 = cast<fp32_t>(load_cache_value(sin_ptr, cache_idx_0));
          const float cos_1 = cast<fp32_t>(load_cache_value(cos_ptr, cache_idx_1));
          const float sin_1 = cast<fp32_t>(load_cache_value(sin_ptr, cache_idx_1));
          elems[i] = x * cos_0 - y * sin_0;
          elems[i + 1] = y * cos_1 + x * sin_1;
        }
      }
    }

#pragma unroll
    for (uint32_t j = 0; j < kVecSize; ++j) {
      input_vec[j] = cast<Packed, fp32x2_t>({elems[2 * j], elems[2 * j + 1]});
    }
    store_as<Storage>(output, input_vec, lane_id);
  }

  PDLTriggerSecondary<kUsePDL>();
}

template <
    int64_t kHeadDim,
    int64_t kRopeDim,
    bool kIsNeox,
    bool kUsePDL,
    typename DType,
    typename CacheDType,
    bool kRoundNormBeforeRope,
    bool kCacheHasFullWidth,
    bool kAtenNormOrder,
    bool kWideHead = false>
struct QKNormRopeKernel {
  static_assert(kHeadDim <= 256, "Only head_dim <= 256 is supported");
  template <typename IdType>
  static constexpr auto kernel = fused_qknorm_rope_warp<
      kHeadDim,
      kRopeDim,
      kIsNeox,
      kUsePDL,
      DType,
      CacheDType,
      kRoundNormBeforeRope,
      false,
      kCacheHasFullWidth,
      kAtenNormOrder,
      kWideHead,
      IdType>;

  static void
  run(const tvm::ffi::TensorView q,
      const tvm::ffi::TensorView k,
      const tvm::ffi::TensorView q_weight,
      const tvm::ffi::TensorView k_weight,
      const tvm::ffi::TensorView cos_sin_cache,
      const tvm::ffi::TensorView positions,
      float eps) {
    using namespace host;

    auto N = SymbolicSize{"num_tokens"};
    auto Q = SymbolicSize{"num_qo_heads"};
    auto K = SymbolicSize{"num_kv_heads"};
    auto D = SymbolicSize{"head_dim"};
    auto R = SymbolicSize{"rope_dim"};
    auto Dq = SymbolicSize{"q_stride"};
    auto Dk = SymbolicSize{"k_stride"};
    auto Dd = SymbolicSize{"head_stride"};
    auto device = SymbolicDevice{};
    auto id_type = SymbolicDType{};
    D.set_value(kHeadDim);
    R.set_value(kRopeDim);
    device.set_options<kDLCUDA>();

    TensorMatcher({N, Q, D}).with_strides({Dq, Dd, 1}).with_dtype<DType>().with_device(device).verify(q);
    TensorMatcher({N, K, D}).with_strides({Dk, Dd, 1}).with_dtype<DType>().with_device(device).verify(k);
    TensorMatcher({D}).with_dtype<DType>().with_device(device).verify(q_weight).verify(k_weight);
    TensorMatcher({-1, kCacheHasFullWidth ? 2 * kRopeDim : kRopeDim})
        .with_dtype<CacheDType>()
        .with_device(device)
        .verify(cos_sin_cache);
    TensorMatcher({N}).with_dtype<int32_t, int64_t>(id_type).with_device(device).verify(positions);

    const auto num_tokens = static_cast<uint32_t>(N.unwrap());
    const auto num_qo_heads = static_cast<uint32_t>(Q.unwrap());
    const auto num_kv_heads = static_cast<uint32_t>(K.unwrap());
    if (num_tokens == 0 || (num_qo_heads == 0 && num_kv_heads == 0)) return;
    const auto q_stride_bytes = static_cast<int64_t>(Dq.unwrap() * sizeof(DType));
    const auto k_stride_bytes = static_cast<int64_t>(Dk.unwrap() * sizeof(DType));
    const auto head_stride_bytes = static_cast<int64_t>(Dd.unwrap() * sizeof(DType));

    if constexpr (kWideHead) {
      // 16 lanes per head load kHeadDim/16 elements per lane in one vector;
      // every row must sit on that vector boundary.
      constexpr int64_t kWideBytes = wide_elems_per_lane<kHeadDim>() * sizeof(DType);
      for (const auto* tensor : {&q, &k, &q_weight, &k_weight}) {
        CHECK_HOST(reinterpret_cast<uintptr_t>(tensor->data_ptr()) % kWideBytes == 0)
            << "wide fused qknorm+rope requires " << kWideBytes << "B-aligned tensors";
      }
      CHECK_HOST(
          q_stride_bytes % kWideBytes == 0 && k_stride_bytes % kWideBytes == 0 && head_stride_bytes % kWideBytes == 0)
          << "wide fused qknorm+rope requires " << kWideBytes << "B-aligned strides";
    }

    const int64_t k_offset = static_cast<int64_t>(num_qo_heads) * head_stride_bytes;
    const auto params = QKNormRopeParams{
        .q_ptr = q.data_ptr(),
        .k_ptr = pointer::offset(k.data_ptr(), -k_offset),
        .q_weight_ptr = q_weight.data_ptr(),
        .k_weight_ptr = k_weight.data_ptr(),
        .cos_sin_cache_ptr = cos_sin_cache.data_ptr(),
        .positions = positions.data_ptr(),
        .q_stride_bytes = q_stride_bytes,
        .k_stride_bytes = k_stride_bytes,
        .head_stride_bytes = head_stride_bytes,
        .num_qo_heads = num_qo_heads,
        .num_kv_heads = num_kv_heads,
        .num_tokens = num_tokens,
        .eps = eps,
        .cache_vec_aligned =
            kWideHead && wide_cache_span_aligned<kHeadDim, kRopeDim, CacheDType>(cos_sin_cache.data_ptr()),
    };

    const auto is_int32 = id_type.is_type<int32_t>();
    const auto selected_kernel = is_int32 ? kernel<int32_t> : kernel<int64_t>;
    const uint32_t kNumSM = runtime::get_sm_count(device.unwrap().device_id);
    static const uint32_t kOccupancyTable[2] = {
        runtime::get_blocks_per_sm(kernel<int32_t>, kThreadsPerBlock),
        runtime::get_blocks_per_sm(kernel<int64_t>, kThreadsPerBlock),
    };
    const auto max_blocks = kOccupancyTable[is_int32 ? 0 : 1] * kNumSM;
    const auto num_works = (num_qo_heads + num_kv_heads) * num_tokens;
    constexpr uint32_t kWorksPerWarpIter = kWideHead ? 2 : 1;
    const auto needed_blocks = div_ceil(num_works, kWarpsPerBlock * kWorksPerWarpIter);
    const auto num_blocks = std::min(max_blocks, needed_blocks);
    LaunchKernel(num_blocks, kThreadsPerBlock, device.unwrap()).enable_pdl(kUsePDL)(selected_kernel, params);
  }
};

template <
    int64_t kHeadDim,
    int64_t kRopeDim,
    bool kIsNeox,
    bool kUsePDL,
    typename DType,
    typename CacheDType,
    bool kRoundNormBeforeRope,
    bool kCacheHasFullWidth,
    bool kAtenNormOrder>
struct QKNormRopePackKVKernel {
  static_assert(!kCacheHasFullWidth, "KV packing does not support full-width cos/sin caches");
  template <typename IdType>
  static constexpr auto kernel = fused_qknorm_rope_warp<
      kHeadDim,
      kRopeDim,
      kIsNeox,
      kUsePDL,
      DType,
      CacheDType,
      kRoundNormBeforeRope,
      true,
      kCacheHasFullWidth,
      kAtenNormOrder,
      /*kWideHead=*/false,
      IdType>;

  static void
  run(const tvm::ffi::TensorView q,
      const tvm::ffi::TensorView k,
      const tvm::ffi::TensorView v,
      const tvm::ffi::TensorView k_prefix,
      const tvm::ffi::TensorView v_prefix,
      const tvm::ffi::TensorView packed_k,
      const tvm::ffi::TensorView packed_v,
      const tvm::ffi::TensorView q_weight,
      const tvm::ffi::TensorView k_weight,
      const tvm::ffi::TensorView cos_sin_cache,
      const tvm::ffi::TensorView positions,
      int64_t batch_size,
      int64_t prefix_tokens,
      int64_t suffix_tokens,
      float eps) {
    using namespace host;

    auto N = SymbolicSize{"num_tokens"};
    auto NP = SymbolicSize{"num_prefix_tokens"};
    auto B = SymbolicSize{"batch_size"};
    auto T = SymbolicSize{"packed_tokens"};
    auto Q = SymbolicSize{"num_qo_heads"};
    auto K = SymbolicSize{"num_kv_heads"};
    auto D = SymbolicSize{"head_dim"};
    auto R = SymbolicSize{"rope_dim"};
    auto Dq = SymbolicSize{"q_stride"};
    auto Dk = SymbolicSize{"k_stride"};
    auto Dv = SymbolicSize{"v_stride"};
    auto Dkp = SymbolicSize{"k_prefix_stride"};
    auto Dvp = SymbolicSize{"v_prefix_stride"};
    auto Dd = SymbolicSize{"head_stride"};
    auto device = SymbolicDevice{};
    auto id_type = SymbolicDType{};
    N.set_value(batch_size * suffix_tokens);
    NP.set_value(batch_size * prefix_tokens);
    B.set_value(batch_size);
    T.set_value(prefix_tokens + suffix_tokens);
    D.set_value(kHeadDim);
    R.set_value(kRopeDim);
    device.set_options<kDLCUDA>();

    TensorMatcher({N, Q, D}).with_strides({Dq, Dd, 1}).with_dtype<DType>().with_device(device).verify(q);
    TensorMatcher({N, K, D}).with_strides({Dk, Dd, 1}).with_dtype<DType>().with_device(device).verify(k);
    TensorMatcher({N, K, D}).with_strides({Dv, Dd, 1}).with_dtype<DType>().with_device(device).verify(v);
    TensorMatcher({NP, K, D}).with_strides({Dkp, Dd, 1}).with_dtype<DType>().with_device(device).verify(k_prefix);
    TensorMatcher({NP, K, D}).with_strides({Dvp, Dd, 1}).with_dtype<DType>().with_device(device).verify(v_prefix);
    TensorMatcher({B, T, K, D}).with_dtype<DType>().with_device(device).verify(packed_k).verify(packed_v);
    RuntimeCheck(packed_k.is_contiguous(), "packed_k must be contiguous");
    RuntimeCheck(packed_v.is_contiguous(), "packed_v must be contiguous");
    TensorMatcher({D}).with_dtype<DType>().with_device(device).verify(q_weight).verify(k_weight);
    TensorMatcher({-1, R}).with_dtype<CacheDType>().with_device(device).verify(cos_sin_cache);
    TensorMatcher({N}).with_dtype<int32_t, int64_t>(id_type).with_device(device).verify(positions);

    const auto num_tokens = static_cast<uint32_t>(N.unwrap());
    const auto num_qo_heads = static_cast<uint32_t>(Q.unwrap());
    const auto num_kv_heads = static_cast<uint32_t>(K.unwrap());
    if (num_tokens == 0 || (num_qo_heads == 0 && num_kv_heads == 0)) return;
    const auto head_stride_bytes = static_cast<int64_t>(Dd.unwrap() * sizeof(DType));
    const int64_t k_offset = static_cast<int64_t>(num_qo_heads) * head_stride_bytes;
    QKNormRopePackKVParams params{};
    params.q_ptr = q.data_ptr();
    params.k_ptr = pointer::offset(k.data_ptr(), -k_offset);
    params.q_weight_ptr = q_weight.data_ptr();
    params.k_weight_ptr = k_weight.data_ptr();
    params.cos_sin_cache_ptr = cos_sin_cache.data_ptr();
    params.positions = positions.data_ptr();
    params.q_stride_bytes = static_cast<int64_t>(Dq.unwrap() * sizeof(DType));
    params.k_stride_bytes = static_cast<int64_t>(Dk.unwrap() * sizeof(DType));
    params.head_stride_bytes = head_stride_bytes;
    params.num_qo_heads = num_qo_heads;
    params.num_kv_heads = num_kv_heads;
    params.num_tokens = num_tokens;
    params.eps = eps;
    params.v_ptr = v.data_ptr();
    params.k_prefix_ptr = k_prefix.data_ptr();
    params.v_prefix_ptr = v_prefix.data_ptr();
    params.packed_k_ptr = packed_k.data_ptr();
    params.packed_v_ptr = packed_v.data_ptr();
    params.v_stride_bytes = static_cast<int64_t>(Dv.unwrap() * sizeof(DType));
    params.k_prefix_stride_bytes = static_cast<int64_t>(Dkp.unwrap() * sizeof(DType));
    params.v_prefix_stride_bytes = static_cast<int64_t>(Dvp.unwrap() * sizeof(DType));
    params.packed_token_stride_bytes = static_cast<int64_t>(num_kv_heads * kHeadDim * sizeof(DType));
    params.packed_head_stride_bytes = static_cast<int64_t>(kHeadDim * sizeof(DType));
    params.batch_size = static_cast<uint32_t>(batch_size);
    params.prefix_tokens = static_cast<uint32_t>(prefix_tokens);
    params.suffix_tokens = static_cast<uint32_t>(suffix_tokens);

    const auto is_int32 = id_type.is_type<int32_t>();
    const auto selected_kernel = is_int32 ? kernel<int32_t> : kernel<int64_t>;
    const uint32_t kNumSM = runtime::get_sm_count(device.unwrap().device_id);
    static const uint32_t kOccupancyTable[2] = {
        runtime::get_blocks_per_sm(kernel<int32_t>, kThreadsPerBlock),
        runtime::get_blocks_per_sm(kernel<int64_t>, kThreadsPerBlock),
    };
    const auto max_blocks = kOccupancyTable[is_int32 ? 0 : 1] * kNumSM;
    const uint32_t num_prefix_works = static_cast<uint32_t>(batch_size * prefix_tokens) * num_kv_heads;
    const uint32_t num_works =
        (num_qo_heads + num_kv_heads) * num_tokens + 2 * num_prefix_works + num_tokens * num_kv_heads;
    const auto needed_blocks = div_ceil(num_works, kWarpsPerBlock);
    const auto num_blocks = std::min(max_blocks, needed_blocks);
    LaunchKernel(num_blocks, kThreadsPerBlock, device.unwrap()).enable_pdl(kUsePDL)(selected_kernel, params);
  }
};

}  // namespace sglang
