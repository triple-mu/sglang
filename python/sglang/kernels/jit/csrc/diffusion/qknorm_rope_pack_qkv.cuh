// Fused per-head QKNorm + RoPE + destination-major QKV pack for the Ulysses
// input all-to-all of packed (thd) DiTs.
//
// It replaces this two-kernel chain:
//   1. fused_qknorm_rope (diffusion/qknorm_rope.cuh, round_norm_before_rope)
//      read q,k -> normalize + rotate -> write q,k back in place
//   2. pack_qkv_destination_major (triton/ulysses_qkv.py)
//      read q,k,v -> write packed[W, N, H/W, 3*D]
//
// The chain moves q,k twice; fusing removes that whole read-modify-write pass.
// The arithmetic is copied verbatim from qknorm_rope.cuh: same lane ownership,
// same warp reduction order, same bf16 rounding points, so the packed output is
// bit-identical to running the two kernels back to back. The pack itself is a
// pure permutation and contributes no arithmetic.

#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <sgl_kernel/impl/norm.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>
#include <type_traits>

namespace sglang {

namespace qknorm_rope_pack_qkv {

namespace {

struct Params {
  void* __restrict__ packed_ptr;
  const void* __restrict__ q_ptr;
  const void* __restrict__ k_ptr;
  const void* __restrict__ v_ptr;
  const void* __restrict__ q_weight_ptr;
  const void* __restrict__ k_weight_ptr;
  const void* __restrict__ cos_sin_cache_ptr;
  const void* __restrict__ positions;
  int64_t q_row_stride_bytes;
  int64_t k_row_stride_bytes;
  int64_t v_row_stride_bytes;
  int64_t head_stride_bytes;
  uint32_t num_heads;
  uint32_t local_heads;
  uint32_t num_tokens;
  float eps;
};

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

/**
 * \brief Rotate one already-normalized head in place, in the activation dtype.
 *
 * Byte-for-byte the `kRoundNormBeforeRope` branch of `fused_qknorm_rope_warp`:
 * the norm result is rounded to `DType` first and the rotation runs in `DType`
 * arithmetic, which is the numerical contract the eager path defines.
 *
 * \tparam kHeadDim  Elements per head.
 * \tparam kRopeDim  Rotary width; lanes beyond it keep the norm result.
 * \tparam kIsNeox   NeoX half-split rotation when true, interleaved otherwise.
 * \param vec        Per-lane slice of the head, updated in place.
 * \param cos_ptr    Row of the cos/sin cache for this token.
 * \param lane_id    Lane index inside the warp.
 */
template <int64_t kHeadDim, int64_t kRopeDim, bool kIsNeox, typename Storage, typename CacheDType>
SGL_DEVICE void apply_rope_rounded(Storage& vec, const CacheDType* cos_ptr, uint32_t lane_id) {
  using namespace device;

  constexpr uint32_t kElemsPerThread = kHeadDim / kWarpThreads;
  constexpr uint32_t kVecSize = kElemsPerThread / 2;
  constexpr uint32_t kRotaryLanes = kRopeDim / kElemsPerThread;
  constexpr uint32_t kHalfRotaryLanes = kRotaryLanes / 2;
  constexpr uint32_t kActiveMask = active_mask<kRotaryLanes>();

  const auto sin_ptr = cos_ptr + kRopeDim / 2;

  if constexpr (kIsNeox) {
    if (lane_id < kRotaryLanes) {
      const auto partner_lane = lane_id < kHalfRotaryLanes ? lane_id + kHalfRotaryLanes : lane_id - kHalfRotaryLanes;
#pragma unroll
      for (uint32_t j = 0; j < kVecSize; ++j) {
        auto partner_vec = vec[j];
        auto partner_bits = reinterpret_cast<const uint32_t&>(partner_vec);
        partner_bits = __shfl_sync(kActiveMask, partner_bits, partner_lane);
        reinterpret_cast<uint32_t&>(partner_vec) = partner_bits;
        auto& values = unpack(vec[j]);
        const auto& partner_values = unpack(partner_vec);
#pragma unroll
        for (uint32_t i = 0; i < 2; ++i) {
          const auto half_idx = (lane_id % kHalfRotaryLanes) * kElemsPerThread + 2 * j + i;
          const auto cos = load_cache_value(cos_ptr, half_idx);
          const auto sin = load_cache_value(sin_ptr, half_idx);
          values[i] = lane_id < kHalfRotaryLanes ? values[i] * cos - partner_values[i] * sin
                                                 : values[i] * cos + partner_values[i] * sin;
        }
      }
    }
  } else {
    if (lane_id < kRotaryLanes) {
#pragma unroll
      for (uint32_t j = 0; j < kVecSize; ++j) {
        auto& values = unpack(vec[j]);
        const auto half_idx = lane_id * kElemsPerThread / 2 + j;
        const auto cos = load_cache_value(cos_ptr, half_idx);
        const auto sin = load_cache_value(sin_ptr, half_idx);
        const auto x = values[0];
        const auto y = values[1];
        values[0] = x * cos - y * sin;
        values[1] = y * cos + x * sin;
      }
    }
  }
}

/**
 * \brief One warp per (token, head): norm+rope q and k, copy v, write packed.
 *
 * `packed[dest, token, local_head]` holds `q | k | v` back to back, where
 * `dest = head / local_heads` is the Ulysses rank that owns the head. The three
 * stores of a warp therefore land in one contiguous `3 * kHeadDim` run.
 */
template <
    int64_t kHeadDim,
    int64_t kRopeDim,
    bool kIsNeox,
    bool kUsePDL,
    typename DType,
    typename CacheDType,
    typename IdType>
__global__ void qknorm_rope_pack_qkv_warp(const Params __grid_constant__ params) {
  using namespace device;

  static_assert(std::is_same_v<DType, fp16_t> || std::is_same_v<DType, bf16_t>);
  static_assert(std::is_same_v<DType, CacheDType>, "Rounded QKNorm+RoPE requires matching cache and activation dtypes");
  static_assert(kHeadDim <= 256, "Only warp-level fused qknorm+rope is supported");
  static_assert(kHeadDim % kWarpThreads == 0, "head_dim must be divisible by warp size");

  constexpr uint32_t kElemsPerThread = kHeadDim / kWarpThreads;
  constexpr uint32_t kVecSize = kElemsPerThread / 2;
  constexpr uint32_t kRotaryLanes = kRopeDim / kElemsPerThread;
  constexpr int64_t kCosSinStrideBytes = kRopeDim * sizeof(CacheDType);
  constexpr int64_t kHeadBytes = kHeadDim * sizeof(DType);

  static_assert(kElemsPerThread % 2 == 0, "Each lane must own an even number of elements");
  static_assert(kRopeDim > 0 && kRopeDim <= kHeadDim, "Invalid rope dimension");
  static_assert(kRopeDim % kElemsPerThread == 0, "rope_dim must align with per-lane vector width");
  static_assert(
      !kIsNeox || (kRotaryLanes >= 2 && kRotaryLanes % 2 == 0),
      "NeoX fused qknorm+rope requires an even rotary lane count");

  using Packed = packed_t<DType>;
  using Storage = AlignedVector<Packed, kVecSize>;

  const auto& [packed_ptr, q_ptr, k_ptr, v_ptr, q_weight_ptr, k_weight_ptr, cos_sin_cache_ptr, positions, q_row_stride_bytes, k_row_stride_bytes, v_row_stride_bytes, head_stride_bytes, num_heads, local_heads, num_tokens, eps] =
      params;

  const uint32_t lane_id = threadIdx.x % kWarpThreads;
  const uint32_t warp_id = threadIdx.x / kWarpThreads;
  const uint32_t start_worker_id = blockIdx.x * kWarpsPerBlock + warp_id;
  const uint32_t num_workers = gridDim.x * kWarpsPerBlock;
  const uint32_t num_works = num_heads * num_tokens;

  PDLWaitPrimary<kUsePDL>();

  const auto q_weight_vec = load_as<Storage>(q_weight_ptr, lane_id);
  const auto k_weight_vec = load_as<Storage>(k_weight_ptr, lane_id);

  for (uint32_t idx = start_worker_id; idx < num_works; idx += num_workers) {
    const uint32_t token_id = idx / num_heads;
    const uint32_t head_id = idx % num_heads;
    const uint32_t dest = head_id / local_heads;
    const uint32_t local_head = head_id % local_heads;

    const int64_t head_offset = static_cast<int64_t>(head_id) * head_stride_bytes;
    const auto q_in = load_as<Storage>(pointer::offset(q_ptr, token_id * q_row_stride_bytes, head_offset), lane_id);
    const auto k_in = load_as<Storage>(pointer::offset(k_ptr, token_id * k_row_stride_bytes, head_offset), lane_id);
    const auto v_in = load_as<Storage>(pointer::offset(v_ptr, token_id * v_row_stride_bytes, head_offset), lane_id);

    const auto pos = static_cast<int64_t>(static_cast<const IdType*>(positions)[token_id]);
    const auto cos_ptr = static_cast<const CacheDType*>(pointer::offset(cos_sin_cache_ptr, pos * kCosSinStrideBytes));

    auto q_out = norm::apply_norm_warp<kHeadDim>(q_in, q_weight_vec, eps);
    apply_rope_rounded<kHeadDim, kRopeDim, kIsNeox>(q_out, cos_ptr, lane_id);
    auto k_out = norm::apply_norm_warp<kHeadDim>(k_in, k_weight_vec, eps);
    apply_rope_rounded<kHeadDim, kRopeDim, kIsNeox>(k_out, cos_ptr, lane_id);

    const int64_t packed_slot = (static_cast<int64_t>(dest) * num_tokens + token_id) * local_heads + local_head;
    void* dst = pointer::offset(packed_ptr, packed_slot * (3 * kHeadBytes));
    store_as<Storage>(dst, q_out, lane_id);
    store_as<Storage>(pointer::offset(dst, kHeadBytes), k_out, lane_id);
    store_as<Storage>(pointer::offset(dst, 2 * kHeadBytes), v_in, lane_id);
  }

  PDLTriggerSecondary<kUsePDL>();
}

}  // namespace

template <int64_t kHeadDim, int64_t kRopeDim, bool kIsNeox, bool kUsePDL, typename DType, typename CacheDType>
struct QKNormRopePackQKVKernel {
  template <typename IdType>
  static constexpr auto kernel =
      qknorm_rope_pack_qkv_warp<kHeadDim, kRopeDim, kIsNeox, kUsePDL, DType, CacheDType, IdType>;

  /**
   * \brief Launch the fused kernel.
   *
   * \param packed  Output [world_size, num_tokens, local_heads, 3 * head_dim].
   * \param q,k,v   [num_tokens, num_heads, head_dim], head dim contiguous.
   * \param q_weight,k_weight  Per-head RMSNorm gains, [head_dim].
   * \param cos_sin_cache      [max_position, rope_dim], cos then sin.
   * \param positions          [num_tokens] rotary positions.
   * \param eps                RMSNorm epsilon.
   */
  static void
  run(tvm::ffi::TensorView packed,
      tvm::ffi::TensorView q,
      tvm::ffi::TensorView k,
      tvm::ffi::TensorView v,
      tvm::ffi::TensorView q_weight,
      tvm::ffi::TensorView k_weight,
      tvm::ffi::TensorView cos_sin_cache,
      tvm::ffi::TensorView positions,
      float eps) {
    using namespace host;

    auto N = SymbolicSize{"num_tokens"};
    auto H = SymbolicSize{"num_heads"};
    auto W = SymbolicSize{"world_size"};
    auto HL = SymbolicSize{"local_heads"};
    auto D = SymbolicSize{"head_dim"};
    auto R = SymbolicSize{"rope_dim"};
    auto Dq = SymbolicSize{"q_row_stride"};
    auto Dk = SymbolicSize{"k_row_stride"};
    auto Dv = SymbolicSize{"v_row_stride"};
    auto Dd = SymbolicSize{"head_stride"};
    auto device = SymbolicDevice{};
    auto id_type = SymbolicDType{};
    D.set_value(kHeadDim);
    R.set_value(kRopeDim);
    device.set_options<kDLCUDA>();

    TensorMatcher({N, H, D}).with_strides({Dq, Dd, 1}).with_dtype<DType>().with_device(device).verify(q);
    TensorMatcher({N, H, D}).with_strides({Dk, Dd, 1}).with_dtype<DType>().with_device(device).verify(k);
    TensorMatcher({N, H, D}).with_strides({Dv, Dd, 1}).with_dtype<DType>().with_device(device).verify(v);
    TensorMatcher({W, N, HL, 3 * kHeadDim}).with_dtype<DType>().with_device(device).verify(packed);
    TensorMatcher({D}).with_dtype<DType>().with_device(device).verify(q_weight).verify(k_weight);
    TensorMatcher({-1, R}).with_dtype<CacheDType>().with_device(device).verify(cos_sin_cache);
    TensorMatcher({N}).with_dtype<int32_t, int64_t>(id_type).with_device(device).verify(positions);

    const auto num_tokens = static_cast<uint32_t>(N.unwrap());
    const auto num_heads = static_cast<uint32_t>(H.unwrap());
    const auto world_size = static_cast<uint32_t>(W.unwrap());
    const auto local_heads = static_cast<uint32_t>(HL.unwrap());
    CHECK_HOST(world_size * local_heads == num_heads)
        << "packed layout must cover every head: world_size " << world_size << " * local_heads " << local_heads
        << " != num_heads " << num_heads;

    const auto params = Params{
        .packed_ptr = packed.data_ptr(),
        .q_ptr = q.data_ptr(),
        .k_ptr = k.data_ptr(),
        .v_ptr = v.data_ptr(),
        .q_weight_ptr = q_weight.data_ptr(),
        .k_weight_ptr = k_weight.data_ptr(),
        .cos_sin_cache_ptr = cos_sin_cache.data_ptr(),
        .positions = positions.data_ptr(),
        .q_row_stride_bytes = static_cast<int64_t>(Dq.unwrap() * sizeof(DType)),
        .k_row_stride_bytes = static_cast<int64_t>(Dk.unwrap() * sizeof(DType)),
        .v_row_stride_bytes = static_cast<int64_t>(Dv.unwrap() * sizeof(DType)),
        .head_stride_bytes = static_cast<int64_t>(Dd.unwrap() * sizeof(DType)),
        .num_heads = num_heads,
        .local_heads = local_heads,
        .num_tokens = num_tokens,
        .eps = eps,
    };

    const auto is_int32 = id_type.is_type<int32_t>();
    const auto selected_kernel = is_int32 ? kernel<int32_t> : kernel<int64_t>;
    const uint32_t num_sm = runtime::get_sm_count(device.unwrap().device_id);
    static const uint32_t occupancy[2] = {
        runtime::get_blocks_per_sm(kernel<int32_t>, kThreadsPerBlock),
        runtime::get_blocks_per_sm(kernel<int64_t>, kThreadsPerBlock),
    };
    const auto max_blocks = occupancy[is_int32 ? 0 : 1] * num_sm;
    const auto needed_blocks = div_ceil(num_heads * num_tokens, kWarpsPerBlock);
    const auto num_blocks = std::max(1u, std::min(max_blocks, needed_blocks));
    LaunchKernel(num_blocks, kThreadsPerBlock, device.unwrap()).enable_pdl(kUsePDL)(selected_kernel, params);
  }
};

}  // namespace qknorm_rope_pack_qkv

}  // namespace sglang
