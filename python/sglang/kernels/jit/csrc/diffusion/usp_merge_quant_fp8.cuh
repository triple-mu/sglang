// Ulysses output merge fused with per-token FP8 quantization, both forms:
// - two-source (the 2-rank IPC output form): two [T, C] bf16 row-strided
//   sources emit one [T, 2C] fp8_e4m3 payload row plus a [T, 1] fp32 scale,
//   bitwise equal to sgl_per_token_quant_fp8(torch.cat((first, second), 1));
// - head-merge (the NCCL all-to-all form): the [W, S, B, H, D] bf16 tensor
//   usp_merge_heads consumes emits [B*S, W*H*D] fp8 rows (row b*S + s) plus
//   scales, bitwise equal to quantizing the merged bf16 rows.
//
// The reference (kernels/jit/csrc/gemm/per_token_quant_fp8.cuh) is a
// --use_fast_math build on SM90, so this file replicates its machine
// arithmetic explicitly instead of relying on compile flags (the same replica
// documented in ops/diffusion/common/fp8_quant_replica.py):
//   amax:      FMNMX.FTZ over |fp32(x)|      -> ftz_f32(fabsf(x)) + fmaxf tree
//   scale:     FMUL.FTZ(amax, rn(1/448))     -> ftz_f32(amax * kRecipFp8Max)
//   scale_inv: MUFU.RCP(scale)               -> rcp.approx.f32 inline asm,
//              zeroed when scale == 0 only under the warp-dispatch guard
//   payload:   FMUL.FTZ(fp32(x), scale_inv)  -> ftz_f32(x) * scale_inv; the
//              missing output flush is unobservable after the E4M3 convert
//   clamp+cvt: FMNMX to [-448, 448] then F2FP.SATFINITE.E4M3
// fmaxf/fminf are IEEE minNum/maxNum, so every reduction tree over the
// non-negative amax terms is bitwise order-invariant.

#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/cta.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <dlpack/dlpack.h>

#include <cstdint>

namespace sglang {

namespace usp_merge_quant_fp8 {

// Elements per 128-bit load; one chunk is also one 64-bit fp8 store.
constexpr uint32_t kVecSize = 8;
// rn(1/448) = 0x3B124925, the compile-time constant the fast-math reference
// multiplies by instead of dividing by 448.
constexpr float kRecipFp8Max = 0.0022321429569274187f;
constexpr float kFp8Max = 448.0f;

namespace {

SGL_DEVICE float ftz_f32(float x) {
  const uint32_t bits = __float_as_uint(x);
  return (bits & 0x7f800000u) == 0u ? __uint_as_float(bits & 0x80000000u) : x;
}

SGL_DEVICE float rcp_approx_f32(float x) {
  float y;
  asm("rcp.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

/**
 * \brief One CTA per token: gather both source rows once, quantize from the
 *        retained registers (single DRAM pass instead of the two-pass read).
 *
 * \tparam kBlockSize       Threads per CTA
 * \tparam kChunksPerThread Retained kVecSize-element chunks per thread; must
 *                          cover 2 * C / kVecSize chunks with kBlockSize
 * \param first        First source, row t at first + t * first_stride
 * \param second       Second source, row t at second + t * second_stride
 * \param q            [T, 2C] fp8 payload, first fills columns [0, C)
 * \param s            [T, 1] fp32 per-token scale
 * \param half_chunks  C / kVecSize
 * \param zero_guard   Whether the reference dispatch zeroes 1/scale at
 *                     scale == 0 (its warp-per-token regime)
 */
template <int kBlockSize, int kChunksPerThread, bool kUsePDL>
__global__ __launch_bounds__(kBlockSize) void merge_two_sources_quant_kernel(
    const bf16_t* __restrict__ first,
    const bf16_t* __restrict__ second,
    fp8_e4m3_t* __restrict__ q,
    float* __restrict__ s,
    int64_t first_stride,
    int64_t second_stride,
    uint32_t half_chunks,
    bool zero_guard) {
  using namespace device;
  using load_vec_t = AlignedVector<bf16_t, kVecSize>;
  using store_vec_t = AlignedVector<fp8_e4m3_t, kVecSize>;

  const int64_t token = blockIdx.x;
  const bf16_t* first_row = first + token * first_stride;
  const bf16_t* second_row = second + token * second_stride;
  const uint32_t num_chunks = 2 * half_chunks;

  PDLWaitPrimary<kUsePDL>();

  // Loads first, amax after: with the reduction loop split out, every LDG is
  // in flight before the first register use instead of one chunk at a time.
  load_vec_t payload[kChunksPerThread];
#pragma unroll
  for (int i = 0; i < kChunksPerThread; ++i) {
    const uint32_t chunk = threadIdx.x + i * kBlockSize;
    if (chunk < num_chunks) {
      const bool in_second = chunk >= half_chunks;
      const bf16_t* row = in_second ? second_row : first_row;
      const uint32_t vec_index = in_second ? chunk - half_chunks : chunk;
      payload[i].load(row, vec_index);
    }
  }
  // amax in the bf16 domain (HMNMX2, one instruction per pair) instead of the
  // reference's per-element FMNMX.FTZ over fp32: bf16 maxNum picks the same
  // element (a denormal never outranks a normal), the bf16->fp32 convert is
  // exact, and an all-denormal amax still flushes to the reference's zero
  // scale because ftz_f32(denormal * kRecipFp8Max) below is zero either way.
  bf16x2_t acc2;
  acc2.x = bf16_t(0.0f);
  acc2.y = bf16_t(0.0f);
#pragma unroll
  for (int i = 0; i < kChunksPerThread; ++i) {
    if (threadIdx.x + i * kBlockSize < num_chunks) {
      const auto* pairs = reinterpret_cast<const bf16x2_t*>(payload[i].data());
#pragma unroll
      for (int j = 0; j < static_cast<int>(kVecSize) / 2; ++j) {
        acc2 = __hmax2(acc2, __habs2(pairs[j]));
      }
    }
  }
  const float thread_amax = fmaxf(static_cast<float>(acc2.x), static_cast<float>(acc2.y));

  __shared__ float reduction_smem[kBlockSize / kWarpThreads];
  __shared__ float scale_inv_smem;
  cta::reduce_max(thread_amax, reduction_smem);
  if (threadIdx.x == 0) {
    const float scale = ftz_f32(reduction_smem[0] * kRecipFp8Max);
    s[token] = scale;
    float scale_inv = rcp_approx_f32(scale);
    if (zero_guard && scale == 0.0f) {
      scale_inv = 0.0f;
    }
    scale_inv_smem = scale_inv;
  }
  __syncthreads();
  const float scale_inv = scale_inv_smem;

  fp8_e4m3_t* out_row = q + token * (2 * static_cast<int64_t>(half_chunks) * kVecSize);
#pragma unroll
  for (int i = 0; i < kChunksPerThread; ++i) {
    const uint32_t chunk = threadIdx.x + i * kBlockSize;
    if (chunk < num_chunks) {
      store_vec_t out_vec;
      auto* out_pairs = reinterpret_cast<__nv_fp8x2_storage_t*>(out_vec.data());
#pragma unroll
      for (int j = 0; j < static_cast<int>(kVecSize) / 2; ++j) {
        // Replica of FMUL.FTZ + FMNMX clamp + F2FP.SATFINITE.E4M3; the pair
        // convert emits one F2FP for two elements with per-lane rn rounding.
        const float lo = ftz_f32(static_cast<float>(payload[i][2 * j])) * scale_inv;
        const float hi = ftz_f32(static_cast<float>(payload[i][2 * j + 1])) * scale_inv;
        const float2 clamped = {fmaxf(fminf(lo, kFp8Max), -kFp8Max), fmaxf(fminf(hi, kFp8Max), -kFp8Max)};
        out_pairs[j] = __nv_cvt_float2_to_fp8x2(clamped, __NV_SATFINITE, __NV_E4M3);
      }
      out_vec.store(out_row, chunk);
    }
  }

  PDLTriggerSecondary<kUsePDL>();
}

/**
 * \brief One CTA per token: gather the token's W head shards once, quantize
 *        from the retained registers (same replica arithmetic as the
 *        two-source kernel above).
 *
 * \tparam kBlockSize       Threads per CTA
 * \tparam kChunksPerThread Retained kVecSize-element chunks per thread; must
 *                          cover W * C / kVecSize chunks with kBlockSize
 * \param x            [W, S, B, H, D] contiguous input; C = H * D
 * \param q            [B * S, W * C] fp8 payload (row b * S + s)
 * \param s            [B * S, 1] fp32 per-token scale
 * \param seq          S
 * \param batch        B
 * \param shard_stride S * B * C, elements between one token's shards
 * \param half_chunks  C / kVecSize, chunks contributed by one shard
 * \param num_chunks   W * C / kVecSize, chunks in one merged row
 * \param zero_guard   Whether the reference dispatch zeroes 1/scale at
 *                     scale == 0 (its warp-per-token regime)
 */
template <int kBlockSize, int kChunksPerThread, bool kUsePDL>
__global__ __launch_bounds__(kBlockSize) void merge_heads_quant_kernel(
    const bf16_t* __restrict__ x,
    fp8_e4m3_t* __restrict__ q,
    float* __restrict__ s,
    uint32_t seq,
    uint32_t batch,
    int64_t shard_stride,
    uint32_t half_chunks,
    uint32_t num_chunks,
    bool zero_guard) {
  using namespace device;
  using load_vec_t = AlignedVector<bf16_t, kVecSize>;
  using store_vec_t = AlignedVector<fp8_e4m3_t, kVecSize>;

  const uint32_t token = blockIdx.x;
  const uint32_t seq_idx = token % seq;
  const uint32_t batch_idx = token / seq;
  const int64_t inner = static_cast<int64_t>(half_chunks) * kVecSize;
  const bf16_t* base = x + (static_cast<int64_t>(seq_idx) * batch + batch_idx) * inner;

  PDLWaitPrimary<kUsePDL>();

  // Loads first, amax after (see the two-source kernel).  Merged-row chunk
  // order equals output-column order, so the store side needs no shard math.
  load_vec_t payload[kChunksPerThread];
#pragma unroll
  for (int i = 0; i < kChunksPerThread; ++i) {
    const uint32_t chunk = threadIdx.x + i * kBlockSize;
    if (chunk < num_chunks) {
      const uint32_t shard = chunk / half_chunks;
      const uint32_t vec_index = chunk - shard * half_chunks;
      payload[i].load(base + static_cast<int64_t>(shard) * shard_stride, vec_index);
    }
  }
  bf16x2_t acc2;
  acc2.x = bf16_t(0.0f);
  acc2.y = bf16_t(0.0f);
#pragma unroll
  for (int i = 0; i < kChunksPerThread; ++i) {
    if (threadIdx.x + i * kBlockSize < num_chunks) {
      const auto* pairs = reinterpret_cast<const bf16x2_t*>(payload[i].data());
#pragma unroll
      for (int j = 0; j < static_cast<int>(kVecSize) / 2; ++j) {
        acc2 = __hmax2(acc2, __habs2(pairs[j]));
      }
    }
  }
  const float thread_amax = fmaxf(static_cast<float>(acc2.x), static_cast<float>(acc2.y));

  __shared__ float reduction_smem[kBlockSize / kWarpThreads];
  __shared__ float scale_inv_smem;
  cta::reduce_max(thread_amax, reduction_smem);
  if (threadIdx.x == 0) {
    const float scale = ftz_f32(reduction_smem[0] * kRecipFp8Max);
    s[token] = scale;
    float scale_inv = rcp_approx_f32(scale);
    if (zero_guard && scale == 0.0f) {
      scale_inv = 0.0f;
    }
    scale_inv_smem = scale_inv;
  }
  __syncthreads();
  const float scale_inv = scale_inv_smem;

  fp8_e4m3_t* out_row = q + static_cast<int64_t>(token) * num_chunks * kVecSize;
#pragma unroll
  for (int i = 0; i < kChunksPerThread; ++i) {
    const uint32_t chunk = threadIdx.x + i * kBlockSize;
    if (chunk < num_chunks) {
      store_vec_t out_vec;
      auto* out_pairs = reinterpret_cast<__nv_fp8x2_storage_t*>(out_vec.data());
#pragma unroll
      for (int j = 0; j < static_cast<int>(kVecSize) / 2; ++j) {
        const float lo = ftz_f32(static_cast<float>(payload[i][2 * j])) * scale_inv;
        const float hi = ftz_f32(static_cast<float>(payload[i][2 * j + 1])) * scale_inv;
        const float2 clamped = {fmaxf(fminf(lo, kFp8Max), -kFp8Max), fmaxf(fminf(hi, kFp8Max), -kFp8Max)};
        out_pairs[j] = __nv_cvt_float2_to_fp8x2(clamped, __NV_SATFINITE, __NV_E4M3);
      }
      out_vec.store(out_row, chunk);
    }
  }

  PDLTriggerSecondary<kUsePDL>();
}

}  // namespace

/** \brief Validate the tensors and launch the fused two-source merge + quant. */
template <int kBlockSize, int kChunksPerThread, bool kUsePDL>
struct MergeTwoSourcesQuantKernel {
  static_assert(kBlockSize % device::kWarpThreads == 0 && kBlockSize <= 1024);
  static_assert(kChunksPerThread >= 1);

  static void
  run(tvm::ffi::TensorView q,
      tvm::ffi::TensorView s,
      tvm::ffi::TensorView first,
      tvm::ffi::TensorView second,
      bool zero_guard) {
    using namespace host;

    auto T = SymbolicSize{"tokens"};
    auto C = SymbolicSize{"inner"};
    auto S_first = SymbolicSize{"first_row_stride"};
    auto S_second = SymbolicSize{"second_row_stride"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({T, C}).with_strides({S_first, 1}).with_dtype<bf16_t>().with_device<kDLCUDA>(device).verify(first);
    TensorMatcher({T, C}).with_strides({S_second, 1}).with_dtype<bf16_t>().with_device<kDLCUDA>(device).verify(second);

    const int64_t tokens = T.unwrap();
    const int64_t inner = C.unwrap();
    CHECK_HOST(tokens > 0 && inner > 0) << "merge_two_sources_quant_fp8: empty input";
    CHECK_HOST(inner % kVecSize == 0) << "merge_two_sources_quant_fp8: inner dim must be divisible by " << kVecSize
                                      << ", got " << inner;

    TensorMatcher({T, inner * 2}).with_dtype<fp8_e4m3_t>().with_device<kDLCUDA>(device).verify(q);
    TensorMatcher({T, 1}).with_dtype<float>().with_device<kDLCUDA>(device).verify(s);

    const int64_t half_chunks = inner / kVecSize;
    CHECK_HOST(2 * half_chunks <= int64_t{kBlockSize} * kChunksPerThread)
        << "merge_two_sources_quant_fp8: inner dim " << inner << " exceeds the " << kBlockSize << "x"
        << kChunksPerThread << " specialization";
    CHECK_HOST(tokens <= INT32_MAX) << "merge_two_sources_quant_fp8: tokens exceed the grid limit";

    const int64_t first_stride = S_first.unwrap();
    const int64_t second_stride = S_second.unwrap();
    const auto* first_ptr = static_cast<const bf16_t*>(first.data_ptr());
    const auto* second_ptr = static_cast<const bf16_t*>(second.data_ptr());
    constexpr uintptr_t kAlignment = sizeof(bf16_t) * kVecSize;
    CHECK_HOST(
        reinterpret_cast<uintptr_t>(first_ptr) % kAlignment == 0 &&
        reinterpret_cast<uintptr_t>(second_ptr) % kAlignment == 0 &&
        (first_stride * sizeof(bf16_t)) % kAlignment == 0 && (second_stride * sizeof(bf16_t)) % kAlignment == 0)
        << "merge_two_sources_quant_fp8: sources must be 16-byte aligned";

    LaunchKernel(static_cast<uint32_t>(tokens), kBlockSize, device.unwrap())
        .enable_pdl(kUsePDL)(
            merge_two_sources_quant_kernel<kBlockSize, kChunksPerThread, kUsePDL>,
            first_ptr,
            second_ptr,
            static_cast<fp8_e4m3_t*>(q.data_ptr()),
            static_cast<float*>(s.data_ptr()),
            first_stride,
            second_stride,
            static_cast<uint32_t>(half_chunks),
            zero_guard);
  }
};

/** \brief Validate the tensors and launch the fused head-merge + quant. */
template <int kBlockSize, int kChunksPerThread, bool kUsePDL>
struct MergeHeadsQuantKernel {
  static_assert(kBlockSize % device::kWarpThreads == 0 && kBlockSize <= 1024);
  static_assert(kChunksPerThread >= 1);

  static void run(tvm::ffi::TensorView q, tvm::ffi::TensorView s, tvm::ffi::TensorView x, bool zero_guard) {
    using namespace host;

    auto W = SymbolicSize{"world_size"};
    auto S = SymbolicSize{"sequence_length"};
    auto B = SymbolicSize{"batch"};
    auto H = SymbolicSize{"local_heads"};
    auto D = SymbolicSize{"head_dim"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({W, S, B, H, D}).with_dtype<bf16_t>().with_device<kDLCUDA>(device).verify(x);

    const int64_t world = W.unwrap();
    const int64_t seq = S.unwrap();
    const int64_t batch = B.unwrap();
    const int64_t inner = H.unwrap() * D.unwrap();
    const int64_t tokens = batch * seq;
    CHECK_HOST(tokens > 0 && world > 0 && inner > 0) << "merge_heads_quant_fp8: empty input";
    CHECK_HOST(inner % kVecSize == 0) << "merge_heads_quant_fp8: inner dim must be divisible by " << kVecSize
                                      << ", got " << inner;

    TensorMatcher({tokens, world * inner}).with_dtype<fp8_e4m3_t>().with_device<kDLCUDA>(device).verify(q);
    TensorMatcher({tokens, 1}).with_dtype<float>().with_device<kDLCUDA>(device).verify(s);

    const int64_t num_chunks = world * inner / kVecSize;
    CHECK_HOST(num_chunks <= int64_t{kBlockSize} * kChunksPerThread)
        << "merge_heads_quant_fp8: merged row of " << world * inner << " elements exceeds the " << kBlockSize << "x"
        << kChunksPerThread << " specialization";
    CHECK_HOST(tokens <= INT32_MAX) << "merge_heads_quant_fp8: tokens exceed the grid limit";

    const auto* x_ptr = static_cast<const bf16_t*>(x.data_ptr());
    constexpr uintptr_t kAlignment = sizeof(bf16_t) * kVecSize;
    // inner % kVecSize == 0 makes every shard offset a kAlignment multiple.
    CHECK_HOST(reinterpret_cast<uintptr_t>(x_ptr) % kAlignment == 0)
        << "merge_heads_quant_fp8: input must be 16-byte aligned";

    LaunchKernel(static_cast<uint32_t>(tokens), kBlockSize, device.unwrap())
        .enable_pdl(kUsePDL)(
            merge_heads_quant_kernel<kBlockSize, kChunksPerThread, kUsePDL>,
            x_ptr,
            static_cast<fp8_e4m3_t*>(q.data_ptr()),
            static_cast<float*>(s.data_ptr()),
            static_cast<uint32_t>(seq),
            static_cast<uint32_t>(batch),
            seq * batch * inner,
            static_cast<uint32_t>(inner / kVecSize),
            static_cast<uint32_t>(num_chunks),
            zero_guard);
  }
};

}  // namespace usp_merge_quant_fp8

}  // namespace sglang
