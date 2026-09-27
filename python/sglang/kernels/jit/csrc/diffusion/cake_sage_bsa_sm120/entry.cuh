#pragma once

// The vendored Cake kernel declares its typedefs, `CakeTensorMap` and the
// `extern "C" __global__` entry at global scope, so it is included before
// `namespace sglang` is opened. Nothing in vendor/ is edited; see README.md.
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>
#include <tvm/ffi/container/tensor.h>

#include "vendor/cake_sage_block_sparse_attention_939d22c4b83f8f4c938f_kernel.cu"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <mutex>
#include <unordered_map>

namespace sglang {

/**
 * \brief Host entry for the vendored Cake SM120 Sage block-sparse attention kernel.
 *
 * The kernel is one generated specialization of flashinfer PR #4951:
 * per-query-row block counts (`HAS_BLOCK_NUMS`), no per-block sizes
 * (`BLOCK_SIZES_MODE 0`, so `seqlen_k` must be a multiple of 64), unordered and
 * possibly empty index rows. Q/K are INT8 BHSD with per-32-token-group and
 * per-64-token-block scales, V is FP8 E4M3 in `[B, H, D, S_k]` with the Sage
 * 16-token permutation baked in, and the output is BF16 BHSD. MHA only, head
 * dimension 128, non-causal, no LSE.
 *
 * Q/K/V/O go to the kernel as TMA descriptors encoded over the *allocated*
 * extents of the tensors; `seqlen_q` / `seqlen_k` bound the tiles the kernel
 * touches, so a caller may hand in a larger resident buffer and only pay for
 * the live rows. Scales and index tables are sized by the live lengths.
 */
namespace cake_sage_bsa_sm120 {

static_assert(
    HAS_BLOCK_NUMS == 1 && BLOCK_SIZES_MODE == 0 && FULL_K64_TILES == 1 && UNIFORM_NONEMPTY == 0 &&
        CONTIGUOUS_BLOCK_INDICES == 0,
    "vendored Cake specialization changed; the launcher below assumes "
    "HAS_BLOCK_NUMS=1 BLOCK_SIZES_MODE=0 FULL_K64_TILES=1 UNIFORM_NONEMPTY=0 CONTIGUOUS_BLOCK_INDICES=0");

constexpr uint32_t kThreads = THREADS;
constexpr uint32_t kDynamicSmemBytes = SMEM_TOTAL;
constexpr int64_t kHeadDim = 128;
constexpr int64_t kBlock = 64;
/// A 128-query TMA tile carries four 32-query quantization groups.
constexpr int64_t kQScaleGroupsPerTile = 4;
constexpr size_t kNumDescriptors = 4;
constexpr size_t kDescriptorWorkspaceBytes = kNumDescriptors * sizeof(CUtensorMap);
static_assert(kDescriptorWorkspaceBytes == 512, "four 128-byte TMA descriptors");

namespace details {

/// The driver entry point, fetched through the runtime so the module links without -lcuda.
inline auto encode_tiled_fn() -> PFN_cuTensorMapEncodeTiled_v12000 {
  static const auto fn = [] {
    void* sym = nullptr;
    cudaDriverEntryPointQueryResult status;
    CHECK_CUDA(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &sym, 12000, cudaEnableDefault, &status));
    CHECK_HOST(status == cudaDriverEntryPointSuccess && sym != nullptr) << "cuTensorMapEncodeTiled unavailable";
    return reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(sym);
  }();
  return fn;
}

inline auto encode(
    CUtensorMapDataType dtype,
    uint32_t rank,
    const void* base,
    const cuuint64_t* dims,
    const cuuint64_t* strides_bytes,
    const cuuint32_t* box,
    CUtensorMapSwizzle swizzle,
    const char* what) -> CUtensorMap {
  const cuuint32_t elem_strides[5] = {1, 1, 1, 1, 1};
  CUtensorMap map{};
  const CUresult res = encode_tiled_fn()(
      &map,
      dtype,
      rank,
      const_cast<void*>(base),
      dims,
      strides_bytes,
      box,
      elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE,
      swizzle,
      CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  CHECK_HOST(res == CUDA_SUCCESS) << "cuTensorMapEncodeTiled(" << what << ") failed: CUresult=" << static_cast<int>(res);
  return map;
}

/// INT8 `[B, H, S, 128]`: box of 64 rows x 128 bytes, 128B swizzle (the kernel's Q/K smem layout).
inline auto encode_qk(const void* base, int64_t batch, int64_t heads, int64_t seq, const char* what) -> CUtensorMap {
  const cuuint64_t dims[4] = {
      static_cast<cuuint64_t>(kHeadDim),
      static_cast<cuuint64_t>(seq),
      static_cast<cuuint64_t>(heads),
      static_cast<cuuint64_t>(batch)};
  const cuuint64_t strides[3] = {
      static_cast<cuuint64_t>(kHeadDim),
      static_cast<cuuint64_t>(kHeadDim * seq),
      static_cast<cuuint64_t>(kHeadDim * seq * heads)};
  const cuuint32_t box[4] = {128, 64, 1, 1};
  return encode(CU_TENSOR_MAP_DATA_TYPE_UINT8, 4, base, dims, strides, box, CU_TENSOR_MAP_SWIZZLE_128B, what);
}

/// FP8 `[B, H, 128, S]`: box of 128 channels x 64 tokens, 64B swizzle.
inline auto encode_v(const void* base, int64_t batch, int64_t heads, int64_t seq) -> CUtensorMap {
  const cuuint64_t dims[4] = {
      static_cast<cuuint64_t>(seq),
      static_cast<cuuint64_t>(kHeadDim),
      static_cast<cuuint64_t>(heads),
      static_cast<cuuint64_t>(batch)};
  const cuuint64_t strides[3] = {
      static_cast<cuuint64_t>(seq),
      static_cast<cuuint64_t>(seq * kHeadDim),
      static_cast<cuuint64_t>(seq * kHeadDim * heads)};
  const cuuint32_t box[4] = {64, 128, 1, 1};
  return encode(CU_TENSOR_MAP_DATA_TYPE_UINT8, 4, base, dims, strides, box, CU_TENSOR_MAP_SWIZZLE_64B, "V");
}

/// BF16 `[B, H, S, 128]` written as two 64-wide halves: 5D map `{64, S, H, B, 2}`, 128B swizzle.
inline auto encode_o(const void* base, int64_t batch, int64_t heads, int64_t seq) -> CUtensorMap {
  constexpr int64_t kElem = sizeof(bf16_t);
  const cuuint64_t dims[5] = {
      64,
      static_cast<cuuint64_t>(seq),
      static_cast<cuuint64_t>(heads),
      static_cast<cuuint64_t>(batch),
      2};
  const cuuint64_t strides[4] = {
      static_cast<cuuint64_t>(kHeadDim * kElem),
      static_cast<cuuint64_t>(kHeadDim * kElem * seq),
      static_cast<cuuint64_t>(kHeadDim * kElem * seq * heads),
      static_cast<cuuint64_t>(64 * kElem)};
  const cuuint32_t box[5] = {64, 64, 1, 1, 2};
  return encode(CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 5, base, dims, strides, box, CU_TENSOR_MAP_SWIZZLE_128B, "O");
}

using DescriptorSet = std::array<CUtensorMap, kNumDescriptors>;

/**
 * \brief Publish the four descriptors into the caller-owned device workspace.
 *
 * The kernel reads its TMA descriptors from global memory (with a tensormap
 * proxy fence), so they live in a 512-byte device buffer the Python side keeps
 * per device. The last bytes uploaded to each workspace are remembered; a
 * repeat call with the same tensors is a pure cache hit, and a change is one
 * stream-ordered 512-byte H2D copy. That copy cannot be recorded into a CUDA
 * graph, so a capture must warm the exact tensors first.
 */
inline auto bind_descriptors(const DescriptorSet& maps, void* workspace, cudaStream_t stream) -> void {
  static std::mutex mu;
  static auto* uploaded = new std::unordered_map<uintptr_t, DescriptorSet>();
  const auto key = reinterpret_cast<uintptr_t>(workspace);
  std::lock_guard<std::mutex> lock(mu);
  if (auto it = uploaded->find(key); it != uploaded->end() && std::memcmp(&it->second, &maps, sizeof(maps)) == 0) {
    return;
  }
  cudaStreamCaptureStatus capturing = cudaStreamCaptureStatusNone;
  CHECK_CUDA(cudaStreamIsCapturing(stream, &capturing));
  CHECK_HOST(capturing == cudaStreamCaptureStatusNone)
      << "cake_sage_bsa_sm120: TMA descriptors changed during CUDA graph capture; "
         "warm the exact tensors before capturing";
  // Pageable source: the runtime stages the bytes before returning, so `maps` may die.
  CHECK_CUDA(cudaMemcpyAsync(workspace, maps.data(), sizeof(maps), cudaMemcpyHostToDevice, stream));
  (*uploaded)[key] = maps;
}

inline auto fits_int32(int64_t value) -> bool {
  return value >= 0 && value <= static_cast<int64_t>(std::numeric_limits<int32_t>::max());
}

}  // namespace details

/**
 * \brief Validate the operands, publish the TMA descriptors and launch the kernel.
 *
 * \param q            INT8 `[B, H, SQ_ALLOC, 128]`, contiguous
 * \param k            INT8 `[B, H, SK_ALLOC, 128]`, contiguous
 * \param v            FP8 E4M3 `[B, H, 128, SK_ALLOC]`, Sage 16-token permutation, contiguous
 * \param out          BF16 `[B, H, SQ_ALLOC, 128]`, contiguous; rows `>= seqlen_q` are left alone
 * \param q_scale      FP32 `[B, H, ceil(seqlen_q / 128) * 4]`, one per 32-query group
 * \param k_scale      FP32 `[B, H, seqlen_k / 64]`, one per 64-key block
 * \param v_scale      FP32 `[B, H, 128]`, one per channel
 * \param q2k_block_index INT32 `[B, H, ceil(seqlen_q / 64), capacity]`, key blocks per query block, any order
 * \param q2k_block_nums  INT32 `[B, H, ceil(seqlen_q / 64)]`, live entries per row, in `[0, capacity]`
 * \param workspace    uint8 `[>= 512]`, 128-byte aligned, holds the four TMA descriptors
 * \param seqlen_q     live query rows, `1 <= seqlen_q <= SQ_ALLOC`
 * \param seqlen_k     live key rows, a multiple of 64, `<= SK_ALLOC`
 * \param softmax_scale finite, positive
 */
inline auto run(
    tvm::ffi::TensorView q,
    tvm::ffi::TensorView k,
    tvm::ffi::TensorView v,
    tvm::ffi::TensorView out,
    tvm::ffi::TensorView q_scale,
    tvm::ffi::TensorView k_scale,
    tvm::ffi::TensorView v_scale,
    tvm::ffi::TensorView q2k_block_index,
    tvm::ffi::TensorView q2k_block_nums,
    tvm::ffi::TensorView workspace,
    int64_t seqlen_q,
    int64_t seqlen_k,
    double softmax_scale) -> void {
  using namespace host;

  CHECK_HOST(seqlen_q >= 1) << "seqlen_q must be positive, got " << seqlen_q;
  CHECK_HOST(seqlen_k >= kBlock && seqlen_k % kBlock == 0)
      << "this Cake specialization needs seqlen_k to be a positive multiple of " << kBlock << ", got " << seqlen_k;
  CHECK_HOST(std::isfinite(softmax_scale) && softmax_scale > 0.0) << "softmax_scale must be finite and positive";

  const int64_t q_blocks = div_ceil(seqlen_q, kBlock);
  const int64_t k_blocks = seqlen_k / kBlock;
  const int64_t q_scale_len = div_ceil(seqlen_q, int64_t{128}) * kQScaleGroupsPerTile;

  SymbolicSize B{"batch"}, H{"heads"}, SQA{"seqlen_q_alloc"}, SKA{"seqlen_k_alloc"}, CAP{"q2k_capacity"},
      WS{"workspace_bytes"};
  SymbolicDevice device;
  device.set_options<kDLCUDA>();

  TensorMatcher({B, H, SQA, kHeadDim}).with_dtype<int8_t>().with_device<kDLCUDA>(device).verify(q);
  TensorMatcher({B, H, SKA, kHeadDim}).with_dtype<int8_t>().with_device<kDLCUDA>(device).verify(k);
  TensorMatcher({B, H, kHeadDim, SKA}).with_dtype<fp8_e4m3_t>().with_device<kDLCUDA>(device).verify(v);
  TensorMatcher({B, H, SQA, kHeadDim}).with_dtype<bf16_t>().with_device<kDLCUDA>(device).verify(out);
  TensorMatcher({B, H, q_scale_len}).with_dtype<fp32_t>().with_device<kDLCUDA>(device).verify(q_scale);
  TensorMatcher({B, H, k_blocks}).with_dtype<fp32_t>().with_device<kDLCUDA>(device).verify(k_scale);
  TensorMatcher({B, H, kHeadDim}).with_dtype<fp32_t>().with_device<kDLCUDA>(device).verify(v_scale);
  TensorMatcher({B, H, q_blocks, CAP}).with_dtype<int32_t>().with_device<kDLCUDA>(device).verify(q2k_block_index);
  TensorMatcher({B, H, q_blocks}).with_dtype<int32_t>().with_device<kDLCUDA>(device).verify(q2k_block_nums);
  TensorMatcher({WS}).with_dtype<uint8_t>().with_device<kDLCUDA>(device).ensure_alignment(128).verify(workspace);

  const int64_t batch = B.unwrap();
  const int64_t heads = H.unwrap();
  const int64_t seqlen_q_alloc = SQA.unwrap();
  const int64_t seqlen_k_alloc = SKA.unwrap();
  const int64_t capacity = CAP.unwrap();
  CHECK_HOST(seqlen_q <= seqlen_q_alloc) << "seqlen_q " << seqlen_q << " exceeds the allocated " << seqlen_q_alloc;
  CHECK_HOST(seqlen_k <= seqlen_k_alloc) << "seqlen_k " << seqlen_k << " exceeds the allocated " << seqlen_k_alloc;
  CHECK_HOST(WS.unwrap() >= static_cast<int64_t>(kDescriptorWorkspaceBytes))
      << "workspace needs at least " << kDescriptorWorkspaceBytes << " bytes";
  for (const int64_t value : {seqlen_q, seqlen_k, heads, capacity, k_blocks, batch * heads * q_blocks}) {
    CHECK_HOST(details::fits_int32(value)) << "value " << value << " does not fit the kernel's int32 arguments";
  }

  const DLDevice dev = device.unwrap();
  auto* stream = static_cast<cudaStream_t>(::TVMFFIEnvGetStream(dev.device_type, dev.device_id));
  details::DescriptorSet maps{
      details::encode_qk(q.data_ptr(), batch, heads, seqlen_q_alloc, "Q"),
      details::encode_qk(k.data_ptr(), batch, heads, seqlen_k_alloc, "K"),
      details::encode_v(v.data_ptr(), batch, heads, seqlen_k_alloc),
      details::encode_o(out.data_ptr(), batch, heads, seqlen_q_alloc)};
  details::bind_descriptors(maps, workspace.data_ptr(), stream);
  auto* slots = static_cast<const CakeTensorMap*>(workspace.data_ptr());

  // `block_sizes` is unread under BLOCK_SIZES_MODE 0 but must be a valid pointer, and
  // `block_sparse_num` is replaced per row by `q2k_block_nums` under HAS_BLOCK_NUMS 1;
  // both take the conservative values the generated binding passes.
  auto* index = static_cast<int*>(q2k_block_index.data_ptr());
  LaunchKernel(dim3(static_cast<uint32_t>(q_blocks), static_cast<uint32_t>(heads), static_cast<uint32_t>(batch)),
               kThreads, dev, kDynamicSmemBytes)(
      kernel_cake_sage_block_sparse_attention_939d22c4b83f8f4c938f,
      slots + 0,
      slots + 1,
      slots + 2,
      slots + 3,
      static_cast<float*>(q_scale.data_ptr()),
      static_cast<float*>(k_scale.data_ptr()),
      static_cast<float*>(v_scale.data_ptr()),
      index,
      static_cast<int*>(q2k_block_nums.data_ptr()),
      index,
      static_cast<int>(seqlen_q),
      static_cast<int>(seqlen_k),
      static_cast<int>(heads),
      static_cast<int>(capacity),
      static_cast<int>(k_blocks),
      static_cast<int>(capacity),
      static_cast<float>(softmax_scale));
}

}  // namespace cake_sage_bsa_sm120
}  // namespace sglang
