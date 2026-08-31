// Channels-last two-pass GroupNorm(+SiLU) over (N, R, C) rows.
//
// C++ port of ops/diffusion/norm/group_norm_silu_twopass_triton.py with the
// same contract: fp32 statistics via per-chunk partial sum/sumsq buffers, the
// affine transform folded into per-(batch, channel) scale/shift, and a pure
// elementwise apply pass with an optional SiLU epilogue.  Held to the eager
// F.group_norm(+F.silu) oracle with a per-dtype tolerance, and deterministic:
// every reduction is a fixed-order sequential sum (no atomics).
//
// C is a power of two <= 2048 divisible by num_groups and by the 16-byte
// vector width; the Python wrapper falls back to the Triton kernels
// otherwise.

#pragma once

#include <sgl_kernel/ffi.h>
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <algorithm>
#include <bit>
#include <cstdint>
#include <type_traits>

namespace sglang {

namespace group_norm_silu_twopass {

constexpr uint32_t kMaxBlockSize = 512;
constexpr uint32_t kFinalizeBlockSize = 256;
constexpr int64_t kMaxGridY = 65535;
// Rows per unrolled apply pass; the launcher sizes blocks to exactly one
// pass so every load in a block is in flight at once (Triton-tile geometry).
// Measured best at 4 for both fp32 (16 elems/thread) and 16-bit (32) on H200.
template <typename T>
constexpr int apply_row_unroll() {
  return 4;
}

namespace {

/**
 * \brief Per-chunk partial channel sums for one (batch, row-chunk) tile.
 *
 * Threads are laid out as (rows_par, cols_vec) with cols_vec = C / kVec, so
 * each thread accumulates kVec fixed channels over its strided rows; the
 * cross-row_lane combine is a fixed-order sequential sum in shared memory.
 *
 * \tparam T Element type: fp32_t | bf16_t | fp16_t.
 * \param x        (N, rows, C) contiguous input.
 * \param partial  (N, nchunks, 2, C) fp32 output: [0] sums, [1] square sums.
 * \param rows     Row count R.
 * \param channels C.
 * \param rows_per_chunk Rows covered by one block.
 * \param col_bits log2(C / kVec); blockDim.x == (C / kVec) << row_bits.
 */
template <typename T>
__launch_bounds__(kMaxBlockSize) __global__ void gn_partial_kernel(
    const T* __restrict__ x,
    float* __restrict__ partial,
    int64_t rows,
    uint32_t channels,
    uint32_t rows_per_chunk,
    uint32_t col_bits) {
  using namespace device;
  constexpr int kVec = 16 / sizeof(T);
  extern __shared__ float smem_red[];  // blockDim.x * 2 * kVec floats when rows_par > 1

  const uint32_t cols_vec = 1u << col_bits;
  const uint32_t col = threadIdx.x & (cols_vec - 1);
  const uint32_t row_lane = threadIdx.x >> col_bits;
  const uint32_t rows_par = blockDim.x >> col_bits;

  const uint32_t batch = blockIdx.y;
  const int64_t row_start = static_cast<int64_t>(blockIdx.x) * rows_per_chunk;
  const int64_t chunk_end = row_start + rows_per_chunk;
  const int64_t row_end = chunk_end < rows ? chunk_end : rows;
  const T* x_batch = x + static_cast<int64_t>(batch) * rows * channels;

  float acc_s[kVec] = {};
  float acc_q[kVec] = {};
  // 4-row unroll: one load in flight per thread starves DRAM at the ~50%
  // occupancy this kernel runs at; four independent loads per iteration
  // restore enough memory-level parallelism to reach the streaming roofline.
  constexpr int kRowUnroll = 4;
  const int64_t stride = rows_par;
  int64_t r = row_start + row_lane;
  for (; r + (kRowUnroll - 1) * stride < row_end; r += kRowUnroll * stride) {
    AlignedVector<T, kVec> v[kRowUnroll];
#pragma unroll
    for (int u = 0; u < kRowUnroll; ++u) {
      v[u].load(x_batch + (r + u * stride) * channels, col);
    }
#pragma unroll
    for (int u = 0; u < kRowUnroll; ++u) {
#pragma unroll
      for (int j = 0; j < kVec; ++j) {
        const float f = static_cast<float>(v[u][j]);
        acc_s[j] += f;
        acc_q[j] += f * f;
      }
    }
  }
  for (; r < row_end; r += stride) {
    AlignedVector<T, kVec> v;
    v.load(x_batch + r * channels, col);
#pragma unroll
    for (int j = 0; j < kVec; ++j) {
      const float f = static_cast<float>(v[j]);
      acc_s[j] += f;
      acc_q[j] += f * f;
    }
  }

  if (rows_par > 1) {
    float* mine = smem_red + static_cast<size_t>(threadIdx.x) * 2 * kVec;
#pragma unroll
    for (int j = 0; j < kVec; ++j) {
      mine[j] = acc_s[j];
      mine[kVec + j] = acc_q[j];
    }
    __syncthreads();
    if (row_lane != 0) {
      return;
    }
    for (uint32_t lane = 1; lane < rows_par; ++lane) {
      const float* other = smem_red + (static_cast<size_t>(lane << col_bits) + col) * 2 * kVec;
#pragma unroll
      for (int j = 0; j < kVec; ++j) {
        acc_s[j] += other[j];
        acc_q[j] += other[kVec + j];
      }
    }
  }

  // fp32 partials are stored in 16B pieces: kVec floats exceed the 16B
  // vector limit when T is 16-bit (kVec == 8).
  constexpr int kFloatVec = 4;
  constexpr int kFloatVecs = kVec / kFloatVec;
  float* out = partial + (static_cast<int64_t>(batch) * gridDim.x + blockIdx.x) * 2 * channels;
#pragma unroll
  for (int p = 0; p < kFloatVecs; ++p) {
    AlignedVector<float, kFloatVec> vs;
    AlignedVector<float, kFloatVec> vq;
#pragma unroll
    for (int j = 0; j < kFloatVec; ++j) {
      vs[j] = acc_s[p * kFloatVec + j];
      vq[j] = acc_q[p * kFloatVec + j];
    }
    vs.store(out, col * kFloatVecs + p);
    vq.store(out + channels, col * kFloatVecs + p);
  }
}

/**
 * \brief Reduce chunk partials into per-(batch, channel) scale/shift.
 *
 * \tparam T Affine parameter element type (matches the activation dtype).
 * \param partial (N, nchunks, 2, C) fp32 chunk partials.
 * \param weight  (C,) affine weight.
 * \param bias    (C,) affine bias.
 * \param ss      (N, 2, C) fp32 output: [0] scale = w * rstd, [1] shift = b - mean * scale.
 * \param nchunks Chunk count of the partial pass.
 * \param channels C.
 * \param cpg     Channels per group.
 * \param group_numel rows * cpg as float (the Triton kernel's divisor).
 * \param eps     Variance epsilon.
 */
template <typename T>
__launch_bounds__(kFinalizeBlockSize) __global__ void gn_finalize_kernel(
    const float* __restrict__ partial,
    const T* __restrict__ weight,
    const T* __restrict__ bias,
    float* __restrict__ ss,
    uint32_t nchunks,
    uint32_t channels,
    uint32_t cpg,
    float group_numel,
    float eps) {
  extern __shared__ float smem[];  // 2 * C channel totals + 2 * (C / cpg) group stats

  const uint32_t batch = blockIdx.x;
  const uint32_t groups = channels / cpg;
  const float* base = partial + static_cast<int64_t>(batch) * nchunks * 2 * channels;

  for (uint32_t c = threadIdx.x; c < channels; c += blockDim.x) {
    float sum = 0.0f;
    float sq = 0.0f;
    for (uint32_t k = 0; k < nchunks; ++k) {
      const float* chunk = base + static_cast<int64_t>(k) * 2 * channels;
      sum += chunk[c];
      sq += chunk[channels + c];
    }
    smem[c] = sum;
    smem[channels + c] = sq;
  }
  __syncthreads();

  float* group_mean = smem + 2 * channels;
  float* group_rstd = group_mean + groups;
  for (uint32_t g = threadIdx.x; g < groups; g += blockDim.x) {
    float sum = 0.0f;
    float sq = 0.0f;
    for (uint32_t i = 0; i < cpg; ++i) {
      sum += smem[g * cpg + i];
      sq += smem[channels + g * cpg + i];
    }
    const float mean = sum / group_numel;
    const float var = fmaxf(sq / group_numel - mean * mean, 0.0f);
    group_mean[g] = mean;
    group_rstd[g] = rsqrtf(var + eps);
  }
  __syncthreads();

  float* ss_batch = ss + static_cast<int64_t>(batch) * 2 * channels;
  for (uint32_t c = threadIdx.x; c < channels; c += blockDim.x) {
    const uint32_t g = c / cpg;
    const float scale = static_cast<float>(weight[c]) * group_rstd[g];
    ss_batch[c] = scale;
    ss_batch[channels + c] = static_cast<float>(bias[c]) - group_mean[g] * scale;
  }
}

/**
 * \brief Elementwise apply pass: y = x * scale + shift, optionally * sigmoid.
 *
 * Same (rows_par, cols_vec) thread layout as the partial pass; each thread
 * loads its kVec channels' scale/shift once and streams its rows.
 */
template <typename T>
__launch_bounds__(kMaxBlockSize) __global__ void gn_apply_kernel(
    const T* __restrict__ x,
    const float* __restrict__ ss,
    T* __restrict__ y,
    int64_t rows,
    uint32_t channels,
    uint32_t rows_per_block,
    uint32_t col_bits,
    bool apply_silu) {
  using namespace device;
  constexpr int kVec = 16 / sizeof(T);

  const uint32_t cols_vec = 1u << col_bits;
  const uint32_t col = threadIdx.x & (cols_vec - 1);
  const uint32_t row_lane = threadIdx.x >> col_bits;
  const uint32_t rows_par = blockDim.x >> col_bits;

  const uint32_t batch = blockIdx.y;
  const int64_t row_start = static_cast<int64_t>(blockIdx.x) * rows_per_block;
  const int64_t block_end = row_start + rows_per_block;
  const int64_t row_end = block_end < rows ? block_end : rows;

  const float* ss_batch = ss + static_cast<int64_t>(batch) * 2 * channels;
  constexpr int kFloatVec = 4;
  constexpr int kFloatVecs = kVec / kFloatVec;
  float scale[kVec];
  float shift[kVec];
#pragma unroll
  for (int p = 0; p < kFloatVecs; ++p) {
    AlignedVector<float, kFloatVec> vsc;
    AlignedVector<float, kFloatVec> vsh;
    vsc.load(ss_batch, col * kFloatVecs + p);
    vsh.load(ss_batch + channels, col * kFloatVecs + p);
#pragma unroll
    for (int j = 0; j < kFloatVec; ++j) {
      scale[p * kFloatVec + j] = vsc[j];
      shift[p * kFloatVec + j] = vsh[j];
    }
  }

  const int64_t batch_offset = static_cast<int64_t>(batch) * rows * channels;
  const T* x_batch = x + batch_offset;
  T* y_batch = y + batch_offset;
  constexpr int kRowUnroll = apply_row_unroll<T>();
  const int64_t stride = rows_par;
  const auto transform = [&](AlignedVector<T, kVec>& v) {
#pragma unroll
    for (int j = 0; j < kVec; ++j) {
      float f = fmaf(static_cast<float>(v[j]), scale[j], shift[j]);
      if (apply_silu) {
        // __expf + __fdividef (ex2.approx + rcp.approx, ~2^-22 rel error)
        // instead of precise expf/div.rn: the tolerance contract absorbs it
        // and the precise forms cost ~4x the instructions per element.
        f = __fdividef(f, 1.0f + __expf(-f));
      }
      v[j] = static_cast<T>(f);
    }
  };
  int64_t r = row_start + row_lane;
  for (; r + (kRowUnroll - 1) * stride < row_end; r += kRowUnroll * stride) {
    AlignedVector<T, kVec> v[kRowUnroll];
#pragma unroll
    for (int u = 0; u < kRowUnroll; ++u) {
      v[u].load(x_batch + (r + u * stride) * channels, col);
    }
#pragma unroll
    for (int u = 0; u < kRowUnroll; ++u) {
      transform(v[u]);
      v[u].store(y_batch + (r + u * stride) * channels, col);
    }
  }
  for (; r < row_end; r += stride) {
    AlignedVector<T, kVec> v;
    v.load(x_batch + r * channels, col);
    transform(v);
    v.store(y_batch + r * channels, col);
  }
}

}  // namespace

/** \brief Two-pass channels-last GroupNorm(+SiLU) over (N, R, C) rows. */
template <typename T>
struct GroupNormSiluTwopassKernel {
  static_assert(std::is_same_v<T, fp32_t> || std::is_same_v<T, bf16_t> || std::is_same_v<T, fp16_t>);
  static constexpr int kVec = 16 / sizeof(T);

  static void
  run(tvm::ffi::TensorView x,
      tvm::ffi::TensorView y,
      tvm::ffi::TensorView weight,
      tvm::ffi::TensorView bias,
      int64_t num_groups,
      double eps,
      bool apply_silu) {
    using namespace host;

    auto N = SymbolicSize{"batch"};
    auto R = SymbolicSize{"rows"};
    auto C = SymbolicSize{"channels"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();
    TensorMatcher({N, R, C}).with_dtype<T>().with_device(device_).verify(x).verify(y);
    TensorMatcher({C}).with_dtype<T>().with_device(device_).verify(weight).verify(bias);

    const int64_t batches = N.unwrap();
    const int64_t rows = R.unwrap();
    const int64_t channels = C.unwrap();
    const DLDevice device = device_.unwrap();

    CHECK_HOST(batches > 0 && rows > 0) << "group_norm_silu: empty input";
    CHECK_HOST(batches <= kMaxGridY) << "group_norm_silu: batch " << batches << " exceeds grid.y";
    CHECK_HOST(channels >= kVec && channels <= 2048 && std::has_single_bit(static_cast<uint64_t>(channels)))
        << "group_norm_silu: channels " << channels << " must be a power of two in [" << kVec << ", 2048]";
    CHECK_HOST(num_groups >= 1 && channels % num_groups == 0)
        << "group_norm_silu: num_groups " << num_groups << " must divide channels " << channels;

    const uint32_t c = static_cast<uint32_t>(channels);
    const uint32_t cols_vec = c / kVec;
    const uint32_t col_bits = std::countr_zero(cols_vec);
    const uint32_t block = std::max<uint32_t>(kMaxBlockSize / 2, cols_vec);
    const uint32_t rows_par = block >> col_bits;
    const uint32_t sm_count = runtime::get_sm_count(device.device_id);

    // Chunk the rows so the partial pass fills the device (~4 blocks per SM
    // across the whole grid) while each block still amortizes its reduction
    // epilogue over many rows.
    const int64_t max_chunks = div_ceil(rows, static_cast<int64_t>(rows_par));
    const int64_t target_chunks =
        std::clamp<int64_t>(div_ceil(static_cast<int64_t>(sm_count) * 4, batches), 1, max_chunks);
    const int64_t rows_per_chunk = div_ceil(div_ceil(rows, target_chunks), static_cast<int64_t>(rows_par)) * rows_par;
    const int64_t nchunks = div_ceil(rows, rows_per_chunk);

    // Workspace: (N, nchunks, 2, C) fp32 partials followed by (N, 2, C) fp32
    // scale/shift.
    const int64_t partial_floats = batches * nchunks * 2 * channels;
    const int64_t ss_floats = batches * 2 * channels;
    auto workspace =
        ffi::alloc_workspace_tensor(static_cast<size_t>(partial_floats + ss_floats) * sizeof(float), device);
    float* partial_ptr = static_cast<float*>(workspace.data_ptr());
    float* ss_ptr = partial_ptr + partial_floats;

    const auto* x_ptr = static_cast<const T*>(x.data_ptr());
    auto* y_ptr = static_cast<T*>(y.data_ptr());
    const auto* w_ptr = static_cast<const T*>(weight.data_ptr());
    const auto* b_ptr = static_cast<const T*>(bias.data_ptr());

    const size_t partial_smem = rows_par > 1 ? static_cast<size_t>(block) * 2 * kVec * sizeof(float) : 0;
    LaunchKernel(dim3(static_cast<uint32_t>(nchunks), static_cast<uint32_t>(batches)), block, device, partial_smem)(
        gn_partial_kernel<T>, x_ptr, partial_ptr, rows, c, static_cast<uint32_t>(rows_per_chunk), col_bits);

    const uint32_t groups = static_cast<uint32_t>(num_groups);
    const size_t finalize_smem = (2 * static_cast<size_t>(c) + 2 * groups) * sizeof(float);
    LaunchKernel(static_cast<uint32_t>(batches), kFinalizeBlockSize, device, finalize_smem)(
        gn_finalize_kernel<T>,
        static_cast<const float*>(partial_ptr),
        w_ptr,
        b_ptr,
        ss_ptr,
        static_cast<uint32_t>(nchunks),
        c,
        c / groups,
        static_cast<float>(rows * (channels / num_groups)),
        static_cast<float>(eps));

    // The apply pass is pure streaming: one fully unrolled pass per block
    // keeps every load of a block in flight simultaneously, and the small
    // blocks give the scheduler many independent streams (Triton geometry).
    const int64_t rows_per_block = static_cast<int64_t>(rows_par) * apply_row_unroll<T>();
    const int64_t apply_blocks = div_ceil(rows, rows_per_block);
    CHECK_HOST(apply_blocks <= INT32_MAX) << "group_norm_silu: apply grid too large";
    LaunchKernel(dim3(static_cast<uint32_t>(apply_blocks), static_cast<uint32_t>(batches)), block, device)(
        gn_apply_kernel<T>,
        x_ptr,
        static_cast<const float*>(ss_ptr),
        y_ptr,
        rows,
        c,
        static_cast<uint32_t>(rows_per_block),
        col_bits,
        apply_silu);
  }
};

}  // namespace group_norm_silu_twopass

}  // namespace sglang
