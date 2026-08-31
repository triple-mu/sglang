// CUDA fast path for the MiniMax-H3 VAE ViT decoder residual triple:
//
//   y[r]   = residual_fp32[r] + widen(branch[r])      (residual, in place)
//   out[r] = cast_rn( weight * (rstd(y[r]) * y[r]) )  (fp16/bf16, for autocast)
//
// One kernel replaces the eager three-launch chain: aten's mixed-dtype add
// (fp32 accumulate), nn.RMSNorm on the fp32 trunk, and the autocast dtype
// cast the next Linear would otherwise issue.
//
// The reduction replicates aten's vectorized_layer_norm_kernel<float, float,
// /*rms_norm=*/true> structurally: a (32, 4) block per row, vec_size 4,
// per-thread sequential FFMA accumulation over grid-strided 16B vectors,
// shfl_down intra-warp tree, two-round smem inter-warp combine, then
// sigma2/N at thread 0 and rsqrtf(sigma2 + eps) everywhere. Verified
// bit-exact vs torch 2.13 on H200 (test_norm.py); retire if torch changes
// that kernel's reduction.

#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/math.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <cstdint>
#include <initializer_list>
#include <limits>
#include <type_traits>

namespace sglang {

namespace residual_rmsnorm_cast {

// aten pins vec_size to 4 for every dtype and launches dim3(32, 4) blocks;
// both are part of the replicated reduction order, not tuning knobs.
constexpr int kVec = 4;
constexpr int kThreads = 128;
constexpr int kWarps = kThreads / 32;

struct RowParams {
  void* residual;      // fp32, updated in place to y
  const void* branch;  // BranchT, read-only
  const void* weight;  // fp32 RMSNorm affine weight (ones when weightless)
  void* out;           // OutT, normalized output
  float eps;
};

template <int kHidden, typename BranchT, typename OutT>
__launch_bounds__(kThreads) __global__ void residual_rmsnorm_cast_kernel(const RowParams __grid_constant__ params) {
  using namespace device;
  static_assert(kHidden % kVec == 0);
  constexpr int kVecs = kHidden / kVec;
  constexpr int kVecsPerThread = (kVecs + kThreads - 1) / kThreads;

  using ResVec = AlignedVector<fp32_t, kVec>;
  using BranchVec = AlignedVector<BranchT, kVec>;
  using OutVec = AlignedVector<OutT, kVec>;

  const int64_t row = blockIdx.x;
  const int tid = threadIdx.x;  // == thrx of aten's (32, 4) block
  const int lane = tid & 31;    // aten threadIdx.x
  const int warp = tid >> 5;    // aten threadIdx.y

  auto* res_row = static_cast<fp32_t*>(params.residual) + row * kHidden;
  const auto* branch_row = static_cast<const BranchT*>(params.branch) + row * kHidden;
  auto* out_row = static_cast<OutT*>(params.out) + row * kHidden;

  // Residual add + per-thread sum of squares, in aten's element order: vector
  // i covers elements [4i, 4i+4), thread t owns vectors t, t+128, ... The add
  // is the eager CUDAFunctor_add<float> after LoadWithCast (one fp32 RN add),
  // and each square folds in via FFMA like aten's compiled `sigma2 += v*v`.
  ResVec y_regs[kVecsPerThread];
  float sigma2 = 0.0f;
#pragma unroll
  for (int k = 0; k < kVecsPerThread; ++k) {
    const int vec_id = tid + k * kThreads;
    if (kVecs % kThreads != 0 && vec_id >= kVecs) {
      break;
    }
    ResVec res_vec;
    BranchVec branch_vec;
    res_vec.load(res_row, vec_id);
    branch_vec.load(branch_row, vec_id);
    ResVec& y_vec = y_regs[k];
#pragma unroll
    for (int i = 0; i < kVec; ++i) {
      y_vec[i] = __fadd_rn(res_vec[i], cast<fp32_t>(branch_vec[i]));
    }
    y_vec.store(res_row, vec_id);
#pragma unroll
    for (int i = 0; i < kVec; ++i) {
      sigma2 = __fmaf_rn(y_vec[i], y_vec[i], sigma2);
    }
  }

  // Intra-warp shfl_down tree, offsets 16..1 (aten WARP_SHFL_DOWN loop).
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    sigma2 = __fadd_rn(sigma2, __shfl_down_sync(0xffffffffu, sigma2, offset));
  }

  // Inter-warp combine, offsets 2 then 1, upper-half warps write and
  // lower-half lanes 0 fold, with aten's two barriers per round.
  __shared__ float sigma_buf[kWarps];
#pragma unroll
  for (int offset = kWarps / 2; offset > 0; offset >>= 1) {
    if (lane == 0 && warp >= offset && warp < 2 * offset) {
      sigma_buf[warp - offset] = sigma2;
    }
    __syncthreads();
    if (lane == 0 && warp < offset) {
      sigma2 = __fadd_rn(sigma2, sigma_buf[warp]);
    }
    __syncthreads();
  }
  if (tid == 0) {
    sigma_buf[0] = __fdiv_rn(sigma2, static_cast<float>(kHidden));
  }
  __syncthreads();
  const float rstd = math::rsqrt(__fadd_rn(sigma_buf[0], params.eps));

  // Output in aten's association, gamma * (rstd * y), one RN cast on store.
  const auto* weight = static_cast<const fp32_t*>(params.weight);
#pragma unroll
  for (int k = 0; k < kVecsPerThread; ++k) {
    const int vec_id = tid + k * kThreads;
    if (kVecs % kThreads != 0 && vec_id >= kVecs) {
      break;
    }
    ResVec weight_vec;
    weight_vec.load(weight, vec_id);
    OutVec out_vec;
#pragma unroll
    for (int i = 0; i < kVec; ++i) {
      out_vec[i] = cast<OutT>(__fmul_rn(weight_vec[i], __fmul_rn(rstd, y_regs[k][i])));
    }
    out_vec.store(out_row, vec_id);
  }
}

inline void verify_alignment(std::initializer_list<const void*> pointers) {
  for (const void* pointer : pointers) {
    CHECK_HOST(reinterpret_cast<uintptr_t>(pointer) % (kVec * sizeof(fp32_t)) == 0)
        << "residual_rmsnorm_cast requires 16-byte aligned tensors";
  }
}

/**
 * \brief Validate and launch the fused residual add + RMSNorm + dtype cast.
 *
 * \tparam kHidden Row width in elements (aten's vectorized kernel needs % 4).
 * \tparam BranchT Branch dtype: fp16_t | bf16_t | fp32_t.
 * \tparam OutT Output (autocast) dtype: fp16_t | bf16_t.
 */
template <int kHidden, typename BranchT, typename OutT>
struct ResidualRMSNormCastKernel {
  static_assert(kHidden % kVec == 0);
  static_assert(std::is_same_v<BranchT, fp16_t> || std::is_same_v<BranchT, bf16_t> || std::is_same_v<BranchT, fp32_t>);
  static_assert(std::is_same_v<OutT, fp16_t> || std::is_same_v<OutT, bf16_t>);

  /**
   * \param out Normalized output rows [rows, kHidden], OutT.
   * \param residual Residual rows [rows, kHidden], fp32, updated in place.
   * \param branch Branch rows [rows, kHidden], BranchT.
   * \param weight RMSNorm affine weight [kHidden], fp32 (ones when weightless).
   * \param eps RMSNorm epsilon.
   */
  static void
  run(tvm::ffi::TensorView out,
      tvm::ffi::TensorView residual,
      tvm::ffi::TensorView branch,
      tvm::ffi::TensorView weight,
      double eps) {
    using namespace host;
    auto R = SymbolicSize{"rows"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({R, kHidden}).with_dtype<OutT>().with_device(device).verify(out);
    TensorMatcher({R, kHidden}).with_dtype<fp32_t>().with_device(device).verify(residual);
    TensorMatcher({R, kHidden}).with_dtype<BranchT>().with_device(device).verify(branch);
    TensorMatcher({kHidden}).with_dtype<fp32_t>().with_device(device).verify(weight);

    const int64_t rows = R.unwrap();
    if (rows == 0) {
      return;
    }
    CHECK_HOST(rows <= std::numeric_limits<uint32_t>::max()) << "rows out of range: " << rows;
    verify_alignment({out.data_ptr(), residual.data_ptr(), branch.data_ptr(), weight.data_ptr()});
    CHECK_HOST(out.data_ptr() != residual.data_ptr() && out.data_ptr() != branch.data_ptr())
        << "out must not alias residual/branch";
    CHECK_HOST(residual.data_ptr() != branch.data_ptr()) << "residual must not alias branch";

    const auto params = RowParams{
        .residual = residual.data_ptr(),
        .branch = branch.data_ptr(),
        .weight = weight.data_ptr(),
        .out = out.data_ptr(),
        .eps = static_cast<float>(eps),
    };
    LaunchKernel(static_cast<uint32_t>(rows), kThreads, device.unwrap())(
        residual_rmsnorm_cast_kernel<kHidden, BranchT, OutT>, params);
  }
};

}  // namespace residual_rmsnorm_cast

}  // namespace sglang
