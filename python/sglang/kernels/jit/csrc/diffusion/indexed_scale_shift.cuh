// CUDA fast path for indexed diffusion adaLN modulation (MiniMax-H3 packed
// timestep/modality groups).
//
// In place: x[r] = round(round(x[r] * round(1 + scale[idx[r]])) + shift[idx[r]])
// with an RNE round to bf16 after (1 + scale), after the product, and on the
// final store -- the same rounding chain as the Triton
// `_indexed_scale_shift_bf16_kernel` it replaces, bitwise for finite values.

#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <algorithm>
#include <cstdint>

namespace sglang {

namespace indexed_scale_shift {

namespace {

constexpr uint32_t kMaxGridY = 65535;
constexpr uintptr_t kAlignment = 16;
constexpr int kVec = static_cast<int>(kAlignment / sizeof(bf16_t));

/// One lane of the eager modulation chain, keeping every bf16 rounding
/// boundary: round(1 + scale), round(x * (1 + scale)), round(prod + shift).
SGL_DEVICE bf16_t modulate_lane(bf16_t x, bf16_t scale, bf16_t shift) {
  const bf16_t one_plus_scale = device::cast<bf16_t>(1.0f + device::cast<fp32_t>(scale));
  const bf16_t product = device::cast<bf16_t>(device::cast<fp32_t>(x) * device::cast<fp32_t>(one_plus_scale));
  return device::cast<bf16_t>(device::cast<fp32_t>(product) + device::cast<fp32_t>(shift));
}

template <uint32_t kThreads, uint32_t kVecsPerThread, typename IdxT>
__launch_bounds__(kThreads) __global__ void indexed_scale_shift_kernel(
    bf16_t* __restrict__ x,  // updated in place; never aliases shift/scale
    const bf16_t* __restrict__ shift,
    const bf16_t* __restrict__ scale,
    const IdxT* __restrict__ indices,
    int64_t rows,
    int64_t row_vecs) {
  using Vec = device::AlignedVector<bf16_t, kVec>;
  constexpr uint32_t kVecsPerBlock = kThreads * kVecsPerThread;
  const int64_t vec_base = static_cast<int64_t>(blockIdx.x) * kVecsPerBlock + threadIdx.x;
  for (int64_t row = blockIdx.y; row < rows; row += gridDim.y) {
    const int64_t group = static_cast<int64_t>(indices[row]);
    const int64_t modulation_base = group * row_vecs;
    const int64_t activation_base = row * row_vecs;
#pragma unroll
    for (uint32_t k = 0; k < kVecsPerThread; ++k) {
      const int64_t col_vec = vec_base + k * kThreads;
      if (col_vec >= row_vecs) {
        break;
      }
      Vec x_vec, scale_vec, shift_vec;
      x_vec.load(x, activation_base + col_vec);
      scale_vec.load(scale, modulation_base + col_vec);
      shift_vec.load(shift, modulation_base + col_vec);
#pragma unroll
      for (int i = 0; i < kVec; ++i) {
        x_vec[i] = modulate_lane(x_vec[i], scale_vec[i], shift_vec[i]);
      }
      x_vec.store(x, activation_base + col_vec);
    }
  }
}

}  // namespace

/**
 * \brief Validate and launch the in-place indexed adaLN modulation.
 *
 * \tparam kThreads Threads per block (whole warps).
 * \tparam kVecsPerThread 16B vectors each thread covers per row.
 */
template <uint32_t kThreads, uint32_t kVecsPerThread>
struct IndexedScaleShiftKernel {
  static_assert(kThreads % 32 == 0);
  static_assert(kVecsPerThread >= 1);

  /**
   * \param x Activation rows [rows, hidden], bf16, updated in place.
   * \param shift Per-group shift rows [groups, hidden], bf16.
   * \param scale Per-group scale rows [groups, hidden], bf16.
   * \param indices Per-row group index [rows], int32 or int64.
   */
  static void
  run(tvm::ffi::TensorView x, tvm::ffi::TensorView shift, tvm::ffi::TensorView scale, tvm::ffi::TensorView indices) {
    using namespace host;

    auto R = SymbolicSize{"rows"};
    auto G = SymbolicSize{"groups"};
    auto D = SymbolicSize{"hidden_size"};
    auto idx_type = SymbolicDType{};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({R, D}).with_dtype<bf16_t>().with_device(device).verify(x);
    TensorMatcher({G, D}).with_dtype<bf16_t>().with_device(device).verify(shift).verify(scale);
    TensorMatcher({R}).with_dtype<int32_t, int64_t>(idx_type).with_device(device).verify(indices);

    const int64_t rows = R.unwrap();
    const int64_t hidden_size = D.unwrap();
    if (rows == 0 || hidden_size == 0) {
      return;
    }
    CHECK_HOST(hidden_size % kVec == 0) << "hidden size must be a multiple of " << kVec;

    auto* x_ptr = static_cast<bf16_t*>(x.data_ptr());
    const auto* shift_ptr = static_cast<const bf16_t*>(shift.data_ptr());
    const auto* scale_ptr = static_cast<const bf16_t*>(scale.data_ptr());
    CHECK_HOST(
        reinterpret_cast<uintptr_t>(x_ptr) % kAlignment == 0 &&
        reinterpret_cast<uintptr_t>(shift_ptr) % kAlignment == 0 &&
        reinterpret_cast<uintptr_t>(scale_ptr) % kAlignment == 0)
        << "indexed_scale_shift requires 16-byte aligned tensors";
    CHECK_HOST(x_ptr != shift_ptr && x_ptr != scale_ptr) << "x must not alias shift/scale";

    const int64_t row_vecs = hidden_size / kVec;
    const auto col_blocks =
        static_cast<uint32_t>(div_ceil(row_vecs, static_cast<int64_t>(kThreads * kVecsPerThread)));
    const auto row_blocks = static_cast<uint32_t>(std::min<int64_t>(rows, kMaxGridY));
    const auto launch = LaunchKernel(dim3(col_blocks, row_blocks), kThreads, device.unwrap());
    if (idx_type.is_type<int32_t>()) {
      launch(
          indexed_scale_shift_kernel<kThreads, kVecsPerThread, int32_t>,
          x_ptr,
          shift_ptr,
          scale_ptr,
          static_cast<const int32_t*>(indices.data_ptr()),
          rows,
          row_vecs);
    } else {
      launch(
          indexed_scale_shift_kernel<kThreads, kVecsPerThread, int64_t>,
          x_ptr,
          shift_ptr,
          scale_ptr,
          static_cast<const int64_t*>(indices.data_ptr()),
          rows,
          row_vecs);
    }
  }
};

}  // namespace indexed_scale_shift

}  // namespace sglang
