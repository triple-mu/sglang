// Indexed adaLN modulation for the MiniMax-H3 DiT (packed timestep/modality
// groups), bitwise equal to the Triton `_indexed_scale_shift_bf16_kernel`:
//
//   out[r] = round(round(x[r] * round(1 + scale[g])) + shift[g]),  g = indices[r]
//
// with an RNE round to bf16 after (1 + scale), after the product and on the
// result. `out` is bf16 (in place when it aliases `x`) or fp32, the latter for
// the final layer, whose rows go straight into the fp32 output heads and no
// longer pay a separate bf16 store plus upcast. Ported from
// origin/minimax-h3-fusion-dit, with the output dtype and a modulation row
// stride added.

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

constexpr uint32_t kMaxGridY = 65535;
constexpr uintptr_t kAlignment = 16;
constexpr int kVec = 8;  // 16 B of bf16 per load

/// One lane of the eager modulation chain, every bf16 rounding boundary kept.
SGL_DEVICE bf16_t modulate_lane(bf16_t x, bf16_t scale, bf16_t shift) {
  const bf16_t one_plus_scale = device::cast<bf16_t>(1.0f + device::cast<fp32_t>(scale));
  const bf16_t product =
      device::cast<bf16_t>(device::cast<fp32_t>(x) * device::cast<fp32_t>(one_plus_scale));
  return device::cast<bf16_t>(device::cast<fp32_t>(product) + device::cast<fp32_t>(shift));
}

template <typename OutT, uint32_t kThreads, uint32_t kVecsPerThread, typename IdxT>
__launch_bounds__(kThreads) __global__ void indexed_scale_shift_kernel(
    OutT* out,  // may alias x when OutT is bf16: one thread reads and writes each vector
    const bf16_t* x,
    const bf16_t* __restrict__ shift,
    const bf16_t* __restrict__ scale,
    const IdxT* __restrict__ indices,
    int64_t rows,
    int64_t row_vecs,
    int64_t group_stride) {
  using InVec = device::AlignedVector<bf16_t, kVec>;
  // Stores stay 16 bytes wide whatever the output type: 8 bf16 or 2 x 4 fp32.
  constexpr int kOutVec = static_cast<int>(kAlignment / sizeof(OutT));
  using OutVec = device::AlignedVector<OutT, kOutVec>;
  constexpr uint32_t kVecsPerBlock = kThreads * kVecsPerThread;
  const int64_t vec_base = static_cast<int64_t>(blockIdx.x) * kVecsPerBlock + threadIdx.x;
  for (int64_t row = blockIdx.y; row < rows; row += gridDim.y) {
    const int64_t group = static_cast<int64_t>(indices[row]);
    const bf16_t* scale_row = scale + group * group_stride;
    const bf16_t* shift_row = shift + group * group_stride;
    const int64_t activation_base = row * row_vecs;
#pragma unroll
    for (uint32_t k = 0; k < kVecsPerThread; ++k) {
      const int64_t col_vec = vec_base + k * kThreads;
      if (col_vec >= row_vecs) break;
      InVec x_vec, scale_vec, shift_vec;
      x_vec.load(x, activation_base + col_vec);
      scale_vec.load(scale_row, col_vec);
      shift_vec.load(shift_row, col_vec);
#pragma unroll
      for (int part = 0; part < kVec / kOutVec; ++part) {
        OutVec out_vec;
#pragma unroll
        for (int i = 0; i < kOutVec; ++i) {
          const int lane = part * kOutVec + i;
          out_vec[i] = device::cast<OutT>(modulate_lane(x_vec[lane], scale_vec[lane], shift_vec[lane]));
        }
        // AlignedVector::store indexes in units of its own width.
        out_vec.store(out, (activation_base + col_vec) * (kVec / kOutVec) + part);
      }
    }
  }
}

/*!
 * \brief Validate and launch the indexed adaLN modulation.
 * \tparam OutT bf16 (in place allowed) or fp32.
 */
template <typename OutT, uint32_t kThreads, uint32_t kVecsPerThread>
struct IndexedScaleShiftKernel {
  static_assert(kThreads % 32 == 0);
  static_assert(kVecsPerThread >= 1);

  /*!
   * \param out     [rows, hidden] OutT; may be `x` itself when OutT is bf16
   * \param x       [rows, hidden] bf16
   * \param shift   [groups, hidden] bf16 rows with a shared row stride
   * \param scale   [groups, hidden] bf16, same stride as `shift`
   * \param indices [rows] int32 or int64 group per row
   */
  static void run(tvm::ffi::TensorView out, tvm::ffi::TensorView x, tvm::ffi::TensorView shift,
                  tvm::ffi::TensorView scale, tvm::ffi::TensorView indices) {
    using namespace host;
    SymbolicSize R{"rows"}, G{"groups"}, D{"hidden_size"}, GS{"group_stride"};
    SymbolicDType idx_type;
    SymbolicDevice device;
    device.set_options<kDLCUDA>();
    TensorMatcher({R, D}).with_dtype<bf16_t>().with_device(device).verify(x);
    TensorMatcher({R, D}).with_dtype<OutT>().with_device(device).verify(out);
    TensorMatcher({G, D})
        .with_strides({GS, 1})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(shift)
        .verify(scale);
    TensorMatcher({R}).with_dtype<int32_t, int64_t>(idx_type).with_device(device).verify(indices);

    const int64_t rows = R.unwrap();
    const int64_t hidden_size = D.unwrap();
    if (rows == 0 || hidden_size == 0) return;
    CHECK_HOST(hidden_size % kVec == 0) << "hidden size must be a multiple of " << kVec;
    CHECK_HOST(GS.unwrap() % kVec == 0) << "modulation row stride must keep 16-byte alignment";
    auto* out_ptr = static_cast<OutT*>(out.data_ptr());
    const auto* x_ptr = static_cast<const bf16_t*>(x.data_ptr());
    const auto* shift_ptr = static_cast<const bf16_t*>(shift.data_ptr());
    const auto* scale_ptr = static_cast<const bf16_t*>(scale.data_ptr());
    for (const void* pointer : {static_cast<const void*>(out_ptr), static_cast<const void*>(x_ptr),
                                static_cast<const void*>(shift_ptr), static_cast<const void*>(scale_ptr)}) {
      CHECK_HOST(reinterpret_cast<uintptr_t>(pointer) % kAlignment == 0)
          << "indexed_scale_shift requires 16-byte aligned tensors";
    }
    if constexpr (!std::is_same_v<OutT, bf16_t>) {
      CHECK_HOST(static_cast<const void*>(out_ptr) != static_cast<const void*>(x_ptr))
          << "an fp32 output cannot alias the bf16 input";
    }

    const int64_t row_vecs = hidden_size / kVec;
    const auto col_blocks =
        static_cast<uint32_t>(div_ceil(row_vecs, static_cast<int64_t>(kThreads * kVecsPerThread)));
    const auto row_blocks = static_cast<uint32_t>(std::min<int64_t>(rows, kMaxGridY));
    const auto launch = LaunchKernel(dim3(col_blocks, row_blocks), kThreads, device.unwrap());
    if (idx_type.is_type<int32_t>()) {
      launch(indexed_scale_shift_kernel<OutT, kThreads, kVecsPerThread, int32_t>, out_ptr, x_ptr,
             shift_ptr, scale_ptr, static_cast<const int32_t*>(indices.data_ptr()), rows, row_vecs,
             GS.unwrap());
    } else {
      launch(indexed_scale_shift_kernel<OutT, kThreads, kVecsPerThread, int64_t>, out_ptr, x_ptr,
             shift_ptr, scale_ptr, static_cast<const int64_t*>(indices.data_ptr()), rows, row_vecs,
             GS.unwrap());
    }
  }
};

}  // namespace indexed_scale_shift

}  // namespace sglang
