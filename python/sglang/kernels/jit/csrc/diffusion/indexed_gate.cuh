// CUDA fast path for the indexed gated residual of MiniMax-H3 adaLN
// (packed timestep/modality groups):
//
//   out[r] = round(x[r] + round_bf16(gate[idx[r]] * other[r]))
//
// with the product RNE-rounded to bf16 precision while staying an fp32
// register, then one final RNE fp32->bf16 store -- the same rounding chain as
// the Triton `_indexed_gate_bf16_kernel` it replaces, bitwise for finite
// values. `out` may alias `x` (the in-place form).

#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <algorithm>
#include <cstdint>

namespace sglang {

namespace indexed_gate {

namespace {

constexpr uint32_t kThreads = 224;  // 7 warps; hidden 5376 -> 672 vecs = 3 blocks exactly
constexpr uint32_t kMaxGridY = 65535;
constexpr uintptr_t kAlignment = 16;
constexpr int kVec = static_cast<int>(kAlignment / sizeof(bf16_t));

/// RNE-round an fp32 value to bf16 precision while keeping an fp32 register.
/// Same bit trick as `round_bf16_to_fp32` in ops/diffusion/common/numerics.py;
/// the integer detour also blocks FMA contraction of the surrounding mul/add.
SGL_DEVICE fp32_t round_bf16_to_fp32(fp32_t value) {
  const uint32_t bits = __float_as_uint(value);
  const uint32_t rounding_bias = 0x7FFFu + ((bits >> 16) & 1u);
  return __uint_as_float((bits + rounding_bias) & 0xFFFF0000u);
}

/// One lane of the gated-residual chain, keeping both bf16 rounding
/// boundaries: round_bf16(gate * other), then round(x + gated) on the store.
SGL_DEVICE bf16_t gate_lane(bf16_t x, bf16_t gate, bf16_t other) {
  const fp32_t gated = round_bf16_to_fp32(device::cast<fp32_t>(gate) * device::cast<fp32_t>(other));
  return device::cast<bf16_t>(device::cast<fp32_t>(x) + gated);
}

template <typename IdxT>
__launch_bounds__(kThreads) __global__ void indexed_gate_kernel(
    bf16_t* out,      // may alias x: each element is read and written once
    const bf16_t* x,  // by the same thread, so no __restrict__ on out/x
    const bf16_t* __restrict__ gate,
    const bf16_t* __restrict__ other,
    const IdxT* __restrict__ indices,
    int64_t rows,
    int64_t row_vecs) {
  using Vec = device::AlignedVector<bf16_t, kVec>;
  const int64_t col_vec = static_cast<int64_t>(blockIdx.x) * kThreads + threadIdx.x;
  if (col_vec >= row_vecs) {
    return;
  }
  for (int64_t row = blockIdx.y; row < rows; row += gridDim.y) {
    const int64_t group = static_cast<int64_t>(indices[row]);
    const int64_t activation_offset = row * row_vecs + col_vec;
    Vec x_vec, gate_vec, other_vec;
    x_vec.load(x, activation_offset);
    gate_vec.load(gate, group * row_vecs + col_vec);
    other_vec.load(other, activation_offset);
#pragma unroll
    for (int i = 0; i < kVec; ++i) {
      x_vec[i] = gate_lane(x_vec[i], gate_vec[i], other_vec[i]);
    }
    x_vec.store(out, activation_offset);
  }
}

}  // namespace

/// \brief Validate and launch the bit-exact indexed gated residual.
struct IndexedGateKernel {
  /**
   * \param out Output rows [rows, hidden], bf16; may alias x (in-place form).
   * \param x Residual rows [rows, hidden], bf16.
   * \param gate Per-group gate rows [groups, hidden], bf16.
   * \param other Update rows [rows, hidden], bf16.
   * \param indices Per-row group index [rows], int32 or int64.
   */
  static void
  run(tvm::ffi::TensorView out,
      tvm::ffi::TensorView x,
      tvm::ffi::TensorView gate,
      tvm::ffi::TensorView other,
      tvm::ffi::TensorView indices) {
    using namespace host;

    auto R = SymbolicSize{"rows"};
    auto G = SymbolicSize{"groups"};
    auto D = SymbolicSize{"hidden_size"};
    auto idx_type = SymbolicDType{};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({R, D}).with_dtype<bf16_t>().with_device(device).verify(out).verify(x).verify(other);
    TensorMatcher({G, D}).with_dtype<bf16_t>().with_device(device).verify(gate);
    TensorMatcher({R}).with_dtype<int32_t, int64_t>(idx_type).with_device(device).verify(indices);

    const int64_t rows = R.unwrap();
    const int64_t hidden_size = D.unwrap();
    if (rows == 0 || hidden_size == 0) {
      return;
    }
    CHECK_HOST(hidden_size % kVec == 0) << "hidden size must be a multiple of " << kVec;

    auto* out_ptr = static_cast<bf16_t*>(out.data_ptr());
    const auto* x_ptr = static_cast<const bf16_t*>(x.data_ptr());
    const auto* gate_ptr = static_cast<const bf16_t*>(gate.data_ptr());
    const auto* other_ptr = static_cast<const bf16_t*>(other.data_ptr());
    CHECK_HOST(
        reinterpret_cast<uintptr_t>(out_ptr) % kAlignment == 0 &&
        reinterpret_cast<uintptr_t>(x_ptr) % kAlignment == 0 &&
        reinterpret_cast<uintptr_t>(gate_ptr) % kAlignment == 0 &&
        reinterpret_cast<uintptr_t>(other_ptr) % kAlignment == 0)
        << "indexed_gate requires 16-byte aligned tensors";
    CHECK_HOST(out_ptr != gate_ptr && out_ptr != other_ptr) << "out must not alias gate/other";

    const int64_t row_vecs = hidden_size / kVec;
    const auto col_blocks = static_cast<uint32_t>(div_ceil(row_vecs, static_cast<int64_t>(kThreads)));
    const auto row_blocks = static_cast<uint32_t>(std::min<int64_t>(rows, kMaxGridY));
    const auto launch = LaunchKernel(dim3(col_blocks, row_blocks), kThreads, device.unwrap());
    if (idx_type.is_type<int32_t>()) {
      launch(
          indexed_gate_kernel<int32_t>,
          out_ptr,
          x_ptr,
          gate_ptr,
          other_ptr,
          static_cast<const int32_t*>(indices.data_ptr()),
          rows,
          row_vecs);
    } else {
      launch(
          indexed_gate_kernel<int64_t>,
          out_ptr,
          x_ptr,
          gate_ptr,
          other_ptr,
          static_cast<const int64_t*>(indices.data_ptr()),
          rows,
          row_vecs);
    }
  }
};

}  // namespace indexed_gate

}  // namespace sglang
