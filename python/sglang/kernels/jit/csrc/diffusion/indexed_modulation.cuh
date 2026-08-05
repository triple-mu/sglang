// CUDA fast path for the MiniMax-H3 indexed AdaLN modulation ops.
//
//   indexed_scale_shift: x[t, :] = round_bf16(x * round_bf16(1 + scale[idx[t]])) + shift[idx[t]]
//   indexed_gate:        x[t, :] = x + round_bf16(gate[idx[t]] * other[t, :])
//
// Both replace a Triton kernel whose BLOCK_N is next_power_of_2(hidden)
// (5376 -> 8192 in production), which wastes 34% of the vector lanes on
// masked-off columns and gives up 128-bit vectorization. Here every access is
// an unpredicated 128-bit LDG/STG and each thread keeps `kRowsPerBlock`
// independent loads in flight.
//
// The Triton `_round_bf16_to_fp32` boundaries are reproduced with native BF16
// arithmetic rather than the integer bit trick, which is what makes the
// scale-shift kernel DRAM-bound instead of ALU-bound (its 15 integer ops per
// element cost as much as the memory traffic):
//   * `round_bf16(1 + scale)` == `add.rn.bf16(1, scale)`. Both operands are
//     BF16, so the FP32 sum is either exact or already lands on a BF16 value;
//     there is no double rounding for the FP32 result to disagree about.
//   * `round_bf16(x * one_plus_scale)` == `mul.rn.bf16(x, one_plus_scale)`.
//     The FP32 product of two BF16 values is exact (16-bit significand), so
//     rounding it to BF16 is a single round-to-nearest-even step.
//   * The trailing `+ shift` stays in FP32 and rounds only at the store, via
//     `cvt.rn.bf16x2.f32`, which was checked against the Triton helper over
//     all 2^32 FP32 inputs (they agree except on NaN payloads).
// No `mul + add` pair survives, so there is nothing for nvcc to contract into
// an FMA and no need for `-fmad=false`. Sweeping all 2^32 BF16 operand pairs
// against the Triton kernel found the outputs identical except when the
// product is literally `0 * inf`, where this kernel returns the IEEE NaN and
// Triton returns 0; neither is reachable from finite activations.

#pragma once

#include <sgl_kernel/tensor.h>  // For TensorMatcher, SymbolicSize, SymbolicDevice
#include <sgl_kernel/utils.h>   // For div_ceil

#include <sgl_kernel/utils.cuh>  // For SGL_DEVICE, bf16_t, LaunchKernel
#include <sgl_kernel/vec.cuh>    // For device::AlignedVector

#include <cstdint>
#include <initializer_list>
#include <type_traits>

namespace sglang_indexed_modulation {

namespace {

constexpr uint32_t kBlockSize = 256;

SGL_DEVICE bf16_t scale_shift_value(bf16_t x, bf16_t shift, bf16_t scale) {
  const bf16_t scaled = __hmul(x, __hadd(__float2bfloat16(1.0f), scale));
  return __float2bfloat16(__bfloat162float(scaled) + __bfloat162float(shift));
}

SGL_DEVICE bf16x2_t scale_shift_value(bf16x2_t x, bf16x2_t shift, bf16x2_t scale) {
  const bf16x2_t scaled = __hmul2(x, __hadd2(__float2bfloat162_rn(1.0f), scale));
  const float2 scaled_f32 = __bfloat1622float2(scaled);
  const float2 shift_f32 = __bfloat1622float2(shift);
  return __float22bfloat162_rn(make_float2(scaled_f32.x + shift_f32.x, scaled_f32.y + shift_f32.y));
}

SGL_DEVICE bf16_t gate_value(bf16_t x, bf16_t gate, bf16_t other) {
  const bf16_t gated = __hmul(gate, other);
  return __float2bfloat16(__bfloat162float(x) + __bfloat162float(gated));
}

SGL_DEVICE bf16x2_t gate_value(bf16x2_t x, bf16x2_t gate, bf16x2_t other) {
  const float2 gated_f32 = __bfloat1622float2(__hmul2(gate, other));
  const float2 x_f32 = __bfloat1622float2(x);
  return __float22bfloat162_rn(make_float2(x_f32.x + gated_f32.x, x_f32.y + gated_f32.y));
}

/// \brief 128-bit access unit: BF16 pairs so the math runs on the x2 pipes.
template <int kVec>
using ModVec = device::
    AlignedVector<std::conditional_t<(kVec > 1), bf16x2_t, bf16_t>, static_cast<std::size_t>(kVec > 1 ? kVec / 2 : 1)>;

template <int kVec>
inline constexpr int kModElems = kVec > 1 ? kVec / 2 : 1;

// The 1-D grid is linearized as `col_block + row_tile * col_blocks` so that
// consecutive CTAs sweep a row tile's full width before moving on, which keeps
// the concurrent DRAM footprint contiguous. Threads of a partial row tile
// re-run the last valid row instead of predicating, which keeps every load
// unconditional; the update stays idempotent because all loads precede all
// stores.
template <int kVec, int kRowsPerBlock>
__global__ void indexed_scale_shift_kernel(
    bf16_t* __restrict__ x,
    const bf16_t* __restrict__ shift,
    const bf16_t* __restrict__ scale,
    const int64_t* __restrict__ indices,
    uint32_t rows,
    uint32_t n_vec,
    uint32_t col_blocks,
    uint32_t x_row_stride,
    uint32_t shift_row_stride,
    uint32_t scale_row_stride) {
  using Vec = ModVec<kVec>;

  const uint32_t col_vec = (blockIdx.x % col_blocks) * blockDim.x + threadIdx.x;
  if (col_vec >= n_vec) {
    return;
  }
  const uint32_t row_base = (blockIdx.x / col_blocks) * kRowsPerBlock;
  const uint32_t last_row = min(row_base + kRowsPerBlock - 1, rows - 1);

  Vec xv[kRowsPerBlock];
  Vec shv[kRowsPerBlock];
  Vec scv[kRowsPerBlock];
  uint32_t row_of[kRowsPerBlock];
#pragma unroll
  for (int r = 0; r < kRowsPerBlock; ++r) {
    const uint32_t row = min(row_base + r, last_row);
    const int64_t index = indices[row];
    row_of[r] = row;
    xv[r].load(x + static_cast<int64_t>(row) * x_row_stride, col_vec);
    shv[r].load(shift + index * shift_row_stride, col_vec);
    scv[r].load(scale + index * scale_row_stride, col_vec);
  }

#pragma unroll
  for (int r = 0; r < kRowsPerBlock; ++r) {
#pragma unroll
    for (int i = 0; i < kModElems<kVec>; ++i) {
      xv[r][i] = scale_shift_value(xv[r][i], shv[r][i], scv[r][i]);
    }
  }

#pragma unroll
  for (int r = 0; r < kRowsPerBlock; ++r) {
    xv[r].store(x + static_cast<int64_t>(row_of[r]) * x_row_stride, col_vec);
  }
}

template <int kVec, int kRowsPerBlock>
__global__ void indexed_gate_kernel(
    bf16_t* __restrict__ x,
    const bf16_t* __restrict__ gate,
    const bf16_t* __restrict__ other,
    const int64_t* __restrict__ indices,
    uint32_t rows,
    uint32_t n_vec,
    uint32_t col_blocks,
    uint32_t x_row_stride,
    uint32_t gate_row_stride,
    uint32_t other_row_stride) {
  using Vec = ModVec<kVec>;

  const uint32_t col_vec = (blockIdx.x % col_blocks) * blockDim.x + threadIdx.x;
  if (col_vec >= n_vec) {
    return;
  }
  const uint32_t row_base = (blockIdx.x / col_blocks) * kRowsPerBlock;
  const uint32_t last_row = min(row_base + kRowsPerBlock - 1, rows - 1);

  Vec xv[kRowsPerBlock];
  Vec gv[kRowsPerBlock];
  Vec ov[kRowsPerBlock];
  uint32_t row_of[kRowsPerBlock];
#pragma unroll
  for (int r = 0; r < kRowsPerBlock; ++r) {
    const uint32_t row = min(row_base + r, last_row);
    const int64_t index = indices[row];
    row_of[r] = row;
    xv[r].load(x + static_cast<int64_t>(row) * x_row_stride, col_vec);
    gv[r].load(gate + index * gate_row_stride, col_vec);
    ov[r].load(other + static_cast<int64_t>(row) * other_row_stride, col_vec);
  }

#pragma unroll
  for (int r = 0; r < kRowsPerBlock; ++r) {
#pragma unroll
    for (int i = 0; i < kModElems<kVec>; ++i) {
      xv[r][i] = gate_value(xv[r][i], gv[r][i], ov[r][i]);
    }
  }

#pragma unroll
  for (int r = 0; r < kRowsPerBlock; ++r) {
    xv[r].store(x + static_cast<int64_t>(row_of[r]) * x_row_stride, col_vec);
  }
}

inline bool aligned16(const void* p) {
  return (reinterpret_cast<uintptr_t>(p) & 0xF) == 0;
}

/// \brief 8 bf16 per access when every base pointer and row stride allows it.
inline bool
can_vectorize(int64_t hidden, std::initializer_list<int64_t> strides, std::initializer_list<const void*> ptrs) {
  constexpr int64_t kVec = 8;
  if (hidden % kVec != 0) {
    return false;
  }
  for (const int64_t stride : strides) {
    if (stride % kVec != 0) {
      return false;
    }
  }
  for (const void* ptr : ptrs) {
    if (!aligned16(ptr)) {
      return false;
    }
  }
  return true;
}

inline int64_t num_col_blocks(int64_t n_vec) {
  return host::div_ceil(n_vec, static_cast<int64_t>(kBlockSize));
}

inline uint32_t num_blocks(int64_t rows, int64_t col_blocks, int rows_per_block) {
  return static_cast<uint32_t>(host::div_ceil(rows, static_cast<int64_t>(rows_per_block)) * col_blocks);
}

}  // namespace

/// \brief In-place `x = round_bf16(x * round_bf16(1 + scale[idx])) + shift[idx]`.
template <int kRowsPerBlock>
struct IndexedScaleShiftKernel {
  static void
  run(tvm::ffi::TensorView x, tvm::ffi::TensorView shift, tvm::ffi::TensorView scale, tvm::ffi::TensorView indices) {
    using namespace host;

    auto rows_ = SymbolicSize{"rows"};
    auto hidden_ = SymbolicSize{"hidden"};
    auto params_ = SymbolicSize{"num_params"};
    auto x_stride_ = SymbolicSize{"x_row_stride"};
    auto shift_stride_ = SymbolicSize{"shift_row_stride"};
    auto scale_stride_ = SymbolicSize{"scale_row_stride"};
    auto device_ = SymbolicDevice{};

    TensorMatcher({rows_, hidden_})
        .with_strides({x_stride_, 1})
        .with_dtype<bf16_t>()
        .with_device<kDLCUDA>(device_)
        .verify(x);
    TensorMatcher({params_, hidden_})
        .with_strides({shift_stride_, 1})
        .with_dtype<bf16_t>()
        .with_device<kDLCUDA>(device_)
        .verify(shift);
    TensorMatcher({params_, hidden_})
        .with_strides({scale_stride_, 1})
        .with_dtype<bf16_t>()
        .with_device<kDLCUDA>(device_)
        .verify(scale);
    TensorMatcher({rows_}).with_dtype<int64_t>().with_device<kDLCUDA>(device_).verify(indices);

    const int64_t rows = rows_.unwrap();
    const int64_t hidden = hidden_.unwrap();
    if (rows == 0 || hidden == 0) {
      return;
    }

    auto* x_ptr = static_cast<bf16_t*>(x.data_ptr());
    const auto* shift_ptr = static_cast<const bf16_t*>(shift.data_ptr());
    const auto* scale_ptr = static_cast<const bf16_t*>(scale.data_ptr());
    const auto* index_ptr = static_cast<const int64_t*>(indices.data_ptr());
    const int64_t x_stride = x_stride_.unwrap();
    const int64_t shift_stride = shift_stride_.unwrap();
    const int64_t scale_stride = scale_stride_.unwrap();

    const int kVec =
        can_vectorize(hidden, {x_stride, shift_stride, scale_stride}, {x_ptr, shift_ptr, scale_ptr}) ? 8 : 1;
    const auto kernel = kVec == 8 ? indexed_scale_shift_kernel<8, kRowsPerBlock>  //
                                  : indexed_scale_shift_kernel<1, kRowsPerBlock>;
    const int64_t n_vec = hidden / kVec;
    const int64_t col_blocks = num_col_blocks(n_vec);

    LaunchKernel(num_blocks(rows, col_blocks, kRowsPerBlock), kBlockSize, device_.unwrap())(
        kernel,
        x_ptr,
        shift_ptr,
        scale_ptr,
        index_ptr,
        static_cast<uint32_t>(rows),
        static_cast<uint32_t>(n_vec),
        static_cast<uint32_t>(col_blocks),
        static_cast<uint32_t>(x_stride),
        static_cast<uint32_t>(shift_stride),
        static_cast<uint32_t>(scale_stride));
  }
};

/// \brief In-place `x = x + round_bf16(gate[idx] * other)`.
template <int kRowsPerBlock>
struct IndexedGateKernel {
  static void
  run(tvm::ffi::TensorView x, tvm::ffi::TensorView gate, tvm::ffi::TensorView other, tvm::ffi::TensorView indices) {
    using namespace host;

    auto rows_ = SymbolicSize{"rows"};
    auto hidden_ = SymbolicSize{"hidden"};
    auto params_ = SymbolicSize{"num_params"};
    auto x_stride_ = SymbolicSize{"x_row_stride"};
    auto gate_stride_ = SymbolicSize{"gate_row_stride"};
    auto other_stride_ = SymbolicSize{"other_row_stride"};
    auto device_ = SymbolicDevice{};

    TensorMatcher({rows_, hidden_})
        .with_strides({x_stride_, 1})
        .with_dtype<bf16_t>()
        .with_device<kDLCUDA>(device_)
        .verify(x);
    TensorMatcher({params_, hidden_})
        .with_strides({gate_stride_, 1})
        .with_dtype<bf16_t>()
        .with_device<kDLCUDA>(device_)
        .verify(gate);
    TensorMatcher({rows_, hidden_})
        .with_strides({other_stride_, 1})
        .with_dtype<bf16_t>()
        .with_device<kDLCUDA>(device_)
        .verify(other);
    TensorMatcher({rows_}).with_dtype<int64_t>().with_device<kDLCUDA>(device_).verify(indices);

    const int64_t rows = rows_.unwrap();
    const int64_t hidden = hidden_.unwrap();
    if (rows == 0 || hidden == 0) {
      return;
    }

    auto* x_ptr = static_cast<bf16_t*>(x.data_ptr());
    const auto* gate_ptr = static_cast<const bf16_t*>(gate.data_ptr());
    const auto* other_ptr = static_cast<const bf16_t*>(other.data_ptr());
    const auto* index_ptr = static_cast<const int64_t*>(indices.data_ptr());
    const int64_t x_stride = x_stride_.unwrap();
    const int64_t gate_stride = gate_stride_.unwrap();
    const int64_t other_stride = other_stride_.unwrap();

    const int kVec = can_vectorize(hidden, {x_stride, gate_stride, other_stride}, {x_ptr, gate_ptr, other_ptr}) ? 8 : 1;
    const auto kernel = kVec == 8 ? indexed_gate_kernel<8, kRowsPerBlock>  //
                                  : indexed_gate_kernel<1, kRowsPerBlock>;
    const int64_t n_vec = hidden / kVec;
    const int64_t col_blocks = num_col_blocks(n_vec);

    LaunchKernel(num_blocks(rows, col_blocks, kRowsPerBlock), kBlockSize, device_.unwrap())(
        kernel,
        x_ptr,
        gate_ptr,
        other_ptr,
        index_ptr,
        static_cast<uint32_t>(rows),
        static_cast<uint32_t>(n_vec),
        static_cast<uint32_t>(col_blocks),
        static_cast<uint32_t>(x_stride),
        static_cast<uint32_t>(gate_stride),
        static_cast<uint32_t>(other_stride));
  }
};

}  // namespace sglang_indexed_modulation
