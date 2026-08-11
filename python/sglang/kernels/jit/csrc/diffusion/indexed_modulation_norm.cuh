// Fused indexed AdaLN modulation + RMSNorm for the MiniMax-H3 DiT block.
//
// Replaces the eager three-kernel chain in
// multimodal_gen/runtime/models/dits/minimax_h3.py:
//
//   residual = residual + round_bf16(gate[idx] * update)  (gate variant only)
//   n        = nn.RMSNorm(residual)
//   y        = round_bf16(n * round_bf16(1 + scale[idx])) + shift[idx]
//
// The kernel is bit-exact with that chain, which constrains the implementation
// in two ways:
//
//  1. The RMS reduction reproduces the geometry and the summation tree of
//     at::native::vectorized_layer_norm_kernel<BFloat16, float, true>, which is
//     the kernel nn.RMSNorm dispatches to for contiguous bf16 rows: 128 threads
//     shaped dim3(32, 4), vec_size 4, each thread accumulating its strided vec4
//     chunks sequentially, a __shfl_down_sync warp tree, a shared-memory tree
//     over the 4 warps, then rsqrtf(sum / N + eps) and gamma * (rstd * x).
//  2. The elementwise rounding mirrors the Triton kernels in
//     kernels/ops/diffusion/triton/indexed_modulation.py, which force a bf16
//     round after `gate * update`, after `1 + scale` and after
//     `norm * (1 + scale)` so nothing contracts into the following add.
//
// gate / scale / shift are AdaLN parameter tables of shape [num_params, H]
// gathered per row through `index`. They are chunks of a wider [num_params, 6H]
// projection output, so their row stride is passed explicitly.

#pragma once

#include <sgl_kernel/tensor.h>  // For TensorMatcher, SymbolicSize, SymbolicDevice
#include <sgl_kernel/utils.h>   // For CHECK_HOST, div_ceil

#include <sgl_kernel/math.cuh>   // For device::math::rsqrt
#include <sgl_kernel/type.cuh>   // For DTypeTrait conversions
#include <sgl_kernel/utils.cuh>  // For LaunchKernel, SGL_DEVICE, bf16_t
#include <sgl_kernel/vec.cuh>    // For device::AlignedVector

#include <cstdint>

namespace sglang {

namespace indexed_modulation_norm {

namespace {

// Geometry of at::native::vectorized_layer_norm_kernel. Changing any of these
// changes the fp32 summation tree and breaks bit-exactness with nn.RMSNorm.
constexpr int kVecSize = 4;
constexpr int kThreadsX = 32;
constexpr int kWarpsPerRow = 4;
constexpr int kThreadsPerRow = kThreadsX * kWarpsPerRow;

/// \brief Pointers and scalars for one fused modulation+norm launch.
struct Params {
  void* y;
  void* res_out;
  const void* x;
  const void* update;
  const void* gate;
  const void* scale;
  const void* shift;
  const void* weight;
  const int64_t* index;
  int64_t gate_row_stride;
  int64_t scale_row_stride;
  int64_t shift_row_stride;
  int64_t num_rows;
  float eps;
};

inline const void* const_data(const tvm::ffi::TensorView& t) {
  return static_cast<const char*>(t.data_ptr()) + t.byte_offset();
}

inline void* mutable_data(const tvm::ffi::TensorView& t) {
  return static_cast<char*>(t.data_ptr()) + t.byte_offset();
}

SGL_DEVICE float to_f32(bf16_t v) {
  return DTypeTrait<fp32_t>::from(v);
}

SGL_DEVICE bf16_t to_bf16(float v) {
  return DTypeTrait<bf16_t>::from(v);
}

/**
 * \brief Fused indexed gate + RMSNorm + indexed scale/shift, one row per CTA.
 *
 * \tparam kHidden  Row width; must be a multiple of kVecSize.
 * \tparam kHasGate Whether to fold the gated residual update in front of the norm.
 * \param p         Kernel parameters; `res_out`, `update` and `gate` are only
 *                  touched when `kHasGate` is true. `res_out` may alias `x`.
 */
template <int kHidden, bool kHasGate>
__global__ void indexed_modulation_norm_kernel(const Params __grid_constant__ p) {
  using Vec = device::AlignedVector<bf16_t, kVecSize>;
  static_assert(kHidden % kVecSize == 0, "hidden size must be a multiple of the vector width");
  static_assert(kHidden * sizeof(bf16_t) <= 48 * 1024, "staged row exceeds the static shared memory limit");
  constexpr int kNumVec = kHidden / kVecSize;
  constexpr int kIters = (kNumVec + kThreadsPerRow - 1) / kThreadsPerRow;

  const int lane = static_cast<int>(threadIdx.x);
  const int warp = static_cast<int>(threadIdx.y);
  const int thrx = lane + warp * kThreadsX;  // torch's `thrx`
  const int64_t row_off = static_cast<int64_t>(blockIdx.x) * kHidden;
  const int64_t param_row = p.index[blockIdx.x];

  const bf16_t* x_row = static_cast<const bf16_t*>(p.x) + row_off;
  // The norm input, kept in shared memory so the epilogue does not re-read it.
  // Every thread reads back exactly the vectors it wrote, so no barrier is needed.
  __shared__ bf16_t staged[kHidden];

  float sigma2 = 0.0f;
#pragma unroll
  for (int k = 0; k < kIters; ++k) {
    const int i = thrx + k * kThreadsPerRow;
    if (i < kNumVec) {
      Vec xv;
      xv.load(x_row, i);
      if constexpr (kHasGate) {
        Vec gv;
        Vec uv;
        gv.load(static_cast<const bf16_t*>(p.gate) + param_row * p.gate_row_stride, i);
        uv.load(static_cast<const bf16_t*>(p.update) + row_off, i);
#pragma unroll
        for (int j = 0; j < kVecSize; ++j) {
          const bf16_t gated = to_bf16(to_f32(gv[j]) * to_f32(uv[j]));
          xv[j] = to_bf16(to_f32(xv[j]) + to_f32(gated));
        }
        xv.store(static_cast<bf16_t*>(p.res_out) + row_off, i);
      }
      xv.store(staged, i);
#pragma unroll
      for (int j = 0; j < kVecSize; ++j) {
        const float v = to_f32(xv[j]);
        sigma2 = sigma2 + v * v;
      }
    }
  }

  // Raw __shfl_down_sync instead of device::warp::reduce_sum: the helper uses
  // shfl_xor, whose tree only provably agrees with torch's at lane 0.
#pragma unroll
  for (int offset = kThreadsX / 2; offset > 0; offset >>= 1) {
    sigma2 += __shfl_down_sync(0xffffffffu, sigma2, offset);
  }

  __shared__ float partial[kWarpsPerRow];
  __shared__ float mean_square;
#pragma unroll
  for (int offset = kWarpsPerRow / 2; offset > 0; offset >>= 1) {
    if (lane == 0 && warp >= offset && warp < 2 * offset) {
      partial[warp - offset] = sigma2;
    }
    __syncthreads();
    if (lane == 0 && warp < offset) {
      sigma2 += partial[warp];
    }
    __syncthreads();
  }
  if (thrx == 0) {
    mean_square = sigma2 / static_cast<float>(kHidden);
  }
  __syncthreads();
  const float rstd = device::math::rsqrt(mean_square + p.eps);

  const bf16_t* scale_row = static_cast<const bf16_t*>(p.scale) + param_row * p.scale_row_stride;
  const bf16_t* shift_row = static_cast<const bf16_t*>(p.shift) + param_row * p.shift_row_stride;
  const bf16_t* weight = static_cast<const bf16_t*>(p.weight);
  bf16_t* y_row = static_cast<bf16_t*>(p.y) + row_off;

#pragma unroll
  for (int k = 0; k < kIters; ++k) {
    const int i = thrx + k * kThreadsPerRow;
    if (i < kNumVec) {
      Vec nv;
      Vec wv;
      Vec scv;
      Vec shv;
      Vec out;
      nv.load(staged, i);
      wv.load(weight, i);
      scv.load(scale_row, i);
      shv.load(shift_row, i);
#pragma unroll
      for (int j = 0; j < kVecSize; ++j) {
        const bf16_t normed = to_bf16(to_f32(wv[j]) * (rstd * to_f32(nv[j])));
        const bf16_t one_plus_scale = to_bf16(1.0f + to_f32(scv[j]));
        const bf16_t scaled = to_bf16(to_f32(normed) * to_f32(one_plus_scale));
        out[j] = to_bf16(to_f32(scaled) + to_f32(shv[j]));
      }
      out.store(y_row, i);
    }
  }
}

/// \brief Verify one [num_params, kHidden] bf16 AdaLN table and return its row stride.
template <int kHidden>
inline int64_t verify_param_table(
    const tvm::ffi::TensorView& table, host::SymbolicSize& num_params, host::SymbolicDevice& device, const char* name) {
  auto row_stride = host::SymbolicSize{"adaln_row_stride"};
  host::TensorMatcher({num_params, kHidden})
      .with_strides({row_stride, 1})
      .with_dtype<bf16_t>()
      .with_device<kDLCUDA>(device)
      .verify(table);
  const int64_t stride = row_stride.unwrap();
  CHECK_HOST(stride % kVecSize == 0) << name << " row stride " << stride << " must be a multiple of " << kVecSize;
  return stride;
}

template <int kHidden, bool kHasGate>
inline void launch(const Params& p, DLDevice device) {
  host::LaunchKernel(dim3(static_cast<uint32_t>(p.num_rows)), dim3(kThreadsX, kWarpsPerRow), device)(
      indexed_modulation_norm_kernel<kHidden, kHasGate>, p);
}

}  // namespace

/// \brief y = round_bf16(RMSNorm(x) * round_bf16(1 + scale[idx])) + shift[idx].
template <int kHidden>
struct IndexedNormScaleShiftKernel {
  static void
  run(tvm::ffi::TensorView y,
      tvm::ffi::TensorView x,
      tvm::ffi::TensorView weight,
      tvm::ffi::TensorView scale,
      tvm::ffi::TensorView shift,
      tvm::ffi::TensorView index,
      double eps) {
    using namespace host;
    auto rows = SymbolicSize{"num_rows"};
    auto num_params = SymbolicSize{"num_params"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({rows, kHidden}).with_dtype<bf16_t>().with_device<kDLCUDA>(device).verify(x).verify(y);
    TensorMatcher({kHidden}).with_dtype<bf16_t>().with_device<kDLCUDA>(device).verify(weight);
    TensorMatcher({rows}).with_dtype<int64_t>().with_device<kDLCUDA>(device).verify(index);

    const Params p = {
        .y = mutable_data(y),
        .res_out = nullptr,
        .x = const_data(x),
        .update = nullptr,
        .gate = nullptr,
        .scale = const_data(scale),
        .shift = const_data(shift),
        .weight = const_data(weight),
        .index = static_cast<const int64_t*>(const_data(index)),
        .gate_row_stride = 0,
        .scale_row_stride = verify_param_table<kHidden>(scale, num_params, device, "scale"),
        .shift_row_stride = verify_param_table<kHidden>(shift, num_params, device, "shift"),
        .num_rows = rows.unwrap(),
        .eps = static_cast<float>(eps),
    };
    if (p.num_rows == 0) {
      return;
    }
    launch<kHidden, false>(p, device.unwrap());
  }
};

/// \brief In-place residual = residual + round_bf16(gate[idx] * update), then the norm+scale/shift above.
template <int kHidden>
struct IndexedGateNormScaleShiftKernel {
  static void
  run(tvm::ffi::TensorView y,
      tvm::ffi::TensorView residual,
      tvm::ffi::TensorView update,
      tvm::ffi::TensorView gate,
      tvm::ffi::TensorView weight,
      tvm::ffi::TensorView scale,
      tvm::ffi::TensorView shift,
      tvm::ffi::TensorView index,
      double eps) {
    using namespace host;
    auto rows = SymbolicSize{"num_rows"};
    auto num_params = SymbolicSize{"num_params"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({rows, kHidden})
        .with_dtype<bf16_t>()
        .with_device<kDLCUDA>(device)
        .verify(residual)
        .verify(update)
        .verify(y);
    TensorMatcher({kHidden}).with_dtype<bf16_t>().with_device<kDLCUDA>(device).verify(weight);
    TensorMatcher({rows}).with_dtype<int64_t>().with_device<kDLCUDA>(device).verify(index);
    // Empty input is a no-op, and its data pointers are all null -- check the
    // aliasing precondition only once there is something to alias.
    if (rows.unwrap() == 0) {
      return;
    }
    CHECK_HOST(mutable_data(y) != mutable_data(residual)) << "y must not alias residual";
    CHECK_HOST(const_data(y) != const_data(update)) << "y must not alias update";

    const Params p = {
        .y = mutable_data(y),
        .res_out = mutable_data(residual),
        .x = const_data(residual),
        .update = const_data(update),
        .gate = const_data(gate),
        .scale = const_data(scale),
        .shift = const_data(shift),
        .weight = const_data(weight),
        .index = static_cast<const int64_t*>(const_data(index)),
        .gate_row_stride = verify_param_table<kHidden>(gate, num_params, device, "gate"),
        .scale_row_stride = verify_param_table<kHidden>(scale, num_params, device, "scale"),
        .shift_row_stride = verify_param_table<kHidden>(shift, num_params, device, "shift"),
        .num_rows = rows.unwrap(),
        .eps = static_cast<float>(eps),
    };
    launch<kHidden, true>(p, device.unwrap());
  }
};

}  // namespace indexed_modulation_norm

}  // namespace sglang
