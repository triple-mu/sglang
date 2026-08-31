// CUDA fast path for the Ulysses destination-major QKV pack:
//   out[dest, row, lh, seg * D + d] =
//       {q, k, v}[seg][row, dest * local_heads + lh, d]
// Pure data movement: bitwise identical to the Triton kernel and to the
// unpacked aten copy chain it replaces (layout/ulysses_qkv_triton.py).

#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>
#include <type_traits>

namespace sglang {

namespace ulysses_qkv_pack {

namespace {

constexpr uint32_t kBlockSize = 512;
constexpr uint32_t kMaxGridY = 65535;
constexpr uintptr_t kAlignment = 16;

/// Per-source strides in vector units (element strides / kVec).
struct SegmentStrides {
  int64_t q_row, q_head;
  int64_t k_row, k_head;
  int64_t v_row, v_head;
};

/**
 * \brief One block packs the contiguous out chunk of a single (dest, row).
 *
 * Stores sweep the chunk linearly (fully coalesced); each 16-byte load sits
 * inside a `head_size`-wide contiguous run of q/k/v, so warps still issue
 * full-width transactions on the strided sources.
 *
 * \tparam T          Element type: fp16_t | bf16_t | fp32_t
 * \tparam kVec       Elements per vector access (1 = scalar fallback)
 * \tparam kHeadVecsCT Compile-time head_size / kVec (0 = use runtime value)
 * \tparam kUsePDL    Whether to emit the PDL wait/trigger pair
 */
template <typename T, int kVec, int kHeadVecsCT, bool kUsePDL>
__global__ __launch_bounds__(kBlockSize) void pack_qkv_destination_major_kernel(
    T* __restrict__ out,
    const T* __restrict__ q,
    const T* __restrict__ k,
    const T* __restrict__ v,
    SegmentStrides strides,
    uint32_t local_heads,
    uint32_t head_vecs_rt) {
  const uint32_t head_vecs = kHeadVecsCT ? kHeadVecsCT : head_vecs_rt;
  const uint32_t chunk_vecs = local_heads * 3u * head_vecs;
  const uint32_t row = blockIdx.x;
  const uint32_t dest = blockIdx.y;
  const int64_t out_base = (static_cast<int64_t>(dest) * gridDim.x + row) * chunk_vecs;
  const uint32_t head_base = dest * local_heads;

  device::PDLWaitPrimary<kUsePDL>();

  for (uint32_t i = threadIdx.x; i < chunk_vecs; i += kBlockSize) {
    const uint32_t c = i % head_vecs;
    const uint32_t r1 = i / head_vecs;
    const uint32_t seg = r1 % 3u;
    const uint32_t lh = r1 / 3u;
    const T* src = seg == 0 ? q : (seg == 1 ? k : v);
    const int64_t stride_row = seg == 0 ? strides.q_row : (seg == 1 ? strides.k_row : strides.v_row);
    const int64_t stride_head = seg == 0 ? strides.q_head : (seg == 1 ? strides.k_head : strides.v_head);
    const int64_t src_vec =
        static_cast<int64_t>(row) * stride_row + static_cast<int64_t>(head_base + lh) * stride_head + c;
    device::AlignedVector<T, kVec> value;
    value.load(src, src_vec);
    value.store(out, out_base + i);
  }

  device::PDLTriggerSecondary<kUsePDL>();
}

}  // namespace

/** \brief Validate and launch the destination-major Ulysses QKV pack. */
template <typename T, bool kUsePDL>
struct PackQkvDestinationMajorKernel {
  static_assert(std::is_same_v<T, fp16_t> || std::is_same_v<T, bf16_t> || std::is_same_v<T, fp32_t>);

  /**
   * \param out Staging buffer [world, rows, local_heads, 3 * head_size], contiguous.
   * \param q   Query [rows, global_heads, head_size], last dim contiguous.
   * \param k   Key, same shape and layout constraints as q.
   * \param v   Value, same shape and layout constraints as q.
   */
  static void run(tvm::ffi::TensorView out, tvm::ffi::TensorView q, tvm::ffi::TensorView k, tvm::ffi::TensorView v) {
    using namespace host;

    auto R = SymbolicSize{"rows"};
    auto G = SymbolicSize{"global_heads"};
    auto D = SymbolicSize{"head_size"};
    auto W = SymbolicSize{"world_size"};
    auto LH = SymbolicSize{"local_heads"};
    auto D3 = SymbolicSize{"packed_head_size"};
    auto QR = SymbolicSize{"q_row_stride"};
    auto QH = SymbolicSize{"q_head_stride"};
    auto KR = SymbolicSize{"k_row_stride"};
    auto KH = SymbolicSize{"k_head_stride"};
    auto VR = SymbolicSize{"v_row_stride"};
    auto VH = SymbolicSize{"v_head_stride"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({R, G, D}).with_strides({QR, QH, 1}).with_dtype<T>().with_device(device).verify(q);
    TensorMatcher({R, G, D}).with_strides({KR, KH, 1}).with_dtype<T>().with_device(device).verify(k);
    TensorMatcher({R, G, D}).with_strides({VR, VH, 1}).with_dtype<T>().with_device(device).verify(v);
    TensorMatcher({W, R, LH, D3}).with_dtype<T>().with_device(device).verify(out);

    const int64_t rows = R.unwrap();
    const int64_t global_heads = G.unwrap();
    const int64_t head_size = D.unwrap();
    const int64_t world_size = W.unwrap();
    const int64_t local_heads = LH.unwrap();
    CHECK_HOST(world_size * local_heads == global_heads) << "out world_size*local_heads (" << world_size << "*"
                                                         << local_heads << ") must equal global_heads " << global_heads;
    CHECK_HOST(D3.unwrap() == 3 * head_size) << "out last dim must be 3*head_size";
    if (rows == 0 || global_heads == 0 || head_size == 0) {
      return;
    }
    CHECK_HOST(world_size <= kMaxGridY) << "world_size exceeds grid.y limit";
    CHECK_HOST(rows <= INT32_MAX) << "rows exceeds grid.x limit";

    auto* out_ptr = static_cast<T*>(out.data_ptr());
    const auto* q_ptr = static_cast<const T*>(q.data_ptr());
    const auto* k_ptr = static_cast<const T*>(k.data_ptr());
    const auto* v_ptr = static_cast<const T*>(v.data_ptr());
    CHECK_HOST(out_ptr != q_ptr && out_ptr != k_ptr && out_ptr != v_ptr) << "output must not alias an input";

    const SegmentStrides elem_strides{QR.unwrap(), QH.unwrap(), KR.unwrap(), KH.unwrap(), VR.unwrap(), VH.unwrap()};

    constexpr int kVec = kAlignment / sizeof(T);
    const auto aligned = [](const void* ptr) { return reinterpret_cast<uintptr_t>(ptr) % kAlignment == 0; };
    const bool vectorized = head_size % kVec == 0 && aligned(out_ptr) && aligned(q_ptr) && aligned(k_ptr) &&
                            aligned(v_ptr) && elem_strides.q_row % kVec == 0 && elem_strides.q_head % kVec == 0 &&
                            elem_strides.k_row % kVec == 0 && elem_strides.k_head % kVec == 0 &&
                            elem_strides.v_row % kVec == 0 && elem_strides.v_head % kVec == 0;

    const auto grid = dim3(static_cast<uint32_t>(rows), static_cast<uint32_t>(world_size));
    const auto launch = [&](auto kernel, const SegmentStrides& strides, uint32_t head_vecs) {
      LaunchKernel(grid, kBlockSize, device.unwrap())
          .enable_pdl(kUsePDL)(
              kernel, out_ptr, q_ptr, k_ptr, v_ptr, strides, static_cast<uint32_t>(local_heads), head_vecs);
    };

    if (vectorized) {
      const SegmentStrides vec_strides{
          elem_strides.q_row / kVec,
          elem_strides.q_head / kVec,
          elem_strides.k_row / kVec,
          elem_strides.k_head / kVec,
          elem_strides.v_row / kVec,
          elem_strides.v_head / kVec};
      const auto head_vecs = static_cast<uint32_t>(head_size / kVec);
      // 16 and 8 cover head_size 128/64 (bf16/fp16); the generic kernel takes
      // the vector width from its runtime argument.
      if (head_vecs == 16) {
        launch(pack_qkv_destination_major_kernel<T, kVec, 16, kUsePDL>, vec_strides, head_vecs);
      } else if (head_vecs == 8) {
        launch(pack_qkv_destination_major_kernel<T, kVec, 8, kUsePDL>, vec_strides, head_vecs);
      } else {
        launch(pack_qkv_destination_major_kernel<T, kVec, 0, kUsePDL>, vec_strides, head_vecs);
      }
    } else {
      launch(pack_qkv_destination_major_kernel<T, 1, 0, kUsePDL>, elem_strides, static_cast<uint32_t>(head_size));
    }
  }
};

}  // namespace ulysses_qkv_pack

}  // namespace sglang
