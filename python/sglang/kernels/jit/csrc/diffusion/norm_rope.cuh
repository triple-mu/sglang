#include <sgl_kernel/tensor.h>

#include <sgl_kernel/math.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/warp.cuh>

#include <dlpack/dlpack.h>

#include <cstdint>
#include <type_traits>

namespace {

// Fused RMSNorm + RoPE over ONE tensor (q or k), in place. Math ported verbatim from
// fast-ulysses qk_norm_rope.cu: fp32 throughout with a single rounding at the store,
// eps inside rsqrt, per-channel fp32 weight, no bias, RoPE over the full head_dim.
// In-place is safe: every load of x is staged to registers/smem before any store.
struct NormRopeParams {
  void* __restrict__ x_ptr;              // [b, seq, n, head_dim] contiguous, mutated in place
  const float* __restrict__ weight_ptr;  // [head_dim] per-head | [n * head_dim] cross-head
  const float* __restrict__ cos_ptr;     // [seq, head_dim / 2]
  const float* __restrict__ sin_ptr;
  int seq;
  int num_heads;
  float eps;
};

constexpr uint32_t kThreadsPerBlock = 256;
constexpr uint32_t kWarpsPerBlock = kThreadsPerBlock / device::kWarpThreads;

// Rotate one normed channel c against its pair partner read from `row` (the staged fp32 row
// of head_dim values); s indexes the cos/sin tables.
template <int64_t kHeadDim, bool kInterleaved>
SGL_DEVICE float rope_pair(const float* row, const float* cos_ptr, const float* sin_ptr, int s, int c) {
  constexpr int kHalf = kHeadDim / 2;
  if constexpr (kInterleaved) {  // GPT-J adjacent pairs (2i, 2i+1)
    const int i = c >> 1;
    const float cs = cos_ptr[s * kHalf + i], sn = sin_ptr[s * kHalf + i];
    return ((c & 1) == 0) ? (row[c] * cs - row[c + 1] * sn) : (row[c] * cs + row[c - 1] * sn);
  } else {  // NeoX half-split pairs (i, i + d/2)
    if (c < kHalf) {
      const float cs = cos_ptr[s * kHalf + c], sn = sin_ptr[s * kHalf + c];
      return row[c] * cs - row[c + kHalf] * sn;
    }
    const int i = c - kHalf;
    const float cs = cos_ptr[s * kHalf + i], sn = sin_ptr[s * kHalf + i];
    return row[c] * cs + row[i] * sn;
  }
}

// One block per head-row, blockDim.x == kHeadDim: smem tree reduction for the per-head
// sum of squares, then RoPE reading the pair partner from the smem-staged normed row.
template <int64_t kHeadDim, bool kInterleaved, bool kUsePDL, typename DType>
__global__ void per_head_norm_rope_kernel(const NormRopeParams __grid_constant__ params) {
  using namespace device;

  auto* x = static_cast<DType*>(params.x_ptr);
  const int64_t row = blockIdx.x;  // over [b * seq * n)
  const int c = threadIdx.x;       // 0..kHeadDim-1
  const int s = static_cast<int>((row / params.num_heads) % params.seq);
  const int64_t base = row * kHeadDim;

  __shared__ float sm[kHeadDim];

  PDLWaitPrimary<kUsePDL>();

  float v = cast<float>(x[base + c]);
  sm[c] = v * v;
  __syncthreads();
#pragma unroll
  for (int stride = kHeadDim / 2; stride > 0; stride >>= 1) {
    if (c < stride) {
      sm[c] += sm[c + stride];
    }
    __syncthreads();
  }
  const float inv = math::rsqrt(sm[0] / kHeadDim + params.eps);
  __syncthreads();  // all threads have read sm[0] before the RoPE staging overwrites it
  v = v * inv * params.weight_ptr[c];

  sm[c] = v;
  __syncthreads();
  x[base + c] = cast<DType>(rope_pair<kHeadDim, kInterleaved>(sm, params.cos_ptr, params.sin_ptr, s, c));

  PDLTriggerSecondary<kUsePDL>();
}

// One block per token: warp-shuffle reduction of the sum of squares over the whole n*d row
// (one RMS scalar shared by all heads of the token), then the token's n*d normed fp32 values
// are staged in dynamic smem so RoPE can read each pair within its head segment.
template <int64_t kHeadDim, bool kInterleaved, bool kUsePDL, typename DType>
__global__ void cross_head_norm_rope_kernel(const NormRopeParams __grid_constant__ params) {
  using namespace device;

  auto* x = static_cast<DType*>(params.x_ptr);
  const int64_t token = blockIdx.x;  // over [b * seq)
  const int s = static_cast<int>(token % params.seq);
  const int nd = params.num_heads * static_cast<int>(kHeadDim);
  const int64_t base = token * static_cast<int64_t>(nd);
  const int tid = threadIdx.x;
  constexpr int kThreads = kThreadsPerBlock;

  extern __shared__ float sm[];  // nd floats: the token's normed values
  __shared__ float wsum[kWarpsPerBlock];

  PDLWaitPrimary<kUsePDL>();

  float local = 0.f;
  for (int j = tid; j < nd; j += kThreads) {
    const float v = cast<float>(x[base + j]);
    local += v * v;
  }
  local = warp::reduce_sum(local);
  const int lane = tid % kWarpThreads;
  const int wid = tid / kWarpThreads;
  if (lane == 0) {
    wsum[wid] = local;
  }
  __syncthreads();
  if (wid == 0) {
    float t = (lane < kWarpsPerBlock) ? wsum[lane] : 0.f;
    t = warp::reduce_sum(t);
    if (lane == 0) {
      wsum[0] = t;
    }
  }
  __syncthreads();
  const float inv = math::rsqrt(wsum[0] / nd + params.eps);

  for (int j = tid; j < nd; j += kThreads) {
    sm[j] = cast<float>(x[base + j]) * inv * params.weight_ptr[j];
  }
  __syncthreads();
  for (int j = tid; j < nd; j += kThreads) {
    const int h = j / kHeadDim;
    const int c = j - h * static_cast<int>(kHeadDim);
    const float* row = sm + static_cast<int64_t>(h) * kHeadDim;
    x[base + j] = cast<DType>(rope_pair<kHeadDim, kInterleaved>(row, params.cos_ptr, params.sin_ptr, s, c));
  }

  PDLTriggerSecondary<kUsePDL>();
}

template <int64_t kHeadDim, bool kCrossHead, bool kInterleaved, bool kUsePDL, typename DType>
struct NormRopeKernel {
  static_assert(std::is_same_v<DType, fp16_t> || std::is_same_v<DType, bf16_t>);
  static_assert(
      kHeadDim >= 2 && kHeadDim <= 1024 && (kHeadDim & (kHeadDim - 1)) == 0,
      "head_dim must be a power of two in [2, 1024]");

  static void
  run(const tvm::ffi::TensorView x,
      const tvm::ffi::TensorView weight,
      const tvm::ffi::TensorView cos,
      const tvm::ffi::TensorView sin,
      float eps) {
    using namespace host;

    auto B = SymbolicSize{"batch"};
    auto S = SymbolicSize{"seq"};
    auto N = SymbolicSize{"num_heads"};
    auto D = SymbolicSize{"head_dim"};
    auto device = SymbolicDevice{};
    D.set_value(kHeadDim);
    device.set_options<kDLCUDA>();

    // No .with_strides(): the default matcher requires packed (contiguous) tensors.
    TensorMatcher({B, S, N, D}).with_dtype<DType>().with_device(device).verify(x);
    const int64_t b = B.unwrap();
    const int64_t seq = S.unwrap();
    const int64_t n = N.unwrap();

    auto W = SymbolicSize{"weight_numel"};
    auto H = SymbolicSize{"half_dim"};
    W.set_value(kCrossHead ? n * kHeadDim : kHeadDim);
    H.set_value(kHeadDim / 2);
    TensorMatcher({W}).with_dtype<fp32_t>().with_device(device).verify(weight);
    // Exact [seq, d/2] (kernels index row s = 0..seq-1): an under-sized table would be an OOB read.
    TensorMatcher({S, H}).with_dtype<fp32_t>().with_device(device).verify(cos).verify(sin);

    const auto params = NormRopeParams{
        .x_ptr = x.data_ptr(),
        .weight_ptr = static_cast<const float*>(weight.data_ptr()),
        .cos_ptr = static_cast<const float*>(cos.data_ptr()),
        .sin_ptr = static_cast<const float*>(sin.data_ptr()),
        .seq = static_cast<int>(seq),
        .num_heads = static_cast<int>(n),
        .eps = eps,
    };

    if constexpr (kCrossHead) {
      constexpr auto kernel = cross_head_norm_rope_kernel<kHeadDim, kInterleaved, kUsePDL, DType>;
      const int64_t smem = n * kHeadDim * static_cast<int64_t>(sizeof(float));
      // One-time opt-in above the default 48KB dynamic-smem limit, up to the device cap
      // minus the kernel's static smem (wsum).
      static const int64_t smem_cap = [] {
        int dev = 0, opt_in_cap = 0;
        RuntimeDeviceCheck(::cudaGetDevice(&dev));
        RuntimeDeviceCheck(::cudaDeviceGetAttribute(&opt_in_cap, ::cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
        ::cudaFuncAttributes fa{};
        RuntimeDeviceCheck(::cudaFuncGetAttributes(&fa, kernel));
        const int dyn = opt_in_cap - static_cast<int>(fa.sharedSizeBytes);
        RuntimeDeviceCheck(::cudaFuncSetAttribute(kernel, ::cudaFuncAttributeMaxDynamicSharedMemorySize, dyn));
        return static_cast<int64_t>(dyn);
      }();
      RuntimeCheck(
          smem <= smem_cap,
          "cross_head norm+rope stages n*head_dim fp32 in shared memory: needs ",
          smem,
          " B > device cap ",
          smem_cap,
          " B (num_heads * head_dim too large)");
      LaunchKernel(static_cast<uint32_t>(b * seq), kThreadsPerBlock, device.unwrap(), static_cast<std::size_t>(smem))
          .enable_pdl(kUsePDL)(kernel, params);
    } else {
      constexpr auto kernel = per_head_norm_rope_kernel<kHeadDim, kInterleaved, kUsePDL, DType>;
      LaunchKernel(static_cast<uint32_t>(b * seq * n), static_cast<uint32_t>(kHeadDim), device.unwrap())
          .enable_pdl(kUsePDL)(kernel, params);
    }
  }
};

}  // namespace
