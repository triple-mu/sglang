// Sub-block routing inputs straight from the Sage INT8 operands.
//
// The SubBlock router ranks key blocks from mean-pooled Q and K cells. Pooling
// BF16 copies of the INT8 operands costs three full passes over the sequence
// (int8 -> bf16 for Q and K, K times its per-tile scale) before the pooling
// kernel reads them again. Pooling the INT8 tensors directly is one read of
// each operand and reproduces those pooled values bit for bit: a K token is
// `bf16(int8 * bf16(tile_scale))`, a Q token is the plain integer, the cell
// mean is `div.full.f32(sum, count) * factor` as the Triton pooling kernel
// computes it, and the result rounds to bf16 once. Sums of at most 64 such
// terms are exact in fp32, so the reduction order does not matter.
//
// The second kernel writes the Cake block tables for a query-block mask in one
// launch: masked-out query blocks keep every key block.
#pragma once
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>
#include <sgl_kernel/utils.cuh>

#include <cstdint>

namespace sglang {
namespace subblock_route_int8 {

constexpr uint32_t kHeadDim = 128;
constexpr uint32_t kScaleTile = 64;  // K tokens sharing one Sage scale
constexpr uint32_t kThreads = 128;   // 4 warps; a lane sums 4 channels, a thread finishes 1
constexpr uint32_t kWarps = kThreads / 32;

SGL_DEVICE float div_full(float a, float b) {
  float r;
  asm("div.full.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}

/// Mean-pool `sub` consecutive tokens of one head into a bf16 cell; z = 0 pools Q, z = 1 pools K.
__global__ void __launch_bounds__(kThreads) PoolInt8Kernel(
    const int8_t* __restrict__ q,
    const int8_t* __restrict__ k,
    const float* __restrict__ k_scale,
    bf16_t* __restrict__ pooled_q,
    bf16_t* __restrict__ pooled_k,
    uint32_t seq_alloc,
    uint32_t used,
    uint32_t sub_q,
    uint32_t sub_k,
    uint32_t cells_q,
    uint32_t cells_k,
    uint32_t k_scale_width,
    float q_factor) {
  const bool is_k = blockIdx.z == 1;
  const uint32_t cells = is_k ? cells_k : cells_q;
  const uint32_t cell = blockIdx.x;
  if (cell >= cells) return;
  const uint32_t head = blockIdx.y;  // batch folded into heads
  const uint32_t sub = is_k ? sub_k : sub_q;
  const uint32_t token0 = cell * sub;
  const uint32_t count = token0 < used ? min(sub, used - token0) : 0u;
  // Cells never straddle a scale tile (sub divides 64), so one scale serves the cell.
  float scale = 1.0f;
  if (is_k && count != 0) {
    const float s = k_scale[static_cast<uint64_t>(head) * k_scale_width + token0 / kScaleTile];
    scale = __bfloat162float(__float2bfloat16_rn(s));
  }
  const uint32_t warp = threadIdx.x / 32;
  const uint32_t lane = threadIdx.x % 32;
  const int8_t* src = (is_k ? k : q) + (static_cast<uint64_t>(head) * seq_alloc + token0) * kHeadDim + lane * 4;
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
  for (uint32_t t = warp; t < count; t += kWarps) {
    const uint32_t packed = *reinterpret_cast<const uint32_t*>(src + static_cast<uint64_t>(t) * kHeadDim);
#pragma unroll
    for (uint32_t j = 0; j < 4; ++j) {
      const float x = static_cast<float>(static_cast<int8_t>(static_cast<uint8_t>(packed >> (8 * j))));
      acc[j] += is_k ? __bfloat162float(__float2bfloat16_rn(x * scale)) : x;
    }
  }
  __shared__ float partial[kWarps][kHeadDim];
#pragma unroll
  for (uint32_t j = 0; j < 4; ++j) partial[warp][lane * 4 + j] = acc[j];
  __syncthreads();
  const uint32_t d = threadIdx.x;
  const float sum = ((partial[0][d] + partial[1][d]) + partial[2][d]) + partial[3][d];
  const float mean = div_full(sum, fmaxf(static_cast<float>(count), 1.0f)) * (is_k ? 1.0f : q_factor);
  bf16_t* dst = is_k ? pooled_k : pooled_q;
  dst[(static_cast<uint64_t>(head) * cells + cell) * kHeadDim + d] = __float2bfloat16_rn(mean);
}

/// `tables[row, j] = j`, replaced by `index[row, j]` for `j < topk` where the query block is sparse.
__global__ void __launch_bounds__(kThreads) BlockTablesKernel(
    const int32_t* __restrict__ index,
    const uint8_t* __restrict__ mask,
    int32_t* __restrict__ tables,
    int32_t* __restrict__ nums,
    uint32_t q_blocks,
    uint32_t topk,
    uint32_t num_blocks) {
  const uint32_t row = blockIdx.x;
  const bool sparse = mask[row % q_blocks] != 0;
  const int32_t* src = index + static_cast<uint64_t>(row) * topk;
  int32_t* dst = tables + static_cast<uint64_t>(row) * num_blocks;
  for (uint32_t j = threadIdx.x; j < num_blocks; j += kThreads) {
    dst[j] = (sparse && j < topk) ? src[j] : static_cast<int32_t>(j);
  }
  if (threadIdx.x == 0) nums[row] = static_cast<int32_t>(sparse ? topk : num_blocks);
}

struct Kernels {
  /*!
   * \param q [B, H, S, 128] int8 Sage Q operand; rows at or past `used` are not read
   * \param k [B, H, S, 128] int8 Sage K operand
   * \param k_scale [B, H, W] fp32 K scales, one per 64 tokens
   * \param pooled_q [B * H, cells_q, 128] bf16 output, Q cells times `q_factor`
   * \param pooled_k [B * H, cells_k, 128] bf16 output
   * \param used live rows of the sequence
   * \param sub_q tokens per Q cell, a divisor of 64
   * \param sub_k tokens per K cell, a divisor of 64
   * \param q_factor multiplier applied to the Q cell means (softmax scale times log2 e)
   */
  static void pool(
      tvm::ffi::TensorView q,
      tvm::ffi::TensorView k,
      tvm::ffi::TensorView k_scale,
      tvm::ffi::TensorView pooled_q,
      tvm::ffi::TensorView pooled_k,
      int64_t used,
      int64_t sub_q,
      int64_t sub_k,
      double q_factor) {
    using namespace host;
    SymbolicSize B{"batch"}, H{"heads"}, S{"rows"}, W{"k_scale_width"}, BH{"batch_heads"}, CQ{"q_cells"}, CK{"k_cells"};
    SymbolicDevice device;
    device.set_options<kDLCUDA>();
    TensorMatcher({B, H, S, kHeadDim}).with_dtype<int8_t>().with_device(device).verify(q).verify(k);
    TensorMatcher({B, H, W}).with_dtype<fp32_t>().with_device(device).verify(k_scale);
    TensorMatcher({BH, CQ, kHeadDim}).with_dtype<bf16_t>().with_device(device).verify(pooled_q);
    TensorMatcher({BH, CK, kHeadDim}).with_dtype<bf16_t>().with_device(device).verify(pooled_k);
    CHECK_HOST(BH.unwrap() == B.unwrap() * H.unwrap()) << "pooled rows must be batch * heads";
    CHECK_HOST(sub_q >= 1 && kScaleTile % sub_q == 0 && sub_k >= 1 && kScaleTile % sub_k == 0)
        << "cell sizes must divide " << kScaleTile << ", got " << sub_q << " and " << sub_k;
    CHECK_HOST(used >= 1 && used <= S.unwrap()) << "used rows must lie in [1, " << S.unwrap() << "]";
    CHECK_HOST(CQ.unwrap() * sub_q >= used && CK.unwrap() * sub_k >= used) << "the cells must cover the used rows";
    CHECK_HOST(div_ceil(used, static_cast<int64_t>(kScaleTile)) <= W.unwrap()) << "k_scale is narrower than the used rows";
    const int64_t cells = CQ.unwrap() > CK.unwrap() ? CQ.unwrap() : CK.unwrap();
    LaunchKernel(dim3(static_cast<uint32_t>(cells), static_cast<uint32_t>(BH.unwrap()), 2), kThreads, device.unwrap())(
        PoolInt8Kernel,
        static_cast<const int8_t*>(q.data_ptr()),
        static_cast<const int8_t*>(k.data_ptr()),
        static_cast<const float*>(k_scale.data_ptr()),
        static_cast<bf16_t*>(pooled_q.data_ptr()),
        static_cast<bf16_t*>(pooled_k.data_ptr()),
        static_cast<uint32_t>(S.unwrap()),
        static_cast<uint32_t>(used),
        static_cast<uint32_t>(sub_q),
        static_cast<uint32_t>(sub_k),
        static_cast<uint32_t>(CQ.unwrap()),
        static_cast<uint32_t>(CK.unwrap()),
        static_cast<uint32_t>(W.unwrap()),
        static_cast<float>(q_factor));
  }

  /*!
   * \param index [B, H, Gq, topk] int32 selected key blocks per query block
   * \param mask [Gq] uint8, nonzero where a query block is routed sparsely
   * \param tables [B, H, Gq, num_blocks] int32 output
   * \param nums [B, H, Gq] int32 output, key blocks each query block visits
   */
  static void block_tables(tvm::ffi::TensorView index, tvm::ffi::TensorView mask, tvm::ffi::TensorView tables, tvm::ffi::TensorView nums) {
    using namespace host;
    SymbolicSize B{"batch"}, H{"heads"}, G{"q_blocks"}, T{"topk"}, N{"num_blocks"};
    SymbolicDevice device;
    device.set_options<kDLCUDA>();
    TensorMatcher({B, H, G, T}).with_dtype<int32_t>().with_device(device).verify(index);
    TensorMatcher({G}).with_dtype<uint8_t>().with_device(device).verify(mask);
    TensorMatcher({B, H, G, N}).with_dtype<int32_t>().with_device(device).verify(tables);
    TensorMatcher({B, H, G}).with_dtype<int32_t>().with_device(device).verify(nums);
    CHECK_HOST(T.unwrap() <= N.unwrap()) << "topk " << T.unwrap() << " exceeds the " << N.unwrap() << " key blocks";
    const int64_t rows = B.unwrap() * H.unwrap() * G.unwrap();
    if (rows == 0) return;
    LaunchKernel(dim3(static_cast<uint32_t>(rows)), kThreads, device.unwrap())(
        BlockTablesKernel,
        static_cast<const int32_t*>(index.data_ptr()),
        static_cast<const uint8_t*>(mask.data_ptr()),
        static_cast<int32_t*>(tables.data_ptr()),
        static_cast<int32_t*>(nums.data_ptr()),
        static_cast<uint32_t>(G.unwrap()),
        static_cast<uint32_t>(T.unwrap()),
        static_cast<uint32_t>(N.unwrap()));
  }
};

}  // namespace subblock_route_int8
}  // namespace sglang
