#pragma once
// RDMA transport for the MiniMax-H3 Ulysses exchange on a single node whose GPUs
// have no NVLink: every rank owns one mlx5 RoCE port and moves its per-peer
// payload with RDMA WRITE (one RC queue pair per peer), while a GPU-side epoch
// barrier over CUDA-IPC signal slots orders the exchange against the streams
// on every rank. Two operations share the machinery:
//
//   mode 2, chunks: [W, C] bytes in, [W, C] bytes out -- all_to_all_single over a
//           payload the sender already laid out destination-major;
//   mode 1, gather_heads: [S_global, H_local, D] in, [S_local, H, D] out -- the
//           Ulysses output all-to-all. The receiver's rows are interleaved by
//           head, so each destination exposes one interleaved UMR MKey per peer
//           and a single RDMA WRITE scatters straight into place.
//
// The NIC only ever reads transport-owned memory: every slot registers its own
// landing (input) and output allocations once at a fixed byte capacity and
// re-binds the operand geometry per call. Ported from flashinfer's PCIe Ulysses
// transport (PR #4951 lineage) with the CUDA P2P routes, mode 0 and the copy
// streams removed; the failure protocol (sticky abort slots, bounded quiesce,
// leak-and-poison teardown) is kept intact.

#include <arpa/inet.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <infiniband/mlx5dv.h>
#include <infiniband/verbs.h>
#include <sgl_kernel/ffi.h>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/utils.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/container/tuple.h>
#include <tvm/ffi/string.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace sglang::rdma_ulysses {

using tvm::ffi::Array;
using tvm::ffi::String;
using tvm::ffi::Tensor;
using tvm::ffi::TensorView;
using tvm::ffi::Tuple;

constexpr int kMaxWorld = 8;
constexpr int kPort = 1;
constexpr int kModeGather = 1;
constexpr int kModeChunks = 2;
// bytes_count and bytes_skip share a 16-bit pair in the UMR repeat entry.
constexpr int64_t kMaxInterleavedStride = 65535;
constexpr uint64_t kAbortSignal = ~uint64_t{0};

// ----------------------------------------------------------------- device --

struct PeerSignals {
  uint64_t* values[kMaxWorld];
};

__device__ __forceinline__ void PublishSignal(uint64_t* address, uint64_t value) {
  asm volatile("st.release.sys.global.u64 [%0], %1;" : : "l"(address), "l"(value) : "memory");
}

__device__ __forceinline__ uint64_t AcquireSignal(const uint64_t* address) {
  uint64_t value;
  asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(value) : "l"(address) : "memory");
  return value;
}

// The epoch lives in device memory so a replayed launch cannot re-publish a
// past epoch; every lane reaches the shuffle before the barrier body returns.
__device__ __forceinline__ uint64_t AdvanceEpoch(uint64_t* counter) {
  uint64_t epoch = 0;
  if (threadIdx.x == 0) {
    epoch = atomicAdd(reinterpret_cast<unsigned long long*>(counter), 1ULL) + 1ULL;
  }
  return __shfl_sync(0xffffffffu, epoch, 0);
}

// One lane per peer: publish this rank's epoch into the peer's slot, then spin
// until every peer's epoch has landed locally. Slots [W, 2W) are the sticky
// abort half; any lane seeing one terminates the barrier for the whole warp.
__device__ __forceinline__ void BarrierBody(uint64_t* local, PeerSignals peers, int world_size,
                                            int rank, uint64_t epoch) {
  const int peer = threadIdx.x;
  bool aborted = peer < world_size && AcquireSignal(local + world_size + peer) == kAbortSignal;
  if (__any_sync(0xffffffffu, aborted)) return;
  if (peer < world_size) PublishSignal(peers.values[peer] + rank, epoch);
  while (true) {
    aborted = peer < world_size && AcquireSignal(local + world_size + peer) == kAbortSignal;
    if (__any_sync(0xffffffffu, aborted)) return;
    const bool ready = peer >= world_size || AcquireSignal(local + peer) >= epoch;
    if (__all_sync(0xffffffffu, ready)) return;
  }
}

static __global__ void BarrierKernel(uint64_t* local, PeerSignals peers, int world_size, int rank,
                                     uint64_t* counter) {
  BarrierBody(local, peers, world_size, rank, AdvanceEpoch(counter));
}

// Rank r is the only writer of abort slot r in every peer's allocation.
static __global__ void PublishAbortKernel(PeerSignals peers, int world_size, int rank) {
  const int peer = threadIdx.x;
  if (peer < world_size) PublishSignal(peers.values[peer] + world_size + rank, kAbortSignal);
}

inline cudaError_t FillPeerSignals(PeerSignals& peers, uint64_t* const* peer_signals,
                                   int world_size) {
  for (int peer = 0; peer < world_size; ++peer) {
    if (peer_signals[peer] == nullptr) return cudaErrorInvalidDevicePointer;
    peers.values[peer] = peer_signals[peer];
  }
  return cudaSuccess;
}

inline cudaError_t EnqueueBarrier(uint64_t* local, uint64_t* const* peer_signals, int world_size,
                                  int rank, uint64_t* counter, cudaStream_t stream) {
  PeerSignals peers{};
  if (const auto status = FillPeerSignals(peers, peer_signals, world_size); status != cudaSuccess)
    return status;
  BarrierKernel<<<1, 32, 0, stream>>>(local, peers, world_size, rank, counter);
  return cudaGetLastError();
}

inline cudaError_t EnqueueAbort(uint64_t* const* peer_signals, int world_size, int rank,
                                cudaStream_t stream) {
  PeerSignals peers{};
  if (const auto status = FillPeerSignals(peers, peer_signals, world_size); status != cudaSuccess)
    return status;
  PublishAbortKernel<<<1, 32, 0, stream>>>(peers, world_size, rank);
  return cudaGetLastError();
}

// ------------------------------------------------------------------- wire --

struct GroupWire {
  uint32_t qpn[kMaxWorld]{};
  uint32_t psn[kMaxWorld]{};
  uint32_t mtu = 0;
  uint8_t gid[16]{};
};

struct SlotWire {
  uint64_t address = 0;
  uint32_t rkey = 0;
  uint32_t destination_rkey[kMaxWorld]{};
  cudaIpcMemHandle_t signal_ipc{};
};

template <typename T>
Array<int64_t> Encode(const T& value) {
  const auto* bytes = reinterpret_cast<const uint8_t*>(&value);
  Array<int64_t> result;
  for (size_t i = 0; i < sizeof(T); ++i) result.push_back(bytes[i]);
  return result;
}

template <typename T>
T DecodeAt(const Array<int64_t>& bytes, size_t offset) {
  CHECK_HOST(offset + sizeof(T) <= bytes.size()) << "truncated RDMA Ulysses metadata";
  T result{};
  auto* output = reinterpret_cast<uint8_t*>(&result);
  for (size_t i = 0; i < sizeof(T); ++i) {
    const int64_t value = bytes[offset + i];
    CHECK_HOST(value >= 0 && value <= 255) << "invalid RDMA Ulysses metadata byte";
    output[i] = static_cast<uint8_t>(value);
  }
  return result;
}

// ---------------------------------------------------------------- helpers --

inline void CheckCuda(cudaError_t status, const char* operation) {
  CHECK_HOST(status == cudaSuccess) << operation << ": " << cudaGetErrorString(status);
}

inline void CheckVerbs(int status, const char* operation) {
  CHECK_HOST(status == 0) << operation << ": " << std::strerror(status > 0 ? status : errno);
}

inline cudaError_t QueryEventUntil(cudaEvent_t event,
                                   std::chrono::steady_clock::time_point deadline) noexcept {
  while (true) {
    const cudaError_t status = cudaEventQuery(event);
    if (status != cudaErrorNotReady) return status;
    if (std::chrono::steady_clock::now() >= deadline) return cudaErrorNotReady;
    std::this_thread::yield();
  }
}

class ScopedCudaDevice {
 public:
  explicit ScopedCudaDevice(int device) noexcept : target_(device) {
    if (cudaGetDevice(&previous_) != cudaSuccess) return;
    if (previous_ != target_ && cudaSetDevice(target_) != cudaSuccess) return;
    active_ = true;
  }
  ~ScopedCudaDevice() noexcept {
    if (active_ && previous_ != target_) cudaSetDevice(previous_);
  }
  bool active() const { return active_; }

 private:
  int previous_ = 0;
  int target_ = 0;
  bool active_ = false;
};

// SYNC_MEMOPS + ibv_reg_mr for cudaMalloc memory, dma-buf for VMM-backed memory.
// SYNC_MEMOPS is allocation-scoped, which is why only transport-owned
// allocations are ever registered.
inline ibv_mr* RegisterGpuMr(ibv_pd* pd, void* pointer, size_t bytes, int access) {
  CHECK_HOST(pointer != nullptr) << "cannot register a null GPU pointer";
  unsigned int sync_memops = 1;
  const CUresult sync_status = cuPointerSetAttribute(&sync_memops, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS,
                                                     reinterpret_cast<CUdeviceptr>(pointer));
  int direct_errno = 0;
  if (sync_status == CUDA_SUCCESS) {
    if (auto* mr = ibv_reg_mr(pd, pointer, bytes, access)) return mr;
    direct_errno = errno;
  }
  const long page_size_value = ::sysconf(_SC_PAGESIZE);
  CHECK_HOST(page_size_value > 0) << "could not read the host page size";
  const auto page_size = static_cast<size_t>(page_size_value);
  const auto address = reinterpret_cast<uintptr_t>(pointer);
  const size_t offset = address % page_size;
  const size_t span = offset + bytes;
  const size_t remainder = span % page_size;
  const size_t export_bytes = span + (remainder == 0 ? 0 : page_size - remainder);
  int fd = -1;
  const CUresult export_status =
      cuMemGetHandleForAddressRange(&fd, static_cast<CUdeviceptr>(address - offset), export_bytes,
                                    CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0);
  CHECK_HOST(export_status == CUDA_SUCCESS)
      << "direct GPU MR registration failed"
      << (direct_errno == 0 ? "" : std::string(": ") + std::strerror(direct_errno))
      << ", and the allocation cannot be exported as dma-buf (CUresult " << export_status << ")";
  auto* mr = ibv_reg_dmabuf_mr(pd, offset, bytes, address, fd, access);
  const int dmabuf_errno = errno;
  ::close(fd);
  CHECK_HOST(mr != nullptr) << "ibv_reg_dmabuf_mr failed: " << std::strerror(dmabuf_errno);
  return mr;
}

// A uint8 view over transport-owned device memory. The DLManagedTensor lives
// on the heap because tvm-ffi keeps the pointer it is handed and calls its
// deleter from the Tensor destructor; the memory itself stays the slot's.
inline Tensor ViewDeviceBytes(void* data, int64_t bytes, int device) {
  struct Blob {
    DLManagedTensor managed{};
    int64_t shape[1]{};
    int64_t strides[1]{1};
  };
  auto* blob = new Blob{};
  blob->shape[0] = bytes;
  blob->managed.dl_tensor = DLTensor{
      .data = data,
      .device = DLDevice{kDLCUDA, device},
      .ndim = 1,
      .dtype = DLDataType{kDLUInt, 8, 1},
      .shape = blob->shape,
      .strides = blob->strides,
      .byte_offset = 0,
  };
  blob->managed.manager_ctx = blob;
  blob->managed.deleter = [](DLManagedTensor* self) { delete static_cast<Blob*>(self->manager_ctx); };
  return Tensor::FromDLPack(&blob->managed);
}

inline int64_t ElementBytes(TensorView tensor) {
  const DLDataType dtype = tensor.dtype();
  return static_cast<int64_t>(dtype.bits) * dtype.lanes / 8;
}

// The per-peer transfer of one exchange: `rows` runs of `width` contiguous
// bytes, `pitch` apart on the interleaved side. Mode 2 is a single row.
struct Geometry {
  int mode = 0;
  int64_t rows = 0;
  int64_t width = 0;
  int64_t pitch = 0;
  int64_t payload = 0;  // rows * width, what one peer sends to one peer
  int64_t total = 0;    // world_size * payload, the operand
  bool operator==(const Geometry& other) const {
    return mode == other.mode && rows == other.rows && width == other.width &&
           pitch == other.pitch;
  }
};

// ------------------------------------------------------------------ slots --

class Transport;

struct Slot {
  Transport* transport = nullptr;
  int mode = 0;
  int64_t capacity = 0;
  void* output = nullptr;
  void* landing = nullptr;
  ibv_mr* output_mr = nullptr;
  ibv_mr* landing_mr = nullptr;
  std::array<mlx5dv_mkey*, kMaxWorld> destination_mkeys{};
  // [0, W) barrier epochs, [W, 2W) sticky abort slots (rank r writes slot r).
  uint64_t* signals = nullptr;
  uint64_t* epoch_device = nullptr;
  std::array<uint64_t*, kMaxWorld> peer_signals{};
  std::array<SlotWire, kMaxWorld> peers{};
  Geometry bound{};
  bool has_geometry = false;
  uint64_t tag = 0;  // host counter naming this exchange's work requests
  bool connected = false;
  bool imports_closed = true;
  bool released = false;

  void Disconnect();
  void Release();
  ~Slot();
};

// --------------------------------------------------------------- transport --

class Transport {
 public:
  int rank = 0;
  int world_size = 0;
  int device = 0;
  std::string nic_name;
  int gid_index = -1;
  std::chrono::milliseconds timeout{10000};
  int write_ordering = cudaGPUDirectRDMAWritesOrderingNone;
  ibv_context* context = nullptr;
  ibv_pd* pd = nullptr;
  ibv_cq* cq = nullptr;
  std::array<ibv_qp*, kMaxWorld> qps{};
  std::array<ibv_qp_ex*, kMaxWorld> qpxs{};
  std::array<mlx5dv_qp_ex*, kMaxWorld> mlx5_qpxs{};
  GroupWire local{};
  std::array<GroupWire, kMaxWorld> peers{};
  uint64_t next_wr_id = 1;
  std::array<uint64_t, kMaxWorld> outstanding_send_wrs{};
  std::array<uint64_t, kMaxWorld> outstanding_recv_wrs{};
  cudaEvent_t phase_done = nullptr;
  cudaStream_t abort_stream = nullptr;
  cudaEvent_t abort_done = nullptr;
  uint64_t* abort_snapshot = nullptr;
  std::vector<std::unique_ptr<Slot>> slots;
  bool connected = false;
  bool failed = false;
  bool phase_inflight = false;
  bool teardown_safe = true;
  bool unsafe_release = false;

  Transport(int rank_arg, int world_size_arg, int device_arg, std::string nic_name_arg,
            int gid_index_arg, int64_t timeout_ms)
      : rank(rank_arg),
        world_size(world_size_arg),
        device(device_arg),
        nic_name(std::move(nic_name_arg)),
        gid_index(gid_index_arg),
        timeout(timeout_ms) {
    try {
      CHECK_HOST(world_size >= 2 && world_size <= kMaxWorld)
          << "RDMA Ulysses needs 2 to " << kMaxWorld << " ranks, got " << world_size;
      CHECK_HOST(rank >= 0 && rank < world_size) << "invalid rank " << rank;
      CHECK_HOST(!nic_name.empty()) << "RDMA Ulysses needs an mlx5 device name";
      CHECK_HOST(gid_index >= 0) << "RDMA Ulysses needs an explicit GID index";
      CHECK_HOST(timeout_ms > 0) << "timeout must be positive";

      ScopedCudaDevice device_guard(device);
      CheckCuda(cudaEventCreateWithFlags(&phase_done, cudaEventDisableTiming),
                "cudaEventCreateWithFlags(phase)");
      int least_priority = 0;
      int greatest_priority = 0;
      CheckCuda(cudaDeviceGetStreamPriorityRange(&least_priority, &greatest_priority),
                "cudaDeviceGetStreamPriorityRange");
      CheckCuda(
          cudaStreamCreateWithPriority(&abort_stream, cudaStreamNonBlocking, greatest_priority),
          "cudaStreamCreateWithPriority(abort)");
      CheckCuda(cudaEventCreateWithFlags(&abort_done, cudaEventDisableTiming),
                "cudaEventCreateWithFlags(abort done)");
      CheckCuda(
          cudaMallocHost(reinterpret_cast<void**>(&abort_snapshot), world_size * sizeof(uint64_t)),
          "cudaMallocHost(abort snapshot)");
      std::memset(abort_snapshot, 0, world_size * sizeof(uint64_t));

      CheckCuda(
          cudaDeviceGetAttribute(&write_ordering, cudaDevAttrGPUDirectRDMAWritesOrdering, device),
          "cudaDeviceGetAttribute(GPUDirectRDMAWritesOrdering)");
      if (write_ordering < cudaGPUDirectRDMAWritesOrderingOwner) {
        int flush_options = 0;
        CheckCuda(cudaDeviceGetAttribute(&flush_options, cudaDevAttrGPUDirectRDMAFlushWritesOptions,
                                         device),
                  "cudaDeviceGetAttribute(GPUDirectRDMAFlushWritesOptions)");
        CHECK_HOST(flush_options & cudaFlushGPUDirectRDMAWritesOptionHost)
            << "device " << device
            << " orders GPUDirect RDMA writes neither by itself nor through a host flush";
      }

      int count = 0;
      ibv_device** list = ibv_get_device_list(&count);
      CHECK_HOST(list != nullptr) << "ibv_get_device_list failed: " << std::strerror(errno);
      mlx5dv_context_attr context_attr{};
      context_attr.flags = MLX5DV_CONTEXT_FLAGS_DEVX;
      for (int i = 0; i < count; ++i) {
        if (nic_name == ibv_get_device_name(list[i])) {
          context = mlx5dv_open_device(list[i], &context_attr);
          break;
        }
      }
      ibv_free_device_list(list);
      CHECK_HOST(context != nullptr) << "cannot open " << nic_name
                                     << " with DEVX: " << std::strerror(errno);
      pd = ibv_alloc_pd(context);
      CHECK_HOST(pd != nullptr) << "ibv_alloc_pd failed: " << std::strerror(errno);
      cq = ibv_create_cq(context, 256, nullptr, nullptr, 0);
      CHECK_HOST(cq != nullptr) << "ibv_create_cq failed: " << std::strerror(errno);

      ibv_port_attr port{};
      ibv_gid gid{};
      ValidatePlannedGid(&port, &gid);
      local.mtu = port.active_mtu;
      std::memcpy(local.gid, &gid, sizeof(gid));

      for (int peer = 0; peer < world_size; ++peer) {
        if (peer == rank) continue;
        ibv_qp_init_attr_ex qp_attr{};
        qp_attr.send_cq = cq;
        qp_attr.recv_cq = cq;
        qp_attr.cap.max_send_wr = 128;
        qp_attr.cap.max_recv_wr = 1;
        qp_attr.cap.max_send_sge = 1;
        qp_attr.cap.max_recv_sge = 1;
        qp_attr.cap.max_inline_data = 128;
        qp_attr.qp_type = IBV_QPT_RC;
        qp_attr.comp_mask = IBV_QP_INIT_ATTR_PD | IBV_QP_INIT_ATTR_SEND_OPS_FLAGS;
        qp_attr.pd = pd;
        qp_attr.send_ops_flags = IBV_QP_EX_WITH_RDMA_WRITE_WITH_IMM;
        mlx5dv_qp_init_attr dv_attr{};
        dv_attr.comp_mask = MLX5DV_QP_INIT_ATTR_MASK_SEND_OPS_FLAGS;
        dv_attr.send_ops_flags = MLX5DV_QP_EX_WITH_MKEY_CONFIGURE;
        auto* qp = mlx5dv_create_qp(context, &qp_attr, &dv_attr);
        CHECK_HOST(qp != nullptr) << "mlx5dv_create_qp failed: " << std::strerror(errno);
        qps[peer] = qp;
        qpxs[peer] = ibv_qp_to_qp_ex(qp);
        mlx5_qpxs[peer] = mlx5dv_qp_ex_from_ibv_qp_ex(qpxs[peer]);
        CHECK_HOST(qpxs[peer] != nullptr && mlx5_qpxs[peer] != nullptr)
            << "cannot create extended mlx5 QP";
        local.qpn[peer] = qp->qp_num;
        local.psn[peer] = 0x120000 + rank * 0x1000 + peer * 0x10;
      }
    } catch (...) {
      Release();
      throw;
    }
  }

  ~Transport() { Release(); }

  void ValidatePlannedGid(ibv_port_attr* port_out, ibv_gid* gid_out) const {
    CHECK_HOST(context != nullptr) << "no open verbs context";
    ibv_port_attr port{};
    CheckVerbs(ibv_query_port(context, kPort, &port), "ibv_query_port");
    CHECK_HOST(port.state == IBV_PORT_ACTIVE) << nic_name << " port 1 is not active";
    CHECK_HOST(port.link_layer == IBV_LINK_LAYER_ETHERNET) << nic_name << " port 1 is not RoCE";
    CHECK_HOST(gid_index < port.gid_tbl_len)
        << "GID index " << gid_index << " is outside " << nic_name << " port 1 table length "
        << port.gid_tbl_len;
    ibv_gid_entry entry{};
    CheckVerbs(ibv_query_gid_ex(context, kPort, gid_index, &entry, 0), "ibv_query_gid_ex");
    CHECK_HOST(entry.gid_type == IBV_GID_TYPE_ROCE_V2) << "GID " << gid_index << " is not RoCE v2";
    CHECK_HOST(entry.ndev_ifindex != 0) << "GID " << gid_index << " has no netdev";
    const auto* raw = entry.gid.raw;
    const bool ipv4_mapped = std::all_of(raw, raw + 10, [](uint8_t b) { return b == 0; }) &&
                             raw[10] == 0xff && raw[11] == 0xff;
    const bool ipv4_nonzero = std::any_of(raw + 12, raw + 16, [](uint8_t b) { return b != 0; });
    CHECK_HOST(ipv4_mapped && ipv4_nonzero) << "GID " << gid_index
                                            << " is not a non-zero IPv4-mapped address";
    if (port_out != nullptr) *port_out = port;
    if (gid_out != nullptr) std::memcpy(gid_out, &entry.gid, sizeof(entry.gid));
  }

  void LeakAndPoison(const char* reason) noexcept {
    std::fprintf(stderr,
                 "RDMA Ulysses: %s; verbs and CUDA resources are intentionally leaked. Call "
                 "shutdown() on every rank before dropping the communicator.\n",
                 reason);
    for (auto& slot : slots) slot.release();
    slots.clear();
  }

  void Release() noexcept {
    ScopedCudaDevice device_guard(device);
    if (!TeardownSafe()) {
      return LeakAndPoison("outstanding RDMA GPU work could not be bounded before teardown");
    }
    for (const auto& slot : slots) {
      if (!slot->imports_closed) {
        return LeakAndPoison("a transport was torn down while a slot still had peer imports");
      }
    }
    if (unsafe_release) return LeakAndPoison("an earlier teardown lost its retry ledger");
    try {
      for (auto& slot : slots) slot->Release();
    } catch (...) {
      return LeakAndPoison("slot teardown failed");
    }
    slots.clear();
    if (phase_done != nullptr) cudaEventDestroy(phase_done);
    phase_done = nullptr;
    if (abort_done != nullptr) cudaEventDestroy(abort_done);
    abort_done = nullptr;
    if (abort_stream != nullptr) cudaStreamDestroy(abort_stream);
    abort_stream = nullptr;
    if (abort_snapshot != nullptr) cudaFreeHost(abort_snapshot);
    abort_snapshot = nullptr;
    for (auto*& qp : qps) {
      if (qp != nullptr) ibv_destroy_qp(qp);
      qp = nullptr;
    }
    if (cq != nullptr) ibv_destroy_cq(cq);
    cq = nullptr;
    if (pd != nullptr) ibv_dealloc_pd(pd);
    pd = nullptr;
    if (context != nullptr) ibv_close_device(context);
    context = nullptr;
  }

  void EnsureHealthy() const {
    CHECK_HOST(!failed) << "RDMA Ulysses transport is poisoned by an earlier failure";
  }

  bool TeardownSafe() const noexcept {
    return teardown_safe && !phase_inflight && !unsafe_release && OutstandingWrs() == 0;
  }

  // The peer and direction ride in the completion so the shared CQ retires the
  // exact per-QP ledger, also while several QPs flush after a failure.
  uint64_t NewWrId(int peer, bool receive) {
    CHECK_HOST(peer >= 0 && peer < world_size) << "invalid WR peer " << peer;
    constexpr uint64_t kReceiveBit = 0x10;
    constexpr unsigned kMetadataBits = 8;
    CHECK_HOST(next_wr_id < (uint64_t{1} << (64 - kMetadataBits))) << "WR id space exhausted";
    return (next_wr_id++ << kMetadataBits) | (receive ? kReceiveBit : 0) |
           static_cast<uint64_t>(peer);
  }

  bool RetireCompletion(const ibv_wc& completion) noexcept {
    constexpr uint64_t kPeerMask = 0x0f;
    constexpr uint64_t kReceiveBit = 0x10;
    const int peer = static_cast<int>(completion.wr_id & kPeerMask);
    if (peer < 0 || peer >= world_size) return false;
    auto& outstanding = (completion.wr_id & kReceiveBit) != 0 ? outstanding_recv_wrs[peer]
                                                              : outstanding_send_wrs[peer];
    if (outstanding == 0) return false;
    --outstanding;
    return true;
  }

  uint64_t OutstandingWrs() const noexcept {
    uint64_t result = 0;
    for (int peer = 0; peer < world_size; ++peer) {
      result += outstanding_send_wrs[peer] + outstanding_recv_wrs[peer];
    }
    return result;
  }

  // Host-synchronous by design: the barrier kernel, the abort-snapshot readback
  // and the event wait bound every phase, and cudaEventSynchronize blocks in
  // the driver instead of flooding API traces with a query spin.
  void RunBarrier(Slot* slot, cudaStream_t stream, bool opening) {
    phase_inflight = true;
    CheckCuda(EnqueueBarrier(slot->signals, slot->peer_signals.data(), world_size, rank,
                             slot->epoch_device, stream),
              opening ? "enqueue opening barrier" : "enqueue closing barrier");
    CheckCuda(cudaMemcpyAsync(abort_snapshot, slot->signals + world_size,
                              world_size * sizeof(uint64_t), cudaMemcpyDeviceToHost, stream),
              "cudaMemcpyAsync(abort snapshot)");
    CheckCuda(cudaEventRecord(phase_done, stream), "cudaEventRecord(barrier)");
    CheckCuda(cudaEventSynchronize(phase_done), "wait for barrier");
    phase_inflight = false;
    for (int peer = 0; peer < world_size; ++peer) {
      CHECK_HOST(abort_snapshot[peer] != kAbortSignal)
          << "RDMA Ulysses peer " << peer << " aborted the exchange";
    }
  }

  bool TryPublishAbort(Slot* slot) noexcept {
    if (slot == nullptr || abort_stream == nullptr || abort_done == nullptr) return false;
    if (EnqueueAbort(slot->peer_signals.data(), world_size, rank, abort_stream) != cudaSuccess)
      return false;
    if (cudaEventRecord(abort_done, abort_stream) != cudaSuccess) return false;
    return QueryEventUntil(abort_done, std::chrono::steady_clock::now() + timeout) == cudaSuccess;
  }

  bool TryDrain(cudaStream_t current) noexcept {
    if (phase_done == nullptr || cudaEventRecord(phase_done, current) != cudaSuccess) return false;
    if (QueryEventUntil(phase_done, std::chrono::steady_clock::now() + timeout) != cudaSuccess)
      return false;
    phase_inflight = false;
    return true;
  }

  // Move every QP to ERR and wait for each posted WR's own or flush completion:
  // registered memory may only be released once the ledger is empty.
  bool Quiesce() noexcept {
    failed = true;
    bool safe = true;
    for (int peer = 0; peer < world_size; ++peer) {
      auto* qp = qps[peer];
      if (qp == nullptr) continue;
      ibv_qp_attr attr{};
      attr.qp_state = IBV_QPS_ERR;
      const int status = ibv_modify_qp(qp, &attr, IBV_QP_STATE);
      if (status != 0 && (outstanding_send_wrs[peer] != 0 || outstanding_recv_wrs[peer] != 0)) {
        safe = false;
      }
    }
    if (cq == nullptr) {
      safe = safe && OutstandingWrs() == 0;
      if (!safe) teardown_safe = false;
      return safe;
    }
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (OutstandingWrs() != 0 && std::chrono::steady_clock::now() < deadline) {
      ibv_wc entries[kMaxWorld]{};
      const int count = ibv_poll_cq(cq, kMaxWorld, entries);
      if (count < 0) {
        safe = false;
        break;
      }
      if (count == 0) {
        std::this_thread::yield();
        continue;
      }
      for (int index = 0; index < count; ++index) {
        if (!RetireCompletion(entries[index])) safe = false;
        if (entries[index].status != IBV_WC_SUCCESS &&
            entries[index].status != IBV_WC_WR_FLUSH_ERR) {
          safe = false;
        }
      }
    }
    if (OutstandingWrs() != 0) safe = false;
    if (!safe) teardown_safe = false;
    return safe;
  }

  void AbortAndQuiesce(Slot* slot, cudaStream_t current) noexcept {
    failed = true;
    const bool abort_published = TryPublishAbort(slot);
    const bool drained = TryDrain(current);
    const bool quiesced = Quiesce();
    if (!abort_published || !drained || !quiesced) teardown_safe = false;
  }

  template <typename Body>
  void RetireOnFailure(Body&& body) {
    EnsureHealthy();
    try {
      body();
    } catch (...) {
      Quiesce();
      throw;
    }
  }

  // With expected_receives == 0 completions are only counted; otherwise every
  // receive must carry this exchange's immediate tag, or a stale completion
  // could let the closing barrier publish before the peer data landed.
  void Poll(int expected_sends, int expected_receives = 0, uint32_t immediate = 0) {
    EnsureHealthy();
    int sends = 0;
    int receives = 0;
    unsigned empty_polls = 0;
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (sends < expected_sends || receives < expected_receives) {
      ibv_wc entries[kMaxWorld]{};
      const int count = ibv_poll_cq(cq, kMaxWorld, entries);
      if (count < 0) teardown_safe = false;
      CHECK_HOST(count >= 0) << "ibv_poll_cq failed";
      bool ledger_ok = true;
      for (int i = 0; i < count; ++i) ledger_ok = RetireCompletion(entries[i]) && ledger_ok;
      if (!ledger_ok) teardown_safe = false;
      CHECK_HOST(ledger_ok) << "mlx5 completion does not match an outstanding WR";
      for (int i = 0; i < count; ++i) {
        const auto& entry = entries[i];
        if (entry.status != IBV_WC_SUCCESS) teardown_safe = false;
        CHECK_HOST(entry.status == IBV_WC_SUCCESS)
            << "mlx5 completion failed: " << ibv_wc_status_str(entry.status)
            << " vendor_err=" << entry.vendor_err;
        if (expected_receives == 0) {
          ++sends;
        } else if (entry.opcode == IBV_WC_RDMA_WRITE) {
          ++sends;
        } else if (entry.opcode == IBV_WC_RECV_RDMA_WITH_IMM) {
          CHECK_HOST(entry.wc_flags & IBV_WC_WITH_IMM) << "receive completion without immediate";
          CHECK_HOST(ntohl(entry.imm_data) == immediate)
              << "receive completion belongs to another exchange";
          ++receives;
        } else {
          CHECK_HOST(false) << "unexpected mlx5 completion opcode " << entry.opcode;
        }
      }
      CHECK_HOST(sends <= expected_sends) << "too many mlx5 send completions";
      CHECK_HOST(receives <= expected_receives) << "too many mlx5 receive completions";
      if (count == 0 && ++empty_polls == 1024) {
        CHECK_HOST(std::chrono::steady_clock::now() < deadline)
            << "timed out waiting for mlx5 completions";
        empty_polls = 0;
      }
    }
  }

  mlx5dv_mkey* CreateMkey() {
    mlx5dv_mkey_init_attr attr{};
    attr.pd = pd;
    attr.create_flags = MLX5DV_MKEY_INIT_ATTR_FLAGS_INDIRECT;
    attr.max_entries = 2;
    auto* result = mlx5dv_create_mkey(&attr);
    CHECK_HOST(result != nullptr) << "mlx5dv_create_mkey failed: " << std::strerror(errno);
    return result;
  }

  // An interleaved layout: `rows` runs of `width` bytes, `skip` bytes apart,
  // starting at `address`. The MKey keeps its rkey across reconfiguration, so
  // a rebind is one local UMR post and needs no collective.
  void ConfigureMkey(int peer, mlx5dv_mkey* mkey, uint32_t access, uint64_t address, uint32_t width,
                     uint32_t skip, uint32_t rows, uint32_t lkey) {
    mlx5dv_mkey_conf_attr config{};
    mlx5dv_mr_interleaved layout{};
    layout.addr = address;
    layout.bytes_count = width;
    layout.bytes_skip = skip;
    layout.lkey = lkey;
    ibv_wr_start(qpxs[peer]);
    qpxs[peer]->wr_id = NewWrId(peer, false);
    qpxs[peer]->wr_flags = IBV_SEND_INLINE | IBV_SEND_SIGNALED;
    mlx5dv_wr_mkey_configure(mlx5_qpxs[peer], mkey, 2, &config);
    mlx5dv_wr_set_mkey_access_flags(mlx5_qpxs[peer], access);
    mlx5dv_wr_set_mkey_layout_interleaved(mlx5_qpxs[peer], rows, 1, &layout);
    const int status = ibv_wr_complete(qpxs[peer]);
    if (status == 0) ++outstanding_send_wrs[peer];
    CHECK_HOST(status == 0) << "configure interleaved MKey failed: " << std::strerror(status);
  }

  void PostWrite(int peer, uint32_t local_key, uint64_t local_address, uint32_t bytes,
                 uint32_t remote_key, uint64_t remote_address, uint32_t immediate) {
    auto* qp = qpxs[peer];
    ibv_wr_start(qp);
    qp->wr_id = NewWrId(peer, false);
    qp->wr_flags = IBV_SEND_SIGNALED;
    ibv_wr_rdma_write_imm(qp, remote_key, remote_address, htonl(immediate));
    ibv_wr_set_sge(qp, local_key, local_address, bytes);
    const int status = ibv_wr_complete(qp);
    if (status == 0) ++outstanding_send_wrs[peer];
    CHECK_HOST(status == 0) << "post RDMA write failed: " << std::strerror(status);
  }

  void PostReceive(int peer) {
    ibv_recv_wr request{};
    request.wr_id = NewWrId(peer, true);
    ibv_recv_wr* bad = nullptr;
    const int status = ibv_post_recv(qps[peer], &request, &bad);
    if (status == 0) ++outstanding_recv_wrs[peer];
    CHECK_HOST(status == 0) << "post RDMA receive failed: " << std::strerror(status);
  }

  void Connect(const Array<int64_t>& flat) {
    EnsureHealthy();
    CHECK_HOST(!connected) << "RDMA Ulysses transport is already connected";
    CHECK_HOST(flat.size() == static_cast<size_t>(world_size) * sizeof(GroupWire))
        << "invalid group metadata length";
    for (int peer = 0; peer < world_size; ++peer) {
      peers[peer] = DecodeAt<GroupWire>(flat, peer * sizeof(GroupWire));
    }
    RetireOnFailure([&] {
      // Re-check the GID and MTU the metadata froze before the QPs consume them.
      ibv_port_attr current_port{};
      ibv_gid current_gid{};
      ValidatePlannedGid(&current_port, &current_gid);
      CHECK_HOST(current_port.active_mtu == local.mtu) << "port MTU changed after metadata exchange";
      CHECK_HOST(std::memcmp(current_gid.raw, local.gid, sizeof(local.gid)) == 0)
          << "GID changed after metadata exchange";
      for (int peer = 0; peer < world_size; ++peer) {
        auto* qp = qps[peer];
        if (qp == nullptr) continue;
        ibv_qp_attr attr{};
        attr.qp_state = IBV_QPS_INIT;
        attr.pkey_index = 0;
        attr.port_num = kPort;
        attr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;
        CheckVerbs(
            ibv_modify_qp(qp, &attr,
                          IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS),
            "QP RESET->INIT");
        attr = {};
        attr.qp_state = IBV_QPS_RTR;
        attr.path_mtu = static_cast<ibv_mtu>(std::min(local.mtu, peers[peer].mtu));
        attr.dest_qp_num = peers[peer].qpn[rank];
        attr.rq_psn = peers[peer].psn[rank];
        attr.max_dest_rd_atomic = 1;
        attr.min_rnr_timer = 12;
        attr.ah_attr.is_global = 1;
        attr.ah_attr.port_num = kPort;
        std::memcpy(&attr.ah_attr.grh.dgid, peers[peer].gid, 16);
        attr.ah_attr.grh.sgid_index = gid_index;
        attr.ah_attr.grh.hop_limit = 64;
        CheckVerbs(
            ibv_modify_qp(qp, &attr,
                          IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
                              IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER),
            "QP INIT->RTR");
        attr = {};
        attr.qp_state = IBV_QPS_RTS;
        attr.timeout = 18;
        attr.retry_cnt = 7;
        attr.rnr_retry = 7;
        attr.sq_psn = local.psn[peer];
        attr.max_rd_atomic = 1;
        CheckVerbs(ibv_modify_qp(qp, &attr,
                                 IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
                                     IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC),
                   "QP RTR->RTS");
      }
    });
    connected = true;
  }
};

inline void Slot::Disconnect() {
  if (imports_closed) return;
  connected = false;
  cudaError_t first_error = cudaSuccess;
  for (auto*& pointer : peer_signals) {
    if (pointer == nullptr || pointer == signals) continue;
    const cudaError_t status = cudaIpcCloseMemHandle(pointer);
    if (status == cudaSuccess) {
      pointer = nullptr;
    } else if (first_error == cudaSuccess) {
      first_error = status;
    }
  }
  if (first_error == cudaSuccess) {
    imports_closed = true;
  } else {
    CheckCuda(first_error, "cudaIpcCloseMemHandle(signals)");
  }
}

inline void Slot::Release() {
  if (released) return;
  CHECK_HOST(imports_closed) << "disconnect peer imports before releasing a slot";
  ScopedCudaDevice device_guard(transport == nullptr ? 0 : transport->device);
  for (auto*& mkey : destination_mkeys) {
    if (mkey == nullptr) continue;
    const int status = mlx5dv_destroy_mkey(mkey);
    if (status == 0) mkey = nullptr;
    CheckVerbs(status, "mlx5dv_destroy_mkey(output)");
  }
  // MRs and MKeys reference the allocations; they go first.
  if (landing_mr != nullptr) {
    const int status = ibv_dereg_mr(landing_mr);
    if (status == 0) landing_mr = nullptr;
    CheckVerbs(status, "ibv_dereg_mr(landing)");
  }
  if (output_mr != nullptr) {
    const int status = ibv_dereg_mr(output_mr);
    if (status == 0) output_mr = nullptr;
    CheckVerbs(status, "ibv_dereg_mr(output)");
  }
  for (void** pointer : {&landing, &output, reinterpret_cast<void**>(&epoch_device),
                         reinterpret_cast<void**>(&signals)}) {
    if (*pointer == nullptr) continue;
    const cudaError_t status = cudaFree(*pointer);
    if (status == cudaSuccess) *pointer = nullptr;
    CheckCuda(status, "cudaFree(slot)");
  }
  released = true;
}

inline Slot::~Slot() {
  bool safe = imports_closed && (transport == nullptr || transport->TeardownSafe());
  if (safe) {
    try {
      Release();
      return;
    } catch (...) {
      safe = false;
    }
  }
  std::fprintf(stderr,
               "RDMA Ulysses: refusing unsafe slot teardown for output %p; native resources are "
               "intentionally leaked\n",
               output);
  if (transport != nullptr) transport->unsafe_release = true;
}

// ---------------------------------------------------------------- binding --

inline Transport* AsTransport(int64_t handle) {
  auto* result = reinterpret_cast<Transport*>(handle);
  CHECK_HOST(result != nullptr) << "null RDMA Ulysses handle";
  return result;
}

inline Slot* FindSlot(Transport* transport, int64_t index) {
  CHECK_HOST(index >= 0 && index < static_cast<int64_t>(transport->slots.size()))
      << "unknown RDMA Ulysses slot " << index;
  return transport->slots[index].get();
}

// Validate the operand pair and derive the per-peer transfer.
inline Geometry DescribeGeometry(int mode, TensorView input, TensorView output, int world_size) {
  Geometry geometry{};
  geometry.mode = mode;
  CHECK_HOST(input.IsContiguous() && output.IsContiguous()) << "RDMA Ulysses operands must be contiguous";
  CHECK_HOST(input.dtype().bits == output.dtype().bits && input.dtype().lanes == output.dtype().lanes)
      << "input/output element size mismatch";
  const int64_t element = ElementBytes(input);
  CHECK_HOST(element == 1 || element == 2 || element == 4)
      << "RDMA Ulysses moves 1-, 2- and 4-byte elements, got " << element << " bytes";
  if (mode == kModeChunks) {
    CHECK_HOST(input.ndim() == 2 && output.ndim() == 2) << "chunk operands must be [world_size, C]";
    CHECK_HOST(input.size(0) == world_size && output.size(0) == world_size)
        << "chunk operands need one chunk per peer";
    CHECK_HOST(input.size(1) == output.size(1) && input.size(1) > 0) << "chunk length mismatch";
    geometry.rows = 1;
    geometry.width = input.size(1) * element;
    geometry.pitch = geometry.width;
  } else {
    CHECK_HOST(mode == kModeGather) << "unhandled RDMA Ulysses mode " << mode;
    CHECK_HOST(input.ndim() == 3 && output.ndim() == 3)
        << "gather operands must be [S_global, H_local, D] and [S_local, H, D]";
    const int64_t s_global = input.size(0), h_local = input.size(1), dim = input.size(2);
    CHECK_HOST(s_global > 0 && s_global % world_size == 0)
        << "gather sequence " << s_global << " must split evenly over " << world_size;
    CHECK_HOST(output.size(0) == s_global / world_size && output.size(1) == h_local * world_size &&
               output.size(2) == dim)
        << "gather output shape does not match its input";
    geometry.rows = s_global / world_size;
    geometry.width = h_local * dim * element;
    geometry.pitch = geometry.width * world_size;
    CHECK_HOST(geometry.pitch <= kMaxInterleavedStride)
        << "H * D * element_size must stay within " << kMaxInterleavedStride
        << " bytes for the interleaved MKey, got " << geometry.pitch;
  }
  geometry.payload = geometry.rows * geometry.width;
  geometry.total = geometry.payload * world_size;
  CHECK_HOST(geometry.payload > 0 && geometry.payload <= UINT32_MAX)
      << "per-peer payload exceeds the mlx5 WR limit";
  return geometry;
}

// Point every peer's destination MKey at the bound gather geometry over the
// output allocation: peer p's rows start at head offset p * width.
inline void ConfigureDestinationMkeys(Transport* transport, Slot* slot, const Geometry& geometry) {
  const uint32_t access = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;
  int configured = 0;
  for (int peer = 0; peer < transport->world_size; ++peer) {
    if (peer == transport->rank) continue;
    CHECK_HOST(slot->destination_mkeys[peer] != nullptr) << "missing destination MKey";
    const uint64_t address = reinterpret_cast<uint64_t>(slot->output) + uint64_t(peer) * geometry.width;
    transport->ConfigureMkey(peer, slot->destination_mkeys[peer], access, address,
                             static_cast<uint32_t>(geometry.width),
                             static_cast<uint32_t>(geometry.pitch - geometry.width),
                             static_cast<uint32_t>(geometry.rows), slot->output_mr->lkey);
    ++configured;
  }
  transport->Poll(configured);
}

inline void BindGeometry(Transport* transport, Slot* slot, const Geometry& geometry) {
  CHECK_HOST(geometry.total <= slot->capacity)
      << "operand of " << geometry.total << " bytes exceeds the slot capacity " << slot->capacity;
  if (slot->has_geometry && slot->bound == geometry) return;
  slot->bound = geometry;
  slot->has_geometry = true;
  if (slot->mode == kModeGather) ConfigureDestinationMkeys(transport, slot, geometry);
}

// The NIC reads the landing buffer. An operand built there is used in place;
// anything else is staged with one copy. Partial overlap is refused because
// the NIC would read one set of bytes and the self-copy another.
inline const void* BindInput(Slot* slot, TensorView input, int64_t bytes, cudaStream_t current) {
  if (input.data_ptr() == slot->landing) return slot->landing;
  const auto* landing_begin = static_cast<const char*>(slot->landing);
  const auto* source_begin = static_cast<const char*>(input.data_ptr());
  CHECK_HOST(source_begin + bytes <= landing_begin || source_begin >= landing_begin + slot->capacity)
      << "input overlaps the landing buffer without being it";
  CheckCuda(cudaMemcpyAsync(slot->landing, input.data_ptr(), static_cast<size_t>(bytes),
                            cudaMemcpyDeviceToDevice, current),
            "cudaMemcpyAsync(landing)");
  return slot->landing;
}

// This rank's own chunk never touches the NIC.
inline void SelfCopy(Transport* transport, Slot* slot, const void* source, const Geometry& geometry,
                     cudaStream_t current) {
  const auto* src = static_cast<const uint8_t*>(source);
  auto* dst = static_cast<uint8_t*>(slot->output);
  const int64_t rank = transport->rank;
  if (geometry.mode == kModeChunks) {
    CheckCuda(cudaMemcpyAsync(dst + rank * geometry.width, src + rank * geometry.width,
                              static_cast<size_t>(geometry.width), cudaMemcpyDeviceToDevice, current),
              "cudaMemcpyAsync(own chunk)");
    return;
  }
  CheckCuda(cudaMemcpy2DAsync(dst + rank * geometry.width, static_cast<size_t>(geometry.pitch),
                              src + rank * geometry.payload, static_cast<size_t>(geometry.width),
                              static_cast<size_t>(geometry.width), static_cast<size_t>(geometry.rows),
                              cudaMemcpyDeviceToDevice, current),
            "cudaMemcpy2DAsync(own heads)");
}

// ---------------------------------------------------------------- exports --

/*!
 * \brief Open the NIC and create one RC queue pair per peer.
 * \return (handle, this rank's GroupWire bytes to all-gather and feed to connect()).
 */
inline Tuple<int64_t, Array<int64_t>> init(int64_t rank, int64_t world_size, int64_t device,
                                           String nic_name, int64_t gid_index, int64_t timeout_ms) {
  auto* transport = new Transport(static_cast<int>(rank), static_cast<int>(world_size),
                                  static_cast<int>(device), std::string(nic_name),
                                  static_cast<int>(gid_index), timeout_ms);
  return Tuple<int64_t, Array<int64_t>>(reinterpret_cast<int64_t>(transport),
                                        Encode(transport->local));
}

inline void connect(int64_t handle, Array<int64_t> flat) {
  auto* transport = AsTransport(handle);
  ScopedCudaDevice device_guard(transport->device);
  transport->Connect(flat);
}

/*!
 * \brief Allocate and register one exchange slot at a fixed byte capacity.
 * \return (slot index, output bytes, landing bytes, this rank's SlotWire to all-gather).
 *
 * Both tensors are uint8 views of transport-owned cudaMalloc memory; callers view
 * a prefix per operand. They stay valid until dispose().
 */
inline Tuple<int64_t, Tensor, Tensor, Array<int64_t>> register_slot(int64_t handle, int64_t mode,
                                                                     int64_t capacity_bytes) {
  auto* transport = AsTransport(handle);
  ScopedCudaDevice device_guard(transport->device);
  transport->EnsureHealthy();
  CHECK_HOST(transport->connected) << "connect the transport before registering slots";
  CHECK_HOST(mode == kModeGather || mode == kModeChunks) << "mode must be 1 (gather) or 2 (chunks)";
  CHECK_HOST(capacity_bytes > 0 && capacity_bytes % 128 == 0)
      << "slot capacity must be a positive multiple of 128 bytes";

  auto slot = std::make_unique<Slot>();
  slot->transport = transport;
  slot->mode = static_cast<int>(mode);
  slot->capacity = capacity_bytes;
  CheckCuda(cudaMalloc(&slot->output, static_cast<size_t>(capacity_bytes)), "cudaMalloc(output)");
  CheckCuda(cudaMalloc(&slot->landing, static_cast<size_t>(capacity_bytes)), "cudaMalloc(landing)");
  const size_t signal_bytes = 2 * transport->world_size * sizeof(uint64_t);
  CheckCuda(cudaMalloc(reinterpret_cast<void**>(&slot->signals), signal_bytes), "cudaMalloc(signals)");
  CheckCuda(cudaMemset(slot->signals, 0, signal_bytes), "cudaMemset(signals)");
  CheckCuda(cudaMalloc(reinterpret_cast<void**>(&slot->epoch_device), sizeof(uint64_t)),
            "cudaMalloc(epoch)");
  CheckCuda(cudaMemset(slot->epoch_device, 0, sizeof(uint64_t)), "cudaMemset(epoch)");
  slot->peer_signals[transport->rank] = slot->signals;

  transport->RetireOnFailure([&] {
    slot->output_mr = RegisterGpuMr(transport->pd, slot->output, static_cast<size_t>(capacity_bytes),
                                    IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                                        IBV_ACCESS_REMOTE_READ);
    slot->landing_mr = RegisterGpuMr(transport->pd, slot->landing,
                                     static_cast<size_t>(capacity_bytes), IBV_ACCESS_LOCAL_WRITE);
    if (mode == kModeGather) {
      for (int peer = 0; peer < transport->world_size; ++peer) {
        if (peer != transport->rank) slot->destination_mkeys[peer] = transport->CreateMkey();
      }
    }
  });

  SlotWire wire{};
  wire.address = reinterpret_cast<uint64_t>(slot->output);
  wire.rkey = slot->output_mr->rkey;
  CheckCuda(cudaIpcGetMemHandle(&wire.signal_ipc, slot->signals), "cudaIpcGetMemHandle(signals)");
  for (int peer = 0; peer < transport->world_size; ++peer) {
    if (slot->destination_mkeys[peer] != nullptr) {
      wire.destination_rkey[peer] = slot->destination_mkeys[peer]->rkey;
    }
  }

  Tensor output = ViewDeviceBytes(slot->output, capacity_bytes, transport->device);
  Tensor landing = ViewDeviceBytes(slot->landing, capacity_bytes, transport->device);
  transport->slots.push_back(std::move(slot));
  return Tuple<int64_t, Tensor, Tensor, Array<int64_t>>(
      static_cast<int64_t>(transport->slots.size() - 1), output, landing, Encode(wire));
}

/*! \brief Import every peer's signal slots and remote keys for one slot. */
inline void connect_slot(int64_t handle, int64_t index, Array<int64_t> flat) {
  auto* transport = AsTransport(handle);
  ScopedCudaDevice device_guard(transport->device);
  transport->EnsureHealthy();
  auto* slot = FindSlot(transport, index);
  CHECK_HOST(!slot->connected) << "slot is already connected";
  CHECK_HOST(flat.size() == static_cast<size_t>(transport->world_size) * sizeof(SlotWire))
      << "invalid slot metadata length";
  std::array<SlotWire, kMaxWorld> peers{};
  std::array<uint64_t*, kMaxWorld> signals{};
  for (int peer = 0; peer < transport->world_size; ++peer) {
    peers[peer] = DecodeAt<SlotWire>(flat, peer * sizeof(SlotWire));
  }
  signals[transport->rank] = slot->signals;
  try {
    for (int peer = 0; peer < transport->world_size; ++peer) {
      if (peer == transport->rank) continue;
      // The epoch barrier is a GPU kernel writing peer memory, so every rank
      // pair needs CUDA peer access even though the payload rides the NIC.
      CheckCuda(cudaIpcOpenMemHandle(reinterpret_cast<void**>(&signals[peer]),
                                     peers[peer].signal_ipc, cudaIpcMemLazyEnablePeerAccess),
                "cudaIpcOpenMemHandle(signals)");
    }
  } catch (...) {
    const std::exception_ptr original = std::current_exception();
    slot->peer_signals = signals;
    slot->imports_closed = false;
    try {
      slot->Disconnect();
    } catch (...) {
    }
    std::rethrow_exception(original);
  }
  slot->peers = peers;
  slot->peer_signals = signals;
  slot->connected = true;
  slot->imports_closed = false;
}

/*!
 * \brief Run one exchange on the caller's stream; blocks the host until every
 *        peer's bytes have landed.
 *
 * `input` is either the slot's landing buffer (viewed as the operand, zero copy)
 * or any contiguous operand (staged once); `output` must be a prefix view of the
 * slot's output buffer. Mode 2: [W, C] -> [W, C]; mode 1: [S_global, H_local, D]
 * -> [S_local, H, D].
 */
inline void exchange(int64_t handle, int64_t index, TensorView input, TensorView output) {
  auto* transport = AsTransport(handle);
  ScopedCudaDevice device_guard(transport->device);
  transport->EnsureHealthy();
  auto* slot = FindSlot(transport, index);
  CHECK_HOST(slot->connected) << "slot is not connected";
  CHECK_HOST(input.device().device_type == kDLCUDA && input.device().device_id == transport->device)
      << "input is on the wrong device";
  CHECK_HOST(output.device().device_type == kDLCUDA && output.device().device_id == transport->device)
      << "output is on the wrong device";
  CHECK_HOST(output.data_ptr() == slot->output) << "output must be a prefix view of the slot output";
  const Geometry geometry = DescribeGeometry(slot->mode, input, output, transport->world_size);
  const cudaStream_t current = host::LaunchKernel::resolve_device(input.device());

  try {
    // Rebinds and staging stay inside the failure envelope so a local setup
    // failure publishes the abort instead of making peers wait out the timeout.
    BindGeometry(transport, slot, geometry);
    const void* source = BindInput(slot, input, geometry.total, current);
    const auto payload = static_cast<uint32_t>(geometry.payload);
    const uint64_t source_base = reinterpret_cast<uint64_t>(source);

    const uint64_t exchange_tag = ++slot->tag;
    const uint32_t immediate = static_cast<uint32_t>(exchange_tag & 0x3fffffffU) | 0x80000000U;
    // A completed opening barrier proves every rank has consumed its previous
    // output and that no sticky abort is pending.
    transport->RunBarrier(slot, current, true);

    const int expected = transport->world_size - 1;
    for (int peer = 0; peer < transport->world_size; ++peer) {
      if (peer != transport->rank) transport->PostReceive(peer);
    }
    for (int peer = 0; peer < transport->world_size; ++peer) {
      if (peer == transport->rank) continue;
      const uint64_t local_address = source_base + uint64_t(peer) * geometry.payload;
      if (geometry.mode == kModeChunks) {
        CHECK_HOST(slot->peers[peer].address != 0 && slot->peers[peer].rkey != 0)
            << "peer " << peer << " has no registered output";
        transport->PostWrite(peer, slot->landing_mr->lkey, local_address, payload,
                             slot->peers[peer].rkey,
                             slot->peers[peer].address + uint64_t(transport->rank) * geometry.payload,
                             immediate);
      } else {
        // The interleaved destination MKey is addressed from zero; the peer
        // configured it over its own output at our head offset.
        const uint32_t remote_key = slot->peers[peer].destination_rkey[transport->rank];
        CHECK_HOST(remote_key != 0) << "peer " << peer << " has no destination MKey for us";
        transport->PostWrite(peer, slot->landing_mr->lkey, local_address, payload, remote_key, 0,
                             immediate);
      }
    }
    SelfCopy(transport, slot, source, geometry, current);

    // Closing is a success vote: only after every completion is verified and
    // GPUDirect writes are visible to the GPU.
    transport->Poll(expected, expected, immediate);
    if (transport->write_ordering < cudaGPUDirectRDMAWritesOrderingOwner) {
      CheckCuda(cudaDeviceFlushGPUDirectRDMAWrites(cudaFlushGPUDirectRDMAWritesTargetCurrentDevice,
                                                   cudaFlushGPUDirectRDMAWritesToOwner),
                "cudaDeviceFlushGPUDirectRDMAWrites");
    }
    transport->RunBarrier(slot, current, false);
  } catch (...) {
    transport->AbortAndQuiesce(slot, current);
    throw;
  }
}

inline int64_t teardown_safe(int64_t handle) {
  return AsTransport(handle)->TeardownSafe() ? 1 : 0;
}

/*! \brief Close every slot's peer imports; every rank must finish this before any rank disposes. */
inline void disconnect(int64_t handle) {
  auto* transport = AsTransport(handle);
  ScopedCudaDevice device_guard(transport->device);
  CHECK_HOST(transport->TeardownSafe())
      << "cannot disconnect after unbounded native GPU work; terminate the process";
  for (auto& slot : transport->slots) slot->Disconnect();
}

inline void dispose(int64_t handle) {
  auto* transport = AsTransport(handle);
  ScopedCudaDevice device_guard(transport->device);
  CHECK_HOST(transport->TeardownSafe())
      << "cannot dispose after unbounded native GPU work; terminate the process";
  CHECK_HOST(!transport->unsafe_release) << "transport has an unrecoverable teardown ledger";
  for (const auto& slot : transport->slots) {
    CHECK_HOST(slot->imports_closed) << "disconnect every rank before disposing the transport";
  }
  for (auto& slot : transport->slots) slot->Release();
  transport->slots.clear();
  CHECK_HOST(!transport->unsafe_release) << "teardown lost a native resource ledger";
  delete transport;
}

/*! \brief Publish this rank's sticky abort for one slot; test-only fault injection. */
inline void publish_abort_for_test(int64_t handle, int64_t index) {
  auto* transport = AsTransport(handle);
  ScopedCudaDevice device_guard(transport->device);
  auto* slot = FindSlot(transport, index);
  CHECK_HOST(slot->connected) << "slot is not connected";
  CHECK_HOST(transport->TryPublishAbort(slot)) << "could not publish the abort";
  transport->failed = true;
}

}  // namespace sglang::rdma_ulysses
