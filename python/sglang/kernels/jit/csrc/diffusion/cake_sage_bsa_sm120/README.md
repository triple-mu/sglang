# Cake SM120 Sage block-sparse attention

`vendor/cake_sage_block_sparse_attention_939d22c4b83f8f4c938f_kernel.cu` is the
generated CUDA source of one specialization of flashinfer PR #4951 ("Cake SM120
sage block-sparse"), taken byte-for-byte from
[flashinfer-ai/flashinfer](https://github.com/flashinfer-ai/flashinfer) at commit
`8db743f1` (`csrc/cake_sage_block_sparse_attention/sm_120a/`). Apache-2.0; see
`vendor/LICENSE`. `vendor/MANIFEST.json` carries the upstream manifest record for
this module (launch geometry, compile flags, source hashes, provenance).

`entry.cuh` is the SGLang side: tensor validation, TMA descriptor encoding and
publication, and the host entry the JIT exports.
`python/sglang/kernels/ops/diffusion/attention/sage_block_sparse_sm120_cake.py`
compiles and calls it.

## What the kernel computes

QK-INT8 / PV-FP8 block-sparse attention over 64-token blocks, the SageAttention
quantization contract: Q is INT8 with one scale per 32-query group, K is INT8
(channel-mean-centred) with one scale per 64-key block, V is FP8 E4M3 with one
scale per channel stored `[B, H, D, S_k]` with the Sage 16-token permutation, P
is quantised to FP8 inside the kernel, the output is BF16 BHSD. MHA only, head
dimension 128, non-causal, no LSE. `q2k_block_index[b, h, q_block, :nums]`
lists the key blocks each query block attends to, in any order.

## Why exactly one specialization

The generator emits six modules keyed on `HAS_BLOCK_NUMS`, `BLOCK_SIZES_MODE`,
`FULL_K64_TILES`, `UNIFORM_NONEMPTY`, `CONTIGUOUS_BLOCK_INDICES`. The MiniMax-H3
sub-block router produces per-row block counts and unordered indices, so only
`(1, 0, 1, 0, 0)` is reachable; `entry.cuh` `static_assert`s those five macros so
a re-sync with a different module fails to compile instead of silently changing
the contract. `BLOCK_SIZES_MODE 0` is why `seqlen_k` must be a multiple of 64:
callers pad K/V to the block boundary and zero the padding rows.

## Descriptors and workspace

The kernel reads its four TMA descriptors (Q, K, V, O) from global memory behind
a `fence.proxy.tensormap` and does not build them itself. `entry.cuh` encodes them
with `cuTensorMapEncodeTiled` (fetched via `cudaGetDriverEntryPointByVersion`, so
the module links without `-lcuda`) over the *allocated* extents of the tensors,
publishes them into a caller-owned 512-byte device workspace, and remembers the
bytes last uploaded to each workspace: a repeat call with the same tensors is a
cache hit, a change is one stream-ordered 512-byte copy. The Python wrapper keeps
one workspace per device. Because that copy cannot be recorded, a CUDA graph must
warm the exact tensors before capture.

`seqlen_q` and `seqlen_k` are explicit arguments and may be smaller than the
allocated rows, so a resident staging buffer can be handed in directly and only
its live prefix is computed. Scales and index tables are sized by the live
lengths, matching the kernel's own indexing.

## Build inputs

`--use_fast_math` is part of the kernel's contract (upstream manifest
`compile_flags`); the module passes it explicitly because the JIT defaults do
not. No extra include paths or link flags. The JIT's default target for a 12.0
device (`sm_120f` under nvcc >= 12.9) compiles to the same SASS as the upstream
`sm_120a` build (verified by `cuobjdump -sass` diff, 2026-09-26).

## Re-syncing with upstream

Copy the `_kernel.cu` of the same specialization into `vendor/`, update the
sha256 in `vendor/MANIFEST.json` and in the Python wrapper, and re-run
`test/registered/kernels/ops/diffusion/test_sage_block_sparse_sm120.py`. The
binding `.cu` is not vendored; its hash is recorded in the manifest for
reference only.
