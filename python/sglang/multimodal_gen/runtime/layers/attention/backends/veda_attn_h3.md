# Veda block-sparse attention for MiniMax-H3 (`veda_attn`)

Veda ([Miowtion](https://github.com/veda-sparse/Miowtion)) permutes the video tokens of the packed
sequence into 128-token 3D tiles, one tile shape per head chosen by a searched plan, scores every
(query tile, key tile) pair with a trained per-head predictor, keeps the top 10% of key tiles per
query tile and runs a FlashAttention-4 CuTe block-sparse kernel on them. Text, audio and condition
rows are global rows (dense both ways). The predictor weights and the tile plans ship together in one
safetensors bundle so they cannot be mispaired. This backend, `veda_attn`, plugs that into the H3 DiT.

## Pins

| Dependency | Version | Why |
|---|---|---|
| Miowtion | commit `92c9748478fcba3884d56a56c66abf1bc1221909` | first commit with both fp8 bundles (`3416cd0849`) and SM120 block sparsity (`75d6629d1e`); `veda_runtime.check_contract()` verifies the API at resolve time |
| `flash-attn-4` (pip) | `4.0.0b32` exactly | Miowtion's vendored SM8x / SM120 block-sparse patch hash-checks the installed `flash_attn.cute` modules and must be installed before anything imports them |
| predictor bundle | `Veda-Sparse/Minimax-H3-T2VA-Veda-8NFE-600Step-Preview/minimax_h3_t2va_veda_8nfe_600step_preview_fp8.safetensors` (275 MB) | fp8 storage with per-head scales, dequantised to bf16 on load; 12 tile plans; keep ratio 0.1 |

```bash
source /workspace/sgl-env/bin/activate
pip install --no-deps 'flash-attn-4==4.0.0b32'
pip install --no-deps 'miowtion @ git+https://github.com/veda-sparse/Miowtion@92c9748478fcba3884d56a56c66abf1bc1221909'
hf download Veda-Sparse/Minimax-H3-T2VA-Veda-8NFE-600Step-Preview --local-dir /workspace/models/veda/Minimax-H3-T2VA-Veda-8NFE-600Step-Preview
python -c "import torch; from miowtion.kernels import fa4; print(fa4.available(torch.device('cuda')), fa4._patch_error)"   # True None
```

sglang's own attention uses its vendored copy `sglang.kernels.ops.attention.flash_attn.cute`, so the pip
`flash-attn-4` upgrade does not touch it. Keep `SGLANG_INKLING_FA4_USE_PIP` unset: with it sglang would
import the pip `flash_attn.cute` unpatched and Miowtion refuses to patch afterwards (the pipeline config
rejects that combination on SM8x / SM120). `subblock_sparse_sage_sm120` and `veda_attn` coexist in one
tree; each has its own enum value and resolver and is selected per run by flag.

## Run

The predictor was distilled against the Turbo LoRA 8-step trajectory, so serve it with that adapter, 9
scheduler steps (8 forwards), no CFG, and one of the 12 geometries the bundle has plans for. The Turbo
LoRA carries AdaLN deltas, which `--minimax-h3-adaln-online` cannot apply, hence tp2 x ul4 instead of
the tp1 x ul8 recipes.

```bash
sglang serve --model-path /workspace/models/MiniMax-H3 --model-variant fl2va \
  --num-gpus 8 --tp-size 2 --ulysses-degree 4 --ring-degree 1 \
  --lora-path /workspace/models/MiniMax-H3-Turbo-Lora \
  --lora-weight-name minimax_h3_turbo_v4_step600_ema.safetensors --lora-scale 1.0 --lora-merge-mode auto \
  --component-attention-backends transformer=veda_attn \
  --attention-backend-config veda_bundle=/workspace/models/veda/Minimax-H3-T2VA-Veda-8NFE-600Step-Preview/minimax_h3_t2va_veda_8nfe_600step_preview_fp8.safetensors \
  --warmup-mode request --warmup-resolutions 1344x768 --warmup-steps 9
# request: num_inference_steps 9, flow_shift 12.0, audio_flow_shift 3.0, target 16:9 / short edge 768 / 5.0 s
```

`--attention-backend-config` keys (`k=v` or JSON; the `k=v` parser splits on commas, so lists need JSON):

| Key | Default | Meaning |
|---|---|---|
| `veda_bundle` | required | predictor bundle path |
| `veda_keep_ratio` | the bundle's (0.1) | block budget in (0, 1]; 1.0 keeps every block |
| `veda_dense_first_n_steps` | 0 | 0-based denoising loop steps that stay dense |
| `veda_dense_layers` | none | DiT blocks that stay dense, e.g. `{"veda_dense_layers": [0, 1]}` |
| `veda_collect_mib` | 256 | bound on one tile-ordered q / k / v / out copy; more MiB means more heads per kernel launch, same result (Miowtion uses 2048 on 96 GB cards) |
| `veda_plan_fallback` | `none` | `none` serves only token grids with an exact plan; `select` applies Miowtion's rule (same aspect, nearest latent_t; else the mirrored aspect's transposed plan) |

A global `--attention-backend veda_attn` also works: the text encoder switches to `torch_sdpa`
automatically and non-DiT layers run dense inside the backend.

Exact grids of the released bundle, as `(latent_t, tokens_h, tokens_w)`: 16:9 `(t, 24, 42)`,
9:16 `(t, 42, 24)`, 4:3 `(t, 24, 32)`, 1:1 `(t, 24, 24)` with `t` in 37 / 72 / 102, i.e. short edge
768 and 5.167 s / 10.125 s / 14.375 s (`duration_seconds: 5.0` in sglang also yields 124 frames and
`t = 37`; 5.2 s jumps to 141 frames and `t = 42`, which has no plan). The predictor was trained on the
5.167 s clips only.

## How it maps onto the H3 attention core

- The core calls `impl.forward_varlen(q, k, v, cu_seqlens=(0, used, seq_len), max_seqlen=used)` after
  the Ulysses all-to-all; q / k / v are `[seq_len, H_local, 128]` bf16 views with unit stride along
  head_dim, post QK-norm and RoPE. The backend passes them to Miowtion's Triton gather as they are.
- t2va packs `[text | audio | video | pad]`, video raster-ordered (t, h, w), padded to 64 rows;
  `video_start = used - T*H*W`. This is the layout the predictor was trained on. Only the video span
  is tiled; every other real row is global.
- Layer index comes from the impl prefix (`^blocks\.(\d+)\.`); the token refiner is dense. Step index
  is `get_forward_context().current_timestep` (0-based).
- **Head parallelism.** Each rank owns the model heads
  `tp_rank * heads_per_tp + ulysses_rank * H_local + j`. Upstream Miowtion attends all heads of a
  layer and uses the same head index for the q / k / v head axis and the predictor projection, so at
  load time `veda_runtime.load_local_bundle` slices the bundle to this rank's heads (predictor tensors
  along dim 0, every plan's `head_shape` row) and the unchanged `SparseStudent` runs on the local
  heads. Every stage is per head (tile gather, pooled features, projection, top-k, block-sparse
  kernel, scatter), so the result is bit-identical to a full-head run; the resident predictor shrinks
  from 0.51 GiB to 69 MB at 7 local heads.
- Padding rows `[used, seq_len)` come back as zeros. Dense fallbacks (refiner, non-varlen calls,
  foreign metadata, dense steps, dense layers, layers beyond the bundle) use sglang's own SDPA
  backend, so they are numerically the `torch_sdpa` baseline.

## Files

| File | Role |
|---|---|
| `runtime/layers/attention/backends/veda_runtime.py` | the only module importing `miowtion`: pins, per-rank bundle slicing, exact-grid plan lookup, `PackedLayout` for sglang's packed sequence, `check_contract()`, `fa4_status()` |
| `runtime/layers/attention/backends/veda_attn_h3.py` | the backend: request metadata, per-process runtime (bundle + cached students per packed layout), `forward_varlen` |
| `runtime/platforms/interface.py`, `runtime/platforms/cuda.py` | `VEDA_ATTN` enum (sparse) and the resolver (contract check, capability 8.x / 9.x / 10.x / 12.x, FA4 block sparsity present) |
| `.../minimax_h3/stages/denoising.py` | `_build_veda_attn_metadata`: video span, `used`, `seq_len` from the packed layout; ref2va rejected |
| `configs/pipeline_configs/minimax_h3.py` | rejects ring, torch.compile, BCG; validates the config keys and the FA4 env |
| `runtime/server_args/server_args.py` | text encoder auto `torch_sdpa` under a global `veda_attn` |
| `test/unit/test_veda_attn_h3.py`, `test/unit/test_veda_runtime.py` | CPU tests: dispatch and fallbacks; slicing exactness against the full-head student, layout equivalence, fp8 round trip, plan lookup, API contract (the Miowtion half skips when it is not installed) |
| `test/unit/manual/veda_attn_h3_gpu_smoke.py` | one-GPU smoke on the real kernels |

## Limits

- Trained on t2va with the Turbo 8-step teacher; fl2va keyframes run as global rows but are not
  validated; ref2va is rejected.
- SM120 runs Miowtion's patched SM80 forward (forward only; inference needs nothing else). SM100 uses
  Miowtion's `q_stage = 1` hook, which Miowtion has not validated on B200.
- No ring attention, torch.compile or breakable CUDA graph (same as VSA-H3).
- Log lines that prove the sparse path: `Veda attention: bundle ... sliced to model heads [a, b) of 56`
  (one per rank), `... -> plan 16x9_t37 (exact, ...)` per packed layout, and
  `Veda attention active: blocks.0.attn at step 0, 7 local heads (model heads a..b)` once per process.

## Differences from the private-fork version

The first version of this backend ran against an unpublished Miowtion fork that added
`PlanTable.select_grid`, `ClipTiling.for_target` and `SparseStudent.forward_heads`. Upstream has none
of them; this version keeps to upstream's public API (`PlanTable.select` / exact grid lookup,
`ClipTiling(layout, config, device)`, `SparseStudent(q, k, v, layer_index)`) and moves the head
mapping into the per-rank bundle slice. Measured results of the fork version (2026-09-26, 8x RTX PRO
5000, tp2 x ul4, Turbo LoRA, 8 forwards): denoise-stage speedups over dense SDPA of 1.24-1.26x on
5 s clips and 1.72x on 10 s clips with warmup; PSNR against the dense run 14-20 dB, which is a
trajectory distance, not an image-quality score. The upstream-API version computes the same tile
selections with the same kernel and is expected to reproduce those outputs; verify with
`test/unit/test_veda_runtime.py` (bitwise against the full-head student) and an on-box A/B.
