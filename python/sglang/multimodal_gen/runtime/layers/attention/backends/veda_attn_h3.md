# Veda 稀疏注意力接入 MiniMax-H3（分支 `feat/h3-veda`）

Veda（[Miowtion](https://github.com/veda-sparse/Miowtion)）：把视频 token 按每头一种 3D tile 形状重排成
128 token 的 tile，用训练好的逐头打分器给 (query tile, key tile) 打分，每个 query tile 只保留 10% 的
key tile，交给 FlashAttention-4 CuTe 的块稀疏 kernel；文本 / 音频 / 条件行是全局行，双向稠密。打分器权重
与 tile 方案打在一个 safetensors bundle 里，避免配错。本分支把它做成 H3 DiT 的一个 attention 后端
`veda_attn`。

## 怎么跑

```bash
# 环境：薄壳 venv 叠在 sgl-env 上（同 h3-unified/env/bootstrap.sh 的做法），另加两个 editable 和锁定版 FA4
python -m venv --system-site-packages .venv && echo /workspace/sgl-env/lib/python3.12/site-packages > .venv/lib/python3.12/site-packages/zzz-sgl-env-deps.pth
.venv/bin/pip install -e python --no-build-isolation --no-deps
.venv/bin/pip install -e /path/to/Miowtion --no-deps
.venv/bin/pip install --no-deps 'flash-attn-4 @ git+https://github.com/Dao-AILab/flash-attention@d15f1531a460ba456f41b01a774f33ab2db8febf#subdirectory=flash_attn/cute'

# 只让 DiT 走 Veda；text encoder / VAE 保持默认。bundle 与 keep ratio 从 --attention-backend-config 传
EP_SUPPRESS_NCCL_CHECK=1 sglang generate --model-path /path/MiniMax-H3 --config request.json \
  --num-gpus 8 --tp-size 2 --ulysses-degree 4 --ring-degree 1 --num-inference-steps 9 \
  --lora-path /path/MiniMax-H3-Turbo-Lora --lora-weight-name minimax_h3_turbo_v4_step600_ema.safetensors \
  --lora-scale 1.0 --lora-merge-mode auto \
  --component-attention-backends transformer=veda_attn \
  --attention-backend-config veda_bundle=/path/veda_predictor.safetensors \
  --output-file-path out.mp4 --perf-dump-path perf.json --save-output
```

`--attention-backend-config` 的键：`veda_bundle`（必填）、`veda_keep_ratio`（默认取 bundle 里的 0.1）、
`veda_dense_first_n_steps`（前 N 个去噪步稠密，默认 0）、`veda_dense_layers`（保持稠密的 block，默认无）。
全局 `--attention-backend veda_attn` 也可以：text encoder 会自动改成 torch_sdpa，VAE 等非 DiT 层在后端里
走稠密 SDPA。

## 改了什么

| 文件 | 改动 |
|---|---|
| `runtime/layers/attention/backends/veda_attn_h3.py`（新） | 后端、请求级元数据 `VedaAttentionMetadata`、每进程一份的 bundle 运行时（按打包布局缓存 `SparseStudent`）、`forward_varlen` |
| `runtime/platforms/interface.py` | 枚举 `VEDA_ATTN`，加入 `is_sparse` |
| `runtime/platforms/cuda.py` | `_VedaAttentionBackendResolver`：import Miowtion，用 `fa4.available()` 按设备判断 FA4 块稀疏是否可用 |
| `.../minimax_h3/stages/denoising.py` | `_build_veda_attn_metadata`：从 `packed["stream_layout"]["target_shape"]` 与 `cu_seqlens` 取视频 span、`used`、`seq_len`，经 forward context 交给后端；ref2va 拒绝 |
| `configs/pipeline_configs/minimax_h3.py` | 校验：拒绝 ring、torch.compile、BCG；`veda_bundle` 必填 |
| `runtime/server_args/server_args.py` | 全局选 `veda_attn` 时 text encoder 自动 torch_sdpa（与 laser_attn 同一规则） |
| `test/unit/test_veda_attn_h3.py`（新） | CPU 单测：注册、层号解析、稠密回退与 SDPA 逐位一致、head 映射、布局校验、stage 元数据 |

## 契约（后端依赖的 H3 行为）

- attention core 在 Ulysses all-to-all 之后调 `impl.forward_varlen(q, k, v, cu_seqlens=(0, used, seq_len),
  max_seqlen=used)`；q / k / v 是 `[seq_len, H_local, 128]` bf16、post QK-norm、post RoPE，staging buffer
  上的 strided view（后端先 `.contiguous()`）。
- t2va 打包顺序 `[文本 | 音频 | 视频 | pad]`，视频按 (t, h, w) 光栅、w 最内层，pad 到 64 的倍数；
  `video_start = used − T·H·W`。这与 Miowtion 训练打分器时的布局一致，所以 t2va 的 bundle 直接可用。
- 层号来自 impl 的 `prefix`（`^blocks\.(\d+)\.`）；token refiner 走稠密。步号来自
  `get_forward_context().current_timestep`（0 起，共 N−1 步）。
- 全局头编号 `tp_rank * heads_per_tp_rank + ulysses_rank * H_local + j`（TP 先按连续区间切头，Ulysses 再切）。
- pad 行 [used, seq_len) 输出置零；稠密回退用 sglang 自己的 SDPA 后端，因此与 baseline 数值一致。

## 限制

- Veda 打分器只在 t2va、Turbo 8 步教师上训过；fl2va 的关键帧按全局行处理能跑但未验证，ref2va 直接拒绝。
- SM120 上 Veda 只有前向（Miowtion 的 FA4 补丁没放开反向）；本后端只做推理。
- Turbo LoRA 含 AdaLN 增量，与 `--minimax-h3-adaln-online` / AdaLN cache 互斥，所以只能 tp2（或 layerwise offload），
  用不上 recipes 里最快的 tp1/ul8 配置。
- 不支持 ring、torch.compile、BCG（与 VSA-H3 同）。

## 结果（2026-09-26，8× RTX PRO 5000 Blackwell sm_120，tp2 × ulysses4，Turbo v4_step600 LoRA，8 次前向，seed 0）

baseline 是同一分支的稠密路径（SM120 上默认 torch SDPA）；Veda 用 `Minimax-H3-T2VA-Veda-8NFE-600Step-Preview`
bundle，keep 0.1，全部步稀疏。10 条 T2VA prompt：2 条官方 10 s，8 条 5 s（sglang cookbook、Comfy-Org 模板、
larryvrh 示例、MovieGen-H3 ×5）。

| prompt | 时长 | 去噪 stage baseline → Veda | stage 加速 | 稳态每步中位 baseline → Veda | 步加速 | PSNR / SSIM（两段之间） |
|---|---|---|---|---|---|---|
| official_t2va_starship | 10 s | 56.2 → 38.5 s | 1.46× | 7.71 → 4.04 s | 1.91× | 14.5 dB / 0.693 |
| official_guide_bakery | 10 s | 55.7 → 38.3 s | 1.45× | 7.69 → 3.99 s | 1.93× | 16.8 dB / 0.672 |
| sglang_cookbook_cats | 5 s | 32.4* → 21.8 s | 1.49×* | 2.72 → 1.96 s | 1.39× | 18.5 dB / 0.721 |
| comfy_template_rooftop | 5 s | 21.2 → 22.2 s | 0.95× | 2.81 → 2.02 s | 1.39× | 18.4 dB / 0.755 |
| larryvrh_corgi | 5 s | 20.4 → 21.5 s | 0.95× | 2.72 → 1.94 s | 1.40× | 16.5 dB / 0.716 |
| moviegen_0000 | 5.17 s | 20.8 → 22.4 s | 0.93× | 2.80 → 2.02 s | 1.39× | 14.1 dB / 0.513 |
| moviegen_0048 | 5.17 s | 21.0 → 22.1 s | 0.95× | 2.80 → 2.00 s | 1.40× | 14.1 dB / 0.695 |
| moviegen_0072 | 5.17 s | 21.0 → 22.2 s | 0.95× | 2.81 → 2.02 s | 1.39× | 14.5 dB / 0.645 |
| moviegen_0360 | 5.17 s | 20.9 → 22.2 s | 0.94× | 2.80 → 2.02 s | 1.39× | 14.7 dB / 0.605 |
| moviegen_0696 | 5.17 s | 21.1 → 22.1 s | 0.95× | 2.80 → 2.00 s | 1.40× | 19.9 dB / 0.670 |

\* 猫咪的 baseline 是 venv 建好后的首次运行，第 0 步 9.1 s（冷缓存），其余 5 s baseline 的第 0 步都是 3.6 s。

- **加速比以去噪 stage 时长为准**（stage 边界有 CUDA 同步；逐步计时不同步，baseline 的逐步数字整体错位约一步，
  表中的"稳态每步中位"只作参考）。无 warmup 时第 0 步 baseline 3.6 s、Veda 7.9 s，多出的约 4 s 是 FA4 CuTe 块稀疏
  kernel 的 JIT，每个新进程付一次，所以 8 步的 5 s 片段在 stage 级不赚（0.95×）。
- **带 warmup 的子集**（同形状合成请求先跑一次，与 `run_minimax_h3.sh` 的做法一致，即常驻服务的稳态）：

  | prompt | baseline | Veda | Veda 首步稠密 |
  |---|---|---|---|
  | sglang_cookbook_cats，5 s | 19.9 s | 15.8 s，1.26× | 16.5 s，1.20× |
  | moviegen_0000，5.17 s | 20.2 s | 16.4 s，1.24× | 17.0 s，1.19× |
  | official_t2va_starship，10 s | 55.9 s | 32.6 s，1.72× | 36.4 s，1.54× |

- 加速比低于单卡 Miowtion 的 1.59× / 3.4×：tp2 × ul4 下每 rank 只算 7 个头的 attention，MLP 与 all-to-all 不变。
- 每条 Veda 运行的日志都有 `Veda attention active: blocks.0.attn at step 0, 7 local heads (model heads 0..6)`。
- PSNR / SSIM 只是两段视频的差异度，不是画质：差异在构图层面，与"同配置换 seed"的 16.6 dB / 0.635
  同量级（v2g-step2-results.md）。抽帧目视三组都清晰、跟随 prompt；画质结论要人看视频。
- 产物：`/workspace/Miowtion/artifacts/sglang_veda_bench/report/index.html`（并排视频 + 逐条数字 + prompt）。
