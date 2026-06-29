# 设计：自定义 NVSHMEM Ulysses all-to-all 算子接入 SGLang Diffusion Wan2.1/2.2

- 日期：2026-06-29
- 分支：`feat/custom-ulysses-a2a-wan`（基于 `triplemu/ulysses-a2a`）
- 算子来源：`/home/ubuntu/workspace/github/self/custom_ulysses_op`（`custom_ulysses_op.UlyssesGroup`，`torch.ops.ulysses.all_to_all_single_4d`）

## 1. 目标

把自定义的、基于 NVSHMEM 对称堆 + NVLink P2P 的 Ulysses 序列并行 all-to-all 算子，叠加到 SGLang Diffusion 的 Wan2.1 / Wan2.2 注意力序列并行路径里，替换现有基于 `torch.distributed.all_to_all_single` 的实现；然后在 H200×8 上分别测「cp+usp 共同叠加」与「单独 usp」两种 SP 拓扑下的 kernel 级性能提升，开启 TMA + tune，用 torch profiler 抓 trace 对比。

### 测试矩阵（已与用户确认）

| 维度 | 取值 |
|---|---|
| 模型 | `Wan-AI/Wan2.1-T2V-14B-Diffusers`、`Wan-AI/Wan2.2-T2V-A14B-Diffusers`（均 40 heads / head_dim 128） |
| 分辨率/帧数 | 1280×720 / 81 帧 → patch 后序列 75600（21 latent 帧 × 45 × 80） |
| SP 配置 | 单独 usp = `--ulysses-degree 8 --ring-degree 1`（sp=8）；cp+usp = `--ulysses-degree 4 --ring-degree 2`（sp=8） |
| 算子 | baseline（torch `all_to_all_single`）vs 自定义 NVSHMEM op（`use_tma=True` + tune） |
| GPU | H200 × 8（hyper00，docker `sglang-diffusion-ulysess`，`source /data/.torch/bin/activate`） |

2 模型 × 2 SP 配置 × 2 算子 = 8 个 cell；每个 cell 一次「干净延迟」run + 一次「profile」run。

## 2. 关键事实（代码层面，已核实）

### 2.1 算子接口

`UlyssesGroup(process_group=None, device=None, initial_pool_bytes=2<<30)`：从 torch ProcessGroup bootstrap，**所有 rank 必须一起构造**（内部 `dist.broadcast` + 两次 `dist.barrier`）。默认 `setdefault` `NVSHMEM_DISABLE_NVLS=1`、`NVSHMEM_REMOTE_TRANSPORT=none`（适配无 NVSwitch fabric / 带 IB 的节点）。

`all_to_all_single_4d(x, *, mode=0, tag="", seq_lens=None, head_splits=None, use_tma=None)`：4D `(b, s, n, d)`，fp16/bf16。

| mode | 输入 | 输出 |
|---|---|---|
| 0 | `(b, s_local, n_global, d)` | `(b, s_global, n_local, d)` |
| 1 | `(b, s_global, n_local, d)` | `(b, s_local, n_global, d)` |

- `use_tma` 三态：`None` 自动（uniform 且 sm90+ → TMA）；`True` 强制 TMA（需 sm90+）；`False` 强制 non-TMA。
- `tag`：对称堆缓冲标签，**不同 shape/路径用不同 tag 复用各自输出块**。同 tag 复用同一 buffer，返回张量是该 buffer 的视图。
- **硬约束**：所有 rank 必须以相同 `(shape, mode, use_tma)` 序列调用；`tune` 与 lazy 兜底互斥（要么全 tune 要么全 lazy，否则整组 hang）。`world_size ∈ {1,2,4,8}`，单节点 NVLink P2P。
- `tune(shape, *, mode, use_tma, verbose)`：预热某 shape 的 launch 配置（集体，只预热不改结果）。

### 2.2 SGLang Diffusion 接入面

- Wan2.1/2.2 的 `WanTransformer3DModel`（`runtime/models/dits/wanvideo.py`）self-attention 用 `USPAttention`（`runtime/layers/attention/layer.py:406`）。cross-attention 用 `WanT2VCrossAttention`，构造时 `skip_sequence_parallel=True` → **不通信，不碰**。
- `causal_wanvideo.py` 走 `LocalAttention` + KV cache，**不在 Ulysses 路径上**，本次不涉及。
- VSA 变体（`UlyssesAttention_VSA`）仅 `--attention-backend video_sparse_attn` 时启用，本次不涉及。
- `USPAttention.forward` 主路径（`layer.py:798-819`）：对 q/k/v 各调一次 `_usp_input_all_to_all(x, head_dim=2)`，attention 后调 `_usp_output_all_to_all(out, head_dim=2)`（`runtime/layers/usp.py:69 / 202`）。q/k/v layout 全程 `[B, S_local, H, D]`，`head_dim=2`。
- shape 对齐：`_usp_input_all_to_all(head_dim=2)`：`[B,S_local,H,D] → [B,S_global,H_local,D]` ＝ 算子 **mode 0**；`_usp_output_all_to_all(head_dim=2)` ＝ **mode 1**。逐位吻合。
- Wan 是 **uniform 切分**（DiT 入口 `enable_sequence_shard` padding 保证整除，`wanvideo.py:1059-1074`），**不走 varlen**。`USPAttention` 无 `seq_lens` 概念。
- Wan 默认 **不开** `enable_packed_qkv_input_a2a`（turbo 的 `async_a2a_communicate`），走的就是 `_usp_input_all_to_all`。
- ulysses 的裸 torch ProcessGroup 直接可取：`get_sp_group().ulysses_group`（`runtime/distributed/parallel_state.py` → `SequenceParallelGroupCoordinator.ulysses_group`，`group_coordinator.py:1271`）。现有代码多处这样用（`layer.py:790,875,951`、`usp.py:39,55`）。
- `sp_degree = ulysses_degree × ring_degree`（`server_args._validate_parallelism`，`server_args.py:2054`）。dp=tp=1 时 `sp_degree == num_gpus`。
- ring_degree>1 时，ulysses all-to-all 之后走 `ring_attn`（`layer.py:803-811`，`usp.py:338`）。USP = Ulysses(all-to-all) + Ring。自定义算子**只加速 Ulysses all-to-all 那一段**，不碰 ring 的 P2P。

### 2.3 各 cell 的实际 all-to-all shape（sp=8，DiT 入口已把 75600 序列切成每 rank 9450）

- **U8R1**：ulysses ws=8。mode0 输入 `[B, 9450, 40, 128]` → 输出 `[B, 75600, 5, 128]`。
- **U4R2**：ulysses ws=4。mode0 输入 `[B, 9450, 40, 128]` → 输出 `[B, 37800, 10, 128]`。
- `B`：单 prompt 时为 1；CFG 以 batch 维处理时为 2（无 cfg-parallel，sp 已占满 8 卡）。对称堆按最大 buffer 估：U8R1 输出 `1×75600×5×128×2 ≈ 96.7 MB`/tag，q/k/v/out 4 个 tag ≈ 390 MB（B=2 → ~780 MB），默认 2GiB pool 足够。

### 2.4 Profiler

- torch profiler 集成在 `runtime/utils/profiler.py`（`SGLDiffusionProfiler`），由采样参数 `--profile` / `--num-profiled-timesteps`(默认5) / `--profile-all-stages` 控制；trace 落 `SGLANG_DIFFUSION_TORCH_PROFILER_DIR`（绝对路径），导出 `*.trace.json.gz`，解析 `cat in ("kernel","gpu_memcpy")`。
- 真实延迟看日志 `(with warmup excluded)` 那行（`diffusion_generator.py:403-411`），`--profile` 会扰动延迟，不能当真实数字。

## 3. 架构与方案选择

### 选定方案 A：在 `_usp_input/output_all_to_all` 内原生吃 4D

在 `_usp_input_all_to_all` / `_usp_output_all_to_all` 内部加一条 gated 分支：当
`开关打开 且 head_dim==2 且 uniform 且 ulysses_ws∈{2,4,8} 且 调用方传了显式 tag`
时，直接 `all_to_all_single_4d(x, mode=0/1, tag, use_tma=True)` 原生吃 4D，**省掉现有 python 路径的 permute/contiguous/多次 reshape host 开销**；否则原样走 torch。

被否决的备选：
- **方案 B**：只在底层 `_usp_all_to_all_single` 拦截 torch collective。保留 python flatten/permute 开销，且算子要 4D 不要 flatten。劣于 A。
- **方案 C**：改 `sequence_model_parallel_all_to_all_4D` / turbo `async_a2a_communicate`。Wan self-attn 主路径根本不走这些。

## 4. 组件

1. **`runtime/layers/custom_ulysses_a2a.py`（新建，单例 wrapper）**
   - 懒构造 `UlyssesGroup(get_sp_group().ulysses_group)`，按 ulysses PG 缓存为单例。首次 attention 在 warmup 期全 rank lockstep 触发，构造是集体安全的。
   - 暴露 `custom_ulysses_a2a(x, mode, tag) -> Tensor`，内部 `all_to_all_single_4d(..., use_tma=_USE_TMA)`。
   - `is_enabled()`：env 开关 + `import custom_ulysses_op` 成功 + `ulysses_ws ∈ {2,4,8}`。import 失败 / 不满足 → False（自动回落 torch）。
   - `initial_pool_bytes` 由 env 配，默认 `2<<30`。
2. **`runtime/layers/usp.py`**：`_usp_input_all_to_all` / `_usp_output_all_to_all` 增加可选 kwarg `a2a_tag: str | None = None` + gated 分支（gate 见方案 A）。其余路径（head_dim==1、varlen、replicated）不变，仍走 torch。
3. **`runtime/layers/attention/layer.py`**：`USPAttention.forward` 主路径（798-819）3 个 input 调用传 `a2a_tag="uin_q"/"uin_k"/"uin_v"`，output 调用传 `a2a_tag="uout"`。**q/k/v 同时存活，必须不同 tag，否则对称堆 buffer 复用互相覆盖（正确性 bug）**。仅约 4 行改动。
4. **env 开关**（沿用 diffusion `envs.py` 风格，`SGLANG_DIFFUSION_*`；实现前过一遍 `env-var-conventions` skill）：
   - `SGLANG_DIFFUSION_CUSTOM_ULYSSES_A2A`：总开关（默认关，保证不影响现有行为）。
   - `SGLANG_DIFFUSION_CUSTOM_ULYSSES_USE_TMA`：默认 True（H200=sm90 支持 TMA）。
   - `SGLANG_DIFFUSION_CUSTOM_ULYSSES_POOL_BYTES`：对称堆字节数，默认 `2<<30`。
5. **tune / TMA**：`use_tma=True`。tune 走**首次调用的 lazy 微基准**（集体一致，warmup 期完成、被排除在测时之外）；保证 warmup 分辨率＝测试分辨率。等价于「开启 tune」，比枚举 shape 预 tune 更稳。显式 `tune()` 作为后续可选增强。

## 5. 正确性验证（性能测量前必做）

固定 seed，自定义 op ON vs OFF 各生成一次，比 latent / 输出帧（或用 `tools/compare_diffusion_trajectory_similarity.py`），确认数值一致（算子自带逐位对拍，端到端 fp 累加差异应可忽略）。**不一致就停，先 debug，不进入性能测量。**

## 6. Benchmark & Profiling 流程

每个 cell 两次 run：

1. **干净延迟**（不带 `--profile`）：看 `(with warmup excluded)` 行 + `--perf-dump-path`。
2. **kernel trace**：`SGLANG_DIFFUSION_TORCH_PROFILER_DIR=<abs> ... --profile --num-profiled-timesteps 5`，导出 trace。

命令骨架（Wan2.2-T2V-A14B，U8R1）：
```bash
export SGLANG_DIFFUSION_TORCH_PROFILER_DIR=/abs/profiles/torch
export SGLANG_DIFFUSION_CUSTOM_ULYSSES_A2A=1   # baseline 时设 0
sglang generate --backend=sglang \
  --model-path=Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --prompt="A cat and a dog baking a cake together in a kitchen." \
  --width=1280 --height=720 --num-frames=81 --seed=42 \
  --num-gpus=8 --ulysses-degree=8 --ring-degree=1 \
  --warmup [--profile --num-profiled-timesteps 5] --save-output
```
cp+usp 换 `--ulysses-degree=4 --ring-degree=2`。

分析（用 `llm-torch-profiler-analysis` skill）：对比 all-to-all 段——baseline 的 NCCL `all_to_all` / `ncclDevKernel` vs 自定义 nvshmem kernel——的 kernel 时间，以及 denoise 端到端延迟。产出一张对比表（每 cell：a2a kernel us、denoise 延迟、加速比）。

## 7. 构建 & 环境

H200 docker `sglang-diffusion-ulysess`：进 docker → `source /data/.torch/bin/activate` → 先 `python -c "import custom_ulysses_op"` 验证；不在位则按 README 构建（`NVSHMEM_HOME=<...> CUSTOM_ULYSSES_CUDA_ARCH=90 pip install -e . --no-build-isolation`，需 NVSHMEM 3.7.0）。把本仓库分支推上去/拉到 docker 内对应路径。

## 8. 风险与约束

- **分支**：在 `feat/custom-ulysses-a2a-wan`（基于当前 `triplemu/ulysses-a2a`）上提交，可回滚 / 可合回。
- **集体一致性**：所有 rank 必须同序调用 `all_to_all_single_4d`，且 use_tma 一致；开关/TMA env 必须全 rank 相同。gate 条件只依赖全 rank 一致的量（env、ulysses_ws、head_dim、tag），不依赖 per-rank 数据，保证一致。
- **tag 唯一性**：q/k/v/out 四个 tag 互不相同（见组件 3）。
- **CFG**：U8R1/U4R2 占满 8 卡，不开 cfg-parallel；lazy tune 覆盖实际 batch。
- **回落**：算子 import 失败、ws∉{2,4,8}、非 head_dim==2、varlen、开关关 → 自动走原 torch 路径，零行为变化。
- **算子整除**：14B/A14B 40 heads，U8(40/8=5)、U4(40/4=10) 均整除，合法。

## 9. 验收标准

1. 开关关时行为与改动前完全一致（现有测试不回归）。
2. 开关开时 Wan2.1-14B / Wan2.2-A14B 在 U8R1 / U4R2 下生成数值与 baseline 一致。
3. 8 个 cell 的 torch profiler trace 全部抓到，产出 kernel 级对比表（a2a kernel 时间 + denoise 延迟 + 加速比），覆盖「单独 usp」与「cp+usp」两拓扑。
