# fast-ulysses 融合进 SGLang Diffusion（Wan2.1 I2V 4×B200）设计文档

日期：2026-07-14
状态：待用户审阅
目标场景：Wan2.1-T2V-14B，720p / 81 帧 / 50 步 / CFG，4×B200 纯 Ulysses（`--ulysses-degree=4`）

> 变更记录（2026-07-14，用户指示）：目标模型由 I2V-14B-720P 改为 **T2V-14B**（无图像条件，
> 基准命令去 `--image-path`；self-attn/USP 路径与全部集成设计不变）。§2 中关于 I2V 的调研
> 事实保留作背景。性能测试须在 GPU 空闲机上进行。

## 1. 背景与目标

`/data/fast-ulysses`（远程 ion-b200 容器 `sglang-diffusion-triplemu`）是基于 NVSHMEM 对称堆 + NVLink P2P 的
Ulysses all-to-all 自定义算子库，提供：

- `UlyssesGroup.all_to_all_single_4d(x, mode, tag, use_tma)`：mode0 `(b,s_local,n,d)→(b,s_global,n_local,d)`，
  mode1 为逆；8×H200 实测比 NCCL 快 1.5–2.1×。
- `all_to_all_single_4d_qk2(q, k, weight_q, weight_k, cos, sin, ...)`：把 QK RMSNorm+RoPE 融进 scatter
  kernel（cross_head + interleaved），对 Wan q+k 场景 H200 实测 1.61×。
- 独立算子 `rms_norm / rope / norm_rope`。

目标：把该库接入 sglang diffusion 运行时的 Ulysses 序列并行路径，在目标场景实测端到端（denoise latency
口径）性能提升。交付形态为**实验开关**（env var），代码按可上游标准编写，暂不加 server flag / CI。

## 2. 调研结论（事实依据）

### 2.1 sglang 侧接入点

- Wan self-attn 走 `USPAttention`（`runtime/layers/attention/layer.py:611`）。每层 4 次 a2a：
  进注意力 q/k/v 各 1 次（layer.py:970-972 → `_usp_input_all_to_all`，head_dim=2），
  出注意力 1 次（layer.py:991 → `_usp_output_all_to_all`）。
- 底层原语是 `torch.distributed.all_to_all_single`（`usp.py:46`），且每次 a2a 前后各有一次
  permute+contiguous 拷贝（usp.py:100-120, 233-253）。
- 形状约定与 fast_ulysses 完全对齐：进注意力 `[b,s_local,h,d]→[b,s_global,h_local,d]` = mode0；
  出注意力 = mode1。fast_ulysses 直接消费 `(b,s,n,d)` 布局，**usp.py 的 permute/contiguous 可整体省去**。
- Ulysses 进程组：`PROCESS_GROUP.ULYSSES_PG`（`runtime/distributed/parallel_groups.py:90`）。
- 无现成插拔机制；收口点为 `usp.py:_usp_input_all_to_all / _usp_output_all_to_all`。
- Wan 不在 BCG 白名单（CUDA graph 不涉及）；torch.compile 默认关，开启时 fullgraph=False，
  collective 处 graph break。

### 2.2 Wan2.1 与 qk2 融合算子的语义对照（逐项吻合）

| 融合算子约定 | Wan2.1 现状（wanvideo.py） | 匹配 |
|---|---|---|
| x 形状 (b, seq, n, d) | q/k unflatten 后 (1, S_local, 40, 128) contiguous | ✅ |
| RMSNorm cross_head，weight [n*d] | `rms_norm_across_heads`，RMSNorm(5120)，整条 hidden 归约（:424-427, :529-538） | ✅ |
| eps 1e-6，fp32 累加回 bf16 | 同（configs eps=1e-6；sgl_kernel rmsnorm fp32 累加） | ✅ |
| cos/sin (seq, d/2) fp32 | `(S_local, 64)` fp32（fp64 生成后 cast；`_compute_rope_for_sequence_shard` :971-992） | ✅ |
| interleaved（GPT-J 相邻配对） | 三条 RoPE 路径均 `is_neox=False`（:543-578） | ✅ |
| RoPE 覆盖全 head_dim | rope_dim_list=[44,42,42]，和=128 | ✅ |

720p 81 帧：S = 21×45×80 = 75600，sp=4 时 S_local=18900、h_local=10，整除无 padding。
每层另有一次 `torch.cat([cos,sin])` 重建 cos_sin_cache（:546-552），融合后可一并消除。

注意点：
- fast_ulysses 要求 norm weight / cos / sin 为 **fp32**；sglang 加载的 norm weight 是 bf16，需一次性
  `.float()` 缓存。
- cross-attn（attn2）`skip_sequence_parallel=True`，无 a2a，不在融合范围。
- v 无 norm/rope，继续走 `all_to_all_single_4d`。

### 2.3 远程环境（ion-b200 / sglang-diffusion-triplemu）

- 8×B200（183 GB/卡）全 NV18 互联，空闲；驱动 580.126.20。
- torch 2.13.0a0 (NGC 26.06) + CUDA/nvcc 13.3 匹配；Python 3.12。
- fast_ulysses 0.0.1 已构建安装（wheel，2026-07-14 构建）可导入全部符号；源码 `/data/fast-ulysses`。
- NVSHMEM 3.7.1 cuda13 archive 在 `/data/libnvshmem-linux-x86_64-3.7.1_cuda13-archive`。
- sglang editable 安装指向 `/data/sglang`，HEAD 1a35440c4 与本地一致（远程仅 pyproject.toml 放开
  torch 版本钉死，勿覆盖）。
- **Wan2.1 权重缺失**；默认 HF 缓存盘 100% 满（剩 2.5G）→ 权重必须放 `/data`（剩 419G）。
- fast_ulysses 的公开性能数字是 8×H200 的，B200 4 卡 op 级收益未实测。

### 2.4 基准口径

- 入口：`sglang generate`（离线、进程内，`local_mode=True`），主指标 **denoise latency**
  （perf dump 中 DenoisingStage 之和），warmup 用 `--warmup`（1 step、单独计时排除）。
- before/after 用 `benchmarks/compare_perf.py`。
- Wan 默认 `auto_dit_layerwise_offload=True` → 基准显式关（`--dit-layerwise-offload false`），
  B200 显存充裕，做纯延迟口径。

## 3. 设计

### 3.1 总体结构（两阶段，两个 env 开关）

- **阶段 1 — a2a 替换**：`SGLANG_DIFFUSION_ENABLE_FAST_ULYSSES=1` 时，USPAttention 等长主路径的 4 次
  a2a 改走 `UlyssesGroup.all_to_all_single_4d`（含省 permute）。搬运语义逐位等价，可强验证。
- **阶段 2 — QK 融合**：`SGLANG_DIFFUSION_ENABLE_FAST_ULYSSES_QK_FUSION=1`（依赖前者）时，Wan block 的
  norm_q/norm_k + RoPE + q/k 进注意力 a2a 融成一次 `all_to_all_single_4d_qk2`。数值上更准
  （全程 fp32 中间），与 baseline 非逐位一致，用容差/轨迹相似度验证。

两个开关分离，便于逐阶段归因。env var 加入 `python/sglang/multimodal_gen/envs.py`
（遵循 env-var-conventions；实现前先读该 skill）。

### 3.2 阶段 1：wrapper 模块 + usp.py 收口改造

新文件 `python/sglang/multimodal_gen/runtime/layers/fast_ulysses_backend.py`：

- `is_enabled()`：env 开 + `import fast_ulysses` 成功 + CUDA 平台。
- `get_group()`：懒初始化单例 `UlyssesGroup(process_group=PROCESS_GROUP.ULYSSES_PG)`。
  SPMD 下首次 forward 全 rank 同步进入，构造安全（内部 broadcast+barrier）；warmup 请求吸收首调
  microbenchmark 开销。
  **启用护栏**：仅当 ulysses 组大小 == `dist.get_world_size()`（纯 Ulysses，无 cfg/tp/ring 子组）
  时启用，否则 warn 一次并永久 fallback——规避多子组各自 NVSHMEM init 的未验证场景。
- `can_use(x)`：4D、bf16/fp16、`s % ws == 0`、`n % ws == 0`、`ws ∈ [2,8]`、d·esize 16B 对齐。

修改 `usp.py`：

- `_usp_input_all_to_all(x, head_dim, tag="")`：head_dim==2 且 **tag 非空** 且 enabled 且 can_use →
  `group.all_to_all_single_4d(x, mode=0, tag=tag)`（跳过全部 permute）；否则原路径。
  tag 为空即不启用 fast path——未显式标注的调用点（如 mask 分支的 q/k/v）若共用默认 tag 会
  复用同一块对称堆输出缓冲而互相覆盖，故只对显式传 tag 的主路径调用点生效。
- `_usp_output_all_to_all(x, head_dim, tag="")`：同理 mode=1。
- head_dim==1 变体、varlen 变体不改（自动 fallback）。

修改 `layer.py` USPAttention 主路径调用点：q/k/v 传 `tag="usp_q"/"usp_k"/"usp_v"`，输出传
`tag="usp_out"`——fast_ulysses 的输出缓冲按 tag 复用，q/k/v 并存必须不同 tag；跨层/跨 step 的
复用安全性由同 stream 串行顺序保证（前一消费 kernel 先于下一次覆盖执行）。

对称堆预算：4 个 tag × 193.5 MB ≈ 0.8 GB < 默认 2 GiB 池，足够（qk2 的 `::q/::k` 复用同级别）。

集体调用序列一致性（fast_ulysses 硬约束）：Wan 等长切分下所有 rank 每层以相同 (shape, mode)
序列调用，天然满足；护栏把不确定场景（mask/varlen/replicated 路径不经过本改动点）排除在外。

### 3.3 阶段 2：USPAttention 增加融合入口

原则：a2a 留在 attention layer 层，模型侧只是"把 norm+rope 的参数下放"。

- `USPAttention.forward(q, k, v, *, qk_fused_ctx=None)`：新增可选参数
  `qk_fused_ctx = (weight_q_fp32, weight_k_fp32, cos, sin, eps)`。
  - ctx 非空时：q/k 走 `all_to_all_single_4d_qk2(q, k, w_q, w_k, cos, sin,
    mode="cross_head", interleaved=True, eps=eps, tag="usp_qk")`，v 走普通 mode0；后续不变。
  - 可用性判定（`can_use_qk2`）全部前置在模型侧 ctx 构造处——ctx 非 None 即保证 fast path
    可用，layer 内只做 assert（不需要二级 fallback）；判定不过则模型侧走原 norm/rope 路径。
- `wanvideo.py` `WanTransformerBlock.forward`：融合开启时跳过本地 norm_q/norm_k 与 RoPE 及每层
  `torch.cat([cos,sin])`，改传 `qk_fused_ctx`（q/k 先 unflatten 成 (b,s,n,d)）。
  - norm weight 的 fp32 副本在首次 forward 缓存于 module 上。
  - cos/sin 直接用 `_compute_rope_for_sequence_shard` 的 (S_local, 64) fp32 原始输出（分离两表，
    不再 cat），每 forward 复用（现有 lru_cache）。
  - `skip_sequence_parallel`（cross-attn）与 sp=1 时不传 ctx，走原路径。

### 3.4 正确性验证

- 阶段 0（op 级）：远程 4×B200 跑库自带 `test_correctness.py`、`test_a2a_qk.py`（ws=4）。
- 阶段 1：a2a 是逐位等价搬运 → (a) 4 卡 torchrun 对拍脚本：随机 (1,18900,40,128) bf16，
  fast 路径 vs NCCL 路径**逐位一致**；(b) e2e 同 seed 下 baseline vs fast 的 denoise 轨迹
  应逐位一致（`tools/compare_diffusion_trajectory_similarity.py` + 输出视频抽查）。
- 阶段 2：融合算子 fp32 中间精度更高，非逐位一致 → (a) 4 卡对拍脚本：fused vs
  "norm→rope→NCCL a2a" 参考路径，容差对齐 fast-ulysses 自带测试阈值；(b) e2e 轨迹相似度
  （余弦/相对误差）+ 视频肉眼验收。
- 对拍脚本放 `python/sglang/multimodal_gen/test/`（实验形态，不注册 CI）。

### 3.5 性能测量（交付数字）

固定：GPU 0-3、seed 42、720p/81帧/50步、CFG 串行（`--cfg-parallel-size=1`）、`--warmup`、
`--dit-layerwise-offload false`、denoise latency 取 3 次中位。

| # | 配置 | 开关 |
|---|---|---|
| 0 | op 级：bench_uniform（N=75600 H=40 D=128, ws=4, mode0/1）+ bench_qk_fused | — |
| 1 | baseline NCCL, eager | 无 |
| 2 | baseline NCCL + torch.compile | 无 |
| 3 | fast a2a, eager | FAST_ULYSSES=1 |
| 4 | fast a2a + compile | FAST_ULYSSES=1 |
| 5 | fast a2a + qk2, eager | 两者=1 |
| 6 | fast a2a + qk2 + compile | 两者=1 |

汇报：denoise latency、e2e、峰值显存（perf dump / compare_perf.py），外加 baseline vs 最优
配置各一份 nsys（归因 a2a kernel 时间变化）。

### 3.6 开发/同步流程

本地仓库开发（新分支）→ rsync 改动文件到远程 `/data/sglang`（editable 安装即时生效）→
远程 4×B200 跑验证与基准 → 结果 JSON 收回本地留档。远程 pyproject.toml 的本地改动不触碰。

## 4. 风险与对策

1. **B200(sm100)+NVSHMEM 3.7.1+torch 2.13 nightly 组合未实测** → 阶段 0 先跑库自带测试/基准，
   B200 op 级加速比直接给出端到端收益上限；若 op 级无收益，止损于阶段 0。
2. **torch.compile 与 custom op 交互**：fast_ulysses 是 TORCH_LIBRARY op，若 dynamo trace 报错
   （缺 fake tensor 实现），给 wrapper 加 `torch._dynamo.disable`，行为退化为与现状
   `dist.all_to_all_single` 相同的 graph break，不阻塞。
3. **首调懒惰 microbenchmark 的 hang 约束**（全 rank 相同调用序列）：护栏限定纯 Ulysses 等长
   主路径，warmup 请求先行吸收。
4. **磁盘**：权重 ~60-70GB 下载到 `/data`（419G 空闲），HF 缓存盘已满不可用。
5. **收益不确定性**：a2a+permute 在 4×B200 端到端中的占比需实测（粗估 denoise 的 10-20%）；
   B200 NVLink 带宽高于 H200，NCCL 基线更强，加速比可能低于 H200 数字——以实测为准。

## 5. 范围外（明确不做）

- ring_degree>1、cfg-parallel、tp>1 组合的 fast path（护栏 fallback）。
- varlen / mask / replicated-token 路径。
- `all_to_all_single_4d_async` 通信重叠（README 已证 cooperative GEMM 下无重叠收益，留待后续）。
- server flag、CI 注册、上游 PR（转正式形态时再做）。
- 其他模型（HunyuanVideo 等 UlyssesAttention/GroupCoordinator 路径）。
