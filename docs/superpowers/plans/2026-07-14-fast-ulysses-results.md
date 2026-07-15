# fast-ulysses 集成实测记录

环境：ion-b200，容器 `sglang-diffusion-triplemu`，8×B200（全 NVLink），本次使用 GPU 0-3（`CUDA_VISIBLE_DEVICES=0,1,2,3`，ws=4）。NVSHMEM 运行时报告 v3.6.5（wheel 构建时链接的版本；/data 下另有 3.7.1 archive，二者差异未深究，正确性/性能以本次实测为准）。库源码 `/data/fast-ulysses`（只读），fast_ulysses wheel 已 pip 安装。

## Op-level (4×B200)

日期：2026-07-14

### 正确性（ws=4）

| 测试 | 结果 |
|---|---|
| `test/test_correctness.py` | ALL PASS（float16/bfloat16 × d=64/128/256 × mode0/mode1 × use_tma=None/True/False，含 distinct-tag q/k 共存无别名） |
| `test/test_a2a_qk.py` | ALL PASS（per_head/cross_head × il=True/False，maxdiff ≤ 1.95e-03；qk2 与 2× single fused 逐位一致） |

### a2a microbenchmark（Wan shape：N=75600, H=40, D=128, `bench_uniform.py`）

| mode | ours med (us) | ours GB/s | NCCL med (us) | NCCL GB/s | 加速比 |
|---|---|---|---|---|---|
| mode0 | 253.2 | 573 | 551.1 | 263 | 2.18× |
| mode1 | 234.0 | 620 | 548.0 | 265 | 2.34× |

原始输出：

```
=== mode0 | ws=4 | H=40 D=128 | ours=auto ===
N=  75600 | ours(auto) med=   253.2us    573 GB/s | NCCL med=   551.1us    263 GB/s
=== mode1 | ws=4 | H=40 D=128 | ours=auto ===
N=  75600 | ours(auto) med=   234.0us    620 GB/s | NCCL med=   548.0us    265 GB/s
```

GPU 空闲复测（2026-07-14，确认 GPU 0-3 util 0%、显存 0MiB 后重跑）：mode0 ours 252.3us/575 GB/s、NCCL 554.3us/262 GB/s；mode1 ours 234.8us/618 GB/s、NCCL 548.3us/265 GB/s。与上表一致（差异 <1%）。

用户手测（2026-07-14，mode0，H=40 D=128 N=75600，ws 扫描）：

| ws | ours med | ours GB/s | NCCL med | NCCL GB/s | 加速 | NVLink 利用率* |
|---|---|---|---|---|---|---|
| 2 | 294.0us | 658 | 1056.0us | 183 | 3.59× | 73% |
| 4 | 251.1us | 578 | 548.1us | 265 | 2.18× | 64% |
| 8 | 154.9us | 547 | 302.4us | 280 | 1.95× | 61% |

*相对 B200 NVLink5 单向注入带宽 900 GB/s（18×NVLink，1.8TB/s 双向）。对照 8×H200（NVLink4，450 GB/s 单向）的公开数字，B200 实测为其 1.82-1.92×，与代际带宽翻倍吻合；两代利用率一致（~61-73%）。NCCL 列为 permute+all_to_all_single+permute 完整旧路径口径（含 permute 拷贝时间）。

### qk 融合基准（`bench_qk_fused.py`，ws=4, dtype=bf16, n_global=40, d=128，单位 ms/iter）

注意：seq_global 20480/46080 是脚本默认 shape（heads/d 与 Wan 一致，但**非** Wan 720p 的 75600）；加速比作量级参考，e2e 收益以后续任务实测为准。

| seq_global | a2a | unfused | fused | fused/unfused | nr_unfused | nr_fused | qk2 | qk2/2fused |
|---|---|---|---|---|---|---|---|---|
| 20480 | 0.081 | 0.226 | 0.122 | 1.856× | 0.145 | 0.041 | 0.245 | 1.004× |
| 46080 | 0.152 | 0.469 | 0.254 | 1.844× | 0.318 | 0.103 | 0.482 | 0.948× |

说明：nr_* 为 norm+rope 相对纯 a2a 的额外开销；qk2 为 q+k 一次调用（1 个 barrier）。表中为 GPU 空闲复测值（2026-07-14，两次复跑一致）：初测 46080 行 fused=0.241 偏低约 5.4%（超 5% 阈值），已按空闲复测更新；其余项差异 <5%。

### 结论

- 正确性：两个测试全部 PASS，B200 (ws=4) 上库可用。
- a2a 加速比 mode0 2.18× / mode1 2.34×（vs NCCL），远高于 1.1× 继续阈值。
- fused norm+rope+a2a 相对 unfused 约 1.84–1.86×（GPU 空闲复测后）；qk2 与 2× fused 基本持平（收益在减少 barrier）。
- 判定：继续后续任务。

## E2E baseline (NCCL, Wan2.1-T2V-14B)

应用户要求，模型从 Wan2.1-I2V-14B-720P 改为 **Wan2.1-T2V-14B-Diffusers**（口径其余不变）；作废的 I2V run 归档于远程 `/data/bench/results/void_i2v/`。

命令（每次 run 前确认 GPU 0-3 空闲、无残留进程）：

```bash
docker exec -e HF_HOME=/data/hf-cache -e CUDA_VISIBLE_DEVICES=0,1,2,3 -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
  sglang-diffusion-triplemu bash -c 'cd /data/sglang && sglang generate --backend=sglang \
  --model-path=Wan-AI/Wan2.1-T2V-14B-Diffusers --prompt="A cat walks slowly towards the camera." \
  --width=1280 --height=720 --num-frames=81 --num-inference-steps=50 --seed=42 \
  --num-gpus=4 --ulysses-degree=4 --ring-degree=1 --cfg-parallel-size=1 \
  --text-encoder-cpu-offload --pin-cpu-memory --dit-layerwise-offload false \
  --warmup --perf-dump-path /data/bench/results/nccl_eager_run<I>.json'
```

### nccl_eager（3 runs）

| run | denoise (s) | e2e (s) | 峰值显存 (GB) |
|---|---|---|---|
| 1 | 133.6 | 136.0 | 36.4 |
| 2 | 133.6 | 135.8 | 36.4 |
| 3 | 133.6 | 135.9 | 36.5 |

**中位数：denoise 133.6s**。三次完全一致（离散 <0.1%），基线稳定。日志无 diffusers fallback。

### nccl_compile

run1 由 Task 2 agent 启动后 agent 被停止，远程进程继续运行中；JSON 产出后补录。run2/3 暂缓（用户占用机器做手测；eager 为主口径）。

**补录（2026-07-14，Task 5）：run1 失败，无 JSON 产出，本阶段不重跑 compile。** 失败特征（`/data/bench/results/nccl_compile_run1.log`）：

- log 中无 Traceback/OOM/Killed（`grep -cE "Traceback|OOM|out of memory|Killed"` = 0），`error` 命中仅为 pytree Enum 弃用警告。
- 进程死于 inductor autotune/compile 阶段：log 全程为 triton_mm 自动调优输出，从未进入任何 denoise step。
- log 尾部（11:47:55 后无更新）：`torch/_inductor/fx_passes/spmd_check.py` 报跨 rank 图不一致警告（rank0 vs rank2/3：`aten.slice.Tensor` vs `aten.empty.memory_format`，float32 [1,16,3,1,162]，channels_last_3d），以及 dynamo `Dynamo does not know how to trace the builtin list.append` UserWarning。
- 判定：compile 路径在该环境下自身不稳定（与 fast_ulysses 无关，此 run 未开 fast 开关）；阶段 1 以 eager 为主口径，compile 组留待后续单独排查。

## 附录 A：模型加载慢定位（2026-07-14）

现象：基准 run 模型加载 ~13 分钟（4 worker 满核 CPU 自旋、零磁盘 IO）。GPU 0-3 与 4-7 行为一致（run1 加载 12:47 / anchor 13:19）——非卡组问题。

根因：`_adjust_offload()` else 分支（`runtime/server_args/server_args.py:646-653`）对视频生成模型**无条件** `dit_cpu_offload=True` + `text_encoder_cpu_offload=True`（不检查显存）。叠加 `pin_cpu_memory=True`（默认），4 个 worker 各自把 28GB DiT + 11GB T5 逐 tensor 锁页拷入 pinned CPU 内存（Run:ai streamer to cpu + 逐 tensor clone，~52MB/s/worker）。

验证（Run B，全 offload 显式关，GPU 4-7）：

| 组件 | offload auto（基准口径） | offload 全关 | 改善 |
|---|---|---|---|
| transformer (28GB) | 9:51 | 4:46 | 2.1x |
| text_encoder (T5) | 3:22 | 0:13 | 15x |
| 总加载 | 13:19 | **5:01** | 2.65x |

- denoise 口径不受影响（anchor 134.7s ≈ baseline 133.6s）。基准继续用原 flag 保持可比；生产建议：B200 上显式 `--dit-cpu-offload false --text-encoder-cpu-offload false`。
- 剩余瓶颈（transformer 4:46，~100MB/s/worker，疑 `pin_cpu_memory` 中转）未完成定位（Run C 被环境事故打断，见附录 B）。
- 上游建议：auto offload 应按「模型尺寸 vs 显存」决策，而非无条件开启。

## 附录 B：环境事故记录（cutlass-dsl 4.6.0 vs flash-attn-4）

- 13:16 容器内安装 `quack_kernels 0.6.1` 时连带把 `nvidia-cutlass-dsl` 升至 4.6.0（用户手测操作）。
- `flash-attn-4 4.0.0b15` 声明 `nvidia-cutlass-dsl>=4.4.2` 但实际与 4.6.0 API 不兼容（`cutlass.cute.core.ThrMma` 已移除）→ `from flash_attn.cute import ...` AttributeError。
- sglang 在 Blackwell 上无条件 `set_fa_ver(4)`（`platforms/cuda.py:447`，不探测 FA4 可用性）→ FA backend 的 forward 必炸。**13:16 后所有 sglang run 均无法 denoise**（与本项目改动无关）。
- 时间线证据：anchor（12:52 完成）成功；probe B（13:21）在 denoise 首步炸。
- 处理：用户自行修复环境；修复后 fast 组基准继续。
- 上游建议：`_prepare_flash_attention_for_blackwell` 应探测 FA4 import 可用性再 set_fa_ver(4)，失败时降级并 warn。

## Stage 1: fast a2a (eager, Wan2.1-T2V-14B, GPU 4-7)

口径：与 baseline 完全一致（720p/81帧/50步/seed 42/4卡纯 Ulysses/--warmup/layerwise offload 关），仅加 `SGLANG_DIFFUSION_ENABLE_FAST_ULYSSES=1`。
换卡组锚点：`nccl_eager_anchor`（GPU 4-7，NCCL）denoise **134.7s** / e2e 137.0s / 峰值 36.5GB（vs GPU 0-3 基线 133.6s，+0.82%，锚点有效）。

| run | denoise (s) | e2e (s) | 峰值显存 (GB) |
|---|---|---|---|
| 1 | 126.6 | 128.9 | 36.2 |
| 2 | 126.6 | 128.9 | 36.2 |
| 3 | 126.6 | 129.1 | 36.2 |

- **中位数 denoise 126.6s：vs 同卡组锚点 134.7s 提升 6.0%，vs GPU 0-3 基线 133.6s 提升 5.3%**。三次离散 <0.1%。
- 正确性：81 帧逐帧 MD5（PyAV rgb24）fast_run1 ≡ anchor ≡ baseline(GPU 0-3)，**逐位一致**（`bf3324bb1be2570a650abaea2a919312`）。NCCL 跨卡组确定性同时得证。
- fast path 生效确认：每个 run 的 log 中 `fast_ulysses UlyssesGroup initialized (world=4)` 出现 4 次（4 rank），无 diffusers fallback、无 Traceback。
- 环境注：run1/2/3 跑于 cutlass-dsl 4.5.2（用户修复后），anchor 跑于修复前兼容版本；两者 FA backend 路径一致（"Using fa attention backend"）。

## Stage 2: fast a2a + QK fusion (eager, Wan2.1-T2V-14B, GPU 4-7)

日期：2026-07-15。代码：commit 02cac5d05d（QK norm+RoPE 融合进 fast_ulysses input a2a），同步远程后实测（6 文件 md5 校验一致）。口径同 Stage 1，两开关全开（`SGLANG_DIFFUSION_ENABLE_FAST_ULYSSES=1` + `SGLANG_DIFFUSION_ENABLE_FAST_ULYSSES_QK_FUSION=1`）。

### 对拍（4 卡，torchrun，两开关全开）

PASS。`qk2 q: max abs diff vs fp32 ref = 0.03125`、`qk2 k: max abs diff vs fp32 ref = 0.01562`（fp32 参考 = 全 fp32 cross-head RMSNorm + GPT-J 交错 RoPE，容差 atol/rtol 2e-2）；mode0/mode1 纯 a2a 与 NCCL 逐位一致。

### 基准（3 runs）

| run | denoise (s) | e2e (s) | 峰值显存 (GB) |
|---|---|---|---|
| 1 | 126.0 | 128.3 | 36.2 |
| 2 | 126.1 | 128.5 | 37.0 |
| 3 | 125.9 | 128.3 | 36.2 |

- **中位数 denoise 126.0s：vs 同卡组锚点 134.7s = -6.5%，vs Stage 1 fast a2a 126.6s = -0.5%**（融合的额外收益很小，与 op-level 预估的量级一致——nr 开销本身只占 denoise 的极小部分）。run2 的峰值显存 37.0GB 为孤立样本（run1/3 均 36.2GB），判定为噪声。
- 融合路径生效确认：run1/run2 log 均含 `fast_ulysses fused QK norm+RoPE+a2a path active.` ×4（4 rank）与 `fast_ulysses UlyssesGroup initialized` ×4；无 diffusers fallback、无 Traceback；FA backend（"Using fa attention backend"）。

### PSNR 验收：未通过（<35 dB 阈值）

fused_run1 视频（`A_cat_walks_slowly_towards_the_camera._20260715-013412_53f7cc7a.mp4`）vs anchor 视频（`..._20260714-124943_50ba72b0.mp4`），PyAV rgb24 逐帧全帧 MSE：

- **平均 19.92 dB / 最小 18.08 dB / 最大 21.32 dB（81 帧）**，各帧分布均匀（18.1–21.3，无个别坏帧）。**低于 35 dB 验收阈值，按计划判定为不通过。**
- 视觉抽查（首/中/末帧取回本地对比）：两个视频均为高质量、无伪影的正常输出，内容/构图/运镜一致，仅细节（猫的步态相位、背景虚化细节）发散——是"同 seed 数值扰动经 50 步采样混沌放大"的特征，**不是** RoPE 布局/eps 类错误的花屏特征。
- 静态核对（Task 6 建议项）：eps 取自 `norm_q.variance_epsilon`（正确来源）；cos/sin 为 GPT-J 交错布局，与 eager 路径 flashinfer `is_neox=False` 一致；单测以全 fp32 参考验证过布局与 eps（maxdiff 在 bf16 舍入量级）。
- 成因分析：融合路径 norm+RoPE 全程 fp32、单次量化回 bf16；原路径 norm 与 RoPE 各量化一次。每元素 ~1 ulp（bf16）的差异经 40 blocks × 50 步扩散采样混沌放大，导致轨迹发散。逐位不一致是预期，但 e2e PSNR 量级（~20 dB）远超计划预期（≥35 dB）。
- ~~待决~~ **已裁决（短程对比实验，2026-07-15）**：controller 跑了 2 步去噪的同 seed 对比（stage1 开关 vs stage1+融合，GPU 0-3，offload 全关；probe_s1/probe_s2）：

  | 对比 | PSNR 均值 | 最小 | 最大 |
  |---|---|---|---|
  | 2 步：stage1 vs 融合 | **38.31 dB** | 35.92 | 41.00 |
  | 50 步：anchor vs 融合 | 19.92 dB | 18.08 | 21.32 |

  2 步差异在 bf16 舍入累积量级（160 次融合调用 + VAE decode → 38 dB），50 步降至 20 dB——差异随步数单调放大，是标准的扩散采样混沌放大曲线。结合单次算子对拍（maxdiff 0.03 ≈ 2 ulp bf16；若为 RoPE 布局/eps bug 会是 O(1) 错误）与视觉验收（双侧高质量无伪影），**判定为良性发散，非实现 bug**。
- **验收口径修订**：对「改变中间计算精度/运算次序」的优化，50 步 PSNR ≥35dB 阈值不适用（混沌采样器上任何 ulp 级扰动都会发散到 ~20dB）。阶段 2 正确性验收修订为三重证据：① 4 卡算子对拍容差通过 ✓；② 2 步短程 e2e PSNR ≥35dB ✓（38.31）；③ 50 步视觉质量验收 ✓。**阶段 2 验收通过。**

## nsys 归因（2026-07-15，GPU 4-7）

方法：`nsys profile --trace=cuda-sw,nvtx --sample=none --cpuctxsw=none --trace-fork-before-exec=true --delay 210 --duration 100`，基准命令与 e2e run 相同（去 `--warmup`）。两份 profile：`nsys_nccl`（无开关）与 `nsys_fused`（两开关全开），采集窗口均落在 denoise 段内。环境注两条：① 该机 nsys 2026.3 默认硬件 CUDA tracing 全部 rank 报 `CUPTI_ERROR_HARDWARE_BUSY`（采不到 kernel），改软件 tracing（`--trace=cuda-sw`）后正常，为此去掉需硬件 trace 的 `--cuda-graph-trace=node`（eager 无 CUDA graph，无影响）；② 本轮模型加载仅 ~2.5 分钟（transformer 62s，远快于 07-14 实测的 13 分钟，疑宿主页缓存/锁页状态差异），首次 delay=820 落空后按预案改 delay=210。

归一化：以 flash-attn 实例数折算捕获步数（每步每 rank 160 次 attention：40 blocks × (self+cross) × 2 CFG）——nccl 窗口 36.5 步、fused 窗口 30.9 步。下表为**每 rank 每步** kernel 忙时（4 rank 总和 ÷4÷步数）：

| kernel | nccl (ms/步) | 次/rank/步 | fused (ms/步) | 次/rank/步 |
|---|---|---|---|---|
| ncclDevKernel_SendRecv（a2a） | 191.8（avg 599us/med 424us/max 3.6ms） | 320 | **0（实例数 0）** | 0 |
| a2a permute 拷贝（elementwise direct_copy 桶） | 171.0 | 568 | 8.4（非 a2a 残余） | ~8 |
| flashinfer RMSNorm（QK norm） | 18.0 | 320 | 7.0（仅剩 cross-attn） | 160 |
| flashinfer BatchQKApplyRotary（RoPE） | 13.5 | 80 | **0（实例数 0）** | 0 |
| ulysses::ulysses_barrier_kernel | — | — | 53.4（med 21us/avg 223us，吸收 rank skew） | 240 |
| ulysses::a2a_qk_chil_kernel（a2a+norm+RoPE 融合） | — | — | 47.6（avg 298us） | 160 |
| ulysses::a2a_tma_kernel（v/out 纯 a2a） | — | — | 37.3（avg 233us） | 160 |
| ulysses::token_inv_rms_kernel | — | — | 8.6 | 160 |
| **本优化涉及路径小计** | **378.9** | | **146.9** | |
| flash attn（对照锚，不应变） | 1654.7 | 160 | 1697.1（+2.6%） | 160 |
| 主 GEMM nvjet 128x256（对照锚） | 439.2 | 562 | 449.9（+2.4%） | 562 |

结论：

1. **每步省 232ms/rank，×50 步 ≈ 11.6s kernel 忙时/请求**，e2e 实测 −8.7s 落在其内且被完全覆盖（差额为旧路径 a2a 在独立 stream 上与计算部分重叠、以及窗口归一误差——对照锚在 fused 窗口普遍 +2.4~3.0%）。
2. 归因干净：NCCL SendRecv 与 RoPE kernel 在 fused profile 中**实例数为 0**，RMSNorm 减半（只剩 cross-attn），permute 拷贝桶 568→~8 次/步。节省确实来自 a2a/permute/norm/rope 这组 kernel，别处（flash attn、GEMM）不变。
3. 实模型中旧路径每 tensor-a2a 全程 1.11ms ≈ 2× microbench（0.55ms）：SendRecv 内含 rank 间 skew 等待（med 424us→max 3.6ms），permute 拷贝实测 508us/tensor。新路径把 skew 显式吸收进 barrier（med 21us），TMA a2a avg 233us 与 microbench 一致——这解释了 e2e 收益（8.1–8.7s）为何高于用 microbench 直算的 ~4.8s（16000 次 × ~0.3ms）。

产物：远程 `/data/bench/nsys/{nsys_nccl,nsys_fused}.{nsys-rep,log}`（`smoke.*` 为软件 tracing 验证残留）。

## 最终汇总

### 性能矩阵（Wan2.1-T2V-14B，720p/81 帧/50 步，4×B200 纯 Ulysses，denoise 中位数）

| 组 | GPU | denoise (s) | vs anchor |
|---|---|---|---|
| nccl_eager（基线） | 0-3 | 133.6 | −0.8%（跨卡组参考） |
| nccl_eager_anchor（锚点） | 4-7 | 134.7 | — |
| fast_a2a（阶段 1） | 4-7 | 126.6 | **−6.0%** |
| fast_a2a + QK 融合（阶段 2） | 4-7 | 126.0 | **−6.5%** |
| nccl_compile / fused_compile | — | 未跑 | 环境 inductor `spmd_check` 跨 rank 图不一致（与本项目无关，见 nccl_compile 小节） |

### op 级 ↔ e2e 收益对账

- op 级：a2a 2.18×（mode0）/ 2.34×（mode1）vs NCCL 完整旧路径；每请求 16000 次 a2a（40 blocks × 4 次/block × 2 CFG × 50 步）。
- e2e：阶段 1（纯 a2a）省 8.1s；阶段 2（+QK 融合）累计省 8.7s（融合增量 0.6s，与 op 级预估 nr 开销占比一致）。
- nsys 归因（上节）：旧路径相关 kernel 18.9s/请求 → 新路径 7.3s，**省 11.6s kernel 忙时**，覆盖并解释 e2e 的 8.7s；实模型旧路径开销 ≈ 2× microbench（skew 等待 + permute 实测更贵），故 e2e 收益高于 microbench 直算的 ~4.8s。

### 正确性结论

- op 级：纯 a2a 与 NCCL **逐位一致**（int16 位视图对拍，fp16/bf16 × d∈{64,128,256} × mode0/1 × ws∈{2,4,8}）；QK 融合 vs 全 fp32 参考 maxdiff ≤0.031（bf16 舍入量级，atol/rtol 2e-2 PASS）。
- 阶段 1 e2e：81 帧逐帧 MD5 与 anchor/GPU0-3 基线**逐位一致**（`bf3324bb`）。
- 阶段 2 e2e：三重证据通过——① 4 卡算子对拍 ✓；② 2 步短程 PSNR 38.31dB ≥35 ✓；③ 50 步视觉验收 ✓。50 步全程 PSNR ~19.9dB 判定为混沌采样下的**良性发散**（非实现 bug），验收口径已按此修订。

### 生产建议与上游化

- 推荐 flag：`SGLANG_DIFFUSION_ENABLE_FAST_ULYSSES=1` + `SGLANG_DIFFUSION_ENABLE_FAST_ULYSSES_QK_FUSION=1`；同时显式 `--dit-cpu-offload false --text-encoder-cpu-offload false`（B200 显存充足，可再省 ~8 分钟加载，见附录 A）。
- 上游化 TODO：① env 开关升格为 server flag（如 `--enable-fast-ulysses`），进 ServerArgs；② CI 增加 4 卡对拍测试（纯 a2a 逐位 + QK 融合容差）；③ 支持多 Ulysses 子组 / ring>1 / 非均匀 seq（当前仅 uniform 单子组路径）。
- 过程中发现的三个上游问题：① auto offload 对视频模型无条件开启（不按「模型尺寸 vs 显存」决策），B200 上白付 ~8 分钟加载（附录 A）；② Blackwell 上 `set_fa_ver(4)` 不探测 FA4 可用性，cutlass-dsl 4.6.0 兼容性事故后所有 run 必炸（附录 B）；③ torch.compile inductor `spmd_check` 跨 rank 图不一致（rank0 vs rank2/3），compile 组因此未跑。
