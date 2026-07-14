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
