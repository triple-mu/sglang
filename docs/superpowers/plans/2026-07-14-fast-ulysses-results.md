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
