# fast-ulysses 集成实测记录

环境：ion-b200，容器 `sglang-diffusion-triplemu`，8×B200（全 NVLink），本次使用 GPU 0-3（`CUDA_VISIBLE_DEVICES=0,1,2,3`，ws=4）。NVSHMEM v3.6.5。库源码 `/data/fast-ulysses`（只读），fast_ulysses wheel 已 pip 安装。

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

### qk 融合基准（`bench_qk_fused.py`，ws=4, dtype=bf16, n_global=40, d=128，单位 ms/iter）

| seq_global | a2a | unfused | fused | fused/unfused | nr_unfused | nr_fused | qk2 | qk2/2fused |
|---|---|---|---|---|---|---|---|---|
| 20480 | 0.080 | 0.225 | 0.121 | 1.856× | 0.144 | 0.041 | 0.240 | 0.992× |
| 46080 | 0.152 | 0.470 | 0.241 | 1.949× | 0.319 | 0.090 | 0.486 | 1.007× |

说明：nr_* 为 norm+rope 相对纯 a2a 的额外开销；qk2 为 q+k 一次调用（1 个 barrier）。

### 结论

- 正确性：两个测试全部 PASS，B200 (ws=4) 上库可用。
- a2a 加速比 mode0 2.18× / mode1 2.34×（vs NCCL），远高于 1.1× 继续阈值。
- fused norm+rope+a2a 相对 unfused 约 1.86–1.95×；qk2 与 2× fused 基本持平（收益在减少 barrier）。
- 判定：继续后续任务。
