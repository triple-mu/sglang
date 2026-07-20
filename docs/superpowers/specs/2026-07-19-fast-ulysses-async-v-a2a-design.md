# fast-ulysses v-first 异步 input a2a（q/k/v 通信 overlap）设计文档

日期：2026-07-19
状态：已实现并验证（正确性红线全过；性能打平，开关保持默认关——见 results 文档 Stage 3）
目标场景：同 Stage-1/2 —— Wan2.1-T2V-14B，720p / 81 帧 / 50 步 / CFG，4×B200 纯 Ulysses（`--ulysses-degree=4`）

> 验证后修订（2026-07-19）：因 ion-b200 sshd 故障，实测改在 hyper00 4×H200（40 步/guidance 4）完成。
> §1 的收益预估未兑现：nsys 显示该 eager 管线 host 气泡本就吸收了 v 通信（scatter 落在 to_q/to_k 之间的
> 主 stream 空隙里，与 GEMM 重叠仅 2-9%，attention kernel ~44ms/块 占主导），async ≈ sync，e2e 差异 ≤0.2%。
> §4.4 的 use_tma 实验已做（bench-only），同样打平。B200 复验留待机器恢复。
前置：`docs/superpowers/specs/2026-07-14-fast-ulysses-integration-design.md`（Stage-1/2），其 §"范围外"曾把
`all_to_all_single_4d_async` 通信重叠列为后续工作；本文即该后续工作的第一步（Stage-3）。

## 1. 背景与目标

Stage-1/2 之后，Wan 的 uniform Ulysses 输入侧为：q/k 走 `qk2_input_a2a`（融合 cross-head RMSNorm + RoPE +
a2a，tag `usp_qk`），v 走普通 mode0 a2a（tag `usp_v`）——全部同步、在主 stream 上，通信全程在关键路径。

Wan block 内 QKV 是三次独立 GEMM（`wanvideo.py` 的 `to_q/to_k/to_v`），投影之后到 attention 之间没有其他
计算，因此输入侧唯一的天然 overlap 窗口是：**v 先投影，v 的 a2a 用库的 `all_to_all_single_4d_async`
（每组一条最高优先级 comm stream + event handle）异步发出，藏进 to_q/to_k 两次 GEMM**。
量级（4×B200，N=75600/H=40/D=128）：窗口 ≈ 两次 `[18900,5120]×[5120,5120]` GEMM ≈ 1.2ms，
v 的 mode0 a2a ≈ 253µs——理论上可完全隐藏。预期 denoise 收益 ~0.7-1.4%（对照 Stage-2 基线 126.0s）。

方案深度经用户确认为**仅 v-first 异步**：qk2 保持同步；不做 head 分块流水；不动 main 上 LTX-2 专用的
`enable_packed_qkv_input_a2a`（TE 风格 NCCL pipeline，与本路径结构性互斥）。

## 2. 设计

### 2.1 开关

`SGLANG_DIFFUSION_ENABLE_FAST_ULYSSES_ASYNC_V_A2A`（默认 False，实验性），依赖基础开关
`SGLANG_DIFFUSION_ENABLE_FAST_ULYSSES`（AND 语义，照 QK_FUSION 模式）。与 QK 融合正交：
v 无 norm/rope，融合开/关两种组合下 async v 均生效。

### 2.2 数据流（每个 WanTransformerBlock）

```
norm1 → to_v GEMM → v reshape [b,s_local,h,d]
      → maybe_async_input_a2a_v(v)  ──发到 comm stream（mode0, tag="usp_v"）──┐
      → to_q GEMM → to_k GEMM（主 stream，与 v a2a 重叠）                     │
      →（非融合分支还有 norm_q/norm_k + RoPE，也在重叠窗口内）                │
      → attn1(q, k, v, qk_fused_ctx=…, v_a2a_handle=hv)                       │
           v = hv.wait()   ←──必须先 wait（barrier epoch 顺序约束）───────────┘
           → qk2_input_a2a（融合）或 q/k 各自同步 fast a2a（非融合）
           → attention → out a2a（mode1, "usp_out"，不变）
```

### 2.3 接入点

- `fast_ulysses_backend.maybe_async_input_a2a_v(v)`：护栏顺序 env → group → shape → BCG capture，
  全部 rank-uniform；不能服务时返回 None（调用方回落现状路径）。计数器 `fast_call_counts["a2a_async"]`。
- `WanTransformerBlock._maybe_async_v_input_a2a`：模型侧只判 attn1 是 USPAttention 且非
  skip_sequence_parallel（照 `_maybe_fast_qk_fused_ctx` 的 lockstep 先例）。
- `USPAttention.forward` 新 kw 参数 `v_a2a_handle`（与 `qk_fused_ctx` 平级）；分派优先级
  handle → qk_fused_ctx → enable_packed_qkv_input_a2a → 普通路径；handle 为 None 时控制流与现状一致。
- uniform-path assert：handle 非 None 时要求无 mask / 无 attn_mask_meta（tail-pad meta 单独置位也会进
  masked 分支）/ 无 replicated / 非 skip-sp / 非 packed——流入不 wait 的分支既数值错误又是全组
  epoch 乱序（hang）隐患，fail-fast。

### 2.4 关键不变量（库约束，fast_ulysses comm.py）

1. **epoch 顺序**：每组 barrier epoch 是单调计数器，主 stream 发下一个同步 fast collective 前必须先
   `handle.wait()`。本设计中 wait 就位于分派处、任何 q/k collective 之前；issue→wait 之间主 stream
   只有本地 kernel（GEMM/norm/rope），无 collective。
2. **rank-uniform 调用序**：async 发起判定全部由 rank 恒等条件构成（env、group 拓扑、shape、BCG 标志）。
3. **CUDA graph**：`_launch_on_comm_stream` 用 `Tensor.record_stream` + 跨 stream event，capture 下非法；
   `is_in_breakable_cuda_graph()` 时返回 None（Wan 亦不在 BCG 白名单，双保险）。

### 2.5 "usp_v" 对称堆 buffer 跨层复用在 async 下的安全性

风险：layer N 的 async scatter 远程写所有 peer 的 usp_v buffer，须证明 peer p 的最后读者
（p 的 layer N−1 attention）先完成。两段链条：
1. **本 rank**：`ev_ready` 在主 stream 记录，trails 本层之前提交的全部工作 ⇒ 本地 scatter 在本地
   layer N−1 out a2a 之后执行。
2. **跨 rank**：rank r 过 layer N−1 out a2a 的 flag barrier ⇔ 所有 rank（含 p）已到达该 epoch；
   p 的到达信号在其主 stream 上排在 out-a2a scatter 之后，而后者数据依赖 attention 输出 ⇒ p 到达时
   其 attention（usp_v 的最后读者）必已完成。∎

首调 autotune microbench 在 comm stream 上执行且 host 阻塞数十 ms：一次性（warmup 吸收）、全 rank
lockstep，可接受。

## 3. 正确性红线

v-first 重排 + async 不改变任何 kernel 的输入（三个投影 GEMM 相互独立、同输入 `norm_hidden_states`；
a2a 是逐位搬运、async 与 sync 同 kernel 同 buffer 仅提交 stream 不同；qk2 输入不变）⇒ 端到端帧必须与
不开 async 的对应 fast 配置**逐位一致**（MD5），任何偏差即实现 bug。本特性没有数值容差借口。

## 4. 验收

1. 4 卡单测 `test_fast_ulysses_a2a.py`：async v 与 NCCL 逐位对拍 ×3 层（buffer 复用）、与同步 qk2 混合
   顺序、计数断言；共 4 条命令（含原有 2 条回归）。
2. e2e 帧 MD5：`(FAST,QK,ASYNC)` ≡ Stage-2 产物；`(FAST,ASYNC)` ≡ Stage-1 产物。
3. denoise walltime：同卡组重锚 Stage-2 基线，3 次中位。
4. nsys：v scatter 与 to_q/to_k GEMM 时间区间真实重叠；若 SM 争抢零重叠（库文档 H200 结论，B200 待验证），
   bench-only 试 `use_tma=True`（几乎不占 SM；全 rank 必须同值），择优后再决定是否固化参数。

## 5. 范围外

- head 分块的通信-attention 深度流水（收益上限更高，视本阶段 nsys 数据另立项）。
- qk2 的 async 化（wait 与 issue 之间无主 stream 计算，无收益）。
- 输出侧 out a2a 的 overlap（to_out GEMM 数据依赖其全部输出，无窗口）。
- LTX-2 / VSA 变体 / varlen / replicated 路径。
