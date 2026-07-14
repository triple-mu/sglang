# fast-ulysses 融合进 SGLang Diffusion 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 fast-ulysses（NVSHMEM+NVLink P2P 的 Ulysses a2a 与 QK norm/RoPE 融合算子）以实验开关接入 sglang diffusion 的 USPAttention 路径，并在远程 4×B200 上实测 Wan2.1-T2V-14B（720p/81帧/50步）的 denoise latency 提升。

**Architecture:** 两阶段。阶段 1 在 `usp.py` 收口点把等长 Ulysses a2a 换成 `UlyssesGroup.all_to_all_single_4d`（逐位等价、省全部 permute）；阶段 2 在 Wan block 把 QK RMSNorm+RoPE+q/k 进注意力 a2a 融成一次 `all_to_all_single_4d_qk2`。新 wrapper 模块 `fast_ulysses_backend.py` 集中所有启用判定与 fallback，两个 env 开关分别控制两阶段。

**Tech Stack:** PyTorch 2.13 (NGC 26.06) / CUDA 13.3 / NVSHMEM 3.7.1 / fast_ulysses 0.0.1（已装于远程容器）/ sglang multimodal_gen 运行时。

**Spec:** `docs/superpowers/specs/2026-07-14-fast-ulysses-integration-design.md`

**变更记录（2026-07-14，用户指示）：** 目标模型由 Wan-AI/Wan2.1-I2V-14B-720P-Diffusers 改为
**Wan-AI/Wan2.1-T2V-14B-Diffusers**（T2V 任务，无图像条件：基准命令去掉 `--image-path`，
cat.png 步骤作废，其余口径不变——T2V 14B 默认即 720p/81帧，序列长度与集成设计不受影响）。
所有性能测试必须先确认 GPU 0-3 空闲（util 0%、无残留进程）；Task 1 的 op 级数字在空闲机上复测。

## Global Constraints

- 实验形态：只加 env var 开关（`SGLANG_DIFFUSION_ENABLE_FAST_ULYSSES`、`SGLANG_DIFFUSION_ENABLE_FAST_ULYSSES_QK_FUSION`，均默认关），不加 server flag、不注册 CI。
- 集体安全：fast path 的每个启用判定必须在 SP rank 间恒等（env、组拓扑、shape/dtype），否则 NVSHMEM 集体会 hang。
- fallback 原则：任何不满足条件的调用透明回落 NCCL 路径，绝不改变默认行为（开关关闭时零行为差异）。
- 远程访问：`ssh ion-b200 "docker exec [-e K=V ...] sglang-diffusion-triplemu bash -c '<cmd>'"`；文件同步用 `tar | ssh docker exec -i tar -x`（宿主机看不到 /data）。
- 远程 `/data/sglang/python/pyproject.toml` 有本地改动（放开 torch 钉版），**永远不要同步覆盖它**。
- 磁盘：HF 默认缓存盘已满，一切权重/输出/结果放 `/data`（`HF_HOME=/data/hf-cache`）。
- 基准口径：denoise latency（perf dump DenoisingStage 之和）、seed 42、`--warmup`、layerwise offload 关、每配置 3 次取中位。
- 代码规范：新数据容器用 `msgspec.Struct`；2+ 参数按 keyword 调用；不用 getattr 防御；不加计划外的"优化"。
- git：全部工作在本地分支 `feat/fast-ulysses-integration` 上，逐任务 commit（spec/plan 文档随首个 commit 入库）。

## 关键背景（执行者需知）

- 本地仓库 `/home/ubuntu/workspace/github/llm/sglang`，远程容器内 `/data/sglang`（editable 安装，改 .py 即时生效），两边 HEAD 同为 1a35440c4。
- Wan 主路径：`WanTransformerBlock.forward`（`wanvideo.py:486`）→ norm_q/k（:529-538，cross-head RMSNorm(5120)）→ unflatten (b,s,40,128)（:539-541）→ RoPE（:543-578，GPT-J interleaved）→ `self.attn1(q,k,v)`（:579，USPAttention）→ layer.py:970-972 q/k/v 各一次 `_usp_input_all_to_all(x, head_dim=2)`（mode0 语义）→ attention → layer.py:991 `_usp_output_all_to_all(out, head_dim=2)`（mode1 语义）。
- 720p 81帧 4 卡：S_local=18900、h=40、h_local=10、d=128、bf16、b=1（CFG 两次串行 forward）。
- fast_ulysses 集体硬约束：所有 rank 以相同 (shape, mode, use_tma) 序列调用；并存结果必须不同 tag（对称堆输出缓冲按 tag+shape+dtype 复用）。
- fast_ulysses 融合 op 要求 weight/cos/sin 为 fp32；Wan 的 freqs_cis 已是 `(S_local, 64) fp32` 两表（`wanvideo.py:1082`），norm weight 为 bf16 需一次性转 fp32 缓存。

---

### Task 1: 建分支 + 远程 op 级验证（4×B200 收益上限）

**Files:** 无代码改动。产出：基准记录文件 `docs/superpowers/plans/2026-07-14-fast-ulysses-results.md`（后续任务持续追加）。

**Interfaces:**
- Produces: B200 ws=4 的 op 级正确性结论与 a2a/qk2 加速比（决定是否继续）。

- [ ] **Step 1: 建本地分支并提交 spec/plan**

```bash
cd /home/ubuntu/workspace/github/llm/sglang
git checkout -b feat/fast-ulysses-integration
git add docs/superpowers/specs/2026-07-14-fast-ulysses-integration-design.md \
        docs/superpowers/plans/2026-07-14-fast-ulysses-integration.md
git commit -m "docs: fast-ulysses integration spec and plan"
```

- [ ] **Step 2: 远程跑 fast-ulysses 自带正确性测试（ws=4）**

```bash
ssh ion-b200 "docker exec -e CUDA_VISIBLE_DEVICES=0,1,2,3 sglang-diffusion-triplemu bash -c \
  'cd /data/fast-ulysses && torchrun --nproc_per_node=4 test/test_correctness.py'"
ssh ion-b200 "docker exec -e CUDA_VISIBLE_DEVICES=0,1,2,3 sglang-diffusion-triplemu bash -c \
  'cd /data/fast-ulysses && torchrun --nproc_per_node=4 test/test_a2a_qk.py'"
```

Expected: 两个测试全部 PASS（逐位/容差对拍通过）。若 FAIL 或 NVSHMEM 初始化 segfault：停止，向用户报告 B200 上库本身不可用——不进入后续任务。

- [ ] **Step 3: 远程跑 op 级 microbenchmark（Wan shape，ws=4）**

```bash
ssh ion-b200 "docker exec -e CUDA_VISIBLE_DEVICES=0,1,2,3 sglang-diffusion-triplemu bash -c \
  'cd /data/fast-ulysses && PROF_N=75600 PROF_H=40 PROF_D=128 PROF_MODE=0 torchrun --nproc_per_node=4 benchmark/bench_uniform.py'"
# mode1 同 shape：
ssh ion-b200 "docker exec -e CUDA_VISIBLE_DEVICES=0,1,2,3 sglang-diffusion-triplemu bash -c \
  'cd /data/fast-ulysses && PROF_N=75600 PROF_H=40 PROF_D=128 PROF_MODE=1 torchrun --nproc_per_node=4 benchmark/bench_uniform.py'"
# qk 融合基准：
ssh ion-b200 "docker exec -e CUDA_VISIBLE_DEVICES=0,1,2,3 sglang-diffusion-triplemu bash -c \
  'cd /data/fast-ulysses && torchrun --nproc_per_node=4 benchmark/bench_qk_fused.py'"
```

Expected: 输出 ours vs NCCL 的 GB/s 与加速比。把三组数字（mode0/mode1/qk_fused）记入 `docs/superpowers/plans/2026-07-14-fast-ulysses-results.md` 新建的 "Op-level (4×B200)" 小节。
判定：若 a2a 加速比 < 1.1×，向用户报告收益上限过低并暂停等待指示；≥1.1× 继续。

- [ ] **Step 4: Commit 结果记录**

```bash
git add docs/superpowers/plans/2026-07-14-fast-ulysses-results.md
git commit -m "bench: fast-ulysses op-level numbers on 4xB200"
```

---

### Task 2: 权重下载 + 基准资产 + NCCL baseline

**Files:** 无代码改动。产出：远程 `/data/hf-cache`（权重）、`/data/bench/cat.png`、`/data/bench/results/*.json`，结果追加到 results 文档。

**Interfaces:**
- Produces: baseline perf dump `nccl_eager_run{1..3}.json`、`nccl_compile_run{1..3}.json`；后续任务用 `benchmarks/compare_perf.py` 与之对比。

- [ ] **Step 1: 后台下载权重到 /data（~70GB，早启动）**

```bash
ssh ion-b200 "docker exec -e HF_HOME=/data/hf-cache sglang-diffusion-triplemu bash -c \
  'mkdir -p /data/bench/results && nohup python -c \"from huggingface_hub import snapshot_download; snapshot_download(\\\"Wan-AI/Wan2.1-T2V-14B-Diffusers\\\")\" > /data/bench/download.log 2>&1 & echo started'"
```

轮询 `tail -2 /data/bench/download.log` 直到完成再进 Step 3（期间可先做 Task 3 的本地代码）。

- [ ] **Step 2: 准备参考图（作废——T2V 无图像条件，跳过）**

```bash
ssh ion-b200 "docker exec sglang-diffusion-triplemu bash -c \
  'wget -q -O /data/bench/cat.png https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png && ls -la /data/bench/cat.png'"
```

- [ ] **Step 3: 确认 offload 关闭 flag 的确切形式**

```bash
ssh ion-b200 "docker exec sglang-diffusion-triplemu bash -c 'cd /data/sglang && sglang generate --help 2>&1 | grep -i -A2 offload'"
```

记下 dit layerwise offload 的布尔 flag 写法（形如 `--dit-layerwise-offload false` 或 `--no-dit-layerwise-offload`），后续所有基准命令用它。

- [ ] **Step 4: 跑 NCCL baseline（eager 3 次 + compile 3 次）**

基准命令模板（`<LABEL>`/`<I>`/`<EXTRA>` 替换；这条模板也是 Task 5/7 的基准命令）：

```bash
ssh ion-b200 "docker exec -e HF_HOME=/data/hf-cache -e CUDA_VISIBLE_DEVICES=0,1,2,3 \
  -e FLASHINFER_DISABLE_VERSION_CHECK=1 <EXTRA_ENV> sglang-diffusion-triplemu bash -c \
  'cd /data/sglang && sglang generate \
    --backend=sglang \
    --model-path=Wan-AI/Wan2.1-T2V-14B-Diffusers \
    --prompt=\"A cat walks slowly towards the camera.\" \
    --width=1280 --height=720 --num-frames=81 --num-inference-steps=50 --seed=42 \
    --num-gpus=4 --ulysses-degree=4 --ring-degree=1 --cfg-parallel-size=1 \
    --text-encoder-cpu-offload --pin-cpu-memory \
    <OFFLOAD_OFF_FLAG> \
    --warmup <EXTRA_FLAGS> \
    --perf-dump-path /data/bench/results/<LABEL>_run<I>.json'"
```

跑 `nccl_eager`（无 EXTRA_FLAGS）×3、`nccl_compile`（`--enable-torch-compile`）×3。
每次检查：日志**不得**出现 `Falling back to diffusers backend`（出现则该次作废）；记录 `(with warmup excluded)` 行。

Expected: 6 个 JSON 落盘。取回并记录 denoise 中位数：

```bash
scp ion-b200:/tmp/... # 若直接 scp 不可行：
ssh ion-b200 "docker exec sglang-diffusion-triplemu bash -c 'cat /data/bench/results/nccl_eager_run1.json'" > /tmp/bench/nccl_eager_run1.json
# （逐个取回，或 docker exec tar -c 打包）
```

把每次 run 的 denoise(s)/e2e(s)/峰值显存与中位数追加到 results 文档 "E2E baseline" 小节。

- [ ] **Step 5: Commit 结果记录**

```bash
git add docs/superpowers/plans/2026-07-14-fast-ulysses-results.md
git commit -m "bench: Wan2.1 I2V 720p 4xB200 NCCL baseline"
```

---

### Task 3: env 开关 + fast_ulysses_backend wrapper（阶段 1 部分）

**Files:**
- Modify: `python/sglang/multimodal_gen/envs.py`（TYPE_CHECKING 块 + environment_variables 字典，参照 `SGLANG_DIFFUSION_SERVER_DEV_MODE` 的 `_lazy_bool` 模式）
- Create: `python/sglang/multimodal_gen/runtime/layers/fast_ulysses_backend.py`

**Interfaces:**
- Produces: `fast_ulysses_backend.maybe_all_to_all_4d(x, *, mode: int, tag: str) -> torch.Tensor | None`（None=调用方走 NCCL 路径）；`fast_ulysses_backend.fast_call_counts: dict[str, int]`（测试断言 fast path 确实生效）。

- [ ] **Step 1: envs.py 增加两个开关**

TYPE_CHECKING 块（放 `SGLANG_DIFFUSION_SERVER_DEV_MODE: bool = False` 附近）：

```python
    SGLANG_DIFFUSION_ENABLE_FAST_ULYSSES: bool = False
    SGLANG_DIFFUSION_ENABLE_FAST_ULYSSES_QK_FUSION: bool = False
```

`environment_variables` 字典（同区域，带注释）：

```python
    # Experimental: route uniform Ulysses SP all-to-all through the
    # fast_ulysses (NVSHMEM + NVLink P2P) library when importable.
    "SGLANG_DIFFUSION_ENABLE_FAST_ULYSSES": _lazy_bool(
        "SGLANG_DIFFUSION_ENABLE_FAST_ULYSSES"
    ),
    # Experimental: additionally fuse Wan-style cross-head QK RMSNorm + RoPE
    # into the fast_ulysses input all-to-all (requires the flag above).
    "SGLANG_DIFFUSION_ENABLE_FAST_ULYSSES_QK_FUSION": _lazy_bool(
        "SGLANG_DIFFUSION_ENABLE_FAST_ULYSSES_QK_FUSION"
    ),
```

注意：先读 envs.py 里 `_lazy_bool` 的实际签名，若需要显式 default 参数则补 `False`。

- [ ] **Step 2: 写 wrapper 模块**

`python/sglang/multimodal_gen/runtime/layers/fast_ulysses_backend.py` 完整内容：

```python
"""Experimental fast_ulysses (NVSHMEM + NVLink P2P) backend for Ulysses SP a2a.

Enabled via SGLANG_DIFFUSION_ENABLE_FAST_ULYSSES (plus
SGLANG_DIFFUSION_ENABLE_FAST_ULYSSES_QK_FUSION for the fused QK norm+RoPE
input a2a). Every entry point returns None / False when the fast path cannot
serve the call, and callers must then run the NCCL path.

Collective safety: every decision here (env flags, group topology, tensor
shape/dtype) is identical across SP ranks under SPMD execution, so for a given
call site either all ranks take the fast path or none does.
"""

import functools
import logging

import msgspec
import torch
import torch.distributed as dist

from sglang.multimodal_gen import envs
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_sp_group,
    get_ulysses_parallel_world_size,
)

logger = logging.getLogger(__name__)

# fast_ulysses supports single-node NVLink P2P, world sizes 1..8; below 2
# there is nothing to communicate.
_MIN_WORLD_SIZE = 2
_MAX_WORLD_SIZE = 8

# Calls actually served by the fast path; tests assert on these to catch
# silent fallbacks.
fast_call_counts: dict[str, int] = {"a2a": 0, "qk2": 0}


class QKFusedCtx(msgspec.Struct):
    """Inputs for the fused cross-head QK RMSNorm + RoPE + input a2a."""

    weight_q: torch.Tensor  # fp32 [num_heads * head_dim]
    weight_k: torch.Tensor  # fp32 [num_heads * head_dim]
    cos: torch.Tensor  # fp32 [s_local, head_dim // 2]
    sin: torch.Tensor  # fp32 [s_local, head_dim // 2]
    eps: float


@functools.cache
def _a2a_enabled() -> bool:
    return bool(envs.SGLANG_DIFFUSION_ENABLE_FAST_ULYSSES)


@functools.cache
def _qk_fusion_enabled() -> bool:
    return _a2a_enabled() and bool(
        envs.SGLANG_DIFFUSION_ENABLE_FAST_ULYSSES_QK_FUSION
    )


@functools.cache
def _get_group():
    """Build the process-wide UlyssesGroup once; None means permanent fallback.

    Lazy construction is collective-safe: under SPMD every SP rank reaches the
    first fast-path call in lockstep, and UlyssesGroup.__init__ runs its own
    broadcast + barriers on the bootstrap group.
    """
    if not _a2a_enabled():
        return None
    try:
        from fast_ulysses import UlyssesGroup
    except ImportError:
        logger.warning(
            "SGLANG_DIFFUSION_ENABLE_FAST_ULYSSES is set but fast_ulysses is "
            "not importable; keeping the NCCL all-to-all path."
        )
        return None
    ulysses_pg = get_sp_group().ulysses_group
    ulysses_world = dist.get_world_size(group=ulysses_pg)
    if ulysses_world != dist.get_world_size():
        # cfg-parallel / tp / ring layouts would need one NVSHMEM domain per
        # sub-group, which is unvalidated; keep NCCL there.
        logger.warning(
            "fast_ulysses disabled: ulysses group has %d ranks but world has "
            "%d; only pure-Ulysses layouts are supported.",
            ulysses_world,
            dist.get_world_size(),
        )
        return None
    if not (_MIN_WORLD_SIZE <= ulysses_world <= _MAX_WORLD_SIZE):
        logger.warning(
            "fast_ulysses disabled: ulysses world size %d outside [%d, %d].",
            ulysses_world,
            _MIN_WORLD_SIZE,
            _MAX_WORLD_SIZE,
        )
        return None
    group = UlyssesGroup(process_group=ulysses_pg)
    logger.info("fast_ulysses UlyssesGroup initialized (world=%d).", ulysses_world)
    return group


def _shape_ok(x: torch.Tensor, *, split_dim: int, world_size: int) -> bool:
    return (
        x.ndim == 4
        and x.is_cuda
        and x.dtype in (torch.bfloat16, torch.float16)
        and x.shape[split_dim] % world_size == 0
        and (x.shape[3] * x.element_size()) % 16 == 0
    )


def maybe_all_to_all_4d(
    x: torch.Tensor, *, mode: int, tag: str
) -> torch.Tensor | None:
    """Fast-path replacement for the uniform Ulysses a2a on [b, s, h, d].

    mode 0: (b, s_local, h, d) -> (b, s_global, h_local, d)   into attention
    mode 1: (b, s_global, h_local, d) -> (b, s_local, h, d)   out of attention

    Returns None whenever the fast path cannot serve the call. ``tag`` must be
    unique among results alive at the same time (symmetric-heap output buffers
    are reused per tag); an empty tag disables the fast path.
    """
    if not tag or not _a2a_enabled():
        return None
    group = _get_group()
    if group is None:
        return None
    split_dim = 2 if mode == 0 else 1
    if not _shape_ok(
        x, split_dim=split_dim, world_size=get_ulysses_parallel_world_size()
    ):
        return None
    out = group.all_to_all_single_4d(x, mode=mode, tag=tag)
    fast_call_counts["a2a"] += 1
    return out


def can_use_qk2(*, num_heads: int, head_dim: int, dtype: torch.dtype) -> bool:
    """Rank-invariant pre-check so the model can skip its own norm/rope."""
    if not _qk_fusion_enabled():
        return False
    if _get_group() is None:
        return False
    world_size = get_ulysses_parallel_world_size()
    return (
        dtype in (torch.bfloat16, torch.float16)
        and num_heads % world_size == 0
        and head_dim >= 2
        and (head_dim & (head_dim - 1)) == 0
        and head_dim * dtype.itemsize >= 32
    )


def qk2_input_a2a(
    q: torch.Tensor, k: torch.Tensor, *, ctx: QKFusedCtx
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused cross-head RMSNorm + RoPE + mode0 a2a for q and k in one
    collective. Caller must have checked can_use_qk2()."""
    group = _get_group()
    assert group is not None, "qk2_input_a2a requires can_use_qk2() == True"
    if fast_call_counts["qk2"] == 0:
        logger.info("fast_ulysses fused QK norm+RoPE+a2a path active.")
    out_q, out_k = group.all_to_all_single_4d_qk2(
        q,
        k,
        ctx.weight_q,
        ctx.weight_k,
        ctx.cos,
        ctx.sin,
        mode="cross_head",
        interleaved=True,
        eps=ctx.eps,
        tag="usp_qk",
    )
    fast_call_counts["qk2"] += 1
    return out_q, out_k
```

（`QKFusedCtx` / `can_use_qk2` / `qk2_input_a2a` 在 Task 6 才被消费，但一次成型避免文件反复动。）

- [ ] **Step 3: 本地 import 冒烟**

```bash
cd /home/ubuntu/workspace/github/llm/sglang
python -c "from sglang.multimodal_gen.runtime.layers import fast_ulysses_backend; print('ok')"
```

Expected: `ok`（本机无 fast_ulysses/GPU 也应可 import——fast_ulysses 的 import 是懒惰的）。

- [ ] **Step 4: Commit**

```bash
git add python/sglang/multimodal_gen/envs.py \
        python/sglang/multimodal_gen/runtime/layers/fast_ulysses_backend.py
git commit -m "feat(diffusion): experimental fast_ulysses backend wrapper and env flags"
```

---

### Task 4: usp.py / layer.py 接线 + 4 卡对拍测试（阶段 1 完成）

**Files:**
- Modify: `python/sglang/multimodal_gen/runtime/layers/usp.py:69-122`（`_usp_input_all_to_all`）、`usp.py:202-255`（`_usp_output_all_to_all`）
- Modify: `python/sglang/multimodal_gen/runtime/layers/attention/layer.py:970-972, 991`
- Create: `python/sglang/multimodal_gen/test/unit/test_fast_ulysses_a2a.py`

**Interfaces:**
- Consumes: `fast_ulysses_backend.maybe_all_to_all_4d`、`fast_call_counts`（Task 3）。
- Produces: `_usp_input_all_to_all(x, head_dim=1, comm_tag="")` / `_usp_output_all_to_all(x, head_dim=1, comm_tag="")` 新参数；layer.py 主路径 tag 约定 `usp_q / usp_k / usp_v / usp_out`（Task 6 复用 `usp_v`）。

- [ ] **Step 1: 先写对拍测试（完整文件）**

`python/sglang/multimodal_gen/test/unit/test_fast_ulysses_a2a.py`：

```python
"""4-GPU parity tests: fast_ulysses backend vs NCCL Ulysses a2a.

Not registered in CI (experimental). Run inside the ion-b200 container:

    # expected FAIL (fast path off -> counter assertion trips):
    torchrun --nproc_per_node=4 \
        python/sglang/multimodal_gen/test/unit/test_fast_ulysses_a2a.py
    # expected PASS:
    SGLANG_DIFFUSION_ENABLE_FAST_ULYSSES=1 \
    SGLANG_DIFFUSION_ENABLE_FAST_ULYSSES_QK_FUSION=1 \
    torchrun --nproc_per_node=4 \
        python/sglang/multimodal_gen/test/unit/test_fast_ulysses_a2a.py
"""

import os

import torch

from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    maybe_init_distributed_environment_and_model_parallel,
)
from sglang.multimodal_gen.runtime.layers import fast_ulysses_backend
from sglang.multimodal_gen.runtime.layers.usp import (
    _usp_input_all_to_all,
    _usp_output_all_to_all,
)

B, S_LOCAL, H, D = 1, 18900, 40, 128  # Wan2.1 14B 720p/81f, sp=4


def _rand(shape, seed, device):
    gen = torch.Generator(device=device).manual_seed(seed)
    return torch.randn(shape, generator=gen, device=device, dtype=torch.bfloat16)


def test_a2a_parity(rank: int, device: torch.device, world: int) -> None:
    x = _rand((B, S_LOCAL, H, D), 1234 + rank, device)
    ref = _usp_input_all_to_all(x, head_dim=2)  # no tag -> NCCL
    fast = _usp_input_all_to_all(x, head_dim=2, comm_tag="t_in")
    assert torch.equal(ref, fast), "mode0 mismatch"

    y = _rand((B, S_LOCAL * world, H // world, D), 4321 + rank, device)
    ref = _usp_output_all_to_all(y, head_dim=2)
    fast = _usp_output_all_to_all(y, head_dim=2, comm_tag="t_out")
    assert torch.equal(ref, fast), "mode1 mismatch"


def _ref_norm_rope(x, weight, cos, sin, eps):
    """Cross-head RMSNorm + interleaved (GPT-J) RoPE, all-fp32 reference."""
    b, s, n, d = x.shape
    xf = x.float().reshape(b, s, n * d)
    inv = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    xf = (xf * inv * weight.float()).reshape(b, s, n, d)
    x1, x2 = xf[..., 0::2], xf[..., 1::2]
    c = cos.view(1, s, 1, d // 2)
    si = sin.view(1, s, 1, d // 2)
    out = torch.empty_like(xf)
    out[..., 0::2] = x1 * c - x2 * si
    out[..., 1::2] = x2 * c + x1 * si
    return out.to(x.dtype)


def test_qk2_parity(rank: int, device: torch.device, world: int) -> None:
    q = _rand((B, S_LOCAL, H, D), 111 + rank, device)
    k = _rand((B, S_LOCAL, H, D), 222 + rank, device)
    gen = torch.Generator(device=device).manual_seed(9)
    wq = 1.0 + 0.1 * torch.randn(H * D, generator=gen, device=device)
    wk = 1.0 + 0.1 * torch.randn(H * D, generator=gen, device=device)
    ang = torch.rand((S_LOCAL, D // 2), generator=gen, device=device) * 6.28
    cos, sin = ang.cos().contiguous(), ang.sin().contiguous()
    eps = 1e-6

    assert fast_ulysses_backend.can_use_qk2(
        num_heads=H, head_dim=D, dtype=torch.bfloat16
    )
    ctx = fast_ulysses_backend.QKFusedCtx(
        weight_q=wq, weight_k=wk, cos=cos, sin=sin, eps=eps
    )
    fq, fk = fast_ulysses_backend.qk2_input_a2a(q=q, k=k, ctx=ctx)

    rq = _usp_input_all_to_all(_ref_norm_rope(q, wq, cos, sin, eps), head_dim=2)
    rk = _usp_input_all_to_all(_ref_norm_rope(k, wk, cos, sin, eps), head_dim=2)
    for name, f, r in (("q", fq, rq), ("k", fk, rk)):
        diff = (f.float() - r.float()).abs().max().item()
        if rank == 0:
            print(f"qk2 {name}: max abs diff vs fp32 ref = {diff:.5f}", flush=True)
        torch.testing.assert_close(f, r, atol=2e-2, rtol=2e-2)


def main() -> None:
    maybe_init_distributed_environment_and_model_parallel(
        tp_size=1, sp_size=4, ulysses_degree=4
    )
    import torch.distributed as dist

    rank = dist.get_rank()
    world = dist.get_world_size()
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", rank)))

    test_a2a_parity(rank, device, world)
    assert (
        fast_ulysses_backend.fast_call_counts["a2a"] == 2
    ), f"fast a2a path not exercised: {fast_ulysses_backend.fast_call_counts}"

    if fast_ulysses_backend.can_use_qk2(
        num_heads=H, head_dim=D, dtype=torch.bfloat16
    ):
        test_qk2_parity(rank, device, world)
        assert fast_ulysses_backend.fast_call_counts["qk2"] >= 1
    elif rank == 0:
        print("qk2 disabled; skipped its parity test", flush=True)

    if rank == 0:
        print("PASS", flush=True)


if __name__ == "__main__":
    main()
```

（qk2 部分在 Task 6 前因 `can_use_qk2` 依赖的函数已在 Task 3 一并落地而可直接跑。）

- [ ] **Step 2: usp.py 接线**

`_usp_input_all_to_all` 签名改为 `def _usp_input_all_to_all(x: torch.Tensor, head_dim: int = 1, comm_tag: str = "") -> torch.Tensor:`，并在 `assert head_dim in (1, 2)`（usp.py:92）之后插入：

```python
    if head_dim == 2 and comm_tag:
        # Experimental NVSHMEM fast path; consumes [b, s, h, d] directly, so
        # the permute/contiguous round-trips below are skipped entirely.
        fast_out = fast_ulysses_backend.maybe_all_to_all_4d(
            x, mode=0, tag=comm_tag
        )
        if fast_out is not None:
            return fast_out
```

`_usp_output_all_to_all` 同样加 `comm_tag: str = ""` 参数，在 `assert head_dim in (1, 2)`（usp.py:225）之后插入相同分支（`mode=1`）。

文件头部 import（与现有 import 块并列）：

```python
from sglang.multimodal_gen.runtime.layers import fast_ulysses_backend
```

注意：确认无循环 import（fast_ulysses_backend 只 import parallel_state 与 envs，安全）。

- [ ] **Step 3: layer.py 主路径传 tag**

layer.py:970-972 改为：

```python
                q = _usp_input_all_to_all(q, head_dim=2, comm_tag="usp_q")
                k = _usp_input_all_to_all(k, head_dim=2, comm_tag="usp_k")
                v = _usp_input_all_to_all(v, head_dim=2, comm_tag="usp_v")
```

layer.py:991 改为：

```python
            out = _usp_output_all_to_all(out, head_dim=2, comm_tag="usp_out")
```

其余调用点（mask 分支等）不动——不传 tag 即永远走 NCCL。

- [ ] **Step 4: 同步远程（本计划的标准同步命令，后续任务复用）**

```bash
cd /home/ubuntu/workspace/github/llm/sglang
tar -cf - \
  python/sglang/multimodal_gen/envs.py \
  python/sglang/multimodal_gen/runtime/layers/fast_ulysses_backend.py \
  python/sglang/multimodal_gen/runtime/layers/usp.py \
  python/sglang/multimodal_gen/runtime/layers/attention/layer.py \
  python/sglang/multimodal_gen/test/unit/test_fast_ulysses_a2a.py \
| ssh ion-b200 "docker exec -i sglang-diffusion-triplemu tar -C /data/sglang -xf -"
```

- [ ] **Step 5: 远程跑对拍——先验证测试会失败（开关关）**

```bash
ssh ion-b200 "docker exec -e CUDA_VISIBLE_DEVICES=0,1,2,3 sglang-diffusion-triplemu bash -c \
  'cd /data/sglang && torchrun --nproc_per_node=4 python/sglang/multimodal_gen/test/unit/test_fast_ulysses_a2a.py'"
```

Expected: FAIL，AssertionError `fast a2a path not exercised`（证明测试确实检验 fast path，而非静默 fallback 假绿）。

- [ ] **Step 6: 远程跑对拍——开关开，应 PASS**

```bash
ssh ion-b200 "docker exec -e CUDA_VISIBLE_DEVICES=0,1,2,3 \
  -e SGLANG_DIFFUSION_ENABLE_FAST_ULYSSES=1 -e SGLANG_DIFFUSION_ENABLE_FAST_ULYSSES_QK_FUSION=1 \
  sglang-diffusion-triplemu bash -c \
  'cd /data/sglang && torchrun --nproc_per_node=4 python/sglang/multimodal_gen/test/unit/test_fast_ulysses_a2a.py'"
```

Expected: rank0 打印 `qk2 q/k: max abs diff ...` 与 `PASS`。若 mode0/mode1 逐位不等或 hang：按 systematic-debugging 排查（先缩小 shape 复现，检查 tag 冲突与调用序列一致性），不要放宽断言。

- [ ] **Step 7: Commit**

```bash
git add python/sglang/multimodal_gen/runtime/layers/usp.py \
        python/sglang/multimodal_gen/runtime/layers/attention/layer.py \
        python/sglang/multimodal_gen/test/unit/test_fast_ulysses_a2a.py
git commit -m "feat(diffusion): route uniform USP a2a through fast_ulysses behind env flag"
```

---

### Task 5: 阶段 1 e2e 验证 + 基准

**Files:** 无代码改动。产出：results 文档追加 "Stage 1 e2e" 小节。

**Interfaces:**
- Consumes: Task 2 的基准命令模板与 baseline JSON。

- [ ] **Step 1: e2e 逐位一致性（framemd5）**

用 Task 2 模板跑两次（都加 `--save-output`，eager）：一次无开关（label `md5_nccl`）、一次 `-e SGLANG_DIFFUSION_ENABLE_FAST_ULYSSES=1`（label `md5_fast`）。生成后：

```bash
ssh ion-b200 "docker exec sglang-diffusion-triplemu bash -c \
  'cd /data/sglang && ls -t outputs/ | head -4'"
# 对两个输出视频分别：
ssh ion-b200 "docker exec sglang-diffusion-triplemu bash -c \
  'ffmpeg -v error -i /data/sglang/outputs/<video1> -f framemd5 - > /tmp/a.md5; \
   ffmpeg -v error -i /data/sglang/outputs/<video2> -f framemd5 - > /tmp/b.md5; \
   diff /tmp/a.md5 /tmp/b.md5 && echo FRAMES_IDENTICAL'"
```

Expected: `FRAMES_IDENTICAL`（阶段 1 是逐位等价搬运）。若不一致：先跑两次纯 NCCL 验证运行本身是否确定；若 NCCL 自身确定而 fast 不同，是集成 bug，回 Task 4 排查。
同时确认 fast 那次的日志包含 `fast_ulysses UlyssesGroup initialized (world=4)`（证明未静默 fallback）。

- [ ] **Step 2: 阶段 1 基准（fast_eager ×3、fast_compile ×3）**

用 Task 2 模板，`<EXTRA_ENV>` 加 `-e SGLANG_DIFFUSION_ENABLE_FAST_ULYSSES=1`，label `fast_a2a_eager` / `fast_a2a_compile`。
若 compile 变体因 dynamo 处理 fast_ulysses custom op 报错：给 `maybe_all_to_all_4d` 加 `@torch._dynamo.disable` 装饰（行为等同现状 collective 的 graph break），重新同步后再跑；在 results 文档记录该情况。

- [ ] **Step 3: compare_perf 对比并记录**

```bash
ssh ion-b200 "docker exec sglang-diffusion-triplemu bash -c \
  'cd /data/sglang && python python/sglang/multimodal_gen/benchmarks/compare_perf.py \
   /data/bench/results/nccl_eager_run2.json /data/bench/results/fast_a2a_eager_run2.json'"
```

（run 编号取各自 denoise 中位的那次。）把 denoise/e2e/显存中位数与提升百分比追加到 results 文档。

- [ ] **Step 4: Commit 结果**

```bash
git add docs/superpowers/plans/2026-07-14-fast-ulysses-results.md
git commit -m "bench: stage-1 fast_ulysses a2a e2e numbers"
```

---

### Task 6: 阶段 2 —— QK norm+RoPE 融合接入

**Files:**
- Modify: `python/sglang/multimodal_gen/runtime/layers/attention/layer.py`（USPAttention.forward 签名 :611-622 与主路径 :955-972）
- Modify: `python/sglang/multimodal_gen/runtime/models/dits/wanvideo.py`（WanTransformerBlock：`__init__` 末尾、forward :523-579）

**Interfaces:**
- Consumes: `fast_ulysses_backend.QKFusedCtx / can_use_qk2 / qk2_input_a2a`（Task 3）；tag 约定 `usp_v`（Task 4）。
- Produces: `USPAttention.forward(..., qk_fused_ctx: QKFusedCtx | None = None)`；`WanTransformerBlock._maybe_fast_qk_fused_ctx(query, freqs_cis) -> QKFusedCtx | None`。

- [ ] **Step 1: USPAttention.forward 增加 qk_fused_ctx**

签名（layer.py:611-622）追加参数：

```python
        qk_fused_ctx: "fast_ulysses_backend.QKFusedCtx | None" = None,
```

文件头部 import `from sglang.multimodal_gen.runtime.layers import fast_ulysses_backend`。
在 `effective_skip_sp` 计算（:653-655）之后加防御（SPMD 下 rank 间一致，assert 不会引起 hang）：

```python
        if qk_fused_ctx is not None:
            assert (
                attn_mask is None
                and num_replicated_prefix == 0
                and num_replicated_suffix == 0
                and num_replicated_kv_prefix == 0
                and not effective_skip_sp
            ), "qk_fused_ctx is only supported on the uniform Ulysses path"
```

主路径（:955-972）改为：

```python
        # Ulysses-style All-to-All for sequence/head sharding
        if sp_size > 1:
            # -> [B, S, H_local, D]
            if qk_fused_ctx is not None:
                # Fused cross-head RMSNorm + RoPE + input a2a for q/k (the
                # caller skipped its own norm/rope); v has no norm/rope.
                q, k = fast_ulysses_backend.qk2_input_a2a(
                    q=q, k=k, ctx=qk_fused_ctx
                )
                v = _usp_input_all_to_all(v, head_dim=2, comm_tag="usp_v")
            elif self.enable_packed_qkv_input_a2a and q.device.type == "cuda":
```

（原 else 分支保持不变。）

- [ ] **Step 2: WanTransformerBlock 融合路径**

`__init__` 末尾（scale_shift_table 之后）加：

```python
        # fp32 copies of the QK norm weights for the fast_ulysses fused path,
        # built lazily on first use.
        self._fast_qk_norm_weights_fp32: tuple[torch.Tensor, torch.Tensor] | None = None
```

新增方法（放 forward 之前）：

```python
    def _maybe_fast_qk_fused_ctx(
        self,
        query: torch.Tensor,
        freqs_cis: tuple[torch.Tensor, torch.Tensor],
    ) -> "fast_ulysses_backend.QKFusedCtx | None":
        """Context for the fused cross-head QK RMSNorm + RoPE + input-a2a.

        None keeps the unfused norm/rope path. Every condition is identical
        across SP ranks, keeping the collective call pattern in lockstep.
        """
        if self.qk_norm != "rms_norm_across_heads" or self.tp_rmsnorm:
            return None
        if not isinstance(self.attn1, USPAttention) or self.attn1.skip_sequence_parallel:
            return None
        if not fast_ulysses_backend.can_use_qk2(
            num_heads=self.local_num_heads,
            head_dim=self.dim_head,
            dtype=query.dtype,
        ):
            return None
        if self._fast_qk_norm_weights_fp32 is None:
            self._fast_qk_norm_weights_fp32 = (
                self.norm_q.weight.detach().float().contiguous(),
                self.norm_k.weight.detach().float().contiguous(),
            )
        weight_q, weight_k = self._fast_qk_norm_weights_fp32
        cos, sin = freqs_cis
        return fast_ulysses_backend.QKFusedCtx(
            weight_q=weight_q,
            weight_k=weight_k,
            cos=cos.contiguous(),
            sin=sin.contiguous(),
            eps=self.norm_q.variance_epsilon,
        )
```

（先确认 `RMSNorm` 的 eps 属性名是 `variance_epsilon`——`runtime/layers/layernorm.py:76` 附近；若不同，以实际为准。）
forward 的 self-attention 段（:525 之后）改为：

```python
        query, _ = self.to_q(norm_hidden_states)
        key, _ = self.to_k(norm_hidden_states)
        value, _ = self.to_v(norm_hidden_states)

        fused_ctx = self._maybe_fast_qk_fused_ctx(query, freqs_cis)
        if fused_ctx is not None:
            query = query.squeeze(1).unflatten(
                2, (self.local_num_heads, self.dim_head)
            )
            key = key.squeeze(1).unflatten(2, (self.local_num_heads, self.dim_head))
            value = value.squeeze(1).unflatten(
                2, (self.local_num_heads, self.dim_head)
            )
            attn_output = self.attn1(query, key, value, qk_fused_ctx=fused_ctx)
        else:
            ...  # 原 529-579 行整段（norm_q/k、unflatten、RoPE、self.attn1(query, key, value)）原样缩进进此分支
```

文件头部 import `from sglang.multimodal_gen.runtime.layers import fast_ulysses_backend`（USPAttention 已有 import）。
`WanTransformerBlock_VSA` 不改（video_sparse_attn 场景，范围外）。

- [ ] **Step 3: 本地 import 冒烟**

```bash
python -c "import sglang.multimodal_gen.runtime.models.dits.wanvideo; print('ok')"
```

Expected: `ok`。

- [ ] **Step 4: 同步远程并复跑 Task 4 的对拍（含 qk2 case）**

用 Task 4 Step 4 的同步命令（文件列表加 `python/sglang/multimodal_gen/runtime/models/dits/wanvideo.py`），然后复跑 Task 4 Step 6 命令。
Expected: `PASS`，且 `qk2 q/k: max abs diff` ≲ 0.02（bf16 量级）。

- [ ] **Step 5: Commit**

```bash
git add python/sglang/multimodal_gen/runtime/layers/attention/layer.py \
        python/sglang/multimodal_gen/runtime/models/dits/wanvideo.py
git commit -m "feat(diffusion): fuse Wan QK norm+RoPE into fast_ulysses input a2a"
```

---

### Task 7: 阶段 2 e2e 验证 + 基准

**Files:** 无代码改动。产出：results 文档追加 "Stage 2 e2e" 小节。

- [ ] **Step 1: e2e 数值验收（PSNR）**

用 Task 2 模板 + `--save-output` 跑一次两开关全开（`-e SGLANG_DIFFUSION_ENABLE_FAST_ULYSSES=1 -e SGLANG_DIFFUSION_ENABLE_FAST_ULYSSES_QK_FUSION=1`，eager，label `psnr_fused`），与 Task 5 Step 1 的 `md5_nccl` 视频对比：

```bash
ssh ion-b200 "docker exec sglang-diffusion-triplemu bash -c \
  'ffmpeg -v info -i /data/sglang/outputs/<nccl_video> -i /data/sglang/outputs/<fused_video> \
   -lavfi psnr -f null - 2>&1 | tail -2'"
```

Expected: average PSNR ≥ 35 dB（融合路径全程 fp32、精度更高，差异应远小于此阈值）。低于 35 dB 视为数值问题，回 Task 6 排查（先核对 cos/sin 布局与 eps）。同时抽查视频内容正常（取回本地看首帧/末帧）。
确认日志出现 `fast_ulysses fused QK norm+RoPE+a2a path active.`。

- [ ] **Step 2: 阶段 2 基准（fused_eager ×3、fused_compile ×3）**

Task 2 模板，`<EXTRA_ENV>` 两个开关全开，label `fused_eager` / `fused_compile`。compile 若报 dynamo 错，与 Task 5 Step 2 同法处理（`torch._dynamo.disable` 到 `qk2_input_a2a`）。

- [ ] **Step 3: 记录并 commit**

compare_perf 对比 `nccl_eager` ↔ `fused_eager`、`fast_a2a_eager` ↔ `fused_eager`，中位数入 results 文档：

```bash
git add docs/superpowers/plans/2026-07-14-fast-ulysses-results.md
git commit -m "bench: stage-2 fused QK norm+RoPE+a2a e2e numbers"
```

---

### Task 8: nsys 归因 + 汇总报告

**Files:** 产出：results 文档完成态（6 配置矩阵 + op 级 + 归因结论）。

- [ ] **Step 1: nsys 采样 baseline 与最优配置各一份**

先从 perf dump 估算 denoise 开始时刻（模型加载+编码约耗时 X 秒），设 `--delay X --duration 60`：

```bash
ssh ion-b200 "docker exec -e HF_HOME=/data/hf-cache -e CUDA_VISIBLE_DEVICES=0,1,2,3 \
  -e FLASHINFER_DISABLE_VERSION_CHECK=1 [开关...] sglang-diffusion-triplemu bash -c \
  'mkdir -p /data/bench/nsys && cd /data/sglang && nsys profile --trace-fork-before-exec=true \
   --cuda-graph-trace=node --force-overwrite=true --delay <X> --duration 60 \
   -o /data/bench/nsys/<label> sglang generate <Task2 模板参数，不带 --warmup>'"
```

用 `nsys stats --report cuda_gpu_kern_sum` 对比两份 report 中 a2a 相关 kernel（NCCL 的 `ncclDevKernel*` vs fast_ulysses kernel）与 permute copy kernel 的总时长，写入归因小节。

- [ ] **Step 2: 汇总矩阵**

results 文档最终包含：

| 配置 | denoise 中位 (s) | vs nccl_eager | e2e (s) | 峰值显存 (GB) |
|---|---|---|---|---|
| nccl_eager / nccl_compile / fast_a2a_eager / fast_a2a_compile / fused_eager / fused_compile | … | … | … | … |

加上 op 级小节（Task 1）、正确性结论（Task 4/5/7）、nsys 归因、以及"后续若上游化需要做什么"（server flag、CI 对拍测试、多子组支持）三行备忘。

- [ ] **Step 3: 最终 commit 并汇报**

```bash
git add docs/superpowers/plans/2026-07-14-fast-ulysses-results.md
git commit -m "bench: final fast-ulysses integration report (4xB200 Wan2.1 I2V 720p)"
```

向用户汇报最终矩阵与结论（不合并、不推送——分支留在本地等用户处置）。

---

## 执行注意事项

1. **hang 处置**：任何 4 卡运行 hang 超过 5 分钟：`ssh ion-b200 "docker exec sglang-diffusion-triplemu bash -c 'pkill -9 -f torchrun; pkill -9 -f sglang'"`，然后按 debug-distributed-hang skill 的 per-rank 日志法排查；优先怀疑「rank 间 fast/fallback 判定分叉」与「tag 冲突」。
2. **GPU 占用**：跑任何 4 卡任务前 `nvidia-smi` 确认 GPU 0-3 空闲（这台机器 8 卡可能被他人使用）。
3. **不要改远程 /data/fast-ulysses**：它是用户的库，本项目只消费其已安装 wheel；若怀疑库 bug，记录复现命令报告用户。
4. **每个基准 run 之间**确认无残留进程与显存占用。
