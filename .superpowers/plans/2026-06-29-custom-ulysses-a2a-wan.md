# 自定义 NVSHMEM Ulysses all-to-all 接入 Wan2.1/2.2 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 SGLang Diffusion 的 `USPAttention` Ulysses 路径里，用 env 开关把 `torch.distributed.all_to_all_single` 替换为自定义 NVSHMEM 算子（`custom_ulysses_op`），不影响默认行为，并为 H200×8 的 kernel 级性能对比做好准备。

**Architecture:** 方案 A——在 `_usp_input_all_to_all` / `_usp_output_all_to_all`（`head_dim==2`、uniform）内加一条 gated 分支，开关打开且算子可用时直接调 `all_to_all_single_4d(mode=0/1)` 原生吃 4D（省 permute/reshape host 开销），否则原样走 torch。`USPAttention` 主路径给 q/k/v/out 传 4 个不同 tag 避免对称堆 buffer 复用互相覆盖。算子 import 失败 / ws∉{2,4,8} / 非 head_dim==2 / 开关关 → 自动回落。

**Tech Stack:** Python, PyTorch distributed, `custom_ulysses_op`（`torch.ops.ulysses.all_to_all_single_4d`，NVSHMEM 对称堆 + NVLink P2P），SGLang Diffusion（multimodal_gen），pytest/unittest。

## Global Constraints

- 算子 4D 语义：mode0 `(b,s_local,n_global,d) → (b,s_global,n_local,d)`；mode1 为逆。fp16/bf16，`world_size ∈ {1,2,4,8}`，单节点 NVLink P2P。
- 集体硬约束：所有 rank 必须以相同 `(shape, mode, use_tma)` 序列调用；同 tag 复用同一对称堆 buffer（返回张量是该 buffer 的视图）。gate 条件只能依赖全 rank 一致的量（env、ulysses_ws、head_dim、tag），不得依赖 per-rank 数据。
- 算子来源仓库（H200 docker 内需可 `import custom_ulysses_op`）：`/home/ubuntu/workspace/github/self/custom_ulysses_op`。
- env 命名：diffusion 子系统用自己的 `python/sglang/multimodal_gen/envs.py`（不是 `srt/environ.py`），前缀 `SGLANG_DIFFUSION_*`，用 `_lazy_bool` / `_lazy_int` 注册，经 `envs.SGLANG_DIFFUSION_*` 访问。
- 默认 OFF：`SGLANG_DIFFUSION_USE_CUSTOM_ULYSSES_A2A` 默认 false，保证现有行为零变化。
- 不碰：`causal_wanvideo.py`（LocalAttention）、VSA 路径、varlen 路径、replicated-prefix 路径、head_dim==1 路径——全部保持 torch。
- 项目规则：`docs/` 下禁止改动（pre-commit 阻断）；新数据容器用 `msgspec.Struct` 不用 `@dataclass`（本计划无新容器）。

## 文件结构

| 文件 | 责任 |
|---|---|
| `python/sglang/multimodal_gen/envs.py`（改） | 注册 3 个 env：开关、TMA、pool 字节数 |
| `python/sglang/multimodal_gen/runtime/layers/custom_ulysses_a2a.py`（新建） | 单例 `UlyssesGroup` 懒构造 + gate（`is_custom_ulysses_a2a_enabled`）+ `custom_ulysses_a2a(x, mode, tag)` |
| `python/sglang/multimodal_gen/runtime/layers/usp.py`（改） | `_usp_input/output_all_to_all` 加 `a2a_tag` kwarg + gated 分支 |
| `python/sglang/multimodal_gen/runtime/layers/attention/layer.py`（改） | `USPAttention` 主路径传 `uin_q/uin_k/uin_v/uout` |
| `python/sglang/multimodal_gen/test/unit/test_custom_ulysses_a2a.py`（新建） | gate 逻辑、usp 分支选择、layer tag 唯一性（CPU-only，mock） |
| `.superpowers/bench/run_ulysses_a2a_matrix.sh`（新建） | H200 8-cell benchmark + profile 命令（turnkey 复现） |

---

### Task 1: 注册 3 个 env var

**Files:**
- Modify: `python/sglang/multimodal_gen/envs.py`（TYPE_CHECKING 块 + `environment_variables` dict）
- Test: `python/sglang/multimodal_gen/test/unit/test_custom_ulysses_a2a.py`

**Interfaces:**
- Produces: `envs.SGLANG_DIFFUSION_USE_CUSTOM_ULYSSES_A2A`(bool, 默认 False)、`envs.SGLANG_DIFFUSION_CUSTOM_ULYSSES_TMA`(bool, 默认 True)、`envs.SGLANG_DIFFUSION_CUSTOM_ULYSSES_POOL_BYTES`(int, 默认 2147483648 = 2<<30)。

- [ ] **Step 1: 写失败测试**

新建 `python/sglang/multimodal_gen/test/unit/test_custom_ulysses_a2a.py`：

```python
import os
import unittest
from unittest.mock import patch

from sglang.multimodal_gen import envs


class TestCustomUlyssesEnvVars(unittest.TestCase):
    def test_defaults(self):
        with patch.dict(os.environ, {}, clear=False):
            for k in (
                "SGLANG_DIFFUSION_USE_CUSTOM_ULYSSES_A2A",
                "SGLANG_DIFFUSION_CUSTOM_ULYSSES_TMA",
                "SGLANG_DIFFUSION_CUSTOM_ULYSSES_POOL_BYTES",
            ):
                os.environ.pop(k, None)
            self.assertFalse(envs.SGLANG_DIFFUSION_USE_CUSTOM_ULYSSES_A2A)
            self.assertTrue(envs.SGLANG_DIFFUSION_CUSTOM_ULYSSES_TMA)
            self.assertEqual(
                envs.SGLANG_DIFFUSION_CUSTOM_ULYSSES_POOL_BYTES, 2 << 30
            )

    def test_overrides(self):
        with patch.dict(
            os.environ,
            {
                "SGLANG_DIFFUSION_USE_CUSTOM_ULYSSES_A2A": "1",
                "SGLANG_DIFFUSION_CUSTOM_ULYSSES_TMA": "0",
                "SGLANG_DIFFUSION_CUSTOM_ULYSSES_POOL_BYTES": "1073741824",
            },
        ):
            self.assertTrue(envs.SGLANG_DIFFUSION_USE_CUSTOM_ULYSSES_A2A)
            self.assertFalse(envs.SGLANG_DIFFUSION_CUSTOM_ULYSSES_TMA)
            self.assertEqual(
                envs.SGLANG_DIFFUSION_CUSTOM_ULYSSES_POOL_BYTES, 1073741824
            )


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest python/sglang/multimodal_gen/test/unit/test_custom_ulysses_a2a.py::TestCustomUlyssesEnvVars -v`
Expected: FAIL（`AttributeError` / `KeyError`，env 尚未注册）

- [ ] **Step 3: 在 envs.py 注册**

在 `TYPE_CHECKING` 块（约 36 行 `SGLANG_DIFFUSION_CFG_GATE_STEP` 附近）加：

```python
    SGLANG_DIFFUSION_USE_CUSTOM_ULYSSES_A2A: bool = False
    SGLANG_DIFFUSION_CUSTOM_ULYSSES_TMA: bool = True
    SGLANG_DIFFUSION_CUSTOM_ULYSSES_POOL_BYTES: int = 2 << 30
```

在 `environment_variables` dict 内（`SGLANG_DIFFUSION_VAE_CHANNELS_LAST_3D` 条目之后、`# ================== cache-dit Env Vars ==================` 之前）加：

```python
    # Custom NVSHMEM Ulysses all-to-all op (custom_ulysses_op) for the
    # USPAttention sequence-parallel path. Opt-in A/B kill-switch; falls back
    # to torch all_to_all when off or when the op is unavailable.
    "SGLANG_DIFFUSION_USE_CUSTOM_ULYSSES_A2A": _lazy_bool(
        "SGLANG_DIFFUSION_USE_CUSTOM_ULYSSES_A2A"
    ),
    # Force the TMA kernel path of the custom Ulysses op (requires sm90+).
    "SGLANG_DIFFUSION_CUSTOM_ULYSSES_TMA": _lazy_bool(
        "SGLANG_DIFFUSION_CUSTOM_ULYSSES_TMA", "true"
    ),
    # NVSHMEM symmetric heap reservation (bytes) for the custom Ulysses op.
    "SGLANG_DIFFUSION_CUSTOM_ULYSSES_POOL_BYTES": _lazy_int(
        "SGLANG_DIFFUSION_CUSTOM_ULYSSES_POOL_BYTES", 2 << 30
    ),
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest python/sglang/multimodal_gen/test/unit/test_custom_ulysses_a2a.py::TestCustomUlyssesEnvVars -v`
Expected: PASS（2 passed）

- [ ] **Step 5: 提交**

```bash
git add python/sglang/multimodal_gen/envs.py python/sglang/multimodal_gen/test/unit/test_custom_ulysses_a2a.py
git commit -m "feat(diffusion): add custom Ulysses a2a env vars"
```

---

### Task 2: `custom_ulysses_a2a` 单例 wrapper + gate

**Files:**
- Create: `python/sglang/multimodal_gen/runtime/layers/custom_ulysses_a2a.py`
- Test: `python/sglang/multimodal_gen/test/unit/test_custom_ulysses_a2a.py`（追加）

**Interfaces:**
- Consumes: `envs.SGLANG_DIFFUSION_USE_CUSTOM_ULYSSES_A2A` / `_CUSTOM_ULYSSES_TMA` / `_POOL_BYTES`（Task 1）；`get_sp_group().ulysses_group`、`get_ulysses_parallel_world_size()`（`runtime/distributed/parallel_state.py`）。
- Produces:
  - `is_custom_ulysses_a2a_enabled() -> bool`
  - `custom_ulysses_a2a(x: torch.Tensor, *, mode: int, tag: str) -> torch.Tensor`
  - `_custom_op_available() -> bool`（模块内可 patch 的辅助）

- [ ] **Step 1: 写失败测试**（追加到 `test_custom_ulysses_a2a.py`）

```python
from sglang.multimodal_gen.runtime.layers import custom_ulysses_a2a as cua


class TestCustomUlyssesGate(unittest.TestCase):
    def test_disabled_when_env_off(self):
        with patch.dict(
            os.environ, {"SGLANG_DIFFUSION_USE_CUSTOM_ULYSSES_A2A": "0"}
        ):
            self.assertFalse(cua.is_custom_ulysses_a2a_enabled())

    def test_disabled_when_unsupported_world_size(self):
        with patch.dict(
            os.environ, {"SGLANG_DIFFUSION_USE_CUSTOM_ULYSSES_A2A": "1"}
        ), patch.object(
            cua, "get_ulysses_parallel_world_size", return_value=3
        ), patch.object(
            cua, "_custom_op_available", return_value=True
        ):
            self.assertFalse(cua.is_custom_ulysses_a2a_enabled())

    def test_disabled_when_op_missing(self):
        with patch.dict(
            os.environ, {"SGLANG_DIFFUSION_USE_CUSTOM_ULYSSES_A2A": "1"}
        ), patch.object(
            cua, "get_ulysses_parallel_world_size", return_value=8
        ), patch.object(
            cua, "_custom_op_available", return_value=False
        ):
            self.assertFalse(cua.is_custom_ulysses_a2a_enabled())

    def test_enabled_when_all_satisfied(self):
        with patch.dict(
            os.environ, {"SGLANG_DIFFUSION_USE_CUSTOM_ULYSSES_A2A": "1"}
        ), patch.object(
            cua, "get_ulysses_parallel_world_size", return_value=8
        ), patch.object(
            cua, "_custom_op_available", return_value=True
        ):
            self.assertTrue(cua.is_custom_ulysses_a2a_enabled())
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest python/sglang/multimodal_gen/test/unit/test_custom_ulysses_a2a.py::TestCustomUlyssesGate -v`
Expected: FAIL（`ModuleNotFoundError: custom_ulysses_a2a`）

- [ ] **Step 3: 写实现**

新建 `python/sglang/multimodal_gen/runtime/layers/custom_ulysses_a2a.py`：

```python
"""Custom NVSHMEM Ulysses all-to-all wrapper for SGLang Diffusion.

Optional drop-in for the torch all_to_all in USPAttention's Ulysses path.
Gated by SGLANG_DIFFUSION_USE_CUSTOM_ULYSSES_A2A; falls back to torch when the
op is unavailable or the parallel config is unsupported.
"""

from __future__ import annotations

import logging

import torch

from sglang.multimodal_gen import envs
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_sp_group,
    get_ulysses_parallel_world_size,
)

logger = logging.getLogger(__name__)

# custom_ulysses_op supports single-node NVLink P2P for these world sizes only.
_SUPPORTED_WORLD_SIZES = (2, 4, 8)

# id(ulysses_pg) -> custom_ulysses_op.UlyssesGroup (collective; built lazily on
# first use while all ranks run the same attention forward in lockstep).
_groups: dict[int, object] = {}

_import_checked = False
_import_ok = False


def _custom_op_available() -> bool:
    global _import_checked, _import_ok
    if not _import_checked:
        _import_checked = True
        try:
            import custom_ulysses_op  # noqa: F401  (triggers TORCH_LIBRARY register)

            _import_ok = True
        except Exception as e:  # ImportError or .so dlopen failure
            _import_ok = False
            logger.warning(
                "custom_ulysses_op unavailable, falling back to torch a2a: %s", e
            )
    return _import_ok


def is_custom_ulysses_a2a_enabled() -> bool:
    """All-rank-consistent gate: env on + supported ulysses ws + op importable."""
    if not envs.SGLANG_DIFFUSION_USE_CUSTOM_ULYSSES_A2A:
        return False
    if get_ulysses_parallel_world_size() not in _SUPPORTED_WORLD_SIZES:
        return False
    return _custom_op_available()


def _get_group():
    ulysses_pg = get_sp_group().ulysses_group
    assert ulysses_pg is not None, "Ulysses process group is not initialized."
    key = id(ulysses_pg)
    group = _groups.get(key)
    if group is None:
        from custom_ulysses_op import UlyssesGroup

        group = UlyssesGroup(
            process_group=ulysses_pg,
            initial_pool_bytes=int(envs.SGLANG_DIFFUSION_CUSTOM_ULYSSES_POOL_BYTES),
        )
        _groups[key] = group
    return group


def custom_ulysses_a2a(x: torch.Tensor, *, mode: int, tag: str) -> torch.Tensor:
    """4D Ulysses all-to-all via the custom NVSHMEM op.

    mode 0: (b, s_local, n_global, d) -> (b, s_global, n_local, d)
    mode 1: inverse.
    Every rank MUST call with the same (shape, mode, use_tma) sequence, and tag
    must be unique per concurrently-live tensor (same tag reuses one heap buffer).
    """
    group = _get_group()
    use_tma = bool(envs.SGLANG_DIFFUSION_CUSTOM_ULYSSES_TMA)
    return group.all_to_all_single_4d(x, mode=mode, tag=tag, use_tma=use_tma)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest python/sglang/multimodal_gen/test/unit/test_custom_ulysses_a2a.py::TestCustomUlyssesGate -v`
Expected: PASS（4 passed）

- [ ] **Step 5: 提交**

```bash
git add python/sglang/multimodal_gen/runtime/layers/custom_ulysses_a2a.py python/sglang/multimodal_gen/test/unit/test_custom_ulysses_a2a.py
git commit -m "feat(diffusion): custom Ulysses a2a singleton wrapper + gate"
```

---

### Task 3: `usp.py` gated 分支 + `a2a_tag`

**Files:**
- Modify: `python/sglang/multimodal_gen/runtime/layers/usp.py`（`_usp_input_all_to_all` 行 69-122、`_usp_output_all_to_all` 行 202-255，及顶部 import）
- Test: `python/sglang/multimodal_gen/test/unit/test_custom_ulysses_a2a.py`（追加）

**Interfaces:**
- Consumes: `custom_ulysses_a2a`、`is_custom_ulysses_a2a_enabled`（Task 2）。
- Produces: `_usp_input_all_to_all(x, head_dim=1, *, a2a_tag=None)` 与 `_usp_output_all_to_all(x, head_dim=1, *, a2a_tag=None)`（签名向后兼容，默认 `a2a_tag=None` → 现有行为）。

- [ ] **Step 1: 写失败测试**（追加）

```python
import torch

from sglang.multimodal_gen.runtime.layers import usp


class TestUspCustomBranch(unittest.TestCase):
    def _x_in(self):
        # head_dim=2 input layout: [b, s_local, h_global, d]
        return torch.zeros(1, 4, 8, 16)

    def test_input_uses_custom_op_mode0(self):
        sentinel = torch.zeros(1, 8, 4, 16)
        with patch.object(
            usp, "get_ulysses_parallel_world_size", return_value=2
        ), patch.object(
            usp, "is_custom_ulysses_a2a_enabled", return_value=True
        ), patch.object(
            usp, "custom_ulysses_a2a", return_value=sentinel
        ) as m:
            out = usp._usp_input_all_to_all(self._x_in(), head_dim=2, a2a_tag="uin_q")
        self.assertEqual(m.call_args.kwargs["mode"], 0)
        self.assertEqual(m.call_args.kwargs["tag"], "uin_q")
        self.assertIs(out, sentinel)

    def test_output_uses_custom_op_mode1(self):
        x = torch.zeros(1, 8, 4, 16)  # [b, s_global, h_local, d]
        sentinel = torch.zeros(1, 4, 8, 16)
        with patch.object(
            usp, "get_ulysses_parallel_world_size", return_value=2
        ), patch.object(
            usp, "is_custom_ulysses_a2a_enabled", return_value=True
        ), patch.object(
            usp, "custom_ulysses_a2a", return_value=sentinel
        ) as m:
            out = usp._usp_output_all_to_all(x, head_dim=2, a2a_tag="uout")
        self.assertEqual(m.call_args.kwargs["mode"], 1)
        self.assertEqual(m.call_args.kwargs["tag"], "uout")
        self.assertIs(out, sentinel)

    def test_fallback_when_disabled(self):
        with patch.object(
            usp, "get_ulysses_parallel_world_size", return_value=2
        ), patch.object(
            usp, "is_custom_ulysses_a2a_enabled", return_value=False
        ), patch.object(
            usp, "custom_ulysses_a2a"
        ) as m_custom, patch.object(
            usp, "_usp_all_to_all_single", side_effect=lambda t: t
        ) as m_torch:
            usp._usp_input_all_to_all(self._x_in(), head_dim=2, a2a_tag="uin_q")
        m_custom.assert_not_called()
        m_torch.assert_called_once()

    def test_fallback_when_no_tag(self):
        with patch.object(
            usp, "get_ulysses_parallel_world_size", return_value=2
        ), patch.object(
            usp, "is_custom_ulysses_a2a_enabled", return_value=True
        ), patch.object(
            usp, "custom_ulysses_a2a"
        ) as m_custom, patch.object(
            usp, "_usp_all_to_all_single", side_effect=lambda t: t
        ) as m_torch:
            usp._usp_input_all_to_all(self._x_in(), head_dim=2, a2a_tag=None)
        m_custom.assert_not_called()
        m_torch.assert_called_once()

    def test_fallback_when_head_dim_1(self):
        x = torch.zeros(1, 8, 4, 16)  # head_dim=1: [b, h_global, s_local, d]
        with patch.object(
            usp, "get_ulysses_parallel_world_size", return_value=2
        ), patch.object(
            usp, "is_custom_ulysses_a2a_enabled", return_value=True
        ), patch.object(
            usp, "custom_ulysses_a2a"
        ) as m_custom, patch.object(
            usp, "_usp_all_to_all_single", side_effect=lambda t: t
        ) as m_torch:
            usp._usp_input_all_to_all(x, head_dim=1, a2a_tag="uin_q")
        m_custom.assert_not_called()
        m_torch.assert_called_once()
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest python/sglang/multimodal_gen/test/unit/test_custom_ulysses_a2a.py::TestUspCustomBranch -v`
Expected: FAIL（`_usp_input_all_to_all() got an unexpected keyword argument 'a2a_tag'`）

- [ ] **Step 3: 改 usp.py**

顶部 import 区（约 11-16 行的 parallel_state import 之后）加：

```python
from sglang.multimodal_gen.runtime.layers.custom_ulysses_a2a import (
    custom_ulysses_a2a,
    is_custom_ulysses_a2a_enabled,
)
```

把 `_usp_input_all_to_all` 签名改为：

```python
def _usp_input_all_to_all(
    x: torch.Tensor, head_dim: int = 1, *, a2a_tag: str | None = None
) -> torch.Tensor:
```

在该函数体里、`world_size <= 1` 早退与两个 `assert`（`x.ndim == 4` / `head_dim in (1, 2)`）之后、`if head_dim == 1:` 之前，插入 gated 分支：

```python
    # Custom NVSHMEM Ulysses all-to-all fast path (mode 0). Native 4D, skips the
    # permute/contiguous/reshape host overhead of the torch path. Only head_dim==2
    # (b, s_local, h_global, d) matches the op's (b, s, n, d) layout.
    if a2a_tag is not None and head_dim == 2 and is_custom_ulysses_a2a_enabled():
        b, s_local, h_global, d = x.shape
        assert (
            h_global % world_size == 0
        ), f"h_global ({h_global}) must be divisible by world_size ({world_size})"
        return custom_ulysses_a2a(x, mode=0, tag=a2a_tag)
```

把 `_usp_output_all_to_all` 签名改为：

```python
def _usp_output_all_to_all(
    x: torch.Tensor, head_dim: int = 1, *, a2a_tag: str | None = None
) -> torch.Tensor:
```

在该函数体里、`world_size <= 1` 早退与两个 `assert` 之后、`if head_dim == 1:` 之前，插入：

```python
    # Custom NVSHMEM Ulysses all-to-all fast path (mode 1, inverse of input).
    if a2a_tag is not None and head_dim == 2 and is_custom_ulysses_a2a_enabled():
        b, s_global, h_local, d = x.shape
        assert (
            s_global % world_size == 0
        ), f"s_global ({s_global}) must be divisible by world_size ({world_size})"
        return custom_ulysses_a2a(x, mode=1, tag=a2a_tag)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest python/sglang/multimodal_gen/test/unit/test_custom_ulysses_a2a.py::TestUspCustomBranch -v`
Expected: PASS（5 passed）

- [ ] **Step 5: 提交**

```bash
git add python/sglang/multimodal_gen/runtime/layers/usp.py python/sglang/multimodal_gen/test/unit/test_custom_ulysses_a2a.py
git commit -m "feat(diffusion): gated custom Ulysses a2a branch in usp.py"
```

---

### Task 4: `layer.py` USPAttention 主路径传唯一 tag

**Files:**
- Modify: `python/sglang/multimodal_gen/runtime/layers/attention/layer.py`（`USPAttention.forward` 主路径，约 798-819 行）
- Test: `python/sglang/multimodal_gen/test/unit/test_custom_ulysses_a2a.py`（追加）

**Interfaces:**
- Consumes: `_usp_input_all_to_all(..., a2a_tag=...)`、`_usp_output_all_to_all(..., a2a_tag=...)`（Task 3）。
- Produces: 主 self-attn 路径 q/k/v/out 分别用 tag `"uin_q"` / `"uin_k"` / `"uin_v"` / `"uout"`（互不相同）。

- [ ] **Step 1: 写失败测试**（追加）

```python
from types import SimpleNamespace

from sglang.multimodal_gen.runtime.layers.attention.layer import USPAttention


class TestUSPAttentionTags(unittest.TestCase):
    def test_main_path_passes_distinct_tags(self):
        captured = []

        def rec_in(x, head_dim=1, *, a2a_tag=None):
            captured.append(("in", head_dim, a2a_tag))
            return x

        def rec_out(x, head_dim=1, *, a2a_tag=None):
            captured.append(("out", head_dim, a2a_tag))
            return x

        attn = object.__new__(USPAttention)
        attn.skip_sequence_parallel = False
        attn.enable_packed_qkv_input_a2a = False
        attn.causal = False
        attn.dropout_p = 0.0
        attn.attn_impl = SimpleNamespace(forward=lambda q, k, v, md: q)

        q = torch.zeros(1, 4, 8, 16)  # [b, s_local, h, d]
        with patch(
            "sglang.multimodal_gen.runtime.layers.attention.layer.get_forward_context",
            return_value=SimpleNamespace(attn_metadata=None),
        ), patch(
            "sglang.multimodal_gen.runtime.layers.attention.layer.get_sequence_parallel_world_size",
            return_value=2,
        ), patch(
            "sglang.multimodal_gen.runtime.layers.attention.layer.get_ulysses_parallel_world_size",
            return_value=2,
        ), patch(
            "sglang.multimodal_gen.runtime.layers.attention.layer.get_ring_parallel_world_size",
            return_value=1,
        ), patch(
            "sglang.multimodal_gen.runtime.layers.attention.layer._usp_input_all_to_all",
            side_effect=rec_in,
        ), patch(
            "sglang.multimodal_gen.runtime.layers.attention.layer._usp_output_all_to_all",
            side_effect=rec_out,
        ):
            attn.forward(q, q.clone(), q.clone())

        tags = [c[2] for c in captured]
        self.assertEqual(tags, ["uin_q", "uin_k", "uin_v", "uout"])
        self.assertEqual(len(set(tags)), 4)
        # all head_dim==2
        self.assertTrue(all(c[1] == 2 for c in captured))
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest python/sglang/multimodal_gen/test/unit/test_custom_ulysses_a2a.py::TestUSPAttentionTags -v`
Expected: FAIL（`tags == [None, None, None, None]`，断言不等）

- [ ] **Step 3: 改 layer.py**

`USPAttention.forward` 的 `if self.enable_packed_qkv_input_a2a ...` 的 `else` 分支（约 797-800 行）：

```python
            else:
                q = _usp_input_all_to_all(q, head_dim=2, a2a_tag="uin_q")
                k = _usp_input_all_to_all(k, head_dim=2, a2a_tag="uin_k")
                v = _usp_input_all_to_all(v, head_dim=2, a2a_tag="uin_v")
```

主路径恢复 sharding（约 819 行）：

```python
        # Ulysses-style All-to-All to restore original sharding
        if sp_size > 1:
            # -> [B, S_local, H, D]
            out = _usp_output_all_to_all(out, head_dim=2, a2a_tag="uout")
```

> 不改 mask 分支（625-754）、replicated-prefix/suffix/kv_prefix 路径（823-1004）——这些 Wan self-attn 主路径不走；它们继续用默认 `a2a_tag=None`（torch），保持原行为。

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest python/sglang/multimodal_gen/test/unit/test_custom_ulysses_a2a.py::TestUSPAttentionTags -v`
Expected: PASS（1 passed）

- [ ] **Step 5: 全单测文件 + 关联回归**

Run:
```bash
python -m pytest python/sglang/multimodal_gen/test/unit/test_custom_ulysses_a2a.py -v
python -m pytest python/sglang/multimodal_gen/test/unit/test_wan_ti2v_helpers.py -v
```
Expected: 全 PASS（新文件全过；Wan 相关单测不回归）

- [ ] **Step 6: 提交**

```bash
git add python/sglang/multimodal_gen/runtime/layers/attention/layer.py python/sglang/multimodal_gen/test/unit/test_custom_ulysses_a2a.py
git commit -m "feat(diffusion): wire distinct a2a tags in USPAttention main path"
```

---

### Task 5: H200 8-cell benchmark + profile 脚本（turnkey）

**Files:**
- Create: `.superpowers/bench/run_ulysses_a2a_matrix.sh`

**Interfaces:**
- Consumes: Task 1 的 env 开关；`sglang generate` CLI（`--num-gpus/--ulysses-degree/--ring-degree/--profile/--num-profiled-timesteps/--warmup/--perf-dump-path`）。
- Produces: 2 模型 × 2 SP 配置 × 2 算子 的 perf-dump + torch trace，落 `$OUT_DIR`。

- [ ] **Step 1: 写脚本**

新建 `.superpowers/bench/run_ulysses_a2a_matrix.sh`：

```bash
#!/usr/bin/env bash
# H200x8 benchmark matrix for the custom Ulysses a2a op.
# 2 models x {U8R1, U4R2} x {baseline, custom op TMA+tune}.
# Run inside docker `sglang-diffusion-ulysess` after `source /data/.torch/bin/activate`.
set -euo pipefail

OUT_DIR="${OUT_DIR:-/data/ulysses_a2a_bench}"
PROMPT="A cat and a dog baking a cake together in a kitchen."
mkdir -p "$OUT_DIR/torch"

run_cell() {
  local model="$1" tag="$2" uly="$3" ring="$4" use_custom="$5"
  local label="${tag}_U${uly}R${ring}_$([ "$use_custom" = 1 ] && echo custom || echo baseline)"
  echo "=== $label ==="
  export SGLANG_DIFFUSION_USE_CUSTOM_ULYSSES_A2A="$use_custom"
  export SGLANG_DIFFUSION_CUSTOM_ULYSSES_TMA=1

  # 1) clean latency (no --profile); read the "(with warmup excluded)" line
  sglang generate --backend=sglang --model-path="$model" --prompt="$PROMPT" \
    --width=1280 --height=720 --num-frames=81 --seed=42 \
    --num-gpus=8 --ulysses-degree="$uly" --ring-degree="$ring" \
    --warmup --perf-dump-path="$OUT_DIR/${label}.perf.json" \
    2>&1 | tee "$OUT_DIR/${label}.latency.log"

  # 2) kernel trace
  SGLANG_DIFFUSION_TORCH_PROFILER_DIR="$OUT_DIR/torch/${label}" \
  sglang generate --backend=sglang --model-path="$model" --prompt="$PROMPT" \
    --width=1280 --height=720 --num-frames=81 --seed=42 \
    --num-gpus=8 --ulysses-degree="$uly" --ring-degree="$ring" \
    --warmup --profile --num-profiled-timesteps=5 \
    2>&1 | tee "$OUT_DIR/${label}.profile.log"
}

for MODEL in "Wan-AI/Wan2.1-T2V-14B-Diffusers" "Wan-AI/Wan2.2-T2V-A14B-Diffusers"; do
  TAG="$(echo "$MODEL" | sed 's#.*/##; s/-Diffusers//')"
  for USE_CUSTOM in 0 1; do
    run_cell "$MODEL" "$TAG" 8 1 "$USE_CUSTOM"   # 单独 usp
    run_cell "$MODEL" "$TAG" 4 2 "$USE_CUSTOM"   # cp+usp
  done
done
echo "Done. Artifacts in $OUT_DIR"
```

- [ ] **Step 2: 语法校验**

Run: `bash -n .superpowers/bench/run_ulysses_a2a_matrix.sh`
Expected: 无输出（语法 OK）

- [ ] **Step 3: 提交**

```bash
git add .superpowers/bench/run_ulysses_a2a_matrix.sh
git commit -m "chore(bench): H200x8 ulysses a2a benchmark matrix script"
```

---

## Phase 2（硬件执行，代码审过后单独进行）

不在本计划的 TDD 任务内，作为 runbook：

1. **推分支 / 拉到 docker**：`ssh hyper00` → `docker exec -it sglang-diffusion-ulysess bash` → `source /data/.torch/bin/activate`，把 `feat/custom-ulysses-a2a-wan` 拉到 docker 内 sglang 路径。
2. **构建/验证算子**：`python -c "import custom_ulysses_op"`；不在位则 `NVSHMEM_HOME=<...> CUSTOM_ULYSSES_CUDA_ARCH=90 pip install -e . --no-build-isolation`（NVSHMEM 3.7.0）。
3. **正确性 gate**（性能前必过）：同 seed，`SGLANG_DIFFUSION_USE_CUSTOM_ULYSSES_A2A=0` vs `=1` 各生成一次（U8R1 一组、U4R2 一组，两模型），比 latent/输出帧或用 `tools/compare_diffusion_trajectory_similarity.py`。不一致就停、debug。
4. **跑矩阵**：`bash .superpowers/bench/run_ulysses_a2a_matrix.sh`。
5. **分析**：用 `llm-torch-profiler-analysis` skill 解析 trace，对比 all-to-all 段（baseline NCCL `all_to_all`/`ncclDevKernel` vs 自定义 nvshmem kernel）kernel 时间 + denoise 端到端延迟，产出每 cell 的对比表（a2a kernel us / denoise latency / 加速比），覆盖「单独 usp」与「cp+usp」。

---

## Self-Review

- **Spec coverage**：spec §3 方案A → Task 3；§4 组件1(wrapper)→Task 2、组件2(usp)→Task 3、组件3(layer tags)→Task 4、组件4(env)→Task 1、组件5(tune/TMA：lazy + use_tma=True)→Task 2 的 `custom_ulysses_a2a`/gate；§5 正确性→Phase 2 step3；§6 benchmark→Task 5 + Phase 2；§7 构建→Phase 2 step1-2；§8 风险(回落/集体一致/tag 唯一/整除)→Task 2 gate + Task 3 assert + Task 4 tag + 测试；§9 验收→Task 4 step5 回归 + Phase 2。无遗漏。
- **Placeholder scan**：每个 code step 均为完整可运行代码；命令带预期输出；无 TBD/TODO。Phase 2 的 `<...>`（NVSHMEM_HOME）是运行时路径，非代码占位。
- **Type consistency**：`is_custom_ulysses_a2a_enabled() -> bool`、`custom_ulysses_a2a(x, *, mode, tag)`、`_custom_op_available()` 在 Task 2 定义，Task 3 一致引用；`_usp_input/output_all_to_all(..., *, a2a_tag=None)` 在 Task 3 定义，Task 4 一致传 `a2a_tag="uin_q/uin_k/uin_v/uout"`；env 名三处一致。
