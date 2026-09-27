# SPDX-License-Identifier: Apache-2.0
"""JIT loader for the RDMA Ulysses transport (`csrc/distributed/rdma_ulysses.cuh`).

A stateful communicator rather than a kernel, so it registers no KernelSpec:
Python drives `init / connect / register_slot / connect_slot / exchange /
teardown_safe / disconnect / dispose` directly on the loaded module.
"""

from __future__ import annotations

import ctypes.util
from typing import TYPE_CHECKING

from sglang.kernels.jit.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module

_EXPORTS = (
    "init",
    "connect",
    "register_slot",
    "connect_slot",
    "exchange",
    "teardown_safe",
    "disconnect",
    "dispose",
    "publish_abort_for_test",
)


def missing_rdma_libraries() -> list[str]:
    """rdma-core libraries this machine lacks; empty when the module can link."""
    return [
        f"lib{name}"
        for name in ("ibverbs", "mlx5")
        if ctypes.util.find_library(name) is None
    ]


@cache_once
def load_rdma_ulysses() -> Module:
    missing = missing_rdma_libraries()
    if missing:
        raise RuntimeError(
            f"the RDMA Ulysses transport needs rdma-core; missing {', '.join(missing)}"
        )
    return load_jit(
        "rdma_ulysses",
        cuda_files=["distributed/rdma_ulysses.cuh"],
        cuda_wrappers=[(name, f"rdma_ulysses::{name}") for name in _EXPORTS],
        extra_ldflags=["-lcuda", "-libverbs", "-lmlx5"],
    )


__all__ = ["load_rdma_ulysses", "missing_rdma_libraries"]
