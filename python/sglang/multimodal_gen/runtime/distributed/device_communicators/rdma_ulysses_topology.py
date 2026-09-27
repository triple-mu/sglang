# SPDX-License-Identifier: Apache-2.0
"""Which mlx5 port and RoCE GID each Ulysses rank drives, decided from sysfs.

Pure functions over a sysfs root so the plan can be unit-tested on a machine
without NICs. Every rank probes itself, the probes are all-gathered, and every
rank runs the same `plan_rdma_route` over the same list, so the group agrees
without a second collective.
"""

from __future__ import annotations

import ipaddress
import socket
from pathlib import Path

import msgspec

from sglang.multimodal_gen import envs

RDMA_PORT = 1


class RankProbe(msgspec.Struct, frozen=True):
    """One rank's view of its own GPU and NIC; `error` set means the rest is void."""

    rank: int
    hostname: str = ""
    gpu_uuid: str = ""
    pci_bus_id: str = ""
    nic: str = ""
    gid_index: int = -1
    error: str | None = None


class RoutePlan(msgspec.Struct, frozen=True):
    nics: tuple[str, ...]
    gid_indices: tuple[int, ...]


def _pci_device_path(pci_bus_id: str, sysfs_root: Path) -> Path:
    bus_id = pci_bus_id.lower()
    parts = bus_id.split(":")
    if len(parts) == 3 and len(parts[0]) > 4:
        bus_id = f"{parts[0][-4:]}:{parts[1]}:{parts[2]}"
    return (sysfs_root / "bus" / "pci" / "devices" / bus_id).resolve(strict=True)


def _path_distance(left: Path, right: Path) -> int:
    common = 0
    for a, b in zip(left.parts, right.parts, strict=False):
        if a != b:
            break
        common += 1
    return len(left.parts) + len(right.parts) - 2 * common


def _rank_override(raw: str | None, rank: int, name: str) -> str | None:
    if not raw:
        return None
    values = [value.strip() for value in raw.split(",")]
    if rank >= len(values) or not values[rank]:
        raise ValueError(f"{name} has no entry for rank {rank}: {raw!r}")
    return values[rank]


def select_nic(pci_bus_id: str, rank: int, *, sysfs_root: Path = Path("/sys")) -> str:
    """The mlx5 device closest to the GPU on the PCI tree.

    Bonded devices are skipped (they carry no RoCE port of their own). A tie
    between the two ports of one card is split by the GPU's PCI bus parity, so
    two GPUs under one switch take different ports whatever CUDA_VISIBLE_DEVICES
    says.
    """
    gpu_path = _pci_device_path(pci_bus_id, sysfs_root)
    configured = _rank_override(
        envs.SGLANG_DIFFUSION_RDMA_ULYSSES_NICS,
        rank,
        "SGLANG_DIFFUSION_RDMA_ULYSSES_NICS",
    )
    if configured is not None:
        if not (sysfs_root / "class" / "infiniband" / configured / "device").exists():
            raise RuntimeError(f"configured RDMA device {configured!r} does not exist")
        return configured
    candidates = []
    for entry in sorted((sysfs_root / "class" / "infiniband").glob("mlx5_*")):
        if entry.name.startswith("mlx5_bond"):
            continue
        try:
            nic_path = (entry / "device").resolve(strict=True)
        except OSError:
            continue
        candidates.append((_path_distance(gpu_path, nic_path), entry.name))
    if not candidates:
        raise RuntimeError("no mlx5 RDMA devices found")
    candidates.sort()
    closest = sorted(
        (name for distance, name in candidates if distance == candidates[0][0]),
        key=lambda name: int(name.rsplit("_", 1)[1]),
    )
    bus = int(gpu_path.name.split(":")[-2], 16)
    return closest[bus % len(closest)]


def select_gid(nic: str, rank: int, *, sysfs_root: Path = Path("/sys")) -> int:
    """Lowest GID index on port 1 that is RoCE v2 with a non-zero IPv4-mapped address."""
    port_root = sysfs_root / "class" / "infiniband" / nic / "ports" / str(RDMA_PORT)
    candidates = []
    for entry in sorted((port_root / "gids").iterdir(), key=lambda e: int(e.name)):
        try:
            address = ipaddress.IPv6Address(entry.read_text().strip())
            gid_type = (
                (port_root / "gid_attrs" / "types" / entry.name).read_text().strip()
            )
        except (OSError, ValueError):
            continue
        ipv4 = address.ipv4_mapped
        if gid_type == "RoCE v2" and ipv4 is not None and not ipv4.is_unspecified:
            candidates.append(int(entry.name))
    configured = _rank_override(
        envs.SGLANG_DIFFUSION_RDMA_ULYSSES_GID_INDICES,
        rank,
        "SGLANG_DIFFUSION_RDMA_ULYSSES_GID_INDICES",
    )
    if configured is not None:
        index = int(configured)
        if index not in candidates:
            raise RuntimeError(
                f"GID index {index} on {nic} is not a usable IPv4 RoCE v2 entry "
                f"(usable: {candidates or 'none'})"
            )
        return index
    if not candidates:
        raise RuntimeError(f"no IPv4 RoCE v2 GID on {nic} port {RDMA_PORT}")
    return candidates[0]


def _gpu_identity(device_index: int) -> tuple[str, str]:
    """(uuid, pci bus id) of one CUDA device, via torch and nvidia-smi."""
    import subprocess

    import torch

    uuid = f"GPU-{torch.cuda.get_device_properties(device_index).uuid}"
    listing = subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid,pci.bus_id", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    for line in listing.splitlines():
        listed_uuid, bus_id = (item.strip() for item in line.split(","))
        if listed_uuid == uuid:
            return uuid, bus_id
    raise RuntimeError(f"nvidia-smi does not list {uuid}")


def probe_rank(
    rank: int, device_index: int, *, sysfs_root: Path = Path("/sys")
) -> RankProbe:
    """Never raises: a failure is carried in `error` so the group can agree on it."""
    try:
        uuid, bus_id = _gpu_identity(device_index)
        nic = select_nic(bus_id, rank, sysfs_root=sysfs_root)
        return RankProbe(
            rank=rank,
            hostname=socket.gethostname(),
            gpu_uuid=uuid,
            pci_bus_id=bus_id,
            nic=nic,
            gid_index=select_gid(nic, rank, sysfs_root=sysfs_root),
        )
    except Exception as error:  # noqa: BLE001 -- every failure must become a vote
        return RankProbe(rank=rank, error=f"{type(error).__name__}: {error}")


def plan_rdma_route(probes: list[RankProbe]) -> RoutePlan:
    """Deterministic in the gathered probes; raises with one reason for the whole group."""
    by_rank = sorted(probes, key=lambda p: p.rank)
    if [p.rank for p in by_rank] != list(range(len(by_rank))):
        raise RuntimeError(f"malformed probe ranks {[p.rank for p in probes]}")
    for probe in by_rank:
        if probe.error is not None:
            raise RuntimeError(f"rank {probe.rank}: {probe.error}")
    hosts = {p.hostname for p in by_rank}
    if len(hosts) != 1:
        raise RuntimeError(f"ranks span several hosts: {sorted(hosts)}")
    uuids = [p.gpu_uuid for p in by_rank]
    if len(set(uuids)) != len(uuids):
        raise RuntimeError(f"two ranks share one GPU: {uuids}")
    nics = [p.nic for p in by_rank]
    if len(set(nics)) != len(nics):
        raise RuntimeError(
            f"one mlx5 device serves two ranks ({nics}); set "
            "SGLANG_DIFFUSION_RDMA_ULYSSES_NICS to a rank-ordered list"
        )
    return RoutePlan(nics=tuple(nics), gid_indices=tuple(p.gid_index for p in by_rank))


__all__ = [
    "RankProbe",
    "RoutePlan",
    "plan_rdma_route",
    "probe_rank",
    "select_gid",
    "select_nic",
]
