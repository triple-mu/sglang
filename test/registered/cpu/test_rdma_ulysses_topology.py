# SPDX-License-Identifier: Apache-2.0
"""Device-free contracts of the RDMA Ulysses transport: NIC and GID choice, route plan, slot views."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from sglang.multimodal_gen.runtime.distributed.device_communicators import (
    rdma_ulysses_a2a as a2a,
)
from sglang.multimodal_gen.runtime.distributed.device_communicators.rdma_ulysses_topology import (
    RankProbe,
    plan_rdma_route,
    select_gid,
    select_nic,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=6, suite="stage-a-test-cpu-intel")

# The pro-5k box: four PCIe switches, each with two GPUs and one dual-port NIC.
_GPUS = {
    "0000:06:00.0": "pci0000:00/0000:00:01.1/0000:01:00.0/0000:02:01.0/0000:04:00.0/0000:05:00.0/0000:06:00.0",
    "0000:09:00.0": "pci0000:00/0000:00:01.1/0000:01:00.0/0000:02:03.0/0000:07:00.0/0000:08:00.0/0000:09:00.0",
    "0000:76:00.0": "pci0000:70/0000:70:01.1/0000:71:00.0/0000:72:01.0/0000:74:00.0/0000:75:00.0/0000:76:00.0",
    "0000:79:00.0": "pci0000:70/0000:70:01.1/0000:71:00.0/0000:72:03.0/0000:77:00.0/0000:78:00.0/0000:79:00.0",
}
_NICS = {
    "mlx5_0": "pci0000:70/0000:70:01.1/0000:71:00.0/0000:72:00.0/0000:73:00.0",
    "mlx5_1": "pci0000:70/0000:70:01.1/0000:71:00.0/0000:72:00.0/0000:73:00.1",
    "mlx5_2": "pci0000:00/0000:00:01.1/0000:01:00.0/0000:02:00.0/0000:03:00.0",
    "mlx5_3": "pci0000:00/0000:00:01.1/0000:01:00.0/0000:02:00.0/0000:03:00.1",
    "mlx5_bond_0": "pci0000:c0/0000:c0:01.1/0000:c1:00.0",
}
_GIDS = [
    ("fe80:0000:0000:0000:5ad5:33ff:fe08:7ec0", "IB/RoCE v1"),
    ("fe80:0000:0000:0000:5ad5:33ff:fe08:7ec0", "RoCE v2"),
    ("0000:0000:0000:0000:0000:ffff:ac10:800d", "IB/RoCE v1"),
    ("0000:0000:0000:0000:0000:ffff:ac10:800d", "RoCE v2"),
    ("0000:0000:0000:0000:0000:0000:0000:0000", "RoCE v2"),
]


def _fake_sysfs(root: Path) -> None:
    for bus, rel in _GPUS.items():
        device = root / "devices" / rel
        device.mkdir(parents=True)
        (root / "bus" / "pci" / "devices").mkdir(parents=True, exist_ok=True)
        (root / "bus" / "pci" / "devices" / bus).symlink_to(device)
    for nic, rel in _NICS.items():
        device = root / "devices" / rel
        device.mkdir(parents=True, exist_ok=True)
        nic_root = root / "class" / "infiniband" / nic
        nic_root.mkdir(parents=True)
        (nic_root / "device").symlink_to(device)
        gids = nic_root / "ports" / "1" / "gids"
        types = nic_root / "ports" / "1" / "gid_attrs" / "types"
        gids.mkdir(parents=True)
        types.mkdir(parents=True)
        for index, (gid, gid_type) in enumerate(_GIDS):
            (gids / str(index)).write_text(gid + "\n")
            (types / str(index)).write_text(gid_type + "\n")


class TestTopology(CustomTestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        _fake_sysfs(self.root)

    def tearDown(self):
        self._tmp.cleanup()

    def test_each_gpu_takes_the_nearest_port_and_neighbours_split_the_card(self):
        expected = {
            "0000:06:00.0": "mlx5_2",
            "0000:09:00.0": "mlx5_3",
            "0000:76:00.0": "mlx5_0",
            "0000:79:00.0": "mlx5_1",
        }
        for bus, nic in expected.items():
            self.assertEqual(select_nic(bus, 0, sysfs_root=self.root), nic)
        # CUDA ordinals do not matter; the bond device is never chosen.
        self.assertEqual(
            select_nic("00000000:06:00.0", 5, sysfs_root=self.root), "mlx5_2"
        )

    def test_nic_override_is_rank_ordered_and_must_exist(self):
        with patch.object(
            a2a.envs, "SGLANG_DIFFUSION_RDMA_ULYSSES_NICS", "mlx5_1,mlx5_0"
        ):
            self.assertEqual(
                select_nic("0000:06:00.0", 1, sysfs_root=self.root), "mlx5_0"
            )
            with self.assertRaisesRegex(ValueError, "no entry for rank 2"):
                select_nic("0000:06:00.0", 2, sysfs_root=self.root)
        with patch.object(a2a.envs, "SGLANG_DIFFUSION_RDMA_ULYSSES_NICS", "mlx5_9"):
            with self.assertRaisesRegex(RuntimeError, "does not exist"):
                select_nic("0000:06:00.0", 0, sysfs_root=self.root)

    def test_gid_is_the_lowest_ipv4_mapped_roce_v2_entry(self):
        self.assertEqual(select_gid("mlx5_2", 0, sysfs_root=self.root), 3)
        with patch.object(a2a.envs, "SGLANG_DIFFUSION_RDMA_ULYSSES_GID_INDICES", "3,3"):
            self.assertEqual(select_gid("mlx5_2", 1, sysfs_root=self.root), 3)
        with patch.object(a2a.envs, "SGLANG_DIFFUSION_RDMA_ULYSSES_GID_INDICES", "1"):
            with self.assertRaisesRegex(RuntimeError, "not a usable IPv4 RoCE v2"):
                select_gid("mlx5_2", 0, sysfs_root=self.root)

    def test_plan_needs_distinct_nics_one_host_and_clean_probes(self):
        good = [
            RankProbe(rank=0, hostname="h", gpu_uuid="a", nic="mlx5_2", gid_index=3),
            RankProbe(rank=1, hostname="h", gpu_uuid="b", nic="mlx5_3", gid_index=3),
        ]
        plan = plan_rdma_route(list(reversed(good)))
        self.assertEqual(plan.nics, ("mlx5_2", "mlx5_3"))
        self.assertEqual(plan.gid_indices, (3, 3))
        with self.assertRaisesRegex(RuntimeError, "serves two ranks"):
            plan_rdma_route(
                [
                    good[0],
                    RankProbe(
                        rank=1, hostname="h", gpu_uuid="b", nic="mlx5_2", gid_index=3
                    ),
                ]
            )
        with self.assertRaisesRegex(RuntimeError, "several hosts"):
            plan_rdma_route(
                [
                    good[0],
                    RankProbe(
                        rank=1, hostname="k", gpu_uuid="b", nic="mlx5_3", gid_index=3
                    ),
                ]
            )
        with self.assertRaisesRegex(RuntimeError, "rank 1: RuntimeError: boom"):
            plan_rdma_route([good[0], RankProbe(rank=1, error="RuntimeError: boom")])
        with self.assertRaisesRegex(RuntimeError, "share one GPU"):
            plan_rdma_route(
                [
                    good[0],
                    RankProbe(
                        rank=1, hostname="h", gpu_uuid="a", nic="mlx5_3", gid_index=3
                    ),
                ]
            )


class _FakeModule:
    """Stands in for the JIT module: slots are CPU byte buffers, exchange records its calls."""

    def __init__(self):
        self.slots = []
        self.exchanges = []

    def register_slot(self, handle, mode, capacity):
        self.slots.append((mode, capacity))
        return (
            len(self.slots) - 1,
            torch.zeros(capacity, dtype=torch.uint8),
            torch.zeros(capacity, dtype=torch.uint8),
            [0],
        )

    def connect_slot(self, handle, index, flat):
        pass

    def exchange(self, handle, index, inp, out, own_in_place):
        self.exchanges.append(
            (index, inp.data_ptr(), tuple(inp.shape), tuple(out.shape), own_in_place)
        )


class TestCommunicatorViews(CustomTestCase):
    def _transport(self):
        module = _FakeModule()
        transport = a2a.RdmaUlyssesA2A(
            module,
            1,
            group=None,
            world_size=2,
            rank=0,
            device=torch.device("cpu"),
            plan=a2a.RoutePlan(nics=("a", "b"), gid_indices=(3, 3)),
        )
        with patch.object(
            a2a, "_gather_object", lambda group, payload, device: [payload, payload]
        ):
            transport._register("chunks", a2a.MODE_CHUNKS, 4096)
            transport._register("gather", a2a.MODE_GATHER, 8192)
            transport._register("stats", a2a.MODE_CHUNKS, 1024)
        return transport, module

    def test_stats_all_gather_replicates_the_record_into_every_chunk(self):
        transport, module = self._transport()
        stats = torch.arange(2 * 4 * 8, dtype=torch.float32).view(2, 1, 4, 8)
        out = transport.exchange_stats(stats)
        self.assertEqual(tuple(out.shape), (2, 2, 1, 4, 8))
        index, _, landing = transport._slots["stats"]
        self.assertEqual(module.exchanges[-1][0], index)
        sent = landing[: 2 * stats.numel() * 4].view(2, -1).view(torch.float32)
        self.assertTrue(torch.equal(sent[0], stats.flatten()))
        self.assertTrue(torch.equal(sent[1], stats.flatten()))

    def test_capacities_cover_the_768p_geometry(self):
        self.assertEqual(a2a.chunks_capacity_bytes(37888, 56, 8) % 128, 0)
        self.assertGreaterEqual(a2a.chunks_capacity_bytes(37888, 56, 8), 8 * 12736640)
        self.assertEqual(
            a2a.gather_capacity_bytes(37888, 56, 128, 2), 37888 * 56 * 128 * 2
        )

    def test_chunks_pack_into_the_landing_and_come_back_as_a_prefix_view(self):
        transport, module = self._transport()
        send = transport.packed_input_buffer((2, 1000))
        self.assertEqual(tuple(send.shape), (2, 1000))
        out = transport.exchange_chunks(send)
        self.assertEqual(tuple(out.shape), (2, 1000))
        self.assertEqual(module.exchanges[0][0], 0)
        self.assertEqual(module.exchanges[0][1], send.data_ptr())
        self.assertEqual(transport.exchanges, 1)
        with self.assertRaisesRegex(
            a2a.RdmaUlyssesError, "exceeds the registered capacity"
        ):
            transport.packed_input_buffer((2, 3000))

    def test_gather_stages_foreign_operands_and_uses_the_landing_in_place(self):
        transport, module = self._transport()
        landing = transport.gather_landing((16, 2, 32), torch.bfloat16)
        out = transport.gather_heads(landing)
        self.assertEqual(tuple(out.shape), (8, 4, 32))
        self.assertEqual(module.exchanges[-1][1], landing.data_ptr())
        foreign = torch.randn(16, 2, 32, dtype=torch.bfloat16)
        transport.gather_heads(foreign)
        self.assertEqual(module.exchanges[-1][1], landing.data_ptr())
        self.assertTrue(torch.equal(landing, foreign))
        transposed = torch.randn(2, 16, 32, dtype=torch.bfloat16).transpose(0, 1)
        transport.gather_heads(transposed)
        self.assertTrue(torch.equal(landing, transposed))


if __name__ == "__main__":
    unittest.main(verbosity=3)
