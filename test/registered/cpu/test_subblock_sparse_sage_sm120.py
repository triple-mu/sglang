# SPDX-License-Identifier: Apache-2.0
"""Device-free contracts of the SM120 SubBlock Sage backend: route guard, block tables, registration."""

import unittest
from unittest.mock import Mock

import torch

from sglang.multimodal_gen.runtime.layers.attention.backends.subblock_sparse_sage_sm120 import (
    cake_block_tables,
    h3_live_rows,
    lowp_route_reason,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.platforms.cuda import (
    _SubBlockSparseSageSM120BackendResolver,
)
from sglang.multimodal_gen.runtime.platforms.interface import DeviceCapability
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=6, suite="stage-a-test-cpu-intel")


def _shard(rows=256, heads=8, head_dim=128, dtype=torch.bfloat16):
    qkv = torch.zeros(rows, 3, heads, head_dim, dtype=dtype)
    return qkv[:, 0], qkv[:, 1], qkv[:, 2]


def _reason(q, k, v, **overrides):
    kwargs = dict(
        cu_seqlens_host=(0, 1000, 1024),
        max_seqlen=1000,
        ring_active=False,
        world_size=4,
        enabled=True,
    )
    kwargs.update(overrides)
    return lowp_route_reason(q, k, v, **kwargs)


class TestLiveRows(CustomTestCase):
    def test_h3_layout_yields_the_live_row_count(self):
        self.assertEqual(h3_live_rows((0, 1000, 1024), 1000, 1024), 1000)
        self.assertEqual(h3_live_rows((0, 1024, 1024), 1024, 1024), 1024)

    def test_other_layouts_are_rejected(self):
        self.assertIsNone(h3_live_rows(None, 1000, 1024))
        self.assertIsNone(h3_live_rows((0, 500, 1000, 1024), 500, 1024))
        self.assertIsNone(h3_live_rows((0, 1000, 1024), 1000, 2048))
        self.assertIsNone(h3_live_rows((0, 1000, 1024), 999, 1024))
        self.assertIsNone(h3_live_rows((0, 0, 1024), 0, 1024))


class TestRouteGuard(CustomTestCase):
    def test_each_guard_names_its_reason(self):
        q, k, v = _shard()
        self.assertEqual(_reason(q, k, v, enabled=False), "disabled")
        self.assertEqual(_reason(q, k, v, world_size=1), "not_ulysses")
        self.assertEqual(_reason(q, k, v, ring_active=True), "ring")
        qh, kh, vh = _shard(dtype=torch.float16)
        self.assertEqual(_reason(qh, kh, vh), "dtype")
        q64, k64, v64 = _shard(head_dim=64)
        self.assertEqual(_reason(q64, k64, v64), "shape")
        self.assertEqual(_reason(q, k, v[:, :4]), "shape")
        q100, k100, v100 = _shard(rows=100)
        self.assertEqual(
            _reason(q100, k100, v100, cu_seqlens_host=(0, 300, 400), max_seqlen=300),
            "alignment",
        )
        self.assertEqual(_reason(q, k, v, world_size=3), "alignment")
        qt = q.transpose(1, 2).transpose(1, 2)  # same shape, still unit stride
        self.assertNotEqual(_reason(qt, k, v), "stride")
        strided = torch.zeros(256, 128, 8, dtype=torch.bfloat16).transpose(1, 2)
        self.assertEqual(_reason(strided, k, v), "stride")
        self.assertEqual(_reason(q, k, v, cu_seqlens_host=None), "packed_layout")
        self.assertEqual(
            _reason(q, k, v, cu_seqlens_host=(0, 500, 500, 1024)), "packed_layout"
        )
        self.assertEqual(
            _reason(q, k, v, cu_seqlens_host=(0, 1000, 2048)), "packed_layout"
        )
        # Every rank-invariant guard passes; only the device is missing here.
        self.assertEqual(_reason(q, k, v), "device")


class TestCakeBlockTables(CustomTestCase):
    def setUp(self):
        self.index = torch.tensor(
            [[[[3, 0], [1, 2], [2, 3]], [[0, 1], [3, 2], [1, 0]]]], dtype=torch.int32
        )  # [1, 2 heads, 3 q blocks, topk=2] of 4 key blocks

    def test_without_a_mask_every_query_block_keeps_topk(self):
        tables, nums = cake_block_tables(self.index, 2, 4, None)
        self.assertTrue(torch.equal(tables, self.index))
        self.assertEqual(nums.dtype, torch.int32)
        self.assertEqual(nums.tolist(), [[[2, 2, 2], [2, 2, 2]]])

    def test_masked_off_query_blocks_keep_every_key_block(self):
        mask = torch.tensor([True, False, True])
        tables, nums = cake_block_tables(self.index, 2, 4, mask)
        self.assertEqual(tuple(tables.shape), (1, 2, 3, 4))
        self.assertEqual(nums.tolist(), [[[2, 4, 2], [2, 4, 2]]])
        for head in range(2):
            self.assertEqual(tables[0, head, 1].tolist(), [0, 1, 2, 3])
            self.assertEqual(
                tables[0, head, 0, :2].tolist(), self.index[0, head, 0].tolist()
            )
            self.assertEqual(
                tables[0, head, 2, :2].tolist(), self.index[0, head, 2].tolist()
            )

    def test_mask_length_must_match_the_query_blocks(self):
        with self.assertRaises(ValueError):
            cake_block_tables(self.index, 2, 4, torch.tensor([True, False]))


class TestRegistration(CustomTestCase):
    def test_enum_is_sparse_and_spells_its_cli_name(self):
        backend = AttentionBackendEnum.SUBBLOCK_SPARSE_SAGE_SM120
        self.assertTrue(backend.is_sparse)
        self.assertEqual(str(backend), "subblock_sparse_sage_sm120")

    def test_resolver_fails_closed_off_sm120(self):
        for capability in (DeviceCapability(9, 0), DeviceCapability(10, 0), None):
            platform = Mock()
            platform.get_device_capability.return_value = capability
            with self.assertRaisesRegex(ValueError, "compute capability 12.0"):
                _SubBlockSparseSageSM120BackendResolver.resolve(platform)


if __name__ == "__main__":
    unittest.main(verbosity=3)
