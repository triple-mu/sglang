import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.multimodal_gen import envs
from sglang.multimodal_gen.runtime.layers import custom_ulysses_a2a as cua
from sglang.multimodal_gen.runtime.layers import usp
from sglang.multimodal_gen.runtime.layers.attention.layer import USPAttention


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
            self.assertEqual(envs.SGLANG_DIFFUSION_CUSTOM_ULYSSES_POOL_BYTES, 2 << 30)

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


class TestCustomUlyssesGate(unittest.TestCase):
    def test_disabled_when_env_off(self):
        with patch.dict(os.environ, {"SGLANG_DIFFUSION_USE_CUSTOM_ULYSSES_A2A": "0"}):
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
        layer_mod = "sglang.multimodal_gen.runtime.layers.attention.layer"
        with patch(
            f"{layer_mod}.get_forward_context",
            return_value=SimpleNamespace(attn_metadata=None),
        ), patch(
            f"{layer_mod}.get_sequence_parallel_world_size", return_value=2
        ), patch(
            f"{layer_mod}.get_ulysses_parallel_world_size", return_value=2
        ), patch(
            f"{layer_mod}.get_ring_parallel_world_size", return_value=1
        ), patch(
            f"{layer_mod}._usp_input_all_to_all", side_effect=rec_in
        ), patch(
            f"{layer_mod}._usp_output_all_to_all", side_effect=rec_out
        ):
            attn.forward(q, q.clone(), q.clone())

        tags = [c[2] for c in captured]
        self.assertEqual(tags, ["uin_q", "uin_k", "uin_v", "uout"])
        self.assertEqual(len(set(tags)), 4)
        self.assertTrue(all(c[1] == 2 for c in captured))


if __name__ == "__main__":
    unittest.main()
