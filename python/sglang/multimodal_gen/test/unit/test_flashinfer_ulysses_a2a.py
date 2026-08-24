# SPDX-License-Identifier: Apache-2.0

import torch

from sglang.multimodal_gen.runtime.distributed.device_communicators.flashinfer_ulysses_a2a import (
    FlashInferUlyssesA2A,
    _TRANSPORTS,
    reset_flashinfer_ulysses_request_stats,
    validate_flashinfer_ulysses_request,
)


class _FakeCommunicator:
    backend = "pcie"
    transport = "hybrid"
    device = torch.device("cpu")

    def __init__(self, world_size: int = 2):
        self.world_size = world_size
        self.allocations = []
        self.exchanges = 0
        self.closed = False

    def allocate_output(self, sample: torch.Tensor, op: str) -> torch.Tensor:
        self.allocations.append((op, tuple(sample.shape)))
        batch, tokens, heads, head_dim = sample.shape
        if op == "scatter_heads":
            shape = (
                batch,
                tokens * self.world_size,
                heads // self.world_size,
                head_dim,
            )
        else:
            shape = (
                batch,
                tokens // self.world_size,
                heads * self.world_size,
                head_dim,
            )
        return torch.empty(shape, dtype=sample.dtype)

    def allocate_input(self, output: torch.Tensor, op: str) -> torch.Tensor:
        batch, tokens, heads, head_dim = output.shape
        assert op == "scatter_heads"
        return torch.empty(
            (
                batch,
                tokens // self.world_size,
                heads * self.world_size,
                head_dim,
            ),
            dtype=output.dtype,
        )

    def scatter_heads(self, source: torch.Tensor, *, out: torch.Tensor):
        self.exchanges += 1
        return out

    def gather_heads(self, source: torch.Tensor, *, out: torch.Tensor):
        self.exchanges += 1
        return out

    def stats(self, reset: bool = False):
        result = {"exchanges": self.exchanges}
        if reset:
            self.exchanges = 0
        return result

    def close(self):
        self.closed = True


def _transport(*, strict: bool = True, max_shapes: int = 4, max_elems: int = 4096):
    comm = _FakeCommunicator()
    return (
        FlashInferUlyssesA2A(
            comm,
            world_size=2,
            max_shapes=max_shapes,
            max_elems=max_elems,
            strict=strict,
        ),
        comm,
    )


def test_prepare_geometry_registers_pair_once_and_reuses_direct_input():
    transport, comm = _transport()

    transport.prepare_geometry(4, 8, 4, torch.float16, direct_input=True)
    transport.prepare_geometry(4, 8, 4, torch.float16, direct_input=True)
    qkv = transport.qkv_buffer(4, 8, 4, torch.float16)

    assert qkv is not None
    assert tuple(qkv.shape) == (4, 8, 3, 4)
    assert comm.allocations == [
        ("scatter_heads", (1, 4, 24, 4)),
        ("gather_heads", (1, 8, 4, 4)),
    ]
    assert transport.stats()["registered_geometries"] == 1


def test_exchange_pair_is_counted_and_preserves_contract_shapes():
    transport, comm = _transport()
    transport.prepare_geometry(4, 8, 4, torch.float16, direct_input=True)

    qkv = transport.qkv_buffer(4, 8, 4, torch.float16)
    q, k, v = transport.exchange_input(qkv)
    gathered = transport.exchange_output(torch.empty((8, 4, 4), dtype=torch.float16))

    assert tuple(q.shape) == tuple(k.shape) == tuple(v.shape) == (8, 4, 4)
    assert tuple(gathered.shape) == (4, 8, 4)
    assert transport.stats()["sglang_exchanges"] == 2
    assert comm.exchanges == 2


def test_strict_mode_rejects_capacity_and_geometry_budget_overflow():
    transport, _ = _transport(max_shapes=1, max_elems=383)

    try:
        transport.prepare_geometry(4, 8, 4, torch.float16, direct_input=True)
    except RuntimeError as error:
        assert "declared capacity" in str(error)
    else:
        raise AssertionError("strict capacity overflow must fail")

    transport, _ = _transport(max_shapes=1)
    transport.prepare_geometry(4, 8, 4, torch.float16, direct_input=False)
    try:
        transport.prepare_geometry(2, 8, 4, torch.float16, direct_input=False)
    except RuntimeError as error:
        assert "geometry budget" in str(error)
    else:
        raise AssertionError("strict geometry budget overflow must fail")


def test_stats_reset_and_collective_shutdown():
    transport, comm = _transport()
    transport.prepare_geometry(4, 8, 4, torch.float16, direct_input=True)
    transport.exchange_input(transport.qkv_buffer(4, 8, 4, torch.float16))

    assert transport.stats(reset=True)["sglang_exchanges"] == 1
    assert transport.stats()["sglang_exchanges"] == 0
    transport.shutdown()
    assert comm.closed is True


def test_request_interval_requires_exact_exchange_delta():
    transport, _ = _transport()
    _TRANSPORTS["test"] = transport
    try:
        transport._exchanges = 17
        reset_flashinfer_ulysses_request_stats()
        transport._exchanges = 20
        assert (
            validate_flashinfer_ulysses_request(20, strict=True)[0]["sglang_exchanges"]
            == 20
        )
        try:
            validate_flashinfer_ulysses_request(19, strict=True)
        except RuntimeError as error:
            assert "expected 19" in str(error)
        else:
            raise AssertionError("strict request exchange mismatch must fail")
    finally:
        _TRANSPORTS.pop("test")
