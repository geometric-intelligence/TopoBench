"""CPU qualification for bounded native graph host/device prefetch."""

from __future__ import annotations

import time
import threading
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Iterator
from dataclasses import replace

import pytest
import torch
from torch_geometric.data import Data, HeteroData

from test.data.dataload.test_disk_graph_datamodule import task8_stores
from test.data.stores.test_typed_graph_store import QualifiedStoreFixture
from topobench.dataloader.device_prefetch import (
    DevicePrefetchLoader,
    PrefetchCapability,
    PrefetchError,
    PrefetchLimitError,
    PrefetchLimits,
    estimate_batch_bytes,
    _cuda_source_mode,
)
from topobench.dataloader.disk_graph import (
    DiskGraphDataModule,
    HomogeneousClusterStrategy,
)

_MIB = 1024 * 1024


def _limits(**changes: object) -> PrefetchLimits:
    values: dict[str, object] = {
        "host_queue_depth": 2,
        "device_queue_depth": 0,
        "max_batch_nodes": 100,
        "max_batch_edges": 200,
        "max_nodes_per_type": {},
        "max_edges_per_relation": {},
        "max_host_batch_bytes": _MIB,
        "max_device_batch_bytes": _MIB,
        "max_host_queue_bytes": 2 * _MIB,
        "max_device_queue_bytes": 0,
        "worst_case_host_bytes": None,
        "worst_case_device_bytes": None,
    }
    values.update(changes)
    return PrefetchLimits(**values)


def _capability(
    device: str = "cpu", *, pin_memory_supported: bool = False
) -> PrefetchCapability:
    return PrefetchCapability(
        device=torch.device(device),
        cuda_available=False,
        pin_memory_supported=pin_memory_supported,
    )


def _batch(sequence_id: int) -> Data:
    storage = torch.arange(12, dtype=torch.float32) + 100 * sequence_id
    batch = Data(
        x=storage[:8].view(4, 2),
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long),
        num_nodes=4,
    )
    batch.alias = storage[4:12].view(4, 2)
    batch.sequence_id = sequence_id
    return batch


def _wait_until(predicate: object, timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():  # type: ignore[operator]
            return
        time.sleep(0.005)
    raise AssertionError("timed out waiting for prefetch lifecycle state")


def test_prefetch_limits_are_frozen_strict_and_reject_unsafe_queue_products() -> None:
    limits = _limits()
    with pytest.raises(AttributeError):
        limits.host_queue_depth = 3  # type: ignore[misc]

    for name, value, error in (
        ("host_queue_depth", True, TypeError),
        ("host_queue_depth", 0, ValueError),
        ("device_queue_depth", -1, ValueError),
        ("max_batch_nodes", 1.5, TypeError),
        ("max_batch_edges", -1, ValueError),
        ("max_host_batch_bytes", True, TypeError),
        ("max_device_queue_bytes", -1, ValueError),
    ):
        with pytest.raises(error, match=name):
            _limits(**{name: value})

    with pytest.raises(ValueError, match="worst_case_host_bytes.*host queue"):
        _limits(
            host_queue_depth=3,
            max_host_queue_bytes=299,
            max_host_batch_bytes=100,
            worst_case_host_bytes=100,
        )
    with pytest.raises(ValueError, match="worst_case_device_bytes.*device queue"):
        _limits(
            device_queue_depth=3,
            max_device_batch_bytes=100,
            max_device_queue_bytes=299,
            worst_case_device_bytes=100,
        )
    with pytest.raises(TypeError, match="max_nodes_per_type"):
        _limits(max_nodes_per_type={1: 2})
    with pytest.raises(ValueError, match="duplicate relation"):
        _limits(
            max_edges_per_relation=[
                (("a", "r", "b"), 1),
                (("a", "r", "b"), 2),
            ]
        )


def test_worst_case_host_budget_includes_live_cuda_staging() -> None:
    with pytest.raises(
        ValueError,
        match="worst_case_host_bytes.*live CUDA staging",
    ):
        _limits(
            host_queue_depth=2,
            device_queue_depth=3,
            max_host_batch_bytes=100,
            max_device_batch_bytes=100,
            max_host_queue_bytes=499,
            max_device_queue_bytes=300,
            worst_case_host_bytes=100,
            worst_case_device_bytes=100,
        )


def test_loader_enforces_one_active_producer_and_reopens_after_close() -> None:
    loader = DevicePrefetchLoader(
        [_batch(1), _batch(2)],
        _limits(host_queue_depth=1),
        capability=_capability(),
    )
    first = iter(loader)
    second = None
    try:
        with pytest.raises(RuntimeError, match="one active prefetch iterator"):
            second = iter(loader)
        first.close()
        reopened = iter(loader)
        assert next(reopened).sequence_id == 1
        reopened.close()
    finally:
        if second is not None:
            second.close()
        first.close()
        loader.close()


def test_capability_rejects_unknown_negative_and_host_device_ring() -> None:
    for device, error in ((True, TypeError), (-1, ValueError), ("xpu", ValueError)):
        with pytest.raises(error):
            PrefetchCapability.detect(device)

    with pytest.raises(ValueError, match="device_queue_depth must be zero"):
        DevicePrefetchLoader(
            [_batch(1)],
            _limits(device_queue_depth=1, max_device_queue_bytes=_MIB),
            capability=_capability("mps"),
        )


def test_data_accounting_counts_alias_storage_once_and_sparse_components() -> None:
    storage = torch.arange(16, dtype=torch.float32)
    sparse = torch.sparse_coo_tensor(
        torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        torch.tensor([2.0, 3.0], dtype=torch.float64),
        size=(3, 3),
    )
    batch = Data(
        x=storage[:8].view(4, 2),
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long),
        edge_weight=torch.tensor([1, 2, 3], dtype=torch.int16),
        num_nodes=4,
    )
    batch.alias = storage[4:12].view(4, 2)
    batch.graph_payload = {"sparse": sparse}

    estimate = estimate_batch_bytes(batch)
    expected = sum(
        {
            tensor.untyped_storage()._cdata: tensor.untyped_storage().nbytes()
            for tensor in (
                storage,
                batch.edge_index,


                batch.edge_weight,
                sparse._indices(),
                sparse._values(),
            )
        }.values()
    )
    assert estimate.total_bytes == expected
    assert estimate.host_bytes == expected
    assert estimate.node_count == 4
    assert estimate.edge_count == 3
    assert estimate.node_counts == (("__homogeneous__", 4),)
    assert estimate.edge_counts == (("__homogeneous__", 3),)
    assert sum(field.admitted_bytes for field in estimate.fields) == expected
    assert tuple(field.field_path for field in estimate.fields) == tuple(
        sorted(field.field_path for field in estimate.fields)
    )
    sparse_fields = [
        field for field in estimate.fields if field.layout == "torch.sparse_coo"
    ]
    assert {field.component for field in sparse_fields} == {"indices", "values"}
    assert {field.dtype for field in sparse_fields} == {
        "torch.int64",
        "torch.float64",
    }
    assert (
        dict(estimate.node_bytes)["__homogeneous__"]
        == storage.untyped_storage().nbytes()
    )
    assert dict(estimate.edge_bytes)["__homogeneous__"] > 0
    assert estimate.global_bytes > 0
def test_prefetch_never_rewrites_caller_native_graphs() -> None:
    homogeneous = _batch(1)
    heterogeneous = HeteroData()
    values = torch.arange(24, dtype=torch.float32).view(6, 4)
    heterogeneous["paper"].x = values
    heterogeneous["paper"].alias = values[:, :2]
    heterogeneous[("paper", "cites", "paper")].edge_index = torch.tensor(
        [[0, 1], [1, 2]],
        dtype=torch.long,
    )

    for original in (homogeneous, heterogeneous):
        bindings = {
            (id(store), key): value
            for store in original.stores
            for key, value in store.items()
            if isinstance(value, torch.Tensor)
        }
        loader = DevicePrefetchLoader(
            [original],
            _limits(),
            capability=_capability(pin_memory_supported=True),
            pin_memory=lambda tensor: tensor.clone(),
        )
        output = next(iter(loader))
        assert output is not original
        for store in original.stores:
            for key, value in store.items():
                if isinstance(value, torch.Tensor):
                    assert value is bindings[(id(store), key)]
        if isinstance(output, Data):
            assert (
                output.x.untyped_storage()._cdata
                == output.alias.untyped_storage()._cdata
            )
        else:
            assert (
                output["paper"].x.untyped_storage()._cdata
                == output["paper"].alias.untyped_storage()._cdata
            )
        loader.close()


def test_cuda_source_classification_is_identity_or_all_cpu_only() -> None:
    target = torch.device("meta")
    resident = Data(
        x=torch.empty((4, 2), device=target),
        edge_index=torch.empty((2, 0), dtype=torch.long, device=target),
        num_nodes=4,
    )
    assert _cuda_source_mode(_batch(1), target) == "host"
    assert _cuda_source_mode(resident, target) == "resident"

    mixed = Data(
        x=resident.x,
        edge_index=torch.empty((2, 0), dtype=torch.long),
        num_nodes=4,
    )
    with pytest.raises(PrefetchLimitError, match="single-device.*cpu.*meta"):
        _cuda_source_mode(mixed, target)
    with pytest.raises(PrefetchLimitError, match="cuda:0.*meta"):
        _cuda_source_mode(resident, torch.device("cuda", 0))

def test_active_iterator_reservation_is_atomic_across_threads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = threading.Event()

    class SlowSource:
        def __iter__(self) -> Iterator[Data]:
            gate.wait()

            return iter((_batch(1),))

    loader = DevicePrefetchLoader(
        SlowSource(),
        _limits(host_queue_depth=1),
        capability=_capability(),
        owns_source=True,
    )
    barrier = threading.Barrier(16)
    constructor_barrier = threading.Barrier(16)
    original_start = threading.Thread.start

    def synchronized_start(thread: threading.Thread) -> None:
        original_start(thread)
        if thread.name.startswith("device-prefetch-"):
            try:
                constructor_barrier.wait(timeout=0.2)
            except threading.BrokenBarrierError:
                pass

    monkeypatch.setattr(threading.Thread, "start", synchronized_start)

    def create_iterator() -> object:
        barrier.wait()
        try:
            return iter(loader)
        except RuntimeError:
            return None

    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = [pool.submit(create_iterator) for _ in range(16)]
        iterators = [future.result() for future in futures]
    gate.set()
    active = [iterator for iterator in iterators if iterator is not None]
    try:
        assert len(active) == 1
    finally:
        for iterator in active:
            iterator.close()
        loader.close()


def test_heterodata_accounting_and_declared_type_relation_caps_are_exact() -> None:
    batch = HeteroData()
    batch["author"].x = torch.ones((2, 3), dtype=torch.float32)
    batch["paper"].x = torch.ones((4, 5), dtype=torch.float16)
    batch["paper"].x_alias = batch["paper"].x[:, :2]
    relation = ("author", "writes", "paper")
    batch[relation].edge_index = torch.tensor([[0, 1, 1], [1, 2, 3]])
    batch[relation].confidence = torch.ones(3, dtype=torch.float64)
    batch.graph_id = torch.tensor([7], dtype=torch.int32)

    estimate = estimate_batch_bytes(batch)
    assert estimate.node_counts == (("author", 2), ("paper", 4))
    assert estimate.edge_counts == ((relation, 3),)
    assert tuple(name for name, _ in estimate.node_bytes) == ("author", "paper")
    assert tuple(name for name, _ in estimate.edge_bytes) == (relation,)
    assert (
        dict(estimate.node_bytes)["paper"]
        == batch["paper"].x.untyped_storage().nbytes()
    )
    assert estimate.global_bytes == batch.graph_id.untyped_storage().nbytes()

    pin_calls: list[torch.Tensor] = []
    loader = DevicePrefetchLoader(
        [batch],
        _limits(max_nodes_per_type={"paper": 3}),
        capability=_capability(pin_memory_supported=True),
        pin_memory=lambda tensor: pin_calls.append(tensor) or tensor.clone(),
    )
    with pytest.raises(PrefetchError, match="host_admission.*paper") as raised:
        next(iter(loader))
    assert isinstance(raised.value.__cause__, PrefetchLimitError)
    assert pin_calls == []
    loader.close()

    loader = DevicePrefetchLoader(
        [batch],
        _limits(max_edges_per_relation={relation: 2}),
        capability=_capability(),
    )
    with pytest.raises(PrefetchError, match="author.*writes.*paper"):
        next(iter(loader))
    loader.close()


def test_host_producer_is_lazy_bounded_ordered_and_pins_unique_storage_once() -> None:
    class Source:
        def __init__(self) -> None:
            self.opened = False
            self.closed = False

        def __iter__(self) -> Iterator[Data]:
            self.opened = True
            try:
                for sequence_id in range(1, 6):
                    yield _batch(sequence_id)
            finally:
                self.closed = True

    source = Source()
    pin_calls: list[int] = []

    def pin_memory(tensor: torch.Tensor) -> torch.Tensor:
        pin_calls.append(tensor.untyped_storage()._cdata)
        return tensor.clone()

    loader = DevicePrefetchLoader(
        source,
        _limits(host_queue_depth=2),
        capability=_capability(pin_memory_supported=True),
        pin_memory=pin_memory,
        owns_source=True,
    )
    assert not source.opened
    iterator = iter(loader)
    _wait_until(lambda: iterator.max_host_queue_size == 2)
    assert iterator.max_host_queue_size <= 2
    assert iterator.max_host_queued_bytes <= loader.limits.max_host_queue_bytes

    batches = list(iterator)
    assert [batch.sequence_id for batch in batches] == [1, 2, 3, 4, 5]
    assert len(pin_calls) == 10
    for batch in batches:
        assert (
            batch.x.untyped_storage()._cdata
            == batch.alias.untyped_storage()._cdata
        )
        batch.x[2, 0] = -float(batch.sequence_id)
        assert batch.alias[0, 0].item() == -float(batch.sequence_id)
    assert source.closed
    assert not iterator.producer_alive
    assert iterator.closed
    loader.close()
    loader.close()


def test_cpu_and_mps_are_explicit_host_only_modes_without_device_transfer() -> None:
    for device in ("cpu", "mps"):
        batch = _batch(1)
        pointers = {
            tensor.untyped_storage()._cdata
            for store in batch.stores
            for tensor in store.values()
            if isinstance(tensor, torch.Tensor)
        }
        loader = DevicePrefetchLoader(
            [batch], _limits(), capability=_capability(device)
        )
        output = next(iter(loader))
        assert loader.status.mode == "host-only"
        assert "CUDA disabled" in loader.status.detail
        assert all(
            tensor.device.type == "cpu"
            for store in output.stores
            for tensor in store.values()
            if isinstance(tensor, torch.Tensor)
        )
        assert pointers == {
            tensor.untyped_storage()._cdata
            for store in output.stores
            for tensor in store.values()
            if isinstance(tensor, torch.Tensor)
        }
        assert loader.active_iterator is not None
        assert loader.active_iterator.transfer_stream is None
        assert loader.active_iterator.completion_events == ()
        loader.close()


def test_single_batch_caps_fail_before_pin_or_queue_admission() -> None:
    pin_calls: list[torch.Tensor] = []
    loader = DevicePrefetchLoader(
        [_batch(7)],
        _limits(max_batch_nodes=3),
        capability=_capability(pin_memory_supported=True),
        pin_memory=lambda tensor: pin_calls.append(tensor) or tensor.clone(),
    )
    iterator = iter(loader)
    with pytest.raises(PrefetchError, match="host_admission.*sequence=7") as raised:
        next(iterator)
    assert isinstance(raised.value.__cause__, PrefetchLimitError)
    assert pin_calls == []
    assert iterator.max_host_queue_size == 0
    assert not iterator.producer_alive
    loader.close()


def test_early_close_and_worker_exception_chain_root_without_leaks() -> None:
    class ClosingSource:
        def __init__(self) -> None:
            self.closed = False

        def __iter__(self) -> Iterator[Data]:
            try:
                for sequence_id in range(1, 100):
                    yield _batch(sequence_id)
            finally:
                self.closed = True

    source = ClosingSource()
    loader = DevicePrefetchLoader(
        source,
        _limits(host_queue_depth=1),
        capability=_capability(),
        owns_source=True,
    )
    iterator = iter(loader)
    assert next(iterator).sequence_id == 1
    iterator.close()
    iterator.close()
    assert iterator.closed
    assert not iterator.producer_alive
    assert source.closed
    loader.close()

    class WorkerFailure(RuntimeError):
        pass

    def failing_source() -> Iterator[Data]:
        yield _batch(11)
        raise WorkerFailure("materialization exploded")

    loader = DevicePrefetchLoader(
        failing_source(),
        _limits(host_queue_depth=1),
        capability=_capability(),
        owns_source=True,
    )
    iterator = iter(loader)
    assert next(iterator).sequence_id == 11
    with pytest.raises(
        PrefetchError,
        match="host_producer.*sequence=12.*materialization exploded",
    ) as raised:
        next(iterator)
    assert isinstance(raised.value.__cause__, WorkerFailure)
    assert not iterator.producer_alive
    assert iterator.closed
    loader.close()


def test_non_native_batch_is_contextual_and_does_not_deadlock_full_queue() -> None:
    loader = DevicePrefetchLoader(
        [_batch(1), {"not": "native"}, _batch(3)],
        _limits(host_queue_depth=1),
        capability=_capability(),
    )
    iterator = iter(loader)
    assert next(iterator).sequence_id == 1
    with pytest.raises(PrefetchError, match="native Data or HeteroData") as raised:
        next(iterator)
    assert isinstance(raised.value.__cause__, TypeError)
    loader.close()
    assert not iterator.producer_alive


def test_disk_datamodule_prefetch_defers_delivery_and_preserves_commit_state(
    task8_stores: dict[str, QualifiedStoreFixture],
) -> None:
    fixture = task8_stores["homogeneous"]
    module = DiskGraphDataModule(
        fixture.store_build.path,
        HomogeneousClusterStrategy(clusters_per_batch=1, seed=19),
        num_workers=0,
        train_shuffle=False,
        prefetch_limits=_limits(
            host_queue_depth=1,
            max_batch_nodes=10_000,
            max_batch_edges=10_000,
            max_host_batch_bytes=64 * _MIB,
            max_host_queue_bytes=64 * _MIB,
        ),
        prefetch_device="cpu",
    )
    module.setup("fit")
    assert module._owner is not None
    assert module._owner._store is None
    state = module.sequence_state
    durable_before = state.state_dict()

    loader = module.train_dataloader()
    assert isinstance(loader, DevicePrefetchLoader)
    iterator = iter(loader)
    _wait_until(lambda: bool(state.prepared))
    assert state.delivered == ()
    assert module._owner._store is not None

    batch = next(iterator)
    assert batch.sequence_id == 1
    assert state.delivered == (1,)
    assert state.state_dict() == durable_before
    iterator.close()

    val_before = state.state_dict()
    list(module.val_dataloader())
    assert state.state_dict() == val_before
    assert state.delivered == (1,)

    module.teardown("fit")
    module.teardown("fit")
    assert module.closed
    assert not loader.active_iterators

def _sequence_module(
    fixture: QualifiedStoreFixture,
) -> DiskGraphDataModule:
    return DiskGraphDataModule(
        fixture.store_build.path,
        HomogeneousClusterStrategy(
            clusters_per_batch=1,
            partition_groups=((0,), (1,), (2,)),
            seed=23,
        ),
        num_workers=0,
        train_shuffle=False,
        prefetch_limits=_limits(
            host_queue_depth=1,
            max_batch_nodes=10_000,
            max_batch_edges=10_000,
            max_host_batch_bytes=64 * _MIB,
            max_host_queue_bytes=64 * _MIB,
        ),
        prefetch_device="cpu",
    )


def test_prefetch_early_break_rolls_back_transient_sequence_for_reiteration(
    task8_stores: dict[str, QualifiedStoreFixture],
) -> None:
    module = _sequence_module(task8_stores["homogeneous"])
    module.setup("fit")
    loader = module.train_dataloader()
    assert isinstance(loader, DevicePrefetchLoader)
    descriptor_count = len(module.descriptors("train"))

    iterator = iter(loader)
    assert next(iterator).sequence_id == 1
    iterator.close()
    state = module.sequence_state
    assert state.issued == ()
    assert state.prepared == ()
    assert state.delivered == ()

    expected = tuple(range(1, descriptor_count + 1))
    assert [batch.sequence_id for batch in loader] == list(expected)
    assert state.delivered == expected
    module.close()
    assert state.delivered == expected


def test_prefetch_error_rolls_back_transient_sequence_for_reiteration(
    task8_stores: dict[str, QualifiedStoreFixture],
) -> None:
    module = _sequence_module(task8_stores["homogeneous"])
    module.setup("fit")
    loader = module.train_dataloader()
    assert isinstance(loader, DevicePrefetchLoader)
    descriptor_count = len(module.descriptors("train"))
    dataset = loader.source.dataset
    delegate = dataset.strategy

    class FailOnceMaterialization:
        def __init__(self) -> None:
            self.calls = 0
            self.failed = False

        def __getattr__(self, name: str) -> object:
            return getattr(delegate, name)

        def materialize(self, source: object, descriptor: object) -> Data:
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("injected materialization failure")
            return delegate.materialize(source, descriptor)

    dataset.strategy = FailOnceMaterialization()
    iterator = iter(loader)
    with pytest.raises(PrefetchError, match="injected materialization failure"):
        next(iterator)
    state = module.sequence_state
    assert state.issued == ()
    assert state.prepared == ()
    assert state.delivered == ()

    assert [batch.sequence_id for batch in loader] == list(
        range(1, descriptor_count + 1)
    )
    module.close()



def test_replacing_limits_does_not_retain_mutable_cap_mappings() -> None:
    node_caps = {"paper": 4}
    limits = _limits(max_nodes_per_type=node_caps)
    node_caps["paper"] = 99
    assert dict(limits.max_nodes_per_type) == {"paper": 4}
    changed = replace(limits, max_batch_nodes=101)
    assert changed.max_batch_nodes == 101
    with pytest.raises(TypeError):
        limits.max_nodes_per_type["paper"] = 7  # type: ignore[index]
