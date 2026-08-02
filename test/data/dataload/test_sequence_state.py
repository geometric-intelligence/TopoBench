"""Commit-safe Task10 sampler sequence and disk-module resume tests."""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from test.data.dataload.test_disk_graph_datamodule import (
    QualifiedStoreFixture,
    exhaustive_fanout,
    task8_stores,
)
from topobench.callbacks.dataloader_commit import DataloaderCommitCallback
from topobench.data.stores.typed_graph_store import TypedGraphStore
from topobench.dataloader.disk_graph import (
    DiskGraphDataModule,
    HeterogeneousNeighborStrategy,
    SamplingDescriptor,
)
from topobench.dataloader.sequence_state import SequenceIdentity, SequenceState


_STATE_KEYS = {
    "format_version",
    "identity",
    "committed_cursor",
    "committed_rng_state",
    "committed_sampler_state",
    "committed_evaluator_sequence",
    "committed_evaluator_count",
    "committed_evaluator_state",
    "committed_global_step",
    "committed_epoch",
}
_IDENTITY_KEYS = {
    "format_version",
    "store_content_sha256",
    "partition_book_identity",
    "active_split_tag",
    "fitted_transform_state_key",
    "sampling_strategy_type",
    "sampling_strategy_config",
    "sampling_strategy_fingerprint",
    "phase_descriptor_digest",
    "phase_descriptor_order",
    "sampler_rng_identity",
}


def _descriptor(
    ordinal: int,
    *,
    strategy: str = "homogeneous-cluster",
    phase: str = "train",
) -> SamplingDescriptor:
    options = (
        {"clusters_per_batch": 1, "partition_groups": None}
        if strategy.endswith("cluster")
        else {
            "batch_size": 1,
            "fanout": [
                {
                    "relation": ["author", "writes", "paper"],
                    "values": [-1],
                }
            ],
            "filter_per_worker": False,
            "replace": False,
            "sample_direction": "forward",
            "subgraph_type": "directional",
        }
    )
    common = {
        "content_sha256": "a" * 64,
        "active_split_tag": "primary",
        "phase": phase,
        "strategy": strategy,
        "strategy_options_json": json.dumps(
            options, sort_keys=True, separators=(",", ":")
        ),
        "batch_ordinal": ordinal,
        "participant_counts": (("node", ordinal + 1),),
        "generator_seed": 100 + ordinal,
        "generator_state_sha256": f"{ordinal + 1:064x}",
    }
    if strategy == "heterogeneous-neighbor":
        common["participant_counts"] = ()
        return SamplingDescriptor(
            **common,
            target_node_type="paper",
            target_seed_ids=(10 + ordinal,),
        )
    return SamplingDescriptor(**common, partition_ids=(ordinal,))


def _identity(
    descriptors: tuple[SamplingDescriptor, ...],
    *,
    partition_book_identity: str = "b" * 64,
) -> SequenceIdentity:
    return SequenceIdentity.from_descriptors(
        descriptors,
        partition_book_identity=partition_book_identity,
        fitted_transform_state_key="c" * 64,
        sampler_state={
            "format_version": "graph-sampling-state-v1",
            "seed": 17,
            "strategy": descriptors[0].strategy,
        },
    )


def _ready(state: SequenceState, descriptor: SamplingDescriptor) -> int:
    sequence_id = state.issue(descriptor)
    state.prepare(sequence_id, descriptor)
    state.deliver(sequence_id)
    state.consume(sequence_id)
    return sequence_id


def _assert_canonical_tuple(value: object) -> None:
    assert value is None or type(value) in {bool, int, float, str, tuple}
    if isinstance(value, tuple):
        for item in value:
            _assert_canonical_tuple(item)


@pytest.mark.parametrize(
    "strategy",
    [
        "homogeneous-cluster",
        "heterogeneous-cluster",
        "heterogeneous-neighbor",
    ],
)
def test_identity_binds_both_descriptor_families_and_normalizes_safe_state(
    strategy: str,
) -> None:
    descriptors = tuple(_descriptor(index, strategy=strategy) for index in range(3))
    identity = _identity(descriptors)

    assert isinstance(hash(identity), int)
    _assert_canonical_tuple(identity.sampling_strategy_config)
    state = SequenceState(identity, descriptors)
    assert _ready(state, descriptors[0]) == 1
    assert state.commit(
        optimizer_succeeded=True,
        model_global_step=1,
        evaluator_sequence=1,
        evaluator_count=1,
        evaluator_state={"sum": torch.tensor(2.0), "labels": ("train",)},
        epoch=0,
    )
    assert state.committed_cursor == 1


def test_identity_rejects_process_metadata_and_descriptor_identity_mismatches() -> None:
    descriptors = (_descriptor(0), _descriptor(1))
    with pytest.raises(ValueError, match="process metadata"):
        SequenceIdentity.from_descriptors(
            descriptors,
            partition_book_identity="b" * 64,
            fitted_transform_state_key=None,
            sampler_state={"pid": 123, "seed": 1},
        )

    state = SequenceState(_identity(descriptors), descriptors)
    with pytest.raises(ValueError, match="train phase"):
        state.issue(replace(descriptors[0], phase="val"))
    with pytest.raises(ValueError, match="descriptor.*sequence 1"):
        state.issue(replace(descriptors[0], partition_ids=(99,)))


def test_out_of_order_preparation_still_delivers_and_consumes_exactly_in_order() -> None:
    descriptors = tuple(_descriptor(index) for index in range(3))
    state = SequenceState(_identity(descriptors), descriptors)
    sequence_ids = tuple(state.issue(descriptor) for descriptor in descriptors)

    state.prepare(sequence_ids[1], descriptors[1])
    state.prepare(sequence_ids[0], descriptors[0])
    with pytest.raises(ValueError, match="expected sequence 1"):
        state.deliver(sequence_ids[1])
    assert state.deliver(sequence_ids[0]) == descriptors[0]
    state.consume(sequence_ids[0])
    assert state.deliver(sequence_ids[1]) == descriptors[1]
    state.consume(sequence_ids[1])
    with pytest.raises(ValueError, match="already prepared"):
        state.prepare(sequence_ids[1], descriptors[1])
    with pytest.raises(ValueError, match="already consumed"):
        state.consume(sequence_ids[1])

    assert state.pending_group == (1, 2)
    before = state.state_dict()
    assert not state.commit(
        optimizer_succeeded=False,
        model_global_step=0,
        evaluator_sequence=2,
        evaluator_count=2,
        evaluator_state={"sum": 2},
        epoch=0,
    )
    assert state.state_dict() == before
    assert not state.commit(
        optimizer_succeeded=True,
        model_global_step=0,
        evaluator_sequence=2,
        evaluator_count=2,
        evaluator_state={"sum": 2},
        epoch=0,
    )
    assert state.state_dict() == before
    with pytest.raises(ValueError, match="evaluator sequence"):
        state.commit(
            optimizer_succeeded=True,
            model_global_step=2,
            evaluator_sequence=1,
            evaluator_count=2,
            evaluator_state={"sum": 2},
            epoch=0,
        )
    assert state.state_dict() == before

    assert state.commit(
        optimizer_succeeded=True,
        model_global_step=2,
        evaluator_sequence=2,
        evaluator_count=2,
        evaluator_state={"sum": 2},
        epoch=0,
    )
    assert state.committed_cursor == 2
    assert state.pending_group == ()


@pytest.mark.parametrize(
    "boundary",
    [
        "before_issue",
        "after_issue",
        "after_collation",
        "after_delivery",
        "after_backward",
        "after_evaluator_update",
        "before_optimizer",
    ],
)
def test_every_uncommitted_interruption_regenerates_once_from_committed_cursor(
    boundary: str,
) -> None:
    descriptor = _descriptor(0)
    identity = _identity((descriptor,))
    state = SequenceState(identity, (descriptor,))
    sequence_id = None
    if boundary != "before_issue":
        sequence_id = state.issue(descriptor)
    if boundary in {
        "after_collation",
        "after_delivery",
        "after_backward",
        "after_evaluator_update",
        "before_optimizer",
    }:
        assert sequence_id is not None
        state.prepare(sequence_id, descriptor)
    if boundary in {
        "after_delivery",
        "after_backward",
        "after_evaluator_update",
        "before_optimizer",
    }:
        assert sequence_id is not None
        state.deliver(sequence_id)
    if boundary in {"after_backward", "after_evaluator_update", "before_optimizer"}:
        assert sequence_id is not None
        state.consume(sequence_id)

    checkpoint = state.state_dict()
    assert checkpoint["committed_cursor"] == 0
    assert not any(
        word in key
        for key in checkpoint
        for word in ("issued", "prepared", "delivered", "consumed", "pending")
    )
    restored = SequenceState(identity, (descriptor,))
    restored.load_state_dict(checkpoint)
    assert restored.issue(descriptor) == 1
    with pytest.raises(ValueError, match="descriptor.*sequence 2"):
        restored.issue(replace(descriptor, partition_ids=(1,)))


def test_checkpoint_whitelist_strict_tamper_rejection_and_committed_resume() -> None:
    descriptors = tuple(_descriptor(index) for index in range(3))
    identity = _identity(descriptors)
    state = SequenceState(identity, descriptors)
    _ready(state, descriptors[0])
    state.commit(
        optimizer_succeeded=True,
        model_global_step=1,
        evaluator_sequence=1,
        evaluator_count=1,
        evaluator_state={"correct": torch.tensor(1), "nested": [True, 2.0]},
        epoch=0,
    )
    checkpoint = state.state_dict()

    assert set(checkpoint) == _STATE_KEYS
    assert set(checkpoint["identity"]) == _IDENTITY_KEYS
    assert checkpoint["committed_cursor"] == 1
    serialized = repr(checkpoint).lower()
    assert "gradient" not in serialized
    assert "worker" not in serialized
    assert "queue" not in serialized
    assert "path" not in serialized
    assert "sampling_descriptor" not in serialized

    restored = SequenceState(identity, descriptors)
    restored.load_state_dict(checkpoint)
    assert restored.issue(descriptors[1]) == 2
    with pytest.raises(ValueError, match="already issued"):
        restored.issue(descriptors[1])

    tampered_states = []
    extra = copy.deepcopy(checkpoint)
    extra["issued_cursor"] = 9
    tampered_states.append(extra)
    wrong_version = copy.deepcopy(checkpoint)
    wrong_version["format_version"] = "sequence-state-v999"
    tampered_states.append(wrong_version)
    wrong_identity = copy.deepcopy(checkpoint)
    wrong_identity["identity"]["active_split_tag"] = "other"
    tampered_states.append(wrong_identity)
    wrong_cursor = copy.deepcopy(checkpoint)
    wrong_cursor["committed_cursor"] = True
    tampered_states.append(wrong_cursor)
    wrong_rng = copy.deepcopy(checkpoint)
    wrong_rng["committed_rng_state"]["generator_seed"] += 1
    tampered_states.append(wrong_rng)
    wrong_evaluator = copy.deepcopy(checkpoint)
    wrong_evaluator["committed_evaluator_sequence"] = 0
    tampered_states.append(wrong_evaluator)
    wrong_global_step = copy.deepcopy(checkpoint)
    wrong_global_step["committed_global_step"] = 0
    tampered_states.append(wrong_global_step)

    for tampered in tampered_states:
        candidate = SequenceState(identity, descriptors)
        with pytest.raises((TypeError, ValueError)):
            candidate.load_state_dict(tampered)
        assert candidate.committed_cursor == 0

    other = SequenceState(
        _identity(descriptors, partition_book_identity="d" * 64), descriptors
    )
    with pytest.raises(ValueError, match="identity mismatch"):
        other.load_state_dict(checkpoint)


@pytest.mark.parametrize("callback_count", [0, 2])
def test_attached_multi_epoch_training_requires_exactly_one_commit_callback(
    task8_stores: dict[str, QualifiedStoreFixture],
    callback_count: int,
) -> None:
    fixture = task8_stores["heterogeneous"]
    with TypedGraphStore.open(fixture.store_build.path) as store:
        fanout = exhaustive_fanout(store.relation_types)
    module = DiskGraphDataModule(
        fixture.store_build.path,
        HeterogeneousNeighborStrategy(
            batch_size=1,
            num_neighbors=fanout,
            seed=31,
        ),
        train_shuffle=False,
    )
    module.trainer = SimpleNamespace(
        callbacks=[DataloaderCommitCallback() for _ in range(callback_count)],
        max_epochs=2,
    )
    with pytest.raises(
        RuntimeError,
        match="exactly one DataloaderCommitCallback",
    ):
        module.train_dataloader()
    module.close()


def test_disk_module_resumes_only_committed_train_sequences_and_isolates_eval(
    task8_stores: dict[str, QualifiedStoreFixture],
) -> None:
    fixture = task8_stores["heterogeneous"]
    with TypedGraphStore.open(fixture.store_build.path) as store:
        fanout = exhaustive_fanout(store.relation_types)
    module = DiskGraphDataModule(
        fixture.store_build.path,
        HeterogeneousNeighborStrategy(
            batch_size=1,
            num_neighbors=fanout,
            seed=29,
        ),
        train_shuffle=False,
    )
    module.setup("fit")
    train_descriptors = module.descriptors("train")
    assert len(train_descriptors) >= 2
    iterator = iter(module.train_dataloader())
    first = next(iterator)
    assert first.sequence_id == 1
    assert first.sampling_descriptor == train_descriptors[0]
    module.consume_sequence(first.sequence_id)
    assert module.commit_optimizer_step(
        optimizer_succeeded=True,
        model_global_step=1,
        evaluator_snapshot={
            "sequence_id": 1,
            "count": 1,
            "state": {"correct": torch.tensor(1)},
        },
        epoch=0,
    )
    checkpoint = module.state_dict()
    module.close()

    resumed = DiskGraphDataModule(
        fixture.store_build.path,
        HeterogeneousNeighborStrategy(
            batch_size=1,
            num_neighbors=fanout,
            seed=29,
        ),
        train_shuffle=False,
    )
    resumed.load_state_dict(checkpoint)
    remaining = list(resumed.train_dataloader())
    assert [batch.sequence_id for batch in remaining] == list(
        range(2, len(train_descriptors) + 1)
    )
    assert [batch.sampling_descriptor for batch in remaining] == list(
        train_descriptors[1:]
    )

    before_eval = resumed.state_dict()
    list(resumed.val_dataloader())
    list(resumed.test_dataloader())
    after_eval = resumed.state_dict()
    assert after_eval.keys() == before_eval.keys()
    assert after_eval["committed_cursor"] == before_eval["committed_cursor"] == 1
    assert after_eval["identity"] == before_eval["identity"]
    assert torch.equal(
        after_eval["committed_evaluator_state"]["correct"],
        before_eval["committed_evaluator_state"]["correct"],
    )
    resumed.close()
