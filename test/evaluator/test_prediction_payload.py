"""Contract tests for row-aligned, immutable prediction payloads."""

from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from torch_geometric.data import Data, HeteroData

from test.data.stores.test_typed_graph_store import QualifiedStoreFixture
from topobench.data.pipelines.base import PredictionRowAdapter
from topobench.data.stores.typed_graph_store import TypedGraphStore
from topobench.evaluator import EvaluationBatch
from topobench.evaluator.prediction import (
    PredictionIdentity,
    PredictionPayload,
)
from topobench.model import (
    DefaultSupervisionAdapter,
    HeterogeneousNodeSupervisionAdapter,
)

pytest_plugins = ("test.data.stores.test_typed_graph_store",)


def _metadata(
    *, task: str, classes: tuple[str, ...] = (), units: str | None = None
):
    metadata = {
        "target": {"role": "target"},
        "raw_output": {"role": "raw_output"},
        "prediction": {"role": "prediction"},
    }
    semantics = {"task": task, "class_vocabulary": classes, "units": units}
    return metadata, semantics


def _payload(
    identity,
    *,
    target,
    raw_output,
    prediction,
    task="classification",
    classes=(),
    units=None,
    extra_columns=None,
    extra_metadata=None,
):
    metadata, semantics = _metadata(task=task, classes=classes, units=units)
    columns = {
        "target": target,
        "raw_output": raw_output,
        **(extra_columns or {}),
    }
    metadata.update(extra_metadata or {})
    return PredictionPayload(
        identity=identity,
        prediction=prediction,
        columns=columns,
        column_metadata=metadata,
        output_semantics=semantics,
    )


def _classification_rows(row_indices: tuple[int, ...]) -> SimpleNamespace:
    count = len(row_indices)
    return SimpleNamespace(
        logits=torch.zeros(count, 2),
        targets=torch.arange(count) % 2,
        num_examples=count,
        row_indices=torch.tensor(row_indices, dtype=torch.long),
    )


def test_graph_sample_identity_preserves_shuffled_unequal_batch_order() -> (
    None
):
    identity = PredictionIdentity(
        columns={"sample_id": np.asarray(["graph-8", "graph-2", "graph-11"])},
        key=("sample_id",),
    )
    payload = _payload(
        identity,
        target=torch.tensor([1, 0, 1]),
        raw_output=torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]]),
        prediction=torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]]),
        classes=("negative", "positive"),
    )

    assert identity.key == ("sample_id",)
    assert identity.columns["sample_id"].tolist() == [
        "graph-8",
        "graph-2",
        "graph-11",
    ]
    assert payload.num_rows == 3


def test_graph_adapter_keeps_shuffle_order_across_unequal_batches() -> None:
    adapter = PredictionRowAdapter(
        source_graph_id="graphs",
        output_kind="graph",
        task_level="graph",
        class_vocabulary=("negative", "positive"),
    )
    batches = (
        (
            Data(sample_id=np.asarray(["graph-8", "graph-2"])),
            {
                "logits": torch.tensor([[0.1, 0.9], [0.8, 0.2]]),
                "labels": torch.tensor([1, 0]),
            },
        ),
        (
            Data(sample_id=np.asarray(["graph-11"])),
            {
                "logits": torch.tensor([[0.3, 0.7]]),
                "labels": torch.tensor([1]),
            },
        ),
    )
    payloads = []
    ordinal = 0
    for batch, model_out in batches:
        selected = DefaultSupervisionAdapter("graph").select(
            model_out, batch, "Test"
        )
        payloads.append(
            adapter.adapt(
                batch,
                selected,
                phase="test",
                split_ordinal_start=ordinal,
            )
        )
        ordinal += selected.num_examples

    assert [
        sample_id
        for payload in payloads
        for sample_id in payload.identity.columns["sample_id"].tolist()
    ] == ["graph-8", "graph-2", "graph-11"]
    assert [
        split_ordinal
        for payload in payloads
        for split_ordinal in payload.identity.columns["split_ordinal"].tolist()
    ] == [0, 1, 2]
    assert payloads[0].columns["raw_output"] is batches[0][1]["logits"]
    assert torch.equal(
        payloads[0].columns["raw_output"],
        batches[0][1]["logits"],
    )


def test_graph_adapter_rejects_surplus_sample_ids() -> None:
    adapter = PredictionRowAdapter(
        source_graph_id="graphs",
        output_kind="graph",
        task_level="graph",
        class_vocabulary=("negative", "positive"),
    )
    batch = Data(sample_id=np.asarray(["graph-8", "graph-2", "graph-surplus"]))

    with pytest.raises(ValueError, match="sample_id.*aligned"):
        adapter.adapt(
            batch,
            _classification_rows((0, 1)),
            phase="test",
        )


@pytest.mark.parametrize("domain", ["homogeneous", "hypergraph"])
def test_node_identity_is_source_graph_and_global_node_id(domain: str) -> None:
    identity = PredictionIdentity(
        columns={
            "source_graph_id": np.asarray([domain, domain]),
            "global_nid": torch.tensor([41, 7]),
        },
        key=("source_graph_id", "global_nid"),
    )

    assert identity.rows == ((domain, 41), (domain, 7))


@pytest.mark.parametrize("output_kind", ["homogeneous", "hypergraph"])
def test_node_adapter_reuses_returned_mask_rows_without_reselection(
    output_kind: str,
) -> None:
    batch = Data(
        global_nid=torch.tensor([40, 41, 42, 43]),
        val_mask=torch.tensor([False, True, False, True]),
    )
    model_out = {
        "logits": torch.arange(8, dtype=torch.float32).reshape(4, 2),
        "labels": torch.tensor([0, 1, 0, 1]),
    }
    supervision = MagicMock(wraps=DefaultSupervisionAdapter("node"))
    selected = supervision.select(model_out, batch, "Validation")
    batch.val_mask = ~batch.val_mask

    payload = PredictionRowAdapter(
        source_graph_id=f"{output_kind}-source",
        output_kind=output_kind,
        class_vocabulary=("negative", "positive"),
    ).adapt(batch, selected, phase="val")

    supervision.select.assert_called_once()
    assert payload.identity.rows == (
        (f"{output_kind}-source", 41),
        (f"{output_kind}-source", 43),
    )
    assert payload.columns["raw_output"] is selected.logits
    assert payload.columns["target"] is selected.targets


def test_heterogeneous_identity_contains_only_target_seed_rows() -> None:
    identity = PredictionIdentity(
        columns={
            "source_graph_id": np.asarray(["mag", "mag"]),
            "target_node_type": np.asarray(["paper", "paper"]),
            "n_id": torch.tensor([19, 4]),
        },
        key=("source_graph_id", "target_node_type", "n_id"),
    )
    payload = _payload(
        identity,
        target=torch.tensor([2, 0]),
        raw_output=torch.randn(2, 3),
        prediction=torch.softmax(torch.randn(2, 3), dim=-1),
        classes=("a", "b", "c"),
    )

    assert identity.rows == (("mag", "paper", 19), ("mag", "paper", 4))
    assert payload.num_rows == 2


def test_heterogeneous_adapter_reuses_only_returned_seed_prefix() -> None:
    batch = HeteroData()
    batch["paper"].n_id = torch.tensor([19, 4, 91, 92])
    batch["paper"].batch_size = 2
    model_out = {
        "logits": torch.arange(12, dtype=torch.float32).reshape(4, 3),
        "labels": torch.tensor([2, 0, 1, 1]),
    }
    supervision = MagicMock(
        wraps=HeterogeneousNodeSupervisionAdapter("paper", "neighbor")
    )
    selected = supervision.select(model_out, batch, "Test")
    batch["paper"].batch_size = 4

    payload = PredictionRowAdapter(
        source_graph_id="mag",
        output_kind="heterogeneous",
        target_node_type="paper",
        sampling_strategy="heterogeneous-neighbor",
        class_vocabulary=("a", "b", "c"),
    ).adapt(batch, selected, phase="test")

    supervision.select.assert_called_once()
    assert payload.identity.rows == (("mag", "paper", 19), ("mag", "paper", 4))
    assert payload.num_rows == 2
    assert payload.columns["raw_output"] is selected.logits
    assert payload.columns["target"] is selected.targets


def test_same_external_id_across_types_and_sources_keeps_distinct_keys() -> (
    None
):
    identity = PredictionIdentity(
        columns={
            "source_graph_id": np.asarray(["mag-a", "mag-a", "mag-b"]),
            "target_node_type": np.asarray(["paper", "author", "paper"]),
            "n_id": torch.tensor([5, 5, 5]),
            "external_id": np.asarray(["shared", "shared", "shared"]),
        },
        key=("source_graph_id", "target_node_type", "n_id"),
    )

    assert identity.key == ("source_graph_id", "target_node_type", "n_id")
    assert "external_id" not in identity.key
    assert identity.columns["external_id"].tolist() == [
        "shared",
        "shared",
        "shared",
    ]
    assert identity.rows == (
        ("mag-a", "paper", 5),
        ("mag-a", "author", 5),
        ("mag-b", "paper", 5),
    )
    assert len(set(identity.rows)) == 3


def test_disk_homogeneous_adapter_restores_integer_external_ids(
    qualified_stores: dict[str, QualifiedStoreFixture],
) -> None:
    fixture = qualified_stores["homogeneous"]
    with TypedGraphStore.open(fixture.store_build.path) as store:
        source_graph_id = store.content_sha256
        adapter = PredictionRowAdapter(
            source_graph_id=source_graph_id,
            output_kind="homogeneous",
            target_node_type=store._manifest["target_node_type"],
            sampling_strategy="homogeneous-cluster",
            class_vocabulary=("negative", "positive"),
            store_path=store.path,
            store_state=store.state(),
        )
    batch = Data(global_nid=torch.tensor([2, 0, 1]))

    payload = adapter.adapt(
        batch,
        _classification_rows((1, 0)),
        phase="test",
    )

    assert payload.identity.key == ("source_graph_id", "global_nid")
    assert payload.identity.rows == (
        (source_graph_id, 0),
        (source_graph_id, 2),
    )
    external_ids = payload.identity.columns["external_id"]
    assert isinstance(external_ids, np.ndarray)
    assert external_ids.dtype.kind in {"i", "u"}
    assert external_ids.tolist() == [-5, 100]


def test_disk_heterogeneous_adapter_restores_string_external_ids(
    qualified_stores: dict[str, QualifiedStoreFixture],
) -> None:
    fixture = qualified_stores["heterogeneous"]
    with TypedGraphStore.open(fixture.store_build.path) as store:
        source_graph_id = store.content_sha256
        target_node_type = store._manifest["target_node_type"]
        adapter = PredictionRowAdapter(
            source_graph_id=source_graph_id,
            output_kind="heterogeneous",
            target_node_type=target_node_type,
            sampling_strategy="heterogeneous-neighbor",
            class_vocabulary=("negative", "positive"),
            store_path=store.path,
            store_state=store.state(),
        )
    batch = HeteroData()
    batch[target_node_type].n_id = torch.tensor([3, 1, 0])

    payload = adapter.adapt(
        batch,
        _classification_rows((2, 0)),
        phase="test",
    )

    assert payload.identity.key == (
        "source_graph_id",
        "target_node_type",
        "n_id",
    )
    assert payload.identity.rows == (
        (source_graph_id, target_node_type, 0),
        (source_graph_id, target_node_type, 3),
    )
    external_ids = payload.identity.columns["external_id"]
    assert isinstance(external_ids, np.ndarray)
    assert external_ids.dtype.kind in {"U", "S"}
    assert external_ids.tolist() == ["a", "d"]


@pytest.mark.parametrize(
    (
        "task",
        "target",
        "raw_output",
        "prediction",
        "classes",
        "units",
        "expected_shapes",
    ),
    [
        (
            "classification",
            torch.tensor([0, 1]),
            torch.tensor([[2.0, -1.0], [-0.5, 1.5]]),
            torch.tensor([[0.95, 0.05], [0.12, 0.88]]),
            ("no", "yes"),
            None,
            ((2,), (2, 2), (2, 2)),
        ),
        (
            "classification",
            torch.tensor([2, 0]),
            torch.tensor([[0.1, 0.2, 0.7], [0.8, 0.1, 0.1]]),
            torch.tensor([[0.1, 0.2, 0.7], [0.8, 0.1, 0.1]]),
            ("a", "b", "c"),
            None,
            ((2,), (2, 3), (2, 3)),
        ),
        (
            "regression",
            torch.tensor([[10.0], [20.0]]),
            torch.tensor([[0.0], [1.0]]),
            torch.tensor([[10.0], [20.0]]),
            (),
            "mg/L",
            ((2, 1), (2, 1), (2, 1)),
        ),
    ],
)
def test_payload_preserves_explicit_task_shapes_and_semantics(
    task, target, raw_output, prediction, classes, units, expected_shapes
) -> None:
    identity = PredictionIdentity(
        columns={"sample_id": torch.tensor([3, 8])}, key=("sample_id",)
    )
    payload = _payload(
        identity,
        target=target,
        raw_output=raw_output,
        prediction=prediction,
        task=task,
        classes=classes,
        units=units,
    )

    assert tuple(payload.columns["target"].shape) == expected_shapes[0]
    assert tuple(payload.columns["raw_output"].shape) == expected_shapes[1]
    assert tuple(payload.prediction.shape) == expected_shapes[2]
    assert payload.output_semantics["class_vocabulary"] == classes
    assert payload.output_semantics["units"] == units


def test_payload_retains_declared_optional_spaces_and_source_metadata() -> (
    None
):
    identity = PredictionIdentity(
        columns={"sample_id": torch.tensor([1, 2])}, key=("sample_id",)
    )
    payload = _payload(
        identity,
        target=torch.tensor([[5.0], [9.0]]),
        raw_output=torch.tensor([[0.0], [1.0]]),
        prediction=torch.tensor([[5.0], [9.0]]),
        task="regression",
        units="eV",
        extra_columns={
            "target_model_space": torch.tensor([[0.0], [1.0]]),
            "target_normalized": torch.tensor([[0.0], [1.0]]),
            "source": np.asarray(["lab-a", "lab-b"]),
        },
        extra_metadata={
            "target_model_space": {"role": "target_model_space"},
            "target_normalized": {"role": "target_normalized"},
            "source": {"role": "metadata"},
        },
    )

    assert payload.columns["source"].tolist() == ["lab-a", "lab-b"]
    assert torch.equal(
        payload.columns["target_model_space"], torch.tensor([[0.0], [1.0]])
    )


def test_identity_and_payload_are_immutable() -> None:
    identity = PredictionIdentity(
        columns={"sample_id": torch.tensor([1])}, key=("sample_id",)
    )
    payload = _payload(
        identity,
        target=torch.tensor([1]),
        raw_output=torch.tensor([[0.0, 1.0]]),
        prediction=torch.tensor([[0.0, 1.0]]),
        classes=("no", "yes"),
    )

    with pytest.raises(FrozenInstanceError):
        payload.prediction = torch.zeros(1, 2)
    with pytest.raises(TypeError):
        identity.columns["sample_id"] = torch.tensor([2])
    with pytest.raises(TypeError):
        payload.column_metadata["source"] = {"role": "metadata"}


def test_identity_owns_tensor_and_ndarray_snapshots() -> None:
    source_graph_ids = np.asarray(["graph-a", "graph-b"])
    global_nids = torch.tensor([41, 7])
    identity = PredictionIdentity(
        columns={
            "source_graph_id": source_graph_ids,
            "global_nid": global_nids,
        },
        key=("source_graph_id", "global_nid"),
    )

    source_graph_ids[0] = "mutated"
    global_nids[1] = 99

    for column in identity.columns.values():
        assert isinstance(column, np.ndarray)
        assert column.flags.owndata
        assert not column.flags.writeable
    assert identity.columns["source_graph_id"].tolist() == [
        "graph-a",
        "graph-b",
    ]
    assert identity.columns["global_nid"].tolist() == [41, 7]
    assert identity.rows == (("graph-a", 41), ("graph-b", 7))


@pytest.mark.parametrize(
    ("columns", "key", "message"),
    [
        ({"sample_id": torch.tensor([1, 1])}, ("sample_id",), "duplicate"),
        ({"sample_id": np.asarray(["ok", ""])}, ("sample_id",), "missing"),
        (
            {"sample_id": torch.tensor([1, 2]), "source": np.asarray(["a"])},
            ("sample_id",),
            "aligned",
        ),
        ({"sample_id": torch.tensor([1, 2])}, ("source_graph_id",), "key"),
    ],
)
def test_identity_rejects_missing_duplicate_or_misaligned_rows(
    columns, key, message
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        PredictionIdentity(columns=columns, key=key)


@pytest.mark.parametrize(
    ("target", "raw_output", "prediction", "message"),
    [
        (torch.tensor([0]), torch.randn(2, 2), torch.randn(2, 2), "aligned"),
        (
            torch.tensor([0, 1]),
            torch.randn(2, 2),
            torch.randn(1, 2),
            "aligned",
        ),
        (
            torch.tensor([[0], [1]]),
            torch.randn(2, 2),
            torch.randn(2, 2),
            "target",
        ),
        (
            torch.tensor([0, 1]),
            torch.randn(2),
            torch.randn(2, 2),
            "raw_output",
        ),
    ],
)
def test_payload_rejects_broadcast_or_misaligned_shapes(
    target, raw_output, prediction, message
) -> None:
    identity = PredictionIdentity(
        columns={"sample_id": torch.tensor([1, 2])}, key=("sample_id",)
    )
    with pytest.raises((TypeError, ValueError), match=message):
        _payload(
            identity,
            target=target,
            raw_output=raw_output,
            prediction=prediction,
            classes=("no", "yes"),
        )


def test_classification_payload_rejects_invalid_class_vocabularies() -> None:
    identity = PredictionIdentity(
        columns={"sample_id": torch.tensor([1, 2])}, key=("sample_id",)
    )
    invalid_vocabularies = (
        ((), 2),
        (("duplicate", "duplicate"), 2),
        (("negative", "positive"), 3),
    )

    for vocabulary, logits_width in invalid_vocabularies:
        logits = torch.zeros(2, logits_width)
        with pytest.raises(ValueError, match="class_vocabulary"):
            _payload(
                identity,
                target=torch.tensor([0, 1]),
                raw_output=logits,
                prediction=logits.clone(),
                classes=vocabulary,
            )


def test_payload_rejects_undeclared_metadata() -> None:
    identity = PredictionIdentity(
        columns={"sample_id": torch.tensor([1, 2])}, key=("sample_id",)
    )
    with pytest.raises(ValueError, match="undeclared"):
        _payload(
            identity,
            target=torch.tensor([0, 1]),
            raw_output=torch.randn(2, 2),
            prediction=torch.randn(2, 2),
            classes=("no", "yes"),
            extra_columns={"secret_batch_attribute": np.asarray(["x", "y"])},
        )


def test_evaluation_batch_carries_the_exact_aligned_payload() -> None:
    outputs = torch.tensor([[2.0, -1.0], [-0.5, 1.5]])
    targets = torch.tensor([0, 1])
    identity = PredictionIdentity(
        columns={"sample_id": torch.tensor([9, 3])}, key=("sample_id",)
    )
    payload = _payload(
        identity,
        target=targets,
        raw_output=outputs,
        prediction=torch.softmax(outputs, dim=-1),
        classes=("no", "yes"),
    )

    batch = EvaluationBatch(outputs, targets, 2, prediction_payload=payload)

    assert batch.prediction_payload is payload
    assert batch.prediction_payload.columns["raw_output"] is outputs
    assert batch.prediction_payload.columns["target"] is targets
