"""Authoritative selected-checkpoint prediction artifact contracts."""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import zipfile
from contextlib import suppress
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from lightning.pytorch.callbacks import Callback

import topobench.callbacks.prediction_artifacts as prediction_artifacts
from topobench.callbacks.prediction_artifacts import (
    SelectedCheckpointArtifactCallback,
    SplitPublication,
)
from topobench.evaluator import (
    EvaluationBatch,
    EvaluationContext,
    EvaluationResult,
    TBEvaluator,
)
from topobench.evaluator.prediction import (
    PredictionIdentity,
    PredictionPayload,
)

_METRICS_SCHEMA = "topobench-selected-checkpoint-metrics-v1"
_MANIFEST_SCHEMA = "topobench-selected-checkpoint-predictions-v1"
_COLUMN_ORDER = (
    "source_graph_id",
    "target_node_type",
    "external_id",
    "split_ordinal",
    "target",
    "raw_output",
    "prediction",
    "source",
)
_IDENTITY_KEY = ("source_graph_id", "target_node_type", "external_id")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fingerprint(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _checkpoint(tmp_path: Path) -> tuple[Path, str]:
    path = tmp_path / "checkpoints" / "selected.ckpt"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"deterministic selected checkpoint\n")
    return path, _sha256(path)


def _context(
    split: str, checkpoint_sha256: str, *, expected_rows: int = 5
) -> EvaluationContext:
    return EvaluationContext(
        split=split,  # type: ignore[arg-type]
        pass_kind="selected_checkpoint",
        policy="exact",
        task="classification",
        num_classes=2,
        expected_num_examples=expected_rows,
        vocabulary_id="binary-v1",
        model_id="toy-model-v1",
        checkpoint_id=checkpoint_sha256,
    )


def _provenance(split: str, num_examples: int) -> dict[str, str | int]:
    return {
        "source_fingerprint": _fingerprint("source:toy"),
        "dataset_fingerprint": _fingerprint("dataset:toy-v1"),
        "split_fingerprint": _fingerprint(f"split:{split}:v1"),
        "model_fingerprint": _fingerprint("model:toy-model-v1"),
        "transform_fingerprint": _fingerprint("transform:none-v1"),
        "num_examples": num_examples,
    }


def _result(
    context: EvaluationContext,
    *,
    num_examples: int | None = None,
) -> EvaluationResult:
    count = (
        context.expected_num_examples if num_examples is None else num_examples
    )
    assert count is not None
    return EvaluationResult(
        metrics={"loss": 0.25, "accuracy": 0.6},
        num_examples=count,
        context=context,
        status={"loss": "exact", "accuracy": "exact"},
        support={"loss": count, "accuracy": count},
        reason={"loss": None, "accuracy": None},
        provenance=_provenance(context.split, count),
    )


def _full_columns(split: str) -> dict[str, np.ndarray | torch.Tensor]:
    split_delta = 0.0 if split == "val" else 0.05
    raw_output = torch.tensor(
        [
            [2.0, -1.0],
            [0.5, 0.25],
            [-0.5, 1.5],
            [1.25, 0.0],
            [-1.0, 2.5],
        ],
        dtype=torch.float32,
    )
    prediction = torch.tensor(
        [
            [0.95, 0.05],
            [0.60, 0.40],
            [0.10, 0.90],
            [0.80, 0.20],
            [0.03, 0.97],
        ],
        dtype=torch.float32,
    )
    if split_delta:
        raw_output = raw_output + split_delta
        prediction = prediction + torch.tensor(
            [[-split_delta, split_delta]], dtype=torch.float32
        )
    return {
        "source_graph_id": np.asarray([7, 7, 7, 8, 8], dtype=np.int64),
        "target_node_type": np.asarray(
            ["drug", "gene", "drug", "gene", "drug"], dtype="<U4"
        ),
        # The same external ID is valid across node types because the full key differs.
        "external_id": np.asarray(
            ["node-a", "node-a", "node-c", "node-d", "node-e"], dtype="<U6"
        ),
        "split_ordinal": np.arange(5, dtype=np.int64),
        "target": torch.tensor([0, 1, 1, 0, 1], dtype=torch.int64),
        "raw_output": raw_output,
        "prediction": prediction,
        "source": np.asarray(
            ["alpha", "alpha", "beta", "beta", "alpha"], dtype="<U5"
        ),
    }


def _batches(
    split: str,
    context: EvaluationContext,
    *,
    row_count: int = 5,
    duplicate_identity: bool = False,
    prediction_offset: float = 0.0,
    non_finite: str | None = None,
    source_width: int | None = None,
) -> tuple[EvaluationBatch, ...]:
    columns = _full_columns(split)
    if duplicate_identity:
        target_node_type = np.asarray(columns["target_node_type"]).copy()
        target_node_type[1] = target_node_type[0]
        columns["target_node_type"] = target_node_type
    if prediction_offset:
        columns["prediction"] = torch.as_tensor(columns["prediction"]).clone()
        columns["prediction"][0, 0] += prediction_offset
    if non_finite == "raw_output":
        columns["raw_output"] = torch.as_tensor(columns["raw_output"]).clone()
        columns["raw_output"][2, 0] = float("nan")
    if non_finite == "prediction":
        columns["prediction"] = torch.as_tensor(columns["prediction"]).clone()
        columns["prediction"][2, 0] = float("inf")
    if source_width is not None:
        columns["source"] = np.asarray(
            ["x" * source_width] * 5, dtype=f"<U{source_width}"
        )

    batches: list[EvaluationBatch] = []
    for start in range(0, row_count, 3):
        stop = min(start + 3, row_count)
        outputs = torch.as_tensor(columns["raw_output"])[start:stop]
        targets = torch.as_tensor(columns["target"])[start:stop]
        identity = PredictionIdentity(
            columns={
                name: np.asarray(columns[name])[start:stop]
                for name in (
                    "source_graph_id",
                    "target_node_type",
                    "external_id",
                    "split_ordinal",
                )
            },
            key=_IDENTITY_KEY,
        )
        payload = PredictionPayload(
            identity=identity,
            prediction=torch.as_tensor(columns["prediction"])[start:stop],
            columns={
                # These are the exact tensors already selected by supervision.
                "target": targets,
                "raw_output": outputs,
                "source": np.asarray(columns["source"])[start:stop],
            },
            column_metadata={
                "target": {
                    "role": "target",
                    "class_vocabulary": ("negative", "positive"),
                },
                "raw_output": {
                    "role": "raw_output",
                    "class_vocabulary": ("negative", "positive"),
                },
                "prediction": {
                    "role": "prediction",
                    "class_vocabulary": ("negative", "positive"),
                },
                "source": {
                    "role": "metadata",
                    "vocabulary": ("alpha", "beta"),
                },
            },
            output_semantics={
                "task": "classification",
                "class_vocabulary": ("negative", "positive"),
                "units": None,
            },
        )
        batches.append(
            EvaluationBatch(
                outputs=outputs,
                targets=targets,
                num_examples=stop - start,
                context=context,
                sequence_id=(split, start),
                prediction_payload=payload,
            )
        )
    return tuple(batches)


def _callback(
    tmp_path: Path,
    *,
    shard_rows: int = 2,
    shard_bytes: int = 1_000_000,
    evaluation_slices: dict[str, object] | None = None,
) -> SelectedCheckpointArtifactCallback:
    return SelectedCheckpointArtifactCallback(
        run_root=tmp_path,
        root=tmp_path / "evaluations" / "best_checkpoint",
        shard_rows=shard_rows,
        shard_bytes=shard_bytes,
        metadata_fields=("source",),
        evaluation_slices=evaluation_slices or {},
        distributed_policy="reject",
        existing_artifact_policy="verify_identical",
    )


def _hold_capture_until_released(
    run_root: Path,
    checkpoint: Path,
    checkpoint_sha256: str,
    ready: Any,
    release: Any,
) -> None:
    callback = _callback(run_root)
    context = _context("val", checkpoint_sha256)
    try:
        callback.begin(
            context,
            checkpoint_path=checkpoint,
            checkpoint_sha256=checkpoint_sha256,
            checkpoint_epoch=3,
            checkpoint_global_step=17,
            world_size=1,
            global_rank=0,
        )
    except BaseException as error:
        ready.send(("error", repr(error)))
        ready.close()
        return
    ready.send(("ready", None))
    ready.close()
    release.recv()
    release.close()
    callback.abort()


def _write_split(
    callback: SelectedCheckpointArtifactCallback,
    checkpoint: Path,
    checkpoint_sha256: str,
    split: str,
    *,
    row_count: int = 5,
    result_count: int | None = None,
    duplicate_identity: bool = False,
    prediction_offset: float = 0.0,
    non_finite: str | None = None,
    source_width: int | None = None,
    expected_rows: int = 5,
) -> SplitPublication:
    context = _context(split, checkpoint_sha256, expected_rows=expected_rows)
    callback.begin(
        context,
        checkpoint_path=checkpoint,
        checkpoint_sha256=checkpoint_sha256,
        checkpoint_epoch=3,
        checkpoint_global_step=17,
        world_size=1,
        global_rank=0,
    )
    for batch in _batches(
        split,
        context,
        row_count=row_count,
        duplicate_identity=duplicate_identity,
        prediction_offset=prediction_offset,
        non_finite=non_finite,
        source_width=source_width,
    ):
        callback.update(batch)
    return callback.finalize(_result(context, num_examples=result_count))


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _publication_files(publication: SplitPublication) -> tuple[Path, ...]:
    return (
        publication.metrics_file.path,
        publication.manifest_file.path,
        *(artifact.path for artifact in publication.shard_files),
    )


def _completed_shard_byte_sizes(
    root: Path, *, row_count: int, shard_rows: int
) -> tuple[int, ...]:
    checkpoint, checkpoint_sha256 = _checkpoint(root)
    publication = _write_split(
        _callback(root, shard_rows=shard_rows),
        checkpoint,
        checkpoint_sha256,
        "val",
        row_count=row_count,
        result_count=row_count,
        expected_rows=row_count,
    )
    return tuple(artifact.byte_size for artifact in publication.shard_files)


def test_source_slices_use_bounded_declared_vocabulary_and_canonical_rows(
    tmp_path: Path,
) -> None:
    checkpoint, checkpoint_sha256 = _checkpoint(tmp_path)
    callback = _callback(
        tmp_path,
        evaluation_slices={"source": {"max_categories": 2, "min_rows": 1}},
    )
    callback.configure_slice_evaluator_factory(
        lambda: TBEvaluator(
            "classification",
            num_classes=2,
            metrics=("accuracy",),
            policy={"train": "online", "val": "exact", "test": "exact"},
        )
    )

    publication = _write_split(
        callback,
        checkpoint,
        checkpoint_sha256,
        "val",
    )

    slices = _json(publication.metrics_file.path)["slices"]["source"]
    assert slices["alpha"]["num_examples"] == 3
    assert slices["alpha"]["metrics"]["accuracy"] == pytest.approx(2 / 3)
    assert slices["beta"]["num_examples"] == 2
    assert slices["beta"]["metrics"]["accuracy"] == pytest.approx(1.0)


def test_source_slices_reject_unbounded_or_undeclared_categories(
    tmp_path: Path,
) -> None:
    checkpoint, checkpoint_sha256 = _checkpoint(tmp_path)
    callback = _callback(
        tmp_path,
        evaluation_slices={"source": {"max_categories": 1, "min_rows": 1}},
    )
    callback.configure_slice_evaluator_factory(
        lambda: TBEvaluator(
            "classification",
            num_classes=2,
            metrics=("accuracy",),
        )
    )

    with pytest.raises(ValueError, match="max_categories"):
        _write_split(callback, checkpoint, checkpoint_sha256, "val")


def test_callback_writes_exact_val_and_test_layout_with_versioned_multi_shard_payloads(
    tmp_path: Path,
) -> None:
    checkpoint, checkpoint_sha256 = _checkpoint(tmp_path)
    callback = _callback(tmp_path)
    assert isinstance(callback, Callback)

    val_publication = _write_split(
        callback, checkpoint, checkpoint_sha256, "val"
    )
    test_publication = _write_split(
        callback, checkpoint, checkpoint_sha256, "test"
    )

    artifact_root = tmp_path / "evaluations" / "best_checkpoint"
    expected_files = {
        Path(split) / relative
        for split in ("val", "test")
        for relative in (
            "metrics.json",
            "predictions/manifest.json",
            "predictions/part-00000.npz",
            "predictions/part-00001.npz",
            "predictions/part-00002.npz",
        )
    }
    observed_files = {
        path.relative_to(artifact_root)
        for path in artifact_root.rglob("*")
        if path.is_file()
    }
    assert observed_files == expected_files
    assert val_publication != test_publication
    assert callback.publications == {
        "val": val_publication,
        "test": test_publication,
    }
    with pytest.raises(TypeError):
        callback.publications["val"] = test_publication  # type: ignore[index]
    with pytest.raises((FrozenInstanceError, AttributeError)):
        val_publication.num_examples = 0  # type: ignore[misc]

    for split, publication in (
        ("val", val_publication),
        ("test", test_publication),
    ):
        metrics = _json(publication.metrics_file.path)
        manifest = _json(publication.manifest_file.path)
        assert metrics["schema_version"] == _METRICS_SCHEMA
        assert manifest["schema_version"] == _MANIFEST_SCHEMA
        assert (
            metrics["split"] == manifest["split"] == publication.split == split
        )
        assert (
            metrics["checkpoint"]
            == manifest["checkpoint"]
            == {
                "path": "checkpoints/selected.ckpt",
                "sha256": checkpoint_sha256,
                "epoch": 3,
                "global_step": 17,
            }
        )
        assert type(metrics["num_examples"]) is int
        assert (
            metrics["num_examples"]
            == manifest["expected_rows"]
            == manifest["observed_rows"]
            == metrics["provenance"]["num_examples"]
            == publication.num_examples
            == 5
        )
        assert list(metrics["metrics"]) == ["loss", "accuracy"]
        assert metrics["metrics"] == {"loss": 0.25, "accuracy": 0.6}
        assert metrics["metric_metadata"] == {
            "loss": {"status": "exact", "support": 5, "reason": None},
            "accuracy": {"status": "exact", "support": 5, "reason": None},
        }
        expected_provenance = _provenance(split, 5)
        assert metrics["provenance"] == expected_provenance
        assert manifest["provenance"] == expected_provenance
        assert manifest["output_semantics"] == {
            "task": "classification",
            "class_vocabulary": ["negative", "positive"],
            "units": None,
        }
        assert manifest["identity"] == {
            "key": list(_IDENTITY_KEY),
            "unique": True,
            "order": "split_ordinal",
        }
        assert [column["name"] for column in manifest["columns"]] == list(
            _COLUMN_ORDER
        )
        column_schema = {
            column["name"]: column for column in manifest["columns"]
        }
        assert (
            column_schema["source_graph_id"]["dtype"] == np.dtype(np.int64).str
        )
        assert column_schema["source_graph_id"]["shape_tail"] == []
        assert (
            column_schema["target_node_type"]["dtype"] == np.dtype("<U4").str
        )
        assert column_schema["external_id"]["dtype"] == np.dtype("<U6").str
        assert column_schema["target"]["dtype"] == np.dtype(np.int64).str
        assert column_schema["target"]["shape_tail"] == []
        assert column_schema["raw_output"]["dtype"] == np.dtype(np.float32).str
        assert column_schema["raw_output"]["shape_tail"] == [2]
        assert column_schema["prediction"]["dtype"] == np.dtype(np.float32).str
        assert column_schema["prediction"]["shape_tail"] == [2]
        assert column_schema["source"]["dtype"] == np.dtype("<U5").str

        assert (
            publication.checkpoint_sha256
            == checkpoint_sha256
            == _sha256(checkpoint)
        )
        assert publication.metrics_file.sha256 == _sha256(
            publication.metrics_file.path
        )
        assert publication.manifest_file.sha256 == _sha256(
            publication.manifest_file.path
        )
        assert (
            publication.metrics_file.registration_name
            == f"best-checkpoint-{split}-metrics"
        )
        assert (
            publication.manifest_file.registration_name
            == f"best-checkpoint-{split}-predictions-manifest"
        )
        assert len(publication.shard_files) == 3
        assert [
            artifact.registration_name for artifact in publication.shard_files
        ] == [
            f"best-checkpoint-{split}-predictions-part-{index:05d}"
            for index in range(3)
        ]
        assert [shard["path"] for shard in manifest["shards"]] == [
            f"part-{index:05d}.npz" for index in range(3)
        ]
        assert [
            (shard["row_start"], shard["row_stop"])
            for shard in manifest["shards"]
        ] == [
            (0, 2),
            (2, 4),
            (4, 5),
        ]

        loaded: dict[str, list[np.ndarray]] = {
            name: [] for name in _COLUMN_ORDER
        }
        for descriptor, shard_record in zip(
            publication.shard_files, manifest["shards"], strict=True
        ):
            assert (
                descriptor.path.parent == artifact_root / split / "predictions"
            )
            assert (
                descriptor.sha256
                == shard_record["sha256"]
                == _sha256(descriptor.path)
            )
            assert (
                descriptor.byte_size
                == shard_record["byte_size"]
                == descriptor.path.stat().st_size
            )
            with zipfile.ZipFile(descriptor.path) as archive:
                assert all(
                    item.compress_type == zipfile.ZIP_STORED
                    for item in archive.infolist()
                )
            with np.load(descriptor.path, allow_pickle=False) as shard:
                assert shard.files == list(_COLUMN_ORDER)
                row_counts = {shard[name].shape[0] for name in shard.files}
                assert len(row_counts) == 1
                for name in _COLUMN_ORDER:
                    assert shard[name].dtype != np.dtype(object)
                    loaded[name].append(shard[name].copy())

        combined = {
            name: np.concatenate(parts, axis=0)
            for name, parts in loaded.items()
        }
        expected = _full_columns(split)
        for name in _COLUMN_ORDER:
            np.testing.assert_array_equal(
                combined[name], np.asarray(expected[name])
            )
        identities = list(
            zip(
                combined["source_graph_id"].tolist(),
                combined["target_node_type"].tolist(),
                combined["external_id"].tolist(),
                strict=True,
            )
        )
        assert len(identities) == len(set(identities)) == 5
        np.testing.assert_array_equal(combined["split_ordinal"], np.arange(5))


def test_finalize_syncs_nested_directories_before_releasing_capture_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, Path | None, frozenset[Path]]] = []
    original_sync_directory = prediction_artifacts._sync_directory
    original_release_lock_descriptor = (
        prediction_artifacts._release_lock_descriptor
    )

    def observe_sync_directory(path: Path) -> None:
        resolved = Path(path).resolve()
        files = frozenset(
            entry.relative_to(resolved)
            for entry in resolved.rglob("*")
            if entry.is_file()
        )
        events.append(("sync", resolved, files))
        original_sync_directory(path)

    def observe_lock_release(descriptor: int) -> None:
        events.append(("release", None, frozenset()))
        original_release_lock_descriptor(descriptor)

    monkeypatch.setattr(
        prediction_artifacts, "_sync_directory", observe_sync_directory
    )
    monkeypatch.setattr(
        prediction_artifacts, "_release_lock_descriptor", observe_lock_release
    )
    checkpoint, checkpoint_sha256 = _checkpoint(tmp_path)
    callback = _callback(tmp_path)
    context = _context("val", checkpoint_sha256)
    callback.begin(
        context,
        checkpoint_path=checkpoint,
        checkpoint_sha256=checkpoint_sha256,
        checkpoint_epoch=3,
        checkpoint_global_step=17,
        world_size=1,
        global_rank=0,
    )
    staging_root = next(callback.root.glob(".val.staging-*")).resolve()
    prediction_root = staging_root / "predictions"
    for batch in _batches("val", context):
        callback.update(batch)

    publication = callback.finalize(_result(context))

    final_root = publication.metrics_file.path.parent.resolve()
    expected_staging_files = frozenset(
        path.resolve().relative_to(final_root)
        for path in _publication_files(publication)
    )
    expected_prediction_files = frozenset(
        path.resolve().relative_to(final_root / "predictions")
        for path in (
            publication.manifest_file.path,
            *[shard.path for shard in publication.shard_files],
        )
    )
    release_indices = [
        index
        for index, (event, _, _) in enumerate(events)
        if event == "release"
    ]
    assert len(release_indices) == 1
    release_index = release_indices[0]
    required_syncs = (
        (prediction_root, expected_prediction_files),
        (staging_root, expected_staging_files),
        (
            callback.root.resolve(),
            frozenset(
                final_root.relative_to(callback.root.resolve()) / path
                for path in expected_staging_files
            ),
        ),
    )
    for required_path, required_files in required_syncs:
        sync_observations = [
            files
            for index, (event, path, files) in enumerate(events)
            if event == "sync"
            and path == required_path
            and index < release_index
        ]
        assert any(
            required_files <= observed_files
            for observed_files in sync_observations
        ), f"{required_path} was not durably synced before lock release"


def test_abort_cleans_staging_without_exposing_partial_split(
    tmp_path: Path,
) -> None:
    checkpoint, checkpoint_sha256 = _checkpoint(tmp_path)
    callback = _callback(tmp_path)
    context = _context("val", checkpoint_sha256)
    callback.begin(
        context,
        checkpoint_path=checkpoint,
        checkpoint_sha256=checkpoint_sha256,
        checkpoint_epoch=3,
        checkpoint_global_step=17,
        world_size=1,
        global_rank=0,
    )
    for batch in _batches("val", context):
        callback.update(batch)

    final_root = tmp_path / "evaluations" / "best_checkpoint" / "val"
    assert not final_root.exists()
    assert any((tmp_path / "evaluations").rglob("*.npz"))

    callback.abort()

    assert not final_root.exists()
    remaining_files = {
        path.relative_to(tmp_path / "evaluations")
        for path in (tmp_path / "evaluations").rglob("*")
        if path.is_file()
    }
    assert remaining_files <= {Path("best_checkpoint") / ".val.capture.lock"}
    assert callback.publications == {}


def test_live_capture_is_rejected_and_dead_process_lock_is_reclaimable(
    tmp_path: Path,
) -> None:
    checkpoint, checkpoint_sha256 = _checkpoint(tmp_path)
    context = _context("val", checkpoint_sha256)
    process_context = mp.get_context("spawn")
    ready_reader, ready_writer = process_context.Pipe(duplex=False)
    release_reader, release_writer = process_context.Pipe(duplex=False)
    holder = process_context.Process(
        target=_hold_capture_until_released,
        args=(
            tmp_path,
            checkpoint,
            checkpoint_sha256,
            ready_writer,
            release_reader,
        ),
    )
    holder.start()
    ready_writer.close()
    release_reader.close()

    try:
        assert ready_reader.poll(30), "capture holder did not become ready"
        status, detail = ready_reader.recv()
        assert status == "ready", detail

        contender = _callback(tmp_path)
        with pytest.raises(RuntimeError, match="lock|concurrent|capture"):
            contender.begin(
                context,
                checkpoint_path=checkpoint,
                checkpoint_sha256=checkpoint_sha256,
                checkpoint_epoch=3,
                checkpoint_global_step=17,
                world_size=1,
                global_rank=0,
            )
        assert contender.publications == {}

        holder.terminate()
        holder.join(timeout=30)
        assert not holder.is_alive()
        lock_path = tmp_path / ".best-checkpoint-val.capture.lock"
        assert lock_path.exists()

        recovered = _callback(tmp_path)
        recovered.begin(
            context,
            checkpoint_path=checkpoint,
            checkpoint_sha256=checkpoint_sha256,
            checkpoint_epoch=3,
            checkpoint_global_step=17,
            world_size=1,
            global_rank=0,
        )
        recovered.abort()
    finally:
        if holder.is_alive():
            with suppress(BrokenPipeError, EOFError, OSError):
                release_writer.send(None)
            holder.join(timeout=10)
        if holder.is_alive():
            holder.kill()
            holder.join(timeout=10)
        ready_reader.close()
        release_writer.close()
        holder.close()


def test_identical_rerun_is_idempotent_and_conflicting_rerun_is_rejected(
    tmp_path: Path,
) -> None:
    checkpoint, checkpoint_sha256 = _checkpoint(tmp_path)
    first = _write_split(
        _callback(tmp_path), checkpoint, checkpoint_sha256, "val"
    )
    before = {path: path.read_bytes() for path in _publication_files(first)}

    identical = _write_split(
        _callback(tmp_path), checkpoint, checkpoint_sha256, "val"
    )

    assert identical == first
    assert {path: path.read_bytes() for path in before} == before

    artifact_root = tmp_path / "evaluations" / "best_checkpoint"
    entries_before_conflict = set(artifact_root.iterdir())
    conflicting = _callback(tmp_path)
    with pytest.raises(RuntimeError, match="conflict|collision|immutable"):
        _write_split(
            conflicting,
            checkpoint,
            checkpoint_sha256,
            "val",
            prediction_offset=0.125,
        )
    candidates = [
        path
        for path in artifact_root.iterdir()
        if path not in entries_before_conflict
        and (path / "predictions" / "manifest.json").is_file()
    ]
    assert len(candidates) == 1
    candidate = candidates[0]
    candidate_snapshot = {
        path.relative_to(candidate): path.read_bytes()
        for path in candidate.rglob("*")
        if path.is_file()
    }

    conflicting.abort()

    preserved_candidates = [
        path
        for path in artifact_root.iterdir()
        if path != artifact_root / "val"
        and (path / "predictions" / "manifest.json").is_file()
    ]
    assert len(preserved_candidates) == 1
    preserved_candidate = preserved_candidates[0]
    assert {
        path.relative_to(preserved_candidate): path.read_bytes()
        for path in preserved_candidate.rglob("*")
        if path.is_file()
    } == candidate_snapshot
    with np.load(
        preserved_candidate / "predictions" / "part-00000.npz",
        allow_pickle=False,
    ) as shard:
        expected = float(
            torch.as_tensor(_full_columns("val")["prediction"])[0, 0]
        )
        assert float(shard["prediction"][0, 0]) == pytest.approx(
            expected + 0.125
        )
    assert {path: path.read_bytes() for path in before} == before


def test_finalize_rejects_evaluator_prediction_count_mismatch(
    tmp_path: Path,
) -> None:
    checkpoint, checkpoint_sha256 = _checkpoint(tmp_path)
    callback = _callback(tmp_path)
    with pytest.raises(ValueError, match="num_examples|count|observed"):
        _write_split(
            callback,
            checkpoint,
            checkpoint_sha256,
            "val",
            result_count=4,
        )
    callback.abort()
    assert not (tmp_path / "evaluations" / "best_checkpoint" / "val").exists()


def test_finalize_rejects_duplicate_identities(tmp_path: Path) -> None:
    checkpoint, checkpoint_sha256 = _checkpoint(tmp_path)
    callback = _callback(tmp_path)
    with pytest.raises(ValueError, match="duplicate|unique|identity"):
        _write_split(
            callback,
            checkpoint,
            checkpoint_sha256,
            "val",
            duplicate_identity=True,
        )
    callback.abort()
    assert not (tmp_path / "evaluations" / "best_checkpoint" / "val").exists()


def test_finalize_rejects_missing_identities(tmp_path: Path) -> None:
    checkpoint, checkpoint_sha256 = _checkpoint(tmp_path)
    callback = _callback(tmp_path)
    with pytest.raises(ValueError, match="missing|expected|identity|count"):
        _write_split(
            callback,
            checkpoint,
            checkpoint_sha256,
            "val",
            row_count=4,
            result_count=4,
        )
    callback.abort()
    assert not (tmp_path / "evaluations" / "best_checkpoint" / "val").exists()


def test_completed_npz_shards_respect_encoded_byte_bound_by_splitting(
    tmp_path: Path,
) -> None:
    one_row_sizes = _completed_shard_byte_sizes(
        tmp_path / "one-row-shards", row_count=2, shard_rows=1
    )
    (two_row_size,) = _completed_shard_byte_sizes(
        tmp_path / "two-row-shard", row_count=2, shard_rows=2
    )
    shard_bytes = max(one_row_sizes)
    assert two_row_size > shard_bytes

    bounded_root = tmp_path / "bounded"
    checkpoint, checkpoint_sha256 = _checkpoint(bounded_root)
    publication = _write_split(
        _callback(bounded_root, shard_rows=2, shard_bytes=shard_bytes),
        checkpoint,
        checkpoint_sha256,
        "val",
        row_count=2,
        result_count=2,
        expected_rows=2,
    )

    assert len(publication.shard_files) == 2
    assert all(
        artifact.byte_size == artifact.path.stat().st_size <= shard_bytes
        for artifact in publication.shard_files
    )


def test_single_encoded_row_larger_than_byte_bound_is_rejected(
    tmp_path: Path,
) -> None:
    (encoded_row_size,) = _completed_shard_byte_sizes(
        tmp_path / "calibration", row_count=1, shard_rows=1
    )
    bounded_root = tmp_path / "bounded"
    checkpoint, checkpoint_sha256 = _checkpoint(bounded_root)
    callback = _callback(
        bounded_root,
        shard_rows=2,
        shard_bytes=encoded_row_size - 1,
    )

    with pytest.raises(
        ValueError, match="single row|encoded|shard_bytes|byte bound"
    ):
        _write_split(
            callback,
            checkpoint,
            checkpoint_sha256,
            "val",
            row_count=1,
            result_count=1,
            expected_rows=1,
        )

    callback.abort()
    assert not (
        bounded_root / "evaluations" / "best_checkpoint" / "val"
    ).exists()


@pytest.mark.parametrize("column", ["raw_output", "prediction"])
def test_non_finite_required_values_are_rejected(
    tmp_path: Path, column: str
) -> None:
    checkpoint, checkpoint_sha256 = _checkpoint(tmp_path)
    callback = _callback(tmp_path)
    with pytest.raises(ValueError, match=f"non-finite|finite|{column}"):
        _write_split(
            callback,
            checkpoint,
            checkpoint_sha256,
            "val",
            non_finite=column,
        )
    callback.abort()
    assert not (tmp_path / "evaluations" / "best_checkpoint" / "val").exists()


def test_multi_rank_capture_is_rejected_before_staging(tmp_path: Path) -> None:
    checkpoint, checkpoint_sha256 = _checkpoint(tmp_path)
    callback = _callback(tmp_path)
    context = _context("val", checkpoint_sha256)

    with pytest.raises(
        RuntimeError, match="world_size|multi-rank|distributed"
    ):
        callback.begin(
            context,
            checkpoint_path=checkpoint,
            checkpoint_sha256=checkpoint_sha256,
            checkpoint_epoch=3,
            checkpoint_global_step=17,
            world_size=2,
            global_rank=0,
        )

    assert callback.publications == {}
    assert not (tmp_path / "evaluations").exists()
