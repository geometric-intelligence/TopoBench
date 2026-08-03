"""Artifact logger publication contracts for selected-checkpoint files."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.loggers.wandb import WandbLogger

import topobench.utils.artifact_logging as artifact_logging
from topobench.callbacks.prediction_artifacts import (
    ArtifactFile,
    SplitPublication,
)
from topobench.utils.artifact_logging import ArtifactLoggerAdapter

_INDEX_SCHEMA = "topobench-artifact-index-v1"
_CHECKPOINT_SHA256 = hashlib.sha256(b"selected checkpoint").hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact(path: Path, registration_name: str) -> ArtifactFile:
    return ArtifactFile(
        path=path,
        sha256=_sha256(path),
        byte_size=path.stat().st_size,
        registration_name=registration_name,
    )


def _publication(
    tmp_path: Path,
    split: str,
    *,
    shard_count: int = 2,
    source_slices: bool = False,
) -> SplitPublication:
    split_root = tmp_path / "evaluations" / "best_checkpoint" / split
    prediction_root = split_root / "predictions"
    prediction_root.mkdir(parents=True)
    metrics_path = split_root / "metrics.json"
    metrics_document = {
        "schema_version": "topobench-selected-checkpoint-metrics-v1",
        "split": split,
        "checkpoint": {
            "path": "checkpoints/selected.ckpt",
            "sha256": _CHECKPOINT_SHA256,
            "epoch": 3,
            "global_step": 17,
        },
        "num_examples": 5,
        "metrics": {"loss": 0.25, "accuracy": 0.6},
        "metric_metadata": {
            "loss": {"status": "exact", "support": 5, "reason": None},
            "accuracy": {
                "status": "exact",
                "support": 5,
                "reason": None,
            },
        },
        "provenance": {"num_examples": 5},
    }
    if source_slices:
        metrics_document["slices"] = {
            "source": {
                "alpha": {
                    "num_examples": 3,
                    "metrics": {"accuracy": 2 / 3},
                    "metric_metadata": {
                        "accuracy": {
                            "status": "exact",
                            "support": 3,
                            "reason": None,
                        }
                    },
                }
            }
        }
    metrics_path.write_text(
        json.dumps(metrics_document, separators=(",", ":")),
        encoding="utf-8",
    )
    manifest_path = prediction_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "topobench-selected-checkpoint-predictions-v1",
                "split": split,
                "checkpoint": {
                    "path": "checkpoints/selected.ckpt",
                    "sha256": _CHECKPOINT_SHA256,
                    "epoch": 3,
                    "global_step": 17,
                },
                "expected_rows": 5,
                "observed_rows": 5,
            },
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    shard_paths = []
    for index in range(shard_count):
        path = prediction_root / f"part-{index:05d}.npz"
        path.write_bytes(f"deterministic shard {split} {index}\n".encode())
        shard_paths.append(path)
    return SplitPublication(
        split=split,
        metrics_file=_artifact(
            metrics_path,
            f"best-checkpoint-{split}-metrics",
        ),
        manifest_file=_artifact(
            manifest_path,
            f"best-checkpoint-{split}-predictions-manifest",
        ),
        shard_files=tuple(
            _artifact(
                path,
                f"best-checkpoint-{split}-predictions-part-{index:05d}",
            )
            for index, path in enumerate(shard_paths)
        ),
        num_examples=5,
        checkpoint_sha256=_CHECKPOINT_SHA256,
    )


def _files(publication: SplitPublication) -> tuple[ArtifactFile, ...]:
    return (
        publication.metrics_file,
        publication.manifest_file,
        *publication.shard_files,
    )


def _with_changed_metric(
    publication: SplitPublication,
) -> SplitPublication:
    metrics_path = publication.metrics_file.path
    document = json.loads(metrics_path.read_text(encoding="utf-8"))
    document["metrics"]["loss"] = 0.5
    metrics_path.write_text(
        json.dumps(document, separators=(",", ":")),
        encoding="utf-8",
    )
    return replace(
        publication,
        metrics_file=_artifact(
            metrics_path,
            publication.metrics_file.registration_name,
        ),
    )


def test_csv_logger_appends_one_uri_and_digest_index_record_per_file(
    tmp_path: Path,
) -> None:
    logger = CSVLogger(
        save_dir=tmp_path / "csv", name="qualified", version="run-0"
    )
    logger.log_metrics = MagicMock()
    adapter = ArtifactLoggerAdapter(loggers=(logger,), run_root=tmp_path)
    val_publication = _publication(tmp_path, "val")
    test_publication = _publication(tmp_path, "test")

    adapter.register(val_publication)
    index_path = Path(logger.log_dir) / "artifact-index.jsonl"
    val_bytes = index_path.read_bytes()
    adapter.register(test_publication)

    complete_bytes = index_path.read_bytes()
    assert complete_bytes.startswith(val_bytes)
    records = [json.loads(line) for line in complete_bytes.splitlines()]
    artifacts = (*_files(val_publication), *_files(test_publication))
    assert len(records) == len(artifacts) == 8
    assert [record["name"] for record in records] == [
        artifact.registration_name for artifact in artifacts
    ]
    for record, artifact in zip(records, artifacts, strict=True):
        assert set(record) == {
            "schema_version",
            "name",
            "split",
            "uri",
            "sha256",
            "byte_size",
            "checkpoint_sha256",
        }
        assert record["schema_version"] == _INDEX_SCHEMA
        assert record["split"] in {"val", "test"}
        assert record["uri"] == artifact.path.resolve().as_uri()
        assert record["sha256"] == artifact.sha256 == _sha256(artifact.path)
        assert record["byte_size"] == artifact.byte_size
        assert record["checkpoint_sha256"] == _CHECKPOINT_SHA256

    assert logger.log_metrics.call_count == 2
    for call, split in zip(
        logger.log_metrics.call_args_list, ("val", "test"), strict=True
    ):
        assert call.args == (
            {
                f"evaluations/best_checkpoint/{split}/loss": 0.25,
                f"evaluations/best_checkpoint/{split}/accuracy": 0.6,
                f"evaluations/best_checkpoint/{split}/num_examples": 5,
            },
        )
        assert call.kwargs == {"step": 17}
        count = call.args[0][
            f"evaluations/best_checkpoint/{split}/num_examples"
        ]
        assert type(count) is int


def test_logger_publishes_stable_source_slice_metric_keys(
    tmp_path: Path,
) -> None:
    logger = CSVLogger(
        save_dir=tmp_path / "csv", name="qualified", version="run-0"
    )
    logger.log_metrics = MagicMock()
    publication = _publication(
        tmp_path,
        "val",
        shard_count=1,
        source_slices=True,
    )

    ArtifactLoggerAdapter((logger,), run_root=tmp_path).register(publication)

    payload = logger.log_metrics.call_args.args[0]
    prefix = "evaluations/best_checkpoint/val/slices/source/alpha/"
    assert payload[f"{prefix}num_examples"] == 3
    assert payload[f"{prefix}accuracy"] == pytest.approx(2 / 3)


class _FakeWandbArtifact:
    def __init__(
        self, name: str, *, type: str, metadata: dict[str, object]
    ) -> None:
        self.name = name
        self.type = type
        self.metadata = metadata
        self.files: list[tuple[str, str | None, str | None]] = []

    def add_file(
        self,
        local_path: str,
        *,
        name: str | None = None,
        policy: str | None = None,
    ) -> None:
        self.files.append((local_path, name, policy))


def _wandb_logger() -> tuple[WandbLogger, MagicMock, list[Mock]]:
    logger = MagicMock(spec=WandbLogger)
    experiment = MagicMock()
    experiment.id = "wandb-run-0"
    waiters: list[Mock] = []

    def log_artifact(_artifact: object) -> SimpleNamespace:
        waiter = Mock()
        waiters.append(waiter)
        return SimpleNamespace(wait=waiter)

    experiment.log_artifact.side_effect = log_artifact
    logger.experiment = experiment
    logger.name = "qualified"
    logger.version = "wandb-run-0"
    logger.log_metrics = MagicMock()
    return logger, experiment, waiters


def test_wandb_uploads_one_immutable_artifact_per_file_with_stable_split_names(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[_FakeWandbArtifact] = []

    def artifact_factory(
        name: str,
        *,
        type: str,
        metadata: dict[str, object],
    ) -> _FakeWandbArtifact:
        artifact = _FakeWandbArtifact(name, type=type, metadata=metadata)
        created.append(artifact)
        return artifact

    monkeypatch.setattr(artifact_logging.wandb, "Artifact", artifact_factory)
    logger, experiment, waiters = _wandb_logger()
    adapter = ArtifactLoggerAdapter(loggers=(logger,), run_root=tmp_path)
    publications = (
        _publication(tmp_path, "val", shard_count=1),
        _publication(tmp_path, "test", shard_count=1),
    )

    for publication in publications:
        adapter.register(publication)

    expected_files = tuple(
        artifact
        for publication in publications
        for artifact in _files(publication)
    )
    assert (
        len(created)
        == experiment.log_artifact.call_count
        == len(expected_files)
        == 6
    )
    assert [artifact.name for artifact in created] == [
        "best-checkpoint-val-metrics",
        "best-checkpoint-val-predictions-manifest",
        "best-checkpoint-val-predictions-part-00000",
        "best-checkpoint-test-metrics",
        "best-checkpoint-test-predictions-manifest",
        "best-checkpoint-test-predictions-part-00000",
    ]
    for created_artifact, local_file in zip(
        created, expected_files, strict=True
    ):
        split = "val" if "-val-" in created_artifact.name else "test"
        assert created_artifact.files == [
            (str(local_file.path), local_file.path.name, "immutable")
        ]
        assert created_artifact.metadata["split"] == split
        assert created_artifact.metadata["sha256"] == local_file.sha256
        assert (
            created_artifact.metadata["checkpoint_sha256"]
            == _CHECKPOINT_SHA256
        )
        assert (
            created_artifact.metadata["uri"]
            == local_file.path.resolve().as_uri()
        )
    assert len(waiters) == len(expected_files)
    assert all(waiter.call_count == 1 for waiter in waiters)

    assert logger.log_metrics.call_count == 2
    for call, split in zip(
        logger.log_metrics.call_args_list, ("val", "test"), strict=True
    ):
        values = call.args[0]
        assert (
            type(values[f"evaluations/best_checkpoint/{split}/num_examples"])
            is int
        )
        assert values[f"evaluations/best_checkpoint/{split}/num_examples"] == 5
        assert call.kwargs == {"step": 17}


def test_default_many_logger_set_publishes_to_csv_and_wandb(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The packaged many-loggers composition publishes through both adapters."""
    monkeypatch.setattr(
        artifact_logging.wandb,
        "Artifact",
        _FakeWandbArtifact,
    )
    csv_logger = CSVLogger(
        save_dir=tmp_path / "csv",
        name="qualified",
        version="run-0",
    )
    csv_logger.log_metrics = MagicMock()
    wandb_logger, experiment, waiters = _wandb_logger()
    publication = _publication(tmp_path, "val", shard_count=1)

    ArtifactLoggerAdapter(
        loggers=(csv_logger, wandb_logger),
        run_root=tmp_path,
    ).register(publication)

    index_path = Path(csv_logger.log_dir) / "artifact-index.jsonl"
    records = [json.loads(line) for line in index_path.read_bytes().splitlines()]
    expected_files = _files(publication)
    assert [record["name"] for record in records] == [
        artifact.registration_name for artifact in expected_files
    ]
    assert experiment.log_artifact.call_count == len(expected_files)
    assert len(waiters) == len(expected_files)
    assert csv_logger.log_metrics.call_count == 1
    assert wandb_logger.log_metrics.call_count == 1


def test_unsupported_logger_is_rejected_explicitly(tmp_path: Path) -> None:
    class UnsupportedLogger:
        def log_metrics(
            self, metrics: dict[str, float], step: int | None = None
        ) -> None:
            del metrics, step

    publication = _publication(tmp_path, "val", shard_count=1)
    with pytest.raises(TypeError, match="unsupported|artifact adapter|logger"):
        adapter = ArtifactLoggerAdapter(
            loggers=(UnsupportedLogger(),),
            run_root=tmp_path,
        )
        adapter.register(publication)


def test_wandb_upload_failure_preserves_every_authoritative_local_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(artifact_logging.wandb, "Artifact", _FakeWandbArtifact)
    logger = MagicMock(spec=WandbLogger)
    failed_upload = SimpleNamespace(
        wait=Mock(side_effect=RuntimeError("upload failed"))
    )
    logger.experiment.log_artifact.return_value = failed_upload
    logger.log_metrics = MagicMock()
    publication = _publication(tmp_path, "test", shard_count=2)
    before = {
        artifact.path: artifact.path.read_bytes()
        for artifact in _files(publication)
    }

    with pytest.raises(RuntimeError, match="upload failed"):
        ArtifactLoggerAdapter(loggers=(logger,), run_root=tmp_path).register(
            publication
        )

    assert failed_upload.wait.call_count == 1
    assert all(path.is_file() for path in before)
    assert {path: path.read_bytes() for path in before} == before


def test_csv_registration_is_durable_and_idempotent_by_name_and_digest(
    tmp_path: Path,
) -> None:
    logger = CSVLogger(
        save_dir=tmp_path / "csv", name="qualified", version="run-0"
    )
    logger.log_metrics = MagicMock()
    publication = _publication(tmp_path, "val", shard_count=1)

    ArtifactLoggerAdapter(loggers=(logger,), run_root=tmp_path).register(
        publication
    )
    index_path = Path(logger.log_dir) / "artifact-index.jsonl"
    completed_index = index_path.read_bytes()

    ArtifactLoggerAdapter(loggers=(logger,), run_root=tmp_path).register(
        publication
    )

    assert index_path.read_bytes() == completed_index
    records = [json.loads(line) for line in completed_index.splitlines()]
    assert [(record["name"], record["sha256"]) for record in records] == [
        (artifact.registration_name, artifact.sha256)
        for artifact in _files(publication)
    ]


def test_csv_reconciliation_truncates_partial_tail_and_resumes_without_duplicates(
    tmp_path: Path,
) -> None:
    logger = CSVLogger(
        save_dir=tmp_path / "csv", name="qualified", version="run-0"
    )
    logger.log_metrics = MagicMock()
    val_publication = _publication(tmp_path, "val", shard_count=1)
    test_publication = _publication(tmp_path, "test", shard_count=1)
    ArtifactLoggerAdapter(loggers=(logger,), run_root=tmp_path).register(
        val_publication
    )
    index_path = Path(logger.log_dir) / "artifact-index.jsonl"
    valid_prefix = index_path.read_bytes()
    with index_path.open("ab") as stream:
        stream.write(
            b'{"schema_version":"topobench-artifact-index-v1",'
            b'"name":"interrupted'
        )

    ArtifactLoggerAdapter(loggers=(logger,), run_root=tmp_path).register(
        test_publication
    )

    completed_index = index_path.read_bytes()
    assert completed_index.startswith(valid_prefix)
    records = [json.loads(line) for line in completed_index.splitlines()]
    expected_artifacts = (
        *_files(val_publication),
        *_files(test_publication),
    )
    assert [(record["name"], record["sha256"]) for record in records] == [
        (artifact.registration_name, artifact.sha256)
        for artifact in expected_artifacts
    ]


def test_wandb_registration_is_durable_and_idempotent_by_name_and_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[_FakeWandbArtifact] = []

    def artifact_factory(
        name: str,
        *,
        type: str,
        metadata: dict[str, object],
    ) -> _FakeWandbArtifact:
        artifact = _FakeWandbArtifact(name, type=type, metadata=metadata)
        created.append(artifact)
        return artifact

    monkeypatch.setattr(artifact_logging.wandb, "Artifact", artifact_factory)
    logger, experiment, _ = _wandb_logger()
    publication = _publication(tmp_path, "test", shard_count=1)

    ArtifactLoggerAdapter(loggers=(logger,), run_root=tmp_path).register(
        publication
    )
    recreated_logger = MagicMock(spec=WandbLogger)
    recreated_logger.experiment = experiment
    recreated_logger.name = logger.name
    recreated_logger.version = logger.version
    recreated_logger.log_metrics = MagicMock()
    ArtifactLoggerAdapter(
        loggers=(recreated_logger,), run_root=tmp_path
    ).register(publication)

    expected_files = _files(publication)
    assert experiment.log_artifact.call_count == len(expected_files)
    assert [artifact.name for artifact in created] == [
        artifact.registration_name for artifact in expected_files
    ]
    assert [
        (artifact.name, artifact.metadata["sha256"]) for artifact in created
    ] == [
        (artifact.registration_name, artifact.sha256)
        for artifact in expected_files
    ]


def test_registration_rejects_a_completed_name_with_a_conflicting_digest(
    tmp_path: Path,
) -> None:
    logger = CSVLogger(
        save_dir=tmp_path / "csv", name="qualified", version="run-0"
    )
    logger.log_metrics = MagicMock()
    publication = _publication(tmp_path, "val", shard_count=1)
    adapter = ArtifactLoggerAdapter(loggers=(logger,), run_root=tmp_path)
    adapter.register(publication)
    index_path = Path(logger.log_dir) / "artifact-index.jsonl"
    completed_index = index_path.read_bytes()
    conflicting_publication = _with_changed_metric(publication)

    with pytest.raises(ValueError, match="conflict|digest"):
        ArtifactLoggerAdapter(loggers=(logger,), run_root=tmp_path).register(
            conflicting_publication
        )

    assert index_path.read_bytes() == completed_index
    assert logger.log_metrics.call_count == 1


def test_retry_resumes_each_logger_without_duplicate_entries_or_uploads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[_FakeWandbArtifact] = []

    def artifact_factory(
        name: str,
        *,
        type: str,
        metadata: dict[str, object],
    ) -> _FakeWandbArtifact:
        artifact = _FakeWandbArtifact(name, type=type, metadata=metadata)
        created.append(artifact)
        return artifact

    monkeypatch.setattr(artifact_logging.wandb, "Artifact", artifact_factory)
    csv_logger = CSVLogger(
        save_dir=tmp_path / "csv", name="qualified", version="run-0"
    )
    csv_logger.log_metrics = MagicMock()
    wandb_logger, experiment, waiters = _wandb_logger()
    upload_attempt = 0

    def fail_third_upload(_artifact: object) -> SimpleNamespace:
        nonlocal upload_attempt
        upload_attempt += 1
        waiter = Mock(
            side_effect=(
                RuntimeError("later upload failed")
                if upload_attempt == 3
                else None
            )
        )
        waiters.append(waiter)
        return SimpleNamespace(wait=waiter)

    experiment.log_artifact.side_effect = fail_third_upload
    publication = _publication(tmp_path, "test", shard_count=2)
    adapter = ArtifactLoggerAdapter(
        loggers=(csv_logger, wandb_logger),
        run_root=tmp_path,
    )

    with pytest.raises(RuntimeError, match="later upload failed"):
        adapter.register(publication)

    ArtifactLoggerAdapter(
        loggers=(csv_logger, wandb_logger),
        run_root=tmp_path,
    ).register(publication)

    expected_names = [
        artifact.registration_name for artifact in _files(publication)
    ]
    index_path = Path(csv_logger.log_dir) / "artifact-index.jsonl"
    records = [
        json.loads(line) for line in index_path.read_bytes().splitlines()
    ]
    assert [record["name"] for record in records] == expected_names
    attempted_names = [artifact.name for artifact in created]
    assert attempted_names == [
        expected_names[0],
        expected_names[1],
        expected_names[2],
        expected_names[2],
        expected_names[3],
    ]
    assert all(waiter.call_count == 1 for waiter in waiters)


def test_csv_metrics_are_flushed_to_disk_before_registration_returns(
    tmp_path: Path,
) -> None:
    logger = CSVLogger(
        save_dir=tmp_path / "csv",
        name="qualified",
        version="run-0",
        flush_logs_every_n_steps=10_000,
    )
    publication = _publication(tmp_path, "val", shard_count=1)

    ArtifactLoggerAdapter(loggers=(logger,), run_root=tmp_path).register(
        publication
    )

    metrics_path = Path(logger.log_dir) / "metrics.csv"
    assert metrics_path.is_file()
    with metrics_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    row = rows[0]
    assert int(row["step"]) == 17
    assert float(row["evaluations/best_checkpoint/val/loss"]) == 0.25
    assert float(row["evaluations/best_checkpoint/val/accuracy"]) == 0.6
    assert int(row["evaluations/best_checkpoint/val/num_examples"]) == 5
