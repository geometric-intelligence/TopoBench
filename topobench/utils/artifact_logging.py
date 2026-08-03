"""Durable experiment-logger registration for prediction artifacts."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from numbers import Real
from pathlib import Path
from threading import Lock
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import wandb
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.loggers.wandb import WandbLogger

ARTIFACT_LOGGER_TARGETS: Mapping[str, str] = MappingProxyType(
    {
        "csv": "lightning.pytorch.loggers.csv_logs.CSVLogger",
        "wandb": "lightning.pytorch.loggers.wandb.WandbLogger",
    }
)
_ARTIFACT_LOGGER_TYPES = (CSVLogger, WandbLogger)

if TYPE_CHECKING:
    from topobench.callbacks.prediction_artifacts import (
        ArtifactFile,
        SplitPublication,
    )

_INDEX_SCHEMA = "topobench-artifact-index-v1"
_WANDB_ARTIFACT_TYPE = "topobench-selected-checkpoint-file"
_SUPPORTED_SPLITS = frozenset({"val", "test"})
_STATE_SCHEMA = "topobench-artifact-registration-state-v1"
_STATE_FILE_NAME = ".artifact-registration-state.json"
_PROCESS_REGISTRATION_LOCK = Lock()


class ArtifactLoggerAdapter:
    """Register authoritative selected-checkpoint files with supported loggers."""

    __slots__ = ("_loggers", "_run_root")

    def __init__(
        self, loggers: Iterable[object], run_root: str | Path
    ) -> None:
        self._run_root = Path(run_root).resolve()
        self._loggers = tuple(loggers)
        for logger in self._loggers:
            if not isinstance(logger, _ARTIFACT_LOGGER_TYPES):
                logger_type = type(logger)
                qualified_name = (
                    f"{logger_type.__module__}.{logger_type.__qualname__}"
                )
                raise TypeError(
                    "unsupported logger for the prediction artifact adapter: "
                    f"{qualified_name}"
                )

    def register(self, publication: SplitPublication) -> None:
        """Register every file, then log the split's finalized scalar values once."""
        files = (
            publication.metrics_file,
            publication.manifest_file,
            *publication.shard_files,
        )
        validated_files = self._validate_files(publication, files)
        metrics, global_step = self._load_metrics(
            publication, validated_files[0][1]
        )
        publication_digest = _publication_digest(
            publication, validated_files, global_step
        )
        state_path = self._run_root / _STATE_FILE_NAME

        with _exclusive_run_root(self._run_root):
            state = _load_registration_state(state_path)
            prepared: list[tuple[object, dict[str, Any]]] = []
            changed = False
            for logger in self._loggers:
                identity = _logger_identity(logger)
                logger_state, created = _logger_state(state, identity)
                changed |= created
                if isinstance(logger, CSVLogger):
                    changed |= _reconcile_csv_index(logger, logger_state)
                changed |= _reserve_files(logger_state, validated_files)
                changed |= _reserve_metrics(
                    logger_state, publication.split, publication_digest
                )
                prepared.append((logger, logger_state))

            completion = [
                _publication_completed(
                    logger_state,
                    publication.split,
                    publication_digest,
                    validated_files,
                )
                for _, logger_state in prepared
            ]

            if changed:
                _atomic_write_state(state_path, state)

            for (logger, logger_state), already_completed in zip(
                prepared, completion, strict=True
            ):
                if already_completed:
                    continue
                if isinstance(logger, CSVLogger):
                    self._register_csv(
                        logger,
                        publication,
                        validated_files,
                        state_path,
                        state,
                        logger_state,
                    )
                else:
                    self._register_wandb(
                        logger,
                        publication,
                        validated_files,
                        state_path,
                        state,
                        logger_state,
                    )
                logger.log_metrics(metrics, step=global_step)
                if isinstance(logger, CSVLogger):
                    logger.save()
                _mark_metrics_completed(
                    logger_state, publication.split, publication_digest
                )
                _atomic_write_state(state_path, state)

    def _validate_files(
        self,
        publication: SplitPublication,
        files: tuple[ArtifactFile, ...],
    ) -> tuple[tuple[ArtifactFile, Path], ...]:
        split = publication.split
        if split not in _SUPPORTED_SPLITS:
            raise ValueError(
                f"unsupported selected-checkpoint split: {split!r}"
            )
        if not files:
            raise ValueError("a split publication must contain artifact files")

        expected_names = (
            f"best-checkpoint-{split}-metrics",
            f"best-checkpoint-{split}-predictions-manifest",
            *(
                f"best-checkpoint-{split}-predictions-part-{index:05d}"
                for index in range(len(publication.shard_files))
            ),
        )
        validated: list[tuple[ArtifactFile, Path]] = []
        seen_paths: set[Path] = set()
        for descriptor, expected_name in zip(
            files, expected_names, strict=True
        ):
            if descriptor.registration_name != expected_name:
                raise ValueError(
                    "artifact registration name is not the canonical split-qualified "
                    f"name: expected {expected_name!r}"
                )
            path = Path(descriptor.path)
            if not path.is_absolute():
                path = self._run_root / path
            try:
                resolved_path = path.resolve(strict=True)
            except OSError as error:
                raise ValueError(
                    "artifact descriptor does not reference a local file"
                ) from error
            try:
                resolved_path.relative_to(self._run_root)
            except ValueError as error:
                raise ValueError(
                    "artifact descriptor escapes the run root"
                ) from error
            if not resolved_path.is_file():
                raise ValueError(
                    "artifact descriptor does not reference a regular file"
                )
            if resolved_path in seen_paths:
                raise ValueError(
                    "a split publication cannot register the same file twice"
                )
            seen_paths.add(resolved_path)

            stat = resolved_path.stat()
            if (
                type(descriptor.byte_size) is not int
                or descriptor.byte_size != stat.st_size
            ):
                raise ValueError(
                    "artifact descriptor byte size does not match the local file"
                )
            digest = _sha256(resolved_path)
            if descriptor.sha256 != digest:
                raise ValueError(
                    "artifact descriptor digest does not match the local file"
                )
            validated.append((descriptor, resolved_path))
        return tuple(validated)

    def _load_metrics(
        self,
        publication: SplitPublication,
        metrics_path: Path,
    ) -> tuple[dict[str, int | float], int]:
        try:
            document = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError(
                "metrics artifact is not valid UTF-8 JSON"
            ) from error
        if not isinstance(document, Mapping):
            raise ValueError("metrics artifact must contain a JSON object")
        if document.get("split") != publication.split:
            raise ValueError(
                "metrics artifact split does not match its publication"
            )

        checkpoint = document.get("checkpoint")
        if not isinstance(checkpoint, Mapping):
            raise ValueError("metrics artifact checkpoint must be an object")
        if checkpoint.get("sha256") != publication.checkpoint_sha256:
            raise ValueError(
                "metrics artifact checkpoint digest does not match its publication"
            )
        global_step = checkpoint.get("global_step")
        if type(global_step) is not int or global_step < 0:
            raise ValueError(
                "metrics artifact checkpoint global_step must be a nonnegative integer"
            )

        num_examples = document.get("num_examples")
        if type(num_examples) is not int or num_examples < 0:
            raise ValueError(
                "metrics artifact num_examples must be a nonnegative integer"
            )
        if num_examples != publication.num_examples:
            raise ValueError(
                "metrics artifact num_examples does not match its publication"
            )

        raw_metrics = document.get("metrics")
        if not isinstance(raw_metrics, Mapping):
            raise ValueError("metrics artifact metrics must be an object")
        prefix = f"evaluations/best_checkpoint/{publication.split}/"
        metrics: dict[str, int | float] = {}
        for name, value in raw_metrics.items():
            if not isinstance(name, str) or not name:
                raise ValueError("metric names must be nonempty strings")
            if not isinstance(value, Real) or isinstance(value, bool):
                raise ValueError(f"metric {name!r} must be a numeric scalar")
            if not math.isfinite(value):
                raise ValueError(f"metric {name!r} must be finite")
            metrics[f"{prefix}{name}"] = value
        raw_slices = document.get("slices", {})
        if not isinstance(raw_slices, Mapping):
            raise ValueError("metrics artifact slices must be an object")
        for field, categories in raw_slices.items():
            if (
                not isinstance(field, str)
                or not field
                or "/" in field
                or not isinstance(categories, Mapping)
            ):
                raise ValueError(
                    "metrics artifact slice fields must be safe object names"
                )
            for category, record in categories.items():
                if (
                    not isinstance(category, str)
                    or not category
                    or "/" in category
                    or not isinstance(record, Mapping)
                ):
                    raise ValueError(
                        "metrics artifact slice categories must be safe objects"
                    )
                count = record.get("num_examples")
                if type(count) is not int or count < 0:
                    raise ValueError(
                        "metrics artifact slice num_examples must be a "
                        "nonnegative integer"
                    )
                slice_metrics = record.get("metrics")
                if not isinstance(slice_metrics, Mapping):
                    raise ValueError(
                        "metrics artifact slice metrics must be an object"
                    )
                slice_prefix = f"{prefix}slices/{field}/{category}/"
                metrics[f"{slice_prefix}num_examples"] = count
                for name, value in slice_metrics.items():
                    if not isinstance(name, str) or not name:
                        raise ValueError(
                            "slice metric names must be nonempty strings"
                        )
                    if not isinstance(value, Real) or isinstance(value, bool):
                        raise ValueError(
                            f"slice metric {name!r} must be a numeric scalar"
                        )
                    if not math.isfinite(value):
                        raise ValueError(
                            f"slice metric {name!r} must be finite"
                        )
                    metrics[f"{slice_prefix}{name}"] = value
        metrics[f"{prefix}num_examples"] = num_examples
        return metrics, global_step

    @staticmethod
    def _register_csv(
        logger: CSVLogger,
        publication: SplitPublication,
        files: tuple[tuple[ArtifactFile, Path], ...],
        state_path: Path,
        state: dict[str, Any],
        logger_state: dict[str, Any],
    ) -> None:
        log_dir = Path(logger.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        index_path = log_dir / "artifact-index.jsonl"
        for descriptor, path in files:
            status = _registration_status(
                logger_state,
                descriptor.registration_name,
                descriptor.sha256,
            )
            if status["completed"]:
                continue
            record = {
                "schema_version": _INDEX_SCHEMA,
                "name": descriptor.registration_name,
                "split": publication.split,
                "uri": path.as_uri(),
                "sha256": descriptor.sha256,
                "byte_size": descriptor.byte_size,
                "checkpoint_sha256": publication.checkpoint_sha256,
            }
            encoded = (
                json.dumps(
                    record,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
                + "\n"
            ).encode("utf-8")
            _durable_append(index_path, encoded)
            status["completed"] = True
            _atomic_write_state(state_path, state)

    @staticmethod
    def _register_wandb(
        logger: WandbLogger,
        publication: SplitPublication,
        files: tuple[tuple[ArtifactFile, Path], ...],
        state_path: Path,
        state: dict[str, Any],
        logger_state: dict[str, Any],
    ) -> None:
        for descriptor, path in files:
            status = _registration_status(
                logger_state,
                descriptor.registration_name,
                descriptor.sha256,
            )
            if status["completed"]:
                continue
            artifact = wandb.Artifact(
                descriptor.registration_name,
                type=_WANDB_ARTIFACT_TYPE,
                metadata={
                    "split": publication.split,
                    "sha256": descriptor.sha256,
                    "byte_size": descriptor.byte_size,
                    "checkpoint_sha256": publication.checkpoint_sha256,
                    "uri": path.as_uri(),
                },
            )
            artifact.add_file(str(path), name=path.name, policy="immutable")
            logger.experiment.log_artifact(artifact).wait()
            status["completed"] = True
            _atomic_write_state(state_path, state)


@contextmanager
def _exclusive_run_root(run_root: Path) -> Iterator[None]:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    with _PROCESS_REGISTRATION_LOCK:
        descriptor = os.open(run_root, flags)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)


def _logger_identity(logger: object) -> str:
    if isinstance(logger, CSVLogger):
        document = {
            "kind": "csv",
            "log_dir": str(Path(logger.log_dir).resolve()),
        }
    else:
        experiment = logger.experiment
        run_id = getattr(experiment, "id", None)
        document = {"kind": "wandb"}
        if isinstance(run_id, str) and run_id:
            document["run_id"] = run_id
            entity = getattr(experiment, "entity", None)
            project = getattr(experiment, "project", None)
            if isinstance(entity, str) and entity:
                document["entity"] = entity
            if isinstance(project, str) and project:
                document["project"] = project
        else:
            name = getattr(logger, "name", None)
            version = getattr(logger, "version", None)
            if isinstance(name, str) and name:
                document["name"] = name
            if isinstance(version, (str, int)):
                document["version"] = str(version)
            if len(document) == 1:
                document["process_local_identity"] = id(logger)
    return json.dumps(
        document,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _load_registration_state(path: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise ValueError(
            "artifact registration state cannot be a symbolic link"
        )
    try:
        payload = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {"schema_version": _STATE_SCHEMA, "loggers": {}}
    except (OSError, UnicodeError) as error:
        raise ValueError(
            "artifact registration state is not readable UTF-8"
        ) from error
    try:
        state = json.loads(payload)
    except json.JSONDecodeError as error:
        raise ValueError(
            "artifact registration state is not valid JSON"
        ) from error
    _validate_registration_state(state)
    return state


def _validate_registration_state(state: object) -> None:
    if (
        not isinstance(state, dict)
        or set(state) != {"schema_version", "loggers"}
        or state.get("schema_version") != _STATE_SCHEMA
    ):
        raise ValueError("artifact registration state has an invalid schema")
    loggers = state.get("loggers")
    if not isinstance(loggers, dict):
        raise ValueError(
            "artifact registration state loggers must be an object"
        )
    for identity, logger_state in loggers.items():
        if not isinstance(identity, str) or not identity:
            raise ValueError(
                "artifact registration logger identity is invalid"
            )
        if not isinstance(logger_state, dict) or set(logger_state) != {
            "registrations",
            "metrics",
        }:
            raise ValueError("artifact registration logger state is invalid")
        registrations = logger_state.get("registrations")
        metrics = logger_state.get("metrics")
        if not isinstance(registrations, dict) or not isinstance(
            metrics, dict
        ):
            raise ValueError("artifact registration entries must be objects")
        for name, digests in registrations.items():
            if not isinstance(name, str) or not name:
                raise ValueError("artifact registration name is invalid")
            _validate_completion_map(digests, "artifact registration")
        for split, digests in metrics.items():
            if split not in _SUPPORTED_SPLITS:
                raise ValueError(
                    "artifact metric registration split is invalid"
                )
            _validate_completion_map(digests, "artifact metric registration")


def _validate_completion_map(value: object, label: str) -> None:
    if not isinstance(value, dict) or len(value) != 1:
        raise ValueError(f"{label} digest state is invalid")
    digest, status = next(iter(value.items()))
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        or not isinstance(status, dict)
        or set(status) != {"completed"}
        or type(status.get("completed")) is not bool
    ):
        raise ValueError(f"{label} completion state is invalid")


def _logger_state(
    state: dict[str, Any], identity: str
) -> tuple[dict[str, Any], bool]:
    loggers = state["loggers"]
    existing = loggers.get(identity)
    if existing is not None:
        return existing, False
    created: dict[str, Any] = {"registrations": {}, "metrics": {}}
    loggers[identity] = created
    return created, True


def _reconcile_csv_index(
    logger: CSVLogger, logger_state: dict[str, Any]
) -> bool:
    index_path = Path(logger.log_dir) / "artifact-index.jsonl"
    try:
        payload = index_path.read_bytes()
    except FileNotFoundError:
        return False
    except OSError as error:
        raise ValueError("CSV artifact index is not readable") from error
    if payload and not payload.endswith(b"\n"):
        payload = _truncate_incomplete_jsonl_tail(index_path, payload)

    changed = False
    for encoded in payload.splitlines():
        try:
            record = json.loads(encoded)
        except (UnicodeError, json.JSONDecodeError) as error:
            raise ValueError(
                "CSV artifact index contains invalid JSON"
            ) from error
        if (
            not isinstance(record, dict)
            or record.get("schema_version") != _INDEX_SCHEMA
        ):
            raise ValueError("CSV artifact index contains an invalid record")
        name = record.get("name")
        digest = record.get("sha256")
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("CSV artifact index identity fields are invalid")
        changed |= _reserve_file(logger_state, name, digest, completed=True)
    return changed


def _truncate_incomplete_jsonl_tail(path: Path, payload: bytes) -> bytes:
    complete_size = payload.rfind(b"\n") + 1
    descriptor = os.open(path, os.O_WRONLY)
    try:
        os.ftruncate(descriptor, complete_size)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _sync_directory(path.parent)
    return payload[:complete_size]


def _reserve_files(
    logger_state: dict[str, Any],
    files: tuple[tuple[ArtifactFile, Path], ...],
) -> bool:
    changed = False
    for descriptor, _ in files:
        changed |= _reserve_file(
            logger_state,
            descriptor.registration_name,
            descriptor.sha256,
            completed=False,
        )
    return changed


def _reserve_file(
    logger_state: dict[str, Any],
    name: str,
    digest: str,
    *,
    completed: bool,
) -> bool:
    registrations = logger_state["registrations"]
    digests = registrations.get(name)
    if digests is None:
        registrations[name] = {digest: {"completed": completed}}
        return True
    if set(digests) != {digest}:
        raise ValueError(
            f"artifact registration name {name!r} has a conflicting digest"
        )
    status = digests[digest]
    if completed and not status["completed"]:
        status["completed"] = True
        return True
    return False


def _reserve_metrics(
    logger_state: dict[str, Any], split: str, digest: str
) -> bool:
    metrics = logger_state["metrics"]
    digests = metrics.get(split)
    if digests is None:
        metrics[split] = {digest: {"completed": False}}
        return True
    if set(digests) != {digest}:
        raise ValueError(
            f"scalar registration for split {split!r} has a conflicting digest"
        )
    return False


def _registration_status(
    logger_state: dict[str, Any], name: str, digest: str
) -> dict[str, bool]:
    return logger_state["registrations"][name][digest]


def _metrics_status(
    logger_state: dict[str, Any], split: str, digest: str
) -> dict[str, bool]:
    return logger_state["metrics"][split][digest]


def _publication_completed(
    logger_state: dict[str, Any],
    split: str,
    publication_digest: str,
    files: tuple[tuple[ArtifactFile, Path], ...],
) -> bool:
    files_completed = all(
        _registration_status(
            logger_state,
            descriptor.registration_name,
            descriptor.sha256,
        )["completed"]
        for descriptor, _ in files
    )
    metrics_completed = _metrics_status(
        logger_state, split, publication_digest
    )["completed"]
    if metrics_completed and not files_completed:
        raise ValueError(
            "artifact registration state completed metrics before its files"
        )
    return files_completed and metrics_completed


def _mark_metrics_completed(
    logger_state: dict[str, Any], split: str, digest: str
) -> None:
    _metrics_status(logger_state, split, digest)["completed"] = True


def _publication_digest(
    publication: SplitPublication,
    files: tuple[tuple[ArtifactFile, Path], ...],
    global_step: int,
) -> str:
    document = {
        "checkpoint_sha256": publication.checkpoint_sha256,
        "files": [
            {
                "name": descriptor.registration_name,
                "sha256": descriptor.sha256,
            }
            for descriptor, _ in files
        ],
        "global_step": global_step,
        "num_examples": publication.num_examples,
        "split": publication.split,
    }
    encoded = json.dumps(
        document,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_write_state(path: Path, state: dict[str, Any]) -> None:
    payload = (
        json.dumps(
            state,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
    )
    temporary_path = Path(temporary_name)
    descriptor_open = True
    try:
        _write_all(descriptor, payload, "artifact registration state write")
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor_open = False
        os.replace(temporary_path, path)
        _sync_directory(path.parent)
    except BaseException:
        if descriptor_open:
            os.close(descriptor)
        temporary_path.unlink(missing_ok=True)
        raise


def _write_all(descriptor: int, payload: bytes, operation: str) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written == 0:
            raise OSError(f"{operation} made no progress")
        view = view[written:]


def _sync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _durable_append(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o666)
    try:
        _write_all(descriptor, payload, "durable artifact index append")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _sync_directory(path.parent)
