"""Generic fitted-transform lifecycle and safe-state contract tests."""

from __future__ import annotations

import errno
import json
import os
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from torch_geometric.data import Data

import topobench.transforms.fittable as fittable_module
from topobench.transforms.fittable import (
    FitContext,
    FitStateError,
    FitStatePublisher,
    FitStatus,
    FittableTransform,
    TransformSpec,
    build_fit_state_key,
)

_SHA_A = "a" * 64
_SHA_B = "b" * 64


class _DeclaredTransform:
    """Small structural implementation used to exercise the generic contract."""

    def __init__(
        self,
        *,
        scale: float = 1.0,
        max_batch_rows: int = 4,
        max_batch_bytes: int = 1024,
    ) -> None:
        self.scale = scale
        self.max_batch_rows = max_batch_rows
        self.max_batch_bytes = max_batch_bytes
        self.status = FitStatus.UNFITTED
        self._state_key: str | None = None
        self.spec = TransformSpec(
            input_kinds=("Data", "HeteroData"),
            output_kind="same",
            deterministic=True,
            device="cpu",
            preserves_node_identity=True,
            preserves_supervision=True,
            feature_width_behavior="preserve",
            edge_effects="none",
            accesses_labels=False,
            target_node_type="author",
            target_field="x",
            input_dtype="float32",
            output_dtype="float32",
            accumulation_dtype="float64",
        )

    @property
    def state_key(self) -> str | None:
        if self.status is not FitStatus.FITTED:
            return None
        return self._state_key

    def canonical_config(self) -> dict[str, object]:
        return {"scale": self.scale}

    def implementation_versions(self) -> dict[str, str]:
        return {"fixture": "1"}

    def begin_fit(self, context: FitContext) -> None:
        self.status = FitStatus.FITTING

    def update_fit(
        self, features: np.ndarray, labels: np.ndarray | None = None
    ) -> None:
        if self.status is not FitStatus.FITTING:
            raise RuntimeError("not fitting")

    def finalize_fit(self, state_root: str | Path) -> object:
        self.status = FitStatus.FITTED
        return object()

    def load_state(
        self, state_root: str | Path, context: FitContext
    ) -> object:
        self.status = FitStatus.FITTED
        return object()

    def transform(self, batch: Data) -> Data:
        return batch.clone()


class _DifferentCodeTransform(_DeclaredTransform):
    """A distinct class must never share fitted state with the fixture class."""


def _context() -> FitContext:
    return FitContext(
        content_sha256=_SHA_A,
        active_split_tag="primary",
        train_ids_sha256=_SHA_B,
        train_source_sha256="c" * 64,
        target_node_type="author",
        target_field="x",
        input_shape=(4, 3),
        input_width=3,
        input_dtype="float32",
        input_schema_sha256="d" * 64,
        package_versions=(("numpy", np.__version__), ("topobench", "test")),
        numeric_precision="float64",
    )


def test_protocol_and_declaration_are_explicit_runtime_immutable_contracts() -> (
    None
):
    transform = _DeclaredTransform()

    assert isinstance(transform, FittableTransform)
    assert transform.status is FitStatus.UNFITTED
    assert transform.state_key is None
    assert transform.spec.input_kinds == ("Data", "HeteroData")
    assert transform.spec.output_kind == "same"
    assert transform.spec.deterministic is True
    assert transform.spec.device == "cpu"
    assert transform.spec.preserves_node_identity is True
    assert transform.spec.preserves_supervision is True
    assert transform.spec.feature_width_behavior == "preserve"
    assert transform.spec.edge_effects == "none"
    assert transform.spec.accesses_labels is False
    assert transform.spec.target_node_type == "author"
    assert transform.spec.target_field == "x"
    with pytest.raises(FrozenInstanceError):
        transform.spec.device = "cuda"  # type: ignore[misc]


def test_state_key_changes_for_every_scientific_identity_component() -> None:
    context = _context()
    transform = _DeclaredTransform()
    baseline = build_fit_state_key(context, transform)
    mutations = (
        replace(context, content_sha256="0" * 64),
        replace(context, active_split_tag="diagnostic"),
        replace(context, train_ids_sha256="1" * 64),
        replace(context, train_source_sha256="2" * 64),
        replace(context, target_node_type="paper"),
        replace(context, target_field="embedding"),
        replace(context, input_shape=(5, 3)),
        replace(context, input_shape=(4, 4), input_width=4),
        replace(context, input_dtype="float64"),
        replace(context, input_schema_sha256="3" * 64),
        replace(context, package_versions=(("numpy", "different"),)),
        replace(context, numeric_precision="float32"),
    )

    assert all(
        build_fit_state_key(value, transform) != baseline
        for value in mutations
    )
    assert (
        build_fit_state_key(context, _DeclaredTransform(scale=2.0)) != baseline
    )
    assert build_fit_state_key(context, _DifferentCodeTransform()) != baseline
    changed_spec = _DeclaredTransform()
    changed_spec.spec = replace(changed_spec.spec, output_dtype="float64")
    assert build_fit_state_key(context, changed_spec) != baseline


def test_state_key_changes_with_fit_driver_and_chunk_schedule_versions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context()
    transform = _DeclaredTransform()
    baseline = build_fit_state_key(context, transform)

    monkeypatch.setattr(
        fittable_module,
        "_CANONICAL_FIT_DRIVER_VERSION",
        "canonical-fit-driver-v2",
    )
    driver_changed = build_fit_state_key(context, transform)
    monkeypatch.setattr(
        fittable_module,
        "_INCREMENTAL_PCA_CHUNK_SCHEDULE_VERSION",
        "incremental-pca-chunk-schedule-v2",
    )
    both_changed = build_fit_state_key(context, transform)

    assert driver_changed != baseline
    assert both_changed != driver_changed


def test_publisher_round_trips_only_validated_json_and_non_executable_arrays(
    tmp_path: Path,
) -> None:
    publisher = FitStatePublisher(tmp_path)
    key = build_fit_state_key(_context(), _DeclaredTransform())
    arrays = {
        "components": np.arange(6, dtype=np.float64).reshape(2, 3),
        "mean": np.array([1.0, 2.0, 3.0], dtype=np.float64),
    }
    metadata: dict[str, Any] = {"kind": "fixture", "sample_count": 4}

    published = publisher.publish(key, metadata=metadata, arrays=arrays)
    loaded = publisher.load(key, expected_metadata=metadata)

    assert published.path == tmp_path / key
    assert loaded.manifest == published.manifest
    assert set(path.name for path in published.path.iterdir()) == {
        "components.npy",
        "manifest.json",
        "mean.npy",
    }
    assert (
        json.loads(
            (published.path / "manifest.json").read_text(encoding="utf-8")
        )["state_key"]
        == key
    )
    assert loaded.arrays["components"].flags.writeable is False
    np.testing.assert_array_equal(
        loaded.arrays["components"], arrays["components"]
    )
    with pytest.raises(
        (TypeError, ValueError), match="object|executable|pickle"
    ):
        publisher.publish(
            "f" * 64,
            metadata={},
            arrays={"unsafe": np.array([object()], dtype=object)},
        )


def test_publisher_never_overwrites_or_reuses_corrupt_or_partial_state(
    tmp_path: Path,
) -> None:
    publisher = FitStatePublisher(tmp_path)
    key = build_fit_state_key(_context(), _DeclaredTransform())
    first = publisher.publish(
        key,
        metadata={"sample_count": 4},
        arrays={"mean": np.ones(3, dtype=np.float64)},
    )
    assert (
        publisher.publish(
            key,
            metadata={"sample_count": 4},
            arrays={"mean": np.ones(3, dtype=np.float64)},
        ).path
        == first.path
    )
    with pytest.raises(FitStateError, match="overwrite|identity|existing"):
        publisher.publish(
            key,
            metadata={"sample_count": 5},
            arrays={"mean": np.ones(3, dtype=np.float64)},
        )

    mean_path = first.path / "mean.npy"
    mean_path.chmod(0o644)
    payload = bytearray(mean_path.read_bytes())
    payload[-1] ^= 1
    mean_path.write_bytes(payload)
    with pytest.raises(FitStateError, match="checksum|corrupt"):
        publisher.load(key)


def test_publisher_treats_atomic_same_key_collision_as_validated_cache_hit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    publisher = FitStatePublisher(tmp_path)
    foreign_staging = tmp_path / ".staging-foreign-publisher"
    foreign_staging.mkdir()
    (foreign_staging / "owned").write_text("active", encoding="utf-8")
    key = build_fit_state_key(_context(), _DeclaredTransform())
    real_rename = os.rename

    def concurrent_winner(source: str | Path, target: str | Path) -> None:
        real_rename(source, target)
        target_path = Path(target)
        for child in target_path.iterdir():
            child.chmod(0o444)
        target_path.chmod(0o555)
        raise OSError(errno.ENOTEMPTY, "concurrent state already published")

    monkeypatch.setattr(os, "rename", concurrent_winner)
    state = publisher.publish(
        key,
        metadata={"sample_count": 4},
        arrays={"mean": np.ones(3, dtype=np.float64)},
    )

    assert state.key == key
    np.testing.assert_array_equal(state.arrays["mean"], np.ones(3))
    assert foreign_staging.is_dir()
    assert (foreign_staging / "owned").read_text(encoding="utf-8") == "active"
