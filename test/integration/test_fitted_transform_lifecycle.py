"""Training-only fitted transform integration with Task8 graph sampling."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from test.data.dataload.test_disk_graph_datamodule import (
    materialized_heterogeneous_reference,
)
from test.data.stores.test_typed_graph_store import QualifiedStoreFixture
from topobench.data.stores.typed_graph_store import TypedGraphStore
from topobench.dataloader.disk_graph import (
    DiskGraphDataModule,
    HeterogeneousClusterStrategy,
    HeterogeneousNeighborStrategy,
    HomogeneousClusterStrategy,
    _fit_ids,
)
from topobench.transforms.fittable import FitStateError, FitStatus
from topobench.transforms.incremental_pca import IncrementalPCATransform


class _CountingPCA(IncrementalPCATransform):
    def __init__(self) -> None:
        super().__init__(
            n_components=1,
            max_batch_rows=2,
            max_batch_bytes=16,
            target_node_type="author",
            target_field="x",
            input_dtype="float32",
            output_dtype="float32",
            accumulation_dtype="float64",
        )
        self.transform_calls = 0
        self.fit_features: list[np.ndarray] = []

    def update_fit(
        self,
        features: np.ndarray,
        labels: np.ndarray | None = None,
    ) -> None:
        self.fit_features.append(np.array(features, copy=True))
        super().update_fit(features, labels)

    def transform(self, batch: Any) -> Any:
        self.transform_calls += 1
        return super().transform(batch)


class _HomogeneousCountingPCA(IncrementalPCATransform):
    def __init__(self) -> None:
        super().__init__(
            n_components=1,
            max_batch_rows=1,
            max_batch_bytes=8,
            target_node_type="node",
            input_dtype="float32",
            output_dtype="float32",
            accumulation_dtype="float64",
        )
        self.transform_calls = 0
        self.fit_features: list[np.ndarray] = []

    def update_fit(
        self,
        features: np.ndarray,
        labels: np.ndarray | None = None,
    ) -> None:
        self.fit_features.append(np.array(features, copy=True))
        super().update_fit(features, labels)

    def transform(self, batch: Any) -> Any:
        self.transform_calls += 1
        return super().transform(batch)


class _MissingStateBindingPCA(_CountingPCA):
    @property
    def state_key(self) -> str | None:
        raise AttributeError("state_key binding is intentionally absent")


class _InPlaceProtectedMutationPCA(_CountingPCA):
    def transform(self, batch: Any) -> Any:
        batch["paper"].x.add_(1)
        return super().transform(batch)


class _ParticipantMutationPCA(_CountingPCA):
    def transform(self, batch: Any) -> Any:
        output = super().transform(batch)
        output.participant_counts = dict(output.participant_counts)
        output.participant_counts["author"] += 1
        return output


def _strategy() -> HeterogeneousClusterStrategy:
    # Repeated partition contexts deliberately expose why fitting through sampled
    # batches would be incorrect: canonical fitting must still visit each train ID once.
    return HeterogeneousClusterStrategy(
        partition_groups=((0, 1, 2), (0, 1, 2)),
        clusters_per_batch=1,
        seed=17,
    )


def _tree_digest(root: Path) -> tuple[tuple[str, str], ...]:
    return tuple(
        (
            str(path.relative_to(root)),
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        for path in sorted(root.rglob("*"))
        if path.is_file()
    )



class _BoundedIdSource:
    ndim = 1
    dtype = np.dtype(np.int32)

    def __init__(self, values: np.ndarray, *, maximum_slice: int) -> None:
        self._values = values
        self.maximum_slice = maximum_slice
        self.slice_lengths: list[int] = []

    def __len__(self) -> int:
        return len(self._values)

    def __getitem__(self, item: slice) -> np.ndarray:
        assert isinstance(item, slice)
        assert item.step in {None, 1}
        start = 0 if item.start is None else item.start
        stop = len(self) if item.stop is None else item.stop
        length = stop - start
        assert length <= self.maximum_slice
        self.slice_lengths.append(length)
        return self._values[item]


def test_canonical_fit_ids_validate_hash_and_reiterate_in_bounded_chunks() -> (
    None
):
    source = _BoundedIdSource(
        np.arange(11, dtype=np.int32),
        maximum_slice=3,
    )

    identifiers = _fit_ids(
        source,
        row_count=11,
        active_split_tag="primary",
        chunk_rows=3,
    )
    chunks = tuple(identifiers.iter_chunks())

    np.testing.assert_array_equal(
        np.concatenate(chunks),
        np.arange(11, dtype=np.int64),
    )
    assert identifiers.count == 11
    assert identifiers.dtype == np.dtype("<i8")
    canonical = _fit_ids(
        np.arange(11, dtype="<i8"),
        row_count=11,
        active_split_tag="primary",
        chunk_rows=4,
    )
    assert identifiers.sha256 == canonical.sha256
    assert all(len(chunk) <= 3 for chunk in chunks)
    assert source.slice_lengths
    assert max(source.slice_lengths) <= 3


def test_canonical_mask_ids_never_emit_empty_fit_updates() -> None:
    mask = torch.tensor(
        [False, True, False, False, False, True],
        dtype=torch.bool,
    )

    identifiers = _fit_ids(
        mask,
        row_count=len(mask),
        active_split_tag="primary",
        chunk_rows=2,
        is_mask=True,
    )
    chunks = tuple(identifiers.iter_chunks())

    assert chunks
    assert all(len(chunk) > 0 for chunk in chunks)
    np.testing.assert_array_equal(
        np.concatenate(chunks),
        np.array([1, 5], dtype=np.int64),
    )


@pytest.mark.parametrize(
    ("values", "message"),
    (
        ([0, 2, 1], "strictly increasing|sorted"),
        ([0, 1, 1], "duplicates"),
        ([-1, 0, 1], r"outside \[0, 3\)"),
        ([0, 1, 3], r"outside \[0, 3\)"),
    ),
)
def test_canonical_fit_ids_reject_malformed_streams_incrementally(
    values: list[int],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _fit_ids(
            _BoundedIdSource(
                np.asarray(values, dtype=np.int32),
                maximum_slice=2,
            ),
            row_count=3,
            active_split_tag="primary",
            chunk_rows=2,
        )

@pytest.mark.parametrize("strategy_name", ("cluster", "neighbor"))
def test_fit_reads_only_sorted_canonical_train_rows_once_and_applies_once_per_batch(
    task8_stores: dict[str, QualifiedStoreFixture],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    strategy_name: str,
) -> None:
    fixture = task8_stores["heterogeneous"]
    with TypedGraphStore.open(fixture.store_build.path) as store:
        expected_ids = np.sort(
            np.array(store.split_ids("primary", "train"), copy=True)
        )
        stored_features = np.array(store.node_features("author"), copy=True)

    feature_reads: list[np.ndarray] = []
    original_features = TypedGraphStore.node_features
    original_labels = TypedGraphStore.node_labels

    def observed_features(
        self: TypedGraphStore, node_type: str, rows: Any = None
    ) -> np.ndarray:
        if node_type == "author" and rows is not None:
            feature_reads.append(np.asarray(rows, dtype=np.int64).copy())
        return original_features(self, node_type, rows)

    def forbidden_labels(
        self: TypedGraphStore, node_type: str, rows: Any = None
    ) -> np.ndarray:
        raise AssertionError(
            f"unsupervised fit opened labels for {node_type!r}"
        )

    monkeypatch.setattr(TypedGraphStore, "node_features", observed_features)
    monkeypatch.setattr(TypedGraphStore, "node_labels", forbidden_labels)
    transform = _CountingPCA()
    strategy = (
        _strategy()
        if strategy_name == "cluster"
        else HeterogeneousNeighborStrategy(
            batch_size=1,
            num_neighbors=[-1],
            seed=17,
        )
    )
    module = DiskGraphDataModule(
        fixture.store_build.path,
        strategy,
        active_split_tag="primary",
        train_shuffle=False,
        fitted_transform=transform,
        fitted_state_root=tmp_path / f"states-{strategy_name}",
    )

    module.setup("fit")

    assert transform.status is FitStatus.FITTED
    np.testing.assert_array_equal(np.concatenate(feature_reads), expected_ids)
    assert sum(len(rows) for rows in feature_reads) == len(expected_ids)
    assert all(len(rows) <= transform.max_batch_rows for rows in feature_reads)
    assert tuple(module._descriptors) == ("train", "val")

    monkeypatch.setattr(TypedGraphStore, "node_labels", original_labels)
    batches = list(module.train_dataloader())
    assert transform.transform_calls == len(batches) == 2
    assert all(batch["author"].x.shape[1] == 1 for batch in batches)
    with TypedGraphStore.open(fixture.store_build.path) as reopened:
        np.testing.assert_array_equal(
            reopened.node_features("author"), stored_features
        )
    module.close()


def test_homogeneous_cluster_fits_complete_train_rows_without_leakage(
    task8_stores: dict[str, QualifiedStoreFixture],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = task8_stores["homogeneous"]
    with TypedGraphStore.open(fixture.store_build.path) as store:
        expected_ids = np.sort(
            np.array(store.split_ids("default", "train"), copy=True)
        )
        expected_features = np.array(
            store.node_features("node", expected_ids), copy=True
        )
    original_labels = TypedGraphStore.node_labels

    def forbidden_labels(
        self: TypedGraphStore,
        node_type: str,
        rows: Any = None,
    ) -> np.ndarray:
        raise AssertionError(
            f"unsupervised fit opened labels for {node_type!r}"
        )

    monkeypatch.setattr(TypedGraphStore, "node_labels", forbidden_labels)
    transform = _HomogeneousCountingPCA()
    module = DiskGraphDataModule(
        fixture.store_build.path,
        HomogeneousClusterStrategy(),
        active_split_tag="default",
        train_shuffle=False,
        fitted_transform=transform,
        fitted_state_root=tmp_path / "homogeneous",
    )
    module.setup("fit")

    np.testing.assert_array_equal(
        np.concatenate(transform.fit_features),
        expected_features,
    )
    assert sum(len(value) for value in transform.fit_features) == len(
        expected_ids
    )
    monkeypatch.setattr(TypedGraphStore, "node_labels", original_labels)
    batches = list(module.train_dataloader())
    assert transform.transform_calls == len(batches) == 1
    assert set(expected_ids).issubset(set(batches[0].global_nid.numpy()))
    np.testing.assert_allclose(
        transform.explained_variance_,
        np.zeros(1, dtype=np.float64),
        rtol=np.finfo(np.float64).eps,
        atol=np.finfo(np.float64).eps,
    )
    module.close()


def test_materialized_neighbor_fits_partial_active_tag_rows_only(
    task8_stores: dict[str, QualifiedStoreFixture],
    tmp_path: Path,
) -> None:
    fixture = task8_stores["heterogeneous"]
    with TypedGraphStore.open(fixture.store_build.path) as store:
        data = materialized_heterogeneous_reference(store)
    data.active_split_tag = "diagnostic"
    for phase in ("train", "val", "test"):
        data["author"][f"{phase}_mask"] = data["author"][
            f"diagnostic_{phase}_mask"
        ].clone()
    original_features = data["author"].x.clone()
    expected_ids = (
        data["author"]
        .diagnostic_train_mask.nonzero(as_tuple=False)
        .reshape(-1)
        .numpy()
    )
    transform = _CountingPCA()
    module = DiskGraphDataModule(
        data,
        HeterogeneousNeighborStrategy(
            batch_size=1,
            num_neighbors=[-1],
            seed=17,
        ),
        active_split_tag="diagnostic",
        train_shuffle=False,
        fitted_transform=transform,
        fitted_state_root=tmp_path / "partial-materialized",
    )
    module.setup("fit")

    np.testing.assert_array_equal(
        np.concatenate(transform.fit_features),
        original_features[expected_ids].numpy(),
    )
    assert sum(len(value) for value in transform.fit_features) == len(
        expected_ids
    )
    batches = list(module.train_dataloader())
    assert transform.transform_calls == len(batches) == len(expected_ids)
    assert torch.equal(data["author"].x, original_features)
    module.close()


def test_state_reuse_survives_restart_download_move_resume_and_worker_spawn(
    task8_stores: dict[str, QualifiedStoreFixture],
    tmp_path: Path,
) -> None:
    fixture = task8_stores["heterogeneous"]
    states = tmp_path / "states"
    initial = _CountingPCA()
    first = DiskGraphDataModule(
        fixture.store_build.path,
        _strategy(),
        active_split_tag="primary",
        train_shuffle=False,
        fitted_transform=initial,
        fitted_state_root=states,
    )
    first.setup("fit")
    expected_key = initial.fitted_state.key
    expected_projection = next(iter(first.val_dataloader()))[
        "author"
    ].x.clone()
    first.close()
    before = _tree_digest(states)

    downloaded = tmp_path / "download" / fixture.store_build.path.name
    downloaded.parent.mkdir(parents=True)
    shutil.copytree(fixture.store_build.path, downloaded)
    resumed = _CountingPCA()

    def forbidden_begin(*args: object, **kwargs: object) -> None:
        raise AssertionError(
            "parent attempted to refit an existing immutable state"
        )

    resumed.begin_fit = forbidden_begin  # type: ignore[method-assign]
    second = DiskGraphDataModule(
        downloaded,
        _strategy(),
        active_split_tag="primary",
        train_shuffle=False,
        fitted_transform=resumed,
        fitted_state_root=states,
        num_workers=1,
    )
    second.setup("fit")
    actual_projection = next(iter(second.val_dataloader()))["author"].x

    assert resumed.fitted_state.key == expected_key
    assert torch.equal(actual_projection, expected_projection)
    assert _tree_digest(states) == before
    second.close()

    validation_restart = _CountingPCA()
    validation_restart.begin_fit = forbidden_begin  # type: ignore[method-assign]
    validation = DiskGraphDataModule(
        downloaded,
        _strategy(),
        active_split_tag="primary",
        train_shuffle=False,
        fitted_transform=validation_restart,
        fitted_state_root=states,
    )
    validation.setup("validate")
    assert validation_restart.fitted_state.key == expected_key
    validation.close()


def test_validation_only_never_fits_and_split_tags_cannot_share_state(
    task8_stores: dict[str, QualifiedStoreFixture],
    tmp_path: Path,
) -> None:
    fixture = task8_stores["heterogeneous"]
    missing = _CountingPCA()
    module = DiskGraphDataModule(
        fixture.store_build.path,
        _strategy(),
        active_split_tag="primary",
        train_shuffle=False,
        fitted_transform=missing,
        fitted_state_root=tmp_path / "missing",
    )
    with pytest.raises(FitStateError, match="existing|not found|validation"):
        module.setup("validate")
    assert missing.status is FitStatus.UNFITTED
    module.close()

    primary = _CountingPCA()
    fitted = DiskGraphDataModule(
        fixture.store_build.path,
        _strategy(),
        active_split_tag="primary",
        train_shuffle=False,
        fitted_transform=primary,
        fitted_state_root=tmp_path / "isolated",
    )
    fitted.setup("fit")
    fitted.close()

    diagnostic = _CountingPCA()
    with pytest.raises(ValueError, match="active split tag"):
        DiskGraphDataModule(
            fixture.store_build.path,
            _strategy(),
            active_split_tag="diagnostic",
            train_shuffle=False,
            fitted_transform=diagnostic,
            fitted_state_root=tmp_path / "isolated",
        )
    assert diagnostic.status is FitStatus.UNFITTED


def test_heterogeneous_fit_requires_explicit_target_before_publication(
    task8_stores: dict[str, QualifiedStoreFixture],
    tmp_path: Path,
) -> None:
    transform = IncrementalPCATransform(
        n_components=1,
        max_batch_rows=2,
        max_batch_bytes=16,
        target_node_type=None,
        input_dtype="float32",
        output_dtype="float32",
        accumulation_dtype="float64",
    )
    state_root = tmp_path / "missing-heterogeneous-target"
    module = DiskGraphDataModule(
        task8_stores["heterogeneous"].store_build.path,
        _strategy(),
        active_split_tag="primary",
        train_shuffle=False,
        fitted_transform=transform,
        fitted_state_root=state_root,
    )

    with pytest.raises(
        ValueError,
        match="heterogeneous.*explicit.*target_node_type",
    ):
        module.setup("fit")
    assert transform.status is FitStatus.UNFITTED
    assert not state_root.exists()
    module.close()


def test_fitted_instance_rejects_cross_store_context_misuse(
    task8_stores: dict[str, QualifiedStoreFixture],
    tmp_path: Path,
) -> None:
    fixture = task8_stores["heterogeneous"]
    with TypedGraphStore.open(fixture.store_build.path) as store:
        first_data = materialized_heterogeneous_reference(store)
    second_data = first_data.clone()
    second_data["paper"].x = second_data["paper"].x + 1
    transform = _CountingPCA()
    first = DiskGraphDataModule(
        first_data,
        HeterogeneousNeighborStrategy(batch_size=1, num_neighbors=[-1]),
        active_split_tag="primary",
        train_shuffle=False,
        fitted_transform=transform,
        fitted_state_root=tmp_path / "cross-store",
    )
    first.setup("fit")
    first.close()

    second = DiskGraphDataModule(
        second_data,
        HeterogeneousNeighborStrategy(batch_size=1, num_neighbors=[-1]),
        active_split_tag="primary",
        train_shuffle=False,
        fitted_transform=transform,
        fitted_state_root=tmp_path / "cross-store",
    )
    with pytest.raises(FitStateError, match="context|identity|store|content"):
        second.setup("fit")
    second.close()


def test_datamodule_rejects_transform_missing_declared_state_binding(
    task8_stores: dict[str, QualifiedStoreFixture],
    tmp_path: Path,
) -> None:
    with pytest.raises(TypeError, match="fitted_transform.*FittableTransform"):
        DiskGraphDataModule(
            task8_stores["heterogeneous"].store_build.path,
            _strategy(),
            active_split_tag="primary",
            train_shuffle=False,
            fitted_transform=_MissingStateBindingPCA(),
            fitted_state_root=tmp_path / "missing-binding",
        )


@pytest.mark.parametrize(
    ("transform_type", "message"),
    (
        (_InPlaceProtectedMutationPCA, "in-place|protected|version"),
        (_ParticipantMutationPCA, "participant_counts|metadata|protected"),
    ),
)
def test_post_assembly_guard_rejects_protected_mutation(
    task8_stores: dict[str, QualifiedStoreFixture],
    tmp_path: Path,
    transform_type: type[_CountingPCA],
    message: str,
) -> None:
    module = DiskGraphDataModule(
        task8_stores["heterogeneous"].store_build.path,
        _strategy(),
        active_split_tag="primary",
        train_shuffle=False,
        fitted_transform=transform_type(),
        fitted_state_root=tmp_path / transform_type.__name__,
    )
    module.setup("fit")
    with pytest.raises(RuntimeError, match=message):
        next(iter(module.train_dataloader()))
    module.close()
