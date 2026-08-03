"""Bounded deterministic incremental-PCA fitting and application tests."""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path

import numpy as np
import pytest
import torch
from sklearn.decomposition import IncrementalPCA
from torch_geometric.data import Data, HeteroData

from topobench.transforms.fittable import (
    FitContext,
    FitStateError,
    FitStatePublisher,
    FitStatus,
    build_fit_state_key,
    derive_fit_chunk_schedule,
)
from topobench.transforms.incremental_pca import IncrementalPCATransform

_X = np.array(
    [
        [1.0, 2.0, 0.0],
        [2.0, 1.0, 1.0],
        [3.0, 4.0, 1.0],
        [4.0, 3.0, 2.0],
        [5.0, 6.0, 2.0],
        [6.0, 5.0, 3.0],
    ],
    dtype=np.float64,
)
_EXPECTED_MEAN = np.array([3.5, 3.5, 1.5], dtype=np.float64)
_EXPECTED_COMPONENTS = np.array(
    [
        [0.67863293, 0.64725284, 0.34716149],
        [-0.43966293, 0.73661538, -0.51390105],
    ],
    dtype=np.float64,
)
_EXPECTED_VARIANCE = np.array([7.23786763, 0.86213237], dtype=np.float64)
_EXPECTED_RATIO = np.array([0.89356390, 0.10643610], dtype=np.float64)
_EXPECTED_SINGULAR = np.array([6.01575749, 2.07621335], dtype=np.float64)
_EXPECTED_PROJECTION = np.array(
    [
        [-3.18820383, 0.76508583],
        [-2.80966225, -0.92509353],
        [-0.18927079, 0.84508968],
        [0.18927079, -0.84508968],
        [2.80966225, 0.92509353],
        [3.18820383, -0.76508583],
    ],
    dtype=np.float64,
)


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _context(
    *, dtype: str = "float64", width: int = 3, rows: int = 6
) -> FitContext:
    ids = np.arange(rows, dtype=np.int64)
    return FitContext(
        content_sha256="a" * 64,
        active_split_tag="primary",
        train_ids_sha256=_sha(ids.tobytes()),
        train_source_sha256=_sha(_X[:rows].tobytes()),
        target_node_type="node",
        target_field="x",
        input_shape=(rows, width),
        input_width=width,
        input_dtype=dtype,
        input_schema_sha256="b" * 64,
        package_versions=(("fixture", "1"),),
        numeric_precision="float64",
    )


def _transform(
    *, whiten: bool = False, output_dtype: str = "float64"
) -> IncrementalPCATransform:
    return IncrementalPCATransform(
        n_components=2,
        max_batch_rows=2,
        max_batch_bytes=48,
        target_node_type="node",
        target_field="x",
        input_dtype="float64",
        output_dtype=output_dtype,
        accumulation_dtype="float64",
        whiten=whiten,
    )


def _fit(transform: IncrementalPCATransform, root: Path) -> None:
    transform.begin_fit(_context())
    for batch in np.split(_X, 3):
        transform.update_fit(batch)
    transform.finalize_fit(root)


def test_incremental_pca_matches_declared_numeric_fixture_and_explicit_state(
    tmp_path: Path,
) -> None:
    transform = _transform()
    _fit(transform, tmp_path)
    tolerance = np.finfo(np.float64).eps * 10**8

    assert transform.status is FitStatus.FITTED
    assert transform.sample_count_ == 6
    assert transform.input_width_ == 3
    assert transform.output_width_ == 2
    np.testing.assert_allclose(
        transform.mean_, _EXPECTED_MEAN, rtol=tolerance, atol=tolerance
    )
    np.testing.assert_allclose(
        transform.components_,
        _EXPECTED_COMPONENTS,
        rtol=tolerance,
        atol=tolerance,
    )
    np.testing.assert_allclose(
        transform.explained_variance_,
        _EXPECTED_VARIANCE,
        rtol=tolerance,
        atol=tolerance,
    )
    np.testing.assert_allclose(
        transform.explained_variance_ratio_,
        _EXPECTED_RATIO,
        rtol=tolerance,
        atol=tolerance,
    )
    np.testing.assert_allclose(
        transform.singular_values_,
        _EXPECTED_SINGULAR,
        rtol=tolerance,
        atol=tolerance,
    )

    manifest = transform.fitted_state.manifest
    assert manifest["metadata"] == {
        "accumulation_dtype": "float64",
        "input_dtype": "float64",
        "input_width": 3,
        "n_components": 2,
        "output_dtype": "float64",
        "output_width": 2,
        "sample_count": 6,
        "target_field": "x",
        "target_node_type": "node",
        "variance_edge_convention": "single_sample_zero",
        "whiten": False,
    }
    assert set(manifest["arrays"]) == {
        "components",
        "explained_variance",
        "explained_variance_ratio",
        "mean",
        "singular_values",
    }


def test_transform_clones_native_batch_and_preserves_identity_supervision_edges(
    tmp_path: Path,
) -> None:
    transform = _transform(output_dtype="float32")
    _fit(transform, tmp_path)
    source = HeteroData()
    source["node"].x = torch.from_numpy(_X.copy())
    source["node"].n_id = torch.arange(6)
    source["node"].y = torch.arange(6)
    source["node"].train_mask = torch.tensor(
        [True, True, False, False, False, False]
    )
    edge_type = ("node", "links", "node")
    source[edge_type].edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
    original_x = source["node"].x.clone()

    output = transform.transform(source)
    tolerance = np.finfo(np.float32).eps * 128

    assert output is not source
    assert output["node"].x.data_ptr() != source["node"].x.data_ptr()
    assert output["node"].x.dtype is torch.float32
    assert tuple(output["node"].x.shape) == (6, 2)
    np.testing.assert_allclose(
        output["node"].x.numpy(),
        _EXPECTED_PROJECTION.astype(np.float32),
        rtol=tolerance,
        atol=tolerance,
    )
    assert torch.equal(source["node"].x, original_x)
    assert torch.equal(output["node"].n_id, source["node"].n_id)
    assert torch.equal(output["node"].y, source["node"].y)
    assert torch.equal(output["node"].train_mask, source["node"].train_mask)
    assert torch.equal(
        output[edge_type].edge_index, source[edge_type].edge_index
    )


def test_whitening_uses_sklearn_semantics_and_restart_reuses_identical_state(
    tmp_path: Path,
) -> None:
    fitted = _transform(whiten=True)
    _fit(fitted, tmp_path)
    restarted = _transform(whiten=True)
    restarted.load_state(tmp_path, _context())
    data = Data(x=torch.from_numpy(_X.copy()))
    data.target_node_type = "node"
    expected = _EXPECTED_PROJECTION / np.sqrt(_EXPECTED_VARIANCE)
    tolerance = np.finfo(np.float64).eps * 10**8

    np.testing.assert_allclose(
        fitted.transform(data).x.numpy(),
        expected,
        rtol=tolerance,
        atol=tolerance,
    )
    np.testing.assert_array_equal(
        restarted.transform(data).x.numpy(), fitted.transform(data).x.numpy()
    )
    assert restarted.fitted_state.key == fitted.fitted_state.key


def test_post_fit_configuration_mutation_invalidates_published_identity(
    tmp_path: Path,
) -> None:
    fitted = _transform()
    _fit(fitted, tmp_path)
    published_key = fitted.state_key
    assert published_key is not None
    data = Data(x=torch.from_numpy(_X.copy()))
    data.target_node_type = "node"

    fitted.whiten = True

    assert build_fit_state_key(_context(), fitted) != published_key
    assert fitted.state_key == published_key
    with pytest.raises(FitStateError, match="configuration|identity"):
        fitted.transform(data)
    with pytest.raises(FitStateError, match="configuration|identity"):
        pickle.dumps(fitted)

    restarted = _transform()
    restarted.load_state(tmp_path, _context())
    assert restarted.state_key == published_key
    np.testing.assert_allclose(
        restarted.transform(data).x.numpy(),
        _EXPECTED_PROJECTION,
        rtol=np.finfo(np.float64).eps * 10**8,
        atol=np.finfo(np.float64).eps * 10**8,
    )


def test_lifecycle_rejects_invalid_order_bounds_width_dtype_and_nonfinite(
    tmp_path: Path,
) -> None:
    transform = _transform()
    data = Data(x=torch.from_numpy(_X.copy()))
    data.target_node_type = "node"
    with pytest.raises(RuntimeError, match="finalized|fitted"):
        transform.transform(data)
    transform.begin_fit(_context())
    with pytest.raises(RuntimeError, match="already|fitting"):
        transform.begin_fit(_context())
    with pytest.raises(ValueError, match="row bound"):
        transform.update_fit(_X[:3])
    with pytest.raises(ValueError, match="byte bound"):
        transform.update_fit(np.ones((2, 3), dtype=np.complex128))
    with pytest.raises(ValueError, match="width"):
        transform.update_fit(np.ones((2, 2), dtype=np.float64))
    with pytest.raises(TypeError, match="dtype"):
        transform.update_fit(np.ones((2, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="finite"):
        invalid = _X[:2].copy()
        invalid[0, 0] = np.nan
        transform.update_fit(invalid)
    for batch in np.split(_X, 3):
        transform.update_fit(batch)
    transform.finalize_fit(tmp_path)
    with pytest.raises(RuntimeError, match="finalized|fitted"):
        transform.finalize_fit(tmp_path)
    with pytest.raises(RuntimeError, match="finalized|fitted"):
        transform.update_fit(_X[:2])


def test_empty_insufficient_and_failed_publication_are_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    empty = _transform()
    empty.begin_fit(_context(rows=0))
    with pytest.raises(ValueError, match="empty|samples"):
        empty.finalize_fit(tmp_path / "empty")
    assert empty.status is FitStatus.FAILED
    with pytest.raises(RuntimeError, match="failed"):
        empty.begin_fit(_context())

    insufficient = _transform()
    insufficient.begin_fit(_context(rows=1))
    insufficient.update_fit(_X[:1])
    with pytest.raises(ValueError, match="n_components|samples"):
        insufficient.finalize_fit(tmp_path / "insufficient")
    assert insufficient.status is FitStatus.FAILED

    interrupted = _transform()
    interrupted.begin_fit(_context())
    for batch in np.split(_X, 3):
        interrupted.update_fit(batch)

    def fail_publish(*args: object, **kwargs: object) -> object:
        raise OSError("interrupted publication")

    monkeypatch.setattr(FitStatePublisher, "publish", fail_publish)
    with pytest.raises(OSError, match="interrupted"):
        interrupted.finalize_fit(tmp_path / "interrupted")
    assert interrupted.status is FitStatus.FAILED
    with pytest.raises(RuntimeError, match="failed"):
        interrupted.transform(Data(x=torch.from_numpy(_X.copy())))
    assert not list((tmp_path / "interrupted").glob(".staging-*"))


def test_state_load_rejects_context_mismatch_and_semantic_array_corruption(
    tmp_path: Path,
) -> None:
    fitted = _transform()
    _fit(fitted, tmp_path)
    with pytest.raises(FitStateError, match="not found|identity|state"):
        _transform().load_state(
            tmp_path, replace_context := _context(dtype="float32")
        )
    assert replace_context.input_dtype == "float32"

    component_path = fitted.fitted_state.path / "components.npy"
    component_path.chmod(0o644)
    component = np.load(component_path, allow_pickle=False)
    component[0, 0] = np.inf
    with component_path.open("wb") as stream:
        np.save(stream, component, allow_pickle=False)
    with pytest.raises(FitStateError, match="checksum|finite|corrupt"):
        _transform().load_state(tmp_path, _context())


def test_internal_updates_and_accumulation_buffer_respect_declared_bounds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_rows: list[int] = []
    original_partial_fit = IncrementalPCA.partial_fit

    def observed_partial_fit(
        self: IncrementalPCA,
        features: np.ndarray,
        *args: object,
        **kwargs: object,
    ) -> IncrementalPCA:
        observed_rows.append(len(features))
        return original_partial_fit(self, features, *args, **kwargs)

    monkeypatch.setattr(IncrementalPCA, "partial_fit", observed_partial_fit)
    transform = _transform()
    transform.begin_fit(_context(rows=3))
    for row in _X[:3]:
        transform.update_fit(row.reshape(1, -1))
    transform.finalize_fit(tmp_path / "bounded")

    assert observed_rows == [2, 1]
    assert max(observed_rows) <= transform.max_batch_rows

    undersized_bytes = IncrementalPCATransform(
        n_components=2,
        max_batch_rows=2,
        max_batch_bytes=24,
        target_node_type="node",
        input_dtype="float32",
        output_dtype="float32",
        accumulation_dtype="float64",
    )
    with pytest.raises(ValueError, match="byte|accumulation"):
        undersized_bytes.begin_fit(_context(dtype="float32", rows=3))


def test_begin_fit_accepts_derived_bootstrap_chunk_below_max_row_bound(
    tmp_path: Path,
) -> None:
    transform = IncrementalPCATransform(
        n_components=2,
        max_batch_rows=8,
        max_batch_bytes=48,
        target_node_type="node",
        input_dtype="float64",
        output_dtype="float64",
        accumulation_dtype="float64",
    )
    context = _context()
    schedule = derive_fit_chunk_schedule(
        input_width=context.input_width,
        input_dtype=context.input_dtype,
        accumulation_dtype=transform.accumulation_dtype,
        max_batch_rows=transform.max_batch_rows,
        max_batch_bytes=transform.max_batch_bytes,
        sample_count=context.input_shape[0],
    )

    assert schedule.chunk_rows == transform.n_components
    assert schedule.chunk_rows < transform.max_batch_rows
    transform.begin_fit(context)
    for start in range(0, len(_X), schedule.chunk_rows):
        transform.update_fit(_X[start : start + schedule.chunk_rows])
    transform.finalize_fit(tmp_path)

    assert transform.status is FitStatus.FITTED
    assert transform.sample_count_ == len(_X)


def test_finalize_allows_samples_equal_components_with_sklearn_variance(
    tmp_path: Path,
) -> None:
    transform = _transform()
    transform.begin_fit(_context(rows=2))
    transform.update_fit(_X[:2])
    transform.finalize_fit(tmp_path / "equal")
    reference = IncrementalPCA(n_components=2).partial_fit(_X[:2])
    tolerance = np.finfo(np.float64).eps * 10**8

    assert transform.sample_count_ == transform.n_components == 2
    np.testing.assert_allclose(
        transform.explained_variance_,
        reference.explained_variance_,
        rtol=tolerance,
        atol=tolerance,
    )
    np.testing.assert_allclose(
        transform.explained_variance_ratio_,
        reference.explained_variance_ratio_,
        rtol=tolerance,
        atol=tolerance,
    )
