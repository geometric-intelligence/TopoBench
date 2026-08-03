"""Bounded deterministic incremental PCA for native PyG feature fields."""

from __future__ import annotations

from collections.abc import Mapping
from importlib.metadata import PackageNotFoundError, version
from numbers import Integral
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import IncrementalPCA
from torch_geometric.data import Data, HeteroData

from topobench.transforms.fittable import (
    FitContext,
    FitStateError,
    FitStatePublisher,
    FitStatus,
    NativeGraph,
    PublishedFitState,
    TransformSpec,
    build_fit_state_key,
    derive_fit_chunk_schedule,
)


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be positive")
    return result


def _dtype(value: object, name: str) -> np.dtype:
    try:
        dtype = np.dtype(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a NumPy dtype") from error
    if dtype.kind != "f":
        raise TypeError(f"{name} must be a floating-point dtype")
    return dtype


def _version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


class IncrementalPCATransform:
    """Fit sklearn-compatible PCA in bounded row chunks and apply immutable state."""

    def __init__(
        self,
        *,
        n_components: int,
        max_batch_rows: int,
        max_batch_bytes: int,
        target_node_type: str | None,
        target_field: str = "x",
        input_dtype: str = "float32",
        output_dtype: str = "float32",
        accumulation_dtype: str = "float64",
        whiten: bool = False,
    ) -> None:
        self.n_components = _positive_integer(n_components, "n_components")
        self.max_batch_rows = _positive_integer(
            max_batch_rows, "max_batch_rows"
        )
        self.max_batch_bytes = _positive_integer(
            max_batch_bytes, "max_batch_bytes"
        )
        if self.max_batch_rows < self.n_components:
            raise ValueError("max_batch_rows must be at least n_components")
        if target_node_type is not None and (
            not isinstance(target_node_type, str) or not target_node_type
        ):
            raise ValueError(
                "target_node_type must be non-empty when provided"
            )
        if not isinstance(target_field, str) or not target_field:
            raise ValueError("target_field must be non-empty")
        if type(whiten) is not bool:
            raise TypeError("whiten must be bool")
        self.target_node_type = target_node_type
        self.target_field = target_field
        self.input_dtype = _dtype(input_dtype, "input_dtype")
        self.output_dtype = _dtype(output_dtype, "output_dtype")
        self.accumulation_dtype = _dtype(
            accumulation_dtype, "accumulation_dtype"
        )
        self.whiten = whiten
        self.spec = TransformSpec(
            input_kinds=("Data", "HeteroData"),
            output_kind="same",
            deterministic=True,
            device="cpu",
            preserves_node_identity=True,
            preserves_supervision=True,
            feature_width_behavior=f"fixed:{self.n_components}",
            edge_effects="none",
            accesses_labels=False,
            target_node_type=target_node_type,
            target_field=target_field,
            input_dtype=self.input_dtype.name,
            output_dtype=self.output_dtype.name,
            accumulation_dtype=self.accumulation_dtype.name,
        )
        self.status = FitStatus.UNFITTED
        self._context: FitContext | None = None
        self._model: IncrementalPCA | None = None
        self._pending: np.ndarray | None = None
        self._sample_count = 0
        self._state: PublishedFitState | None = None
        self._state_root: Path | None = None

    def canonical_config(self) -> Mapping[str, object]:
        return {
            "accumulation_dtype": self.accumulation_dtype.name,
            "input_dtype": self.input_dtype.name,
            "max_batch_bytes": self.max_batch_bytes,
            "max_batch_rows": self.max_batch_rows,
            "n_components": self.n_components,
            "output_dtype": self.output_dtype.name,
            "target_field": self.target_field,
            "target_node_type": self.target_node_type,
            "variance_edge_convention": "single_sample_zero",
            "whiten": self.whiten,
        }

    def implementation_versions(self) -> Mapping[str, str]:
        return {
            "numpy": np.__version__,
            "scikit-learn": _version("scikit-learn"),
            "torch": str(torch.__version__),
            "torch-geometric": _version("torch-geometric"),
            "topobench": _version("topobench"),
        }

    def __getstate__(self) -> dict[str, object]:
        if self.status is FitStatus.FITTED:
            self._validated_fitted_state()
        state = dict(self.__dict__)
        state.pop("begin_fit", None)
        state["_model"] = None
        state["_pending"] = None
        if self.status is FitStatus.FITTED:
            if self._context is None or self._state_root is None:
                raise RuntimeError(
                    "fitted worker state lacks immutable reload identity"
                )
            state["_state"] = None
            state["status"] = FitStatus.UNFITTED
        return state

    def __setstate__(self, state: dict[str, object]) -> None:
        self.__dict__.update(state)
        if self._context is None or self._state_root is None:
            return
        loaded = FitStatePublisher(self._state_root).load(
            build_fit_state_key(self._context, self),
            expected_metadata=self._metadata(self._context),
        )
        self._install(loaded, self._context)

    @property
    def state_key(self) -> str | None:
        if self.status is not FitStatus.FITTED:
            return None
        if self._state is None:
            raise FitStateError(
                "fitted transform has no published state binding"
            )
        return self._state.key

    @property
    def fitted_state(self) -> PublishedFitState:
        if self._state is None or self.status is not FitStatus.FITTED:
            raise RuntimeError(
                "fitted state is unavailable before valid finalize/load"
            )
        return self._state

    def _array(self, name: str) -> np.ndarray:
        return self.fitted_state.arrays[name]

    @property
    def mean_(self) -> np.ndarray:
        return self._array("mean")

    @property
    def components_(self) -> np.ndarray:
        return self._array("components")

    @property
    def explained_variance_(self) -> np.ndarray:
        return self._array("explained_variance")

    @property
    def explained_variance_ratio_(self) -> np.ndarray:
        return self._array("explained_variance_ratio")

    @property
    def singular_values_(self) -> np.ndarray:
        return self._array("singular_values")

    @property
    def sample_count_(self) -> int:
        return int(self.fitted_state.manifest["metadata"]["sample_count"])

    @property
    def input_width_(self) -> int:
        return int(self.fitted_state.manifest["metadata"]["input_width"])

    @property
    def output_width_(self) -> int:
        return int(self.fitted_state.manifest["metadata"]["output_width"])

    def _validated_fitted_state(self) -> PublishedFitState:
        if self._state is None or self._context is None:
            raise FitStateError(
                "fitted transform has no published state binding"
            )
        expected_key = build_fit_state_key(self._context, self)
        if expected_key != self._state.key:
            raise FitStateError(
                "fitted transform configuration no longer matches "
                "its published state identity"
            )
        return self._state

    def _require_unfitted(self) -> None:
        if self.status is FitStatus.FAILED:
            raise RuntimeError(
                "failed fitted-transform instances cannot be reused"
            )
        if self.status is not FitStatus.UNFITTED:
            raise RuntimeError(
                f"fit has already begun or finalized ({self.status.value})"
            )

    def begin_fit(self, context: FitContext) -> None:
        self._require_unfitted()
        if context.target_field != self.target_field:
            raise ValueError(
                "fit context target field does not match transform"
            )
        if (
            self.target_node_type is not None
            and context.target_node_type != self.target_node_type
        ):
            raise ValueError(
                "fit context target node type does not match transform"
            )
        if self.n_components > context.input_width:
            raise ValueError("n_components cannot exceed input feature width")
        if np.dtype(context.input_dtype) != self.input_dtype:
            raise TypeError(
                "fit context input dtype does not match transform dtype"
            )
        if context.numeric_precision != self.accumulation_dtype.name:
            raise ValueError(
                "fit context numeric precision does not match transform"
            )
        schedule = derive_fit_chunk_schedule(
            input_width=context.input_width,
            input_dtype=self.input_dtype,
            accumulation_dtype=self.accumulation_dtype,
            max_batch_rows=self.max_batch_rows,
            max_batch_bytes=self.max_batch_bytes,
            sample_count=context.input_shape[0],
        )
        if schedule.chunk_rows < self.n_components:
            raise ValueError(
                "byte-derived fit chunk cannot hold the n_components "
                "bootstrap rows"
            )
        self._context = context
        self._model = IncrementalPCA(
            n_components=self.n_components, whiten=self.whiten, copy=True
        )
        self._pending = np.empty(
            (0, context.input_width), dtype=self.accumulation_dtype
        )
        self._sample_count = 0
        self.status = FitStatus.FITTING

    def _require_fitting(
        self,
    ) -> tuple[FitContext, IncrementalPCA, np.ndarray]:
        if self.status is FitStatus.FAILED:
            raise RuntimeError(
                "failed fitted-transform instances cannot be reused"
            )
        if self.status is not FitStatus.FITTING:
            raise RuntimeError("fit is not active or has already finalized")
        assert (
            self._context is not None
            and self._model is not None
            and self._pending is not None
        )
        return self._context, self._model, self._pending

    @staticmethod
    def _partial_fit(model: IncrementalPCA, features: np.ndarray) -> None:
        with np.errstate(divide="ignore", invalid="ignore"):
            model.partial_fit(features)

    def update_fit(
        self, features: np.ndarray, labels: np.ndarray | None = None
    ) -> None:
        context, model, pending = self._require_fitting()
        if labels is not None:
            raise PermissionError(
                "incremental PCA does not declare label access"
            )
        value = np.asarray(features)
        if value.ndim != 2:
            raise ValueError("fit features must be a two-dimensional array")
        if value.shape[0] < 1:
            raise ValueError("fit update must contain at least one row")
        if value.shape[0] > self.max_batch_rows:
            raise ValueError("fit update exceeds configured row bound")
        if value.nbytes > self.max_batch_bytes:
            raise ValueError("fit update exceeds configured byte bound")
        if value.shape[1] != context.input_width:
            raise ValueError("fit update feature width is incompatible")
        if value.dtype != self.input_dtype:
            raise TypeError(
                f"fit update dtype {value.dtype.name} does not match {self.input_dtype.name}"
            )
        accumulation_bytes = (
            value.shape[0] * value.shape[1] * self.accumulation_dtype.itemsize
        )
        if accumulation_bytes > self.max_batch_bytes:
            raise ValueError(
                "fit update exceeds configured accumulation byte bound"
            )
        if not np.isfinite(value).all():
            raise ValueError("fit update features must be finite")
        accumulated = np.asarray(value, dtype=self.accumulation_dtype)
        offset = 0
        if not hasattr(model, "components_") and len(pending):
            needed = self.n_components - len(pending)
            taken = min(needed, len(accumulated))
            pending = np.concatenate(
                (pending, accumulated[:taken]),
                axis=0,
            )
            offset = taken
            if len(pending) == self.n_components:
                self._partial_fit(model, pending)
                pending = np.empty(
                    (0, context.input_width),
                    dtype=self.accumulation_dtype,
                )
        if not hasattr(model, "components_"):
            remaining = len(accumulated) - offset
            if remaining < self.n_components:
                self._pending = (
                    pending
                    if len(pending)
                    else np.array(
                        accumulated[offset:],
                        copy=True,
                        order="C",
                    )
                )
                self._sample_count += len(value)
                return
            feed_rows = min(self.max_batch_rows, remaining)
            self._partial_fit(model, accumulated[offset : offset + feed_rows])
            offset += feed_rows
        while offset < len(accumulated):
            feed_rows = min(self.max_batch_rows, len(accumulated) - offset)
            self._partial_fit(model, accumulated[offset : offset + feed_rows])
            offset += feed_rows
        self._pending = pending
        self._sample_count += len(value)

    def _metadata(self, context: FitContext) -> dict[str, object]:
        return {
            "accumulation_dtype": self.accumulation_dtype.name,
            "input_dtype": self.input_dtype.name,
            "input_width": context.input_width,
            "n_components": self.n_components,
            "output_dtype": self.output_dtype.name,
            "output_width": self.n_components,
            "sample_count": context.input_shape[0],
            "target_field": context.target_field,
            "target_node_type": context.target_node_type,
            "variance_edge_convention": "single_sample_zero",
            "whiten": self.whiten,
        }

    def _fail(self) -> None:
        self.status = FitStatus.FAILED
        self._model = None
        self._pending = None

    def finalize_fit(self, state_root: str | Path) -> PublishedFitState:
        context, model, pending = self._require_fitting()
        try:
            if self._sample_count == 0:
                raise ValueError(
                    "cannot fit incremental PCA on empty training input"
                )
            if self._sample_count != context.input_shape[0]:
                raise ValueError(
                    "fit sample count does not match canonical training context"
                )
            if self._sample_count < self.n_components:
                raise ValueError(
                    "incremental PCA requires at least n_components samples"
                )
            if self._sample_count == 1 and self.whiten:
                raise ValueError(
                    "whitening is undefined for a single-sample PCA fit"
                )
            if not hasattr(model, "components_") or len(pending):
                raise ValueError("insufficient final samples for n_components")
            arrays = {
                "components": np.asarray(
                    model.components_, dtype=self.accumulation_dtype
                ),
                "explained_variance": np.asarray(
                    model.explained_variance_, dtype=self.accumulation_dtype
                ),
                "explained_variance_ratio": np.asarray(
                    model.explained_variance_ratio_,
                    dtype=self.accumulation_dtype,
                ),
                "mean": np.asarray(model.mean_, dtype=self.accumulation_dtype),
                "singular_values": np.asarray(
                    model.singular_values_, dtype=self.accumulation_dtype
                ),
            }
            if self._sample_count == 1:
                arrays["explained_variance"] = np.zeros(
                    self.n_components,
                    dtype=self.accumulation_dtype,
                )
                arrays["explained_variance_ratio"] = np.zeros(
                    self.n_components,
                    dtype=self.accumulation_dtype,
                )
                arrays["singular_values"] = np.zeros(
                    self.n_components,
                    dtype=self.accumulation_dtype,
                )
            state = FitStatePublisher(state_root).publish(
                build_fit_state_key(context, self),
                metadata=self._metadata(context),
                arrays=arrays,
            )
            self._state_root = Path(state_root)
            self._install(state, context)
            return state
        except BaseException:
            self._fail()
            raise
        finally:
            self._model = None
            self._pending = None

    def load_state(
        self, state_root: str | Path, context: FitContext
    ) -> PublishedFitState:
        self._require_unfitted()
        if context.target_field != self.target_field or (
            self.target_node_type is not None
            and context.target_node_type != self.target_node_type
        ):
            raise FitStateError(
                "fitted state context target identity mismatch"
            )
        if np.dtype(context.input_dtype) != self.input_dtype:
            raise FitStateError("fitted state context dtype mismatch")
        state = FitStatePublisher(state_root).load(
            build_fit_state_key(context, self),
            expected_metadata=self._metadata(context),
        )
        self._state_root = Path(state_root)
        self._install(state, context)
        return state

    def _install(self, state: PublishedFitState, context: FitContext) -> None:
        arrays = state.arrays
        expected_shapes = {
            "components": (self.n_components, context.input_width),
            "explained_variance": (self.n_components,),
            "explained_variance_ratio": (self.n_components,),
            "mean": (context.input_width,),
            "singular_values": (self.n_components,),
        }
        if set(arrays) != set(expected_shapes):
            raise FitStateError("incremental PCA state array set is invalid")
        for name, shape in expected_shapes.items():
            value = arrays[name]
            if value.shape != shape or value.dtype != self.accumulation_dtype:
                raise FitStateError(
                    f"incremental PCA state {name!r} shape/dtype is invalid"
                )
            if not np.isfinite(value).all():
                raise FitStateError(
                    f"incremental PCA state {name!r} must be finite"
                )
        tolerance = np.finfo(self.accumulation_dtype).eps * max(
            128, context.input_width * 16
        )
        gram = arrays["components"] @ arrays["components"].T
        if not np.allclose(
            gram,
            np.eye(self.n_components, dtype=self.accumulation_dtype),
            rtol=tolerance,
            atol=tolerance,
        ):
            raise FitStateError(
                "incremental PCA components are not orthonormal"
            )
        if np.any(arrays["explained_variance"] < 0) or np.any(
            arrays["explained_variance_ratio"] < 0
        ):
            raise FitStateError("incremental PCA variance state is invalid")
        if np.any(arrays["singular_values"] < 0):
            raise FitStateError("incremental PCA singular values are invalid")
        if self.whiten and np.any(arrays["explained_variance"] <= 0):
            raise FitStateError(
                "whitened incremental PCA requires positive variance"
            )
        self._context = context
        self._state = state
        self.status = FitStatus.FITTED

    def transform(self, batch: NativeGraph) -> NativeGraph:
        if self.status is FitStatus.FAILED:
            raise RuntimeError(
                "failed fitted-transform instances cannot be reused"
            )
        if self.status is not FitStatus.FITTED:
            raise RuntimeError(
                "transform requires valid finalized fitted state"
            )
        self._validated_fitted_state()
        if isinstance(batch, HeteroData):
            if self.target_node_type is None:
                raise ValueError(
                    "HeteroData transform requires target_node_type"
                )
            if self.target_node_type not in batch.node_types:
                raise ValueError(
                    f"batch does not contain target node type {self.target_node_type!r}"
                )
            storage = batch[self.target_node_type]
        elif isinstance(batch, Data):
            storage = batch
        else:
            raise TypeError(
                "incremental PCA accepts native Data or HeteroData"
            )
        if self.target_field not in storage:
            raise ValueError(
                f"batch does not contain feature field {self.target_field!r}"
            )
        tensor = storage[self.target_field]
        if not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu":
            raise TypeError(
                "incremental PCA requires a CPU tensor feature field"
            )
        if tensor.ndim != 2 or tensor.shape[1] != self.input_width_:
            raise ValueError(
                "batch feature width is incompatible with fitted state"
            )
        expected_dtype = torch.from_numpy(
            np.empty((), dtype=self.input_dtype)
        ).dtype
        if tensor.dtype != expected_dtype:
            raise TypeError(
                "batch feature dtype is incompatible with fitted state"
            )
        centered = (
            np.asarray(tensor.detach().numpy(), dtype=self.accumulation_dtype)
            - self.mean_
        )
        projected = centered @ self.components_.T
        if self.whiten:
            projected = projected / np.sqrt(self.explained_variance_)
        output_value = torch.from_numpy(
            np.array(projected, dtype=self.output_dtype, copy=True, order="C")
        )
        output = batch.clone()
        if isinstance(output, HeteroData):
            assert self.target_node_type is not None
            output[self.target_node_type][self.target_field] = output_value
        else:
            output[self.target_field] = output_value
        return output


__all__ = ["IncrementalPCATransform"]
