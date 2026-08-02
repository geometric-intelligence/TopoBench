"""Dependency-free raw parsers for native hypergraph datasets."""

from __future__ import annotations

import hashlib
import json
import zipfile
from collections import defaultdict
from collections.abc import Hashable, Iterable, Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse as scipy_sparse
import torch
from torch import Tensor

from topobench.data import (
    HYPERGRAPH_REPRESENTATION_VERSION,
    HypergraphData,
    validate_hypergraph_structure,
)

SAFE_HYPERGRAPH_FORMAT = "topobench.hypergraph.npz"
SAFE_HYPERGRAPH_FORMAT_VERSION = 1
SAFE_HYPERGRAPH_CONVERTER_VERSION = "topobench-hypergraph-converter-v1"
_SAFE_METADATA_KEYS = frozenset(
    {
        "arrays",
        "converter_version",
        "feature_storage",
        "format",
        "format_version",
        "incidence_roles",
        "npz_sha256",
        "num_hyperedges",
        "num_nodes",
        "padding_count",
        "padding_sentinel",
        "raw_sha256",
    }
)


def incidence_pairs(
    hyperedges: Mapping[Hashable, Iterable[int]],
    num_nodes: int,
) -> tuple[Tensor, int]:
    """Convert raw hyperedges into canonical node-to-hyperedge pairs."""
    if isinstance(num_nodes, bool) or not isinstance(num_nodes, Integral):
        raise TypeError("num_nodes must be an integer")
    num_nodes = int(num_nodes)
    if num_nodes < 0:
        raise ValueError("num_nodes must be nonnegative")

    ordered_ids = sorted(
        hyperedges,
        key=lambda value: (type(value).__name__, repr(value)),
    )
    hyperedge_id = {raw: index for index, raw in enumerate(ordered_ids)}
    pairs: list[tuple[int, int]] = []
    for raw in ordered_ids:
        nodes = tuple(hyperedges[raw])
        if not nodes:
            raise ValueError(f"empty hyperedge is unsupported: {raw!r}")
        for node in nodes:
            if isinstance(node, bool) or not isinstance(node, Integral):
                raise TypeError("hyperedge node IDs must be integers")
            node_id = int(node)
            if node_id < 0 or node_id >= num_nodes:
                raise ValueError("hyperedge contains an out-of-bounds node")
            pairs.append((node_id, hyperedge_id[raw]))

    if not pairs:
        return torch.empty((2, 0), dtype=torch.long), len(ordered_ids)
    index = torch.tensor(sorted(pairs), dtype=torch.long).t().contiguous()
    return index, len(ordered_ids)


def _resolve_data_dir(data_dir: str | Path, data_name: str) -> Path:
    """Resolve assets that either retain or flatten their dataset folder."""
    root = Path(data_dir)
    nested = root / data_name
    if nested.is_dir():
        return nested
    return root


_DEFAULT_MAX_DENSE_FEATURE_BYTES = 512 * 1024**2
_DEFAULT_MAX_ARRAY_MEMBER_BYTES = 512 * 1024**2
_DEFAULT_MAX_TOTAL_ARRAY_BYTES = 1024 * 1024**2
_FINITE_VALIDATION_CHUNK_ELEMENTS = 1_048_576
_FLOAT32_ELEMENT_BYTES = 4


def _validate_dense_byte_limit(max_dense_bytes: int) -> int:
    """Return a positive dense allocation ceiling without boolean coercion."""
    if isinstance(max_dense_bytes, bool) or not isinstance(
        max_dense_bytes, Integral
    ):
        raise TypeError("max_dense_bytes must be an integer")
    limit = int(max_dense_bytes)
    if limit <= 0:
        raise ValueError("max_dense_bytes must be positive")
    return limit


def _dense_allocation_estimate(shape: tuple[int, ...]) -> int:
    """Calculate float32 output bytes with Python's overflow-safe integers."""
    elements = 1
    for dimension in shape:
        elements *= dimension
    return elements * _FLOAT32_ELEMENT_BYTES


def _preflight_dense_features(
    *,
    shape: tuple[int, ...],
    dtype: object,
    max_dense_bytes: int,
) -> None:
    """Reject a dense float32 conversion before allocating its output."""
    estimate = _dense_allocation_estimate(shape)
    if estimate > max_dense_bytes:
        raise ValueError(
            "dense feature allocation rejected before allocation: "
            f"shape={shape}, dtype={dtype}, estimate={estimate} bytes, "
            f"limit={max_dense_bytes} bytes"
        )


def _numpy_values_are_finite(array: np.ndarray) -> bool:
    """Check finiteness with temporary storage bounded independently of shape."""
    iterator = np.nditer(
        array,
        flags=["external_loop", "buffered", "zerosize_ok"],
        op_flags=["readonly"],
        buffersize=_FINITE_VALIDATION_CHUNK_ELEMENTS,
    )
    return all(bool(np.isfinite(chunk).all()) for chunk in iterator)


def _validate_sparse_coo_tensor(features: Tensor) -> Tensor:
    """Validate and canonicalize one fully sparse floating COO tensor."""
    if features.layout != torch.sparse_coo:
        raise TypeError(
            "sparse feature tensors must use torch sparse COO layout"
        )
    if features.ndim != 2 or features.sparse_dim() != 2:
        raise ValueError("sparse feature tensors must be fully sparse rank-2")
    if features.dense_dim() != 0:
        raise TypeError("sparse feature tensors must not have dense dimensions")
    if not features.is_floating_point():
        raise TypeError("sparse feature values must have a floating dtype")

    indices = features._indices()
    if indices.dtype != torch.long:
        raise TypeError("sparse feature indices must use torch.int64")
    if indices.numel() and (
        bool((indices < 0).any())
        or bool((indices[0] >= features.size(0)).any())
        or bool((indices[1] >= features.size(1)).any())
    ):
        raise ValueError(
            "sparse feature indices must be within declared shape bounds"
        )
    coalesced = features.coalesce()
    if not bool(torch.isfinite(coalesced.values()).all()):
        raise ValueError("sparse feature values must contain only finite values")

    converted = coalesced.to(dtype=torch.float32).coalesce()
    if not bool(torch.isfinite(converted.values()).all()):
        raise ValueError(
            "sparse feature values must remain finite in torch.float32"
        )
    return converted


def _csr_feature_tensor(features: Any) -> Tensor:
    """Convert validated scipy CSR storage using work proportional to NNZ."""
    if getattr(features, "format", None) != "csr":
        raise TypeError("scipy sparse features must use scipy CSR layout")
    if getattr(features, "ndim", None) != 2:
        raise ValueError("scipy CSR features must be rank-2")

    shape = tuple(int(dimension) for dimension in features.shape)
    values = np.asarray(features.data)
    column_indices = np.asarray(features.indices)
    indptr = np.asarray(features.indptr)
    if not np.issubdtype(values.dtype, np.floating):
        raise TypeError("scipy CSR feature values must have a floating dtype")
    if not np.issubdtype(column_indices.dtype, np.integer) or not np.issubdtype(
        indptr.dtype, np.integer
    ):
        raise TypeError("scipy CSR indices must have integer dtypes")
    if values.ndim != 1 or column_indices.ndim != 1 or indptr.ndim != 1:
        raise ValueError("scipy CSR storage arrays must be rank-1")
    if indptr.size != shape[0] + 1:
        raise ValueError("scipy CSR indptr length does not match row count")
    if indptr.size == 0 or int(indptr[0]) != 0:
        raise ValueError("scipy CSR indptr must begin at zero")
    if bool((np.diff(indptr) < 0).any()):
        raise ValueError("scipy CSR indptr must be nondecreasing")
    if int(indptr[-1]) != values.size or column_indices.size != values.size:
        raise ValueError("scipy CSR storage lengths are inconsistent")
    if column_indices.size and (
        bool((column_indices < 0).any())
        or bool((column_indices >= shape[1]).any())
    ):
        raise ValueError(
            "scipy CSR column indices must be within declared shape bounds"
        )
    if not _numpy_values_are_finite(values):
        raise ValueError("scipy CSR feature values must contain only finite values")

    positions = np.arange(values.size, dtype=np.int64)
    row_indices = np.searchsorted(indptr, positions, side="right") - 1
    indices = torch.from_numpy(
        np.stack(
            (
                row_indices.astype(np.int64, copy=False),
                column_indices.astype(np.int64, copy=False),
            ),
            axis=0,
        )
    )
    tensor_values = torch.from_numpy(
        values.astype(np.float32, copy=True)
    )
    tensor = torch.sparse_coo_tensor(
        indices,
        tensor_values,
        size=shape,
        dtype=torch.float32,
    )
    return _validate_sparse_coo_tensor(tensor)


def _as_feature_tensor(
    features: Any,
    *,
    max_dense_bytes: int = _DEFAULT_MAX_DENSE_FEATURE_BYTES,
) -> Tensor:
    """Convert finite dense or CSR features without shape-sized sparse work."""
    limit = _validate_dense_byte_limit(max_dense_bytes)
    if scipy_sparse.issparse(features):
        return _csr_feature_tensor(features)
    if isinstance(features, Tensor) and features.layout != torch.strided:
        return _validate_sparse_coo_tensor(features)

    if isinstance(features, Tensor):
        if features.ndim != 2:
            raise ValueError("dense features must be rank-2")
        if not features.is_floating_point():
            raise TypeError("dense features must have a floating dtype")
        shape = tuple(int(dimension) for dimension in features.shape)
        _preflight_dense_features(
            shape=shape,
            dtype=features.dtype,
            max_dense_bytes=limit,
        )
        if not bool(torch.isfinite(features).all()):
            raise ValueError("dense features must contain only finite values")
        converted = features.to(dtype=torch.float32)
        if not bool(torch.isfinite(converted).all()):
            raise ValueError(
                "dense features must remain finite in torch.float32"
            )
        return converted

    array = np.asarray(features)
    if array.ndim != 2:
        raise ValueError("dense features must be rank-2")
    if not np.issubdtype(array.dtype, np.floating):
        raise TypeError("dense features must have a floating dtype")
    shape = tuple(int(dimension) for dimension in array.shape)
    _preflight_dense_features(
        shape=shape,
        dtype=array.dtype,
        max_dense_bytes=limit,
    )
    if not _numpy_values_are_finite(array):
        raise ValueError("dense features must contain only finite values")
    tensor = torch.as_tensor(array, dtype=torch.float32)
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError("dense features must remain finite in torch.float32")
    return tensor


def _reject_json_constant(value: str) -> None:
    """Reject non-standard JSON numeric constants."""
    raise ValueError(f"metadata contains non-JSON numeric constant {value!r}")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Build one JSON object while rejecting duplicate member names."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"metadata contains duplicate key {key!r}")
        result[key] = value
    return result


def _canonical_json_bytes(value: Any) -> bytes:
    """Encode primitive metadata into the one accepted canonical form."""
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _read_safe_metadata(path: Path) -> dict[str, Any]:
    """Read canonical JSON without extensions, duplicates, or object hooks."""
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
        value = json.loads(
            text,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid safe hypergraph metadata: {path.name}") from error
    if not isinstance(value, dict):
        raise TypeError("safe hypergraph metadata must be a JSON object")
    if raw != _canonical_json_bytes(value):
        raise ValueError("safe hypergraph metadata must use canonical JSON")
    return value


def _sha256_file(path: Path) -> str:
    """Hash one payload without a shape-sized in-memory copy."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    """Return whether a value is one lowercase hexadecimal SHA-256 digest."""
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _strict_positive_int(value: Any, field_name: str) -> int:
    """Validate a positive built-in JSON integer."""
    if type(value) is not int:
        raise TypeError(f"{field_name} must be a JSON integer")
    if value <= 0:
        raise ValueError(f"{field_name} must be positive")
    return value


def _array_descriptor(
    name: str,
    value: Any,
) -> tuple[np.dtype[Any], tuple[int, ...]]:
    """Validate one exact JSON array descriptor without loading its payload."""
    if not isinstance(value, dict) or set(value) != {"dtype", "shape"}:
        raise ValueError(
            f"array descriptor for {name} must contain exactly dtype and shape"
        )
    dtype_value = value["dtype"]
    if not isinstance(dtype_value, str):
        raise TypeError(f"{name} dtype must be a string")
    try:
        dtype = np.dtype(dtype_value)
    except TypeError as error:
        raise ValueError(f"{name} declares an invalid dtype") from error
    if dtype.hasobject:
        raise TypeError(f"{name} declares an unsupported object dtype")
    shape_value = value["shape"]
    if not isinstance(shape_value, list) or any(
        type(dimension) is not int or dimension < 0
        for dimension in shape_value
    ):
        raise TypeError(f"{name} shape must be nonnegative JSON integers")
    return dtype, tuple(shape_value)


def _validate_archive_byte_limit(value: int, field_name: str) -> int:
    """Return a positive archive ceiling without boolean coercion."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{field_name} must be an integer")
    limit = int(value)
    if limit <= 0:
        raise ValueError(f"{field_name} must be positive")
    return limit


def _array_payload_bytes(
    shape: tuple[int, ...],
    dtype: np.dtype[Any],
) -> int:
    """Calculate exact uncompressed NPY payload bytes with Python integers."""
    elements = 1
    for dimension in shape:
        elements *= dimension
    return elements * dtype.itemsize


def _preflight_safe_descriptors(
    metadata: Mapping[str, Any],
    descriptors: Mapping[str, tuple[np.dtype[Any], tuple[int, ...]]],
    *,
    num_nodes: int,
    max_dense_bytes: int,
) -> None:
    """Reject impossible or oversized descriptors before member access."""
    label_dtype, label_shape = descriptors["labels"]
    if label_shape != (num_nodes,):
        raise ValueError(
            f"labels descriptor shape must equal [{num_nodes}]"
        )
    if label_dtype.kind not in "iuf":
        raise TypeError("labels descriptor must declare a real numeric dtype")
    incidence_dtype, incidence_shape = descriptors["incidence"]
    if len(incidence_shape) != 2 or incidence_shape[0] != 2:
        raise ValueError("incidence descriptor must have shape [2, M]")
    if not np.issubdtype(incidence_dtype, np.integer):
        raise TypeError("incidence descriptor must declare an integer dtype")

    if metadata["feature_storage"] == "dense":
        feature_dtype, feature_shape = descriptors["features"]
        if (
            len(feature_shape) != 2
            or feature_shape[0] != num_nodes
            or feature_shape[1] <= 0
        ):
            raise ValueError(
                "features descriptor must have shape "
                "[num_nodes, positive feature width]"
            )
        if not np.issubdtype(feature_dtype, np.floating):
            raise TypeError(
                "features descriptor must declare a floating dtype"
            )
        _preflight_dense_features(
            shape=feature_shape,
            dtype=feature_dtype,
            max_dense_bytes=max_dense_bytes,
        )
        return

    index_dtype, index_shape = descriptors["feature_indices"]
    value_dtype, value_shape = descriptors["feature_values"]
    shape_dtype, shape_shape = descriptors["feature_shape"]
    if (
        index_dtype != np.dtype(np.int64)
        or len(index_shape) != 2
        or index_shape[0] != 2
    ):
        raise TypeError(
            "feature_indices descriptor must declare int64 shape [2, NNZ]"
        )
    if (
        not np.issubdtype(value_dtype, np.floating)
        or value_shape != (index_shape[1],)
    ):
        raise TypeError(
            "feature_values descriptor must declare floating shape [NNZ]"
        )
    if shape_dtype != np.dtype(np.int64) or shape_shape != (2,):
        raise TypeError(
            "feature_shape descriptor must declare int64 shape [2]"
        )


def _read_npy_header(
    stream: Any,
    member_name: str,
) -> tuple[np.dtype[Any], tuple[int, ...], int]:
    """Read one bounded standard NPY header without touching array payload."""
    try:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, fortran_order, dtype = (
                np.lib.format.read_array_header_1_0(stream)
            )
        elif version == (2, 0):
            shape, fortran_order, dtype = (
                np.lib.format.read_array_header_2_0(stream)
            )
        else:
            raise ValueError(
                f"{member_name} uses unsupported NPY version {version}"
            )
    except (EOFError, OSError, ValueError) as error:
        raise ValueError(f"{member_name} has an invalid NPY header") from error
    dtype = np.dtype(dtype)
    shape = tuple(int(dimension) for dimension in shape)
    if dtype.hasobject:
        raise TypeError(f"{member_name} declares an unsupported object dtype")
    if fortran_order:
        raise ValueError(f"{member_name} must use C-contiguous NPY storage")
    return dtype, shape, int(stream.tell())


def _preflight_npz_archive(
    path: Path,
    descriptors: Mapping[str, tuple[np.dtype[Any], tuple[int, ...]]],
    *,
    max_member_bytes: int,
    max_total_bytes: int,
) -> None:
    """Validate raw ZIP members and NPY headers before NumPy allocation."""
    expected_names = {f"{name}.npy" for name in descriptors}
    try:
        with zipfile.ZipFile(path, mode="r") as archive:
            members = archive.infolist()
            member_names = [member.filename for member in members]
            if (
                len(member_names) != len(expected_names)
                or set(member_names) != expected_names
            ):
                raise ValueError(
                    "safe NPZ must contain exact array members with no "
                    "directories, extras, or duplicates"
                )

            total_file_bytes = 0
            total_payload_bytes = 0
            for member in members:
                if member.is_dir():
                    raise ValueError("safe NPZ must not contain directories")
                if member.compress_type != zipfile.ZIP_STORED:
                    raise ValueError(
                        "safe NPZ members must use ZIP_STORED"
                    )
                if member.flag_bits & 0x1:
                    raise ValueError("safe NPZ members must not be encrypted")
                if member.compress_size != member.file_size:
                    raise ValueError(
                        "ZIP_STORED member sizes must match exactly"
                    )
                if member.file_size > max_member_bytes:
                    raise ValueError(
                        f"{member.filename} expanded bytes exceed "
                        f"per-member limit={max_member_bytes}"
                    )
                total_file_bytes += member.file_size
                if total_file_bytes > max_total_bytes:
                    raise ValueError(
                        "safe NPZ expanded bytes exceed "
                        f"total limit={max_total_bytes}"
                    )

                array_name = member.filename.removesuffix(".npy")
                declared_dtype, declared_shape = descriptors[array_name]
                with archive.open(member, mode="r") as stream:
                    header_dtype, header_shape, header_bytes = (
                        _read_npy_header(stream, member.filename)
                    )
                if header_dtype != declared_dtype:
                    raise TypeError(
                        f"{array_name} NPY header dtype does not match "
                        "metadata"
                    )
                if header_shape != declared_shape:
                    raise ValueError(
                        f"{array_name} NPY header shape does not match "
                        "metadata"
                    )
                payload_bytes = _array_payload_bytes(
                    header_shape,
                    header_dtype,
                )
                expected_member_bytes = header_bytes + payload_bytes
                if expected_member_bytes != member.file_size:
                    raise ValueError(
                        f"{array_name} NPY member size does not match its "
                        "header"
                    )
                if payload_bytes > max_member_bytes:
                    raise ValueError(
                        f"{array_name} expanded bytes exceed "
                        f"per-member limit={max_member_bytes}"
                    )
                total_payload_bytes += payload_bytes
                if total_payload_bytes > max_total_bytes:
                    raise ValueError(
                        "safe NPZ array expanded bytes exceed "
                        f"total limit={max_total_bytes}"
                    )
    except zipfile.BadZipFile as error:
        raise ValueError("safe NPZ must be a valid ZIP archive") from error


def _validate_safe_metadata(
    metadata: dict[str, Any],
) -> tuple[dict[str, tuple[np.dtype[Any], tuple[int, ...]]], int, int]:
    """Validate the closed safe-format schema before opening its NPZ."""
    if set(metadata) != _SAFE_METADATA_KEYS:
        raise ValueError(
            "safe hypergraph metadata schema mismatch; "
            f"expected={sorted(_SAFE_METADATA_KEYS)!r}, "
            f"received={sorted(metadata)!r}"
        )
    if metadata["format"] != SAFE_HYPERGRAPH_FORMAT:
        raise ValueError("unsupported safe hypergraph format")
    if metadata["format_version"] != SAFE_HYPERGRAPH_FORMAT_VERSION:
        raise ValueError(
            "unsupported safe hypergraph format_version; "
            f"expected {SAFE_HYPERGRAPH_FORMAT_VERSION}"
        )
    if metadata["converter_version"] != SAFE_HYPERGRAPH_CONVERTER_VERSION:
        raise ValueError("unsupported safe hypergraph converter_version")
    if metadata["incidence_roles"] != ["node", "hyperedge"]:
        raise ValueError(
            "incidence_roles must explicitly equal ['node', 'hyperedge']"
        )
    if metadata["feature_storage"] not in {"dense", "coo"}:
        raise ValueError("feature_storage must equal 'dense' or 'coo'")
    if not _is_sha256(metadata["raw_sha256"]):
        raise ValueError("raw_sha256 must be a lowercase SHA-256 digest")
    if not _is_sha256(metadata["npz_sha256"]):
        raise ValueError("npz_sha256 must be a lowercase SHA-256 digest")

    num_nodes = _strict_positive_int(metadata["num_nodes"], "num_nodes")
    num_hyperedges = _strict_positive_int(
        metadata["num_hyperedges"],
        "num_hyperedges",
    )
    padding_count = metadata["padding_count"]
    if type(padding_count) is not int:
        raise TypeError("padding_count must be a JSON integer")
    if padding_count < 0:
        raise ValueError("padding_count must be nonnegative")
    padding_sentinel = metadata["padding_sentinel"]
    if isinstance(padding_sentinel, bool) or not isinstance(
        padding_sentinel,
        (str, int, float, type(None)),
    ):
        raise TypeError("padding_sentinel must be a JSON primitive")
    if isinstance(padding_sentinel, float) and not np.isfinite(
        padding_sentinel
    ):
        raise ValueError("padding_sentinel must be finite")

    expected_arrays = (
        {"features", "incidence", "labels"}
        if metadata["feature_storage"] == "dense"
        else {
            "feature_indices",
            "feature_shape",
            "feature_values",
            "incidence",
            "labels",
        }
    )
    arrays = metadata["arrays"]
    if not isinstance(arrays, dict) or set(arrays) != expected_arrays:
        raise ValueError(
            "safe hypergraph array schema mismatch; "
            f"expected={sorted(expected_arrays)!r}"
        )
    descriptors = {
        name: _array_descriptor(name, descriptor)
        for name, descriptor in arrays.items()
    }
    return descriptors, num_nodes, num_hyperedges


def _validated_numeric_labels(
    labels: np.ndarray,
    *,
    num_nodes: int,
) -> Tensor:
    """Require finite integer-valued numeric labels before converting to long."""
    if labels.ndim != 1 or labels.shape[0] != num_nodes:
        raise ValueError(
            f"labels must have shape [{num_nodes}]; received {labels.shape}"
        )
    if labels.dtype.hasobject or labels.dtype.kind not in "iuf":
        raise TypeError("labels must have a real numeric dtype")
    if not _numpy_values_are_finite(labels) or (
        np.issubdtype(labels.dtype, np.floating)
        and not bool(np.equal(labels, np.floor(labels)).all())
    ):
        raise ValueError("labels must contain only finite integer values")
    if np.issubdtype(labels.dtype, np.floating):
        out_of_range = labels.size and (
            bool((labels < float(-(2**63))).any())
            or bool((labels >= float(2**63)).any())
        )
    elif np.issubdtype(labels.dtype, np.unsignedinteger):
        out_of_range = labels.size and bool(
            (labels > np.uint64(np.iinfo(np.int64).max)).any()
        )
    else:
        int64 = np.iinfo(np.int64)
        out_of_range = labels.size and (
            bool((labels < int64.min).any())
            or bool((labels > int64.max).any())
        )
    if out_of_range:
        raise ValueError(
            "labels must contain only finite integer values within "
            "the torch.long range"
        )
    return torch.from_numpy(labels.astype(np.int64, copy=True))


def _validated_text_labels(
    labels: np.ndarray,
    *,
    num_nodes: int,
) -> Tensor:
    """Parse decimal label tokens exactly before any integer conversion."""
    if labels.ndim != 1 or labels.shape[0] != num_nodes:
        raise ValueError(
            f"labels must have shape [{num_nodes}]; received {labels.shape}"
        )
    lower_bound = Decimal(-(2**63))
    upper_bound = Decimal(2**63)
    parsed: list[int] = []
    for token in labels:
        try:
            value = Decimal(str(token))
        except InvalidOperation as error:
            raise ValueError(
                "labels must contain only finite integer values"
            ) from error
        if (
            not value.is_finite()
            or value != value.to_integral_value()
            or value < lower_bound
            or value >= upper_bound
        ):
            raise ValueError(
                "labels must contain only finite integer values within "
                "the torch.long range"
            )
        parsed.append(int(value))
    class_id = {
        value: index for index, value in enumerate(sorted(set(parsed)))
    }
    return torch.tensor(
        [class_id[value] for value in parsed],
        dtype=torch.long,
    )


def _validated_incidence(
    incidence: np.ndarray,
    *,
    num_nodes: int,
    num_hyperedges: int,
) -> Tensor:
    """Validate explicit incidence roles and preserve every declared pair."""
    if incidence.ndim != 2 or incidence.shape[0] != 2:
        raise ValueError(
            "incidence must have shape [2, M]; "
            f"received {incidence.shape}"
        )
    if incidence.dtype.hasobject or not np.issubdtype(
        incidence.dtype,
        np.integer,
    ):
        raise TypeError("incidence must have an integer dtype")
    if incidence.size and (
        bool((incidence < 0).any())
        or bool((incidence[0] >= num_nodes).any())
        or bool((incidence[1] >= num_hyperedges).any())
    ):
        raise ValueError("incidence indices exceed their declared role bounds")
    return torch.from_numpy(incidence.astype(np.int64, copy=True))


def _safe_feature_tensor(
    arrays: Mapping[str, np.ndarray],
    *,
    feature_storage: str,
    num_nodes: int,
    max_dense_bytes: int,
) -> Tensor:
    """Build dense or explicit COO features without implicit densification."""
    if feature_storage == "dense":
        features = _as_feature_tensor(
            arrays["features"],
            max_dense_bytes=max_dense_bytes,
        )
        if int(features.size(0)) != num_nodes:
            raise ValueError(
                "features row count must equal metadata num_nodes"
            )
        return features

    indices = arrays["feature_indices"]
    values = arrays["feature_values"]
    shape_array = arrays["feature_shape"]
    if indices.dtype != np.dtype(np.int64):
        raise TypeError("feature_indices must use int64")
    if indices.ndim != 2 or indices.shape[0] != 2:
        raise ValueError("feature_indices must have shape [2, NNZ]")
    if values.ndim != 1 or values.shape[0] != indices.shape[1]:
        raise ValueError("feature_values must have one entry per COO index")
    if values.dtype.hasobject or not np.issubdtype(
        values.dtype,
        np.floating,
    ):
        raise TypeError("feature_values must have a floating dtype")
    if shape_array.dtype != np.dtype(np.int64) or shape_array.shape != (2,):
        raise TypeError("feature_shape must be an int64 array with shape [2]")
    shape = (int(shape_array[0]), int(shape_array[1]))
    if shape[0] != num_nodes or shape[1] <= 0:
        raise ValueError(
            "feature_shape must equal [num_nodes, positive feature width]"
        )
    if indices.size and (
        bool((indices < 0).any())
        or bool((indices[0] >= shape[0]).any())
        or bool((indices[1] >= shape[1]).any())
    ):
        raise ValueError("feature_indices exceed feature_shape bounds")
    tensor = torch.sparse_coo_tensor(
        torch.from_numpy(indices),
        torch.from_numpy(values),
        size=shape,
    )
    return _as_feature_tensor(tensor, max_dense_bytes=max_dense_bytes)


def load_hypergraph_npz_dataset(
    data_dir: str | Path,
    data_name: str,
    *,
    max_dense_bytes: int = _DEFAULT_MAX_DENSE_FEATURE_BYTES,
    max_array_member_bytes: int = _DEFAULT_MAX_ARRAY_MEMBER_BYTES,
    max_total_array_bytes: int = _DEFAULT_MAX_TOTAL_ARRAY_BYTES,
) -> tuple[HypergraphData, str]:
    """Load one closed-schema NPZ+JSON hypergraph without executable payloads."""
    dense_limit = _validate_dense_byte_limit(max_dense_bytes)
    member_limit = _validate_archive_byte_limit(
        max_array_member_bytes,
        "max_array_member_bytes",
    )
    total_limit = _validate_archive_byte_limit(
        max_total_array_bytes,
        "max_total_array_bytes",
    )
    resolved_dir = _resolve_data_dir(data_dir, data_name)
    metadata_path = resolved_dir / f"{data_name}.json"
    npz_path = resolved_dir / f"{data_name}.npz"
    if not metadata_path.is_file() or not npz_path.is_file():
        raise FileNotFoundError(
            "safe hypergraph assets require both "
            f"{data_name}.json and {data_name}.npz"
        )
    metadata = _read_safe_metadata(metadata_path)
    descriptors, num_nodes, num_hyperedges = _validate_safe_metadata(metadata)
    _preflight_safe_descriptors(
        metadata,
        descriptors,
        num_nodes=num_nodes,
        max_dense_bytes=dense_limit,
    )
    if _sha256_file(npz_path) != metadata["npz_sha256"]:
        raise ValueError("safe hypergraph NPZ SHA-256 mismatch")
    _preflight_npz_archive(
        npz_path,
        descriptors,
        max_member_bytes=member_limit,
        max_total_bytes=total_limit,
    )

    arrays: dict[str, np.ndarray] = {}
    with np.load(npz_path, allow_pickle=False) as archive:
        if len(archive.files) != len(descriptors) or set(
            archive.files
        ) != set(descriptors):
            raise ValueError("safe hypergraph NPZ array schema mismatch")
        for name, (declared_dtype, declared_shape) in descriptors.items():
            array = archive[name]
            if array.dtype != declared_dtype:
                raise TypeError(
                    f"{name} dtype does not match metadata; "
                    f"declared={declared_dtype.str}, actual={array.dtype.str}"
                )
            if array.shape != declared_shape:
                raise ValueError(
                    f"{name} shape does not match metadata; "
                    f"declared={declared_shape}, actual={array.shape}"
                )
            arrays[name] = array

    features = _safe_feature_tensor(
        arrays,
        feature_storage=metadata["feature_storage"],
        num_nodes=num_nodes,
        max_dense_bytes=dense_limit,
    )
    labels = _validated_numeric_labels(arrays["labels"], num_nodes=num_nodes)
    hyperedge_index = _validated_incidence(
        arrays["incidence"],
        num_nodes=num_nodes,
        num_hyperedges=num_hyperedges,
    )
    data = HypergraphData(
        x=features,
        y=labels,
        hyperedge_index=hyperedge_index,
        num_hyperedges=num_hyperedges,
        representation_version=HYPERGRAPH_REPRESENTATION_VERSION,
    )
    return validate_hypergraph_structure(data), str(resolved_dir)


def validate_hypergraph_npz_assets(
    data_dir: str | Path,
    data_name: str,
) -> None:
    """Validate staged safe assets for atomic downloader promotion."""
    load_hypergraph_npz_dataset(data_dir, data_name)


@dataclass(frozen=True, slots=True)
class ContentRoleSpec:
    """Explicit row-role rule for a legacy non-executable content table."""

    num_node_rows: int
    num_padding_rows: int = 0
    padding_sentinel: str | int | float | None = None

    def __post_init__(self) -> None:
        if isinstance(self.num_node_rows, bool) or not isinstance(
            self.num_node_rows,
            Integral,
        ):
            raise TypeError("num_node_rows must be an integer")
        if self.num_node_rows <= 0:
            raise ValueError("num_node_rows must be positive")
        if isinstance(self.num_padding_rows, bool) or not isinstance(
            self.num_padding_rows,
            Integral,
        ):
            raise TypeError("num_padding_rows must be an integer")
        if self.num_padding_rows < 0:
            raise ValueError("num_padding_rows must be nonnegative")
        if isinstance(self.padding_sentinel, bool) or not isinstance(
            self.padding_sentinel,
            (str, int, float, type(None)),
        ):
            raise TypeError("padding_sentinel must be a scalar primitive")
        if isinstance(self.padding_sentinel, float) and not np.isfinite(
            self.padding_sentinel
        ):
            raise ValueError("padding_sentinel must be finite")


def _parse_raw_id(value: Any) -> Hashable:
    """Preserve textual IDs while recognizing canonical integer spellings."""
    if isinstance(value, Integral) and not isinstance(value, bool):
        return int(value)
    text = str(value).strip()
    try:
        integer = int(text)
    except ValueError:
        return text
    return integer if str(integer) == text else text


def _read_rows(path: Path, *, minimum_columns: int) -> np.ndarray:
    """Read a whitespace-delimited raw table with stable two-dimensional shape."""
    rows = np.genfromtxt(path, dtype=str)
    rows = np.asarray(rows)
    if rows.size == 0:
        raise ValueError(f"raw hypergraph file is empty: {path.name}")
    rows = np.atleast_2d(rows)
    if rows.shape[1] < minimum_columns:
        raise ValueError(
            f"{path.name} must contain at least {minimum_columns} columns"
        )
    return rows


def load_hypergraph_content_dataset(
    data_dir: str | Path,
    data_name: str,
    *,
    role_spec: ContentRoleSpec,
) -> tuple[HypergraphData, str]:
    """Parse a legacy text table using only explicitly declared row roles."""
    if not isinstance(role_spec, ContentRoleSpec):
        raise TypeError("role_spec must be a ContentRoleSpec")
    resolved_dir = Path(data_dir)
    content = _read_rows(
        resolved_dir / f"{data_name}.content",
        minimum_columns=3,
    )
    edges = _read_rows(
        resolved_dir / f"{data_name}.edges",
        minimum_columns=2,
    )
    if edges.shape[1] != 2:
        raise ValueError(f"{data_name}.edges must contain exactly two columns")

    content_ids = [_parse_raw_id(value) for value in content[:, 0]]
    if len(set(content_ids)) != len(content_ids):
        raise ValueError("content IDs must be unique")
    raw_edges = [
        (_parse_raw_id(node), _parse_raw_id(hyperedge))
        for node, hyperedge in edges
    ]
    raw_node_ids = {node for node, _ in raw_edges}
    raw_hyperedge_ids = {hyperedge for _, hyperedge in raw_edges}
    if raw_node_ids & raw_hyperedge_ids:
        raise ValueError(
            "declared node and hyperedge roles must be disjoint"
        )

    num_rows = len(content_ids)
    num_node_rows = int(role_spec.num_node_rows)
    num_padding_rows = int(role_spec.num_padding_rows)
    if num_node_rows + num_padding_rows > num_rows:
        raise ValueError("declared node and padding row counts exceed content")
    node_rows = list(range(num_node_rows))
    padding_start = num_rows - num_padding_rows
    padding_rows = list(range(padding_start, num_rows))
    role_rows = list(range(num_node_rows, padding_start))
    declared_node_ids = {content_ids[row] for row in node_rows}
    declared_padding_ids = {content_ids[row] for row in padding_rows}
    if raw_hyperedge_ids & declared_node_ids:
        raise ValueError(
            "declared node and hyperedge roles must be disjoint"
        )
    if (raw_node_ids - declared_node_ids) or (
        raw_hyperedge_ids & declared_padding_ids
    ):
        raise ValueError("edges conflict with declared content row roles")
    ambiguous_ids = [
        content_ids[row]
        for row in role_rows
        if content_ids[row] not in raw_hyperedge_ids
    ]
    if ambiguous_ids:
        raise ValueError(
            "ambiguous content row roles require an explicit node or "
            f"padding count; rows={ambiguous_ids!r}"
        )

    if num_padding_rows and role_spec.padding_sentinel is not None:
        padding_payload = content[padding_rows, 1:]
        try:
            numeric_payload = padding_payload.astype(np.float64)
            sentinel = float(role_spec.padding_sentinel)
        except (TypeError, ValueError):
            if not bool(
                (padding_payload == str(role_spec.padding_sentinel)).all()
            ):
                raise ValueError(
                    "declared padding rows do not match padding_sentinel"
                )
        else:
            if not bool((numeric_payload == sentinel).all()):
                raise ValueError(
                    "declared padding rows do not match padding_sentinel"
                )

    node_id = {
        content_ids[row]: index for index, row in enumerate(node_rows)
    }
    try:
        raw_features = content[node_rows, 1:-1].astype(np.float64)
    except ValueError as error:
        raise ValueError("content features must be numeric") from error
    features = _as_feature_tensor(raw_features)
    labels = _validated_text_labels(
        content[node_rows, -1],
        num_nodes=num_node_rows,
    )

    hyperedges: dict[Hashable, list[int]] = defaultdict(list)
    for raw_node, raw_hyperedge in raw_edges:
        hyperedges[raw_hyperedge].append(node_id[raw_node])
    hyperedge_index, num_hyperedges = incidence_pairs(
        hyperedges,
        num_node_rows,
    )
    data = HypergraphData(
        x=features,
        y=labels,
        hyperedge_index=hyperedge_index,
        num_hyperedges=num_hyperedges,
        representation_version=HYPERGRAPH_REPRESENTATION_VERSION,
    )
    return validate_hypergraph_structure(data), str(resolved_dir)


__all__ = [
    "SAFE_HYPERGRAPH_CONVERTER_VERSION",
    "SAFE_HYPERGRAPH_FORMAT",
    "SAFE_HYPERGRAPH_FORMAT_VERSION",
    "ContentRoleSpec",
    "incidence_pairs",
    "load_hypergraph_content_dataset",
    "load_hypergraph_npz_dataset",
    "validate_hypergraph_npz_assets",
]
