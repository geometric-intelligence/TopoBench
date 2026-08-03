"""Loaders for GraphUniverse [1] datasets.

[1] "GraphUniverse: Enabling Systematic Evaluation of Inductive Generalization" by Louis Van Langendonck and Guillermo Bernardez and Nina Miolane and Pere Barlet-Ros
Accepted at The Fourteenth International Conference on Learning Representations, 2026},
https://openreview.net/forum?id=jRWxvQnqUt
"""

import ast
import io
import json
import math
import os
import stat
import struct
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig
from torch_geometric.data import Data, Dataset, InMemoryDataset

from topobench.data.loaders.base import AbstractLoader, resolve_cache_config

_GRAPH_UNIVERSE_REPRESENTATION_VERSION = "pyg-data-v1"
_GRAPH_UNIVERSE_PARSER_VERSION = "graph-universe-v1"
_GENERATION_TIMEOUT_SECONDS = 3600
_GRAPH_UNIVERSE_IPC_VERSION = "graph-universe-ipc-v2"
_GRAPH_UNIVERSE_IPC_SCHEMA = "static-pyg-data-v1"
_GRAPH_RECORD_KEYS = frozenset({"edge_index", "x", "y"})
_GRAPH_KEY_ORDER = ("edge_index", "x", "y")
_METADATA_ARRAY_NAME = "metadata"
_MAX_METADATA_BYTES = 1_048_576
_MAX_NPY_HEADER_BYTES = 65_536
_MAX_ARRAY_MEMBER_BYTES = 512 * 1024**2
_MAX_TOTAL_ARRAY_BYTES = 1024 * 1024**2
_SAFE_ARRAY_DTYPES = frozenset(
    np.dtype(dtype).str
    for dtype in (
        np.bool_,
        np.uint8,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.float16,
        np.float32,
        np.float64,
    )
)

_NpyDescriptor = tuple[np.dtype, tuple[int, ...], int]
_FileSignature = tuple[int, int, int, int, int, int, int, int, int]
_GENERATION_SCRIPT = """
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from graph_universe import GraphUniverseDataset
from torch_geometric.data import Data

VERSION = "graph-universe-ipc-v2"
SCHEMA = "static-pyg-data-v1"
KEYS = ("edge_index", "x", "y")
SAFE_DTYPES = {
    np.dtype(dtype).str
    for dtype in (
        np.bool_,
        np.uint8,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.float16,
        np.float32,
        np.float64,
    )
}


def validate_array(array, key, index):
    context = f"GraphUniverse graph {index}.{key}"
    if array.dtype.str not in SAFE_DTYPES:
        raise TypeError(f"{context} has unsupported dtype {array.dtype}")
    if key == "edge_index":
        if array.dtype != np.int64:
            raise TypeError(f"{context} must use int64")
        if array.ndim != 2 or array.shape[0] != 2:
            raise ValueError(f"{context} must have shape [2, num_edges]")
    elif key == "x":
        if array.ndim != 2 or array.shape[0] <= 0 or array.shape[1] <= 0:
            raise ValueError(
                f"{context} must have shape [num_nodes, num_features]"
            )
    elif array.ndim not in {0, 1, 2} or array.size == 0:
        raise ValueError(f"{context} must be a non-empty scalar or matrix")
    if not np.isfinite(array).all():
        raise ValueError(f"{context} must contain only finite values")


with open(sys.argv[2], encoding="utf-8") as file:
    parameters = json.load(file)
dataset = GraphUniverseDataset(root=sys.argv[1], parameters=parameters)
arrays = {}
graph_descriptors = []
for index in range(len(dataset)):
    graph = dataset[index]
    if type(graph) is not Data:
        raise TypeError(
            "GraphUniverse generation returned "
            f"{type(graph).__name__}, expected a static Data object"
        )
    record = graph.to_dict()
    if type(record) is not dict or set(record) != set(KEYS):
        raise ValueError(
            "GraphUniverse graph records must contain exactly "
            "edge_index, x, and y"
        )
    descriptors = {}
    for key in KEYS:
        value = record[key]
        if type(value) is not torch.Tensor:
            raise TypeError("GraphUniverse graph attributes must be tensors")
        if value.layout != torch.strided:
            raise TypeError("GraphUniverse graph attributes must be dense")
        array = value.detach().cpu().numpy()
        validate_array(array, key, index)
        name = f"graph_{index}_{key}"
        arrays[name] = array
        descriptors[key] = {
            "array": name,
            "dtype": array.dtype.str,
            "shape": list(array.shape),
        }
    graph_descriptors.append(descriptors)
if not graph_descriptors:
    raise ValueError("GraphUniverse generation returned no graphs")

metadata = {
    "version": VERSION,
    "schema": SCHEMA,
    "graphs": graph_descriptors,
}
arrays["metadata"] = np.frombuffer(
    json.dumps(
        metadata,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8"),
    dtype=np.uint8,
)
result_path = Path(sys.argv[3])
temporary_path = result_path.with_name(result_path.name + ".tmp")
try:
    with temporary_path.open("wb") as stream:
        np.savez(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary_path, result_path)
finally:
    temporary_path.unlink(missing_ok=True)
"""


def _reject_json_constant(value: str) -> None:
    """Reject non-standard JSON numeric constants."""
    raise ValueError(f"GraphUniverse IPC metadata contains {value}")


def _unique_json_object(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    """Reject duplicate JSON members instead of silently replacing them."""
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"GraphUniverse IPC metadata repeats key {key!r}")
        result[key] = value
    return result


def _read_npy_header(
    stream,
    *,
    member_name: str,
    member_size: int,
) -> _NpyDescriptor:
    """Read a bounded NPY header without materializing its array payload."""
    prefix = stream.read(8)
    if len(prefix) != 8 or prefix[:6] != b"\x93NUMPY":
        raise ValueError(
            f"GraphUniverse IPC member {member_name!r} is not NPY"
        )
    version = tuple(prefix[6:8])
    major = version[0]
    if version == (1, 0):
        header_size_bytes = 2
        encoded_size = stream.read(header_size_bytes)
        if len(encoded_size) != header_size_bytes:
            raise ValueError(
                f"GraphUniverse IPC member {member_name!r} has a truncated header"
            )
        header_size = struct.unpack("<H", encoded_size)[0]
        encoding = "latin1"
    elif version in {(2, 0), (3, 0)}:
        header_size_bytes = 4
        encoded_size = stream.read(header_size_bytes)
        if len(encoded_size) != header_size_bytes:
            raise ValueError(
                f"GraphUniverse IPC member {member_name!r} has a truncated header"
            )
        header_size = struct.unpack("<I", encoded_size)[0]
        encoding = "utf-8" if major == 3 else "latin1"
    else:
        raise ValueError(
            f"GraphUniverse IPC member {member_name!r} uses unsupported NPY "
            f"version {version}"
        )
    if header_size <= 0 or header_size > _MAX_NPY_HEADER_BYTES:
        raise ValueError(
            f"GraphUniverse IPC member {member_name!r} has an oversized NPY header"
        )
    header = stream.read(header_size)
    if len(header) != header_size:
        raise ValueError(
            f"GraphUniverse IPC member {member_name!r} has a truncated NPY header"
        )
    try:
        descriptor = ast.literal_eval(header.decode(encoding))
    except (SyntaxError, UnicodeDecodeError, ValueError) as error:
        raise ValueError(
            f"GraphUniverse IPC member {member_name!r} has an invalid NPY header"
        ) from error
    if (
        type(descriptor) is not dict
        or set(descriptor) != {"descr", "fortran_order", "shape"}
        or type(descriptor["fortran_order"]) is not bool
        or type(descriptor["shape"]) is not tuple
        or any(
            type(dimension) is not int or dimension < 0
            for dimension in descriptor["shape"]
        )
    ):
        raise ValueError(
            f"GraphUniverse IPC member {member_name!r} has an invalid NPY schema"
        )
    try:
        dtype = np.dtype(descriptor["descr"])
    except (TypeError, ValueError) as error:
        raise TypeError(
            f"GraphUniverse IPC member {member_name!r} has an invalid dtype"
        ) from error
    if dtype.str not in _SAFE_ARRAY_DTYPES:
        raise TypeError(
            f"GraphUniverse IPC member {member_name!r} has unsupported dtype "
            f"{dtype}"
        )
    shape = descriptor["shape"]
    payload_size = math.prod(shape) * dtype.itemsize
    encoded_header_size = 8 + header_size_bytes + header_size
    if encoded_header_size + payload_size != member_size:
        raise ValueError(
            f"GraphUniverse IPC member {member_name!r} size does not match "
            "its NPY header"
        )
    return dtype, shape, payload_size


def _preflight_npz(payload: bytes) -> dict[str, _NpyDescriptor]:
    """Bound and validate every NPZ member before NumPy materializes it."""
    if len(payload) > _MAX_TOTAL_ARRAY_BYTES:
        raise ValueError("GraphUniverse IPC archive is too large")
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            infos = archive.infolist()
            if not infos:
                raise ValueError("GraphUniverse IPC archive is empty")
            if len({info.filename for info in infos}) != len(infos):
                raise ValueError("GraphUniverse IPC archive repeats a member")
            if any(
                info.flag_bits & 1
                or "/" in info.filename
                or not info.filename.endswith(".npy")
                for info in infos
            ):
                raise ValueError(
                    "GraphUniverse IPC archive contains an invalid member name"
                )
            total_size = sum(info.file_size for info in infos)
            if total_size > _MAX_TOTAL_ARRAY_BYTES or any(
                info.file_size > _MAX_ARRAY_MEMBER_BYTES for info in infos
            ):
                raise ValueError(
                    "GraphUniverse IPC archive exceeds the safe array size limit"
                )
            headers = {}
            for info in infos:
                name = info.filename[:-4]
                if (
                    name == _METADATA_ARRAY_NAME
                    and info.file_size
                    > _MAX_METADATA_BYTES + _MAX_NPY_HEADER_BYTES + 12
                ):
                    raise ValueError(
                        "GraphUniverse IPC metadata exceeds the safe size limit"
                    )
                with archive.open(info) as stream:
                    headers[name] = _read_npy_header(
                        stream,
                        member_name=name,
                        member_size=info.file_size,
                    )
            return headers
    except zipfile.BadZipFile as error:
        raise ValueError(
            "GraphUniverse IPC payload must be an NPZ archive"
        ) from error


def _file_signature(status: os.stat_result) -> _FileSignature:
    """Return the file identity and mutation-sensitive metadata."""
    return (
        status.st_dev,
        status.st_ino,
        status.st_mode,
        status.st_nlink,
        status.st_uid,
        status.st_gid,
        status.st_size,
        status.st_mtime_ns,
        status.st_ctime_ns,
    )


def _regular_file_status(result_path: Path) -> os.stat_result:
    """Inspect a path without following its final component."""
    try:
        status = result_path.lstat()
    except OSError as error:
        raise ValueError(
            "GraphUniverse IPC source must be a stable regular file"
        ) from error
    if not stat.S_ISREG(status.st_mode):
        raise ValueError(
            "GraphUniverse IPC source must be a regular file, not a symlink"
        )
    return status


def _assert_source_unchanged(
    result_path: Path,
    expected: _FileSignature,
) -> None:
    """Reject replacement or mutation of the source path."""
    if _file_signature(_regular_file_status(result_path)) != expected:
        raise ValueError("GraphUniverse IPC source changed during validation")


def _read_immutable_npz(
    result_path: Path,
) -> tuple[bytes, _FileSignature]:
    """Read one bounded no-follow regular-file snapshot."""
    before_open = _regular_file_status(result_path)
    no_follow = getattr(os, "O_NOFOLLOW", None)
    if no_follow is None:
        raise RuntimeError(
            "GraphUniverse IPC loading requires no-follow file support"
        )
    flags = (
        os.O_RDONLY
        | no_follow
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        descriptor = os.open(result_path, flags)
    except OSError as error:
        raise ValueError(
            "GraphUniverse IPC source must be a stable regular file"
        ) from error
    try:
        opened = os.fstat(descriptor)
        signature = _file_signature(opened)
        if not stat.S_ISREG(opened.st_mode) or signature != _file_signature(
            before_open
        ):
            raise ValueError(
                "GraphUniverse IPC source changed during validation"
            )
        if opened.st_size > _MAX_TOTAL_ARRAY_BYTES:
            raise ValueError("GraphUniverse IPC archive is too large")
        try:
            with open(descriptor, "rb", closefd=False) as stream:
                payload = stream.read(opened.st_size + 1)
        except OSError as error:
            raise ValueError(
                "GraphUniverse IPC source changed during validation"
            ) from error
        if (
            len(payload) != opened.st_size
            or _file_signature(os.fstat(descriptor)) != signature
        ):
            raise ValueError(
                "GraphUniverse IPC source changed during validation"
            )
        _assert_source_unchanged(result_path, signature)
        return payload, signature
    finally:
        os.close(descriptor)


def _validated_loaded_array(
    array: np.ndarray,
    descriptor: _NpyDescriptor,
    *,
    member_name: str,
) -> np.ndarray:
    """Match a loaded array exactly to its bounded preflight descriptor."""
    context = f"GraphUniverse IPC member {member_name!r}"
    if type(array) is not np.ndarray:
        raise TypeError(f"{context} did not load as an array")
    expected_dtype, expected_shape, expected_bytes = descriptor
    if array.dtype.str != expected_dtype.str:
        raise TypeError(f"{context} loaded dtype does not match preflight")
    if array.shape != expected_shape:
        raise ValueError(f"{context} loaded shape does not match preflight")
    if array.nbytes != expected_bytes:
        raise ValueError(
            f"{context} loaded byte size does not match preflight"
        )
    return array


def _metadata_from_array(array: np.ndarray) -> dict[str, object]:
    """Decode the bounded uint8 JSON metadata member."""
    if (
        type(array) is not np.ndarray
        or array.dtype != np.uint8
        or array.ndim != 1
        or not 0 < array.size <= _MAX_METADATA_BYTES
    ):
        raise ValueError(
            "GraphUniverse IPC metadata must be a non-empty bounded uint8 vector"
        )
    try:
        encoded = array.tobytes().decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("GraphUniverse IPC metadata must be UTF-8") from error
    metadata = json.loads(
        encoded,
        parse_constant=_reject_json_constant,
        object_pairs_hook=_unique_json_object,
    )
    if type(metadata) is not dict:
        raise ValueError("GraphUniverse IPC metadata must be an object")
    return metadata


def _validated_tensor(
    array: np.ndarray,
    *,
    key: str,
    graph_index: int,
) -> torch.Tensor:
    """Validate one closed numeric array before constructing a tensor."""
    context = f"GraphUniverse IPC graph {graph_index}.{key}"
    if (
        type(array) is not np.ndarray
        or array.dtype.str not in _SAFE_ARRAY_DTYPES
    ):
        observed_dtype = getattr(array, "dtype", type(array).__name__)
        raise TypeError(f"{context} has unsupported dtype {observed_dtype}")
    if key == "edge_index":
        if array.dtype != np.int64:
            raise TypeError(f"{context} must use int64")
        if array.ndim != 2 or array.shape[0] != 2:
            raise ValueError(f"{context} must have shape [2, num_edges]")
    elif key == "x":
        if array.ndim != 2 or array.shape[0] <= 0 or array.shape[1] <= 0:
            raise ValueError(
                f"{context} must have shape [num_nodes, num_features]"
            )
    elif array.ndim not in {0, 1, 2} or array.size == 0:
        raise ValueError(f"{context} must be a non-empty scalar or matrix")
    if not bool(np.isfinite(array).all()):
        raise ValueError(f"{context} must contain only finite values")
    return torch.from_numpy(array.copy())


def _graphs_from_npz(result_path: Path) -> list[Data]:
    """Validate a closed NPZ+JSON schema and reconstruct static PyG graphs."""
    payload, source_signature = _read_immutable_npz(result_path)
    headers = _preflight_npz(payload)
    _assert_source_unchanged(result_path, source_signature)
    if _METADATA_ARRAY_NAME not in headers:
        raise ValueError(
            "GraphUniverse IPC archive must contain exactly one metadata member"
        )
    metadata_dtype, metadata_shape, _ = headers[_METADATA_ARRAY_NAME]
    if (
        metadata_dtype != np.uint8
        or len(metadata_shape) != 1
        or not 0 < metadata_shape[0] <= _MAX_METADATA_BYTES
    ):
        raise ValueError(
            "GraphUniverse IPC metadata must be a non-empty bounded uint8 vector"
        )

    loaded = np.load(io.BytesIO(payload), allow_pickle=False)
    if isinstance(loaded, np.ndarray):
        raise ValueError("GraphUniverse IPC payload must be an NPZ archive")
    with loaded as archive:
        if archive.files.count(_METADATA_ARRAY_NAME) != 1 or set(
            archive.files
        ) != set(headers):
            raise ValueError(
                "GraphUniverse IPC archive schema changed after preflight"
            )
        metadata_array = _validated_loaded_array(
            archive[_METADATA_ARRAY_NAME],
            headers[_METADATA_ARRAY_NAME],
            member_name=_METADATA_ARRAY_NAME,
        )
        metadata = _metadata_from_array(metadata_array)
        if set(metadata) != {"graphs", "schema", "version"}:
            raise ValueError(
                "GraphUniverse IPC metadata must contain exactly "
                "graphs, schema, and version"
            )
        if metadata["version"] != _GRAPH_UNIVERSE_IPC_VERSION:
            raise ValueError(
                "unsupported GraphUniverse IPC version: "
                f"{metadata['version']!r}"
            )
        if metadata["schema"] != _GRAPH_UNIVERSE_IPC_SCHEMA:
            raise ValueError(
                f"unsupported GraphUniverse IPC schema: {metadata['schema']!r}"
            )
        graph_descriptors = metadata["graphs"]
        if type(graph_descriptors) is not list or not graph_descriptors:
            raise ValueError(
                "GraphUniverse IPC metadata must describe at least one graph"
            )

        expected_names = {_METADATA_ARRAY_NAME}
        for index, descriptors in enumerate(graph_descriptors):
            if (
                type(descriptors) is not dict
                or set(descriptors) != _GRAPH_RECORD_KEYS
            ):
                raise ValueError(
                    f"GraphUniverse IPC graph {index} must contain exactly "
                    "edge_index, x, and y descriptors"
                )
            for key in _GRAPH_KEY_ORDER:
                descriptor = descriptors[key]
                if type(descriptor) is not dict or set(descriptor) != {
                    "array",
                    "dtype",
                    "shape",
                }:
                    raise ValueError(
                        f"GraphUniverse IPC graph {index}.{key} descriptor "
                        "must contain exactly array, dtype, and shape"
                    )
                expected_name = f"graph_{index}_{key}"
                if descriptor["array"] != expected_name:
                    raise ValueError(
                        f"GraphUniverse IPC graph {index}.{key} has "
                        "an invalid array name"
                    )
                declared_dtype = descriptor["dtype"]
                declared_shape = descriptor["shape"]
                if (
                    type(declared_dtype) is not str
                    or declared_dtype not in _SAFE_ARRAY_DTYPES
                ):
                    raise TypeError(
                        f"GraphUniverse IPC graph {index}.{key} declares "
                        "an unsupported dtype"
                    )
                if type(declared_shape) is not list or any(
                    type(dimension) is not int or dimension < 0
                    for dimension in declared_shape
                ):
                    raise ValueError(
                        f"GraphUniverse IPC graph {index}.{key} declares "
                        "an invalid shape"
                    )
                expected_names.add(expected_name)
                if expected_name not in headers:
                    raise ValueError(
                        f"GraphUniverse IPC archive is missing "
                        f"{expected_name!r}"
                    )

        if set(headers) != expected_names:
            raise ValueError(
                "GraphUniverse IPC archive contains unexpected array members"
            )

        records: list[dict[str, torch.Tensor]] = []
        for index, descriptors in enumerate(graph_descriptors):
            record = {}
            for key in _GRAPH_KEY_ORDER:
                descriptor = descriptors[key]
                name = descriptor["array"]
                header_dtype, header_shape, _ = headers[name]
                array = _validated_loaded_array(
                    archive[name],
                    headers[name],
                    member_name=name,
                )
                if header_dtype.str != descriptor["dtype"]:
                    raise TypeError(
                        f"GraphUniverse IPC graph {index}.{key} dtype "
                        "does not match metadata"
                    )
                if list(header_shape) != descriptor["shape"]:
                    raise ValueError(
                        f"GraphUniverse IPC graph {index}.{key} shape "
                        "does not match metadata"
                    )
                record[key] = _validated_tensor(
                    array,
                    key=key,
                    graph_index=index,
                )
            records.append(record)

    _assert_source_unchanged(result_path, source_signature)
    graphs = []
    for record in records:
        graph = Data.from_dict(record)
        graph.validate(raise_on_error=True)
        graphs.append(graph)
    return graphs


class _SafeGraphUniverseDataset(InMemoryDataset):
    """In-memory PyG dataset reconstructed from the closed IPC schema."""

    def __init__(
        self,
        *,
        canonical_data_dir: Path,
        parameters: dict,
        graphs: list[Data],
    ) -> None:
        self.name = canonical_data_dir.name
        self.parameters = parameters
        self.graph_list: list[Data] = []
        super().__init__(root=None)
        self.root = str(canonical_data_dir.parent)
        self.processed_root = str(canonical_data_dir)
        self._data, self.slices = self.collate(graphs)
        self._data_list = None

    @property
    def raw_dir(self) -> str:
        """Return the selector directory used by the loader."""
        return self.processed_root

    @property
    def processed_dir(self) -> str:
        """Return the selector directory without enabling disk processing."""
        return self.processed_root

    def get_data_dir(self) -> str:
        """Return the selector directory used by GraphUniverse callers."""
        return self.processed_root


def _materialize_in_isolated_process(
    canonical_data_dir: Path,
    parameters: dict,
) -> _SafeGraphUniverseDataset:
    """Generate graphs in isolation and accept only the closed IPC schema."""
    canonical_data_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".graph-universe-",
        dir=canonical_data_dir.parent,
    ) as temporary_directory:
        temporary_root = Path(temporary_directory)
        source_root = temporary_root / "source"
        parameters_path = temporary_root / "parameters.json"
        result_path = temporary_root / "result.npz"
        parameters_path.write_text(
            json.dumps(parameters, allow_nan=False, sort_keys=True),
            encoding="utf-8",
        )
        with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as errors:
            try:
                subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        _GENERATION_SCRIPT,
                        str(source_root),
                        str(parameters_path),
                        str(result_path),
                    ],
                    check=True,
                    stderr=errors,
                    timeout=_GENERATION_TIMEOUT_SECONDS,
                )
            except subprocess.TimeoutExpired as error:
                raise RuntimeError(
                    "GraphUniverse generation did not finish within "
                    f"{_GENERATION_TIMEOUT_SECONDS} seconds"
                ) from error
            except subprocess.CalledProcessError as error:
                errors.seek(0)
                details = errors.read().strip()
                message = (
                    "GraphUniverse generation failed in isolated process "
                    f"with exit code {error.returncode}"
                )
                if details:
                    message = f"{message}:\n{details}"
                raise RuntimeError(message) from error
        graphs = _graphs_from_npz(result_path)

    return _SafeGraphUniverseDataset(
        canonical_data_dir=canonical_data_dir,
        parameters=parameters,
        graphs=graphs,
    )


class GraphUniverseDatasetLoader(AbstractLoader):
    """Load Graph Universe datasets.

    Parameters
    ----------
    parameters : DictConfig
        Configuration parameters containing:
            - data_dir: Root directory for data
            - data_name: Name of the dataset
            - data_type: Type of the dataset (e.g., "graph_classification")
    """

    def __init__(self, parameters: DictConfig) -> None:
        super().__init__(parameters)

    def load_dataset(self) -> Dataset:
        """Load Graph Universe dataset.

        Returns
        -------
        Dataset
            The loaded Graph Universe dataset.

        Raises
        ------
        RuntimeError
            If dataset loading fails.
        """
        parameters = resolve_cache_config(
            self.parameters["generation_parameters"],
            context="GraphUniverse generation parameters",
        )
        if not isinstance(parameters, dict):
            raise TypeError("generation_parameters must resolve to a mapping")
        dataset = _materialize_in_isolated_process(
            Path(self.get_data_dir()),
            parameters,
        )
        dataset.feature_policy = "continuous"
        dataset.representation_version = _GRAPH_UNIVERSE_REPRESENTATION_VERSION
        dataset.parser_version = _GRAPH_UNIVERSE_PARSER_VERSION
        return dataset

    def load(self, **kwargs) -> tuple[Data, str]:
        """Load data.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        tuple[torch_geometric.data.Data, str]
            Tuple containing the loaded data and the data directory.
        """
        dataset, _ = super().load(**kwargs)
        data_dir = dataset.raw_dir

        return dataset, data_dir
