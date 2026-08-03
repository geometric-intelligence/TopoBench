"""Security tests for GraphUniverse child-to-parent materialization."""

import json
import math
import pickle
import subprocess
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

import topobench.data.loaders.graph.graph_universe_loader as loader_module

from topobench.data.loaders.graph.graph_universe_loader import (
    _graphs_from_npz,
    _materialize_in_isolated_process,
)


def _write_marker(path: str) -> dict[str, object]:
    Path(path).write_text("executed", encoding="utf-8")
    return {"version": "graph-universe-ipc-v2", "graphs": []}


class _ReducerCanary:
    def __init__(self, marker: Path) -> None:
        self.marker = marker

    def __reduce__(self):
        return _write_marker, (str(self.marker),)


def _write_safe_result(
    path: Path,
    records: list[dict[str, np.ndarray]],
    *,
    extra_arrays: dict[str, np.ndarray] | None = None,
) -> None:
    arrays: dict[str, np.ndarray] = {}
    graph_descriptors = []
    for index, record in enumerate(records):
        descriptors = {}
        for key in ("edge_index", "x", "y"):
            name = f"graph_{index}_{key}"
            array = record[key]
            arrays[name] = array
            descriptors[key] = {
                "array": name,
                "dtype": array.dtype.str,
                "shape": list(array.shape),
            }
        graph_descriptors.append(descriptors)
    metadata = {
        "version": "graph-universe-ipc-v2",
        "schema": "static-pyg-data-v1",
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
    if extra_arrays is not None:
        arrays.update(extra_arrays)
    with path.open("wb") as stream:
        np.savez(stream, **arrays)


def test_valid_graph_universe_npz_round_trips_as_static_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = {
        "x": np.ones((3, 2), dtype=np.float32),
        "edge_index": np.array([[0, 1], [1, 2]], dtype=np.int64),
        "y": np.array([1.0], dtype=np.float32),
    }

    def publish_valid_result(
        command: list[str],
        *,
        check: bool,
        stderr,
        timeout: int,
    ) -> subprocess.CompletedProcess:
        del check, stderr, timeout
        _write_safe_result(Path(command[-1]), [record])
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", publish_valid_result)

    dataset = _materialize_in_isolated_process(
        tmp_path / "GraphUniverse",
        {},
    )
    graph = dataset[0]

    assert type(graph) is Data
    assert torch.equal(graph.edge_index, torch.from_numpy(record["edge_index"]))
    assert torch.equal(graph.x, torch.from_numpy(record["x"]))
    assert torch.equal(graph.y, torch.from_numpy(record["y"]))


@pytest.mark.parametrize(
    ("key", "value", "error_type", "message"),
    [
        (
            "edge_index",
            np.array([[0], [1]], dtype=np.int32),
            TypeError,
            "edge_index must use int64",
        ),
        (
            "x",
            np.ones(3, dtype=np.float32),
            ValueError,
            "x must have shape",
        ),
        (
            "y",
            np.array([np.inf], dtype=np.float32),
            ValueError,
            "y must contain only finite values",
        ),
    ],
)
def test_parent_rejects_invalid_graph_arrays(
    tmp_path: Path,
    key: str,
    value: np.ndarray,
    error_type: type[Exception],
    message: str,
) -> None:
    record = {
        "edge_index": np.array([[0, 1], [1, 2]], dtype=np.int64),
        "x": np.ones((3, 2), dtype=np.float32),
        "y": np.array([1.0], dtype=np.float32),
    }
    record[key] = value
    result_path = tmp_path / "result.npz"
    _write_safe_result(result_path, [record])

    with pytest.raises(error_type, match=message):
        _graphs_from_npz(result_path)


def test_parent_rejects_symlinked_npz(tmp_path: Path) -> None:
    target_path = tmp_path / "target.npz"
    result_path = tmp_path / "result.npz"
    _write_safe_result(
        target_path,
        [
            {
                "edge_index": np.array([[0], [1]], dtype=np.int64),
                "x": np.ones((2, 1), dtype=np.float32),
                "y": np.array([0], dtype=np.int64),
            }
        ],
    )
    result_path.symlink_to(target_path)

    with pytest.raises(ValueError, match="regular file|symlink"):
        _graphs_from_npz(result_path)


def test_parent_rejects_path_replacement_between_preflight_and_decode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result_path = tmp_path / "result.npz"
    replacement_path = tmp_path / "replacement.npz"
    record = {
        "edge_index": np.array([[0], [1]], dtype=np.int64),
        "x": np.ones((2, 1), dtype=np.float32),
        "y": np.array([0], dtype=np.int64),
    }
    _write_safe_result(result_path, [record])
    _write_safe_result(
        replacement_path,
        [{**record, "x": np.zeros((2, 1), dtype=np.float32)}],
    )
    preflight = loader_module._preflight_npz

    def replace_after_preflight(payload: bytes):
        headers = preflight(payload)
        replacement_path.replace(result_path)
        return headers

    monkeypatch.setattr(
        loader_module,
        "_preflight_npz",
        replace_after_preflight,
    )

    with pytest.raises(ValueError, match="changed during validation"):
        _graphs_from_npz(result_path)


def test_parent_rejects_in_place_mutation_between_preflight_and_decode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result_path = tmp_path / "result.npz"
    record = {
        "edge_index": np.array([[0], [1]], dtype=np.int64),
        "x": np.ones((2, 1), dtype=np.float32),
        "y": np.array([0], dtype=np.int64),
    }
    _write_safe_result(result_path, [record])
    preflight = loader_module._preflight_npz

    def mutate_after_preflight(payload: bytes):
        headers = preflight(payload)
        _write_safe_result(
            result_path,
            [{**record, "x": np.zeros((3, 2), dtype=np.float32)}],
        )
        return headers

    monkeypatch.setattr(
        loader_module,
        "_preflight_npz",
        mutate_after_preflight,
    )

    with pytest.raises(ValueError, match="changed during validation"):
        _graphs_from_npz(result_path)


@pytest.mark.parametrize(
    ("field", "error_type", "message"),
    [
        ("dtype", TypeError, "loaded dtype does not match preflight"),
        ("shape", ValueError, "loaded shape does not match preflight"),
        ("byte_size", ValueError, "loaded byte size does not match preflight"),
    ],
)
def test_parent_rejects_loaded_array_descriptor_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    error_type: type[Exception],
    message: str,
) -> None:
    result_path = tmp_path / "result.npz"
    _write_safe_result(
        result_path,
        [
            {
                "edge_index": np.array([[0], [1]], dtype=np.int64),
                "x": np.ones((2, 1), dtype=np.float32),
                "y": np.array([0], dtype=np.int64),
            }
        ],
    )
    preflight = loader_module._preflight_npz

    def alter_descriptor_after_preflight(payload: bytes):
        headers = preflight(payload)
        descriptor = headers["graph_0_x"]
        dtype, shape = descriptor[:2]
        byte_size = (
            descriptor[2]
            if len(descriptor) == 3
            else math.prod(shape) * dtype.itemsize
        )
        if field == "dtype":
            dtype = np.dtype(np.float64)
        elif field == "shape":
            shape = (shape[0] + 1, *shape[1:])
        else:
            byte_size += dtype.itemsize
        headers["graph_0_x"] = (dtype, shape, byte_size)
        return headers

    monkeypatch.setattr(
        loader_module,
        "_preflight_npz",
        alter_descriptor_after_preflight,
    )

    with pytest.raises(error_type, match=message):
        _graphs_from_npz(result_path)


def test_parent_rejects_unexpected_graph_array_members(
    tmp_path: Path,
) -> None:
    result_path = tmp_path / "result.npz"
    _write_safe_result(
        result_path,
        [
            {
                "edge_index": np.array([[0], [1]], dtype=np.int64),
                "x": np.ones((2, 1), dtype=np.float32),
                "y": np.array([0], dtype=np.int64),
            }
        ],
        extra_arrays={"graph_0_code": np.array([1], dtype=np.int64)},
    )

    with pytest.raises(ValueError, match="unexpected array members"):
        _graphs_from_npz(result_path)


def _write_pickle_payload(path: Path, payload: object) -> None:
    with path.open("wb") as stream:
        pickle.dump(payload, stream)


def _write_torch_payload(path: Path, payload: object) -> None:
    torch.save(payload, path)


@pytest.mark.parametrize(
    "serializer",
    [_write_pickle_payload, _write_torch_payload],
    ids=["pickle", "torch"],
)
def test_parent_rejects_executable_payload_without_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    serializer: Callable[[Path, object], None],
) -> None:
    marker = tmp_path / "executed.txt"

    def write_poisoned_result(
        command: list[str],
        *,
        check: bool,
        stderr,
        timeout: int,
    ) -> subprocess.CompletedProcess:
        del check, stderr, timeout
        serializer(Path(command[-1]), _ReducerCanary(marker))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", write_poisoned_result)

    with pytest.raises(ValueError):
        _materialize_in_isolated_process(tmp_path / "GraphUniverse", {})

    assert not marker.exists()
