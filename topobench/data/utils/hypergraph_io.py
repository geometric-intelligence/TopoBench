"""Dependency-free raw parsers for native hypergraph datasets."""

from __future__ import annotations

import pickle
from collections import defaultdict
from collections.abc import Hashable, Iterable, Mapping
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from topobench.data import (
    HYPERGRAPH_REPRESENTATION_VERSION,
    HypergraphData,
    validate_hypergraph_structure,
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
    pairs: set[tuple[int, int]] = set()
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
            pairs.add((node_id, hyperedge_id[raw]))

    if not pairs:
        return torch.empty((2, 0), dtype=torch.long), len(ordered_ids)
    index = torch.tensor(sorted(pairs), dtype=torch.long).t().contiguous()
    return index, len(ordered_ids)


def _pickle_data_dir(data_dir: str | Path, data_name: str) -> Path:
    """Resolve archives that either retain or flatten their dataset folder."""
    root = Path(data_dir)
    nested = root / data_name
    if nested.is_dir():
        return nested
    return root


def _as_feature_tensor(features: Any) -> Tensor:
    """Convert dense or scipy-style pickled features without row reordering."""
    if hasattr(features, "toarray"):
        features = features.toarray()
    elif hasattr(features, "todense"):
        features = features.todense()
    return torch.as_tensor(np.asarray(features), dtype=torch.float32)


def _isolated_hyperedge_id(
    hyperedges: Mapping[Hashable, Iterable[int]],
    node: int,
) -> Hashable:
    """Return a deterministic collision-free raw ID for an isolated singleton."""
    candidate: tuple[Any, ...] = ("__topobench_isolated_node__", node)
    suffix = 0
    while candidate in hyperedges:
        suffix += 1
        candidate = ("__topobench_isolated_node__", node, suffix)
    return candidate


def load_hypergraph_pickle_dataset(
    data_dir: str | Path,
    data_name: str,
) -> tuple[HypergraphData, str]:
    """Parse HyperGCN-style pickle files into native dense incidence pairs."""
    resolved_dir = _pickle_data_dir(data_dir, data_name)
    with (resolved_dir / "features.pickle").open("rb") as stream:
        features = _as_feature_tensor(pickle.load(stream))
    with (resolved_dir / "labels.pickle").open("rb") as stream:
        labels = torch.as_tensor(np.asarray(pickle.load(stream)), dtype=torch.long)
    with (resolved_dir / "hypergraph.pickle").open("rb") as stream:
        raw_hypergraph = pickle.load(stream)

    if features.ndim != 2:
        raise ValueError("pickled features must be rank-2")
    labels = labels.reshape(-1)
    num_nodes = int(features.size(0))
    if labels.numel() != num_nodes:
        raise ValueError("pickled labels must contain one entry per feature row")
    if not isinstance(raw_hypergraph, Mapping):
        raise TypeError("pickled hypergraph must be a mapping")

    hypergraph = {raw_id: tuple(nodes) for raw_id, nodes in raw_hypergraph.items()}
    initial_index, _ = incidence_pairs(hypergraph, num_nodes)
    occupied_nodes = set(initial_index[0].tolist())
    for node in range(num_nodes):
        if node not in occupied_nodes:
            hypergraph[_isolated_hyperedge_id(hypergraph, node)] = (node,)

    hyperedge_index, num_hyperedges = incidence_pairs(hypergraph, num_nodes)
    data = HypergraphData(
        x=features,
        y=labels,
        hyperedge_index=hyperedge_index,
        num_hyperedges=num_hyperedges,
        representation_version=HYPERGRAPH_REPRESENTATION_VERSION,
    )
    return validate_hypergraph_structure(data), str(resolved_dir)


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
) -> tuple[HypergraphData, str]:
    """Parse content/edges incidence files without inventing hyperedges."""
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
    raw_hyperedge_ids = {hyperedge for _, hyperedge in raw_edges}
    if any(node in raw_hyperedge_ids for node, _ in raw_edges):
        raise ValueError("raw node and hyperedge IDs must be disjoint")

    node_rows = [
        row_index
        for row_index, raw_id in enumerate(content_ids)
        if raw_id not in raw_hyperedge_ids
    ]
    node_id = {content_ids[row]: index for index, row in enumerate(node_rows)}
    missing_nodes = sorted(
        {raw for raw, _ in raw_edges if raw not in node_id},
        key=lambda value: (type(value).__name__, repr(value)),
    )
    if missing_nodes:
        raise ValueError(
            "edges reference node IDs without feature rows: "
            f"{missing_nodes!r}"
        )

    features = torch.as_tensor(
        content[node_rows, 1:-1].astype(np.float32),
        dtype=torch.float32,
    )
    labels = torch.as_tensor(
        content[node_rows, -1].astype(np.float64),
        dtype=torch.long,
    )
    labels = labels - labels.min()

    hyperedges: dict[Hashable, list[int]] = defaultdict(list)
    for raw_node, raw_hyperedge in raw_edges:
        hyperedges[raw_hyperedge].append(node_id[raw_node])
    hyperedge_index, num_hyperedges = incidence_pairs(
        hyperedges,
        int(features.size(0)),
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
    "incidence_pairs",
    "load_hypergraph_content_dataset",
    "load_hypergraph_pickle_dataset",
]
