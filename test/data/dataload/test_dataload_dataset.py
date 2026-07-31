"""Tests for index-backed graph split views and transductive reuse."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest
import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset, Subset
from torch_geometric.data import Data

from topobench.data.utils.split_utils import (
    load_inductive_splits,
    load_transductive_splits,
)
from topobench.dataloader import GraphDataModule


class TrackingGraphDataset(Dataset[Data]):
    """Graph source that records lazy item access."""

    def __init__(self, graphs: Sequence[Data]) -> None:
        self.graphs = tuple(graphs)
        self.accesses: list[int] = []

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, index: int) -> Data:
        self.accesses.append(index)
        return self.graphs[index]


def _graph(label: int, num_nodes: int = 3) -> Data:
    nodes = torch.arange(num_nodes)
    return Data(
        x=torch.ones(num_nodes, 2),
        edge_index=torch.stack([nodes, (nodes + 1) % num_nodes]),
        y=torch.tensor([label]),
    )


def test_fixed_inductive_splits_are_lazy_views_over_one_source() -> None:
    """Fixed splitting stores indices without copying or eagerly reading graphs."""
    graphs = [_graph(index % 2) for index in range(6)]
    source = TrackingGraphDataset(graphs)
    source.split_idx = {
        "train": np.array([0, 2, 4]),
        "valid": np.array([1]),
        "test": np.array([3, 5]),
    }

    train, val, test = load_inductive_splits(
        source,
        DictConfig({"split_type": "fixed"}),
    )

    assert all(isinstance(view, Subset) for view in (train, val, test))
    assert train.dataset is val.dataset is test.dataset is source
    assert list(train.indices) == [0, 2, 4]
    assert list(val.indices) == [1]
    assert list(test.indices) == [3, 5]
    assert source.accesses == []
    assert all("train_mask" not in graph for graph in graphs)

    module = GraphDataModule(
        dataset_train=train,
        dataset_val=val,
        dataset_test=test,
        learning_setting="inductive",
        batch_size=2,
    )
    next(iter(module.val_dataloader()))
    assert source.accesses == [1]


def test_transductive_split_returns_native_singleton_source(tmp_path) -> None:
    """The split boundary returns one native graph and no phase copies."""
    data = Data(
        x=torch.ones(6, 2),
        y=torch.tensor([0, 1, 0, 1, 0, 1]),
    )
    source = TrackingGraphDataset([data])

    train, val, test = load_transductive_splits(
        source,
        DictConfig(
            {
                "split_type": "random",
                "data_seed": 0,
                "train_prop": 0.5,
                "data_split_dir": str(tmp_path),
            }
        ),
    )

    assert train == [data]
    assert val is None
    assert test is None

    module = GraphDataModule(
        dataset_train=train,
        learning_setting="transductive",
        batch_size=1,
    )
    assert module.dataset_train is module.dataset_val is module.dataset_test
    assert next(iter(module.train_dataloader())).num_graphs == 1
    assert next(iter(module.val_dataloader())).num_graphs == 1
    assert next(iter(module.test_dataloader())).num_graphs == 1


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"batch_size": 2}, "transductive graph loading requires batch_size=1"),
        (
            {"dataset_train": [_graph(0), _graph(1)]},
            "transductive graph loading requires exactly one graph",
        ),
        (
            {"dataset_val": [_graph(0)]},
            "transductive phases must reuse the source graph",
        ),
        (
            {"dataset_test": [_graph(0)]},
            "transductive phases must reuse the source graph",
        ),
    ],
)
def test_transductive_loading_rejects_incompatible_phase_settings(
    kwargs: dict[str, object],
    message: str,
) -> None:
    """A transductive module accepts only its one source graph."""
    arguments: dict[str, object] = {
        "dataset_train": [_graph(0)],
        "learning_setting": "transductive",
        "batch_size": 1,
    }
    arguments.update(kwargs)

    with pytest.raises(ValueError, match=f"^{message}$"):
        GraphDataModule(**arguments)  # type: ignore[arg-type]
