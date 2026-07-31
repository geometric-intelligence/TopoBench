"""Behavioral tests for native homogeneous PyG graph loading."""

from __future__ import annotations

from collections.abc import Sequence

import pytest
import torch
from torch.utils.data import Dataset, RandomSampler, SequentialSampler, Subset
from torch_geometric.data import Batch, Data

from topobench.dataloader import GraphDataModule


class TinyGraphDataset(Dataset[Data]):
    """Small index-backed graph source with observable item access."""

    def __init__(self, graphs: Sequence[Data]) -> None:
        self.graphs = tuple(graphs)
        self.accesses = 0

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, index: int) -> Data:
        self.accesses += 1
        return self.graphs[index]


def _graph(num_nodes: int, label: int) -> Data:
    source = torch.arange(num_nodes, dtype=torch.long)
    target = (source + 1) % num_nodes
    return Data(
        x=torch.arange(num_nodes * 2, dtype=torch.float).view(num_nodes, 2),
        edge_index=torch.stack([source, target]),
        y=torch.tensor([label]),
    )


def test_graph_datamodule_uses_native_pyg_batching() -> None:
    """A homogeneous batch keeps native PyG fields and node assignments."""
    graph_a = _graph(2, 0)
    graph_b = _graph(3, 1)
    graph_c = _graph(4, 0)
    source = TinyGraphDataset([graph_a, graph_b, graph_c])
    module = GraphDataModule(
        dataset_train=Subset(source, [0, 1]),
        dataset_val=Subset(source, [0, 1]),
        dataset_test=Subset(source, [2]),
        learning_setting="inductive",
        batch_size=2,
        num_workers=0,
    )

    batch = next(iter(module.val_dataloader()))

    assert isinstance(batch, Batch)
    assert batch.num_graphs == 2
    assert batch.batch.tolist() == [0] * graph_a.num_nodes + [1] * graph_b.num_nodes
    assert "x_0" not in batch
    assert "batch_0" not in batch
    assert isinstance(module.train_dataloader().sampler, RandomSampler)
    assert isinstance(module.val_dataloader().sampler, SequentialSampler)
    assert isinstance(module.test_dataloader().sampler, SequentialSampler)


@pytest.mark.parametrize(
    ("name", "value", "error", "message"),
    [
        ("batch_size", True, TypeError, "batch_size must be an integer"),
        ("batch_size", 1.5, TypeError, "batch_size must be an integer"),
        ("batch_size", 0, ValueError, "batch_size must be positive"),
        ("num_workers", False, TypeError, "num_workers must be an integer"),
        ("num_workers", 1.5, TypeError, "num_workers must be an integer"),
        ("num_workers", -1, ValueError, "num_workers must be non-negative"),
    ],
)
def test_graph_datamodule_validates_numeric_loader_settings(
    name: str,
    value: object,
    error: type[Exception],
    message: str,
) -> None:
    """Boolean, fractional, and out-of-range loader settings are rejected."""
    source = TinyGraphDataset([_graph(2, 0), _graph(3, 1), _graph(4, 0)])
    kwargs = {"batch_size": 1, "num_workers": 0, name: value}

    with pytest.raises(error, match=f"^{message}$"):
        GraphDataModule(
            dataset_train=Subset(source, [0]),
            dataset_val=Subset(source, [1]),
            dataset_test=Subset(source, [2]),
            learning_setting="inductive",
            **kwargs,
        )


def test_zero_workers_disables_persistent_workers() -> None:
    """PyTorch's invalid zero-worker persistent mode is normalized away."""
    source = TinyGraphDataset([_graph(2, 0), _graph(3, 1), _graph(4, 0)])
    module = GraphDataModule(
        dataset_train=Subset(source, [0]),
        dataset_val=Subset(source, [1]),
        dataset_test=Subset(source, [2]),
        learning_setting="inductive",
        num_workers=0,
        persistent_workers=True,
    )

    assert module.train_dataloader().persistent_workers is False


@pytest.mark.parametrize("phase", ["train", "validation", "test"])
def test_inductive_loading_rejects_empty_phase_views(phase: str) -> None:
    """Every inductive phase must contain at least one graph."""
    source = TinyGraphDataset([_graph(2, 0), _graph(3, 1), _graph(4, 0)])
    views = {
        "train": Subset(source, [0]),
        "validation": Subset(source, [1]),
        "test": Subset(source, [2]),
    }
    views[phase] = Subset(source, [])

    with pytest.raises(ValueError, match=f"^dataset_{phase} must not be empty$"):
        GraphDataModule(
            dataset_train=views["train"],
            dataset_val=views["validation"],
            dataset_test=views["test"],
            learning_setting="inductive",
        )


@pytest.mark.parametrize("missing_phase", ["validation", "test"])
def test_inductive_loading_requires_every_phase(missing_phase: str) -> None:
    """Inductive loading never aliases or invents a missing phase."""
    source = TinyGraphDataset([_graph(2, 0), _graph(3, 1), _graph(4, 0)])
    dataset_val = None if missing_phase == "validation" else Subset(source, [1])
    dataset_test = None if missing_phase == "test" else Subset(source, [2])

    with pytest.raises(
        ValueError,
        match="^inductive loading requires train, validation, and test views$",
    ):
        GraphDataModule(
            dataset_train=Subset(source, [0]),
            dataset_val=dataset_val,
            dataset_test=dataset_test,
            learning_setting="inductive",
        )


def test_inductive_loading_requires_index_backed_views() -> None:
    """Materialized phase datasets cannot replace lazy source views."""
    source = TinyGraphDataset([_graph(2, 0), _graph(3, 1), _graph(4, 0)])

    with pytest.raises(
        ValueError,
        match="^inductive loading requires index-backed Subset views$",
    ):
        GraphDataModule(
            dataset_train=source,
            dataset_val=source,
            dataset_test=source,
            learning_setting="inductive",
        )


def test_inductive_loading_requires_one_shared_source() -> None:
    """All phase indices address one source dataset rather than graph copies."""
    first = TinyGraphDataset([_graph(2, 0), _graph(3, 1), _graph(4, 0)])
    second = TinyGraphDataset([_graph(2, 0), _graph(3, 1), _graph(4, 0)])

    with pytest.raises(
        ValueError,
        match="^inductive phase views must share one source dataset$",
    ):
        GraphDataModule(
            dataset_train=Subset(first, [0]),
            dataset_val=Subset(second, [1]),
            dataset_test=Subset(first, [2]),
            learning_setting="inductive",
        )


def test_graph_datamodule_rejects_unknown_learning_setting() -> None:
    """Learning mode selection is explicit rather than inferred from phases."""
    source = TinyGraphDataset([_graph(2, 0)])

    with pytest.raises(
        ValueError,
        match=r"^unsupported learning_setting: 'semi-supervised'$",
    ):
        GraphDataModule(
            dataset_train=source,
            learning_setting="semi-supervised",  # type: ignore[arg-type]
        )
