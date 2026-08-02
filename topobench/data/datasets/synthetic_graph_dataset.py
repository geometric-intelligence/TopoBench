"""Deterministic native homogeneous graph fixtures."""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset

from topobench.data.splits import apply_transductive_split

SyntheticGraphTask = Literal[
    "graph_classification",
    "graph_regression",
    "node_classification",
]


def _ring_edges(num_nodes: int) -> torch.Tensor:
    """Return a bidirectional ring edge index."""
    nodes = torch.arange(num_nodes, dtype=torch.long)
    successors = (nodes + 1) % num_nodes
    return torch.cat(
        [
            torch.stack([nodes, successors]),
            torch.stack([successors, nodes]),
        ],
        dim=1,
    )


def _graph_examples(
    *,
    task: Literal["graph_classification", "graph_regression"],
    generator: torch.Generator,
) -> list[Data]:
    """Build twelve learnable graph-level examples."""
    examples: list[Data] = []
    for graph_id in range(12):
        num_nodes = 4 + graph_id % 4
        if task == "graph_classification":
            label = graph_id % 2
            x = 0.02 * torch.randn(num_nodes, 4, generator=generator)
            x[:, label] += 1.0
            y = torch.tensor([label], dtype=torch.long)
        else:
            target = (graph_id + 1) / 4.0
            x = 0.02 * torch.randn(num_nodes, 4, generator=generator)
            x[:, 0] += target
            x[:, 1] += num_nodes / 10.0
            y = torch.tensor([target], dtype=torch.float)

        data = Data(x=x, edge_index=_ring_edges(num_nodes), y=y)
        data.validate(raise_on_error=True)
        examples.append(data)
    return examples


def _node_classification_example(
    *,
    generator: torch.Generator,
) -> Data:
    """Build one node-classification graph with a complete fixed split."""
    num_nodes = 18
    labels = torch.arange(num_nodes, dtype=torch.long) % 2
    x = 0.02 * torch.randn(num_nodes, 4, generator=generator)
    x[torch.arange(num_nodes), labels] += 1.0
    data = Data(x=x, edge_index=_ring_edges(num_nodes), y=labels)
    apply_transductive_split(
        data,
        train=torch.arange(0, 10),
        val=torch.arange(10, 14),
        test=torch.arange(14, 18),
    )
    data.validate(raise_on_error=True)
    return data


class SyntheticGraphDataset(InMemoryDataset):
    """Package deterministic native PyG fixtures for homogeneous graph tasks.

    Parameters
    ----------
    task : {"graph_classification", "graph_regression", "node_classification"}
        Fixture contract to construct.
    seed : int, default=0
        Seed for a fixture-local random generator.
    """

    feature_policy = "continuous"
    representation_version = "pyg-data-v1"
    parser_version = "synthetic-graph-v1"

    @property
    def cache_parameters(self) -> dict[str, int | str]:
        """Return effective loader defaults that determine fixture content."""
        return {"task": self.task, "seed": self.seed}

    def __init__(
        self,
        *,
        task: SyntheticGraphTask = "graph_classification",
        seed: int = 0,
    ) -> None:
        if task not in {
            "graph_classification",
            "graph_regression",
            "node_classification",
        }:
            raise ValueError(f"unsupported synthetic graph task: {task!r}")
        self.task = task
        self.seed = int(seed)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed)

        if task == "node_classification":
            examples = [_node_classification_example(generator=generator)]
        else:
            examples = _graph_examples(task=task, generator=generator)

        super().__init__(root=None)
        self.data, self.slices = self.collate(examples)
        if task != "node_classification":
            self.split_idx = {
                "train": np.arange(0, 8, dtype=np.int64),
                "valid": np.arange(8, 10, dtype=np.int64),
                "test": np.arange(10, 12, dtype=np.int64),
            }


__all__ = ["SyntheticGraphDataset", "SyntheticGraphTask"]
