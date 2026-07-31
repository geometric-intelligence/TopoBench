"""Native GraphMLP backbone and auxiliary-loss gates."""

import torch
import pytest
from torch_geometric.data import Data

from topobench.loss.model import GraphMLPLoss
from topobench.nn.backbones.graph import GraphMLP
from topobench.nn.wrappers.graph import GraphMLPWrapper


def test_graph_mlp_returns_only_embeddings_without_global_distance_matrix() -> None:
    model = GraphMLP(in_channels=4, hidden_channels=8)
    model.train()

    output = model(torch.randn(7, 4))

    assert isinstance(output, torch.Tensor)
    assert output.shape == (7, 8)


def test_graph_mlp_wrapper_preserves_exact_native_output_contract() -> None:
    data = Data(
        x=torch.randn(7, 4),
        y=torch.arange(7, dtype=torch.long) % 2,
        edge_index=torch.tensor(
            [[0, 1, 2, 3, 4, 5, 6], [1, 2, 0, 4, 5, 6, 3]],
            dtype=torch.long,
        ),
        batch=torch.zeros(7, dtype=torch.long),
    )
    wrapper = GraphMLPWrapper(
        GraphMLP(in_channels=4, hidden_channels=8),
        edge_attr_mode="reject",
        edge_weight_mode="reject",
    )

    output = wrapper(data)

    assert set(output) == {"x", "labels", "batch"}
    assert output["x"].shape == (7, 8)


def test_graph_mlp_loss_rejects_cross_graph_edges() -> None:
    data = Data(
        x=torch.randn(4, 8),
        edge_index=torch.tensor([[0, 1], [2, 3]], dtype=torch.long),
        batch=torch.tensor([0, 0, 1, 1], dtype=torch.long),
    )

    with pytest.raises(ValueError, match="crosses graph boundaries"):
        GraphMLPLoss()({"x": data.x}, data)
