"""Per-graph contrastive loss for the native GraphMLP candidate."""

from __future__ import annotations

from numbers import Integral

import torch
import torch.nn.functional as functional
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import remove_self_loops, to_dense_adj

from topobench.loss.base import AbstractLoss


class GraphMLPLoss(AbstractLoss):
    """Compute GraphMLP contrastive terms without cross-graph pairs."""

    def __init__(
        self,
        r_adj_power: int = 2,
        tau: float = 1.0,
        loss_weight: float = 0.5,
    ) -> None:
        super().__init__()
        if (
            isinstance(r_adj_power, bool)
            or not isinstance(r_adj_power, Integral)
            or r_adj_power < 1
        ):
            raise ValueError("r_adj_power must be a positive integer")
        self.r_adj_power = int(r_adj_power)
        self.tau = float(tau)
        self.loss_weight = float(loss_weight)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"r_adj_power={self.r_adj_power}, tau={self.tau}, "
            f"loss_weight={self.loss_weight})"
        )

    def get_power_adj(
        self,
        edge_index: Tensor,
        *,
        num_nodes: int,
    ) -> Tensor:
        """Return a dense powered adjacency for one graph only."""
        edge_index, _ = remove_self_loops(edge_index)
        adjacency = to_dense_adj(
            edge_index,
            max_num_nodes=num_nodes,
        ).squeeze(0)
        adjacency_power = adjacency
        for _ in range(self.r_adj_power - 1):
            adjacency_power = adjacency_power @ adjacency
        return adjacency_power

    def graph_mlp_contrast_loss(
        self,
        similarity: Tensor,
        adjacency: Tensor,
    ) -> Tensor:
        """Return the mean node contrastive loss for one graph."""
        exponentiated = torch.exp(self.tau * similarity)
        denominator = exponentiated.sum(dim=1)
        positive = (exponentiated * adjacency).sum(dim=1)
        return -torch.log(positive / denominator + 1e-8).mean()

    def forward(self, model_out: dict, batch: Data) -> Tensor:
        """Compute isolated per-graph terms from native ``x`` and ``batch``."""
        embeddings = model_out.get("x")
        if not isinstance(embeddings, Tensor) or embeddings.ndim != 2:
            raise TypeError("model_out['x'] must be a rank-2 tensor")
        model_state = batch.get("model_state")
        valid_model_states = ("Training", "Validation", "Test")
        if (
            not isinstance(model_state, str)
            or model_state not in valid_model_states
        ):
            raise ValueError(
                "batch['model_state'] must be one of "
                "'Training', 'Validation', or 'Test'; "
                f"got {model_state!r}"
            )
        if model_state != "Training":
            return embeddings.new_zeros(())
        edge_index = batch.get("edge_index")
        if (
            not isinstance(edge_index, Tensor)
            or edge_index.ndim != 2
            or edge_index.size(0) != 2
        ):
            raise ValueError("batch.edge_index must have shape [2, E]")
        batch_index = model_out.get("batch", batch.get("batch"))
        if (
            not isinstance(batch_index, Tensor)
            or batch_index.ndim != 1
            or batch_index.dtype is not torch.long
            or batch_index.numel() != embeddings.size(0)
        ):
            raise ValueError(
                "model_out['batch'] must be a rank-1 long tensor "
                "matching model_out['x']"
            )
        if batch_index.numel() == 0:
            raise ValueError("model_out['batch'] must not be empty")
        graph_count = int(batch_index.max()) + 1
        if int(batch_index.min()) < 0 or not torch.equal(
            torch.unique(batch_index),
            torch.arange(graph_count, device=batch_index.device),
        ):
            raise ValueError("model_out['batch'] must be contiguous from zero")
        if edge_index.numel() and not torch.equal(
            batch_index[edge_index[0]],
            batch_index[edge_index[1]],
        ):
            raise ValueError("batch.edge_index crosses graph boundaries")

        local_positions = torch.empty_like(batch_index)
        node_indices: list[Tensor] = []
        for graph_id in range(graph_count):
            indices = torch.nonzero(
                batch_index == graph_id,
                as_tuple=False,
            ).flatten()
            if indices.numel() == 0:
                raise ValueError("model_out['batch'] contains an empty graph")
            local_positions[indices] = torch.arange(
                indices.numel(),
                device=indices.device,
            )
            node_indices.append(indices)

        weighted_loss = embeddings.new_zeros(())
        normalized = functional.normalize(embeddings, p=2, dim=-1)
        for graph_id, indices in enumerate(node_indices):
            edge_mask = batch_index[edge_index[0]] == graph_id
            local_edges = local_positions[edge_index[:, edge_mask]]
            adjacency = self.get_power_adj(
                local_edges,
                num_nodes=int(indices.numel()),
            )
            similarity = normalized[indices] @ normalized[indices].T
            similarity = similarity.masked_fill(
                torch.eye(
                    indices.numel(),
                    dtype=torch.bool,
                    device=indices.device,
                ),
                0,
            )
            graph_loss = self.graph_mlp_contrast_loss(
                similarity,
                adjacency,
            )
            weighted_loss = weighted_loss + indices.numel() * graph_loss
        return self.loss_weight * weighted_loss / embeddings.size(0)
