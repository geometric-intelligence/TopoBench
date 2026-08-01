"""Batch-isolated differentiable graph construction for a native GCN."""

from __future__ import annotations

from numbers import Integral

import torch
from torch import Tensor, nn
from torch_geometric.nn.models import GCN


class _StructureEncoder(nn.Module):
    """Embed nodes solely to learn differentiable auxiliary edge weights."""

    def __init__(self, in_channels: int, hidden_channels: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_channels, hidden_channels)
        self.activation = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        """Return structure embeddings."""
        return self.activation(self.linear(x))


class GCNDGM(nn.Module):
    """Run GCN over learned, strictly within-example auxiliary edges."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        k: int = 5,
        dropout: float = 0.0,
        act: str = "relu",
    ) -> None:
        super().__init__()
        for name, value in (
            ("in_channels", in_channels),
            ("hidden_channels", hidden_channels),
            ("num_layers", num_layers),
            ("k", k),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, Integral)
                or value < 1
            ):
                raise ValueError(f"{name} must be a positive integer")
        self.k = int(k)
        self.structure_encoder = _StructureEncoder(
            int(in_channels),
            int(hidden_channels),
        )
        self.log_temperature = nn.Parameter(torch.zeros(()))
        self.gcn = GCN(
            in_channels=int(in_channels),
            hidden_channels=int(hidden_channels),
            num_layers=int(num_layers),
            out_channels=int(hidden_channels),
            dropout=float(dropout),
            act=act,
        )
        self.last_auxiliary_edge_index: Tensor | None = None
        self.last_auxiliary_logprobs: Tensor | None = None

    def _validate_batch(self, x: Tensor, batch: Tensor | None) -> Tensor:
        """Return contiguous native graph membership for every node."""
        if not isinstance(x, Tensor) or x.ndim != 2:
            raise ValueError("x must be a rank-2 tensor")
        if x.size(0) == 0:
            raise ValueError("x must contain at least one node")
        if batch is None:
            return torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        if (
            not isinstance(batch, Tensor)
            or batch.ndim != 1
            or batch.dtype is not torch.long
            or batch.numel() != x.size(0)
        ):
            raise ValueError(
                "batch must be a rank-1 long tensor matching x rows"
            )
        graph_count = int(batch.max()) + 1
        if int(batch.min()) < 0 or not torch.equal(
            torch.unique(batch),
            torch.arange(graph_count, device=batch.device),
        ):
            raise ValueError("batch must be contiguous from zero")
        return batch

    def _learned_edges(
        self,
        structure: Tensor,
        batch: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Build nearest-neighbor edges independently inside each graph."""
        graph_count = int(batch.max()) + 1
        graph_nodes = [
            torch.nonzero(batch == graph_id, as_tuple=False).flatten()
            for graph_id in range(graph_count)
        ]
        edges: list[Tensor] = []
        edge_weights: list[Tensor] = []
        logprobs: list[Tensor] = []
        temperature = self.log_temperature.exp().clamp_min(1e-6)

        for nodes in graph_nodes:
            local_structure = structure[nodes]
            effective_k = min(
                self.k,
                max(int(nodes.numel()) - 1, 1),
            )
            if nodes.numel() == 1:
                local_targets = torch.zeros(
                    (1, effective_k),
                    dtype=torch.long,
                    device=nodes.device,
                )
                selected_distances = local_structure.new_zeros(
                    (1, effective_k)
                )
            else:
                distances = torch.cdist(local_structure, local_structure)
                distances = distances.masked_fill(
                    torch.eye(
                        nodes.numel(),
                        dtype=torch.bool,
                        device=nodes.device,
                    ),
                    torch.inf,
                )
                selected_distances, local_targets = torch.topk(
                    distances,
                    k=effective_k,
                    dim=1,
                    largest=False,
                    sorted=False,
                )
            local_logprobs = torch.log_softmax(
                -selected_distances / temperature,
                dim=1,
            )
            local_sources = (
                torch.arange(
                    nodes.numel(),
                    device=nodes.device,
                )
                .unsqueeze(1)
                .expand(-1, effective_k)
            )
            edges.append(
                torch.stack(
                    (
                        nodes[local_sources.reshape(-1)],
                        nodes[local_targets.reshape(-1)],
                    )
                )
            )
            edge_weights.append(local_logprobs.exp().reshape(-1))
            logprobs.append(local_logprobs.reshape(-1))

        edge_index = torch.cat(edges, dim=1)
        weights = torch.cat(edge_weights)
        return edge_index, weights, torch.cat(logprobs)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: Tensor | None = None,
        **_: object,
    ) -> Tensor:
        """Return only node embeddings from isolated learned topology."""
        del edge_index
        batch = self._validate_batch(x, batch)
        structure = self.structure_encoder(x)
        learned_edges, edge_weights, logprobs = self._learned_edges(
            structure,
            batch,
        )
        self.last_auxiliary_edge_index = learned_edges.detach()
        self.last_auxiliary_logprobs = logprobs.detach()
        return self.gcn(x, learned_edges, edge_weight=edge_weights)


__all__ = ["GCNDGM"]
