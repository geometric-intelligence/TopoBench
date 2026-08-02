"""Batch-isolated differentiable graph construction for a native GCN."""

from __future__ import annotations

from numbers import Integral

import torch
from torch import Tensor, nn
from torch_geometric.nn.models import GCN

_DEFAULT_QUERY_CHUNK_SIZE = 256
_DEFAULT_MAX_NODES = 20_000
_DEFAULT_MAX_WORKSPACE_BYTES = 768 * 1024**2
_MAX_DISTANCE_ELEMENT_SIZE = 8
_LONG_BYTES = 8
_BOOL_BYTES = 1
_SELECTION_FLOAT_TENSORS = 3
_SELECTION_INDEX_TENSORS = 2
_SELECTION_BOOL_TENSORS = 2
_PAIR_FEATURE_TENSORS = 6
_PAIR_SCALAR_TENSORS = 10
_PAIR_INDEX_TENSORS = 3


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
        query_chunk_size: int = _DEFAULT_QUERY_CHUNK_SIZE,
        max_nodes: int = _DEFAULT_MAX_NODES,
        max_workspace_bytes: int = _DEFAULT_MAX_WORKSPACE_BYTES,
    ) -> None:
        super().__init__()
        for name, value in (
            ("in_channels", in_channels),
            ("hidden_channels", hidden_channels),
            ("num_layers", num_layers),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, Integral)
                or value < 1
            ):
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(k, bool) or not isinstance(k, Integral):
            raise TypeError("k must be an integer")
        if k < 2:
            raise ValueError("k must be at least 2")
        for name, value in (
            ("query_chunk_size", query_chunk_size),
            ("max_nodes", max_nodes),
            ("max_workspace_bytes", max_workspace_bytes),
        ):
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer")
            if value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if max_nodes <= k:
            raise ValueError("max_nodes must be greater than k")
        if query_chunk_size > max_nodes:
            raise ValueError("query_chunk_size must not exceed max_nodes")
        required_workspace = self.estimate_workspace_bytes(
            node_count=int(max_nodes),
            query_chunk_size=int(query_chunk_size),
            k=int(k),
            feature_dim=int(hidden_channels),
            element_size=_MAX_DISTANCE_ELEMENT_SIZE,
        )
        if max_workspace_bytes < required_workspace:
            raise ValueError(
                f"max_workspace_bytes={max_workspace_bytes} cannot admit "
                f"max_nodes={max_nodes} with "
                f"query_chunk_size={query_chunk_size}; requires "
                f"{required_workspace} bytes"
            )
        self.k = int(k)
        self.query_chunk_size = int(query_chunk_size)
        self.max_nodes = int(max_nodes)
        self.max_workspace_bytes = int(max_workspace_bytes)
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

    @staticmethod
    def estimate_workspace_bytes(
        *,
        node_count: int,
        query_chunk_size: int,
        k: int,
        feature_dim: int,
        element_size: int,
    ) -> int:
        """Conservatively bound exact-selection and selected-pair tensors."""
        chunk_rows = min(query_chunk_size, max(node_count - 1, 1))
        effective_k = min(k, max(node_count - 1, 1))
        chunk_elements = chunk_rows * node_count
        selected_pairs = node_count * effective_k
        selection_bytes = chunk_elements * (
            _SELECTION_FLOAT_TENSORS * element_size
            + _SELECTION_INDEX_TENSORS * _LONG_BYTES
            + _SELECTION_BOOL_TENSORS * _BOOL_BYTES
        )
        retained_pair_bytes = selected_pairs * (
            _PAIR_FEATURE_TENSORS * feature_dim * element_size
            + _PAIR_SCALAR_TENSORS * element_size
            + _PAIR_INDEX_TENSORS * _LONG_BYTES
        )
        return selection_bytes + retained_pair_bytes

    @staticmethod
    def _canonical_topk(
        distances: Tensor,
        k: int,
    ) -> tuple[Tensor, Tensor]:
        """Select exact nearest neighbors, breaking equal distances by index."""
        selected_distances, selected_indices = torch.topk(
            distances,
            k=k,
            dim=1,
            largest=False,
            sorted=True,
        )
        threshold = selected_distances[:, -1:]
        candidate_indices = torch.arange(
            distances.size(1),
            device=distances.device,
        ).expand(distances.size(0), -1)
        threshold_candidates = torch.where(
            distances == threshold,
            candidate_indices,
            distances.size(1),
        )
        canonical_threshold = torch.topk(
            threshold_candidates,
            k=k,
            dim=1,
            largest=False,
            sorted=True,
        ).values
        below_threshold = selected_distances < threshold
        threshold_rank = (~below_threshold).cumsum(dim=1) - 1
        canonical_ties = canonical_threshold.gather(
            1,
            threshold_rank.clamp_min(0),
        )
        selected_indices = torch.where(
            below_threshold,
            selected_indices,
            canonical_ties,
        )
        selected_indices = selected_indices.sort(dim=1).values
        selected_distances = distances.gather(1, selected_indices)
        distance_order = torch.argsort(
            selected_distances,
            dim=1,
            stable=True,
        )
        selected_indices = selected_indices.gather(1, distance_order)
        return distances.gather(1, selected_indices), selected_indices

    @staticmethod
    @torch.no_grad()
    def _select_chunk_indices(
        queries: Tensor,
        candidates: Tensor,
        *,
        query_start: int,
        k: int,
    ) -> Tensor:
        """Select canonical neighbors without retaining pairwise autograd state."""
        distances = torch.cdist(queries, candidates)
        candidate_indices = torch.arange(
            candidates.size(0),
            device=candidates.device,
        ).expand(queries.size(0), -1)
        query_indices = torch.arange(
            query_start,
            query_start + queries.size(0),
            device=queries.device,
        )
        distances = distances.masked_fill(
            candidate_indices == query_indices.unsqueeze(1),
            torch.inf,
        )
        return GCNDGM._canonical_topk(distances, k)[1]

    def _chunked_neighbors(
        self,
        structure: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Return exact local distances and indices without an N-by-N tensor."""
        node_count = int(structure.size(0))
        if node_count > self.max_nodes:
            raise ValueError(
                f"GCN-DGM node count {node_count} exceeds "
                f"max_nodes={self.max_nodes} before distance allocation"
            )
        effective_k = min(self.k, max(node_count - 1, 1))
        if node_count == 1:
            return (
                structure.new_zeros((1, effective_k)),
                torch.zeros(
                    (1, effective_k),
                    dtype=torch.long,
                    device=structure.device,
                ),
            )
        required_workspace = self.estimate_workspace_bytes(
            node_count=node_count,
            query_chunk_size=self.query_chunk_size,
            k=effective_k,
            feature_dim=int(structure.size(1)),
            element_size=structure.element_size(),
        )
        if required_workspace > self.max_workspace_bytes:
            raise ValueError(
                f"GCN-DGM node count {node_count} requires "
                f"{required_workspace} workspace bytes, exceeding "
                f"max_workspace_bytes={self.max_workspace_bytes} before "
                "distance allocation"
            )

        selected_indices: list[Tensor] = []
        chunk_size = min(self.query_chunk_size, node_count - 1)
        for start in range(0, node_count, chunk_size):
            stop = min(start + chunk_size, node_count)
            selected_indices.append(
                self._select_chunk_indices(
                    structure[start:stop],
                    structure,
                    query_start=start,
                    k=effective_k,
                )
            )
        local_targets = torch.cat(selected_indices)
        selected_differences = (
            structure[local_targets] - structure.unsqueeze(1)
        )
        selected_distances = torch.linalg.vector_norm(
            selected_differences,
            dim=-1,
        )
        return selected_distances, local_targets

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
            selected_distances, local_targets = self._chunked_neighbors(
                local_structure
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
                        nodes[local_targets.reshape(-1)],
                        nodes[local_sources.reshape(-1)],
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
