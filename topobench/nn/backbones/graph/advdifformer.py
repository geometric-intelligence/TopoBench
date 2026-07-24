"""Advective Diffusion Transformer graph backbone.

This module implements a TopoBench-compatible AdvDIFFormer encoder following
Wu et al., "Supercharging Graph Transformers with Advective Diffusion"
(ICML 2025) and the official ``qitianwu/AdvDIFFormer`` implementation.
The propagation layer corresponds to the model's global attentive diffusion
plus observed-graph advection operator.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.utils import coalesce


class _PreparedGraph:
    """Row-normalized graph data reused throughout one encoder forward."""

    __slots__ = ("dst", "norm_weight", "num_nodes", "src")

    def __init__(self, src, dst, norm_weight, num_nodes) -> None:
        self.src = src
        self.dst = dst
        self.norm_weight = norm_weight
        self.num_nodes = num_nodes

    def matmul(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the prepared row-normalized adjacency to node features."""
        if self.src.numel() == 0:
            return torch.zeros_like(x)
        messages = x.index_select(0, self.src) * self.norm_weight.unsqueeze(-1)
        out = torch.zeros_like(x)
        out.index_add_(0, self.dst, messages)
        return out


class _PreparedBatch:
    """Canonical graph assignments and contiguous graph segments."""

    __slots__ = (
        "batch",
        "counts",
        "counts_list",
        "num_graphs",
        "ptr",
        "restore_indices",
        "sorted_batch",
        "sorted_indices",
    )

    def __init__(
        self,
        batch,
        sorted_batch,
        ptr,
        counts,
        counts_list,
        sorted_indices,
        restore_indices,
    ) -> None:
        self.batch = batch
        self.sorted_batch = sorted_batch
        self.ptr = ptr
        self.counts = counts
        self.counts_list = counts_list
        self.num_graphs = counts.numel()
        self.sorted_indices = sorted_indices
        self.restore_indices = restore_indices


def _prepare_batch(batch: torch.Tensor | None, num_nodes: int, device) -> _PreparedBatch:
    """Canonicalize graph IDs and build contiguous segments once."""
    identity = torch.arange(num_nodes, device=device)
    if batch is None or num_nodes == 0:
        canonical = torch.zeros(num_nodes, dtype=torch.long, device=device)
        counts = torch.tensor([num_nodes], dtype=torch.long, device=device)
        ptr = torch.tensor([0, num_nodes], dtype=torch.long, device=device)
        return _PreparedBatch(
            canonical,
            canonical,
            ptr,
            counts,
            [num_nodes],
            identity,
            identity,
        )

    _, canonical = torch.unique(batch, sorted=True, return_inverse=True)
    counts = torch.bincount(canonical)
    ptr = torch.cat([counts.new_zeros(1), counts.cumsum(0)])

    is_contiguous = bool(
        num_nodes < 2 or torch.all(canonical[1:] >= canonical[:-1]).item()
    )
    if is_contiguous:
        sorted_indices = identity
        restore_indices = identity
        sorted_batch = canonical
    else:
        sorted_indices = torch.argsort(canonical, stable=True)
        restore_indices = torch.empty_like(sorted_indices)
        restore_indices[sorted_indices] = identity
        sorted_batch = canonical.index_select(0, sorted_indices)

    return _PreparedBatch(
        canonical,
        sorted_batch,
        ptr,
        counts,
        counts.tolist(),
        sorted_indices,
        restore_indices,
    )


def _prepare_graph(
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor | None,
    num_nodes: int,
    dtype: torch.dtype,
    device: torch.device,
    make_undirected: bool,
) -> _PreparedGraph:
    """Prepare receiving-node row normalization once per encoder forward."""
    if edge_index.numel() == 0:
        empty = edge_index.new_empty(0)
        return _PreparedGraph(empty, empty, torch.empty(0, dtype=dtype, device=device), num_nodes)

    edge_index = edge_index.to(device=device)
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.shape[1], dtype=dtype, device=device)
    else:
        edge_weight = edge_weight.to(dtype=dtype, device=device)
    if make_undirected:
        edge_index, edge_weight = _as_undirected(edge_index, edge_weight)

    src, dst = edge_index
    degree = torch.zeros(num_nodes, dtype=dtype, device=device)
    degree.index_add_(0, dst, edge_weight)
    inverse_degree = torch.zeros_like(degree)
    positive = degree > 0
    inverse_degree[positive] = degree[positive].reciprocal()
    return _PreparedGraph(src, dst, inverse_degree[dst] * edge_weight, num_nodes)


def _as_undirected(
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Return an undirected version of an edge list."""
    if edge_index.numel() == 0:
        return edge_index, edge_weight

    rev_edge_index = edge_index.flip(0)
    edge_index = torch.cat([edge_index, rev_edge_index], dim=1)

    if edge_weight is not None:
        edge_weight = torch.cat([edge_weight, edge_weight], dim=0)

    return coalesce(edge_index, edge_weight, reduce="mean")


def _normalized_adjacency_matmul(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor | None = None,
    make_undirected: bool = False,
) -> torch.Tensor:
    """Compute D^{-1} A X with sparse edge indices.

    The official implementation constructs a transposed sparse adjacency and
    left-normalizes it by the receiving-node degree before multiplying node
    features, which is equivalent to row-normalized message aggregation.
    """
    prepared = _prepare_graph(
        edge_index,
        edge_weight,
        x.shape[0],
        x.dtype,
        x.device,
        make_undirected,
    )
    return prepared.matmul(x)


def _dense_normalized_adjacency(
    num_nodes: int,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor | None,
    device: torch.device,
    dtype: torch.dtype,
    make_undirected: bool = False,
) -> torch.Tensor:
    """Build a dense row-normalized adjacency matrix."""
    if edge_index.numel() == 0:
        return torch.zeros(num_nodes, num_nodes, device=device, dtype=dtype)

    if edge_weight is None:
        edge_weight = torch.ones(edge_index.shape[1], device=device, dtype=dtype)
    else:
        edge_weight = edge_weight.to(dtype=dtype, device=device)

    if make_undirected:
        edge_index, edge_weight = _as_undirected(edge_index, edge_weight)

    row, col = edge_index
    deg = torch.zeros(num_nodes, device=device, dtype=dtype)
    deg.index_add_(0, col, edge_weight)
    deg_inv = torch.zeros_like(deg)
    nonzero = deg > 0
    deg_inv[nonzero] = deg[nonzero].reciprocal()
    norm = deg_inv[col] * edge_weight

    adj = torch.zeros(num_nodes, num_nodes, device=device, dtype=dtype)
    adj[col, row] = norm
    return adj


class AdvDIFFormerLayer(nn.Module):
    """One AdvDIFFormer propagation layer.

    Parameters
    ----------
    hidden_dim : int
        Dimension of node embeddings.
    heads : int
        Number of propagation heads.
    variant : str
        Propagation variant. ``"series"`` uses the scalable polynomial
        propagation, while ``"inverse"`` uses a dense linear solve. The
        aliases ``"s"`` and ``"i"`` are accepted for compatibility.
    propagation_steps : int
        Number of propagation powers, called ``K_order`` in the official
        implementation, for the scalable variant.
    beta : float
        Weight of observed-graph advection.
    theta : float
        Identity coefficient for the inverse variant.
    dropout : float
        Dropout rate applied inside the layer.
    head_aggregation : str
        Either ``"sum"`` or ``"mean"`` for combining heads.
    make_undirected : bool
        Whether to symmetrize the input edge list before local propagation.
    """

    def __init__(
        self,
        hidden_dim: int,
        heads: int = 1,
        variant: str = "series",
        propagation_steps: int = 2,
        beta: float = 0.5,
        theta: float = 0.0,
        dropout: float = 0.0,
        head_aggregation: str = "sum",
        make_undirected: bool = False,
    ) -> None:
        super().__init__()
        variant_aliases = {"s": "series", "i": "inverse"}
        variant = variant_aliases.get(variant, variant)
        if variant not in {"series", "inverse"}:
            raise ValueError("variant must be 'series'/'s' or 'inverse'/'i'.")
        if propagation_steps < 1:
            raise ValueError("propagation_steps must be at least 1.")
        if head_aggregation not in {"sum", "mean"}:
            raise ValueError("head_aggregation must be either 'sum' or 'mean'.")

        self.hidden_dim = hidden_dim
        self.heads = heads
        self.variant = variant
        self.propagation_steps = propagation_steps
        self.beta = beta
        self.theta = theta
        self.head_aggregation = head_aggregation
        self.make_undirected = make_undirected

        self.query = nn.ModuleList(
            nn.Linear(hidden_dim, hidden_dim, bias=False)
            for _ in range(heads)
        )
        self.key = nn.ModuleList(
            nn.Linear(hidden_dim, hidden_dim, bias=False)
            for _ in range(heads)
        )

        output_input_dim = (
            hidden_dim * (propagation_steps + 1)
            if variant == "series"
            else hidden_dim
        )
        self.output = nn.ModuleList(
            nn.Linear(output_input_dim, hidden_dim, bias=False)
            for _ in range(heads)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor | None = None,
        edge_weight: torch.Tensor | None = None,
        prepared_graph: _PreparedGraph | None = None,
        prepared_batch: _PreparedBatch | None = None,
    ) -> torch.Tensor:
        """Forward pass."""
        if prepared_batch is None:
            prepared_batch = _prepare_batch(batch, x.shape[0], x.device)
        batch = prepared_batch.batch
        if prepared_graph is None:
            prepared_graph = _prepare_graph(
                edge_index,
                edge_weight,
                x.shape[0],
                x.dtype,
                x.device,
                self.make_undirected,
            )

        if self.variant == "series":
            return self._forward_series_all_heads(x, prepared_graph, prepared_batch)

        head_outputs = []
        for head in range(self.heads):
            q = F.normalize(self.query[head](x), p=2, dim=-1, eps=1e-12)
            k = F.normalize(self.key[head](x), p=2, dim=-1, eps=1e-12)

            out = self._forward_inverse(
                x, q, k, edge_index, batch, edge_weight
            )
            projected = self.output[head](out)
            head_outputs.append(projected)

        out = torch.stack(head_outputs, dim=0).sum(dim=0)
        if self.head_aggregation == "mean":
            out = out / self.heads
        return self.dropout(out)

    def _forward_series_all_heads(
        self,
        x: torch.Tensor,
        graph: _PreparedGraph,
        batch: _PreparedBatch,
    ) -> torch.Tensor:
        """Apply AdvDIFFormer-S while batching heads in the hot path."""
        q = torch.stack(
            [
                F.normalize(query(x), p=2, dim=-1, eps=1e-12)
                for query in self.query
            ],
            dim=0,
        )
        k = torch.stack(
            [
                F.normalize(key(x), p=2, dim=-1, eps=1e-12)
                for key in self.key
            ],
            dim=0,
        )
        context = self._prepare_multihead_attention_context(q, k, batch)
        weights = torch.stack([output.weight for output in self.output], dim=0)
        weight_blocks = weights.split(self.hidden_dim, dim=2)

        current = x.unsqueeze(0).expand(self.heads, -1, -1)
        projected = torch.einsum("hni,hoi->hno", current, weight_blocks[0])
        for step in range(self.propagation_steps):
            attentive = self._linear_multihead_attention_prepared(current, context)
            advective = self._multihead_graph_matmul(graph, current)
            current = attentive + self.beta * advective
            projected = projected + torch.einsum(
                "hni,hoi->hno",
                current,
                weight_blocks[step + 1],
            )

        out = projected.sum(dim=0)
        if self.head_aggregation == "mean":
            out = out / self.heads
        return self.dropout(out)

    def _multihead_graph_matmul(
        self,
        graph: _PreparedGraph,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Apply local propagation to all heads with one graph aggregation."""
        flat = x.permute(1, 0, 2).reshape(x.shape[1], self.heads * self.hidden_dim)
        out = graph.matmul(flat)
        return out.view(x.shape[1], self.heads, self.hidden_dim).permute(1, 0, 2)

    def _forward_scalable_projected(
        self,
        x: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        graph: _PreparedGraph,
        batch: _PreparedBatch,
        output: nn.Linear,
    ) -> torch.Tensor:
        """Propagate while applying output weight blocks incrementally."""
        context = self._prepare_attention_context(q, k, batch)
        weights = output.weight.split(self.hidden_dim, dim=1)
        projected = F.linear(x, weights[0])
        current = x
        for step in range(self.propagation_steps):
            attentive = self._linear_attention_prepared(q, k, current, context)
            current = attentive + self.beta * graph.matmul(current)
            projected = projected + F.linear(current, weights[step + 1])
        return projected

    def _forward_scalable(
        self,
        x: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        edge_weight: torch.Tensor | None,
    ) -> torch.Tensor:
        """Apply the linear-complexity AdvDIFFormer-S propagation."""
        states = [x]
        current = x
        for _ in range(self.propagation_steps):
            attentive = self._linear_attention_matmul(q, k, current, batch)

            advective = _normalized_adjacency_matmul(
                current,
                edge_index,
                edge_weight=edge_weight,
                make_undirected=self.make_undirected,
            )
            current = attentive + self.beta * advective
            states.append(current)

        return torch.cat(states, dim=-1)

    def _forward_inverse(
        self,
        x: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        edge_weight: torch.Tensor | None,
    ) -> torch.Tensor:
        """Apply the dense AdvDIFFormer-I linear solve."""
        num_nodes = x.shape[0]
        attention = torch.zeros(
            num_nodes, num_nodes, device=x.device, dtype=x.dtype
        )
        for graph_id in batch.unique(sorted=True):
            mask = batch == graph_id
            idx = mask.nonzero(as_tuple=True)[0]
            attention[idx[:, None], idx[None, :]] = self._dense_attention(
                q[mask], k[mask]
            )

        adj = _dense_normalized_adjacency(
            num_nodes,
            edge_index,
            edge_weight,
            device=x.device,
            dtype=x.dtype,
            make_undirected=self.make_undirected,
        )
        identity = torch.eye(num_nodes, device=x.device, dtype=x.dtype)
        operator = (1 + self.theta) * identity - attention - self.beta * adj

        jitter = 1e-6 * identity
        return torch.linalg.solve(operator + jitter, x)

    @staticmethod
    def _linear_attention_matmul(
        q: torch.Tensor,
        k: torch.Tensor,
        values: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """Compute C values for eta(q, k) = 1 + cosine(q, k).

        This vectorized form is algebraically the same as applying the
        operation independently to each graph in a batch, but avoids an inner
        Python loop over graphs for every layer, head, and propagation step.
        """
        prepared = _prepare_batch(batch, values.shape[0], values.device)
        context = AdvDIFFormerLayer._prepare_attention_context(q, k, prepared)
        return AdvDIFFormerLayer._linear_attention_prepared(q, k, values, context)

    @staticmethod
    def _prepare_attention_context(
        q: torch.Tensor,
        k: torch.Tensor,
        batch: _PreparedBatch,
    ) -> tuple[
        _PreparedBatch,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Compute graph-local denominators once for all propagation steps."""
        q_sorted = q.index_select(0, batch.sorted_indices)
        k_sorted = k.index_select(0, batch.sorted_indices)
        key_sums = k.new_zeros(batch.num_graphs, k.shape[-1])
        key_sums.index_add_(0, batch.sorted_batch, k_sorted)

        counts = batch.counts.to(dtype=q.dtype).unsqueeze(-1)
        denominator = counts.index_select(0, batch.sorted_batch)
        denominator = denominator + (q_sorted * key_sums.index_select(0, batch.sorted_batch)).sum(dim=-1, keepdim=True)
        return batch, q_sorted, k_sorted, denominator.clamp_min(1e-12)

    @staticmethod
    def _linear_attention_prepared(
        q: torch.Tensor,
        k: torch.Tensor,
        values: torch.Tensor,
        context: tuple[
            _PreparedBatch,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
    ) -> torch.Tensor:
        """Graphwise linear attention without a node-wise outer-product tensor."""
        batch, q_sorted, k_sorted, denominator = context
        values_sorted = values.index_select(0, batch.sorted_indices)
        parts = []
        q_parts = q_sorted.split(batch.counts_list, dim=0)
        k_parts = k_sorted.split(batch.counts_list, dim=0)
        value_parts = values_sorted.split(batch.counts_list, dim=0)
        denominator_parts = denominator.split(batch.counts_list, dim=0)
        for q_graph, k_graph, value_graph, denominator_graph in zip(
            q_parts,
            k_parts,
            value_parts,
            denominator_parts,
            strict=True,
        ):
            numerator = value_graph.sum(dim=0, keepdim=True) + q_graph @ (
                k_graph.T @ value_graph
            )
            parts.append(numerator / denominator_graph)

        out_sorted = parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)
        return out_sorted.index_select(0, batch.restore_indices)

    @staticmethod
    def _prepare_multihead_attention_context(
        q: torch.Tensor,
        k: torch.Tensor,
        batch: _PreparedBatch,
    ) -> tuple[_PreparedBatch, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute graph-local denominators for all heads at once."""
        q_sorted = q.index_select(1, batch.sorted_indices)
        k_sorted = k.index_select(1, batch.sorted_indices)
        heads, _, hidden_dim = k_sorted.shape
        graph_ids = batch.sorted_batch.unsqueeze(0) + torch.arange(heads, device=k.device).unsqueeze(-1) * batch.num_graphs

        key_sums = k.new_zeros(heads * batch.num_graphs, hidden_dim)
        key_sums.index_add_(0, graph_ids.reshape(-1), k_sorted.reshape(-1, hidden_dim))
        key_sums = key_sums.view(heads, batch.num_graphs, hidden_dim)

        counts = batch.counts.to(dtype=q.dtype).view(1, batch.num_graphs, 1)
        denominator = counts.index_select(1, batch.sorted_batch)
        denominator = denominator + (q_sorted * key_sums.index_select(1, batch.sorted_batch)).sum(dim=-1, keepdim=True)
        return batch, q_sorted, k_sorted, denominator.clamp_min(1e-12)

    @staticmethod
    def _linear_multihead_attention_prepared(
        values: torch.Tensor,
        context: tuple[_PreparedBatch, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Graphwise linear attention for values shaped [heads, nodes, dim]."""
        batch, q_sorted, k_sorted, denominator = context
        values_sorted = values.index_select(1, batch.sorted_indices)
        parts = []
        q_parts = q_sorted.split(batch.counts_list, dim=1)
        k_parts = k_sorted.split(batch.counts_list, dim=1)
        value_parts = values_sorted.split(batch.counts_list, dim=1)
        denominator_parts = denominator.split(batch.counts_list, dim=1)
        for q_graph, k_graph, value_graph, denominator_graph in zip(
            q_parts,
            k_parts,
            value_parts,
            denominator_parts,
            strict=True,
        ):
            key_value = k_graph.transpose(-1, -2) @ value_graph
            numerator = value_graph.sum(dim=1, keepdim=True) + q_graph @ key_value
            parts.append(numerator / denominator_graph)

        out_sorted = parts[0] if len(parts) == 1 else torch.cat(parts, dim=1)
        return out_sorted.index_select(1, batch.restore_indices)

    @staticmethod
    def _dense_attention(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """Build the dense row-normalized positive similarity matrix."""
        sim = 1 + q @ k.T
        return sim / sim.sum(dim=-1, keepdim=True).clamp(min=1e-12)


class AdvDIFFormerEncoder(nn.Module):
    """Advective Diffusion Transformer encoder for graph node embeddings.

    Parameters
    ----------
    input_dim : int
        Dimension of input node features.
    hidden_dim : int
        Dimension of hidden node embeddings.
    num_layers : int, optional
        Number of stacked AdvDIFFormer propagation layers.
    heads : int, optional
        Number of heads in each propagation layer.
    variant : str, optional
        ``"series"``/``"s"`` for AdvDIFFormer-S or ``"inverse"``/``"i"``
        for AdvDIFFormer-I.
    propagation_steps : int, optional
        Number of propagation powers for AdvDIFFormer-S.
    beta : float, optional
        Weight of observed-graph advection.
    theta : float, optional
        Identity coefficient for AdvDIFFormer-I.
    dropout : float, optional
        Dropout rate.
    input_dropout : float, optional
        Dropout applied after the input projection.
    residual : bool, optional
        Whether to use residual connections around propagation layers.
    layer_norm : bool, optional
        Whether to apply layer normalization after each layer.
    head_aggregation : str, optional
        Either ``"sum"`` or ``"mean"``.
    make_undirected : bool, optional
        Whether to symmetrize the input edge list before local propagation.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        heads: int = 1,
        variant: str = "series",
        propagation_steps: int = 2,
        beta: float = 0.5,
        theta: float = 0.0,
        dropout: float = 0.0,
        input_dropout: float = 0.0,
        residual: bool = True,
        layer_norm: bool = True,
        head_aggregation: str = "sum",
        make_undirected: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1.")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.out_channels = hidden_dim
        self.num_layers = num_layers
        self.heads = heads
        variant_aliases = {"s": "series", "i": "inverse"}
        self.variant = variant_aliases.get(variant, variant)
        self.propagation_steps = propagation_steps
        self.beta = beta
        self.theta = theta
        self.residual = residual
        self.layer_norm = layer_norm
        self.make_undirected = make_undirected

        self.input_proj = (
            nn.Identity()
            if input_dim == hidden_dim
            else nn.Linear(input_dim, hidden_dim)
        )
        self.input_dropout = nn.Dropout(input_dropout)
        self.layers = nn.ModuleList(
            AdvDIFFormerLayer(
                hidden_dim=hidden_dim,
                heads=heads,
                variant=self.variant,
                propagation_steps=propagation_steps,
                beta=beta,
                theta=theta,
                dropout=dropout,
                head_aggregation=head_aggregation,
                make_undirected=make_undirected,
            )
            for _ in range(num_layers)
        )
        self.norms = nn.ModuleList(
            nn.LayerNorm(hidden_dim) if layer_norm else nn.Identity()
            for _ in range(num_layers)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor | None = None,
        edge_weight: torch.Tensor | None = None,
        edge_attr: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Node feature matrix of shape ``[num_nodes, input_dim]``.
        edge_index : torch.Tensor
            Edge indices of shape ``[2, num_edges]``.
        batch : torch.Tensor, optional
            Batch assignment for each node.
        edge_weight : torch.Tensor, optional
            Optional scalar edge weights.
        edge_attr : torch.Tensor, optional
            Ignored unless it is one-dimensional and ``edge_weight`` is unset.
        **kwargs : dict
            Additional arguments ignored for wrapper compatibility.

        Returns
        -------
        torch.Tensor
            Node embeddings of shape ``[num_nodes, hidden_dim]``.
        """
        if edge_weight is None and edge_attr is not None and edge_attr.dim() == 1:
            edge_weight = edge_attr

        x = self.input_dropout(self.input_proj(x))
        prepared_batch = _prepare_batch(batch, x.shape[0], x.device)
        prepared_graph = _prepare_graph(
            edge_index,
            edge_weight,
            x.shape[0],
            x.dtype,
            x.device,
            self.make_undirected,
        )
        for layer, norm in zip(self.layers, self.norms, strict=False):
            propagated = layer(
                x,
                edge_index,
                batch=prepared_batch.batch,
                edge_weight=edge_weight,
                prepared_graph=prepared_graph,
                prepared_batch=prepared_batch,
            )
            if self.residual:
                x = x + self.dropout(propagated)
            else:
                x = self.dropout(propagated)
            x = norm(x)

        return x
