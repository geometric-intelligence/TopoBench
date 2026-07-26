"""Multi-Track Graph Convolutional Network backbone.

This module implements a TopoBench-compatible adaptation of Pei et al.,
"Multi-Track Message Passing: Tackling Oversmoothing and Oversquashing in
Graph Learning via Preventing Heterophily Mixing" (ICML 2024), roughly following the
official https://github.com/XJTU-Graph-Intelligence-Lab/Multi-Track-Message-Passing. 
Unlike their full pipeline, this only implements the multi track encoder and is not specific to any application.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.utils import add_remaining_self_loops, coalesce


class _PreparedGraph:
    """Sparse normalized graph, this is reused by all layers of the encoder."""

    __slots__ = ("dst", "norm_weight", "src")

    def __init__(self, src, dst, norm_weight) -> None:
        self.src = src
        self.dst = dst
        self.norm_weight = norm_weight

    def matmul(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the normalized adjacency to arbitrary dimensions."""
        if self.src.numel() == 0:
            return torch.zeros_like(x)
        weight_shape = (-1,) + (1,) * (x.dim() - 1)
        messages = x.index_select(0, self.src) * self.norm_weight.view(
            weight_shape
        )
        return torch.zeros_like(x).index_add(0, self.dst, messages)


def _as_undirected(
    edge_index: torch.Tensor, edge_weight: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetrize an edge list while preserving scalar edge weights."""
    reverse = edge_index.flip(0)
    edge_index = torch.cat((edge_index, reverse), dim=1)
    edge_weight = torch.cat((edge_weight, edge_weight), dim=0)
    return coalesce(edge_index, edge_weight, reduce="mean")


def _prepare_graph(
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor | None,
    num_nodes: int,
    dtype: torch.dtype,
    device: torch.device,
    normalization: str,
    add_self_loops: bool,
    make_undirected: bool,
) -> _PreparedGraph:
    """Build the sparse normalized adjacency used by all propagation layers."""
    edge_index = edge_index.to(device=device)
    if edge_weight is None:
        edge_weight = torch.ones(
            edge_index.shape[1], dtype=dtype, device=device
        )
    else:
        edge_weight = edge_weight.to(dtype=dtype, device=device)

    if make_undirected and edge_index.numel() > 0:
        edge_index, edge_weight = _as_undirected(edge_index, edge_weight)
    if add_self_loops:
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value=1.0, num_nodes=num_nodes
        )
    if edge_index.numel() == 0:
        empty = edge_index.new_empty(0)
        return _PreparedGraph(empty, empty, edge_weight)

    src, dst = edge_index
    degree = torch.zeros(num_nodes, dtype=dtype, device=device)
    degree.index_add_(0, dst, edge_weight)
    inverse_degree = torch.zeros_like(degree)
    positive = degree > 0
    if normalization == "symmetric":
        inverse_degree[positive] = degree[positive].pow(-0.5)
        norm_weight = inverse_degree[src] * edge_weight * inverse_degree[dst]
    else:
        inverse_degree[positive] = degree[positive].reciprocal()
        norm_weight = inverse_degree[dst] * edge_weight
    return _PreparedGraph(src, dst, norm_weight)


class MTGCNEncoder(nn.Module):
    """Multi-Track Message Passing graph encoder.

    The encoder implements MTGCN's loading, independent multi-track message
    passing, and acquiring stages. Track prototypes are learnable because a
    reusable TopoBench backbone cannot access labels or pseudo-label stages.

    Parameters
    ----------
    in_channels : int
        Number of input node features.
    hidden_channels : int
        Number of output features per node.
    num_layers : int, optional
        Number of multi-track propagation layers.
    num_tracks : int, optional
        Number of semantic propagation tracks.
    num_heads : int, optional
        Number of track-assignment heads.
    dropout : float, optional
        Dropout probability used by the reference architecture.
    propagation_weight : float, optional
        Weight of propagated messages in each initial-residual update.
    use_residual : bool, optional
        Whether to retain the initial-message residual.
    normalization : str, optional
        Sparse adjacency normalization, ``"symmetric"`` or ``"row"``.
    add_self_loops : bool, optional
        Whether to add the self-loops used by the reference model.
    make_undirected : bool, optional
        Whether to symmetrize the observed graph before propagation.
    temperature : float, optional
        Soft track-assignment temperature.
    affiliation_source : str, optional
        Source for track affiliations: ``"features"``, ``"auxiliary"``, or
        ``"hybrid"``.
    use_entropy_sharpening : bool, optional
        Whether to sharpen affiliations before loading messages onto tracks.
    sharpening_power : float, optional
        Power used by normalized affiliation sharpening.
    use_output_residual : bool, optional
        Whether to add a projected input residual after acquiring messages.
    output_norm : str, optional
        Output normalization, either ``"none"`` or ``"layer"``.
    **kwargs : dict
        Additional wrapper arguments, ignored.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int = 4,
        num_tracks: int = 2,
        num_heads: int = 1,
        dropout: float = 0.1,
        propagation_weight: float = 0.8,
        use_residual: bool = True,
        normalization: str = "symmetric",
        add_self_loops: bool = True,
        make_undirected: bool = False,
        temperature: float = 1.0,
        affiliation_source: str = "hybrid",
        use_entropy_sharpening: bool = False,
        sharpening_power: float = 2.0,
        use_output_residual: bool = True,
        output_norm: str = "layer",
        **kwargs,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1.")
        if num_tracks < 2:
            raise ValueError("num_tracks must be at least 2.")
        if num_heads < 1:
            raise ValueError("num_heads must be at least 1.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        if not 0.0 <= propagation_weight <= 1.0:
            raise ValueError("propagation_weight must be in [0, 1].")
        if normalization not in {"symmetric", "row"}:
            raise ValueError("normalization must be 'symmetric' or 'row'.")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive.")
        if affiliation_source not in {"features", "auxiliary", "hybrid"}:
            raise ValueError(
                "affiliation_source must be 'features', 'auxiliary', or 'hybrid'."
            )
        if sharpening_power <= 0.0:
            raise ValueError("sharpening_power must be positive.")
        if output_norm not in {"none", "layer"}:
            raise ValueError("output_norm must be 'none' or 'layer'.")

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = hidden_channels
        self.num_layers = num_layers
        self.num_tracks = num_tracks
        self.num_heads = num_heads
        self.dropout = dropout
        self.propagation_weight = propagation_weight
        self.use_residual = use_residual
        self.normalization = normalization
        self.add_self_loops = add_self_loops
        self.make_undirected = make_undirected
        self.temperature = temperature
        self.affiliation_source = affiliation_source
        self.use_entropy_sharpening = use_entropy_sharpening
        self.sharpening_power = sharpening_power
        self.use_output_residual = use_output_residual
        self.output_norm = output_norm

        projected_channels = num_heads * hidden_channels
        self.value_projection = nn.Linear(in_channels, projected_channels)
        self.affiliation_input = nn.Linear(in_channels, hidden_channels)
        self.output_residual = (
            nn.Identity()
            if in_channels == hidden_channels
            else nn.Linear(in_channels, hidden_channels)
        )
        self.auxiliary_input = nn.Linear(in_channels, hidden_channels)
        self.auxiliary_output = nn.Linear(hidden_channels, hidden_channels)
        self.query_projection = nn.Linear(hidden_channels, projected_channels)
        self.key_projection = nn.Linear(hidden_channels, projected_channels)
        self.track_prototypes = nn.Parameter(
            torch.empty(num_tracks, hidden_channels)
        )
        self.head_fusion = nn.Linear(num_heads, 1)
        self.norm = (
            nn.LayerNorm(hidden_channels)
            if output_norm == "layer"
            else nn.Identity()
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset learnable parameters."""
        for module in (
            self.value_projection,
            self.affiliation_input,
            self.auxiliary_input,
            self.auxiliary_output,
            self.query_projection,
            self.key_projection,
            self.head_fusion,
        ):
            module.reset_parameters()
        if isinstance(self.output_residual, nn.Linear):
            self.output_residual.reset_parameters()
        if isinstance(self.norm, nn.LayerNorm):
            self.norm.reset_parameters()
        nn.init.xavier_uniform_(self.track_prototypes)

    def _auxiliary_embeddings(
        self, x: torch.Tensor, graph: _PreparedGraph
    ) -> torch.Tensor:
        """Compute the two-layer GCN representation used for affiliation."""
        hidden = graph.matmul(self.auxiliary_input(x))
        hidden = F.relu(hidden)
        hidden = F.dropout(hidden, self.dropout, training=self.training)
        return graph.matmul(self.auxiliary_output(hidden))

    def _affiliation_embeddings(
        self, x: torch.Tensor, graph: _PreparedGraph
    ) -> torch.Tensor:
        """Select the representation used for semantic track assignment."""
        feature_affiliation = self.affiliation_input(x)
        if self.affiliation_source == "features":
            return feature_affiliation

        auxiliary = self._auxiliary_embeddings(x, graph)
        if self.affiliation_source == "auxiliary":
            return auxiliary
        return feature_affiliation + auxiliary

    def _compute_affiliations(
        self, affiliation_input: torch.Tensor
    ) -> torch.Tensor:
        """Compute soft node-to-track affiliations for every head."""
        query = self.query_projection(affiliation_input).view(
            -1, self.num_heads, self.hidden_channels
        )
        keys = self.key_projection(self.track_prototypes).view(
            self.num_tracks, self.num_heads, self.hidden_channels
        )
        scores = torch.einsum("nhd,thd->nht", query, keys)
        return F.softmax(scores / self.temperature, dim=-1)

    def _track_weights(self, affiliations: torch.Tensor) -> torch.Tensor:
        """Return normalized node-to-track loading weights."""
        if not self.use_entropy_sharpening:
            return affiliations
        sharpened = affiliations.pow(self.sharpening_power)
        return sharpened / sharpened.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    def _load_tracks(
        self, x: torch.Tensor, affiliations: torch.Tensor
    ) -> torch.Tensor:
        """Load node values onto semantic tracks."""
        values = self.value_projection(x).view(
            -1, self.num_heads, self.hidden_channels
        )
        values = F.dropout(values, self.dropout, training=self.training)
        track_weights = self._track_weights(affiliations).permute(0, 2, 1)
        return track_weights.unsqueeze(-1) * values.unsqueeze(1)

    def _propagate_tracks(
        self, initial_tracks: torch.Tensor, graph: _PreparedGraph
    ) -> torch.Tensor:
        """Run the released model's sparse two-hop residual track layers."""
        tracks = initial_tracks
        for _ in range(self.num_layers):
            tracks = F.dropout(tracks, self.dropout, training=self.training)
            tracks = graph.matmul(tracks)
            tracks = self.propagation_weight * graph.matmul(tracks)
            if self.use_residual:
                tracks = (
                    tracks + (1.0 - self.propagation_weight) * initial_tracks
                )
        return tracks

    def _acquire_messages(
        self, tracks: torch.Tensor, affiliations: torch.Tensor
    ) -> torch.Tensor:
        """Acquire head-specific track messages and fuse heads."""
        # tracks: [N, T, H, C], affiliations: [N, H, T], per_head: [N, H, C].
        track_weights = self._track_weights(affiliations).permute(0, 2, 1)
        per_head = (track_weights.unsqueeze(-1) * tracks).sum(dim=1)
        per_head = F.dropout(per_head, self.dropout, training=self.training)
        return self.head_fusion(per_head.transpose(1, 2)).squeeze(-1)

    def affiliation_diagnostics(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
        edge_attr: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return affiliation entropy and track-usage diagnostics."""
        if (
            edge_weight is None
            and edge_attr is not None
            and edge_attr.dim() == 1
        ):
            edge_weight = edge_attr
        graph = _prepare_graph(
            edge_index,
            edge_weight,
            x.shape[0],
            x.dtype,
            x.device,
            self.normalization,
            self.add_self_loops,
            self.make_undirected,
        )
        affiliations = self._compute_affiliations(
            self._affiliation_embeddings(x, graph)
        )
        usage = affiliations.mean(dim=(0, 1))
        entropy = -(affiliations * affiliations.clamp_min(1e-12).log()).sum(
            dim=-1
        )
        return {
            "mean_entropy": entropy.mean(),
            "mean_track_usage": usage.mean(),
            "max_track_usage": usage.max(),
            "min_track_usage": usage.min(),
        }

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
        edge_attr: torch.Tensor | None = None,
        batch: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Encode node features with graph-local multi-track propagation.

        Parameters
        ----------
        x : torch.Tensor
            Node features of shape ``[num_nodes, in_channels]``.
        edge_index : torch.Tensor
            Sparse edge indices of shape ``[2, num_edges]``.
        edge_weight : torch.Tensor, optional
            Scalar edge weights.
        edge_attr : torch.Tensor, optional
            Used as edge weights when one-dimensional and ``edge_weight`` is
            not supplied.
        batch : torch.Tensor, optional
            Accepted for wrapper compatibility. Disjoint ``edge_index`` keeps
            propagation graph-local.
        **kwargs : dict
            Additional wrapper arguments, ignored.

        Returns
        -------
        torch.Tensor
            Node embeddings of shape ``[num_nodes, hidden_channels]``.
        """
        if (
            edge_weight is None
            and edge_attr is not None
            and edge_attr.dim() == 1
        ):
            edge_weight = edge_attr
        graph = _prepare_graph(
            edge_index,
            edge_weight,
            x.shape[0],
            x.dtype,
            x.device,
            self.normalization,
            self.add_self_loops,
            self.make_undirected,
        )
        affiliations = self._compute_affiliations(
            self._affiliation_embeddings(x, graph)
        )
        initial_tracks = self._load_tracks(x, affiliations)
        tracks = self._propagate_tracks(initial_tracks, graph)
        output = self._acquire_messages(tracks, affiliations)
        if self.use_output_residual:
            output = output + self.output_residual(x)
        return self.norm(output)
