"""LapPE structural-coordinate ETNN backbone for combinatorial complexes.

This module implements a coordinate-enabled TopoBench adaptation of
E(n)-Equivariant Topological Neural Networks (ETNNs) from Battiloro et al.,
``E(n) Equivariant Topological Neural Networks``, arXiv:2405.15429, and the
official implementation at
``https://github.com/NSAPH-Projects/topological-equivariant-networks``.

The coordinate-free ETNN backbone in ``etnn.py`` implements the combinatorial
feature-update part of ETNN without geometric inputs. This module keeps that
feature-update structure but adds a conservative structural coordinate frame
for graph datasets, such as GraphUniverse, that do not provide physical
Euclidean coordinates.

The adaptation has three steps:

1. TopoBench computes normalized graph Laplacian eigenvectors as rank-0
   structural pseudo-coordinates with the existing ``LapPE`` transform.
2. Rank-0 coordinates are lifted to higher-rank cells by recursive incidence
   averaging:

       p_0(v) = LapPE(v)
       p_r(c) = mean_{d incident to c} p_{r-1}(d)

3. ETNN relation messages receive the sparse TopoBench neighborhood value and
   one E(n)-invariant structural-distance feature:

       z_{d,c,N} = concat(h_d, h_c, a_{d,c,N}, ||p_d - p_c||^2)
       m_{c,N}   = sum_{d in N(c)} psi_N(z_{d,c,N})
       h'_c      = h_c + beta_rank(c)(h_c, concat_N m_{c,N})

This follows the CCMPN/ETNN neighborhood aggregation and rank-wise feature
update while specializing the geometric input to distances in a graph-derived
structural embedding. The coordinate update from the full ETNN formulation is
not applied: coordinates are fixed auxiliary inputs, not learned dynamical
states. Consequently, the model is invariant to rigid transformations of the
chosen structural coordinate frame, but it should not be read as a physical
coordinate ETNN for datasets that do not contain physical Euclidean positions.
"""

from __future__ import annotations

from collections import defaultdict

import torch
from torch import nn

from topobench.data.utils import get_routes_from_neighborhoods
from topobench.nn.backbones.combinatorial.etnn import (
    _ETNNMessagePassing,
    _make_mlp,
    _neighborhood_to_edge_index,
)


class ETNNLapPE(nn.Module):
    """ETNN feature backbone with LapPE structural distance messages.

    The class mirrors the coordinate-free ``ETNN`` backbone but requires a
    rank-0 coordinate attribute, usually ``LapPE``, produced before
    graph-to-combinatorial lifting. For each relation edge from sender cell
    ``d`` to receiver cell ``c``, the relation-specific message function
    receives both the sparse TopoBench relation value ``a_{d,c,N}`` and the
    invariant structural-distance scalar ``||p_d - p_c||^2``.

    This variant keeps ETNN's topological feature update structure:

        m_{c,N} = sum_{d in N(c)} psi_N(h_d, h_c, a_{d,c,N}, ||p_d-p_c||^2)
        h'_c    = h_c + beta_rank(c)(h_c, concat_N m_{c,N})

    The LapPE term is used only as an invariant message feature. Coordinates
    are not updated, so this module should be understood as a structural
    pseudo-coordinate adaptation rather than a full coordinate-dynamical ETNN.

    The backbone expects the lifting/feature-encoding pipeline to provide
    feature tensors for every rank from 0 to ``max_rank``. Empty ranks should
    still be represented by zero-row tensors such as ``x_2.shape == [0, d]``;
    missing rank attributes are treated as malformed input.

    Parameters
    ----------
    in_channels : int
        Input feature dimension for every visible cell rank.
    hidden_channels : int
        Hidden dimension used by ETNN layers.
    out_channels : int
        Output feature dimension for every visible cell rank.
    neighborhoods : list[str]
        TopoBench neighborhood names, e.g. ``"up_adjacency-0"`` or
        ``"down_incidence-1"``.
    num_layers : int, optional
        Number of ETNN message-passing layers.
    dropout : float, optional
        Dropout probability used inside message and update blocks.
    activation : str, optional
        Activation function name.
    use_batch_norm : bool, optional
        Whether to use batch normalization inside MLP blocks.
    coordinate_attr : str, optional
        Batch attribute containing rank-0 structural coordinates.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        neighborhoods: list[str],
        num_layers: int = 2,
        dropout: float = 0.0,
        activation: str = "silu",
        use_batch_norm: bool = False,
        coordinate_attr: str = "LapPE",
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError(
                "ETNNLapPE requires at least one message-passing layer."
            )
        if len(neighborhoods) == 0:
            raise ValueError("ETNNLapPE requires at least one neighborhood.")

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.coordinate_attr = coordinate_attr

        # Keep the public neighborhood config identical to ETNN, then derive
        # source/destination ranks for relation-wise message passing.
        self.neighborhoods = list(neighborhoods)
        self.routes = get_routes_from_neighborhoods(self.neighborhoods)
        self.num_layers = num_layers
        self.max_rank = max(max(route) for route in self.routes)

        # AllCellFeatureEncoder projects every selected rank to the same hidden
        # size, so one projection shape works across ranks.
        self.input_projection = nn.ModuleDict(
            {
                str(rank): nn.Linear(in_channels, hidden_channels)
                for rank in range(self.max_rank + 1)
            }
        )

        # LapPE layers use the same topological relations as ETNN, with one
        # additional scalar structural-distance feature per relation edge.
        self.layers = nn.ModuleList(
            [
                _ETNNLapPELayer(
                    neighborhoods=self.neighborhoods,
                    routes=self.routes,
                    hidden_channels=hidden_channels,
                    dropout=dropout,
                    activation=activation,
                    use_batch_norm=use_batch_norm,
                )
                for _ in range(num_layers)
            ]
        )

        self.output_projection = nn.ModuleDict(
            {
                str(rank): nn.Linear(hidden_channels, out_channels)
                for rank in range(self.max_rank + 1)
            }
        )

    def forward(self, batch) -> dict[int, torch.Tensor]:
        """Run LapPE-distance ETNN message passing.

        Parameters
        ----------
        batch : torch_geometric.data.Data
            Lifted TopoBench batch containing rank-wise features ``x_i``,
            sparse neighborhood tensors, incidence matrices, and rank-0
            structural coordinates.

        Returns
        -------
        dict[int, torch.Tensor]
            Rank-indexed output embeddings compatible with ``TuneWrapper``.
        """
        x = {}
        for rank in range(self.max_rank + 1):
            key = f"x_{rank}"

            # The model config determines which ranks are visible through its
            # neighborhoods. Every visible rank must already have been encoded
            # by AllCellFeatureEncoder as ``x_0``, ``x_1``, ...
            if not hasattr(batch, key):
                raise AttributeError(
                    f"ETNNLapPE expected rank-{rank} features at `{key}`."
                )

            # Project every rank into the common hidden space used by ETNN
            # message and update functions.
            x[rank] = self.input_projection[str(rank)](getattr(batch, key))

        # Build structural coordinates once so every layer uses the same
        # coordinate frame and the same rank-wise cell summaries. This is done
        # inside the forward pass for the first variant because TopoBench
        # batches are assembled dynamically and moved across devices by
        # Lightning. If profiling shows this is a bottleneck, the same
        # coordinate lifting can be moved into a preprocessing transform.
        coordinates = _build_lappe_cell_coordinates(
            batch=batch,
            coordinate_attr=self.coordinate_attr,
            max_rank=self.max_rank,
            device=x[0].device,
            dtype=x[0].dtype,
        )

        # Apply relation-wise message passing. Coordinates are fixed auxiliary
        # features in this variant; only cell features are updated by layers.
        for layer in self.layers:
            x = layer(x, batch, coordinates=coordinates)

        # Return rank-indexed embeddings for the standard TopoBench wrapper.
        return {
            rank: self.output_projection[str(rank)](features)
            for rank, features in x.items()
        }


class _ETNNLapPELayer(nn.Module):
    """One ETNN layer with LapPE structural distance edge attributes.

    The layer has the same two aggregation levels as the coordinate-free ETNN:

    1. For each configured neighborhood relation ``N``, sender-cell messages
       are summed into receiver cells.
    2. For each destination rank, messages from all incoming relation types are
       concatenated and passed through a rank-specific update MLP.

    The only architectural difference is the relation edge attribute. Instead
    of using only the sparse neighborhood value ``a_{d,c,N}``, each message
    also sees ``||p_d - p_c||^2`` computed from LapPE-derived cell
    coordinates. This keeps the geometric signal invariant to translations,
    rotations, and reflections of the structural coordinate frame.

    Relation-message order is part of the model definition: messages are
    concatenated in the order given by ``self.neighborhoods``. The Hydra config
    therefore fixes both which relations are used and the channel order seen by
    the rank-wise update MLPs.

    Parameters
    ----------
    neighborhoods : list[str]
        TopoBench sparse neighborhood names used as ETNN relation types.
    routes : list[list[int]]
        Source and destination rank pairs inferred from ``neighborhoods``.
    hidden_channels : int
        Hidden feature dimension for every rank.
    dropout : float
        Dropout probability used in message and update MLPs.
    activation : str
        Activation name used in message and update MLPs.
    use_batch_norm : bool
        Whether to insert batch normalization in MLP blocks.
    """

    def __init__(
        self,
        neighborhoods: list[str],
        routes: list[list[int]],
        hidden_channels: int,
        dropout: float,
        activation: str,
        use_batch_norm: bool,
    ) -> None:
        super().__init__()
        self.neighborhoods = list(neighborhoods)
        self.routes = [tuple(route) for route in routes]
        if len(self.neighborhoods) != len(self.routes):
            raise ValueError(
                "ETNNLapPE expected one route per neighborhood, but found "
                f"{len(self.neighborhoods)} neighborhoods and "
                f"{len(self.routes)} routes."
            )

        # Two scalar edge attributes are supplied to each relation message:
        # the sparse-neighborhood value and the squared structural distance.
        self.message_passing = nn.ModuleList(
            [
                _ETNNMessagePassing(
                    hidden_channels=hidden_channels,
                    edge_channels=2,
                    dropout=dropout,
                    activation=activation,
                    use_batch_norm=use_batch_norm,
                )
                for _ in self.neighborhoods
            ]
        )

        # Count how many relation types send messages into each rank. The
        # update MLP input is the current state plus one aggregated message per
        # incoming relation.
        incoming_counts = defaultdict(int)
        for _, dst_rank in self.routes:
            incoming_counts[dst_rank] += 1

        # ETNN uses rank-wise update functions. A node, edge, and face can
        # receive different numbers and types of relation messages.
        ranks = sorted({rank for route in self.routes for rank in route})
        self.update = nn.ModuleDict(
            {
                str(rank): _make_mlp(
                    in_channels=(1 + incoming_counts[rank]) * hidden_channels,
                    hidden_channels=hidden_channels,
                    out_channels=hidden_channels,
                    dropout=dropout,
                    activation=activation,
                    use_batch_norm=use_batch_norm,
                )
                for rank in ranks
            }
        )

    def forward(
        self,
        x: dict[int, torch.Tensor],
        batch,
        coordinates: dict[int, torch.Tensor],
    ) -> dict[int, torch.Tensor]:
        """Apply one LapPE-distance ETNN layer.

        Parameters
        ----------
        x : dict[int, torch.Tensor]
            Rank-indexed hidden cell features.
        batch : torch_geometric.data.Data
            Lifted TopoBench batch containing sparse neighborhoods.
        coordinates : dict[int, torch.Tensor]
            Rank-indexed structural cell coordinates. The keys must match the
            ranks in ``x``.

        Returns
        -------
        dict[int, torch.Tensor]
            Updated rank-indexed hidden cell features.
        """
        # Accumulate one aggregated message tensor per destination rank and per
        # incoming relation type. The list order follows ``self.neighborhoods``
        # and is therefore deterministic for a fixed config.
        messages_by_rank = defaultdict(list)

        for route_idx, (neighborhood, route) in enumerate(
            zip(self.neighborhoods, self.routes, strict=True)
        ):
            src_rank, dst_rank = route

            # Convert the sparse TopoBench relation into explicit sender and
            # receiver indices. The returned edge attribute is the scalar sparse
            # neighborhood value ``a_{d,c,N}``.
            edge_index, edge_attr = _neighborhood_to_edge_index(
                batch=batch,
                neighborhood=neighborhood,
                src_rank=src_rank,
                dst_rank=dst_rank,
                device=x[src_rank].device,
                dtype=x[src_rank].dtype,
                num_src_cells=x[src_rank].shape[0],
                num_dst_cells=x[dst_rank].shape[0],
            )
            if edge_attr.ndim != 2 or edge_attr.shape[1] != 1:
                raise ValueError(
                    "ETNNLapPE expected `_neighborhood_to_edge_index` to "
                    "return scalar edge attributes with shape [num_edges, 1], "
                    f"but found shape {tuple(edge_attr.shape)}."
                )

            # Add the invariant structural-coordinate distance. This is the
            # only place where LapPE coordinates enter the message update. The
            # message MLP can learn how strongly to weight this scalar, so this
            # first variant does not introduce a separate distance scale.
            distance_attr = _squared_coordinate_distances(
                src_coordinates=coordinates[src_rank],
                dst_coordinates=coordinates[dst_rank],
                edge_index=edge_index,
                dtype=edge_attr.dtype,
            )
            edge_attr = torch.cat([edge_attr, distance_attr], dim=-1)

            # Apply the relation-specific ETNN message function and store the
            # aggregated receiver messages under the destination rank.
            message = self.message_passing[route_idx](
                x_src=x[src_rank],
                x_dst=x[dst_rank],
                edge_index=edge_index,
                edge_attr=edge_attr,
            )
            messages_by_rank[dst_rank].append(message)

        out = {}
        for rank, features in x.items():
            # Concatenate the current rank state with all messages arriving at
            # that rank, then apply the rank-specific residual update.
            update_input = torch.cat(
                [features, *messages_by_rank.get(rank, [])], dim=-1
            )
            out[rank] = features + self.update[str(rank)](update_input)
        return out


def _build_lappe_cell_coordinates(
    batch,
    coordinate_attr: str,
    max_rank: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[int, torch.Tensor]:
    """Construct rank-wise structural cell coordinates from rank-0 LapPE.

    Rank 0 uses the coordinate matrix stored in ``coordinate_attr``. Each
    higher rank is obtained by averaging the coordinates of incident
    lower-rank cells through the corresponding absolute incidence matrix.

    The expected incidence convention is the TopoBench convention used by
    combinatorial complexes: ``incidence_r`` has rows for rank ``r-1`` cells
    and columns for rank ``r`` cells. Absolute values are used because
    orientation signs are meaningful for boundary operators but not for
    barycentric coordinate averaging.

    Parameters
    ----------
    batch : torch_geometric.data.Data
        Lifted TopoBench batch containing rank-wise features, incidence
        matrices, and the rank-0 coordinate attribute.
    coordinate_attr : str
        Attribute containing rank-0 structural coordinates, usually ``LapPE``.
    max_rank : int
        Maximum cell rank that needs coordinates.
    device : torch.device
        Device for returned coordinate tensors.
    dtype : torch.dtype
        Floating dtype for returned coordinate tensors.

    Returns
    -------
    dict[int, torch.Tensor]
        Rank-indexed structural coordinate tensors.
    """
    if not hasattr(batch, coordinate_attr):
        raise AttributeError(
            f"ETNNLapPE expected rank-0 coordinates at `{coordinate_attr}`."
        )

    # Rank-0 coordinates come directly from the preprocessing transform. We
    # move them to the active model device because Lightning may transfer the
    # batch before the backbone sees it.
    rank_0_coordinates = getattr(batch, coordinate_attr).to(
        device=device, dtype=dtype
    )

    # Fail loudly if preprocessing produced a malformed coordinate tensor. A
    # silent row mismatch would attach coordinates to the wrong cells.
    if rank_0_coordinates.ndim != 2:
        raise ValueError(
            "ETNNLapPE expected a rank-0 coordinate matrix at "
            f"`{coordinate_attr}`, but found shape "
            f"{tuple(rank_0_coordinates.shape)}."
        )
    if rank_0_coordinates.shape[0] != batch.x_0.shape[0]:
        raise ValueError(
            "ETNNLapPE expected one rank-0 coordinate row per rank-0 cell, "
            f"but found {rank_0_coordinates.shape[0]} coordinates for "
            f"{batch.x_0.shape[0]} cells."
        )

    coordinates = {0: rank_0_coordinates}
    for rank in range(1, max_rank + 1):
        feature_key = f"x_{rank}"
        if not hasattr(batch, feature_key):
            raise AttributeError(
                f"ETNNLapPE expected rank-{rank} features at `{feature_key}`."
            )

        incidence_key = f"incidence_{rank}"
        if not hasattr(batch, incidence_key):
            raise AttributeError(
                "ETNNLapPE needs incidence matrices to lift coordinates, but "
                f"`{incidence_key}` is missing."
            )

        # Lift coordinates one rank at a time: vertices -> edges -> faces.
        # This avoids adding a new preprocessing contract for direct vertex-cell
        # incidence while still producing a coordinate for every visible cell.
        incidence = getattr(batch, incidence_key).coalesce().to(device)
        coordinates[rank] = _average_coordinates_through_incidence(
            lower_coordinates=coordinates[rank - 1],
            incidence=incidence,
            num_cells=getattr(batch, feature_key).shape[0],
        )
    return coordinates


def _average_coordinates_through_incidence(
    lower_coordinates: torch.Tensor,
    incidence: torch.Tensor,
    num_cells: int,
) -> torch.Tensor:
    """Average lower-rank coordinates over each incident higher-rank cell.

    Parameters
    ----------
    lower_coordinates : torch.Tensor
        Coordinates for rank ``r-1`` cells with shape
        ``[num_lower_cells, coordinate_dim]``.
    incidence : torch.Tensor
        Sparse incidence matrix with rows as rank ``r-1`` cells and columns as
        rank ``r`` cells.
    num_cells : int
        Number of real rank ``r`` cells in the current batch.

    Returns
    -------
    torch.Tensor
        Coordinate matrix for rank ``r`` cells with shape
        ``[num_cells, coordinate_dim]``.
    """
    if num_cells == 0:
        return lower_coordinates.new_empty((0, lower_coordinates.shape[1]))

    incidence = incidence.coalesce()

    # Check both sparse axes before indexing. These errors usually indicate an
    # incompatible lifting or a malformed batch, so failing here is better than
    # silently producing incorrect distances.
    if incidence.shape[0] != lower_coordinates.shape[0]:
        raise ValueError(
            "Cannot lift ETNNLapPE structural coordinates: incidence has "
            f"{incidence.shape[0]} source rows, but the lower rank has "
            f"{lower_coordinates.shape[0]} coordinate rows."
        )
    if incidence.shape[1] != num_cells:
        raise ValueError(
            "Cannot lift ETNNLapPE structural coordinates: incidence has "
            f"{incidence.shape[1]} columns, but the target rank has "
            f"{num_cells} cells."
        )

    # Drop explicit zero entries. They may appear as empty-rank placeholders and
    # should not contribute to coordinate averages.
    indices = incidence.indices()
    values = incidence.values().abs().to(lower_coordinates.dtype)
    nonzero_mask = values != 0
    indices = indices[:, nonzero_mask]
    values = values[nonzero_mask]

    # Accumulate weighted coordinate sums and total incidence weights for each
    # target cell. Absolute incidence values ignore orientation signs.
    coordinates = lower_coordinates.new_zeros(
        (num_cells, lower_coordinates.shape[1])
    )
    weights = lower_coordinates.new_zeros((num_cells, 1))
    if values.numel() == 0:
        return coordinates

    lower_idx = indices[0]
    cell_idx = indices[1]
    coordinates.index_add_(
        0, cell_idx, lower_coordinates[lower_idx] * values.unsqueeze(-1)
    )
    weights.index_add_(0, cell_idx, values.unsqueeze(-1))

    # Isolated or degenerate target cells keep zero coordinates. The clamp
    # avoids division by zero without changing nonzero averages.
    weights = weights.clamp_min(torch.finfo(weights.dtype).eps)
    return coordinates / weights


def _squared_coordinate_distances(
    src_coordinates: torch.Tensor,
    dst_coordinates: torch.Tensor,
    edge_index: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Compute invariant squared distances for relation edges.

    Parameters
    ----------
    src_coordinates : torch.Tensor
        Coordinates for sender-rank cells.
    dst_coordinates : torch.Tensor
        Coordinates for receiver-rank cells.
    edge_index : torch.Tensor
        Relation edges in ``[sender, receiver]`` format.
    dtype : torch.dtype
        Floating dtype for the returned distance feature.

    Returns
    -------
    torch.Tensor
        Squared distance feature with shape ``[num_edges, 1]``.
    """
    if edge_index.numel() == 0:
        return src_coordinates.new_empty((0, 1), dtype=dtype)

    sender, receiver = edge_index

    # Squared Euclidean distance is invariant to translations, rotations, and
    # reflections of the structural coordinate frame.
    delta = src_coordinates[sender] - dst_coordinates[receiver]
    return delta.pow(2).sum(dim=-1, keepdim=True).to(dtype)
