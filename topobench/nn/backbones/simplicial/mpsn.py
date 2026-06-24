"""Message Passing Simplicial Network (MPSN) backbone.

Native implementation of the boundary + upper-adjacency, GIN-style MPSN variant.

Plain message-passing GNNs are bounded by the 1-Weisfeiler-Lehman (1-WL) test and
cannot count triangles. MPSN lifts each triangle to a 2-simplex (clique lifting)
and runs message passing over the resulting simplicial complex; its Simplicial-WL
colour refinement is strictly more powerful than 1-WL and no less powerful than
3-WL, so triangles become first-class cells the network can read and count [1].

We implement the canonical efficient variant (matching CIN): for a simplex of rank
r we use boundary and upper-adjacency messages only (co-boundary and lower-adjacency
are dropped, which the paper proves retains full SWL power). With an injective,
GIN-style update (Xu et al., 2019) and a learnable per-rank epsilon, one layer is

    h_r^{t+1} = MLP_r( (1 + eps_r) h_r + m_boundary + m_upper ),

where the boundary message sums the features of a simplex's (r-1)-faces and the
upper message sums, over rank-r neighbours sharing a common (r+1)-coface, both the
neighbour feature and the shared-coface feature.

Routing uses only the unsigned clique-lifted incidence matrices B1 (node-to-edge)
and B2 (edge-to-triangle); the boundary, upper-adjacency and shared-coface terms
are all derived from these (no Hodge Laplacians). The lifting is unsigned, so no
orientation/sign handling is needed and the MLP activation is a plain ReLU.

References
----------
.. [1] Bodnar, Frasca, Wang, Otter, Montúfar, Liò, Bronstein.
   Weisfeiler and Lehman Go Topological: Message Passing Simplicial Networks.
   ICML 2021 (Spotlight). https://arxiv.org/abs/2103.03212
"""

import torch


def _to_dense(matrix: torch.Tensor) -> torch.Tensor:
    """Return a dense, unsigned (0/1) version of an incidence matrix.

    The clique lifting emits sparse COO incidence matrices with ``signed=False``,
    i.e. already 0/1. We densify (toy/complex-sized routing operators) and take
    the absolute value so the support is used regardless of any sign convention.

    Parameters
    ----------
    matrix : torch.Tensor
        Sparse or dense incidence matrix.

    Returns
    -------
    torch.Tensor
        Dense unsigned incidence matrix.
    """
    if matrix.is_sparse:
        matrix = matrix.to_dense()
    return matrix.abs()


def _upper_adjacency(boundary_next: torch.Tensor) -> torch.Tensor:
    """Build the rank-r upper-adjacency neighbour operator.

    Two rank-r simplices are upper-adjacent when they share a common (r+1)-coface.
    With ``boundary_next`` the unsigned rank-r to rank-(r+1) incidence B (shape
    ``[N_r, N_{r+1}]``), the co-adjacency counts are the off-diagonal of
    ``boundary_next @ boundary_next.T``; the diagonal (a simplex's own coface
    count) is removed so a simplex is not its own neighbour.

    Parameters
    ----------
    boundary_next : torch.Tensor
        Dense unsigned incidence of shape ``[N_r, N_{r+1}]``.

    Returns
    -------
    torch.Tensor
        Dense ``[N_r, N_r]`` upper-adjacency operator (zero diagonal). Applying it
        to rank-r features sums, for each simplex, its upper-adjacent neighbours.
    """
    adjacency = boundary_next @ boundary_next.transpose(0, 1)
    adjacency = adjacency - torch.diag(torch.diagonal(adjacency))
    return adjacency


class MPSNLayer(torch.nn.Module):
    """One MPSN layer: boundary + upper-adjacency messages, GIN-style update.

    Applies, for every rank r in {0, 1, 2} simultaneously, the boundary + upper
    update of Bodnar et al. (2021),

        h_r^{t+1} = MLP_r( (1 + eps_r) h_r + m_boundary + m_upper ).

    Each rank owns a distinct MLP and a learnable scalar epsilon (initialised to
    0.0), matching the injective GIN-style aggregator of Xu et al. (2019). Ranks
    whose boundary or upper neighbourhood is empty simply omit that term (rank 0
    has no boundary; rank 2 has no upper-adjacency).

    Parameters
    ----------
    hidden_dim : int
        Feature width shared by all ranks.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

        # One distinct MLP per rank (0, 1, 2): Linear -> ReLU -> Linear, no norm.
        for r in range(3):
            mlp = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, hidden_dim),
            )
            setattr(self, f"mlp_{r}", mlp)
            # Learnable per-rank epsilon, init 0.0 (GIN-style injective update).
            setattr(
                self,
                f"eps_{r}",
                torch.nn.Parameter(torch.zeros(())),
            )

    def _boundary_message(
        self, boundary: torch.Tensor, x_face: torch.Tensor
    ) -> torch.Tensor:
        """Boundary message: sum of the features of a simplex's (r-1)-faces.

        With ``boundary`` the rank-(r-1) to rank-r incidence B (shape
        ``[N_{r-1}, N_r]``), ``boundary.T @ x_face`` gathers, for each rank-r
        simplex, the sum of its faces' features.

        Parameters
        ----------
        boundary : torch.Tensor
            Dense unsigned incidence of shape ``[N_{r-1}, N_r]``.
        x_face : torch.Tensor
            Features of the (r-1)-faces, shape ``[N_{r-1}, hidden]``.

        Returns
        -------
        torch.Tensor
            Boundary message per rank-r simplex, shape ``[N_r, hidden]``.
        """
        return boundary.transpose(0, 1) @ x_face

    def _upper_message(
        self,
        upper_adjacency: torch.Tensor,
        boundary_next: torch.Tensor,
        x_self: torch.Tensor,
        x_coface: torch.Tensor,
        coface_multiplicity: float,
    ) -> torch.Tensor:
        """Upper-adjacency message for one rank.

        Sums, over each rank-r simplex's upper-adjacent neighbours, both the
        neighbour feature and the shared-coface feature. The neighbour term is the
        upper-adjacency operator applied to the rank's own features. The
        shared-coface term is computed exactly from the coface multiplicity: in a
        simple-graph clique complex with ``complex_dim = 2`` two upper-adjacent
        simplices share exactly one coface, so summing cofaces (via
        ``boundary_next @ x_coface``) weighted by ``|faces_r(coface)| - 1`` gives
        the shared-coface contribution -- multiplicity 1 for rank 0 (each edge has
        2 nodes) and 2 for rank 1 (each triangle has 3 edges).

        Parameters
        ----------
        upper_adjacency : torch.Tensor
            Dense ``[N_r, N_r]`` upper-adjacency operator (zero diagonal).
        boundary_next : torch.Tensor
            Dense unsigned incidence of shape ``[N_r, N_{r+1}]``.
        x_self : torch.Tensor
            Features of the rank-r simplices, shape ``[N_r, hidden]``.
        x_coface : torch.Tensor
            Features of the rank-(r+1) cofaces, shape ``[N_{r+1}, hidden]``.
        coface_multiplicity : float
            ``|faces_r(coface)| - 1``: 1 for rank 0, 2 for rank 1.

        Returns
        -------
        torch.Tensor
            Upper message per rank-r simplex, shape ``[N_r, hidden]``.
        """
        neighbour_term = upper_adjacency @ x_self
        coface_term = coface_multiplicity * (boundary_next @ x_coface)
        return neighbour_term + coface_term

    def _update(
        self, rank: int, x_self: torch.Tensor, message: torch.Tensor
    ) -> torch.Tensor:
        """GIN-style injective update for one rank.

        Returns ``MLP_r( (1 + eps_r) * x_self + message )``, where ``message`` is
        the sum of the available boundary and upper messages for the rank.

        Parameters
        ----------
        rank : int
            Rank r in {0, 1, 2}.
        x_self : torch.Tensor
            Current features of the rank-r simplices.
        message : torch.Tensor
            Aggregated boundary + upper message for this rank.

        Returns
        -------
        torch.Tensor
            Updated rank-r features.
        """
        eps = getattr(self, f"eps_{rank}")
        mlp = getattr(self, f"mlp_{rank}")
        return mlp((1.0 + eps) * x_self + message)

    def forward(self, x_all, incidence_all):
        """Apply one MPSN layer to all three ranks.

        Parameters
        ----------
        x_all : tuple of torch.Tensor
            Current features ``(x_0, x_1, x_2)``.
        incidence_all : tuple of torch.Tensor
            Dense unsigned operators ``(B1, B2, upper_adj_0, upper_adj_1)`` where
            ``B1`` is the node-to-edge incidence, ``B2`` the edge-to-triangle
            incidence, and ``upper_adj_0`` / ``upper_adj_1`` the precomputed rank-0
            / rank-1 upper-adjacency operators.

        Returns
        -------
        tuple of torch.Tensor
            Updated features ``(x_0, x_1, x_2)``.
        """
        x_0, x_1, x_2 = x_all
        b1, b2, upper_adj_0, upper_adj_1 = incidence_all

        # ---- rank 0 (nodes): no boundary; upper-adjacency via shared edge ----
        msg_0 = self._upper_message(
            upper_adjacency=upper_adj_0,
            boundary_next=b1,
            x_self=x_0,
            x_coface=x_1,
            coface_multiplicity=1.0,  # each edge-coface has 2 nodes -> 2-1 = 1
        )
        new_x_0 = self._update(0, x_0, msg_0)

        # ---- rank 1 (edges): boundary = 2 nodes; upper via shared triangle ----
        msg_1 = self._boundary_message(b1, x_0)
        if x_2.shape[0] > 0:
            msg_1 = (
                msg_1
                + self._upper_message(
                    upper_adjacency=upper_adj_1,
                    boundary_next=b2,
                    x_self=x_1,
                    x_coface=x_2,
                    coface_multiplicity=2.0,  # each tri-coface has 3 edges -> 3-1 = 2
                )
            )
        new_x_1 = self._update(1, x_1, msg_1)

        # ---- rank 2 (triangles): boundary = 3 edges; upper is empty ----
        if x_2.shape[0] > 0:
            msg_2 = self._boundary_message(b2, x_1)
            new_x_2 = self._update(2, x_2, msg_2)
        else:
            new_x_2 = x_2

        return new_x_0, new_x_1, new_x_2


class MPSN(torch.nn.Module):
    """Message Passing Simplicial Network (boundary + upper-adjacency variant).

    Lifts node features to higher-rank cells with per-rank input projections, then
    stacks ``n_layers`` :class:`MPSNLayer` blocks. Each layer performs the
    injective, GIN-style simplicial update of Bodnar et al. (2021) (boundary +
    upper-adjacency messages), whose SWL colour refinement is strictly more
    powerful than 1-WL and no weaker than 3-WL, so the 2-cells (triangles) become
    countable, first-class objects.

    Parameters
    ----------
    in_channels_all : tuple of int
        Input feature dimensions on ``(nodes, edges, triangles)``.
    hidden_dim : int
        Hidden width shared across all ranks (default 64).
    n_layers : int
        Number of stacked MPSN layers (default 3).
    """

    def __init__(
        self, in_channels_all, hidden_dim: int = 64, n_layers: int = 3
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

        # Per-rank input projection to the shared hidden width.
        self.in_linear_0 = torch.nn.Linear(in_channels_all[0], hidden_dim)
        self.in_linear_1 = torch.nn.Linear(in_channels_all[1], hidden_dim)
        self.in_linear_2 = torch.nn.Linear(in_channels_all[2], hidden_dim)

        self.layers = torch.nn.ModuleList(
            MPSNLayer(hidden_dim) for _ in range(n_layers)
        )

    def forward(self, x_all, incidence_all):
        """Forward pass over the simplicial complex.

        Parameters
        ----------
        x_all : tuple of torch.Tensor
            Input features ``(x_0, x_1, x_2)`` on nodes, edges, triangles.
        incidence_all : tuple of torch.Tensor
            ``(incidence_1, incidence_2)``: the unsigned clique-lifted incidence
            matrices B1 (node-to-edge) and B2 (edge-to-triangle). May be sparse
            COO; densified and ``.abs()``-ed internally. Used only for routing.

        Returns
        -------
        tuple of torch.Tensor
            Final hidden states ``(x_0, x_1, x_2)``, each of width ``hidden_dim``.
        """
        x_0, x_1, x_2 = x_all
        b1 = _to_dense(incidence_all[0])
        b2 = _to_dense(incidence_all[1])

        # Precompute the rank-0 and rank-1 upper-adjacency operators once.
        upper_adj_0 = _upper_adjacency(b1)
        upper_adj_1 = _upper_adjacency(b2)

        x_0 = self.in_linear_0(x_0)
        x_1 = self.in_linear_1(x_1)
        x_2 = self.in_linear_2(x_2)

        packed_incidence = (b1, b2, upper_adj_0, upper_adj_1)
        x_all = (x_0, x_1, x_2)
        for layer in self.layers:
            x_all = layer(x_all, packed_incidence)

        return x_all
