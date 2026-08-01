"""Graph Neural Reaction-Diffusion Network backbone.

This module implements GREAD following Choi et al., "GREAD: Graph Neural
Reaction-Diffusion Networks" (ICML 2023). It supports all seven reaction
terms from the paper and sparse original or learned soft adjacency. Fixed-step
Euler and Runge-Kutta integration is implemented locally instead of through
the ``torchdiffeq`` package used by the reference implementation.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.utils import add_remaining_self_loops, coalesce, softmax

_REACTION_ALIASES = {
    "ac": "allen_cahn",
    "allen-cahn": "allen_cahn",
    "bs": "blurring_sharpening",
    "bspm": "blurring_sharpening",
    "f": "fisher",
    "fb": "filter_bank",
    "fb*": "filter_bank_star",
    "fb3": "filter_bank_star",
    "st": "source",
    "z": "zeldovich",
}
_REACTIONS = {
    "allen_cahn",
    "blurring_sharpening",
    "filter_bank",
    "filter_bank_star",
    "fisher",
    "source",
    "zeldovich",
}


class _SparseOperator:
    """Sparse matrix in PyG source-to-target convention."""

    __slots__ = ("num_nodes", "source", "target", "weight")

    def __init__(self, source, target, weight, num_nodes) -> None:
        self.source = source
        self.target = target
        self.weight = weight
        self.num_nodes = num_nodes

    def matmul(self, x: torch.Tensor) -> torch.Tensor:
        """Multiply the sparse operator by node features."""
        if self.source.numel() == 0:
            return torch.zeros_like(x)
        messages = x.index_select(0, self.source) * self.weight.unsqueeze(-1)
        return torch.zeros_like(x).index_add(0, self.target, messages)


class _PreparedEdges:
    """Canonical sparse graph structure reused during one forward pass."""

    __slots__ = ("edge_weight", "num_nodes", "source", "target")

    def __init__(self, source, target, edge_weight, num_nodes) -> None:
        self.source = source
        self.target = target
        self.edge_weight = edge_weight
        self.num_nodes = num_nodes


def _prepare_edges(
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor | None,
    num_nodes: int,
    dtype: torch.dtype,
    device: torch.device,
    make_undirected: bool,
    self_loop_weight: float,
) -> _PreparedEdges:
    """Move, symmetrize, coalesce, and optionally augment an edge list."""
    edge_index = edge_index.to(device=device)
    if edge_weight is None:
        edge_weight = torch.ones(
            edge_index.shape[1], dtype=dtype, device=device
        )
    else:
        edge_weight = edge_weight.to(dtype=dtype, device=device)

    if make_undirected and edge_index.numel() > 0:
        edge_index = torch.cat((edge_index, edge_index.flip(0)), dim=1)
        edge_weight = torch.cat((edge_weight, edge_weight), dim=0)
        edge_index, edge_weight = coalesce(
            edge_index, edge_weight, num_nodes, reduce="mean"
        )

    if self_loop_weight != 0:
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index,
            edge_weight,
            fill_value=self_loop_weight,
            num_nodes=num_nodes,
        )

    if edge_index.numel() == 0:
        empty = edge_index.new_empty(0)
        weights = torch.empty(0, dtype=dtype, device=device)
        return _PreparedEdges(empty, empty, weights, num_nodes)

    edge_index, edge_weight = coalesce(
        edge_index, edge_weight, num_nodes, reduce="mean"
    )
    return _PreparedEdges(edge_index[0], edge_index[1], edge_weight, num_nodes)


def _original_operator(
    edges: _PreparedEdges,
    normalization: str,
) -> _SparseOperator:
    """Build the normalized original adjacency operator."""
    degree = torch.zeros(
        edges.num_nodes,
        dtype=edges.edge_weight.dtype,
        device=edges.edge_weight.device,
    )
    degree.index_add_(0, edges.target, edges.edge_weight)

    if normalization == "row":
        inverse_degree = torch.zeros_like(degree)
        positive = degree > 0
        inverse_degree[positive] = degree[positive].reciprocal()
        weight = inverse_degree[edges.target] * edges.edge_weight
    else:
        inverse_sqrt = torch.zeros_like(degree)
        positive = degree > 0
        inverse_sqrt[positive] = degree[positive].rsqrt()
        weight = (
            inverse_sqrt[edges.target]
            * edges.edge_weight
            * inverse_sqrt[edges.source]
        )
    return _SparseOperator(edges.source, edges.target, weight, edges.num_nodes)


class GREADEncoder(nn.Module):
    """TopoBench-compatible GREAD node encoder.

    Parameters
    ----------
    input_dim : int
        Dimension of input node features.
    hidden_dim : int
        Dimension of hidden node representations.
    reaction_term : str, optional
        One of ``"blurring_sharpening"``, ``"fisher"``, ``"allen_cahn"``,
        ``"zeldovich"``, ``"source"``, ``"filter_bank"``, or
        ``"filter_bank_star"``. Paper and official-code aliases are accepted.
    adjacency : str, optional
        ``"soft"`` for learned sparse attention or ``"original"`` for the
        normalized observed adjacency.
    normalization : str, optional
        ``"row"`` or ``"symmetric"`` normalization for original adjacency.
    alpha_mode : str, optional
        ``"scalar"`` or ``"channel"`` diffusion coefficient.
    beta_mode : str, optional
        ``"scalar"`` or ``"channel"`` reaction coefficient.
    constrain_alpha : bool, optional
        Apply a sigmoid to the learned diffusion coefficient.
    constrain_beta : bool, optional
        Apply a sigmoid to the learned reaction coefficient.
    integration_time : float, optional
        Terminal time of the reaction-diffusion process.
    step_size : float, optional
        Maximum fixed integration step size.
    solver : str, optional
        Fixed-step ``"euler"`` or fourth-order ``"rk4"`` solver.
    heads : int, optional
        Number of soft-adjacency attention heads.
    attention_dim : int, optional
        Total query and key projection dimension.
    reweight_attention : bool, optional
        Multiply soft-adjacency logits by supplied edge weights.
    dynamic_adjacency : bool, optional
        Recompute learned soft adjacency from every intermediate ODE state.
        By default, compute it once from the initial state as in the reference
        implementation.
    self_loop_weight : float, optional
        Weight of self-loops added before graph preparation. Zero disables
        self-loop insertion.
    make_undirected : bool, optional
        Symmetrize the observed edge list before propagation.
    input_dropout : float, optional
        Dropout applied after the input projection.
    output_dropout : float, optional
        Dropout applied to the terminal ODE state.
    terminal_activation : bool, optional
        Apply ReLU to the terminal ODE state.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        reaction_term: str = "blurring_sharpening",
        adjacency: str = "soft",
        normalization: str = "row",
        alpha_mode: str = "scalar",
        beta_mode: str = "channel",
        constrain_alpha: bool = True,
        constrain_beta: bool = False,
        integration_time: float = 1.0,
        step_size: float = 0.5,
        solver: str = "rk4",
        heads: int = 4,
        attention_dim: int | None = None,
        reweight_attention: bool = False,
        dynamic_adjacency: bool = False,
        self_loop_weight: float = 0.0,
        make_undirected: bool = True,
        input_dropout: float = 0.0,
        output_dropout: float = 0.0,
        terminal_activation: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        reaction_term = _REACTION_ALIASES.get(reaction_term, reaction_term)
        if reaction_term not in _REACTIONS:
            raise ValueError(f"Unsupported reaction term: {reaction_term}.")
        if adjacency not in {"original", "soft"}:
            raise ValueError("adjacency must be 'original' or 'soft'.")
        if normalization not in {"row", "symmetric"}:
            raise ValueError("normalization must be 'row' or 'symmetric'.")
        if alpha_mode not in {"channel", "scalar"}:
            raise ValueError("alpha_mode must be 'scalar' or 'channel'.")
        if beta_mode not in {"channel", "scalar"}:
            raise ValueError("beta_mode must be 'scalar' or 'channel'.")
        if solver not in {"euler", "rk4"}:
            raise ValueError("solver must be 'euler' or 'rk4'.")
        if integration_time <= 0 or step_size <= 0:
            raise ValueError(
                "integration_time and step_size must be positive."
            )
        if heads < 1:
            raise ValueError("heads must be at least 1.")

        attention_dim = attention_dim or hidden_dim
        if attention_dim % heads != 0:
            raise ValueError("attention_dim must be divisible by heads.")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.out_channels = hidden_dim
        self.reaction_term = reaction_term
        self.adjacency = adjacency
        self.normalization = normalization
        self.alpha_mode = alpha_mode
        self.beta_mode = beta_mode
        self.constrain_alpha = constrain_alpha
        self.constrain_beta = constrain_beta
        self.integration_time = integration_time
        self.step_size = step_size
        self.solver = solver
        self.heads = heads
        self.attention_dim = attention_dim
        self.reweight_attention = reweight_attention
        self.dynamic_adjacency = dynamic_adjacency
        self.self_loop_weight = self_loop_weight
        self.make_undirected = make_undirected
        self.terminal_activation = terminal_activation

        self.input_proj = (
            nn.Identity()
            if input_dim == hidden_dim
            else nn.Linear(input_dim, hidden_dim)
        )
        self.input_dropout = nn.Dropout(input_dropout)
        self.output_dropout = nn.Dropout(output_dropout)

        alpha_shape = (1,) if alpha_mode == "scalar" else (hidden_dim,)
        beta_shape = (1,) if beta_mode == "scalar" else (hidden_dim,)
        self.alpha = nn.Parameter(torch.zeros(alpha_shape))
        self.beta = nn.Parameter(torch.empty(beta_shape))
        if constrain_beta:
            nn.init.zeros_(self.beta)
        else:
            nn.init.uniform_(self.beta, -0.1, 0.1)

        if adjacency == "soft":
            self.query = nn.Linear(hidden_dim, attention_dim)
            self.key = nn.Linear(hidden_dim, attention_dim)
            nn.init.xavier_uniform_(self.query.weight)
            nn.init.xavier_uniform_(self.key.weight)
            nn.init.zeros_(self.query.bias)
            nn.init.zeros_(self.key.bias)
        else:
            self.query = None
            self.key = None

    def _soft_operator(
        self,
        x: torch.Tensor,
        edges: _PreparedEdges,
    ) -> _SparseOperator:
        """Generate graph-local sparse soft adjacency."""
        head_dim = self.attention_dim // self.heads
        query = self.query(x).view(-1, self.heads, head_dim)
        key = self.key(x).view(-1, self.heads, head_dim)
        logits = (
            query.index_select(0, edges.target)
            * key.index_select(0, edges.source)
        ).sum(dim=-1) / math.sqrt(head_dim)
        if self.reweight_attention:
            logits = logits * edges.edge_weight.unsqueeze(-1)
        weight = softmax(logits, edges.target, num_nodes=edges.num_nodes).mean(
            dim=-1
        )
        return _SparseOperator(
            edges.source, edges.target, weight, edges.num_nodes
        )

    def _coefficients(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return effective diffusion and reaction coefficients."""
        alpha = (
            torch.sigmoid(self.alpha) if self.constrain_alpha else self.alpha
        )
        beta = torch.sigmoid(self.beta) if self.constrain_beta else self.beta
        return alpha, beta

    def _reaction(
        self,
        x: torch.Tensor,
        x0: torch.Tensor,
        diffusion: torch.Tensor,
        operator: _SparseOperator,
    ) -> torch.Tensor:
        """Evaluate the configured reaction term from Eq. (10)."""
        if self.reaction_term == "blurring_sharpening":
            return -operator.matmul(diffusion)
        if self.reaction_term == "fisher":
            return x * (1 - x)
        if self.reaction_term == "allen_cahn":
            return x * (1 - x.square())
        if self.reaction_term == "zeldovich":
            return x * (x - x.square())
        if self.reaction_term == "source":
            return x0
        if self.reaction_term == "filter_bank":
            # Eq. (10): Lx = x - Ax = -diffusion.
            return -diffusion
        return x - diffusion

    def _current_operator(
        self,
        x: torch.Tensor,
        fixed_operator: _SparseOperator | None,
        edges: _PreparedEdges,
    ) -> _SparseOperator:
        """Return the fixed operator or generate state-dependent adjacency."""
        if fixed_operator is not None:
            return fixed_operator
        return self._soft_operator(x, edges)

    def _derivative(
        self,
        x: torch.Tensor,
        x0: torch.Tensor,
        fixed_operator: _SparseOperator | None,
        edges: _PreparedEdges,
        alpha: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate the GREAD reaction-diffusion vector field."""
        operator = self._current_operator(x, fixed_operator, edges)
        diffusion = operator.matmul(x) - x
        reaction = self._reaction(x, x0, diffusion, operator)
        return alpha * diffusion + beta * reaction

    def _integrate(
        self,
        x0: torch.Tensor,
        fixed_operator: _SparseOperator | None,
        edges: _PreparedEdges,
    ) -> torch.Tensor:
        """Integrate the reaction-diffusion ODE with a fixed-step solver.

        The local implementation avoids an additional ODE-solver dependency.
        """
        num_steps = max(1, math.ceil(self.integration_time / self.step_size))
        step = self.integration_time / num_steps
        alpha, beta = self._coefficients()
        x = x0
        for _ in range(num_steps):
            if self.solver == "euler":
                derivative = self._derivative(
                    x, x0, fixed_operator, edges, alpha, beta
                )
                x = x + step * derivative
                continue
            k1 = self._derivative(x, x0, fixed_operator, edges, alpha, beta)
            k2 = self._derivative(
                x + 0.5 * step * k1,
                x0,
                fixed_operator,
                edges,
                alpha,
                beta,
            )
            k3 = self._derivative(
                x + 0.5 * step * k2,
                x0,
                fixed_operator,
                edges,
                alpha,
                beta,
            )
            k4 = self._derivative(
                x + step * k3, x0, fixed_operator, edges, alpha, beta
            )
            x = x + (step / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        return x

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor | None = None,
        edge_weight: torch.Tensor | None = None,
        edge_attr: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Encode node features with graph neural reaction-diffusion."""
        # Batched PyG graphs remain separated because propagation only follows
        # the supplied, disjoint edge sets.
        del batch
        if (
            edge_weight is None
            and edge_attr is not None
            and edge_attr.dim() == 1
        ):
            edge_weight = edge_attr

        x0 = self.input_dropout(self.input_proj(x))
        edges = _prepare_edges(
            edge_index,
            edge_weight,
            x0.shape[0],
            x0.dtype,
            x0.device,
            self.make_undirected,
            self.self_loop_weight,
        )
        if self.adjacency == "soft" and not self.dynamic_adjacency:
            fixed_operator = self._soft_operator(x0, edges)
        elif self.adjacency == "soft":
            fixed_operator = None
        else:
            fixed_operator = _original_operator(edges, self.normalization)

        out = self._integrate(x0, fixed_operator, edges)
        if self.terminal_activation:
            out = F.relu(out)
        return self.output_dropout(out)
