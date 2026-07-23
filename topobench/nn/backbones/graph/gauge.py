"""Riemannian Graph Foundation Model with neural vector bundles.

This module implements the gauge-equivariant graph model described in
"Are Common Substructures Transferable: Riemannian Graph Foundation Model
with Neural Vector Bundles".

The model learns, for every node, a local orthonormal frame (a gauge) that
spans an ``r``-dimensional subspace of the ``d``-dimensional embedding space,
smooths those frames across the graph, and uses them to update the node
features. The building blocks correspond directly to the equations of the
paper:

- :class:`LocalCoordinatesLayer` -- equations (2)-(4).
- :class:`GatedFlatteningLayer` -- equations (5)-(8).
- :class:`NodeUpdateLayer` -- equations (9)-(10).
- :class:`GaugeLayer` -- one full message-passing step combining the above.
- :class:`GaugeModel` -- the full stack of gauge layers.
"""

import math
from collections.abc import Callable

import torch
from torch import Tensor, nn
from torch_scatter import scatter_add, scatter_mean, scatter_softmax

activation_dict: dict[str, Callable] = {
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "gelu": nn.GELU,
    "sigmoid": nn.Sigmoid,
}


class MultiHeadLinear(nn.Module):
    """Per-head (per-subspace) linear layer.

    Applies ``r_dim`` independent linear maps, one per head, so that head ``h``
    transforms its own slice of the input with its own weight matrix and bias.
    The per-head weights are stored stacked in a single parameter and applied
    with a batched ``einsum``.

    Parameters
    ----------
    in_channels : int
        Number of input features per head.
    out_channels : int
        Number of output features per head.
    r_dim : int
        Number of heads (subspaces) ``r``.
    bias : bool, optional
        Whether each head uses a bias term (default: True).
    device : torch.device or str or None, optional
        Device on which to allocate the parameters (default: None).
    dtype : torch.dtype or None, optional
        Data type of the parameters (default: None).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        r_dim: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()

        factory_kwargs = {"device": device, "dtype": dtype}

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bias = bias
        self.r_dim = r_dim

        self.superW = nn.Parameter(
            torch.empty(
                self.r_dim,
                self.out_channels,
                self.in_channels,
                **factory_kwargs,
            )
        )

        if self.bias:
            self.superB = nn.Parameter(
                torch.empty(self.r_dim, self.out_channels, **factory_kwargs),
            )
        else:
            self.register_parameter("superB", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize the per-head weights and biases.

        Uses the same scheme as :class:`torch.nn.Linear` (a uniform
        distribution bounded by ``1 / sqrt(in_channels)``), with the fan-in
        taken per head rather than over the stacked parameter.
        """
        bound = 1 / math.sqrt(self.in_channels)

        torch.nn.init.uniform_(self.superW, -bound, bound)

        if self.bias:
            torch.nn.init.uniform_(self.superB, -bound, bound)

    def forward(self, Z: Tensor) -> Tensor:
        """Apply the per-head linear maps.

        Parameters
        ----------
        Z : Tensor
            Input tensor of shape ``[N, r, in_channels]`` where ``r`` is the
            number of heads.

        Returns
        -------
        Tensor
            Output tensor of shape ``[N, r, out_channels]``.
        """
        if self.bias:
            return torch.einsum("roi,Nri->Nro", self.superW, Z) + self.superB

        return torch.einsum("roi,Nri->Nro", self.superW, Z)


class MultiHeadFF(nn.Module):
    """Per-head feed-forward network (a distinct MLP per subspace).

    Stacks :class:`MultiHeadLinear` layers interleaved with activations and
    dropout so that each of the ``r`` heads is transformed by its own
    multi-layer perceptron. Activations and dropout are applied only between
    layers, never after the output layer.

    Parameters
    ----------
    in_channels : int
        Number of input features per head.
    out_channels : int
        Number of output features per head.
    r : int
        Number of heads (subspaces) ``r``.
    hidden_dims : list of int or None, optional
        Widths of the hidden layers. If ``None`` the network is a single
        per-head linear map (default: None).
    act : str, optional
        Name of the activation applied between layers, resolved via
        ``activation_dict`` (default: "leaky_relu").
    drop : float, optional
        Dropout probability applied between layers (default: 0.0).
    bias : bool, optional
        Whether each per-head linear layer uses a bias term (default: True).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        r: int,
        hidden_dims: list[int] | None = None,
        act: str = "leaky_relu",
        drop: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act = act
        self.dropout = drop
        self.r = r
        self.bias = bias

        self.layer_sizes = [self.in_channels]

        if hidden_dims is not None:
            self.layer_sizes += [j for j in hidden_dims]
        self.layer_sizes += [self.out_channels]

        els = []
        for j in range(len(self.layer_sizes) - 1):
            els.append(
                MultiHeadLinear(
                    self.layer_sizes[j],
                    self.layer_sizes[j + 1],
                    r_dim=self.r,
                    bias=self.bias,
                )
            )

            if j < len(self.layer_sizes) - 2:
                els.append(activation_dict[self.act]())
                els.append(nn.Dropout(self.dropout))

        self.model = nn.Sequential(*els)

    def forward(self, Z: Tensor) -> Tensor:
        """Apply the per-head feed-forward network.

        Parameters
        ----------
        Z : Tensor
            Input tensor of shape ``[N, r, in_channels]`` where ``r`` is the
            number of heads.

        Returns
        -------
        Tensor
            Output tensor of shape ``[N, r, out_channels]``.
        """
        Zhat = self.model(Z)

        return Zhat


class FFBlock(nn.Module):
    """Feed-forward block with activations, dropout and a final LayerNorm.

    The block consists of ``n_hidden_layers`` hidden linear layers with an
    activation resolved via ``activation_dict`` followed by an output linear
    layer, with dropout applied after every layer and layer normalization
    applied to the output.

    Parameters
    ----------
    in_channels : int
        Number of input features.
    out_channels : int
        Number of output features.
    hidden_dim : int
        Number of hidden units in the intermediate layers.
    n_hidden_layers : int, optional
        Number of hidden layers (must be at least 1) (default: 1).
    drop : float, optional
        Dropout probability (default: 0.3).
    bias : bool, optional
        Whether the linear layers use a bias term (default: True).
    act : str, optional
        Name of the activation applied after each hidden layer, resolved via
        ``activation_dict`` (default: "gelu").
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_dim: int,
        n_hidden_layers: int = 1,
        drop: float = 0.3,
        bias: bool = True,
        act: str = "gelu",
    ):
        super().__init__()

        self.dropout = drop
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_dimension = hidden_dim
        self.bias = bias
        self.n_hidden_layers = n_hidden_layers
        self.act = act

        assert self.n_hidden_layers >= 1

        els = []

        for layer_index in range(self.n_hidden_layers + 1):
            if layer_index == 0:
                els.append(
                    nn.Linear(
                        self.in_channels, self.hidden_dimension, bias=self.bias
                    )
                )
                els.append(activation_dict[self.act]())

            elif layer_index < self.n_hidden_layers:
                els.append(
                    nn.Linear(
                        self.hidden_dimension,
                        self.hidden_dimension,
                        bias=self.bias,
                    )
                )
                els.append(activation_dict[self.act]())

            else:
                els.append(
                    nn.Linear(
                        self.hidden_dimension,
                        self.out_channels,
                        bias=self.bias,
                    )
                )

            els.append(nn.Dropout(self.dropout))

        self.model = torch.nn.Sequential(*els)
        self.norm = nn.LayerNorm(self.in_channels)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape ``[..., in_channels]``.

        Returns
        -------
        Tensor
            Output tensor of shape ``[..., out_channels]``.
        """

        x = self.norm(x)
        x = self.model(x)

        return x


class LocalCoordinatesLayer(torch.nn.Module):
    """Local coordinate frame layer (equations (2)-(4)).

    For each node this layer projects the node embeddings into ``r`` different
    subspaces, aggregates a smoothed reconstruction over the neighborhood using
    attention weights, and applies a QR decomposition to obtain, per node, an
    orthonormal basis (a local gauge) spanning an ``r``-dimensional subspace of
    the embedding space.

    Parameters
    ----------
    r_subspaces : int
        Number of subspaces (frame vectors) ``r`` learned per node.
    d_embedd : int
        Dimension ``d`` of the node embeddings.
    tau : float, optional
        Temperature used to scale the attention logits (default: 1.0).
    bias : bool, optional
        Whether the linear layers use a bias term (default: True).
    act : str, optional
        Name of the activation used by the feed-forward block, resolved via
        ``activation_dict`` (default: "gelu").
    f_sim_act : str, optional
        Name of the activation used by the per-head similarity network
        ``f_sim``, resolved via ``activation_dict`` (default: "leaky_relu").
    """

    # eqns. 2-4
    def __init__(
        self,
        r_subspaces: int,
        d_embedd: int,
        tau: float = 1.0,
        bias: bool = True,
        act: str = "gelu",
        f_sim_act: str = "leaky_relu",
    ):
        super().__init__()

        self.r = r_subspaces
        self.tau = tau
        self.d = d_embedd
        self.bias = bias
        self.act = act
        self.f_sim_act = f_sim_act

        # combine the projectors into a single nn.Linear layer, reshape afterwards!
        self.initial_projector = torch.nn.Linear(
            self.d, self.d * self.r, bias=self.bias
        )

        # f_sim = f, computing similarity of node features
        self.f_sim = MultiHeadFF(
            2 * self.d,
            1,
            r=self.r,
            hidden_dims=[2 * self.d],
            act=self.f_sim_act,
        )

        self.fflayer = FFBlock(
            self.d, self.d, self.d, bias=self.bias, act=self.act
        )
        self.preqr_norm = nn.LayerNorm(self.d)

    def forward(self, Z: Tensor, edge_index: Tensor) -> Tensor:
        """Forward pass computing per-node local orthonormal frames.

        Implements equations (2)-(4): the neighborhood-smoothed reconstruction
        of the node embeddings (equations (2)-(3)) followed by a QR
        decomposition yielding an orthonormal basis per node (equation (4)).

        Parameters
        ----------
        Z : Tensor
            Node embeddings of shape ``[N, d]``.
        edge_index : Tensor
            Edge index tensor of shape ``[2, E]`` with source and destination
            node indices.

        Returns
        -------
        Tensor
            Per-node orthonormal frames of shape ``[N, r, d]``.
        """

        N = Z.size(0)  # num_nodes
        src, dst = edge_index[0], edge_index[1]

        # nr. 1: we project the input matrix x into r different subspaces
        Zh = self.initial_projector(Z)  # [N, r*d]
        Zh = Zh.reshape(N, self.r, self.d)  # [N, r, d]

        # EQUATION no. (3)
        # f_vals has shape [N, r, 1]
        f_vals = (
            self.f_sim(torch.concat((Zh[src], Zh[dst]), dim=-1)) / self.tau
        )

        f_vals = f_vals.squeeze(-1)  # remove last singleton dimension
        alphas = torch.softmax(f_vals, dim=-1).unsqueeze(-1)  # [E, r, 1]

        # EQUATION no. (2)
        out = scatter_add(
            alphas * Zh[src, :, :], index=dst, dim=0, dim_size=N
        )  # tensor of shape (E, r, d)

        # we need to clamp as nodes with degree 0 would have a scatter_add of 0
        # this should give us a tensor of shape [N, r]
        norm = 1 / (scatter_add(alphas, dst, dim=0, dim_size=N).clamp(1e-6))

        # norm*out should be [N, r, d] with norm broadcasted along the last dimension (d)
        # norm*out is of shape [N,r,d] while Z is of shape [N,d], hence we insert a new axis at -2
        qhat = Z.unsqueeze(-2) - norm * out

        # feedforward followed by a LayerNorm, then QR (eq. 4)
        qhat = self.fflayer(qhat)
        qhat = self.preqr_norm(qhat)

        # EQUATION no. (4)
        # xx has shape [N, r, d] so now we can do the QR decomposition to obtain an orthonormal basis
        Q, _ = torch.linalg.qr(qhat.mT)

        return Q.mT


class GatedFlatteningLayer(nn.Module):
    """Gated flattening layer that smooths local frames (equations (5)-(8)).

    This layer aligns each node's local frame with those of its neighbors. It
    computes gating weights from the overlap between neighboring frames
    (equation (6)), forms a gated aggregate that is blended with the original
    frame (equation (7)), and re-orthonormalizes the result via a QR
    decomposition (equation (8)).

    Parameters
    ----------
    r : int
        Number of subspaces (frame vectors) ``r`` per node.
    gamma : float, optional
        Blending coefficient between the original frame and the aggregated
        neighbor frames (default: 0.01).
    tau : float, optional
        Temperature used to scale the gating logits (default: 1.0).
    """

    # eqns.5-8

    def __init__(self, r: int, gamma: float = 0.01, tau: float = 1.0):
        super().__init__()

        self.r = r
        self.gamma = gamma
        self.tau = tau

    def forward(self, Q: Tensor, edge_index: Tensor) -> Tensor:
        """Forward pass smoothing the per-node frames over the graph.

        Implements equations (6)-(8): gating weights from neighboring frame
        overlaps (equation (6)), a gated aggregate blended with the input frame
        (equation (7)), and a final QR re-orthonormalization (equation (8)).

        Parameters
        ----------
        Q : Tensor
            Per-node orthonormal frames of shape ``[N, r, d]``.
        edge_index : Tensor
            Edge index tensor of shape ``[2, E]`` with source and destination
            node indices.

        Returns
        -------
        Tensor
            Smoothed per-node orthonormal frames of shape ``[N, r, d]``.
        """
        # eqns. (6-8)

        N = Q.size(0)  # num_nodes
        src, dst = edge_index[0], edge_index[1]
        k = Q.size(-2)  # with k fixed, the trace of eye(k) = k

        # EQUATION no. (6)
        # the trace of the identity of size k = k
        g_vec = scatter_softmax(
            ((Q[src] * Q[dst]).sum((-2, -1)) - k) / self.tau,
            index=dst,
            dim_size=N,
        )

        # technical note: in theory wed need to compute gij
        # for all pairs of nodes which becomes unnecessary only because we only
        # sum over neighbors menaning that non-neighbor entries are irrelevant

        # EQUATION no. (7)
        Qagg = scatter_add(
            g_vec[:, None, None] * Q[src], dim=0, index=dst, dim_size=N
        )
        Qhat = (1 - self.gamma) * Q + self.gamma * Qagg

        # EQUATION no. (8)
        # lastly we do the QR decomposition again to obtain an orthonormal basis:
        Qnew, _ = torch.linalg.qr(Qhat.mT)

        return Qnew.mT


class NodeUpdateLayer(torch.nn.Module):
    """Node feature update layer using the local frames (equations (9)-(10)).

    Each node embedding is projected onto the subspace spanned by its local
    frame and mapped through a learnable matrix (equation (9)). The projected
    features are then aggregated over the neighborhood and, when the residual is
    enabled, combined with a learnable transformation ``phi`` of the original
    embedding (equation (10)). Setting ``phi_hidden_layers`` to ``None`` disables
    the residual and recovers the reference behavior.

    Parameters
    ----------
    in_channels : int
        Number of input features.
    out_channels : int
        Number of output features.
    phi_hidden_layers : int or None, optional
        Number of hidden layers of the MLP residual ``phi``. Must be at least 1
        when not ``None``. If ``None`` the residual is disabled (matching the
        reference implementation) (default: 1).
    phi_hidden_dim : int or None, optional
        Hidden width of the residual MLP ``phi``. Defaults to
        ``max(in_channels, out_channels)`` when ``None`` (default: None).
    act : str, optional
        Name of the activation used by the residual MLP ``phi``, resolved via
        ``activation_dict`` (default: "gelu").
    """

    # eqns. 9-10
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        phi_hidden_layers: int | None = 1,
        phi_hidden_dim: int | None = None,
        act: str = "gelu",
    ):
        super().__init__()

        self.phi = None
        # this is the learnable function applied to z (if phi_hidden_layers isn't None)
        if phi_hidden_layers is not None:
            self.phi = FFBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                hidden_dim=phi_hidden_dim
                if phi_hidden_dim is not None
                else max(in_channels, out_channels),
                n_hidden_layers=phi_hidden_layers,
                act=act,
            )

        # this is the learnable matrix applied to tilde(z)
        self.W = torch.nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, Z: Tensor, Q: Tensor, edge_index: Tensor) -> Tensor:
        """Forward pass updating the node features.

        Implements equations (9)-(10): the frame projection of the node
        embeddings (equation (9)) followed by the neighborhood aggregation and,
        when enabled, a learnable residual connection (equation (10)).

        Parameters
        ----------
        Z : Tensor
            Node embeddings of shape ``[N, d]``.
        Q : Tensor
            Per-node orthonormal frames of shape ``[N, r, d]``.
        edge_index : Tensor
            Edge index tensor of shape ``[2, E]`` with source and destination
            node indices.

        Returns
        -------
        Tensor
            Updated node embeddings of shape ``[N, out_channels]``.
        """

        # step 0: bind commonly used values to variable names
        src, dst = edge_index[0], edge_index[1]
        N = Z.size(0)  # num_nodes for scatter ops

        # step 1: calculate tilde(z)

        # EQUATION no. (9)
        # Q has shape [N,r,d] and z has shape [N, d]
        # we want to transform each vector in z via the matrix [r,d] batching over the first dimension
        QtZ = torch.einsum("ijk,ik->ij", Q, Z)
        Z_tilde = torch.einsum("ikj, ik->ij", Q, QtZ)
        Z_tilde = self.W(Z_tilde)

        # EQUATION no. (10)
        # DIVERGENCE FROM REFERENCE IMPLEMENTATION
        # Contrary to the reference implementation we optionally add a "residual
        # connection" realized via the self.phi function. It is enabled by
        # default and can be disabled (recovering the reference behavior) by
        # passing phi_hidden_layers=None, in which case self.phi is None.
        # cf. equation (10)
        Znew = scatter_mean(Z_tilde[src], index=dst, dim=0, dim_size=N)

        if self.phi is not None:
            Znew = Znew + self.phi(Z)

        return Znew


class GaugeLayer(torch.nn.Module):
    """A single gauge message-passing layer.

    One layer computes per-node local frames with a
    :class:`LocalCoordinatesLayer` (equations (2)-(4)), smooths them through a
    stack of ``n_gated`` :class:`GatedFlatteningLayer` modules (equations
    (5)-(8)), and updates the node features with a :class:`NodeUpdateLayer`
    (equations (9)-(10)).

    Parameters
    ----------
    d_embedd : int
        Dimension ``d`` of the node embeddings.
    r : int
        Number of subspaces (frame vectors) ``r`` per node.
    n_gated : int, optional
        Number of gated flattening layers applied to the frames (default: 1).
    gamma : float, optional
        Blending coefficient used by the gated flattening layers (default: 0.01).
    tau : float, optional
        Temperature used to scale the attention and gating logits (default: 1.0).
    bias : bool, optional
        Whether the linear layers use a bias term (default: True).
    phi_hidden_layers : int or None, optional
        Number of hidden layers of the MLP residual ``phi`` in the node update.
        Must be at least 1 when not ``None``. If ``None`` the residual is
        disabled (matching the reference implementation) (default: 1).
    phi_hidden_dim : int or None, optional
        Hidden width of the residual MLP ``phi``. Defaults to ``d_embedd`` when
        ``None`` (default: None).
    act : str, optional
        Name of the activation used by the feed-forward blocks, resolved via
        ``activation_dict`` (default: "gelu").
    f_sim_act : str, optional
        Name of the activation used by the per-head similarity network
        ``f_sim``, resolved via ``activation_dict`` (default: "leaky_relu").
    """

    def __init__(
        self,
        d_embedd: int,
        r: int,
        n_gated: int = 1,
        gamma: float = 0.01,
        tau: float = 1.0,
        bias=True,
        phi_hidden_layers: int | None = 1,
        phi_hidden_dim: int | None = None,
        act: str = "gelu",
        f_sim_act: str = "leaky_relu",
    ):
        super().__init__()

        self.r = r
        self.bias = bias
        self.tau = tau
        self.n_gated = n_gated
        self.gamma = gamma
        self.d_embedd = d_embedd
        self.act = act
        self.f_sim_act = f_sim_act

        self.local_coords_layer = LocalCoordinatesLayer(
            r_subspaces=r,
            d_embedd=d_embedd,
            tau=tau,
            bias=bias,
            act=act,
            f_sim_act=f_sim_act,
        )

        self.gated_flattening_layers = nn.ModuleList(
            [
                GatedFlatteningLayer(self.r, self.gamma, self.tau)
                for _ in range(n_gated)
            ]
        )

        self.node_update_layer = NodeUpdateLayer(
            self.d_embedd,
            self.d_embedd,
            phi_hidden_layers=phi_hidden_layers,
            phi_hidden_dim=phi_hidden_dim,
            act=act,
        )

    def forward(self, x: Tensor, edge_index: Tensor) -> tuple[Tensor, Tensor]:
        """Forward pass of a single gauge layer.

        Parameters
        ----------
        x : Tensor
            Node embeddings of shape ``[N, d]``.
        edge_index : Tensor
            Edge index tensor of shape ``[2, E]`` with source and destination
            node indices.

        Returns
        -------
        Znew : Tensor
            Updated node embeddings of shape ``[N, d]``.
        Q : Tensor
            Per-node orthonormal frames of shape ``[N, r, d]``.
        """

        Q = self.local_coords_layer(x, edge_index)

        for layer in self.gated_flattening_layers:
            Q = layer(Q, edge_index)

        Znew = self.node_update_layer(x, Q, edge_index)

        return Znew, Q


class GaugeModel(nn.Module):
    """Riemannian graph foundation model with neural vector bundles.

    The model first projects the input features into a ``d_embedd``-dimensional
    embedding space and then applies a stack of :class:`GaugeLayer` modules,
    each of which learns per-node local frames and uses them to update the node
    features.

    Parameters
    ----------
    n_layers : int
        Number of gauge layers.
    in_channels : int
        Number of input features.
    r : int
        Number of subspaces (frame vectors) ``r`` per node.
    d_embedd : int
        Dimension ``d`` of the node embeddings.
    n_gated : int, optional
        Number of gated flattening layers per gauge layer (default: 1).
    gamma : float, optional
        Blending coefficient used by the gated flattening layers (default: 0.01).
    tau : float, optional
        Temperature used to scale the attention and gating logits (default: 1.0).
    bias : bool, optional
        Whether the linear layers use a bias term (default: True).
    phi_hidden_layers : int or None, optional
        Number of hidden layers of the MLP residual ``phi`` in the node update.
        Must be at least 1 when not ``None``. If ``None`` the residual is
        disabled (matching the reference implementation) (default: 1).
    phi_hidden_dim : int or None, optional
        Hidden width of the residual MLP ``phi``. Defaults to ``d_embedd`` when
        ``None`` (default: None).
    act : str, optional
        Name of the activation used by the feed-forward blocks, resolved via
        ``activation_dict`` (default: "gelu").
    f_sim_act : str, optional
        Name of the activation used by the per-head similarity network
        ``f_sim``, resolved via ``activation_dict`` (default: "leaky_relu").
    """

    def __init__(
        self,
        n_layers: int,
        in_channels: int,
        r: int,
        d_embedd: int,
        n_gated: int = 1,
        gamma=0.01,
        tau: float = 1.0,
        bias=True,
        phi_hidden_layers: int | None = 1,
        phi_hidden_dim: int | None = None,
        act: str = "gelu",
        f_sim_act: str = "leaky_relu",
    ):
        super().__init__()

        self.n_layers = n_layers
        self.gamma = gamma
        self.tau = tau
        self.bias = bias
        self.in_channels = in_channels
        self.r = r
        self.d_embedd = d_embedd
        self.n_gated = n_gated
        self.act = act
        self.f_sim_act = f_sim_act

        self.input_projector = nn.Sequential(
            nn.Linear(in_channels, d_embedd), nn.LayerNorm(d_embedd)
        )

        self.layers = nn.ModuleList(
            [
                GaugeLayer(
                    d_embedd=self.d_embedd,
                    r=self.r,
                    gamma=self.gamma,
                    tau=self.tau,
                    n_gated=self.n_gated,
                    bias=bias,
                    phi_hidden_layers=phi_hidden_layers,
                    phi_hidden_dim=phi_hidden_dim,
                    act=self.act,
                    f_sim_act=self.f_sim_act,
                )
                for _ in range(self.n_layers)
            ]
        )

    def forward(self, x: Tensor, edge_index: Tensor) -> tuple[Tensor, Tensor]:
        """Forward pass of the full model.

        Parameters
        ----------
        x : Tensor
            Input node features of shape ``[N, in_channels]``.
        edge_index : Tensor
            Edge index tensor of shape ``[2, E]`` with source and destination
            node indices.

        Returns
        -------
        z : Tensor
            Final node embeddings of shape ``[N, d_embedd]``.
        Q : Tensor
            Per-node orthonormal frames of shape ``[N, r, d_embedd]`` from the
            last gauge layer.
        """
        z = self.input_projector(x)

        for layer in self.layers:
            z, Q = layer(z, edge_index)

        return z, Q
