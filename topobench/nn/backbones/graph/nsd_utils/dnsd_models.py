"""Core layers for Deep Neural Sheaf Diffusion (DNSD).

This module ports the reference DNSD implementation [1] into a self-contained
set of PyTorch Geometric ``MessagePassing`` building blocks:

- :class:`Orthogonal` — differentiable orthogonal-map parameterizations.
- :class:`FullRestrictionBuilder`, :class:`DiagonalRestrictionBuilder`,
  :class:`OrthogonalRestrictionBuilder` — learn the ``d x d`` restriction map
  ``F_{v<e}`` on each edge from the concatenated endpoint features.
- :class:`DNSDConv` — one DNSD diffusion step.
- :class:`DeepNeuralSheafDiffusion` — the full stacked model.

Unlike the ``spmm``-based Neural Sheaf Diffusion already in
:mod:`~topobench.nn.backbones.graph.nsd_utils.inductive_discrete_models`, DNSD
replaces the sheaf *Laplacian* with a sheaf *adjacency* operator (the
``directional`` flag), which keeps the disagreement signal informative at
depth, and complements it with normalization, an odd (``tanh``) nonlinearity,
and per-stalk gating [1].

References
----------
[1] Bourgerie, Girdzijauskas and Fodor. "Deep Neural Sheaf Diffusion."
    ICML 2026 Workshop on Graph Foundation Models. https://arxiv.org/abs/2605.19021
    Reference code: https://github.com/remibourgerie/deep-neural-sheaf-diffusion
"""

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree


class Orthogonal(nn.Module):
    """
    Differentiable orthogonal-matrix parameterization.

    Maps a batch of ``d x d`` parameter matrices to orthogonal matrices, used
    to build orthogonal restriction maps. All methods are pure PyTorch.

    This is intentionally kept separate from
    :class:`~topobench.nn.backbones.graph.nsd_utils.orthogonal.Orthogonal`:
    the two are not interchangeable. The NSD one consumes lower-triangular
    skew parameters and offers only ``matrix_exp``/``cayley``, whereas DNSD is
    ported faithfully from the reference implementation, which takes a full
    ``d x d`` parameter matrix and additionally supports ``qr`` and
    ``householder``. Sharing a class would silently change the map
    parameterization; unifying (if ever) should promote this richer version.

    Parameters
    ----------
    d : int
        Size of the (square) orthogonal matrices.
    method : str, optional
        Parameterization to use. One of ``'cayley'`` (default; Cayley
        transform, well-behaved gradients), ``'matrix_exp'`` (matrix
        exponential of a skew-symmetric matrix), ``'householder'`` (product of
        Householder reflectors) or ``'qr'`` (QR decomposition).

    Raises
    ------
    ValueError
        If ``method`` is not one of the supported options.
    """

    def __init__(self, d, method="cayley"):
        super().__init__()
        self.d = d
        if method not in ("cayley", "matrix_exp", "householder", "qr"):
            raise ValueError(f"Unknown orthogonalization method '{method}'.")
        self.method = method

    def forward(self, params):
        """
        Orthogonalize a batch of parameter matrices.

        Parameters
        ----------
        params : torch.Tensor
            Parameter tensor of shape [num_edges, d * d].

        Returns
        -------
        torch.Tensor
            Orthogonal matrices of shape [num_edges, d, d].
        """
        e = params.size(0)
        d = self.d
        a = params.view(e, d, d)

        if self.method == "householder":
            eye = torch.eye(d, device=params.device).unsqueeze(0)
            q = eye.expand(e, -1, -1).clone()
            for i in range(d):
                v = a[:, i, :]
                v = v / (v.norm(dim=-1, keepdim=True) + 1e-8)
                h = eye - 2 * torch.einsum("bi,bj->bij", v, v)
                q = q @ h
            return q
        if self.method == "matrix_exp":
            skew = a - a.transpose(-1, -2)
            return torch.matrix_exp(skew)
        if self.method == "cayley":
            skew = a - a.transpose(-1, -2)
            eye = torch.eye(d, device=skew.device).unsqueeze(0)
            return torch.linalg.solve(eye + skew, eye - skew)
        # qr
        a = a + 1e-6 * torch.eye(d, device=a.device).unsqueeze(0)
        q, _ = torch.linalg.qr(a)
        return q


class _RestrictionBuilder(nn.Module):
    """
    Base class for edge-wise restriction-map builders.

    A restriction-map builder produces the ``d x d`` map ``F_{v<e}`` for every
    directed edge from the concatenated features of its two endpoints.
    """

    def forward(self, x, edge_index):
        """
        Build restriction maps for every edge.

        Parameters
        ----------
        x : torch.Tensor
            Node feature matrix of shape [num_nodes, hidden_dim].
        edge_index : torch.Tensor
            Edge indices of shape [2, num_edges].

        Raises
        ------
        NotImplementedError
            This is an abstract method that subclasses must implement.
        """
        raise NotImplementedError


class FullRestrictionBuilder(_RestrictionBuilder):
    """
    Learns unconstrained (full) ``d x d`` restriction maps.

    Parameters
    ----------
    hidden_dim : int
        Node feature dimension (per endpoint).
    num_stalks : int
        Stalk dimension ``d``; each map is ``d x d``.
    init_scale : float, optional
        Multiplier applied to the initial linear weights/bias. Default is 1.0.
    """

    def __init__(self, hidden_dim, num_stalks, init_scale=1.0):
        super().__init__()
        self.d = num_stalks
        self.lin = nn.Linear(2 * hidden_dim, num_stalks * num_stalks)
        _scale_linear_(self.lin, init_scale)

    def forward(self, x, edge_index):
        """
        Build ``tanh``-bounded full restriction maps.

        Parameters
        ----------
        x : torch.Tensor
            Node feature matrix of shape [num_nodes, hidden_dim].
        edge_index : torch.Tensor
            Edge indices of shape [2, num_edges].

        Returns
        -------
        torch.Tensor
            Restriction maps of shape [num_edges, d, d].
        """
        u, v = edge_index
        x_cat = torch.cat([x[u], x[v]], dim=-1)
        maps = self.lin(x_cat).view(-1, self.d, self.d)
        return torch.tanh(maps)


class DiagonalRestrictionBuilder(_RestrictionBuilder):
    """
    Learns diagonal ``d x d`` restriction maps.

    Only the ``d`` diagonal entries are parameterized, giving a cheaper,
    channel-independent sheaf.

    Parameters
    ----------
    hidden_dim : int
        Node feature dimension (per endpoint).
    num_stalks : int
        Stalk dimension ``d``.
    init_scale : float, optional
        Multiplier applied to the initial linear weights/bias. Default is 1.0.
    """

    def __init__(self, hidden_dim, num_stalks, init_scale=1.0):
        super().__init__()
        self.d = num_stalks
        self.lin = nn.Linear(2 * hidden_dim, num_stalks)
        _scale_linear_(self.lin, init_scale)

    def forward(self, x, edge_index):
        """
        Build ``tanh``-bounded diagonal restriction maps.

        Parameters
        ----------
        x : torch.Tensor
            Node feature matrix of shape [num_nodes, hidden_dim].
        edge_index : torch.Tensor
            Edge indices of shape [2, num_edges].

        Returns
        -------
        torch.Tensor
            Diagonal restriction maps of shape [num_edges, d, d].
        """
        u, v = edge_index
        x_cat = torch.cat([x[u], x[v]], dim=-1)
        f = torch.tanh(self.lin(x_cat))
        return torch.diag_embed(f)


class OrthogonalRestrictionBuilder(_RestrictionBuilder):
    """
    Learns orthogonal ``d x d`` restriction maps.

    The linear layer outputs raw parameters that :class:`Orthogonal` maps onto
    the orthogonal group, yielding norm-preserving transport maps.

    Parameters
    ----------
    hidden_dim : int
        Node feature dimension (per endpoint).
    num_stalks : int
        Stalk dimension ``d``.
    init_scale : float, optional
        Multiplier applied to the initial linear weights/bias. Default is 1.0.
    ortho_method : str, optional
        Orthogonalization method passed to :class:`Orthogonal`. Default is
        ``'qr'`` (the paper's / reference implementation's default).
    """

    def __init__(
        self, hidden_dim, num_stalks, init_scale=1.0, ortho_method="qr"
    ):
        super().__init__()
        self.d = num_stalks
        self.lin = nn.Linear(2 * hidden_dim, num_stalks * num_stalks)
        self.ortho = Orthogonal(num_stalks, method=ortho_method)
        _scale_linear_(self.lin, init_scale)

    def forward(self, x, edge_index):
        """
        Build orthogonal restriction maps.

        Parameters
        ----------
        x : torch.Tensor
            Node feature matrix of shape [num_nodes, hidden_dim].
        edge_index : torch.Tensor
            Edge indices of shape [2, num_edges].

        Returns
        -------
        torch.Tensor
            Orthogonal restriction maps of shape [num_edges, d, d].
        """
        u, v = edge_index
        x_cat = torch.cat([x[u], x[v]], dim=-1)
        return self.ortho(self.lin(x_cat))


def _scale_linear_(lin, init_scale):
    """
    Scale a linear layer's initial weights and bias in place.

    Parameters
    ----------
    lin : torch.nn.Linear
        Linear layer whose parameters are rescaled.
    init_scale : float
        Multiplier applied to the weights and bias. A no-op when equal to 1.0.

    Returns
    -------
    None
        None.
    """
    if init_scale == 1.0:
        return
    with torch.no_grad():
        lin.weight.data *= init_scale
        if lin.bias is not None:
            lin.bias.data *= init_scale


_BUILDER_MAP = {
    "full": FullRestrictionBuilder,
    "diagonal": DiagonalRestrictionBuilder,
    "orthogonal": OrthogonalRestrictionBuilder,
}


class DNSDConv(MessagePassing):
    """
    One Deep Neural Sheaf Diffusion diffusion step.

    Computes, per stalk ``s`` and channel block ``h``::

        x' = (1 + eps) * x - phi( W1 @ agg(x) @ W2 )

    where ``agg`` is the degree-normalized sheaf aggregation. The three DNSD
    flags (Table 4 in [1]) switch on the deep-friendly behavior:

    - ``directional``: aggregate ``F_target x_target`` (a sheaf *adjacency*
      operator) instead of the Laplacian disagreement
      ``F_source x_source - F_target x_target``.
    - ``odd``: use ``tanh`` (odd) instead of ``ReLU`` for ``phi``.
    - ``use_stalk_gate``: apply a learned per-stalk sigmoid gate to the update.

    Parameters
    ----------
    hidden_dim : int
        Total hidden dimension; ``stalk_dim = hidden_dim // num_stalks``.
    num_stalks : int
        Stalk dimension ``d``.
    stalk_dropout : float, optional
        Probability of dropping a whole stalk of the update during training.
        Default is 0.0.
    directional : bool, optional
        Use the sheaf adjacency operator instead of the Laplacian. Default is
        False.
    odd : bool, optional
        Use an odd (``tanh``) nonlinearity. Default is False.
    use_stalk_gate : bool, optional
        Enable the per-stalk sigmoid gate. Default is False.

    References
    ----------
    [1] Bourgerie et al. "Deep Neural Sheaf Diffusion."
        https://arxiv.org/abs/2605.19021
    """

    def __init__(
        self,
        hidden_dim,
        num_stalks,
        stalk_dropout=0.0,
        directional=False,
        odd=False,
        use_stalk_gate=False,
    ):
        super().__init__(aggr="add", flow="target_to_source")
        self.hidden_dim = hidden_dim
        self.num_stalks = num_stalks
        self.stalk_dim = hidden_dim // num_stalks
        self.stalk_dropout = stalk_dropout
        self.directional = directional
        self.odd = odd
        self.use_stalk_gate = use_stalk_gate

        # Channel-mixing weights, initialized near identity so the untrained
        # layer is close to a plain diffusion step.
        self.W1 = nn.Parameter(
            torch.eye(num_stalks) + 0.01 * torch.randn(num_stalks, num_stalks)
        )
        self.W2 = nn.Parameter(
            torch.eye(self.stalk_dim)
            + 0.01 * torch.randn(self.stalk_dim, self.stalk_dim)
        )
        self.eps = nn.Parameter(torch.zeros(num_stalks))

        if use_stalk_gate:
            # Gate starts at sigmoid(0) = 0.5; zero-init keeps it symmetric.
            self.gate_lin = nn.Linear(2 * self.stalk_dim, 1, bias=True)
            nn.init.zeros_(self.gate_lin.weight)
            nn.init.zeros_(self.gate_lin.bias)

    def forward(self, x, edge_index, f_source, f_target):
        """
        Run one diffusion step.

        Parameters
        ----------
        x : torch.Tensor
            Node features of shape [num_nodes, hidden_dim].
        edge_index : torch.Tensor
            Edge indices of shape [2, num_edges].
        f_source : torch.Tensor
            Source restriction maps of shape [num_edges, d, d].
        f_target : torch.Tensor
            Target restriction maps of shape [num_edges, d, d].

        Returns
        -------
        torch.Tensor
            Updated node features of shape [num_nodes, hidden_dim].
        """
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.clamp(min=1).pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        return self.propagate(
            edge_index, x=x, f_source=f_source, f_target=f_target, norm=norm
        )

    def message(self, x_i, x_j, f_source, f_target, norm):
        """
        Build the per-edge sheaf message.

        Parameters
        ----------
        x_i : torch.Tensor
            Source-node features of shape [num_edges, hidden_dim].
        x_j : torch.Tensor
            Target-node features of shape [num_edges, hidden_dim].
        f_source : torch.Tensor
            Source restriction maps of shape [num_edges, d, d].
        f_target : torch.Tensor
            Target restriction maps of shape [num_edges, d, d].
        norm : torch.Tensor
            Symmetric degree normalization of shape [num_edges].

        Returns
        -------
        torch.Tensor
            Messages of shape [num_edges, hidden_dim].
        """
        e = x_i.size(0)
        s, h = self.num_stalks, self.stalk_dim
        x_target = x_j.view(e, s, h)

        fx_target = torch.bmm(f_target, x_target)
        if self.directional:
            diff = fx_target
        else:
            x_source = x_i.view(e, s, h)
            fx_source = torch.bmm(f_source, x_source)
            diff = fx_source - fx_target

        msg = torch.bmm(f_source.transpose(1, 2), diff)
        return (msg * norm.view(-1, 1, 1)).view(e, -1)

    def update(self, aggr_out, x):
        """
        Apply the channel mix, nonlinearity, optional gate, and residual.

        Parameters
        ----------
        aggr_out : torch.Tensor
            Aggregated messages of shape [num_nodes, hidden_dim].
        x : torch.Tensor
            Node features before the step, of shape [num_nodes, hidden_dim].

        Returns
        -------
        torch.Tensor
            Updated node features of shape [num_nodes, hidden_dim].
        """
        n = x.size(0)
        s, h = self.num_stalks, self.stalk_dim
        x_stalk = x.view(n, s, h)
        agg_stalk = aggr_out.view(n, s, h)

        transformed = torch.einsum("ij,njk,kl->nil", self.W1, agg_stalk, self.W2)
        transformed = torch.tanh(transformed) if self.odd else F.relu(transformed)

        if self.use_stalk_gate:
            gate_input = torch.cat([x_stalk, agg_stalk], dim=-1)
            gate = torch.sigmoid(
                self.gate_lin(gate_input.view(-1, 2 * h))
            ).view(n, s, 1)
            transformed = gate * transformed

        if self.stalk_dropout > 0.0 and self.training:
            mask = torch.ones(n, s, 1, device=transformed.device)
            mask = F.dropout(mask, p=self.stalk_dropout, training=True)
            transformed = transformed * mask

        eps = (1 + self.eps).view(1, s, 1)
        return (eps * x_stalk - transformed).view(n, -1)


class DeepNeuralSheafDiffusion(nn.Module):
    """
    Deep Neural Sheaf Diffusion model [1].

    Stacks :class:`DNSDConv` layers, each with its own learned source and
    target restriction maps, optional per-layer LayerNorm, and the DNSD deep
    flags. The final linear layer projects the stalk-major hidden state to
    ``output_dim``.

    Parameters
    ----------
    input_dim : int
        Input feature dimension.
    hidden_dim : int
        Total hidden dimension. Must be divisible by ``num_stalks``.
    output_dim : int
        Output dimension.
    num_layers : int, optional
        Number of diffusion layers. Default is 4.
    num_stalks : int, optional
        Stalk dimension ``d``. Default is 3.
    map_type : str, optional
        Restriction-map family: ``'full'``, ``'diagonal'`` or
        ``'orthogonal'``. Default is ``'diagonal'``.
    layer_norm_mode : str or None, optional
        LayerNorm placement: ``'embedding'`` (over ``hidden_dim``),
        ``'stalk'`` (over ``stalk_dim``) or None. Default is ``'embedding'``.
    ortho_method : str, optional
        Orthogonalization method for the orthogonal map type. Default is
        ``'qr'`` (matches the paper / reference implementation).
    stalk_dropout : float, optional
        Per-stalk dropout inside each conv. Default is 0.0.
    input_dropout : float, optional
        Dropout applied to the input features before the first layer. Default
        is 0.0.
    directional : bool, optional
        Use the sheaf adjacency operator (``adj`` flag). Default is False.
    odd : bool, optional
        Use an odd (``tanh``) nonlinearity. Default is False.
    use_stalk_gate : bool, optional
        Enable the per-stalk gate. Default is False.

    Raises
    ------
    ValueError
        If ``hidden_dim`` is not divisible by ``num_stalks``, if ``map_type``
        is unknown, or if ``layer_norm_mode`` is invalid.

    References
    ----------
    [1] Bourgerie et al. "Deep Neural Sheaf Diffusion."
        https://arxiv.org/abs/2605.19021
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        num_layers=4,
        num_stalks=3,
        map_type="diagonal",
        layer_norm_mode="embedding",
        ortho_method="qr",
        stalk_dropout=0.0,
        input_dropout=0.0,
        directional=False,
        odd=False,
        use_stalk_gate=False,
    ):
        super().__init__()
        if hidden_dim % num_stalks != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by num_stalks "
                f"({num_stalks})."
            )
        if map_type not in _BUILDER_MAP:
            raise ValueError(
                f"Unknown map_type '{map_type}'. "
                f"Choose from {list(_BUILDER_MAP)}."
            )

        self.hidden_dim = hidden_dim
        self.num_stalks = num_stalks
        self.stalk_dim = hidden_dim // num_stalks
        self.num_layers = num_layers
        self.input_dropout = input_dropout

        self.lin_in = nn.Linear(input_dim, hidden_dim)

        builder_cls = _BUILDER_MAP[map_type]
        # 1/sqrt(2) init keeps the two composed maps from doubling the scale.
        builder_kwargs = {"init_scale": 1.0 / math.sqrt(2)}
        if map_type == "orthogonal":
            builder_kwargs["ortho_method"] = ortho_method

        self.source_builders = nn.ModuleList(
            builder_cls(hidden_dim, num_stalks, **builder_kwargs)
            for _ in range(num_layers)
        )
        self.target_builders = nn.ModuleList(
            builder_cls(hidden_dim, num_stalks, **builder_kwargs)
            for _ in range(num_layers)
        )

        self.convs = nn.ModuleList(
            DNSDConv(
                hidden_dim,
                num_stalks,
                stalk_dropout=stalk_dropout,
                directional=directional,
                odd=odd,
                use_stalk_gate=use_stalk_gate,
            )
            for _ in range(num_layers)
        )

        self.layer_norm_mode = layer_norm_mode
        if layer_norm_mode == "embedding":
            self.norms = nn.ModuleList(
                nn.LayerNorm(hidden_dim) for _ in range(num_layers)
            )
        elif layer_norm_mode == "stalk":
            self.norms = nn.ModuleList(
                nn.LayerNorm(self.stalk_dim) for _ in range(num_layers)
            )
        elif layer_norm_mode in (None, False):
            self.norms = None
        else:
            raise ValueError(
                "layer_norm_mode must be 'embedding', 'stalk', or None."
            )

        self.lin_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        """
        Forward pass of Deep Neural Sheaf Diffusion.

        Parameters
        ----------
        x : torch.Tensor
            Node feature matrix of shape [num_nodes, input_dim].
        edge_index : torch.Tensor
            Edge indices of shape [2, num_edges].

        Returns
        -------
        torch.Tensor
            Output node features of shape [num_nodes, output_dim].
        """
        n = x.size(0)
        x = F.dropout(x, p=self.input_dropout, training=self.training)
        x = self.lin_in(x)
        for i in range(self.num_layers):
            f_source = self.source_builders[i](x, edge_index)
            f_target = self.target_builders[i](x, edge_index)
            x = self.convs[i](x, edge_index, f_source, f_target)
            if self.norms is not None:
                if self.layer_norm_mode == "embedding":
                    x = self.norms[i](x)
                else:  # 'stalk'
                    x = self.norms[i](
                        x.view(n, self.num_stalks, self.stalk_dim)
                    ).view(n, -1)
        return self.lin_out(x)
