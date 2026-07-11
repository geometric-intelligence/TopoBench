"""Deep Neural Sheaf Diffusion (DNSD) encoder for the training framework.

Deep Neural Sheaf Diffusion [1] tackles the depth limitation of Neural Sheaf
Diffusion: as layers stack, the sheaf-Laplacian disagreement signal vanishes
and deeper layers stop contributing. DNSD replaces the sheaf Laplacian with a
sheaf *adjacency* operator and adds normalization, an odd (``tanh``)
nonlinearity, and per-stalk gating, letting the model exploit deep aggregation.

This encoder is a thin wrapper that adapts the ``MessagePassing`` DNSD core in
:mod:`~topobench.nn.backbones.graph.nsd_utils.dnsd_models` to the framework's
``forward(x, edge_index) -> [num_nodes, hidden_dim]`` backbone contract.

[1] Bourgerie, Girdzijauskas and Fodor. "Deep Neural Sheaf Diffusion."
    ICML 2026 Workshop on Graph Foundation Models. https://arxiv.org/abs/2605.19021
"""

from torch.nn import Module
from torch_geometric.utils import to_undirected

from topobench.nn.backbones.graph.nsd_utils.dnsd_models import (
    DeepNeuralSheafDiffusion,
)


class DNSDEncoder(Module):
    """
    Deep Neural Sheaf Diffusion encoder.

    Learns per-layer source and target restriction maps and diffuses node
    features with the DNSD sheaf-adjacency operator. Produces node embeddings
    of dimension ``hidden_dim`` for a downstream readout.

    Parameters
    ----------
    input_dim : int
        Dimension of input node features.
    hidden_dim : int
        Dimension of hidden layers and of the output embedding. Must be
        divisible by ``d``.
    num_layers : int, optional
        Number of sheaf diffusion layers. DNSD is designed to stay effective
        when this is large. Default is 4.
    d : int, optional
        Stalk dimension. For ``map_type='diagonal'``, ``d >= 1``; for
        ``'full'``/``'orthogonal'``, ``d > 1`` is meaningful. Default is 2.
    map_type : str, optional
        Restriction-map family: ``'diagonal'``, ``'full'`` or
        ``'orthogonal'``. Default is ``'diagonal'``.
    directional : bool, optional
        Use the sheaf adjacency operator instead of the Laplacian (the ``adj``
        flag in the paper, and the core DNSD ingredient). Default is True.
    odd : bool, optional
        Use an odd (``tanh``) nonlinearity instead of ReLU. Default is True.
    use_stalk_gate : bool, optional
        Enable the per-stalk sigmoid gate on diffusion updates. Default is
        False.
    layer_norm_mode : str or None, optional
        LayerNorm placement: ``'embedding'``, ``'stalk'`` or None. Default is
        ``'embedding'``.
    ortho_method : str, optional
        Orthogonalization method used when ``map_type='orthogonal'``. One of
        ``'cayley'``, ``'matrix_exp'``, ``'householder'``, ``'qr'``. Default is
        ``'qr'`` (matches the paper / reference implementation).
    dropout : float, optional
        Per-stalk dropout applied inside each diffusion layer. Default is 0.0.
    input_dropout : float, optional
        Dropout applied to input features before the first layer. Default is
        0.0.
    **kwargs : dict
        Additional keyword arguments (ignored).

    References
    ----------
    [1] Bourgerie et al. "Deep Neural Sheaf Diffusion."
        https://arxiv.org/abs/2605.19021
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_layers=4,
        d=2,
        map_type="diagonal",
        directional=True,
        odd=True,
        use_stalk_gate=False,
        layer_norm_mode="embedding",
        ortho_method="qr",
        dropout=0.0,
        input_dropout=0.0,
        **kwargs,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.d = d
        self.map_type = map_type

        # output_dim == hidden_dim: this backbone emits node embeddings and the
        # framework readout handles the task-specific projection.
        self.dnsd = DeepNeuralSheafDiffusion(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_layers=num_layers,
            num_stalks=d,
            map_type=map_type,
            layer_norm_mode=layer_norm_mode,
            ortho_method=ortho_method,
            stalk_dropout=dropout,
            input_dropout=input_dropout,
            directional=directional,
            odd=odd,
            use_stalk_gate=use_stalk_gate,
        )

    def forward(
        self,
        x,
        edge_index,
        edge_attr=None,
        edge_weight=None,
        batch=None,
        **kwargs,
    ):
        """
        Forward pass of the Deep Neural Sheaf Diffusion encoder.

        Parameters
        ----------
        x : torch.Tensor
            Node feature matrix of shape [num_nodes, input_dim].
        edge_index : torch.Tensor
            Edge indices of shape [2, num_edges]. Converted to undirected.
        edge_attr : torch.Tensor, optional
            Edge features (not used). Default is None.
        edge_weight : torch.Tensor, optional
            Edge weights (not used). Default is None.
        batch : torch.Tensor, optional
            Batch assignment vector (not used). Default is None.
        **kwargs : dict
            Additional arguments (not used).

        Returns
        -------
        torch.Tensor
            Output node feature matrix of shape [num_nodes, hidden_dim].
        """
        # Sheaf diffusion assumes a symmetric edge set (both directions
        # present); make the graph undirected before propagating.
        edge_index = to_undirected(edge_index)
        return self.dnsd(x, edge_index)
