"""Neural Sheaf Propagation (NSP) backbone for the training framework.

NSP is the **wave** counterpart of Neural Sheaf Diffusion: instead of the heat
equation (which dissipates energy and oversmooths), it discretises the sheaf
**wave** equation ``X''(t) = -Delta_F X(t)`` with the leapfrog method, conserving
energy and keeping representations non-smooth -- useful on heterophilic graphs.

This is a corrected implementation: the leapfrog force term is scaled by a
``step_size`` coefficient (the stability / CFL control the wave update requires).

[1] Suk et al. "Surfing on the Neural Sheaf." NeurIPS 2022 Workshop on Symmetry
and Geometry in Neural Representations. OpenReview:xOXFkyRzTlu.
"""

from torch.nn import Module
from torch_geometric.utils import to_undirected

from topobench.nn.backbones.graph.nsd_utils.inductive_discrete_models import (
    InductiveDiscreteBundleSheafPropagation,
    InductiveDiscreteDiagSheafPropagation,
    InductiveDiscreteGeneralSheafPropagation,
)


class NSPEncoder(Module):
    """
    Neural Sheaf Propagation encoder (diagonal sheaf, wave dynamics).

    Runs sheaf wave propagation with diagonal restriction maps, discretised with
    a stabilised leapfrog scheme. The ``step_size`` coefficient controls leapfrog
    stability (without it the wave update diverges on dense graphs).

    Parameters
    ----------
    input_dim : int
        Dimension of input node features.
    hidden_dim : int
        Dimension of hidden layers. If not divisible by d, the effective inner
        width is floored to ``(hidden_dim // d) * d`` while the output stays
        ``hidden_dim``.
    num_layers : int, optional
        Number of propagation layers. Default is 2.
    d : int, optional
        Stalk dimension (d >= 1). Default is 2.
    sheaf_type : str, optional
        Restriction-map family: 'diag' (diagonal), 'bundle' (orthogonal O(d))
        or 'general' (full d x d). 'bundle' and 'general' require d > 1.
        Default is 'diag'.
    step_size : float, optional
        Leapfrog step ``h``; the force term is scaled by ``h**2`` (stability
        control). Default is 0.5.
    second_linear : bool, optional
        If True, apply an extra input projection after the first encoder linear
        and before propagation begins. Default is False.
    new_laplacian_each_step : bool, optional
        If True, recompute the sheaf Laplacian from the current features at every
        layer (dynamic geometry); if False, build it once and reuse it (fixed
        geometry). Default is True.
    dropout : float, optional
        Dropout rate for hidden layers. Default is 0.1.
    input_dropout : float, optional
        Dropout rate for the input layer. Default is 0.1.
    sheaf_act : str, optional
        Activation for the sheaf learner. Default is 'tanh'.
    device : str, optional
        Device to run on. Default is 'cpu'.
    **kwargs : dict
        Additional keyword arguments (not used).

    References
    ----------
    [1] Suk et al. "Surfing on the Neural Sheaf." NeurIPS 2022 Workshop on
        Symmetry and Geometry in Neural Representations. OpenReview:xOXFkyRzTlu.
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_layers=2,
        d=2,
        sheaf_type="diag",
        step_size=0.5,
        second_linear=False,
        new_laplacian_each_step=True,
        dropout=0.1,
        input_dropout=0.1,
        sheaf_act="tanh",
        device="cpu",
        **kwargs,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.d = d
        self.sheaf_type = sheaf_type
        self.num_layers = num_layers
        self.step_size = step_size
        self.second_linear = second_linear
        self.new_laplacian_each_step = new_laplacian_each_step
        self.device = device

        # Select the restriction-map variant and validate d BEFORE building the
        # config (which divides by d). Bundle (orthogonal) and general (full)
        # maps require a stalk dimension d > 1.
        if sheaf_type == "diag":
            assert d >= 1
            propagation_class = InductiveDiscreteDiagSheafPropagation
        elif sheaf_type == "bundle":
            assert d > 1
            propagation_class = InductiveDiscreteBundleSheafPropagation
        elif sheaf_type == "general":
            assert d > 1
            propagation_class = InductiveDiscreteGeneralSheafPropagation
        else:
            raise ValueError(f"Unknown sheaf_type: {sheaf_type}")

        self.sheaf_config = {
            "d": d,
            "layers": num_layers,
            "hidden_channels": hidden_dim // d,
            "input_dim": input_dim,
            "output_dim": hidden_dim,
            "device": device,
            "input_dropout": input_dropout,
            "dropout": dropout,
            "sheaf_act": sheaf_act,
            "orth": "cayley",
            "step_size": step_size,
            "second_linear": second_linear,
            "new_laplacian_each_step": new_laplacian_each_step,
        }

        self.sheaf_propagation_model = propagation_class(self.sheaf_config)

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
        Forward pass of the Neural Sheaf Propagation encoder.

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
            Batch vector (not used). Default is None.
        **kwargs : dict
            Additional arguments (not used).

        Returns
        -------
        torch.Tensor
            Output node feature matrix of shape [num_nodes, hidden_dim].
        """
        edge_index = to_undirected(edge_index)
        return self.sheaf_propagation_model(x, edge_index)

    def get_sheaf_propagation_model(self):
        """
        Get the underlying sheaf propagation model.

        Returns
        -------
        InductiveDiscreteDiagSheafPropagation
            The sheaf wave-propagation model instance.
        """
        return self.sheaf_propagation_model
