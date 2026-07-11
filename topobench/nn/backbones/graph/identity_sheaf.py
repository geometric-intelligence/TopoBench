"""Identity Sheaf Network (ISN) backbone for the training framework.

An Identity Sheaf Network is a Neural Sheaf Diffusion model in which the
learnable sheaf is replaced by a fixed Identity Sheaf: every restriction map
``F_{v<e}`` equals the identity, so the sheaf Laplacian reduces to the standard
graph Laplacian. ISN serves as a baseline that ablates sheaf learning -- if it
matches learnable Sheaf Neural Networks, the benefit of learning restriction
maps is called into question [1].

[1] Hernandez Caralt et al. "On the Necessity of Learnable Sheaf Laplacians."
GRaM Workshop, ICLR 2026. https://arxiv.org/abs/2603.05395
"""

from torch.nn import Module
from torch_geometric.utils import to_undirected

from topobench.nn.backbones.graph.nsd_utils.inductive_discrete_models import (
    InductiveDiscreteIdentitySheafDiffusion,
)


class ISNEncoder(Module):
    """
    Identity Sheaf Network encoder for the training framework.

    This encoder runs sheaf diffusion with fixed identity restriction maps.
    Because the restriction maps are not learned, the sheaf Laplacian equals
    the standard graph Laplacian; the model is therefore the sheaf-free
    baseline used to ablate learnable Sheaf Neural Networks [1].

    Parameters
    ----------
    input_dim : int
        Dimension of input node features.
    hidden_dim : int
        Dimension of hidden layers. Must be divisible by d.
    num_layers : int, optional
        Number of sheaf diffusion layers. Default is 2.
    d : int, optional
        Dimension of the stalk space. For the identity sheaf, d >= 1.
        Default is 2.
    dropout : float, optional
        Dropout rate for hidden layers. Default is 0.1.
    input_dropout : float, optional
        Dropout rate for input layer. Default is 0.1.
    add_hp : bool, optional
        Append a fixed high-pass channel (grows each stalk by one). Default is False.
    add_lp : bool, optional
        Append a fixed low-pass channel (grows each stalk by one). Default is False.
    normalised : bool, optional
        Use the augmented normalized sheaf Laplacian ((D+1)^-1/2 L (D+1)^-1/2). Default is False.
    deg_normalised : bool, optional
        Use degree normalization instead (mutually exclusive). Default is False.
    second_linear : bool, optional
        Apply an extra input projection before diffusion. Default is False.
    device : str, optional
        Device to run the model on ('cpu' or 'cuda'). Default is 'cpu'.
    **kwargs : dict
        Additional keyword arguments (not used).

    References
    ----------
    [1] Hernandez Caralt et al. "On the Necessity of Learnable Sheaf
        Laplacians." GRaM Workshop, ICLR 2026. https://arxiv.org/abs/2603.05395
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_layers=2,
        d=2,
        dropout=0.1,
        input_dropout=0.1,
        add_hp=False,
        add_lp=False,
        normalised=False,
        deg_normalised=False,
        second_linear=False,
        device="cpu",
        **kwargs,
    ):
        super().__init__()

        assert d >= 1

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.d = d
        self.num_layers = num_layers
        self.device = device

        self.sheaf_config = {
            "d": d,
            "layers": num_layers,
            "hidden_channels": hidden_dim // d,
            "input_dim": input_dim,
            "output_dim": hidden_dim,
            "device": device,
            "input_dropout": input_dropout,
            "dropout": dropout,
            # ISN-paper ablation knobs: fixed high-/low-pass channels,
            # Laplacian normalization, and the optional extra input projection.
            "add_hp": add_hp,
            "add_lp": add_lp,
            "normalised": normalised,
            "deg_normalised": deg_normalised,
            "second_linear": second_linear,
            # Identity sheaf does not learn maps; these are unused but kept
            # for compatibility with the diagonal diffusion config.
            "sheaf_act": "id",
            "orth": "cayley",
        }

        # Create the identity sheaf model (diagonal diffusion with fixed maps).
        self.sheaf_model = InductiveDiscreteIdentitySheafDiffusion(
            self.sheaf_config
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
        Forward pass of the Identity Sheaf Network encoder.

        Parameters
        ----------
        x : torch.Tensor
            Node feature matrix of shape [num_nodes, input_dim].
        edge_index : torch.Tensor
            Edge indices of shape [2, num_edges]. Will be automatically
            converted to undirected.
        edge_attr : torch.Tensor, optional
            Edge feature matrix (not used). Default is None.
        edge_weight : torch.Tensor, optional
            Edge weights (not used). Default is None.
        batch : torch.Tensor, optional
            Batch vector assigning each node to a specific graph (not used).
            Default is None.
        **kwargs : dict
            Additional arguments (not used).

        Returns
        -------
        torch.Tensor
            Output node feature matrix of shape [num_nodes, hidden_dim].
        """
        # Sheaf diffusion requires undirected graphs (bidirectional edges).
        edge_index = to_undirected(edge_index)

        return self.sheaf_model(x, edge_index)

    def get_sheaf_model(self):
        """
        Get the underlying identity sheaf diffusion model.

        Returns
        -------
        InductiveDiscreteIdentitySheafDiffusion
            The identity sheaf diffusion model instance.
        """
        return self.sheaf_model
