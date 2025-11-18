import torch
import torch.nn as nn
from torch_geometric.nn import FAConv

class FAGCNEncoder(nn.Module):
    """Frequency Adaptive Graph Convolutional Network (FAGCN) with identity activation function.

    FAGCN uses FAConv layers that adaptively control low and high frequency information
    in graph convolutions. This implementation stacks multiple FAConv layers without
    activation functions between them.

    Parameters
    ----------
    in_channels : int
        Number of input features.
    hidden_channels : int
        Number of hidden units.
    out_channels : int
        Number of output features.
    num_layers : int
        Number of layers.
    eps : float, optional
        Epsilon value for mixing initial features. Defaults to 0.1.
    dropout : float, optional
        Dropout rate for attention coefficients. Defaults to 0.0.
    cached : bool, optional
        Whether to cache normalized edge indices. Defaults to False.
    add_self_loops : bool, optional
        Whether to add self-loops to the graph. Defaults to True.
    normalize : bool, optional
        Whether to normalize the adjacency matrix. Defaults to True.
    """

    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers,
        eps=0.1,
        dropout=0.0,
        cached=False,
        add_self_loops=True,
        normalize=True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.eps = eps
        self.dropout = dropout

        # Input projection
        self.lin_in = nn.Linear(in_channels, hidden_channels)

        # FAConv layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                FAConv(
                    channels=hidden_channels,
                    eps=eps,
                    dropout=dropout,
                    cached=cached,
                    add_self_loops=add_self_loops,
                    normalize=normalize,
                )
            )

        # Output projection
        self.lin_out = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch=None, edge_weight=None):
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input node features.
        edge_index : torch.Tensor
            Edge indices.

        Returns
        -------
        torch.Tensor
            Output node features.
        """
        # Project input features
        x = self.lin_in(x)
        x_0 = x  # Store initial features for FAConv

        # Apply FAConv layers
        for conv in self.convs:
            x = conv(x, x_0, edge_index)

        # Project to output dimension
        x = self.lin_out(x)

        return x