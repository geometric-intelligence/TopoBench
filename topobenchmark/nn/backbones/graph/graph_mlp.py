"""Graph MLP backbone from https://github.com/yanghu819/Graph-MLP/blob/master/models.py."""

import torch
import torch.nn as nn
from torch.nn import Dropout, LayerNorm, Linear


class GraphMLP(nn.Module):
    """ "Graph MLP backbone.

    Parameters
    ----------
    in_channels : int
        Number of input features.
    hidden_channels : int
        Number of hidden units.
    order : int, optional
        To compute order-th power of adj matrix (default: 1).
    dropout : float, optional
        Dropout rate (default: 0.0).
    **kwargs
        Additional arguments.
    """

    def __init__(
        self, in_channels, hidden_channels, order=1, dropout=0.0, **kwargs
    ):
        super().__init__()
        self.out_channels = hidden_channels
        self.order = order
        self.mlp = Mlp(in_channels, self.out_channels, dropout)

    def forward(self, x):
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        x = self.mlp(x)
        Z = x

        x_dis = get_feature_dis(Z) if self.training else None

        return x, x_dis


class Mlp(nn.Module):
    """MLP module.

    Parameters
    ----------
    input_dim : int
        Input dimension.
    hid_dim : int
        Hidden dimension.
    dropout : float
        Dropout rate.
    """

    def __init__(self, input_dim, hid_dim, dropout):
        super().__init__()
        self.fc1 = Linear(input_dim, hid_dim)
        self.fc2 = Linear(hid_dim, hid_dim)
        self.act_fn = torch.nn.functional.gelu
        self._init_weights()

        self.dropout = Dropout(dropout)
        self.layernorm = LayerNorm(hid_dim, eps=1e-6)

    def _init_weights(self):
        """Initialize weights."""
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.layernorm(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def get_feature_dis(x):
    """Get feature distance matrix.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.

    Returns
    -------
    torch.Tensor
        Feature distance matrix.
    """
    x_dis = x @ x.T
    mask = torch.eye(x_dis.shape[0]).to(x_dis.device)
    x_sum = torch.sum(x**2, 1).reshape(-1, 1)
    x_sum = torch.sqrt(x_sum).reshape(-1, 1)
    x_sum = x_sum @ x_sum.T
    x_dis = x_dis * (x_sum ** (-1))
    x_dis = (1 - mask) * x_dis
    return x_dis
