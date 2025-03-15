"""Convolutional Cell Convolutional Network (MLP) model."""

import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    r"""MLP model.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    n_layers : int, optional
        Number of layers (default: 2).
    dropout : float, optional
        Dropout rate (default: 0).
    last_act : bool, optional
        If True, the last activation function is applied (default: False).
    """

    def __init__(self, in_channels, n_layers=2, dropout=0.0, last_act=False, **kwargs):
        super().__init__()
        self.d = dropout
        self.convs = nn.ModuleList()
        self.last_act = last_act
        for _ in range(n_layers):
            self.convs.append(nn.Linear(in_channels, in_channels))

    def forward(self, x_0):
        r"""Forward pass.

        Parameters
        ----------
        x_0 : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        for i, c in enumerate(self.convs):
            x_0 = c(F.dropout(x_0, p=self.d, training=self.training))
            if i == len(self.convs):
                break
            x_0 = x_0.relu()
        return x_0

