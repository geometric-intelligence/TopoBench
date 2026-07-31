"""Graph MLP backbone from https://github.com/yanghu819/Graph-MLP/blob/master/models.py."""

import torch
import torch.nn as nn
from torch.nn import Dropout, LayerNorm, Linear


class GraphMLP(nn.Module):
    """Apply a feature-only MLP independently to every native graph node."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        dropout: float = 0.0,
        loss: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.out_channels = hidden_channels
        self.mlp = Mlp(in_channels, hidden_channels, dropout)
        self.loss = loss

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return node embeddings without allocating a global pair matrix."""
        return self.mlp(x)


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
