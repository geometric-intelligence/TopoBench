"""
This module contains the implementation of identity GNNs.
"""

import torch
from torch_geometric.nn import GAT, GCN, GIN, GraphSAGE


class IdentityGAT(torch.nn.Module):
    """Graph Attention Network (GAT) with identity activation function.

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
    norm : torch.nn.Module
        Normalization layer.
    heads : int, optional
        Number of attention heads. Defaults to 1.
    dropout : float, optional
        Dropout rate. Defaults to 0.0.
    """

    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers,
        norm,
        heads=1,
        dropout=0.0,
    ):
        super().__init__()
        self.model = GAT(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            heads=heads,
            dropout=dropout,
            norm=norm,
            act=torch.nn.Identity(),
        )

        self.hidden_channels = hidden_channels
        self.out_channels = out_channels

    def forward(self, x, edge_index):
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
        return self.model(x, edge_index)


class IdentityGCN(torch.nn.Module):
    """Graph Convolutional Network (GCN) with identity activation function.

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
    norm : torch.nn.Module
        Normalization layer.
    dropout : float, optional
        Dropout rate. Defaults to 0.0.
    """

    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers,
        norm,
        dropout=0.0,
    ):
        super().__init__()
        self.model = GCN(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            dropout=dropout,
            norm=norm,
            act=torch.nn.Identity(),
        )
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels

    def forward(self, x, edge_index):
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
        return self.model(x, edge_index)


class IdentitySAGE(torch.nn.Module):
    """GraphSAGE with identity activation function.

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
    norm : torch.nn.Module
        Normalization layer.
    dropout : float, optional
        Dropout rate. Defaults to 0.0.
    """

    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers,
        norm,
        dropout=0.0,
    ):
        super().__init__()
        self.model = GraphSAGE(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            dropout=dropout,
            norm=norm,
            act=torch.nn.Identity(),
        )
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels

    def forward(self, x, edge_index):
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
        return self.model(x, edge_index)


class IdentityGIN(torch.nn.Module):
    """Graph Isomorphism Network (GIN) with identity activation function.

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
    norm : torch.nn.Module
        Normalization layer.
    dropout : float, optional
        Dropout rate. Defaults to 0.0.
    """

    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers,
        norm,
        dropout=0.0,
    ):
        super().__init__()
        self.model = GIN(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            dropout=dropout,
            norm=norm,
            act=torch.nn.Identity(),
        )
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels

    def forward(self, x, edge_index):
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
        return self.model(x, edge_index)
