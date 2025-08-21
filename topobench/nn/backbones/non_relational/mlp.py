"""MLP implementation."""

import torch.nn as nn
from torch_geometric.nn.resolver import (
    activation_resolver,
    normalization_resolver,
)


class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP).

    This class implements a multi-layer perceptron architecture with customizable
    activation functions and normalization layers.

    Parameters
    ----------
    in_channels : int
        The dimensionality of the input features.
    hidden_channels : int
        The dimensionality of the hidden features.
    out_channels : int
        The dimensionality of the output features.
    dropout : float, optional
        The dropout rate (default 0.25).
    norm : str, optional
        The normalization layer to use (default None).
    norm_kwargs : dict, optional
        Additional keyword arguments for the normalization layer (default None).
    act : str, optional
        The activation function to use (default "relu").
    act_kwargs : dict, optional
        Additional keyword arguments for the activation function (default None).
    final_act : str, optional
        The final activation function to use (default "sigmoid").
    final_act_kwargs : dict, optional
        Additional keyword arguments for the final activation function (default None).
    num_nodes : int, optional
        The number of nodes in the input graph (default None).
    **kwargs
        Additional keyword arguments.
    """

    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        dropout=0.25,
        norm=None,
        norm_kwargs=None,
        act=None,
        act_kwargs=None,
        final_act=None,
        final_act_kwargs=None,
        num_nodes=None,
        **kwargs,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.dropout = dropout
        self.norm_layers = self.build_norm_layers(norm, norm_kwargs)
        self.act = (
            activation_resolver(act, **(act_kwargs or {}))
            if act is not None
            else nn.Identity()
        )
        self.final_act = (
            activation_resolver(final_act, **(final_act_kwargs or {}))
            if final_act is not None
            else nn.Identity()
        )
        self.out_channels = out_channels
        self.mlp_layers = self.build_mlp_layers()
        self.num_nodes = num_nodes

    def build_norm_layers(self, norm, norm_kwargs):
        """Build the normalization layers.

        Parameters
        ----------
        norm : str
            The normalization layer to use.
        norm_kwargs : dict
            Additional keyword arguments for the normalization layer.

        Returns
        -------
        list
            A list of normalization layers.
        """
        layers = []
        for hidden_dim in self.hidden_channels:
            if norm is not None:
                layers.append(
                    normalization_resolver(
                        norm,
                        hidden_dim,
                        **(norm_kwargs or {}),
                    )
                )
            else:
                layers.append(nn.Identity())
        return layers

    def build_mlp_layers(self):
        """Build the MLP layers.

        Returns
        -------
        nn.Sequential
            The MLP layers.
        """
        layers = []
        fc_layer_input_dim = self.in_channels
        for fc_dim, norm_layer in zip(
            self.hidden_channels, self.norm_layers, strict=False
        ):
            layers.append(
                nn.Sequential(
                    nn.Linear(fc_layer_input_dim, fc_dim),
                    norm_layer,
                    self.act,
                    nn.AlphaDropout(p=self.dropout, inplace=True),
                )
            )
            fc_layer_input_dim = fc_dim
        layers.append(nn.Linear(fc_dim, self.out_channels))
        layers.append(self.final_act)
        return nn.Sequential(*layers)

    def forward(self, x, batch_size):
        """Forward pass through the MLP.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        batch_size : int
            Batch size.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        flattened_x = x.view(batch_size, -1)
        x = self.mlp_layers(flattened_x)
        return (
            x.view(batch_size, self.num_nodes, -1)
            if batch_size > 1
            else x.view(self.num_nodes, -1)
        )

    def __call__(self, model_out) -> dict:
        """Backbone logic based on model_output.

        Parameters
        ----------
        model_out : dict
            Dictionary containing the encoder output.

        Returns
        -------
        dict
            Dictionary containing the updated model output.
        """
        model_out["x_0"] = self.forward(model_out["x_0"], model_out.batch_size)
        model_out["logits"] = model_out["x_0"]

        return model_out
