"""MLP-based Readout over 0-cells (i.e. nodes)."""

import torch.nn as nn
from torch_geometric.utils import scatter

from topobench.nn.backbones.non_relational.mlp import MLP as MLPBackbone


class MLPReadout(MLPBackbone):
    """MLP-based Readout over 0-cells (i.e. nodes).

    This class implements a readout layer for graph neural networks, allowing for customizable MLP layers and pooling strategies.

    Parameters
    ----------
    in_channels : int
        The dimensionality of the input features.
    hidden_layers : int
        The dimensionality of the hidden MLP layers.
    out_channels : int
        The dimensionality of the output features.
    pooling_type : str
        Pooling type for readout layer. Either "max", "sum" or "mean".
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
    task_level : str
        Task level for readout layer. Either "graph" or "node".
    **kwargs
        Additional keyword arguments.
    """

    def __init__(
        self,
        in_channels,
        hidden_layers,
        out_channels,
        pooling_type="sum",
        dropout=0.25,
        norm=None,
        norm_kwargs=None,
        act="relu",
        act_kwargs=None,
        final_act=None,
        final_act_kwargs=None,
        task_level=None,
        **kwargs,
    ):
        super().__init__(
            in_channels=in_channels,
            hidden_layers=hidden_layers,
            out_channels=out_channels,
            dropout=dropout,
            norm=norm,
            norm_kwargs=norm_kwargs,
            act=act,
            act_kwargs=act_kwargs,
            final_act=final_act,
            final_act_kwargs=final_act_kwargs,
            task_level=task_level,
            **kwargs,
        )
        self.pooling_type = pooling_type
        if self.task_level == "graph":
            self.mlp_layers = self.mlp_layers[
                :-2
            ]  # remove last linear and final activation
            mlp_final_dim = (
                self.hidden_layers[-1]
                if len(self.hidden_layers) > 0
                else self.in_channels
            )
            self.graph_readout_layer = nn.Linear(mlp_final_dim, out_channels)
            self.graph_readout_activation = self.final_act

    def forward(self, model_out, batch):
        """Readout logic based on model_output.

        Parameters
        ----------
        model_out : dict
            Dictionary containing the model output.
        batch : torch_geometric.data.Data
            Batch object containing the batched domain data.

        Returns
        -------
        dict
            Dictionary containing the updated model output.
        """
        model_out["x_0"] = self.mlp_layers(model_out["x_0"])

        if self.task_level == "graph":
            model_out["x_0"] = scatter(
                model_out["x_0"],
                batch["batch_0"],
                dim=0,
                reduce=self.pooling_type,
            )
            model_out["x_0"] = self.graph_readout_layer(model_out["x_0"])
            model_out["x_0"] = self.graph_readout_activation(model_out["x_0"])

        return model_out
    
    def __call__(self, model_out, batch):
        """Readout logic based on model_output.

        Parameters
        ----------
        model_out : dict
            Dictionary containing the model output.
        batch : torch_geometric.data.Data
            Batch object containing the batched domain data.

        Returns
        -------
        dict
            Dictionary containing the updated model output.
        """
        model_out = self.forward(model_out, batch)
        model_out["logits"] = model_out["x_0"]

        return model_out
