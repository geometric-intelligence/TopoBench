"""A transform that has the gated GConve layer model."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.graphgym.register as register
import torch_geometric.nn as pyg_nn
from torch_geometric.graphgym.models.layer import LayerConfig
from torch_geometric.graphgym.register import register_layer
from torch_geometric.nn.conv.res_gated_graph_conv import ResGatedGraphConv
from torch_scatter import scatter


class GatedGCNLayer(pyg_nn.conv.MessagePassing):
    """GatedGCN layer Residual Gated Graph ConvNets .https://arxiv.org/pdf/1711.07553.pdf.

    Parameters
    ----------
    in_dim : int
        Input dimension.
    out_dim : int
        Output dimension.
    dropout : float
        Dropout rate.
    residual : bool
        Whether to use residual connections.
    act : str
        Activation function to use. Default is "relu".
    equivstable_pe : bool
        Whether to use equivariant stable positional encoding. Default is False.
    **kwargs : dict
        Additional keyword arguments.
    """

    def __init__(
        self,
        in_dim,
        out_dim,
        dropout,
        residual,
        act="relu",
        equivstable_pe=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.activation = register.act_dict[act]
        self.A = pyg_nn.Linear(in_dim, out_dim, bias=True)
        self.B = pyg_nn.Linear(in_dim, out_dim, bias=True)
        self.C = pyg_nn.Linear(in_dim, out_dim, bias=True)
        self.D = pyg_nn.Linear(in_dim, out_dim, bias=True)
        self.E = pyg_nn.Linear(in_dim, out_dim, bias=True)

        # Handling for Equivariant and Stable PE using LapPE
        # ICLR 2022 https://openreview.net/pdf?id=e95i1IHcWj
        self.EquivStablePE = equivstable_pe
        if self.EquivStablePE:
            self.mlp_r_ij = nn.Sequential(
                nn.Linear(1, out_dim),
                self.activation(),
                nn.Linear(out_dim, 1),
                nn.Sigmoid(),
            )

        self.bn_node_x = nn.BatchNorm1d(out_dim)
        self.bn_edge_e = nn.BatchNorm1d(out_dim)
        self.act_fn_x = self.activation()
        self.act_fn_e = self.activation()
        self.dropout = dropout
        self.residual = residual
        self.e = None

    def forward(self, batch):
        """
        Forward pass for the Gated Graph Convolution layer.

        Parameters
        ----------
        batch : Batch
            A batch of data containing node features, edge attributes, and edge indices.

        Returns
        -------
        Tensor
            Updated node features after applying the Gated Graph Convolution.
        """
        x, e, edge_index = batch.x, batch.edge_attr, batch.edge_index

        """
        x               : [n_nodes, in_dim]
        e               : [n_edges, in_dim]
        edge_index      : [2, n_edges]
        """
        if self.residual:
            x_in = x
            e_in = e

        Ax = self.A(x)
        Bx = self.B(x)
        Ce = self.C(e)
        Dx = self.D(x)
        Ex = self.E(x)

        # Handling for Equivariant and Stable PE using LapPE
        # ICLR 2022 https://openreview.net/pdf?id=e95i1IHcWj
        pe_LapPE = batch.pe_EquivStableLapPE if self.EquivStablePE else None

        x, e = self.propagate(
            edge_index, Bx=Bx, Dx=Dx, Ex=Ex, Ce=Ce, e=e, Ax=Ax, PE=pe_LapPE
        )

        x = self.bn_node_x(x)
        e = self.bn_edge_e(e)

        x = self.act_fn_x(x)
        e = self.act_fn_e(e)

        x = F.dropout(x, self.dropout, training=self.training)
        e = F.dropout(e, self.dropout, training=self.training)

        if self.residual:
            x = x_in + x
            e = e_in + e

        batch.x = x
        batch.edge_attr = e

        return batch

    def message(self, Dx_i, Ex_j, PE_i, PE_j, Ce):
        """Perform message computation for each edge.

        Parameters
        ----------
        Dx_i : torch.Tensor
            Transformed node features for source nodes. Shape: [n_edges, out_dim].
        Ex_j : torch.Tensor
            Transformed node features for target nodes. Shape: [n_edges, out_dim].
        PE_i : torch.Tensor
            Positional encoding for source nodes. Shape: [n_edges, out_dim].
        PE_j : torch.Tensor
            Positional encoding for target nodes. Shape: [n_edges, out_dim].
        Ce : torch.Tensor
            Transformed edge features. Shape: [n_edges, out_dim].

        Returns
        -------
        Tensor
            Computed messages for each edge.
        """
        """
        {}x_i           : [n_edges, out_dim]
        {}x_j           : [n_edges, out_dim]
        {}e             : [n_edges, out_dim]
        """
        e_ij = Dx_i + Ex_j + Ce
        sigma_ij = torch.sigmoid(e_ij)

        # Handling for Equivariant and Stable PE using LapPE
        # ICLR 2022 https://openreview.net/pdf?id=e95i1IHcWj
        if self.EquivStablePE:
            r_ij = ((PE_i - PE_j) ** 2).sum(dim=-1, keepdim=True)
            r_ij = self.mlp_r_ij(
                r_ij
            )  # the MLP is 1 dim --> hidden_dim --> 1 dim
            sigma_ij = sigma_ij * r_ij

        self.e = e_ij
        return sigma_ij

    def aggregate(self, sigma_ij, index, Bx_j, Bx):
        """Aggregate messages for the Gated Graph Convolution layer.

        Parameters
        ----------
        sigma_ij : Tensor
            Output from the message() function. Shape: [n_edges, out_dim].
        index : Tensor
            Edge indices. Shape: [n_edges].
        Bx_j : Tensor
            Transformed node features for target nodes. Shape: [n_edges, out_dim].
        Bx : Tensor
            Transformed node features for all nodes. Shape: [n_nodes, out_dim].

        Returns
        -------
        Tensor
            Aggregated node features. Shape: [n_nodes, out_dim].
        """
        """
        sigma_ij        : [n_edges, out_dim]  ; is the output from message() function
        index           : [n_edges]
        {}x_j           : [n_edges, out_dim]
        """
        dim_size = Bx.shape[0]  # or None ??   <--- Double check this

        sum_sigma_x = sigma_ij * Bx_j
        numerator_eta_xj = scatter(
            sum_sigma_x, index, 0, None, dim_size, reduce="sum"
        )

        sum_sigma = sigma_ij
        denominator_eta_xj = scatter(
            sum_sigma, index, 0, None, dim_size, reduce="sum"
        )

        out = numerator_eta_xj / (denominator_eta_xj + 1e-6)
        return out

    def update(self, aggr_out, Ax):
        """Update node and edge features after aggregation.

        Parameters
        ----------
        aggr_out : Tensor
            Output from the aggregate() function after the aggregation. Shape: [n_nodes, out_dim].
        Ax : Tensor
            Transformed node features. Shape: [n_nodes, out_dim].

        Returns
        -------
        Tuple[Tensor, Tensor]
            Updated node features and edge features.
        """
        """
        aggr_out        : [n_nodes, out_dim] ; is the output from aggregate() function after the aggregation
        {}x             : [n_nodes, out_dim]
        """
        x = Ax + aggr_out
        e_out = self.e
        del self.e
        return x, e_out


@register_layer("gatedgcnconv")
class GatedGCNGraphGymLayer(nn.Module):
    """Initialize the GatedGCNGraphGymLayer https://arxiv.org/pdf/1711.07553.pdf.

    Parameters
    ----------
    layer_config : LayerConfig
        Configuration for the layer.
    **kwargs : dict
        Additional keyword arguments.
    """

    def __init__(self, layer_config: LayerConfig, **kwargs):
        super().__init__()
        self.model = GatedGCNLayer(
            in_dim=layer_config.dim_in,
            out_dim=layer_config.dim_out,
            dropout=0.0,  # Dropout is handled by GraphGym's `GeneralLayer` wrapper
            residual=False,  # Residual connections are handled by GraphGym's `GNNStackStage` wrapper
            act=layer_config.act,
            **kwargs,
        )

    def forward(self, batch):
        """Forward pass for the Gated Graph Convolution layer.

        Parameters
        ----------
        batch : Batch
            A batch of data containing node features, edge attributes, and edge indices.

        Returns
        -------
        Tensor
            Updated node features after applying the Gated Graph Convolution Layer.
        """
        return self.model(batch)


@register_layer("resgatedgcnconv")
class ResGatedGCNConvGraphGymLayer(nn.Module):
    """Initialize the ResGatedGCNConvGraphGymLayer.

    Parameters
    ----------
    layer_config : LayerConfig
        Configuration for the layer.
    **kwargs : dict
        Additional keyword arguments.
    """

    def __init__(self, layer_config: LayerConfig, **kwargs):
        super().__init__()
        self.model = ResGatedGraphConv(
            layer_config.dim_in,
            layer_config.dim_out,
            bias=layer_config.has_bias,
            **kwargs,
        )

    def forward(self, batch):
        """Forward pass for the Gated Graph Convolution layer.

        Parameters
        ----------
        batch : Batch
            A batch of data containing node features, edge attributes, and edge indices.

        Returns
        -------
        Tensor
            Updated node features after applying the Gated Graph Convolution Layer.
        """
        batch.x = self.model(batch.x, batch.edge_index)
        return batch


class ResGatedGCNConvLayer(nn.Module):
    """Initialize the ResGatedGCNConvLayer.

    Parameters
    ----------
    in_dim : int
        Input dimension.
    out_dim : int
        Output dimension.
    dropout : float
        Dropout rate.
    residual : bool
        Whether to use residual connections.
    act : str, optional
        Activation function to use. Default is "relu".
    **kwargs : dict
        Additional keyword arguments.
    """

    def __init__(
        self, in_dim, out_dim, dropout, residual, act="relu", **kwargs
    ):
        super().__init__()
        self.model = ResGatedGraphConv(
            in_dim,
            out_dim,
            dropout=dropout,
            act=register.act_dict[act](),
            residual=residual,
            **kwargs,
        )

    def forward(self, batch):
        """
        Forward pass for the Gated Graph Convolution layer.

        Parameters
        ----------
        batch : Batch
            A batch of data containing node features, edge attributes, and edge indices.

        Returns
        -------
        Tensor
            Updated node features after applying the Gated Graph Convolution Layer.
        """
        batch.x = self.model(batch.x, batch.edge_index)
        return batch
