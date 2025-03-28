"""Define the TopoTune class, which, given a choice of hyperparameters, instantiates a GCCN expecting a single Hasse graph as input."""

import copy

import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from topobench.data.utils import get_routes_from_neighborhoods


class TopoTune_OneHasse(torch.nn.Module):
    """Tunes a GNN model using higher-order relations.

    This class takes a GNN and its kwargs as inputs, and tunes it with specified additional relations.
    Unlike the case of TopoTune, this class expects a single Hasse graph as input, where all
    higher-order neighborhoods are represented as a single adjacency matrix.

    Parameters
    ----------
    GNN : torch.nn.Module, a class not an object
        The GNN class to use. ex: GAT, GCN.
    neighborhoods : list of lists
        The neighborhoods of interest.
    layers : int
        The number of layers to use. Each layer contains one GNN.
    use_edge_attr : bool
        Whether to use edge attributes.
    activation : str
        The activation function to use. ex: 'relu', 'tanh', 'sigmoid'.
    """

    def __init__(
        self,
        GNN,
        neighborhoods,
        layers,
        use_edge_attr,
        activation,
    ):
        super().__init__()
        self.routes = get_routes_from_neighborhoods(neighborhoods)
        self.neighborhoods = neighborhoods

        self.layers = layers
        self.use_edge_attr = use_edge_attr
        self.max_rank = 2
        self.graph_routes = torch.nn.ModuleList()
        self.GNN = [i for i in GNN.named_modules()]
        self.activation = activation

        # Instantiate GNN layers
        for _ in range(self.layers):
            self.graph_routes.append(copy.deepcopy(GNN))

        self.hidden_channels = GNN.hidden_channels
        self.out_channels = GNN.out_channels

    def all_nbhds_expand(self, params, membership):
        """Expand the complex into a single Hasse graph which contains all ranks and all nbhd.

        Parameters
        ----------
        params : dict
            The parameters of the batch, containing the complex.
        membership : dict
            The batch membership of the graphs per rank.

        Returns
        -------
        torch_geometric.data.Data
            The expanded Hasse graph.
        """

        device = params.x_0.device

        x = torch.cat(
            [getattr(params, f"x_{i}") for i in range(self.max_rank + 1)],
            dim=0,
        ).to(device)

        max_node_id = params.x_0.shape[0]
        max_edge_id = params.x_1.shape[0]

        edge_indices = []
        edge_attrs = []

        for route, neighborhood in zip(
            self.routes, self.neighborhoods, strict=False
        ):
            src_rank, dst_rank = route

            if (
                "up_laplacian" in neighborhood
                or "up_adjacency" in neighborhood
            ):
                if src_rank == 0:  # node-to-edge
                    adjustment = torch.tensor([[0], [0]]).to(device)
                elif src_rank == 1:  # edge-to-face
                    adjustment = torch.tensor(
                        [[max_node_id], [max_node_id]]
                    ).to(device)
                else:
                    raise ValueError(
                        f"Unsupported src_rank for 'up' neighborhood: {src_rank}"
                    )

                edge_indices.append(
                    getattr(params, neighborhood).indices().to(device)
                    + adjustment
                )
                edge_attrs.append(
                    getattr(params, neighborhood).values().squeeze()
                )

            elif (
                "down_laplacian" in neighborhood
                or "down_adjacency" in neighborhood
            ):
                if src_rank == 1:  # edge-to-node
                    adjustment = torch.tensor(
                        [[max_node_id], [max_node_id]]
                    ).to(device)
                elif src_rank == 2:  # face-to-edge
                    adjustment = torch.tensor(
                        [
                            [max_node_id + max_edge_id],
                            [max_node_id + max_edge_id],
                        ]
                    ).to(device)
                else:
                    raise ValueError(
                        f"Unsupported src_rank for 'down' neighborhood: {src_rank}"
                    )

                edge_indices.append(
                    getattr(params, neighborhood).indices().to(device)
                    + adjustment
                )
                edge_attrs.append(
                    getattr(params, neighborhood).values().squeeze()
                )

            elif "down_incidence" in neighborhood:
                if src_rank == 1:  # edge-to-face
                    adjustment = torch.tensor([[0], [max_node_id]]).to(device)
                elif src_rank == 2:  # face-to-edge
                    adjustment = torch.tensor(
                        [[max_node_id], [max_node_id + max_edge_id]]
                    ).to(device)
                else:
                    raise ValueError(
                        f"Unsupported src_rank for 'down_incidence' neighborhood: {src_rank}"
                    )

                edge_indices.append(
                    getattr(params, neighborhood)
                    .coalesce()
                    .indices()
                    .to(device)
                    + adjustment
                )
                edge_attrs.append(
                    getattr(params, neighborhood).values().squeeze()
                )

            elif "up_incidence" in neighborhood:
                if src_rank == 0:  # node-to-edge
                    adjustment = torch.tensor([[max_node_id], [0]]).to(device)
                elif src_rank == 1:  # edge-to-face
                    adjustment = torch.tensor(
                        [[max_node_id + max_edge_id], [max_node_id]]
                    ).to(device)
                else:
                    raise ValueError(
                        f"Unsupported src_rank for 'up_incidence' neighborhood: {src_rank}"
                    )
                coincidence_indices = (
                    getattr(params, neighborhood)
                    .T.coalesce()
                    .indices()
                    .to(device)
                    + adjustment
                )

                edge_indices.append(coincidence_indices)
                # edge_attrs.append(
                #     getattr(params, neighborhood)
                #     .T.coalesce()
                #     .values()
                #     .squeeze()
                # )

        edge_index = torch.cat(edge_indices, dim=1)
        # edge_attr = torch.cat(edge_attrs, dim=0)

        batch_expanded = torch.cat(
            [membership[0], membership[1], membership[2]], dim=0
        ).to(device)

        return Data(
            x=x,
            edge_index=edge_index,
            # edge_attr=edge_attr,
            batch=batch_expanded,
        )

    def all_nbhds_gnn_forward(
        self,
        batch_route,
        layer_idx,
    ):
        """Forward pass of the GNN (one layer) for an intrarank Hasse graph.

        Parameters
        ----------
        batch_route : torch_geometric.data.Data
            The batch of intrarank Hasse graphs for this route.
        layer_idx : int
            The index of the TopoTune layer.

        Returns
        -------
        torch.tensor
            The output of the GNN (updated features).
        """
        out = self.graph_routes[layer_idx](
            batch_route.x,
            batch_route.edge_index,
            #    batch_route.edge_weight, # TODO : some gnns take edge_weight (1d) and some take edge_attr.
            #    batch_route.edge_attr,
        )
        return out

    def aggregate_inter_nbhd(self, x_out):
        """Aggregate the outputs of the GNN for each rank.

        While the GNN takes care of intra-nbhd aggregation,
        this will take care of inter-nbhd aggregation.
        Default: sum.

        Parameters
        ----------
        x_out : torch.tensor
            The output of the GNN, concatenated features of each rank.

        Returns
        -------
        dict
            The aggregated outputs of the GNN for each rank.
        """
        x_out_per_rank = {}
        start_idx = 0
        for rank in range(self.max_rank + 1):
            rank_size = self.membership[rank].shape[0]
            end_idx = start_idx + rank_size
            if end_idx > x_out.shape[0]:
                raise IndexError(
                    f"End index {end_idx} out of bounds for x_out with shape {x_out.shape[0]}"
                )
            x_out_per_rank[rank] = x_out[start_idx:end_idx]
            start_idx = end_idx
        return x_out_per_rank

    def generate_membership_vectors(self, batch: Data):
        """Generate membership vectors based on batch.cell_statistics.

        Parameters
        ----------
        batch : torch_geometric.data.Data
            Batch object containing the batched domain data.

        Returns
        -------
        dict
            The batch membership of the graphs per rank.
        """
        max_dim = batch.cell_statistics.shape[1]
        cell_statistics = batch.cell_statistics
        membership = {
            j: torch.tensor(
                [
                    elem
                    for list in [
                        [i] * x for i, x in enumerate(cell_statistics[:, j])
                    ]
                    for elem in list
                ]
            )
            for j in range(max_dim)
        }
        return membership

    def forward(self, batch):
        """Forward pass of the model.

        Parameters
        ----------
        batch : Complex or ComplexBatch(Complex)
            The input data.

        Returns
        -------
        dict
            The output hidden states of the model per rank.
        """
        act = get_activation(self.activation)

        self.membership = self.generate_membership_vectors(batch)
        if batch.x_2.shape[0] == 0:
            x_out_per_rank = {}
            x_out_per_rank[0] = batch.x_0
            x_out_per_rank[1] = batch.x_1
            x_out_per_rank[2] = batch.x_2
            return x_out_per_rank

        for layer_idx in range(self.layers):
            batch_route = self.all_nbhds_expand(batch, self.membership)
            x_out = self.all_nbhds_gnn_forward(
                batch_route,
                layer_idx,
            )

            # aggregate across neighborhoods
            x_out_per_rank = self.aggregate_inter_nbhd(x_out)

            # update and replace the features for next layer
            for rank in x_out_per_rank:
                x_out_per_rank[rank] = act(x_out_per_rank[rank])
                setattr(batch, f"x_{rank}", x_out_per_rank[rank])

        for rank in range(self.max_rank + 1):
            if rank not in x_out_per_rank:
                x_out_per_rank[rank] = getattr(batch, f"x_{rank}")

        return x_out_per_rank


def get_activation(nonlinearity, return_module=False):
    """Activation resolver from CWN.

    Parameters
    ----------
    nonlinearity : str
        The nonlinearity to use.
    return_module : bool
        Whether to return the module or the function.

    Returns
    -------
    module or function
        The module or the function.
    """
    if nonlinearity == "relu":
        module = torch.nn.ReLU
        function = F.relu
    elif nonlinearity == "elu":
        module = torch.nn.ELU
        function = F.elu
    elif nonlinearity == "id":
        module = torch.nn.Identity

        def function(x):
            return x
    elif nonlinearity == "sigmoid":
        module = torch.nn.Sigmoid
        function = F.sigmoid
    elif nonlinearity == "tanh":
        module = torch.nn.Tanh
        function = torch.tanh
    else:
        raise NotImplementedError(
            f"Nonlinearity {nonlinearity} is not currently supported."
        )
    if return_module:
        return module
    return function
