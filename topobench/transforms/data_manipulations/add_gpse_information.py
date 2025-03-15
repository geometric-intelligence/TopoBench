"""A transform that adds positional information to the graph."""

import os

import torch
import torch_geometric
import torch_geometric.data
from torch_geometric.data import Data
from torch_geometric.graphgym.config import cfg, load_cfg, set_cfg
from torch_geometric.graphgym.model_builder import create_model

from topobench.data.utils import get_routes_from_neighborhoods


class dotdict(dict):
    """Dot.notation access to dictionary attributes."""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class AddGPSEInformation(torch_geometric.transforms.BaseTransform):
    r"""A transform that uses a pre-trained GPSE to add positional and strutural information to the graph.

    Parameters
    ----------
    **kwargs : optional
        Parameters for the transform.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.type = "add_gpse_information"
        self.parameters = kwargs

        self.max_rank = kwargs["max_rank"]
        self.copy_initial = kwargs["copy_initial"]
        self.neighborhoods = kwargs["neighborhoods"]
        # TODO Add the posibility of having differet hops with shifted information ?
        # self.hop_num = kwargs["hop_num"]
        # self.shift = kwargs["shift"]

        self.device = (
            "cpu" if kwargs["device"] == "cpu" else f"cuda:{kwargs['cuda'][0]}"
        )
        self.init_config()
        self.model = create_model(
            dim_in=cfg.dim_in, dim_out=self.parameters["dim_out"]
        )
        model_state_dict = torch.load(
            f"{os.getcwd()}/data/pretrained_models/gpse_{self.parameters['pretrain_model'].lower()}.pt",
            map_location=torch.device(self.device),
        )
        self.hidden_dim = (
            self.parameters["dim_target_node"]
            + self.parameters["dim_target_graph"]
        )

        # remove_keys = [s for s in model_state_dict["model_state"] if s.startswith("model.post_mp")]

        # model.post_mp.node_post_mps.1.model.0.Layer_0.layer.model.weight
        # model_state_dict['model_state'] = {k: v for k, v in model_state_dict['model_state'].items() if k not in remove_keys}

        self.model.load_state_dict(model_state_dict["model_state"])

    def init_config(self):
        """Initialize GraphGym configuration.

        Returns
        -------
        None
            Nothing to return.
        """
        set_cfg(cfg)
        cfg.set_new_allowed(True)

        # TODO Fix this configuration parameters
        params = dotdict(
            {
                "cfg_file": f"configs/extras/gpse_{self.parameters['pretrain_model'].lower()}.yaml",
                "opts": [],
            }
        )
        load_cfg(cfg, params)
        cfg.share.num_node_targets = self.parameters["dim_target_node"]
        cfg.share.num_graph_targets = self.parameters["dim_target_graph"]
        cfg.accelerator = (
            self.device
        )  # "cuda:0" if torch.cuda.is_available() else "cpu"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(type={self.type!r}, parameters={self.parameters!r})"

    def intrarank_expand(self, params, src_rank, nbhd):
        """Expand the complex into an intrarank Hasse graph.

        Parameters
        ----------
        params : dict
            The parameters of the batch, containting the complex.
        src_rank : int
            The source rank.
        nbhd : str
            The neighborhood to use.

        Returns
        -------
        torch_geometric.data.Data
            The expanded batch of intrarank Hasse graphs for this route.
        """

        setattr(params, nbhd, getattr(params, nbhd).coalesce())

        batch_route = Data(
            x=getattr(params, f"x_{src_rank}"),
            edge_index=getattr(params, nbhd).indices(),
            edge_weight=getattr(params, nbhd).values().squeeze(),
            edge_attr=getattr(params, nbhd).values().squeeze(),
            requires_grad=True,
        )

        return batch_route

    def interrank_expand(self, params, src_rank, dst_rank, nbhd_cache):
        """Expand the complex into an interrank Hasse graph.

        Parameters
        ----------
        params : dict
            The parameters of the batch, containting the complex.
        src_rank : int
            The source rank.
        dst_rank : int
            The destination rank.
        nbhd_cache : dict
            The neighborhood cache containing the expanded boundary index and edge attributes.

        Returns
        -------
        torch_geometric.data.Data
            The expanded batch of interrank Hasse graphs for this route.
        """
        src_batch = params[f"x_{src_rank}"]
        dst_batch = params[f"x_{dst_rank}"]
        edge_index, edge_attr = nbhd_cache
        # setattr(
        #     params,
        #     f"x_{src_rank}",
        #     getattr(params, f"x_{src_rank}"),
        # )
        feat_on_dst = torch.zeros_like(
            getattr(params, f"x_{dst_rank}"), device=self.device
        )
        x_in = torch.vstack(
            [feat_on_dst, getattr(params, f"x_{src_rank}").to(self.device)]
        )
        batch_expanded = torch.cat([dst_batch, src_batch], dim=0)

        batch_route = Data(
            x=x_in.to(self.device),
            edge_index=edge_index.to(self.device),
            edge_attr=edge_attr.to(self.device),
            edge_weight=edge_attr.to(self.device),
            batch=batch_expanded.to(self.device),
        )

        return batch_route

    def aggregate_inter_nbhd(self, x_out_per_route):
        """Aggregate the outputs of the GNN for each rank.

        While the GNN takes care of intra-nbhd aggregation,
        this will take care of inter-nbhd aggregation.
        Default: sum.

        Parameters
        ----------
        x_out_per_route : dict
            The outputs of the GNN for each route.

        Returns
        -------
        dict
            The aggregated outputs of the GNN for each rank.
        """
        x_out_per_rank = {}
        for route_index, (_, dst_rank) in enumerate(self.routes):
            if dst_rank not in x_out_per_rank:
                # # This happens when there are not destination cells for a route
                # # In this case, we do not need to aggregate
                # if route_index not in x_out_per_route:
                #     continue
                x_out_per_rank[dst_rank] = x_out_per_route[route_index]
            else:
                x_out_per_rank[dst_rank] = torch.cat(
                    [x_out_per_rank[dst_rank], x_out_per_route[route_index]],
                    dim=1,
                )
        return x_out_per_rank

    def interrank_boundary_index(x_src, boundary_index, n_dst_nodes):
        """
        Recover lifted graph.

        Edge-to-node boundary relationships of a graph with n_nodes and n_edges
        can be represented as up-adjacency node relations. There are n_nodes+n_edges nodes in this lifted graph.
        Desgiend to work for regular (edge-to-node and face-to-edge) boundary relationships.

        Parameters
        ----------
        x_src : torch.tensor
            Source node features. Shape [n_src_nodes, n_features]. Should represent edge or face features.
        boundary_index : list of lists or list of tensors
            List boundary_index[0] stores node ids in the boundary of edge stored in boundary_index[1].
            List boundary_index[1] stores list of edges.
        n_dst_nodes : int
            Number of destination nodes.

        Returns
        -------
        edge_index : list of lists
            The edge_index[0][i] and edge_index[1][i] are the two nodes of edge i.
        edge_attr : tensor
            Edge features are given by feature of bounding node represnting an edge. Shape [n_edges, n_features].
        """
        node_ids = (
            boundary_index[0]
            if torch.is_tensor(boundary_index[0])
            else torch.tensor(boundary_index[0], dtype=torch.int32)
        )
        edge_ids = (
            boundary_index[1]
            if torch.is_tensor(boundary_index[1])
            else torch.tensor(boundary_index[1], dtype=torch.int32)
        )

        max_node_id = n_dst_nodes
        adjusted_edge_ids = edge_ids + max_node_id

        edge_index = torch.zeros(
            (2, node_ids.numel()), dtype=node_ids.dtype, device=x_src.device
        )
        edge_index[0, :] = node_ids
        edge_index[1, :] = adjusted_edge_ids

        edge_attr = x_src[edge_ids].squeeze()

        return edge_index, edge_attr

    def get_nbhd_cache(self, params):
        """Cache the nbhd information into a dict for the complex at hand.

        Parameters
        ----------
        params : dict
            The parameters of the batch, containing the complex.

        Returns
        -------
        dict
            The neighborhood cache.
        """
        nbhd_cache = {}
        for neighborhood, route in zip(
            self.neighborhoods, self.routes, strict=False
        ):
            src_rank, dst_rank = route
            if src_rank != dst_rank and (src_rank, dst_rank) not in nbhd_cache:
                n_dst_nodes = getattr(params, f"x_{dst_rank}").shape[0]

                # There is no neighbourhood in question
                if n_dst_nodes == 0:
                    nbhd_cache[(src_rank, dst_rank)] = None
                elif src_rank > dst_rank:
                    boundary = getattr(params, neighborhood).coalesce()
                    nbhd_cache[(src_rank, dst_rank)] = (
                        interrank_boundary_index(
                            getattr(params, f"x_{src_rank}"),
                            boundary.indices(),
                            n_dst_nodes,
                        )
                    )
                elif src_rank < dst_rank:
                    coboundary = getattr(params, neighborhood).coalesce()
                    # print(f'neighborhood: {neighborhood}')
                    # print(f'src_rank: {src_rank}')
                    # print(f'dst_rank: {dst_rank}')
                    # x_src = getattr(params, f"x_{src_rank}")
                    # x_dst = getattr(params, f"x_{dst_rank}")
                    # print(f'x_src: {x_src.shape}')
                    # print(f'x_dst: {x_dst.shape}')
                    # print(f'neighborhood: {coboundary.shape}')
                    # print(coboundary.to_dense().max())
                    nbhd_cache[(src_rank, dst_rank)] = (
                        interrank_boundary_index(
                            getattr(params, f"x_{src_rank}"),
                            coboundary.indices(),
                            n_dst_nodes,
                        )
                    )
        return nbhd_cache

    def forward_intrarank(
        self, src_rank, route_index, data: torch_geometric.data.Data
    ):
        """Forward for cells where src_rank==dst_rank.

        Parameters
        ----------
        src_rank : int
            Source rank of the transmitting cell.
        route_index : int
            The index of this particular message passing route.
        data : torch_geometric.data.Data
            The input data.

        Returns
        -------
        data
            The data object with messages passed.
        """
        nbhd = self.neighborhoods[route_index]
        batch_route = self.intrarank_expand(data, src_rank, nbhd)

        input_nodes = torch.normal(
            0,
            1,
            size=(batch_route.x.shape[0], cfg.dim_in),
            device=self.device,
        )
        input_graph = torch_geometric.data.Data(
            x=input_nodes,
            edge_index=batch_route.edge_index,
            y=torch.ones(batch_route.x.shape[0], 51, device=self.device),
            y_graph=torch.ones(
                1, self.parameters["dim_out"], device=self.device
            ),
            batch=torch.zeros(
                batch_route.x.shape[0],
                dtype=torch.int64,
                device=self.device,
            ),
        ).to(self.device)
        self.model.eval()
        with torch.inference_mode():
            x_out, _ = self.model(input_graph)
        return x_out

    def forward_interank(
        self, src_rank, dst_rank, nbhd_cache, data: torch_geometric.data.Data
    ):
        """Forward for cells where src_rank!=dst_rank.

        Parameters
        ----------
        src_rank : int
            Source rank of the transmitting cell.
        dst_rank : int
            Destinatino rank of the transmitting cell.
        nbhd_cache : dict
            Cache of the neighbourhood information.
        data : toch_geometric.data.Data
            The input data.

        Returns
        -------
        data
            The data object with messages passed.
        """
        # This has the boudary index
        nbhd = nbhd_cache[(src_rank, dst_rank)]

        # The actual data to pass to the GNN
        batch_route = self.interrank_expand(data, src_rank, dst_rank, nbhd)
        # The number of destination cells
        n_dst_cells = data[f"x_{dst_rank}"].shape[0]

        input_nodes = torch.normal(
            0, 1, size=(batch_route.x.shape[0], cfg.dim_in), device=self.device
        )

        # Express everything in terms of an input graph
        input_graph = torch_geometric.data.Data(
            x=input_nodes,
            edge_index=batch_route.edge_index,
            y=torch.ones(batch_route.x.shape[0], 51, device=self.device),
            y_graph=torch.ones(
                1, self.parameters["dim_out"], device=self.device
            ),
            batch=torch.zeros(
                batch_route.x.shape[0], dtype=torch.int64, device=self.device
            ),
        ).to(self.device)
        self.model.eval()
        with torch.inference_mode():
            expanded_out, _ = self.model.forward(input_graph)

        # Only grab the cells we are interested in
        x_out = expanded_out[:n_dst_cells]

        return x_out

    def forward(self, data: torch_geometric.data.Data):
        r"""Apply the transform to the input data.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data.

        Returns
        -------
        torch_geometric.data.Data
            The transformed data.
        """

        # Copy initial values as first hop
        if self.copy_initial:
            for i in range(self.max_rank + 1):
                x_i = getattr(data, f"x_{i}").float().to(self.device)
                setattr(data, f"x{i}_0", x_i)
        self.routes = get_routes_from_neighborhoods(self.neighborhoods)
        nbhd_cache = self.get_nbhd_cache(data)

        x_out_per_route = {}
        # Interate over the routes (i, [0, 1])
        for route_index, route in enumerate(self.routes):
            src_rank, dst_rank = route

            if src_rank == dst_rank:
                # If there are no nodes in this rank, we skip
                if getattr(data, f"x_{src_rank}").shape[0] == 0:
                    x_out_per_route[route_index] = torch.zeros(
                        (0, self.hidden_dim),
                        dtype=torch.float,
                        device=self.device,
                    )
                    continue
                # We cannot use GPSE embeddings on single-node graphs
                if getattr(data, f"x_{src_rank}").shape[0] == 1:
                    x_out_per_route[route_index] = torch.zeros(
                        (1, self.hidden_dim),
                        dtype=torch.float,
                        device=self.device,
                    )
                    continue
                x_out = self.forward_intrarank(src_rank, route_index, data)
                x_out_per_route[route_index] = x_out

            elif src_rank != dst_rank:
                # If there is no neighborhood, we skip
                if nbhd_cache[(src_rank, dst_rank)] is None:
                    x_out_per_route[route_index] = torch.zeros(
                        (0, self.hidden_dim),
                        dtype=torch.float,
                        device=self.device,
                    )
                    continue
                x_out = self.forward_interank(
                    src_rank, dst_rank, nbhd_cache, data
                )
                # Outputs of this particular route
                x_out_per_route[route_index] = x_out

        # aggregate across neighborhoods
        x_out_per_rank = self.aggregate_inter_nbhd(x_out_per_route)

        # If no information was passed to a rank, then we initialize an empty vector
        # with the output dimension of the pre-trained model
        # and set the features as 0
        for rank in range(self.max_rank + 1):
            if rank not in x_out_per_rank:
                x_out_per_rank[rank] = torch.zeros(
                    (data[f"x_{rank}"].shape[0], self.hidden_dim),
                    dtype=torch.float,
                    device=self.device,
                )

        hop_num = int(self.copy_initial)
        for key, value in x_out_per_rank.items():
            setattr(data, f"x{key}_{hop_num}", value.float().to(self.device))

        return data


def interrank_boundary_index(x_src, boundary_index, n_dst_nodes):
    """
    Recover lifted graph.

    Edge-to-node boundary relationships of a graph with n_nodes and n_edges
    can be represented as up-adjacency node relations. There are n_nodes+n_edges nodes in this lifted graph.
    Desgiend to work for regular (edge-to-node and face-to-edge) boundary relationships.

    Parameters
    ----------
    x_src : torch.tensor
        Source node features. Shape [n_src_nodes, n_features]. Should represent edge or face features.
    boundary_index : list of lists or list of tensors
        List boundary_index[0] stores node ids in the boundary of edge stored in boundary_index[1].
        List boundary_index[1] stores list of edges.
    n_dst_nodes : int
        Number of destination nodes.

    Returns
    -------
    edge_index : list of lists
        The edge_index[0][i] and edge_index[1][i] are the two nodes of edge i.
    edge_attr : tensor
        Edge features are given by feature of bounding node represnting an edge. Shape [n_edges, n_features].
    """
    node_ids = (
        boundary_index[0]
        if torch.is_tensor(boundary_index[0])
        else torch.tensor(boundary_index[0], dtype=torch.int32)
    )
    edge_ids = (
        boundary_index[1]
        if torch.is_tensor(boundary_index[1])
        else torch.tensor(boundary_index[1], dtype=torch.int32)
    )

    max_node_id = n_dst_nodes
    adjusted_edge_ids = edge_ids + max_node_id

    edge_index = torch.zeros((2, node_ids.numel()), dtype=node_ids.dtype)
    edge_index[0, :] = node_ids
    edge_index[1, :] = adjusted_edge_ids

    edge_attr = x_src[edge_ids].squeeze()

    return edge_index, edge_attr


##### GPSE model functions

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


"""A transform that has the GPSE_encoder model."""

import torch
import torch.nn as nn
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.layer import MLP, new_layer_config
from torch_geometric.graphgym.register import pooling_dict, register_head


def _pad_and_stack(x1: torch.Tensor, x2: torch.Tensor, pad1: int, pad2: int):
    """
    Pad and stack two tensors.

    Parameters
    ----------
    x1 : torch.Tensor
        First tensor to pad and stack.
    x2 : torch.Tensor
        Second tensor to pad and stack.
    pad1 : int
        Padding size for the first tensor.
    pad2 : int
        Padding size for the second tensor.

    Returns
    -------
    torch.Tensor
        Padded and stacked tensor.
    """
    padded_x1 = nn.functional.pad(x1, (0, pad2))
    padded_x2 = nn.functional.pad(x2, (pad1, 0))
    return torch.vstack([padded_x1, padded_x2])


def _apply_index(batch, virtual_node: bool, pad_node: int, pad_graph: int):
    """
    Apply index to batch data, handling virtual nodes and padding.

    Parameters
    ----------
    batch : Batch
        A batch of data containing node features, graph features, and labels.
    virtual_node : bool
        Whether to handle virtual nodes.
    pad_node : int
        Padding size for node features.
    pad_graph : int
        Padding size for graph features.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        Padded and stacked predictions and true values.
    """
    graph_pred, graph_true = batch.graph_feature, batch.y_graph
    node_pred, node_true = batch.node_feature, batch.y
    if virtual_node:
        # Remove virtual node
        idx = torch.concat(
            [
                torch.where(batch.batch == i)[0][:-1]
                for i in range(batch.batch.max().item() + 1)
            ]
        )
        node_pred, node_true = node_pred[idx], node_true[idx]

    # Stack node predictions on top of graph predictions and pad with zeros
    pred = _pad_and_stack(node_pred, graph_pred, pad_node, pad_graph)
    true = _pad_and_stack(node_true, graph_true, pad_node, pad_graph)

    return pred, true


@register_head("inductive_hybrid_multi")
class GNNInductiveHybridMultiHead(nn.Module):
    """GNN prediction head for inductive node and graph prediction tasks using individual MLP for each task.

    Parameters
    ----------
    dim_in : int
        Input dimension.
    dim_out : int
        Output dimension. Not used. Use share.num_node_targets and share.num_graph_targets instead.
    """

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.node_target_dim = cfg.share.num_node_targets
        self.graph_target_dim = cfg.share.num_graph_targets
        self.virtual_node = cfg.virtual_node
        num_layers = cfg.gnn.layers_post_mp

        layer_config = new_layer_config(
            dim_in, 1, num_layers, has_act=False, has_bias=True, cfg=cfg
        )
        if cfg.gnn.multi_head_dim_inner is not None:
            layer_config.dim_inner = cfg.gnn.multi_head_dim_inner
        self.node_post_mps = nn.ModuleList(
            [MLP(layer_config) for _ in range(self.node_target_dim)]
        )

        self.graph_pooling = pooling_dict[cfg.model.graph_pooling]
        self.graph_post_mp = MLP(
            new_layer_config(
                dim_in,
                self.graph_target_dim,
                num_layers,
                has_act=False,
                has_bias=True,
                cfg=cfg,
            )
        )
        self.counter = 0

    def forward(self, batch):
        """Forward pass for the GNNInductiveHybridMultiHead.

        Parameters
        ----------
        batch : Batch
            A batch of data containing node features, edge attributes, and edge indices.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Padded and stacked predictions and true values.
        """
        batch.node_feature = torch.hstack(
            [m(batch.x) for m in self.node_post_mps]
        )
        # if self.counter < 178:
        #     self.counter += 1
        graph_emb = self.graph_pooling(batch.x, batch.batch)
        # else:
        #     pass
        batch.graph_feature = self.graph_post_mp(graph_emb)
        return _apply_index(
            batch,
            self.virtual_node,
            self.node_target_dim,
            self.graph_target_dim,
        )
