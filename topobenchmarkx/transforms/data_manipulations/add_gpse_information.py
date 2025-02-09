"""A transform that adds positional information to the graph."""

import os

import torch
import torch_geometric
import torch_geometric.data
from torch_geometric.data import Data
from torch_geometric.graphgym.config import cfg, load_cfg, set_cfg
from torch_geometric.graphgym.model_builder import create_model

from topobenchmarkx.data.utils import get_routes_from_neighborhoods_simplex


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
        nbhd_t = params[f"adjacency_{src_rank}"]
        batch_route = Data(
            x=getattr(params, f"x_{src_rank}"),
            edge_index=nbhd_t.indices(),
            edge_weight=nbhd_t.values().squeeze(),
            edge_attr=nbhd_t.values().squeeze(),
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
        device = getattr(params, f"x_{src_rank}").device
        feat_on_dst = torch.zeros_like(getattr(params, f"x_{dst_rank}"))
        x_in = torch.vstack([feat_on_dst, getattr(params, f"x_{src_rank}")])
        batch_expanded = torch.cat([dst_batch, src_batch], dim=0)

        batch_route = Data(
            x=x_in.to(device),
            edge_index=edge_index.to(device),
            edge_attr=edge_attr.to(device),
            edge_weight=edge_attr.to(device),
            batch=batch_expanded.to(device),
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
                # This happens when there are not destination cells for a route
                # In this case, we do not need to aggregate
                if route_index not in x_out_per_route:
                    continue
                x_out_per_rank[dst_rank] = x_out_per_route[route_index]
            else:
                x_out_per_rank[dst_rank] += x_out_per_route[route_index]
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
                # If destination nodes is 0 then messages cannot be passed to
                # non existent node
                if n_dst_nodes == 0:
                    continue
                if src_rank > dst_rank:
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

        if batch_route.x.shape[0] < 2:
            return batch_route["x"]
        else:
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
        if n_dst_cells == 0:
            x_out = expanded_out
        else:
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

        self.routes = get_routes_from_neighborhoods_simplex(self.neighborhoods)

        nbhd_cache = self.get_nbhd_cache(data)

        x_out_per_route = {}
        # Interate over the routes (i, [0, 1])
        for route_index, route in enumerate(self.routes):
            src_rank, dst_rank = route

            if src_rank == dst_rank:
                x_out = self.forward_intrarank(src_rank, route_index, data)
                x_out_per_route[route_index] = x_out

            elif src_rank != dst_rank:
                # If there are no destination cells, then there is no need to forward
                if data[f"x_{dst_rank}"].shape[0] == 0:
                    continue

                x_out = self.forward_interank(
                    src_rank, dst_rank, nbhd_cache, data
                )
                # Outputs of this particular route
                x_out_per_route[route_index] = x_out

        # aggregate across neighborhoods
        self.model.eval()
        with torch.inference_mode():
            x_out_per_rank = self.aggregate_inter_nbhd(x_out_per_route)

        # If no information was passed to a rank, then we initialize an empty vector
        # with the output dimension of the pre-trained model
        # and set the features as 1
        hidden_dim = (
            self.parameters["dim_target_node"]
            + self.parameters["dim_target_graph"]
        )
        for rank in range(self.max_rank + 1):
            if rank not in x_out_per_rank:
                x_out_per_rank[rank] = torch.ones(
                    (data[f"x_{rank}"].shape[0], hidden_dim),
                    dtype=torch.float,
                    device=self.device,
                )

        hop_num = int(self.copy_initial)
        for key, value in x_out_per_rank.items():
            setattr(data, f"x{key}_{hop_num}", value.float())

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
