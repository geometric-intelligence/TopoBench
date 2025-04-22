"""A transform that adds positional information to the graph."""

import torch
import torch_geometric
import torch_geometric.data
from torch_geometric.data import Data

from topobench.data.utils import get_routes_from_neighborhoods
from topobench.transforms.data_manipulations.positional_structural_encodings import (
    DerivePS,
)


class dotdict(dict):
    """Dot.notation access to dictionary attributes."""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class HOPSE_PE_Information(torch_geometric.transforms.BaseTransform):
    r"""A transform that uses a positional and structural information added to the graph.

    Parameters
    ----------
    **kwargs : optional
        Parameters for the transform.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.type = "hopse_ps_information"
        self.parameters = kwargs

        self.max_rank = kwargs["max_rank"]
        self.copy_initial = kwargs["copy_initial"]
        self.neighborhoods = kwargs["neighborhoods"]

        self.device = (
            "cpu" if kwargs["device"] == "cpu" else f"cuda:{kwargs['cuda'][0]}"
        )

        self.ps = DerivePS(**kwargs)
        self.num_pe_considered = len(kwargs["pe_types"])
        # self.model = create_model(
        #     dim_in=cfg.dim_in, dim_out=self.parameters["dim_out"]
        # )
        # model_state_dict = torch.load(
        #     f"{os.getcwd()}/data/pretrained_models/gpse_{self.parameters['pretrain_model'].lower()}.pt",
        #     map_location=torch.device(self.device),
        # )
        # self.model.load_state_dict(model_state_dict["model_state"])
        self.hidden_dim = self.parameters["dim_target_node"]

        # + self.parameters["dim_target_graph"]

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
                    dim=2,
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
            size=(batch_route.x.shape[0], 1),  # cfg.dim_in
            device=self.device,
        )
        input_graph = torch_geometric.data.Data(
            x=input_nodes,
            edge_index=batch_route.edge_index,
            batch=torch.zeros(
                batch_route.x.shape[0],
                dtype=torch.int64,
                device=self.device,
            ),
        ).to(self.device)

        x_out = self.ps(input_graph)
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
            0, 1, size=(batch_route.x.shape[0], 1), device=self.device
        )  # cfg.dim_in

        # Express everything in terms of an input graph
        input_graph = torch_geometric.data.Data(
            x=input_nodes,
            edge_index=batch_route.edge_index,
            batch=torch.zeros(
                batch_route.x.shape[0], dtype=torch.int64, device=self.device
            ),
        ).to(self.device)

        expanded_out = self.ps(input_graph)
        # TODO: Is this correct? I mean logically correct given the our PEs
        # Only grab the cells we are interested in
        x_out = expanded_out[:, :n_dst_cells, :]

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
                        (self.num_pe_considered, 0, self.hidden_dim),
                        dtype=torch.float,
                        device=self.device,
                    )
                    continue
                # We cannot use PE embeddings on single-node graphs
                if getattr(data, f"x_{src_rank}").shape[0] == 1:
                    x_out_per_route[route_index] = torch.zeros(
                        (self.num_pe_considered, 1, self.hidden_dim),
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
                        (self.num_pe_considered, 0, self.hidden_dim),
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
                    (
                        self.num_pe_considered,
                        data[f"x_{rank}"].shape[0],
                        self.hidden_dim,
                    ),  # num of pes, data[f"x_{rank}"].shape[0], self.hidden_dim
                    dtype=torch.float,
                    device=self.device,
                )

        hop_num = int(self.copy_initial)
        for key, value in x_out_per_rank.items():
            for num_pe in range(
                self.num_pe_considered
            ):  # for bla bla bla:  iter num of pes + initial features
                setattr(
                    data,
                    f"x{key}_{num_pe + hop_num}",
                    value[num_pe].float().to(self.device),
                )

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
