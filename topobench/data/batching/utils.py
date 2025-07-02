"""Utility functions for batching cells of different ranks."""

import copy

import torch
import torch_geometric.typing
from torch import Tensor
from torch_geometric.data import Data
from torch_sparse import SparseTensor


def reduce_higher_ranks_incidences(
    batch, cells_ids, rank, max_rank, is_hypergraph=False
):
    """Reduce the incidences with higher rank than the specified one.

    Parameters
    ----------
    batch : torch_geometric.data.Data
        The input data.
    cells_ids : list[torch.Tensor]
        List of tensors containing the ids of the cells. The length of the list should be equal to the maximum rank.
    rank : int
        The rank to select the higher order incidences.
    max_rank : int
        The maximum rank of the incidences.
    is_hypergraph : bool
        Whether the data represents an hypergraph.

    Returns
    -------
    torch_geometric.data.Data
        The output data with the reduced incidences.
    list[torch.Tensor]
        The updated indices of the cells. Each element of the list is a tensor containing the ids of the cells of the corresponding rank.
    """
    for i in range(rank + 1, max_rank + 1):
        if is_hypergraph:
            incidence = batch.incidence_hyperedges
        else:
            incidence = batch[f"incidence_{i}"]

        # if i != rank+1:
        incidence = torch.index_select(incidence, 0, cells_ids[i - 1])
        cells_ids[i] = torch.where(torch.sum(incidence, dim=0).to_dense() > 1)[
            0
        ]
        incidence = torch.index_select(incidence, 1, cells_ids[i])
        if is_hypergraph:
            batch.incidence_hyperedges = incidence
        else:
            batch[f"incidence_{i}"] = incidence

    return batch, cells_ids


def reduce_lower_ranks_incidences(batch, cells_ids, rank, is_hypergraph=False):
    """Reduce the incidences with lower rank than the specified one.

    Parameters
    ----------
        batch : torch_geometric.data.Data
            The input data.
        cells_ids : list[torch.Tensor]
            List of tensors containing the ids of the cells. The length of the list should be equal to the maximum rank.
        rank : int
            The rank of the cells to consider.
        is_hypergraph : bool
            Whether the data represents an hypergraph.

    Returns
    -------
        torch.Tensor
            The indices of the nodes contained by the cells.
        list[torch.Tensor]
            The updated indices of the cells. Each element of the list is a tensor containing the ids of the cells of the corresponding rank.
    """
    for i in range(rank, 0, -1):
        if is_hypergraph:
            incidence = batch.incidence_hyperedges
        else:
            incidence = batch[f"incidence_{i}"]
        incidence = torch.index_select(incidence, 1, cells_ids[i])
        cells_ids[i - 1] = torch.where(
            torch.sum(incidence, dim=1).to_dense() > 0
        )[0]
        incidence = torch.index_select(incidence, 0, cells_ids[i - 1])
        if is_hypergraph:
            batch.incidence_hyperedges = incidence
        else:
            batch[f"incidence_{i}"] = incidence

    if not is_hypergraph:
        incidence = batch["incidence_0"]
        incidence = torch.index_select(incidence, 1, cells_ids[0])
        batch["incidence_0"] = incidence
    return batch, cells_ids


def reduce_matrices(batch, cells_ids, names, max_rank):
    """Reduce the matrices using the indices in cells_ids.

    The matrices are assumed to be in the batch with the names specified in the list names.

    Parameters
    ----------
    batch : torch_geometric.data.Data
        The input data.
    cells_ids : list[torch.Tensor]
        List of tensors containing the ids of the cells. The length of the list should be equal to the maximum rank.
    names : list[str]
        List of names of the matrices in the batch. They should appear in the format f"{name}{i}" where i is the rank of the matrix.
    max_rank : int
        The maximum rank of the matrices.

    Returns
    -------
    torch_geometric.data.Data
        The output data with the reduced matrices.
    """
    for i in range(max_rank + 1):
        for name in names:
            if f"{name}{i}" in batch.keys():  # noqa
                matrix = batch[f"{name}{i}"]
                matrix = torch.index_select(matrix, 0, cells_ids[i])
                matrix = torch.index_select(matrix, 1, cells_ids[i])
                batch[f"{name}{i}"] = matrix
    return batch


def reduce_neighborhoods(batch, node, rank=0, remove_self_loops=True):
    """Reduce the neighborhoods of the cells in the batch.

    Parameters
    ----------
    batch : torch_geometric.data.Data
        The input data.
    node : torch.Tensor
        The indices of the cells to batch over.
    rank : int
        The rank of the cells to batch over.
    remove_self_loops : bool
        Whether to remove self loops from the edge_index.

    Returns
    -------
    torch_geometric.data.Data
        The output data with the reduced neighborhoods.
    """
    is_hypergraph = False
    if hasattr(batch, "incidence_hyperedges"):
        is_hypergraph = True
        max_rank = 1
    else:
        max_rank = len([key for key in batch.keys() if "incidence" in key]) - 1  # noqa

    # In the graph case we don't create incidences
    if max_rank == -1:
        cells_ids = [None for _ in range(2)]
        cells_ids[0] = node
        total_nodes = batch.x.shape[0]

        adjacency = SparseTensor(
            row=batch["edge_index"][0],
            col=batch["edge_index"][1],
            sparse_sizes=(total_nodes, total_nodes),
        )
        reduced_adjacency = adjacency.index_select(
            0, cells_ids[0]
        ).index_select(1, cells_ids[0])
        reduced_adjacency = reduced_adjacency.coalesce()
        new_edge_index = torch.stack(
            [reduced_adjacency.storage.row(), reduced_adjacency.storage.col()]
        )
        batch["edge_index"] = new_edge_index

        batch["x"] = batch.x[cells_ids[0]]
        batch["y"] = batch.y[cells_ids[0]] if hasattr(batch, "y") else None

        batch["batch_0"] = torch.zeros_like(batch["x"][:, 0], dtype=torch.long)
        return batch

    if rank > max_rank:
        raise ValueError(
            f"Rank {rank} is greater than the maximum rank {max_rank} in the dataset."
        )

    cells_ids = [None for _ in range(max_rank + 1)]

    # the indices of the cells selected by the NeighborhoodLoader are saved in the batch in the attribute n_id
    cells_ids[rank] = node

    batch, cells_ids = reduce_higher_ranks_incidences(
        batch, cells_ids, rank, max_rank, is_hypergraph
    )
    batch, cells_ids = reduce_lower_ranks_incidences(
        batch, cells_ids, rank, is_hypergraph
    )

    batch = reduce_matrices(
        batch,
        cells_ids,
        names=[
            # TODO: quick fix here, commented initial neightbourhoods and substituted with some random guess
            "down_laplacian_",
            "up_laplacian_",
            "hodge_laplacian_",
            "adjacency_",
            # added
            "coadjacency_",
            "up_adjacency-",
            "down_adjacency-",
            "up_incidence-",
            "down_incidence-",
            # "incidence",
            # "down_laplacian",
            # "up_laplacian",
            # "adjacency",
            # "coadjacency",
            # "hodge_laplacian",
        ],
        max_rank=max_rank,
    )

    # reduce the feature matrices
    for i in range(max_rank + 1):
        if f"x_{i}" in batch.keys():  # noqa
            batch[f"x_{i}"] = batch[f"x_{i}"][cells_ids[i]]

            # For the hopse encodings:
            hopse_keys = [key for key in batch.keys() if f"x{i}_" in key]
            if len(hopse_keys) > 0:
                for hopse_key in hopse_keys:
                    batch[hopse_key] = batch[hopse_key][cells_ids[i]]

        elif "x_hyperedges" in batch.keys() and i == 1:  # noqa
            batch["x_hyperedges"] = batch["x_hyperedges"][cells_ids[i]]

    # fix edge_index
    if not is_hypergraph:
        # TODO: random quick fix here, rework
        # Check if adjacency_0  exists and take the edge_index from there
        if "adjacency_0" in batch.keys():
            edge_index = batch.adjacency_0.coalesce().indices()
        elif "up_adjacency-0" in batch.keys():
            edge_index = batch["up_adjacency-0"].coalesce().indices()
        else:
            raise ValueError(
                "Either adjacency_0 or up_adjacency-0 should exist in the batch."
            )
        # adjacency_0 = batch.adjacency_0.coalesce()
        # edge_index = adjacency_0.indices()

        if remove_self_loops:
            edge_index = torch_geometric.utils.remove_self_loops(edge_index)[0]
        batch.edge_index = edge_index

    # fix x
    batch.x = batch["x_0"]
    if hasattr(batch, "num_nodes"):
        batch.num_nodes = batch.x.shape[0]

    if hasattr(batch, "y"):
        batch.y = batch.y[cells_ids[rank]]
        # The task is defined for the nodes! hence always 0
        # batch.y = batch.y[cells_ids[0]]

    batch.cells_ids = cells_ids
    return batch


def filter_data(data: Data, cells: Tensor, rank: int) -> Data:
    """Filter the attributes of the data based on the cells passed.

    The function uses the indices passed to select the cells of the specified rank. The cells of lower or higher ranks are selected using the incidence matrices.

    Parameters
    ----------
    data : torch_geometric.data.Data
        The input data.
    cells : Tensor
        Tensor containing the indices of the cells of the specified rank to keep.
    rank : int
        Rank of the cells of interest.

    Returns
    -------
    torch_geometric.data.Data
        The output data with the filtered attributes.
    """
    out = copy.copy(data)
    out = reduce_neighborhoods(out, cells, rank=rank)
    out.n_id = cells
    return out


def get_sampled_neighborhood(data, rank=0, n_hops=1, is_hypergraph=False):
    """Update the edge_index attribute of torch_geometric.data.Data.

    The function finds cells, of the specified rank, that are either upper or lower neighbors.

    Parameters
    ----------
    data : torch_geometric.data.Data
        The input data.
    rank : int
        The rank of the cells that you want to batch over.
    n_hops : int
        Two cells are considered neighbors if they are connected by n hops in the upper or lower neighborhoods.
    is_hypergraph : bool
        Whether the data represents an hypergraph.

    Returns
    -------
    torch_geometric.data.Data
        The output data with updated edge_index.
        edge_index contains indices of connected cells of the specified rank K.
        Two cells of rank K are connected if they are either lower or upper neighbors.
    """
    if rank == 0:
        data.edge_index = torch_geometric.utils.to_undirected(data.edge_index)
        return data
    if is_hypergraph:
        if rank > 1:
            raise ValueError(
                "Hypergraphs are not supported for ranks greater than 1."
            )
        if rank == 1:
            incidence = data.incidence_hyperedges
            A = torch.sparse.mm(incidence, incidence.T)  # lower adj matrix
        else:
            incidence = data.incidence_hyperedges
            A = torch.sparse.mm(incidence.T, incidence)
        for _ in range(n_hops - 1):
            A = torch.sparse.mm(A, A)
        edges = A.indices()
    else:
        # get number of incidences
        max_rank = len([key for key in data.keys() if "incidence" in key]) - 1  # noqa
        if rank > max_rank:
            raise ValueError(
                f"Rank {rank} is greater than the maximum rank {max_rank} in the data."
            )

        # This considers the upper adjacencies
        n_cells = data[f"x_{rank}"].shape[0]
        A_sum = torch.sparse_coo_tensor([[], []], [], (n_cells, n_cells))
        if rank == max_rank:
            edges = torch.empty((2, 0), dtype=torch.long)
        else:
            incidence = data[f"incidence_{rank + 1}"]
            A = torch.sparse.mm(incidence, incidence.T)
            for _ in range(n_hops - 1):
                A = torch.sparse.mm(A, A)
            A_sum += A

        # This is for selecting the whole upper cells
        # for i in range(rank+1, max_rank):
        #     P = torch.sparse.mm(P, data[f"incidence_{i+1}"])
        #     Q = torch.sparse.mm(P,P.T)
        #     edges = torch.cat((edges, Q.indices()), dim=1)

        # This considers the lower adjacencies
        if rank != 0:
            incidence = data[f"incidence_{rank}"]
            A = torch.sparse.mm(incidence.T, incidence)
            for _ in range(n_hops - 1):
                A = torch.sparse.mm(A, A)
            A_sum += A

        # This is for selecting cells if they share any node
        # for i in range(rank-1, 0, -1):
        #     P = torch.sparse.mm(data[f"incidence_{i}"], P)
        #     Q = torch.sparse.mm(P.T,P)
        #     edges = torch.cat((edges, Q.indices()), dim=1)

        edges = A_sum.coalesce().indices()
    # Remove self edges
    mask = edges[0, :] != edges[1, :]
    edges = edges[:, mask]

    data.edge_index = edges

    # We need to set x to x_{rank} since NeighborLoader will take the number of nodes from the x attribute
    # The correct x is given after the reduce_neighborhoods function
    if is_hypergraph and rank == 1:
        data.x = data.x_hyperedges
    else:
        data.x = data[f"x_{rank}"]

    if hasattr(data, "num_nodes"):
        data.num_nodes = data.x.shape[0]
    return data
