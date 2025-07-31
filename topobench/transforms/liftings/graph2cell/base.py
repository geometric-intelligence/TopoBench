"""Abstract class for lifting graphs to cell complexes."""

import networkx as nx
import torch
from toponetx.classes import CellComplex
import numpy as np
import torch_geometric

from topobench.data.utils.utils import get_complex_connectivity
from topobench.transforms.liftings import GraphLifting


class Graph2CellLifting(GraphLifting):
    r"""Abstract class for lifting graphs to cell complexes.

    Parameters
    ----------
    complex_dim : int, optional
        The dimension of the cell complex to be generated. Default is 2.
    **kwargs : optional
        Additional arguments for the class.
    """

    def __init__(self, complex_dim=2, **kwargs):
        super().__init__(**kwargs)
        self.betti_numbers = True
        self.cross_diameter = True
        self.complex_dim = complex_dim
        self.type = "graph2cell"

    def _get_lifted_topology(
        self, cell_complex: CellComplex, graph: nx.Graph
    ) -> dict:
        r"""Return the lifted topology.

        Parameters
        ----------
        cell_complex : CellComplex
            The cell complex.
        graph : nx.Graph
            The input graph.

        Returns
        -------
        dict
            The lifted topology.
        """
        lifted_topology = get_complex_connectivity(
            cell_complex, self.complex_dim, neighborhoods=self.neighborhoods
        )
        lifted_topology["x_0"] = torch.stack(
            list(cell_complex.get_cell_attributes("features", 0).values())
        )
        # If new edges have been added during the lifting process, we discard the edge attributes
        if self.contains_edge_attr and cell_complex.shape[1] == (
            graph.number_of_edges()
        ):
            lifted_topology["x_1"] = torch.stack(
                list(cell_complex.get_cell_attributes("features", 1).values())
            )
        # If betti numbers want to be computed we can do it here
        if self.betti_numbers:
            b_0, b_1, b_2 = get_betti_numbers(graph, max_cycle_size=self.max_cell_length)
            lifted_topology["b_0"] = torch.tensor(b_0)
            lifted_topology["b_1"] = torch.tensor(b_1)
            lifted_topology["b_2"] = torch.tensor(b_2)

        # Same for cross diameter
        if self.cross_diameter:
            cd = get_cross_diameter(lifted_topology, cell_complex)
            lifted_topology["cross_diameter"] = torch.tensor(cd)
        return lifted_topology

def get_betti_numbers(G: nx.Graph, max_cycle_size: int = 18):
    """
    returns betti numbers b_0, b_1, b_2 of a graph lifted to a CC with cyclic lift
    """
    num_nodes = G.number_of_nodes()
    cycle_incidence_matrix = get_cycle_incidence_matrix(G=G, k=max_cycle_size)
    num_edges, num_cycles = cycle_incidence_matrix.shape
    euler_number = num_nodes - num_edges + num_cycles
    b_0 = nx.number_connected_components(G)

    # kernel_sets = find_kernel_sets(cycle_incidence_matrix)
    b_2 = num_cycles - matrix_rank_mod_2(cycle_incidence_matrix)
    b_1 = b_0 + b_2 - euler_number
    return b_0, b_1, b_2

def matrix_rank_mod_2(matrix) -> int:
    """
    computes the dimension of the linear space spanned by the columns of a 0-1 matrix
    where the space is over the field of size 2.
    """

    # Convert the matrix to F₂ (i.e., take all elements modulo 2)
    matrix = (matrix % 2).astype(int)

    # Get the number of rows and columns
    rows, cols = matrix.shape

    # Initialize rank
    rank = 0

    # Perform Gaussian elimination
    for j in range(cols):
        # Find a non-zero element in this column
        for i in range(rank, rows):
            if matrix[i, j] == 1:
                # Swap rows
                matrix[rank], matrix[i] = matrix[i].copy(), matrix[rank].copy()

                # Eliminate this variable from other equations
                for k in range(rank + 1, rows):
                    if matrix[k, j] == 1:
                        matrix[k] ^= matrix[rank]  # XOR operation (addition in F₂)

                rank += 1
                break

        if rank == rows:
            break

    return rank

def get_cycle_incidence_matrix(G, k):
    """
    given a networkx graph computes the edge-cycle incidence matrix without counting 
    chords as incident edges. 

    returns a nupy array of shape (n,m) where n is the number of edges and m is the 
    number of cycles.
    """

    def find_cycles(graph, start, current, path, visited, depth):
        if depth > k:
            return

        visited.add(current)
        path.append(current)

        for neighbor in graph.neighbors(current):
            if neighbor == start and depth > 2:
                cycle = path[:]
                cycles.append(cycle)
            elif neighbor not in visited:
                find_cycles(graph, start, neighbor, path, visited, depth + 1)

        path.pop()
        visited.remove(current)

    cycles = []
    for node in G.nodes():
        find_cycles(G, node, node, [], set(), 1)

    # Remove duplicate cycles
    unique_cycles = []
    cycle_sets = set()
    for cycle in cycles:
        cycle_set = frozenset(cycle)
        if cycle_set not in cycle_sets and len(cycle_set) == len(cycle):
            cycle_sets.add(cycle_set)
            unique_cycles.append(cycle)

    # Create edge to index mapping
    edge_to_index = {frozenset(edge): i for i, edge in enumerate(G.edges())}

    # Create incidence matrix
    incidence_matrix = np.zeros((len(G.edges()), len(unique_cycles)), dtype=int)

    for j, cycle in enumerate(unique_cycles):
        cycle_edges = list(zip(cycle, cycle[1:] + cycle[:1]))
        for edge in cycle_edges:
            edge_set = frozenset(edge)
            if edge_set in edge_to_index:
                i = edge_to_index[edge_set]
                incidence_matrix[i, j] = 1

    return incidence_matrix

def get_cross_diameter(lifted_topology, CC) -> int:
    num_cells_low = CC.number_of_nodes()
    num_cells_high = CC.number_of_cells()

    incidence = lifted_topology['2-up_incidence-0'].to_dense().T.nonzero().t().contiguous()
    low_adjacency = lifted_topology['up_adjacency-0'].to_dense().nonzero().t().contiguous()
    distance_matrix = get_subcomplex_distance_node_marking(
        incidence=incidence,
        num_cells_low=num_cells_low,
        num_cells_high=num_cells_high,
        edge_index_low=low_adjacency,
    )
    distance_matrix[distance_matrix == 1001] = -1
    if distance_matrix.shape[0] == 0:
        return -1
    return distance_matrix.max().item()

def get_subcomplex_distance_node_marking(
    incidence: torch.Tensor,
    edge_index_low: torch.Tensor,
    num_cells_low: int,
    num_cells_high: int,
) -> torch.Tensor:
    low_rank_spd = get_all_pairs_shortest_paths(
        edge_index_low, max_num_nodes=num_cells_low
    )
    if num_cells_high == 0:
        return torch.empty(0, 1)
    spd_encode_list = []

    for i in range(num_cells_high):
        low_rank_indices = incidence[0][incidence[1] == i]
        spd_cell, _ = torch.min(low_rank_spd[:, low_rank_indices], dim=1)

        spd_encode_list.append(spd_cell.view(-1, 1))
    return torch.cat(spd_encode_list, dim=1).reshape(-1, 1)

def get_all_pairs_shortest_paths(
    edge_index, max_num_nodes: int, imputing_val: int = 1001
):
    """
    input is an adjacency of the original complex. Computes spd on this as graph
    """
    adj = torch_geometric.utils.to_dense_adj(
        edge_index, max_num_nodes=max_num_nodes
    ).squeeze(0)

    spd = torch.where(
        ~torch.eye(len(adj), dtype=bool) & (adj == 0),
        torch.full_like(adj, imputing_val),
        adj,
    )
    # Floyd-Warshall

    for k in range(len(spd)):
        dist_from_source_to_k = spd[:, [k]]
        dist_from_k_to_target = spd[[k], :]
        dist_from_source_to_target_via_k = (
            dist_from_source_to_k + dist_from_k_to_target
        )
        spd = torch.minimum(spd, dist_from_source_to_target_via_k)
    return spd
