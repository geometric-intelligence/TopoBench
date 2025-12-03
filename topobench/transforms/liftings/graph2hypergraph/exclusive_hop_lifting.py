"""This module implements the k-hop lifting of graphs to hypergraphs."""

import torch
import torch_geometric

from topobench.transforms.liftings.graph2hypergraph import (
    Graph2HypergraphLifting,
)


class HypergraphExclusiveHopLifting(Graph2HypergraphLifting):
    r"""Lift graph to hypergraphs by considering exclusive k-hop neighborhoods.

    The class transforms graphs to hypergraph domain by considering k-hop neighborhoods of
    a node, but excluding the k-1 hop neighbors. This lifting extracts a number of hyperedges equal to k times the number of
    nodes in the graph.

    Parameters
    ----------
    k_value : int, optional
        The number of hops to consider. Default is 1.
    **kwargs : optional
        Additional arguments for the class.
    """

    def __init__(self, k_value=2, **kwargs):
        super().__init__(**kwargs)
        self.k = k_value

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(k={self.k!r})"

    def lift_topology(self, data: torch_geometric.data.Data) -> dict:
        r"""Lift a graphs to hypergraphs by considering exclusive k-hop neighborhoods.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data to be lifted.

        Returns
        -------
        dict
            The lifted topology.
        """
        if hasattr(data, "x") and data.x is not None:
            num_nodes = data.x.shape[0]
        else:
            num_nodes = data.num_nodes

        # The number of hyperedges is k times the number of nodes
        num_hyperedges = num_nodes * self.k

        edge_index = torch_geometric.utils.to_undirected(data.edge_index)
        device = edge_index.device

        # Detect isolated nodes
        isolated_nodes = [
            i for i in range(num_nodes) if i not in edge_index[0]
        ]
        if len(isolated_nodes) > 0:
            # Add self-loops for isolated nodes
            isolated_tensor = torch.tensor(
                [isolated_nodes, isolated_nodes], dtype=torch.long
            ).to(device)
            edge_index = torch.cat(
                [
                    edge_index,
                    isolated_tensor,
                ],
                dim=1,
            )

        all_indices = []

        for n in range(num_nodes):
            nodes_up_to_k_minus_1 = torch.tensor([n], device=device)

            for k_i in range(1, self.k + 1):
                nodes_up_to_k, _, _, _ = torch_geometric.utils.k_hop_subgraph(
                    n, k_i, edge_index
                )

                is_common = torch.isin(nodes_up_to_k, nodes_up_to_k_minus_1)
                exclusive_neighbors = nodes_up_to_k[~is_common]

                hyperedge_idx = n * self.k + k_i - 1

                anchor_node = torch.tensor([n], device=device)
                nodes_in_hyperedge = torch.cat(
                    [exclusive_neighbors, anchor_node]
                )
                nodes_in_hyperedge = torch.unique(
                    nodes_in_hyperedge
                )  # Ensure no duplicates

                num_nodes_in_he = nodes_in_hyperedge.shape[0]
                if num_nodes_in_he > 0:
                    # Row indices (nodes)
                    row_indices = nodes_in_hyperedge.to(torch.long)
                    # Col indices (hyperedge)
                    col_indices = torch.full(
                        (num_nodes_in_he,),
                        fill_value=hyperedge_idx,
                        dtype=torch.long,
                        device=device,
                    )

                    current_indices = torch.stack(
                        [row_indices, col_indices], dim=0
                    )
                    all_indices.append(current_indices)

                nodes_up_to_k_minus_1 = nodes_up_to_k

        if not all_indices:
            # Handle empty graph
            indices = torch.empty((2, 0), dtype=torch.long, device=device)
            values = torch.empty(0, dtype=torch.float32, device=device)
        else:
            indices = torch.cat(all_indices, dim=1)
            num_non_zero = indices.shape[1]
            values = torch.ones(
                num_non_zero, dtype=torch.float32, device=device
            )

        # Create the sparse incidence matrix: [N_nodes, N_hyperedges]
        incidence_1 = torch.sparse_coo_tensor(
            indices=indices,
            values=values,
            size=(num_nodes, num_hyperedges),
        )

        return {
            "incidence_hyperedges": incidence_1,
            "num_hyperedges": num_hyperedges,
            "x_0": data.x,
        }
