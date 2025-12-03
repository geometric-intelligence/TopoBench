"""This module implements the k-hop lifting of graphs to hypergraphs."""

import torch
import torch_geometric

from topobench.transforms.liftings.graph2hypergraph import (
    Graph2HypergraphLifting,
)


class HypergraphKHopLifting(Graph2HypergraphLifting):
    r"""Lift graph to hypergraphs by considering k-hop neighborhoods.

    The class transforms graphs to hypergraph domain by considering k-hop neighborhoods of
    a node. This lifting extracts a number of hyperedges equal to the number of
    nodes in the graph.

    Parameters
    ----------
    k_value : int, optional
        The number of hops to consider. Default is 1.
    **kwargs : optional
        Additional arguments for the class.
    """

    def __init__(self, k_value=1, **kwargs):
        super().__init__(**kwargs)
        self.k = k_value

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(k={self.k!r})"

    def lift_topology(self, data: torch_geometric.data.Data) -> dict:
        r"""Lift a graphs to hypergraphs by considering k-hop neighborhoods.

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

        # The number of hyperedges is equal to the number of nodes
        num_hyperedges = num_nodes

        # Get the undirected edge index
        edge_index = torch_geometric.utils.to_undirected(data.edge_index)

        # Detect isolated nodes
        isolated_nodes = [
            i for i in range(num_nodes) if i not in edge_index[0]
        ]
        if len(isolated_nodes) > 0:
            # Add self-loops for isolated nodes to ensure they are
            # included in their own k-hop neighborhood.
            isolated_tensor = torch.tensor(
                [isolated_nodes, isolated_nodes], dtype=torch.long
            ).to(edge_index.device)

            edge_index = torch.cat(
                [
                    edge_index,
                    isolated_tensor,
                ],
                dim=1,
            )
        all_indices = []

        for n in range(num_nodes):
            neighbors, _, _, _ = torch_geometric.utils.k_hop_subgraph(
                n, self.k, edge_index
            )

            num_neighbors = neighbors.shape[0]
            if num_neighbors > 0:
                row = torch.full(
                    (num_neighbors,), fill_value=n, dtype=torch.long
                )
                col = neighbors.to(torch.long)

                hyperedge_indices = torch.stack([row, col], dim=0)
                all_indices.append(hyperedge_indices)

        if not all_indices:
            # Handle empty graph
            indices = torch.empty(
                (2, 0), dtype=torch.long, device=edge_index.device
            )
            values = torch.empty(
                0, dtype=torch.float32, device=edge_index.device
            )
        else:
            indices = torch.cat(all_indices, dim=1).to(edge_index.device)
            num_non_zero = indices.shape[1]
            values = torch.ones(
                num_non_zero, dtype=torch.float32, device=edge_index.device
            )

        incidence_1 = torch.sparse_coo_tensor(
            indices=indices,
            values=values,
            size=(num_hyperedges, num_nodes),
        )

        return {
            "incidence_hyperedges": incidence_1,
            "num_hyperedges": num_hyperedges,
            "x_0": data.x,
        }
