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
        # Check if data has instance x:
        if hasattr(data, "x") and data.x is not None:
            num_nodes = data.x.shape[0]
        else:
            num_nodes = data.num_nodes

        incidence_1 = torch.zeros(num_nodes, num_nodes * self.k)
        edge_index = torch_geometric.utils.to_undirected(data.edge_index)

        # Detect isolated nodes
        isolated_nodes = [
            i for i in range(num_nodes) if i not in edge_index[0]
        ]
        if len(isolated_nodes) > 0:
            # Add completely isolated nodes to the edge_index
            edge_index = torch.cat(
                [
                    edge_index,
                    torch.tensor(
                        [isolated_nodes, isolated_nodes], dtype=torch.long
                    ),
                ],
                dim=1,
            )

        for n in range(num_nodes):
            exclude_neighbors = torch.tensor([n])
            for k in range(1, self.k + 1):
                neighbors, _, _, _ = torch_geometric.utils.k_hop_subgraph(
                    n, k, edge_index
                )
                common_idxs = torch.isin(neighbors, exclude_neighbors)
                neighbors = neighbors[~common_idxs]
                hyperedge_idx = n * self.k + k - 1
                incidence_1[neighbors, hyperedge_idx] = 1
                incidence_1[n, hyperedge_idx] = 1
                exclude_neighbors = torch.cat([exclude_neighbors, neighbors])

        num_hyperedges = incidence_1.shape[1]
        incidence_1 = torch.Tensor(incidence_1).to_sparse_coo()
        return {
            "incidence_hyperedges": incidence_1,
            "num_hyperedges": num_hyperedges,
            "x_0": data.x,
        }
