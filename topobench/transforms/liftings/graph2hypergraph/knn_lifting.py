"""This module implements the HypergraphKNNLifting class."""

import torch
import torch_geometric

from topobench.transforms.liftings.graph2hypergraph import (
    Graph2HypergraphLifting,
)


class HypergraphKNNLifting(Graph2HypergraphLifting):
    r"""Lift graphs to hypergraph domain by considering k-nearest neighbors.

    Parameters
    ----------
    k_value : int, optional
        The number of nearest neighbors to consider. Must be positive. Default is 1.
    loop : bool, optional
        If True the hyperedges will contain the node they were created from.
    **kwargs : optional
        Additional arguments for the class.

    Raises
    ------
    ValueError
        If k_value is less than 1.
    TypeError
        If k_value is not an integer or if loop is not a boolean.
    """

    def __init__(self, k_value=1, loop=True, **kwargs):
        super().__init__(**kwargs)

        # Validate k_value
        if not isinstance(k_value, int):
            raise TypeError("k_value must be an integer")
        if k_value < 1:
            raise ValueError("k_value must be greater than or equal to 1")

        # Validate loop
        if not isinstance(loop, bool):
            raise TypeError("loop must be a boolean")

        self.k = k_value
        self.loop = loop
        self.transform = torch_geometric.transforms.KNNGraph(self.k, self.loop)

    def lift_topology(self, data: torch_geometric.data.Data) -> dict:
        r"""Lift a graph to hypergraph by considering k-nearest neighbors.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data to be lifted.

        Returns
        -------
        dict
            The lifted topology.
        """
        num_nodes = data.x.shape[0]
        data.pos = data.x
        num_hyperedges = num_nodes
        incidence_1 = torch.zeros(num_nodes, num_nodes)
        data_lifted = self.transform(data)
        # check for loops, since KNNGraph is inconsistent with nodes with equal features
        if self.loop:
            for i in range(num_nodes):
                if not torch.any(
                    torch.all(
                        data_lifted.edge_index == torch.tensor([[i, i]]).T,
                        dim=0,
                    )
                ):
                    connected_nodes = data_lifted.edge_index[
                        0, data_lifted.edge_index[1] == i
                    ]
                    dists = torch.sqrt(
                        torch.sum(
                            (
                                data.pos[connected_nodes]
                                - data.pos[i].unsqueeze(0) ** 2
                            ),
                            dim=1,
                        )
                    )
                    furthest = torch.argmax(dists)
                    idx = torch.where(
                        torch.all(
                            data_lifted.edge_index
                            == torch.tensor(
                                [[connected_nodes[furthest], i]]
                            ).T,
                            dim=0,
                        )
                    )[0]
                    data_lifted.edge_index[:, idx] = torch.tensor([[i, i]]).T

        incidence_1[data_lifted.edge_index[1], data_lifted.edge_index[0]] = 1
        incidence_1 = torch.Tensor(incidence_1).to_sparse_coo()
        return {
            "incidence_hyperedges": incidence_1,
            "num_hyperedges": num_hyperedges,
            "x_0": data.x,
        }
