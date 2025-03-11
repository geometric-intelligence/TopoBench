"""This module implements the NeighborhoodComplexLifting class, which lifts graphs to simplicial complexes."""

from toponetx.classes import SimplicialComplex
from torch_geometric.data import Data

from topobench.transforms.liftings.graph2simplicial.base import (
    Graph2SimplicialLifting,
)


class NeighborhoodComplexLifting(Graph2SimplicialLifting):
    """Lifts graphs to a simplicial complex domain by identifying the neighborhood complex as k-simplices.

    Parameters
    ----------
    **kwargs : optional
        Additional arguments for the class.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def lift_topology(self, data: Data) -> dict:
        r"""Lift the topology of a graph to a simplicial complex.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data to be lifted.

        Returns
        -------
        dict
            The lifted topology.
        """
        graph = self._generate_graph_from_data(data)
        graph = graph.to_undirected()
        simplicial_complex = SimplicialComplex(simplices=graph)

        # For every node u
        for u in graph.nodes:
            neighbourhood_complex = set()
            neighbourhood_complex.add(u)
            # Check it's neighbours
            for v in graph.neighbors(u):
                # For every other node w != u ^ w != v
                for w in graph.nodes:
                    # w == u
                    if w == u:
                        continue
                    # w == v
                    if w == v:
                        continue

                    # w and u share v as it's neighbour
                    if v in graph.neighbors(w):
                        neighbourhood_complex.add(w)
            # Do not add 0-simplices
            if len(neighbourhood_complex) < 2:
                continue
            # Do not add i-simplices if the maximum dimension is lower
            if len(neighbourhood_complex) > self.complex_dim + 1:
                continue
            simplicial_complex.add_simplex(neighbourhood_complex)

        feature_dict = {i: f for i, f in enumerate(data["x"])}

        simplicial_complex.set_simplex_attributes(
            feature_dict, name="features"
        )

        return self._get_lifted_topology(simplicial_complex, graph)
