"""Abstract class for lifting graphs to combinatorial complexes."""

import networkx as nx
from toponetx.classes import CombinatorialComplex

from topobench.data.utils.utils import get_combinatorial_complex_connectivity
from topobench.transforms.liftings import GraphLifting


class Graph2CombinatorialLifting(GraphLifting):
    r"""Abstract class for lifting graphs to combinatorial complexes.

    Parameters
    ----------
    **kwargs : optional
        Additional arguments for the class.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.type = "graph2combinatorial"

    def _get_lifted_topology(
        self, combinatorial_complex: CombinatorialComplex, graph: nx.Graph
    ) -> dict:
        r"""Return the lifted topology.

        Parameters
        ----------
        combinatorial_complex : CellComplex
            The cell complex.
        graph : nx.Graph
            The input graph.

        Returns
        -------
        dict
            The lifted topology.
        """
        lifted_topology = get_combinatorial_complex_connectivity(
            combinatorial_complex,
            self.complex_dim,
            neighborhoods=self.neighborhoods,
        )

        return lifted_topology
