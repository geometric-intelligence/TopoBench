"""Abstract class for lifting graphs to cell complexes."""

import networkx as nx
import torch
from toponetx.classes import CellComplex

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
        return lifted_topology
