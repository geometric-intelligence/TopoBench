"""This module implements the neighborhood lifting for graphs to cell complexes.

Definition:
* 0-cells: Vertices of the graph.
* 1-cells: Edges of the graph.
* Higher-dimensional cells: Defined based on the neighborhoods of vertices.
A 2-cell is added for each vertex and its immediate neighbors.

Characteristics:
Star-like Structure: Star-like structures centered around a vertex and include all its adjacent vertices.
Flexibility: This approach can generate higher-dimensional cells even in graphs that do not have cycles.
Local Connectivity: The focus is on local connectivity rather than global cycles.
"""

from toponetx.classes import CellComplex

from topobenchmark.transforms.liftings.base import LiftingMap


class NeighborhoodLifting(LiftingMap):
    """Lifts graphs to cell complexes by identifying the cycles as 2-cells.

    Parameters
    ----------
    max_cell_length : int, optional
        The maximum length of the cycles to be lifted. Default is None.
    """

    def __init__(self, max_cell_length=None):
        super().__init__()
        self.max_cell_length = max_cell_length

    def lift(self, domain):
        """Finds the cycles of a graph and lifts them to 2-cells.

        Parameters
        ----------
        domain : nx.Graph
            Graph to be lifted.

        Returns
        -------
        CellComplex
            Lifted cell complex.
        """
        graph = domain

        cell_complex = CellComplex(graph)

        vertices = list(graph.nodes())
        for v in vertices:
            cell_complex.add_node(v, rank=0)

        edges = list(graph.edges())
        for edge in edges:
            cell_complex.add_cell(edge, rank=1)

        for v in vertices:
            neighbors = list(graph.neighbors(v))
            if len(neighbors) > 1:
                two_cell = [v, *neighbors]
                if (
                    self.max_cell_length is not None
                    and len(two_cell) > self.max_cell_length
                ):
                    pass
                else:
                    cell_complex.add_cell(two_cell, rank=2)

        return cell_complex
