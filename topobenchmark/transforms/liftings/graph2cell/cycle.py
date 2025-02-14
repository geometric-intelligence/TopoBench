"""This module implements the cycle lifting for graphs to cell complexes."""

import networkx as nx
from toponetx.classes import CellComplex

from topobenchmark.transforms.liftings.base import LiftingMap


class CellCycleLifting(LiftingMap):
    r"""Lift graphs to cell complexes.

    The algorithm creates 2-cells by identifying the cycles and considering them as 2-cells.

    Parameters
    ----------
    max_cell_length : int, optional
        The maximum length of the cycles to be lifted. Default is None.
    """

    def __init__(self, max_cell_length=None):
        super().__init__()
        self._complex_dim = 2
        self.max_cell_length = max_cell_length

    def lift(self, domain):
        r"""Find the cycles of a graph and lifts them to 2-cells.

        Parameters
        ----------
        domain : nx.Graph
            Graph to be lifted.

        Returns
        -------
        CellComplex
            The cell complex.
        """
        graph = domain

        cycles = nx.cycle_basis(graph)
        cell_complex = CellComplex(graph)

        # Eliminate self-loop cycles
        cycles = [cycle for cycle in cycles if len(cycle) != 1]

        # Eliminate cycles that are greater than the max_cell_length
        if self.max_cell_length is not None:
            cycles = [
                cycle for cycle in cycles if len(cycle) <= self.max_cell_length
            ]
        if len(cycles) != 0:
            cell_complex.add_cells_from(cycles, rank=self._complex_dim)

        return cell_complex
