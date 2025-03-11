"""The CofaceCCLifting lifting."""

from toponetx.classes.combinatorial_complex import CombinatorialComplex
from toponetx.classes.hyperedge import HyperEdge
from torch_geometric.data import Data

from topobench.data.utils.utils import (
    get_combinatorial_complex_connectivity,
)
from topobench.transforms.liftings.simplicial2combinatorial.base import (
    Simplicial2CombinatorialLifting,
)


class CofaceCCLifting(Simplicial2CombinatorialLifting):
    """The CofaceCCLifting class.

    This class lifts a simplicial complex to a combinatorial complex
    by using the coface relation between the simplicial cells.

    Parameters
    ----------
    **kwargs : dict
        The keyword arguments.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.keep_features = kwargs.get("keep_features", False)

    def get_lower_cells(self, data: Data) -> list[HyperEdge]:
        """Get the lower cells of the complex.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data.

        Returns
        -------
        list
            The list of lower cells.
        """
        cells: list[HyperEdge] = []

        ## Add 0-cells
        for cell in range(data["x_0"].size(0)):
            zero_cell = HyperEdge([cell], rank=0)
            cells.append(zero_cell)

        ## Add 1-cells
        for inc_c_1 in data["incidence_1"].to_dense().T:
            # Get the 0-cells that are incident to the 1-cell
            cell_0_bound = inc_c_1.nonzero().flatten().tolist()
            assert len(cell_0_bound) == 2
            one_cell = HyperEdge(cell_0_bound, rank=1)
            cells.append(one_cell)

        ## Add 2-cells
        for inc_c_2 in data["incidence_2"].to_dense().T:
            # Get the 1-cells that are incident to the 2-cell
            cell_1_bound = inc_c_2.nonzero().flatten()
            # Get the 0-cells that are incident to the 1-cells
            cell_0_bound = (
                data["incidence_1"].to_dense().T[cell_1_bound].nonzero()
            )
            # Get the actual 0-cells since nonzero()
            # indexes in 2D
            cell_0_bound = cell_0_bound[:, 1]
            # Remove redudants and convert to tuple
            two_cell = HyperEdge(tuple(set(cell_0_bound.tolist())), rank=2)
            cells.append(two_cell)

        return cells

    def lift_topology(self, data: Data) -> dict:
        """Lift the simplicial topology to a combinatorial complex.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data.

        Returns
        -------
        dict
            The lifted connectivity dict.
        """

        # Check that the dataset has the required fields
        # assume that it's a simplicial dataset
        assert "incidence_1" in data
        assert "incidence_2" in data

        cells = self.get_lower_cells(data)

        ccc = CombinatorialComplex(cells, graph_based=False)

        # Iterate over the 2-cells and add the 3-cells
        for r_cell in ccc.skeleton(rank=2):
            # Get the coface of the 2-cell
            indices, coface = ccc.coadjacency_matrix(2, 1, index=True)

            # Get the indices of the 2-cell that are co-adjacent
            coface_indices = (
                coface.todense()[indices[r_cell]].nonzero()[1].tolist()
            )
            cell_3 = set(r_cell)

            # Iterate over the indices of the 2-cells
            # and add their 0-cells as a 3-cell
            for idx in coface_indices:
                cell_3 = cell_3.union(set(ccc.skeleton(rank=2)[idx]))

            # Adding a rank 3 cell with less than 4 vertices
            # will take this cell from the skeleton of 2-cells if it exists
            # so in the interest of keeping features the user
            # can choose to recompute all feature embeddings
            if len(cell_3) < 4 and self.keep_features:
                continue
            # Get the cofaces incident to the 2-cell `cell` and add `cell` to the set
            ccc.add_cell(cell_3, rank=3)

        # Create the incidence, adjacency and laplacian matrices
        lifted_data = get_combinatorial_complex_connectivity(ccc, 3)

        # If the user wants to keep the features
        # from the r-cells aside from the first x_0
        if self.keep_features:
            lifted_data = {
                "x_0": data["x_0"],
                "x_1": data["x_1"],
                "x_2": data["x_2"],
                **lifted_data,
            }
        else:
            lifted_data = {"x_0": data["x_0"], **lifted_data}

        return lifted_data

    def forward(self, data: Data) -> Data:
        """Forward pass.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data.

        Returns
        -------
        torch_geometric.data.Data
            The updated lifted data.
        """
        initial_data = data.to_dict()
        lifted_topology = self.lift_topology(data)
        lifted_topology = self.feature_lifting(lifted_topology)

        # Make sure to remove passing of duplicated data
        # so that the constructor of Data does not raise an error

        for k in lifted_topology:
            if k in initial_data:
                del initial_data[k]
        return Data(**initial_data, **lifted_topology)
