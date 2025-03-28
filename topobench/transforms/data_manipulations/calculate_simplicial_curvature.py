"""A transform that calculates the simplicial curvature of the input graph."""

import torch
import torch_geometric


class CalculateSimplicialCurvature(torch_geometric.transforms.BaseTransform):
    r"""A transform that calculates the simplicial curvature of the input graph.

    Parameters
    ----------
    **kwargs : optional
        Parameters for the transform.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.type = "simplicial_curvature"
        self.parameters = kwargs

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(type={self.type!r}, parameters={self.parameters!r})"

    def forward(self, data: torch_geometric.data.Data):
        r"""Apply the transform to the input data.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data.

        Returns
        -------
        torch_geometric.data.Data
            The transformed data.
        """
        data = self.one_cell_curvature(data)
        data = self.zero_cell_curvature(data)
        data = self.two_cell_curvature(data)
        return data

    def zero_cell_curvature(
        self,
        data: torch_geometric.data.Data,
    ) -> torch_geometric.data.Data:
        r"""Calculate the zero cell curvature of the input data.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data.

        Returns
        -------
        torch_geometric.data.Data
            Data with the zero cell curvature.
        """
        data["0_cell_curvature"] = torch.mm(
            abs(data["incidence_1"]), data["1_cell_curvature"]
        )
        return data

    def one_cell_curvature(
        self,
        data: torch_geometric.data.Data,
    ) -> torch_geometric.data.Data:
        r"""Calculate the one cell curvature of the input data.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data.

        Returns
        -------
        torch_geometric.data.Data
            Data with the one cell curvature.
        """
        data["1_cell_curvature"] = (
            4
            - torch.mm(abs(data["incidence_1"]).T, data["0_cell_degrees"])
            + 3 * data["1_cell_degrees"]
        )
        return data

    def two_cell_curvature(
        self,
        data: torch_geometric.data.Data,
    ) -> torch_geometric.data.Data:
        r"""Calculate the two cell curvature of the input data.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data.

        Returns
        -------
        torch_geometric.data.Data
            Data with the two cell curvature.
        """
        # Term 1 is simply the degree of the 2-cell (i.e. each triangle belong to n tetrahedrons)
        term1 = data["2_cell_degrees"]
        # Find triangles that belong to multiple tetrahedrons
        two_cell_degrees = data["2_cell_degrees"].clone()
        idx = torch.where(data["2_cell_degrees"] > 1)[0]
        two_cell_degrees[idx] = 0
        up = data["incidence_3"].to_dense() @ data["incidence_3"].to_dense().T
        down = (
            data["incidence_2"].to_dense().T @ data["incidence_2"].to_dense()
        )
        mask = torch.eye(up.size()[0]).bool()
        up.masked_fill_(mask, 0)
        down.masked_fill_(mask, 0)
        diff = (down - up) * 1
        term2 = diff.sum(1, keepdim=True)
        data["2_cell_curvature"] = 3 + term1 - term2
        return data
