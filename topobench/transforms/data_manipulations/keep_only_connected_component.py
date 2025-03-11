"""KeepOnlyConnectedComponent class definition."""

import torch_geometric
from torch_geometric.transforms import LargestConnectedComponents


class KeepOnlyConnectedComponent(torch_geometric.transforms.BaseTransform):
    """Class to keep only the largest connected components of the input graph.

    Parameters
    ----------
    **kwargs : optional
        Parameters for the base transform.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.type = "keep_connected_component"
        self.parameters = kwargs

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(type={self.type!r}, parameters={self.parameters!r})"

    def forward(self, data: torch_geometric.data.Data):
        """Apply the transform to the input data.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data.

        Returns
        -------
        torch_geometric.data.Data
            The transformed data.
        """
        # torch_geometric.transforms.largest_connected_components()
        num_components = self.parameters["num_components"]
        lcc = LargestConnectedComponents(
            num_components=num_components, connection="strong"
        )
        data = lcc(data)
        return data
