"""An transform that redifines simplicial complex neighbourhood."""

import torch_geometric


class RedefineSimplicialNeighbourhoods(
    torch_geometric.transforms.BaseTransform
):
    r"""An transform that redifines simplicial complex neighbourhood.

    Parameters
    ----------
    **kwargs : optional
        Parameters for the base transform.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.type = "RedefineSimplicialNeighbourhoods"
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
            The same data.
        """
        return data
