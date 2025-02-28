"""A transform that converts the node features of the input graph to float."""

import torch_geometric


class NodeFeaturesToFloat(torch_geometric.transforms.BaseTransform):
    r"""A transform that converts the node features of the input graph to float.

    Parameters
    ----------
    **kwargs : optional
        Parameters for the base transform.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.type = "map_node_features_to_float"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(type={self.type!r})"

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
        data.x = data.x.float()
        return data
