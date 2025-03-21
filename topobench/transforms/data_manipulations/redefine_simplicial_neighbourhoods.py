"""An transform that redifines simplicial complex neighbourhood."""

import torch
import torch_geometric

from topobench.data.utils import data2simplicial
from topobench.data.utils.utils import get_complex_connectivity


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

        keys_to_keep = self.parameters["keys_to_keep"]
        simplicial_complex = data2simplicial(data)

        lifted_topology = get_complex_connectivity(
            simplicial_complex,
            self.parameters["complex_dim"],
            neighborhoods=self.parameters["neighborhoods"],
            signed=self.parameters["signed"],
        )

        # Get rid of the old keys
        for key, _ in data:
            if key not in keys_to_keep:
                if key in lifted_topology:
                    assert torch.equal(
                        data[key].indices(), lifted_topology[key].indices()
                    ), (
                        f"Key {key} has different indices in data and lifted topology, although should be the same regardless sign True/False"
                    )
                data.pop(key)

        # Assign new topology
        for key in lifted_topology:
            data[key] = lifted_topology[key]

        return data
