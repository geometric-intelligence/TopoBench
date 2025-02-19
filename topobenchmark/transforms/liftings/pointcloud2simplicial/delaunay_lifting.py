import torch
import torch_geometric
from scipy.spatial import Delaunay
from toponetx.classes import SimplicialComplex

from modules.data.utils.utils import get_complex_connectivity
from modules.transforms.liftings.pointcloud2simplicial.base import (
    PointCloud2SimplicialLifting,
)


class DelaunayLifting(PointCloud2SimplicialLifting):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _get_lifted_topology(
        self, simplicial_complex: SimplicialComplex
    ) -> dict:
        r"""Returns the lifted topology.

        Parameters
        ----------
        simplicial_complex : SimplicialComplex
            The simplicial complex.
        Returns
        ---------
        dict
            The lifted topology.
        """
        lifted_topology = get_complex_connectivity(
            simplicial_complex, self.complex_dim
        )
        lifted_topology["x_0"] = torch.stack(
            list(
                simplicial_complex.get_simplex_attributes(
                    "features", 0
                ).values()
            )
        )

        return lifted_topology

    def lift_topology(self, data: torch_geometric.data.Data, **kwargs) -> dict:
        tri = Delaunay(data.pos, **kwargs)
        simplicial_complex = SimplicialComplex(tri.simplices)
        self.complex_dim = simplicial_complex.dim

        node_features = {i: data.x[i, :] for i in range(data.x.shape[0])}
        simplicial_complex.set_simplex_attributes(
            node_features, name="features"
        )

        return self._get_lifted_topology(simplicial_complex)
