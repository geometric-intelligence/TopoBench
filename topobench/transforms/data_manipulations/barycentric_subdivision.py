"""Transform that performs barycentric subdivision on a simplicial complex."""

import torch
import torch_geometric
from topomodelx.utils.sparse import from_sparse
from toponetx import SimplicialComplex

from topobench.data.utils import data2simplicial
from topobench.data.utils.utils import get_complex_connectivity
from topomodelx.utils.sparse import from_sparse


# class BarycentricSubdivisionTransform(
#     torch_geometric.transforms.BaseTransform
# ):
#     r"""A transform that performs barycentric subdivision on a simplicial complex.

#     The barycentric subdivision of a simplicial complex K is a new simplicial complex Sd(K)
#     where each simplex in K is replaced by a collection of simplices, resulting in a finer
#     triangulation of the underlying space.

#     Parameters
#     ----------
#     **kwargs : optional
#         Parameters for the base transform.
#     """

#     def __init__(self, **kwargs):
#         super().__init__()
#         self.type = "domain2domain"
#         self.parameters = kwargs

#     def __repr__(self) -> str:
#         return f"{self.__class__.__name__}(type={self.type!r}, parameters={self.parameters!r})"

#     def forward(self, data: torch_geometric.data.Data):
#         r"""Apply the barycentric subdivision to the input data.

#         Parameters
#         ----------
#         data : torch_geometric.data.Data
#             The input data, expected to contain a simplicial complex.

#         Returns
#         -------
#         torch_geometric.data.Data
#             The data with a subdivided simplicial complex.
#         """
#         keys_to_keep = self.parameters["keys_to_keep"]
#         # Transform torch_geometric.data.Data to simplicial complex
#         simplicial_complex = data2simplicial(data)

#         # Apply barycentric subdivision
#         subdivided_complex, simplex_to_index = self._barycentric_subdivision(
#             simplicial_complex
#         )

#         lifted_topology = get_complex_connectivity(
#             subdivided_complex,
#             self.parameters["complex_dim"],
#             neighborhoods=self.parameters["neighborhoods"],
#             signed=self.parameters["signed"],
#         )

#         # Get rid of the old keys
#         for key, _ in data:
#             if key not in keys_to_keep:
#                 # if key in lifted_topology:
#                 #
#                 # assert torch.equal(
#                 #     data[key].indices(), lifted_topology[key].indices()
#                 # ), (
#                 #     f"Key {key} has different indices in data and lifted topology, although should be the same regardless sign True/False"
#                 # )
#                 data.pop(key)

#         # Assign new topology
#         for key in lifted_topology:
#             data[key] = lifted_topology[key]

#         zero_cells, one_cells = data.incidence_1.shape
#         two_cells, three_cells = data.incidence_3.shape

#         data["shape"] = torch.tensor(
#             [zero_cells, one_cells, two_cells, three_cells]
#         )
#         for idx, n in enumerate(data["shape"]):
#             if idx == 0:
#                 data[f"x"] = torch.ones((n, 1))
#             if n > 0:
#                 data[f"x_{idx}"] = torch.ones((n, 1))

#         data["edge_index"] = torch.Tensor(
#             from_sparse(subdivided_complex.adjacency_matrix(rank=0)).indices()
#         )

#         return data

#     def _barycentric_subdivision(self, K: SimplicialComplex) -> tuple:
#         """Perform barycentric subdivision on a simplicial complex.

#         Parameters
#         ----------
#         K : SimplicialComplex
#             The input simplicial complex.

#         Returns
#         -------
#         tuple
#             A tuple containing the subdivided simplicial complex and a mapping from
#             simplices to indices.
#         """
#         # Create a new SimplicialComplex to store the subdivision
#         Sd_K = SimplicialComplex()

#         # Check if K has the required attributes
#         if not hasattr(K, "simplices"):
#             raise AttributeError(
#                 "The simplicial complex must have a 'simplices' attribute"
#             )
#         if not hasattr(K, "dim"):
#             raise AttributeError(
#                 "The simplicial complex must have a 'dim' property"
#             )

#         new_simplices = {dim: set() for dim in range(K.dim + 1)}

#         # Add new vertices to Sd_K. Each simplex of Sd_K is a chain of simplices of K
#         for simplex in K.simplices:
#             new_simplices[0].add((simplex,))

#         # Give now an index to each simplex
#         simplex_to_index = {
#             simplex[0]: i for i, simplex in enumerate(new_simplices[0])
#         }

#         # Now, we add simplices from dimension 1 to K.dim
#         for dim in range(1, K.dim + 1):
#             # Get all simplices of the previous dimension, and try to add more simplices to the chain
#             previous_simplices = new_simplices[dim - 1]
#             for simplex_sub in previous_simplices:
#                 last_simplex = simplex_sub[-1]
#                 for simplex in K.simplices:
#                     # Check if simplex is a face of simplex_sub
#                     # Note: The '<' operator is assumed to check if last_simplex is a face of simplex
#                     if last_simplex < simplex:
#                         new_simplices[dim].add(simplex_sub + (simplex,))

#         # Now convert the simplices to indexes
#         all_simplices = []
#         for dim in range(K.dim + 1):
#             for simplex in new_simplices[dim]:
#                 all_simplices.append(
#                     [simplex_to_index[or_simplex] for or_simplex in simplex]
#                 )

#         # Add the simplices to the new SimplicialComplex
#         Sd_K.add_simplices_from(all_simplices)


#         return Sd_K, simplex_to_index
class BarycentricSubdivisionTransform(
    torch_geometric.transforms.BaseTransform
):
    r"""A transform that selectively performs barycentric subdivision on a simplicial complex.

    Parameters
    ----------
    subdivide_ranks : list of int
        Ranks (dimensions) of simplices to subdivide (e.g., [2] to subdivide only triangles).
    **kwargs : optional
        Parameters for the base transform.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.type = "domain2domain"
        self.parameters = kwargs
        self.apply_to = kwargs.get("apply_to", ["train", "val", "test"])
        subdivide_ranks = kwargs.get("subdivide_ranks", None)
        self.subdivide_ranks = (
            subdivide_ranks if subdivide_ranks is not None else []
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(type={self.type!r}, ranks={self.subdivide_ranks!r}, parameters={self.parameters!r})"

    def forward(self, data: torch_geometric.data.Data):
        if all(
            [
                getattr(data, f"{split}_mask").sum() == 0
                for split in self.apply_to
            ]
        ):
            return data

        keys_to_keep = self.parameters["keys_to_keep"]
        simplicial_complex = data2simplicial(data)

        subdivided_complex, simplex_to_index = barycentric_subdivision(
            simplicial_complex, self.subdivide_ranks
        )

        lifted_topology = get_complex_connectivity(
            subdivided_complex,
            self.parameters["complex_dim"],
            neighborhoods=self.parameters["neighborhoods"],
            signed=self.parameters["signed"],
        )

        for key, _ in data:
            if key not in keys_to_keep:
                data.pop(key)

        for key in lifted_topology:
            data[key] = lifted_topology[key]

        zero_cells, one_cells = data.incidence_1.shape
        two_cells, three_cells = data.incidence_3.shape

        data["shape"] = torch.tensor(
            [zero_cells, one_cells, two_cells, three_cells]
        )
        for idx, n in enumerate(data["shape"]):
            if idx == 0:
                data["x"] = torch.ones((n, 1))
            if n > 0:
                data[f"x_{idx}"] = torch.ones((n, 1))

        data["edge_index"] = torch.Tensor(
            from_sparse(subdivided_complex.adjacency_matrix(rank=0)).indices()
        )

        return data


def barycentric_subdivision(K: SimplicialComplex, ranks_to_subdivide):
    """
    Perform selective barycentric subdivision on specified ranks of simplices.

    Parameters:
    -----------
    K : SimplicialComplex
        The original simplicial complex
    ranks_to_subdivide : list
        List of dimensions/ranks to subdivide (e.g., [1, 2] for edges and triangles)

    Returns:
    --------
    Sd_K : SimplicialComplex
        The subdivided complex
    simplex_to_index : dict
        Mapping from original simplices to their barycenter indices
    """
    Sd_K = SimplicialComplex()

    # Get vertices properly - handle potential different implementations
    vertices = []
    try:
        # Try to get 0-simplices directly
        vertices = list(K.get_simplices(0))
    except (AttributeError, KeyError):
        try:
            # Alternative approach - filter by length
            vertices = [s for s in K.simplices if len(s) == 1]
        except:
            # Last resort - try to get nodes from the complex
            vertices = [(v,) for v in K.nodes]

    # First, add all original vertices to the new complex
    for v in vertices:
        Sd_K.add_simplex(v)

    # Get the highest vertex index to start numbering barycenters
    try:
        next_index = max([v[0] for v in vertices]) + 1
    except:
        # If vertices aren't accessible as tuples, try different approach
        try:
            next_index = max(K.nodes) + 1
        except:
            next_index = len(vertices)

    # Map to store simplex to barycenter index
    simplex_to_index = {}

    # Map vertices to their original indices
    for v in vertices:
        if isinstance(v, tuple) and len(v) == 1:
            simplex_to_index[v] = v[0]
        else:
            simplex_to_index[v] = v

    # Get all simplices by dimension - use dim instead of dimension
    all_simplices = []
    for dim in range(K.dim + 1):
        try:
            dim_simplices = list(K.get_simplices(dim))
            all_simplices.extend(dim_simplices)
        except (AttributeError, KeyError):
            # Handle cases where get_simplices might not be available
            pass

    # If we couldn't get simplices by dimension, try using the simplices attribute directly
    if not all_simplices:
        try:
            all_simplices = list(K.simplices)
        except:
            # Last resort - try to reconstruct from available data
            for dim in range(1, K.dim + 1):
                try:
                    all_simplices.extend(getattr(K, f"_{dim}_simplices", []))
                except:
                    pass

    # Process simplices by dimension for consecutive indexing
    for dim in range(1, K.dim + 1):  # Skip vertices (dim=0)
        # Filter simplices of this dimension
        dim_simplices = [s for s in all_simplices if len(s) - 1 == dim]

        for simplex in dim_simplices:
            # Assign consecutive index
            simplex_to_index[simplex] = next_index

            # If we're subdividing this rank, add the barycenter as a vertex
            if dim in ranks_to_subdivide:
                Sd_K.add_simplex((next_index,))

            # Increment index for next simplex
            next_index += 1

    # Process each simplex for subdivision
    for simplex in all_simplices:
        dim = len(simplex) - 1

        # If not subdividing this rank, keep the original simplex
        if dim not in ranks_to_subdivide:
            if dim > 0:  # Skip vertices as they're already added
                Sd_K.add_simplex(simplex)
            continue

        # For simplices we're subdividing
        if dim == 1:  # Edge case: subdividing an edge
            v1, v2 = simplex
            bc = simplex_to_index[simplex]
            # Create two new edges
            Sd_K.add_simplex((v1, bc))
            Sd_K.add_simplex((v2, bc))

        elif dim == 2:  # Triangle case: subdividing a triangle
            # Get vertices and barycenter
            v1, v2, v3 = simplex
            bc = simplex_to_index[simplex]

            # Create edges from vertices to barycenter
            Sd_K.add_simplex((v1, bc))
            Sd_K.add_simplex((v2, bc))
            Sd_K.add_simplex((v3, bc))

            # Create new triangles (connect two original vertices with barycenter)
            Sd_K.add_simplex((v1, v2, bc))
            Sd_K.add_simplex((v2, v3, bc))
            Sd_K.add_simplex((v3, v1, bc))

        elif dim == 3:  # Tetrahedron case
            v1, v2, v3, v4 = simplex
            bc = simplex_to_index[simplex]

            # Connect vertices to barycenter
            Sd_K.add_simplex((v1, bc))
            Sd_K.add_simplex((v2, bc))
            Sd_K.add_simplex((v3, bc))
            Sd_K.add_simplex((v4, bc))

            # Connect each triangle face with barycenter to form tetrahedra
            Sd_K.add_simplex((v1, v2, v3, bc))
            Sd_K.add_simplex((v1, v2, v4, bc))
            Sd_K.add_simplex((v1, v3, v4, bc))
            Sd_K.add_simplex((v2, v3, v4, bc))

    return Sd_K, simplex_to_index
