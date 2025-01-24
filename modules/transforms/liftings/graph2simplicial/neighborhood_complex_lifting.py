import networkx as nx
import torch
from toponetx.classes import SimplicialComplex
from torch_geometric.data import Data
from torch_geometric.utils.convert import to_networkx

from modules.transforms.liftings.graph2simplicial.base import Graph2SimplicialLifting


class NeighborhoodComplexLifting(Graph2SimplicialLifting):
    """ Lifts graphs to a simplicial complex domain by identifying the neighborhood complex as k-simplices.
        The neighborhood complex of a node u is the set of nodes that share a neighbor with u.

    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def lift_topology(self, data: Data) -> dict:
        graph: nx.Graph = to_networkx(data, to_undirected=True)
        simplicial_complex = SimplicialComplex(simplices=graph)

        # For every node u
        for u in graph.nodes:
            neighbourhood_complex = set()
            neighbourhood_complex.add(u)
            # Check it's neighbours
            for v in graph.neighbors(u):
                # For every other node w != u ^ w != v
                for w in  graph.nodes:
                    # w == u
                    if w == u:
                        continue
                    # w == v
                    if w == v:
                        continue

                    # w and u share v as it's neighbour
                    if v in graph.neighbors(w):
                        neighbourhood_complex.add(w)
            # Do not add 0-simplices
            if len(neighbourhood_complex) < 2:
                continue
            # Do not add i-simplices if the maximum dimension is lower
            if len(neighbourhood_complex) > self.complex_dim + 1:
                continue
            simplicial_complex.add_simplex(neighbourhood_complex)

        feature_dict = {
            i: torch.zeros(data["x"].size(1)) for i in range(data["x"].size(0))
        }

        simplicial_complex.set_simplex_attributes(feature_dict, name="features")

        return self._get_lifted_topology(simplicial_complex, graph)

    def _get_lifted_topology(self, simplicial_complex: SimplicialComplex, graph: nx.Graph) -> dict:
        data = super()._get_lifted_topology(simplicial_complex, graph)

        for r in range(simplicial_complex.dim):
            data[f"x_idx_{r}"] = torch.tensor(simplicial_complex.skeleton(r))

        return data
