r"""This module implements the neighborhood complex lifting.

Given that connections based on neighbourhoods of nodes are already
present in GNN literature, the notion of a *neighbourhood complex* becomes
of interest.
Following from previous work, defining a neighbourhoods as the nodes with
which a node shares a neighbour.
Formally, :math:`N(G)` is defined in terms of a simplex.
Where a simplex math:`\sigma v` is the neighbourhood simplex of node
`v \in V(G)`, composed of all :math:`u \in V(G)` given that
:math:`\exists w:(v, w) \in E(G) \wedge(u, w) \in E(G)` [L78]_.

This structure has been proven to have certain properties that could be
interesting in certain domains such as :math:`k`-colorability.
As stablished by Lovasz ([L78]_), if :math:`N(G)` is :math:`(k + 2)`-connected,
then, :math:`G` is not :math:`k`-colorable.
Additionally, [L78]_ shows a relationship between the homotopy invariance of
:math:`N(G)` and the :math:`k`-colorability of :math:`G`.

*Neighbourhood complexes* can be used to calculate other more interesting structures
in induced by graphs, such as the *dominating set* of :math:`G` which is the
*Alexander dual* of :math:`N(G^-)` (neighbourhood complex of the complement of
:math:`G`).
This is useful for computing homology groups of *dominance complexes* without having to
actually calculated the dominance set [MW25]_.
In future implementations, adding a basic transformation pertaining to the
*Alexander Dual* would help in having a *Dominating Complex*, namely,
a simplicial complex composed of simplices where the complements of the nodes
composing the simplifies are dominating in G .

References
----------
.. [L78] Lovász, L. “Kneser's Conjecture, Chromatic Number, and Homotopy.”
    Journal of Combinatorial Theory, Series A 25, no. 3 (November 1, 1978): 319-24.
    https://doi.org/10.1016/0097-3165(78)90022-5.
.. [MW25] Matsushita, T., Wakatsuki, S., 2025. Dominance complexes, neighborhood complexes
    and combinatorial Alexander duals.
    Journal of Combinatorial Theory, Series A 211, 105978.
    https://doi.org/10.1016/j.jcta.2024.105978
"""

from toponetx.classes import SimplicialComplex

from topobenchmark.transforms.liftings.base import LiftingMap


class NeighborhoodComplexLifting(LiftingMap):
    """Lifts graphs to a simplicial complex domain.

    Identifies the neighborhood complex as k-simplices.
    The neighborhood complex of a node u is the set of
    nodes that share a neighbor with u.

    Parameters
    ----------
    complex_dim : int
        Dimension of the subcomplex.
    """

    def __init__(self, complex_dim=2):
        super().__init__()
        self.complex_dim = complex_dim

    def lift(self, domain):
        """Lift the topology of a graph to a simplicial complex.

        Parameters
        ----------
        domain : nx.Graph
            Graph to be lifted.

        Returns
        -------
        toponetx.SimplicialComplex
            Lifted simplicial complex.
        """
        graph = domain

        simplicial_complex = SimplicialComplex(simplices=graph)

        for u in graph.nodes:
            neighbourhood_complex = set()
            neighbourhood_complex.add(u)

            # Check it's neighbours
            for v in graph.neighbors(u):
                for w in graph.nodes:
                    if w in (u, v):
                        continue

                    # w and u share v as it's neighbour
                    if v in graph.neighbors(w):
                        neighbourhood_complex.add(w)

            if (
                len(neighbourhood_complex) < 2  # Do not add 0-simplices
                or len(neighbourhood_complex)
                > self.complex_dim
                + 1  # Do not add i-simplices if the maximum dimension is lower
            ):
                continue

            simplicial_complex.add_simplex(neighbourhood_complex)

        feature_dict = {
            node: attrs["x"] for node, attrs in graph.nodes(data=True)
        }
        simplicial_complex.set_simplex_attributes(feature_dict, name="x")

        # because ComplexData pads unexisting dimensions with empty matrices
        simplicial_complex.practical_dim = self.complex_dim

        return simplicial_complex
