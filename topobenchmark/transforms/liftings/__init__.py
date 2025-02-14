"""This module implements the liftings for the topological transforms."""

from .base import (  # noqa: F401
    Graph2CellLiftingTransform,
    Graph2ComplexLiftingTransform,
    Graph2HypergraphLiftingTransform,
    Graph2SimplicialLiftingTransform,
    LiftingMap,
    LiftingTransform,
)
from .graph2cell import GRAPH2CELL_LIFTINGS
from .graph2hypergraph import GRAPH2HYPERGRAPH_LIFTINGS
from .graph2simplicial import GRAPH2SIMPLICIAL_LIFTINGS

LIFTINGS = {
    **GRAPH2CELL_LIFTINGS,
    **GRAPH2HYPERGRAPH_LIFTINGS,
    **GRAPH2SIMPLICIAL_LIFTINGS,
}


locals().update(LIFTINGS)
