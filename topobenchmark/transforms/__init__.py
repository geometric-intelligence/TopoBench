"""This module contains the transforms for the topobenchmark package."""

from .data_manipulations import DATA_MANIPULATIONS
from .feature_liftings import FEATURE_LIFTINGS
from .liftings import (
    GRAPH2CELL_LIFTINGS,
    GRAPH2HYPERGRAPH_LIFTINGS,
    GRAPH2SIMPLICIAL_LIFTINGS,
    LIFTINGS,
)

TRANSFORMS = {
    **LIFTINGS,
    **FEATURE_LIFTINGS,
    **DATA_MANIPULATIONS,
}


_map_lifting_type_to_dict = {
    "graph2cell": GRAPH2CELL_LIFTINGS,
    "graph2hypergraph": GRAPH2HYPERGRAPH_LIFTINGS,
    "graph2simplicial": GRAPH2SIMPLICIAL_LIFTINGS,
}


def add_lifting_map(LiftingMap, lifting_type, name=None):
    if name is None:
        name = LiftingMap.__name__

    liftings_dict = _map_lifting_type_to_dict[lifting_type]

    for dict_ in (liftings_dict, LIFTINGS, TRANSFORMS):
        dict_[name] = LiftingMap
