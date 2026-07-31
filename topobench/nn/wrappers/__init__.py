"""Wrappers exposed by the supported data domains."""

from .graph import WRAPPER_CLASSES as GRAPH_WRAPPERS
from .heterogeneous import WRAPPER_CLASSES as HETEROGENEOUS_WRAPPERS
from .hypergraph import WRAPPER_CLASSES as HYPERGRAPH_WRAPPERS

WRAPPER_CLASSES = dict(
    sorted(
        {
            **GRAPH_WRAPPERS,
            **HETEROGENEOUS_WRAPPERS,
            **HYPERGRAPH_WRAPPERS,
        }.items()
    )
)

globals().update(WRAPPER_CLASSES)

__all__ = [*WRAPPER_CLASSES, "WRAPPER_CLASSES"]
