"""Wrappers for the hypergraph domain."""

from .hypergraph_wrapper import HypergraphWrapper

WRAPPER_CLASSES = dict(
    sorted({HypergraphWrapper.__name__: HypergraphWrapper}.items())
)

globals().update(WRAPPER_CLASSES)

__all__ = [*WRAPPER_CLASSES, "WRAPPER_CLASSES"]
