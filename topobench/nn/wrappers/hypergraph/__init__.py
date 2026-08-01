"""Wrappers for the hypergraph domain."""

from .hypergraph_wrapper import HypergraphWrapper

WRAPPER_CLASSES = {
    "HypergraphWrapper": HypergraphWrapper,
}

__all__ = [
    "HypergraphWrapper",
    "WRAPPER_CLASSES",
]
