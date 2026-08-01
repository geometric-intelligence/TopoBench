"""Wrappers for the heterogeneous graph domain."""

from .heterogeneous_wrapper import HeterogeneousWrapper

WRAPPER_CLASSES = {
    "HeterogeneousWrapper": HeterogeneousWrapper,
}

__all__ = [
    "HeterogeneousWrapper",
    "WRAPPER_CLASSES",
]
