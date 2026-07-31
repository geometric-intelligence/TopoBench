"""Wrappers for the heterogeneous graph domain."""

from .heterogeneous_wrapper import HeterogeneousWrapper

WRAPPER_CLASSES = dict(
    sorted({HeterogeneousWrapper.__name__: HeterogeneousWrapper}.items())
)

globals().update(WRAPPER_CLASSES)

__all__ = [*WRAPPER_CLASSES, "WRAPPER_CLASSES"]
