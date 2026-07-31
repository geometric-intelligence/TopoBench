"""Backbones for the hypergraph domain."""

from .edgnn import EDGNN

BACKBONE_CLASSES = dict(
    sorted({EDGNN.__name__: EDGNN}.items())
)

globals().update(BACKBONE_CLASSES)

__all__ = [*BACKBONE_CLASSES, "BACKBONE_CLASSES"]
