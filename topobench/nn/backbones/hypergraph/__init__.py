"""Backbones for the hypergraph domain."""

from .edgnn import EDGNN
from .hypergraph_conv import HypergraphConvBackbone

BACKBONE_CLASSES = dict(
    sorted(
        {
            EDGNN.__name__: EDGNN,
            HypergraphConvBackbone.__name__: HypergraphConvBackbone,
        }.items()
    )
)

globals().update(BACKBONE_CLASSES)

__all__ = [*BACKBONE_CLASSES, "BACKBONE_CLASSES"]
