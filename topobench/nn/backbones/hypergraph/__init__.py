"""Backbones for the hypergraph domain."""

from .edgnn import EDGNN
from .hypergraph_conv import HypergraphConvBackbone

BACKBONE_CLASSES = {
    "EDGNN": EDGNN,
    "HypergraphConvBackbone": HypergraphConvBackbone,
}

__all__ = [
    "EDGNN",
    "HypergraphConvBackbone",
    "BACKBONE_CLASSES",
]
