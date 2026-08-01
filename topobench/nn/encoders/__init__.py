"""Explicit public feature-encoder registry."""

from .dgm_encoder import DGMStructureFeatureEncoder
from .graph_node_encoder import GraphNodeFeatureEncoder
from .heterogeneous_node_encoder import HeterogeneousNodeFeatureEncoder

FEATURE_ENCODERS = {
    "DGMStructureFeatureEncoder": DGMStructureFeatureEncoder,
    "GraphNodeFeatureEncoder": GraphNodeFeatureEncoder,
    "HeterogeneousNodeFeatureEncoder": HeterogeneousNodeFeatureEncoder,
}

__all__ = [
    "DGMStructureFeatureEncoder",
    "GraphNodeFeatureEncoder",
    "HeterogeneousNodeFeatureEncoder",
    "FEATURE_ENCODERS",
]
