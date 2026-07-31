"""Backbones exposed by the supported data domains."""

from .graph import BACKBONE_CLASSES as GRAPH_BACKBONES
from .heterogeneous import BACKBONE_CLASSES as HETEROGENEOUS_BACKBONES
from .hypergraph import BACKBONE_CLASSES as HYPERGRAPH_BACKBONES

MODEL_CLASSES = dict(
    sorted(
        {
            **GRAPH_BACKBONES,
            **HETEROGENEOUS_BACKBONES,
            **HYPERGRAPH_BACKBONES,
        }.items()
    )
)

globals().update(MODEL_CLASSES)

__all__ = [*MODEL_CLASSES, "MODEL_CLASSES"]
