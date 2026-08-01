"""Backbones exposed by the supported data domains."""

from .graph import GCNDGM, GPSEncoder, GraphMLP, NSDEncoder
from .heterogeneous import HeteroSAGEBackbone, HGTBackbone
from .hypergraph import EDGNN, HypergraphConvBackbone

MODEL_CLASSES = {
    "EDGNN": EDGNN,
    "GCNDGM": GCNDGM,
    "GPSEncoder": GPSEncoder,
    "GraphMLP": GraphMLP,
    "HGTBackbone": HGTBackbone,
    "HeteroSAGEBackbone": HeteroSAGEBackbone,
    "HypergraphConvBackbone": HypergraphConvBackbone,
    "NSDEncoder": NSDEncoder,
}

__all__ = [
    "EDGNN",
    "GCNDGM",
    "GPSEncoder",
    "GraphMLP",
    "HGTBackbone",
    "HeteroSAGEBackbone",
    "HypergraphConvBackbone",
    "NSDEncoder",
    "MODEL_CLASSES",
]
