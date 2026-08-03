"""Neural-network components shared by TopoBench models."""

from topobench.nn.activation import make_activation
from topobench.nn.capabilities import (
    GRAPH_MODEL_CAPABILITIES,
    MODEL_CAPABILITY_MANIFEST,
    CapabilityValidation,
    GraphModelCapability,
    ModelCapability,
    validate_capability_composition,
    validate_graph_composition,
)

__all__ = [
    "CapabilityValidation",
    "GRAPH_MODEL_CAPABILITIES",
    "MODEL_CAPABILITY_MANIFEST",
    "GraphModelCapability",
    "ModelCapability",
    "make_activation",
    "validate_capability_composition",
    "validate_graph_composition",
]
