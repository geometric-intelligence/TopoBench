"""Neural-network components shared by TopoBench models."""

from topobench.nn.activation import make_activation
from topobench.nn.capabilities import (
    GRAPH_MODEL_CAPABILITIES,
    GraphModelCapability,
    validate_graph_composition,
)

__all__ = [
    "GRAPH_MODEL_CAPABILITIES",
    "GraphModelCapability",
    "make_activation",
    "validate_graph_composition",
]
