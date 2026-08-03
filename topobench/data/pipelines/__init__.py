"""Configuration-driven data-pipeline boundaries."""

from topobench.data.capabilities import RuntimeDataCapability

from .base import (
    AbstractDataPipeline,
    CanonicalPredictionIdentity,
    DataPipelineOutput,
    PredictionRowAdapter,
    is_parquet_typed_graph_config,
)
from .default import DefaultDataPipeline
from .heterogeneous import HeterogeneousNodeDataPipeline
from .hypergraph import HypergraphNodeDataPipeline

__all__ = [
    "AbstractDataPipeline",
    "CanonicalPredictionIdentity",
    "DataPipelineOutput",
    "RuntimeDataCapability",
    "PredictionRowAdapter",
    "is_parquet_typed_graph_config",
    "DefaultDataPipeline",
    "HeterogeneousNodeDataPipeline",
    "HypergraphNodeDataPipeline",
]
