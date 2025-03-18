"""TopoBench: A library for benchmarking of topological models."""

# Import submodules
from . import (
    data,
    dataloader,
    evaluator,
    loss,
    model,
    nn,
    transforms,
    utils,
)
from .run import initialize_hydra

__all__ = [
    "data",
    "dataloader",
    "evaluator",
    "initialize_hydra",
    "loss",
    "model",
    "nn",
    "transforms",
    "utils",
]


__version__ = "0.0.1"
