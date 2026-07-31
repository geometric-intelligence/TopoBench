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
from .domains import SUPPORTED_DOMAINS, require_supported_domain
from .run import initialize_hydra

__all__ = [
    "SUPPORTED_DOMAINS",
    "data",
    "dataloader",
    "evaluator",
    "initialize_hydra",
    "loss",
    "model",
    "nn",
    "require_supported_domain",
    "transforms",
    "utils",
]


__version__ = "0.0.1"
