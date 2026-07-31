"""Shared interfaces for configuration-driven data pipelines."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from numbers import Real

import hydra
from lightning import LightningDataModule
from omegaconf import DictConfig

from topobench.data.heterogeneous import HeterogeneousDataSpec
from topobench.data.preprocessor import PreProcessor


@dataclass(frozen=True)
class DataPipelineOutput:
    """Validated objects and runtime metadata produced by a data pipeline.

    The frozen container prevents replacing its references. It does not freeze
    the Lightning data module itself, whose internal state remains mutable for
    framework lifecycle hooks. ``HeterogeneousDataSpec`` is independently
    immutable.
    """

    datamodule: LightningDataModule
    preprocessing_time: float
    data_spec: HeterogeneousDataSpec | None = None

    def __post_init__(self) -> None:
        """Validate and normalize values crossing the pipeline boundary."""
        if not isinstance(self.datamodule, LightningDataModule):
            raise TypeError("datamodule must be a LightningDataModule")
        if self.data_spec is not None and not isinstance(
            self.data_spec,
            HeterogeneousDataSpec,
        ):
            raise TypeError(
                "data_spec must be a HeterogeneousDataSpec or None"
            )
        if isinstance(self.preprocessing_time, bool) or not isinstance(
            self.preprocessing_time,
            Real,
        ):
            raise TypeError("preprocessing_time must be a real numeric scalar")

        preprocessing_time = float(self.preprocessing_time)
        if not math.isfinite(preprocessing_time):
            raise ValueError("preprocessing_time must be finite")
        if preprocessing_time < 0:
            raise ValueError("preprocessing_time must be non-negative")
        object.__setattr__(self, "preprocessing_time", preprocessing_time)


class AbstractDataPipeline(ABC):
    """Build a data module and its runtime data contract."""

    @staticmethod
    def preprocess(cfg: DictConfig) -> PreProcessor:
        """Load and preprocess a dataset using the configured transforms."""
        loader = hydra.utils.instantiate(cfg.dataset.loader)
        dataset, dataset_dir = loader.load()
        transforms = (
            hydra.utils.instantiate(cfg.transforms)
            if cfg.get("transforms") is not None
            else None
        )
        return PreProcessor(dataset, dataset_dir, transforms)

    @abstractmethod
    def build(self, cfg: DictConfig) -> DataPipelineOutput:
        """Build a Lightning data module and its runtime data contract."""
