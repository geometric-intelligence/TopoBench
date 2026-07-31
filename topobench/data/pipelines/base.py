"""Shared interfaces for configuration-driven data pipelines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import hydra
from lightning import LightningDataModule
from omegaconf import DictConfig

from topobench.data.heterogeneous import HeterogeneousDataSpec
from topobench.data.preprocessor import PreProcessor


@dataclass(frozen=True)
class DataPipelineOutput:
    """Objects and runtime metadata produced by a data pipeline."""

    datamodule: LightningDataModule
    preprocessing_time: float
    data_spec: HeterogeneousDataSpec | None = None


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
