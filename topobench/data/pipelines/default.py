"""The existing homogeneous TopoBench data pipeline."""

from __future__ import annotations

from omegaconf import DictConfig

from topobench.dataloader import GraphDataModule

from .base import AbstractDataPipeline, DataPipelineOutput


class DefaultDataPipeline(AbstractDataPipeline):
    """Build the established split-based TopoBench data module."""

    def build(self, cfg: DictConfig) -> DataPipelineOutput:
        """Load, preprocess, split, and batch a homogeneous dataset."""
        preprocessor = self.preprocess(cfg)
        train, val, test = preprocessor.load_dataset_splits(
            cfg.dataset.split_params
        )

        if cfg.dataset.parameters.task_level not in ["node", "graph"]:
            raise ValueError("Invalid task_level")

        datamodule = GraphDataModule(
            dataset_train=train,
            dataset_val=val,
            dataset_test=test,
            learning_setting=cfg.dataset.split_params.learning_setting,
            **cfg.dataset.get("dataloader_params", {}),
        )
        return DataPipelineOutput(
            datamodule=datamodule,
            preprocessing_time=float(preprocessor.preprocessing_time),
        )
