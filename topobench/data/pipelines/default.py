"""The existing homogeneous TopoBench data pipeline."""

from __future__ import annotations

from omegaconf import DictConfig

from topobench.data.capabilities import qualify_graph_dataset
from topobench.data.features import prepare_graph_features
from topobench.data.utils.split_utils import validate_split_type_qualification
from topobench.dataloader import GraphDataModule
from topobench.utils.config_resolvers import infer_in_channels

from .base import AbstractDataPipeline, DataPipelineOutput


class DefaultDataPipeline(AbstractDataPipeline):
    """Build the established split-based TopoBench data module."""

    def build(self, cfg: DictConfig) -> DataPipelineOutput:
        """Load, preprocess, split, and batch a homogeneous dataset."""
        validate_split_type_qualification(
            cfg.dataset.split_params.get("split_type")
        )
        preprocessor = self.preprocess(cfg)
        train, val, test = preprocessor.load_dataset_splits(
            cfg.dataset.split_params
        )

        if cfg.dataset.parameters.task_level not in ["node", "graph"]:
            raise ValueError("Invalid task_level")

        if cfg.dataset.loader.parameters.data_domain == "graph":
            base_channels = qualify_graph_dataset(
                cfg.dataset
            ).feature_width
            total_channels = infer_in_channels(
                cfg.dataset,
                cfg.get("transforms"),
            )
            prepare_graph_features(
                train,
                val,
                test,
                feature_policy=cfg.dataset.parameters.feature_policy,
                base_num_features=base_channels,
                total_num_features=total_channels,
            )

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
