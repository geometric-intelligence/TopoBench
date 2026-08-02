"""The existing homogeneous TopoBench data pipeline."""

from __future__ import annotations

from omegaconf import DictConfig

from topobench.data.capabilities import qualify_graph_dataset
from topobench.data.features import (
    prepare_graph_features,
    validate_qualified_graph_source,
)
from topobench.data.utils.split_utils import validate_split_type_qualification
from topobench.dataloader import GraphDataModule
from topobench.utils.config_resolvers import infer_in_channels

from .base import (
    AbstractDataPipeline,
    DataPipelineOutput,
    is_parquet_typed_graph_config,
)


class DefaultDataPipeline(AbstractDataPipeline):
    """Build the established split-based TopoBench data module."""

    def build(self, cfg: DictConfig) -> DataPipelineOutput:
        """Load, preprocess, split, and batch a homogeneous dataset."""
        if is_parquet_typed_graph_config(cfg):
            return self.build_parquet(
                cfg,
                expected_output_kind="homogeneous",
            )
        validate_split_type_qualification(
            cfg.dataset.split_params.get("split_type")
        )
        if cfg.dataset.parameters.task_level not in ["node", "graph"]:
            raise ValueError("Invalid task_level")

        preprocessor = self.preprocess(cfg)
        capability = None
        total_channels = None
        if cfg.dataset.loader.parameters.data_domain == "graph":
            capability = qualify_graph_dataset(cfg.dataset)
            total_channels = infer_in_channels(
                cfg.dataset,
                cfg.get("transforms"),
            )
            validate_qualified_graph_source(
                preprocessor,
                capability=capability,
                configured_num_classes=cfg.dataset.parameters.num_classes,
                total_num_features=total_channels,
            )

        train, val, test = preprocessor.load_dataset_splits(
            cfg.dataset.split_params
        )


        if cfg.dataset.loader.parameters.data_domain == "graph":
            assert capability is not None
            assert total_channels is not None
            base_channels = capability.stored_feature_width
            prepare_graph_features(
                train,
                val,
                test,
                feature_policy=cfg.dataset.parameters.feature_policy,
                base_num_features=base_channels,
                total_num_features=total_channels,
                categorical_cardinalities=capability.feature_cardinalities,
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
