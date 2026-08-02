"""Native transductive hypergraph node-classification data pipeline."""

from __future__ import annotations

from omegaconf import DictConfig

from topobench.data.hypergraph import (
    HypergraphData,
    validate_hypergraph_node_data,
    validate_hypergraph_source,
)
from topobench.data.splits import validate_transductive_masks
from topobench.data.utils.split_utils import load_transductive_splits
from topobench.dataloader import GraphDataModule

from .base import AbstractDataPipeline, DataPipelineOutput


class HypergraphNodeDataPipeline(AbstractDataPipeline):
    """Build one native transductive hypergraph node-classification graph."""

    def build(self, cfg: DictConfig) -> DataPipelineOutput:
        """Preprocess, split, validate, and batch exactly one hypergraph."""
        preprocessor = self.preprocess(cfg)
        if len(preprocessor) != 1:
            raise ValueError(
                "Hypergraph node classification v1 requires exactly one "
                f"processed graph; received {len(preprocessor)}"
            )

        source_data = preprocessor[0]
        if not isinstance(source_data, HypergraphData):
            raise TypeError(
                "The hypergraph pipeline requires native HypergraphData; "
                f"received {type(source_data).__name__}"
            )

        split_params = cfg.dataset.split_params
        if split_params.get("learning_setting") != "transductive":
            raise ValueError(
                "Hypergraph node classification v1 requires transductive "
                "learning_setting"
            )

        parameters = cfg.dataset.get("parameters", {})
        num_classes = parameters.get("num_classes")
        loader = cfg.dataset.get("loader", {})
        selector = loader.get("parameters", {}).get(
            "data_name",
            cfg.dataset.get("selector", "<hypergraph>"),
        )
        validate_hypergraph_source(
            source_data,
            selector=selector,
            num_classes=num_classes,
        )
        runtime_data = source_data.clone()

        if split_params.get("split_type") == "fixed":
            validate_transductive_masks(runtime_data)

        train, val, test = load_transductive_splits(
            [runtime_data],
            split_params,
        )
        data = train[0]
        validate_hypergraph_node_data(
            data,
            selector=selector,
            num_classes=num_classes,
        )

        datamodule = GraphDataModule(
            dataset_train=train,
            dataset_val=val,
            dataset_test=test,
            learning_setting="transductive",
            **cfg.dataset.get("dataloader_params", {}),
        )
        return DataPipelineOutput(
            datamodule=datamodule,
            preprocessing_time=preprocessor.preprocessing_time,
        )


__all__ = ["HypergraphNodeDataPipeline"]
