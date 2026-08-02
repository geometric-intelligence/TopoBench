"""Native heterogeneous node-classification data pipeline."""

from __future__ import annotations

from omegaconf import DictConfig
from torch_geometric.data import HeteroData

from topobench.data.capabilities import qualify_heterogeneous_dataset
from topobench.data.heterogeneous import validate_heterogeneous_node_data
from topobench.dataloader.heterogeneous import HeterogeneousNodeDataModule

from .base import AbstractDataPipeline, DataPipelineOutput


class HeterogeneousNodeDataPipeline(AbstractDataPipeline):
    """Build a separately batched native heterogeneous node data module."""

    def build(self, cfg: DictConfig) -> DataPipelineOutput:
        """Preprocess, validate, and batch exactly one heterogeneous graph."""
        qualification = qualify_heterogeneous_dataset(cfg.dataset)
        preprocessor = self.preprocess(cfg)
        if len(preprocessor) != 1:
            raise ValueError(
                "Heterogeneous node classification v1 requires exactly "
                f"one processed graph; received {len(preprocessor)}"
            )
        data = preprocessor[0]
        if not isinstance(data, HeteroData):
            raise TypeError(
                "The heterogeneous pipeline requires native HeteroData; "
                f"received {type(data).__name__}"
            )
        selector = qualification.selector
        spec = validate_heterogeneous_node_data(
            data,
            target_node_type=cfg.dataset.parameters.target_node_type,
            num_classes=cfg.dataset.parameters.num_classes,
            selector=selector,
        )
        datamodule = HeterogeneousNodeDataModule(
            data=data,
            spec=spec,
            **cfg.dataset.dataloader_params,
        )
        return DataPipelineOutput(
            datamodule=datamodule,
            preprocessing_time=preprocessor.preprocessing_time,
            data_spec=spec,
        )


__all__ = ["HeterogeneousNodeDataPipeline"]
