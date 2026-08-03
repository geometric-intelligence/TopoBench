"""Native heterogeneous node-classification data pipeline."""

from __future__ import annotations

from types import MappingProxyType

import torch
from omegaconf import DictConfig
from torch_geometric.data import HeteroData

from topobench.data.capabilities import RuntimeDataCapability, qualify_dataset
from topobench.data.heterogeneous import validate_heterogeneous_node_data
from topobench.dataloader.heterogeneous import HeterogeneousNodeDataModule

from .base import (
    AbstractDataPipeline,
    DataPipelineOutput,
    _native_provenance_fingerprints,
    is_parquet_typed_graph_config,
    native_prediction_row_adapter,
)


class HeterogeneousNodeDataPipeline(AbstractDataPipeline):
    """Build a separately batched native heterogeneous node data module."""

    def build(self, cfg: DictConfig) -> DataPipelineOutput:
        """Preprocess, validate, and batch exactly one heterogeneous graph."""
        if is_parquet_typed_graph_config(cfg):
            return self.build_parquet(
                cfg,
                expected_output_kind="heterogeneous",
            )
        qualification = qualify_dataset(cfg.dataset)
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
        for node_type in data.node_types:
            store = data[node_type]
            node_count = int(store.num_nodes)
            n_id = store.get("n_id")
            if n_id is None:
                store.n_id = torch.arange(node_count, dtype=torch.long)
            elif (
                not isinstance(n_id, torch.Tensor)
                or n_id.dtype == torch.bool
                or n_id.is_floating_point()
                or n_id.is_complex()
                or n_id.ndim != 1
                or n_id.numel() != node_count
                or bool(torch.any(n_id < 0))
                or torch.unique(n_id).numel() != node_count
            ):
                raise ValueError(
                    f"{node_type}.n_id must contain unique non-negative "
                    "source ordinals aligned to nodes"
                )
        target_store = data[spec.target_node_type]
        capability_spec = RuntimeDataCapability(
            selector=qualification.selector,
            data_domain="heterogeneous",
            output_kind="heterogeneous",
            feature_widths=spec.input_channels,
            num_classes=int(target_store.y.max().item()) + 1,
            target_node_type=spec.target_node_type,
        )
        supervision_counts = {
            phase: int(target_store[f"{phase}_mask"].count_nonzero().item())
            for phase in ("train", "val", "test")
        }
        mode = str(cfg.dataset.dataloader_params.get("mode", "full_batch"))
        source_graph_id, prediction_adapter = native_prediction_row_adapter(
            cfg,
            output_kind="heterogeneous",
            target_node_type=spec.target_node_type,
            sampling_strategy=f"heterogeneous-{mode.replace('_batch', '')}",
        )
        datamodule = HeterogeneousNodeDataModule(
            data=data,
            spec=spec,
            **cfg.dataset.dataloader_params,
        )
        fingerprints = _native_provenance_fingerprints(
            {"train": (data,), "val": None, "test": None},
            output_kind="heterogeneous",
            target_node_type=spec.target_node_type,
        )
        split_declaration = MappingProxyType(
            {
                str(key): value
                for key, value in cfg.dataset.split_params.items()
                if key != "learning_setting"
            }
        )
        provenance = MappingProxyType(
            {
                "source_graph_id": source_graph_id,
                "split_declaration": split_declaration,
                "learning_setting": str(
                    cfg.dataset.split_params.learning_setting
                ),
                "task": prediction_adapter.task,
                "data_domain": str(cfg.dataset.loader.parameters.data_domain),
                "output_kind": "heterogeneous",
                "supervision_counts": MappingProxyType(
                    dict(supervision_counts)
                ),
                "qualification_selector": qualification.selector,
                **fingerprints,
            }
        )
        return DataPipelineOutput(
            datamodule=datamodule,
            preprocessing_time=preprocessor.preprocessing_time,
            data_spec=spec,
            capability_spec=capability_spec,
            source_graph_id=source_graph_id,
            prediction_row_adapter=prediction_adapter,
            supervision_counts=supervision_counts,
            provenance_input=provenance,
        )


__all__ = ["HeterogeneousNodeDataPipeline"]
