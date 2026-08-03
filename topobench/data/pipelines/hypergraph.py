"""Native transductive hypergraph node-classification data pipeline."""

from __future__ import annotations

from types import MappingProxyType

from omegaconf import DictConfig

from topobench.data.capabilities import RuntimeDataCapability, qualify_dataset
from topobench.data.hypergraph import (
    HypergraphData,
    validate_hypergraph_source,
)
from topobench.data.utils.split_utils import load_transductive_splits
from topobench.dataloader import GraphDataModule
from topobench.dataloader.graph import (
    has_hypergraph_validation,
    mark_hypergraph_validated,
)

from .base import (
    AbstractDataPipeline,
    DataPipelineOutput,
    _native_provenance_fingerprints,
    native_prediction_row_adapter,
)


class HypergraphNodeDataPipeline(AbstractDataPipeline):
    """Build one native transductive hypergraph node-classification graph."""

    def build(self, cfg: DictConfig) -> DataPipelineOutput:
        """Preprocess, split, validate, and batch exactly one hypergraph."""
        qualification = qualify_dataset(cfg.dataset)
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
        selector = qualification.selector
        if not has_hypergraph_validation(
            source_data,
            selector=selector,
            num_classes=num_classes,
        ):
            validate_hypergraph_source(
                source_data,
                selector=selector,
                num_classes=num_classes,
            )

        train, val, test = load_transductive_splits(
            [source_data],
            split_params,
        )
        mark_hypergraph_validated(
            train[0],
            selector=selector,
            num_classes=num_classes,
        )
        capability_spec = RuntimeDataCapability(
            selector=qualification.selector,
            data_domain="hypergraph",
            output_kind="hypergraph",
            feature_widths=(("node", int(train[0].x.size(-1))),),
            num_classes=int(train[0].y.max().item()) + 1,
            target_node_type=None,
        )

        supervision_counts = {
            phase: int(
                getattr(train[0], f"{phase}_mask").count_nonzero().item()
            )
            for phase in ("train", "val", "test")
        }
        source_graph_id, prediction_adapter = native_prediction_row_adapter(
            cfg,
            output_kind="hypergraph",
            sampling_strategy="hypergraph-full",
        )
        datamodule = GraphDataModule(
            dataset_train=train,
            dataset_val=val,
            dataset_test=test,
            learning_setting="transductive",
            **cfg.dataset.get("dataloader_params", {}),
        )
        fingerprints = _native_provenance_fingerprints(
            {"train": train, "val": val, "test": test},
            output_kind="hypergraph",
        )
        split_declaration = MappingProxyType(
            {
                str(key): value
                for key, value in split_params.items()
                if key != "learning_setting"
            }
        )
        provenance = MappingProxyType(
            {
                "source_graph_id": source_graph_id,
                "split_declaration": split_declaration,
                "learning_setting": "transductive",
                "task": prediction_adapter.task,
                "data_domain": str(cfg.dataset.loader.parameters.data_domain),
                "output_kind": "hypergraph",
                "supervision_counts": MappingProxyType(
                    dict(supervision_counts)
                ),
                "source_validation_selector": str(selector),
                **fingerprints,
            }
        )
        return DataPipelineOutput(
            datamodule=datamodule,
            preprocessing_time=preprocessor.preprocessing_time,
            capability_spec=capability_spec,
            source_graph_id=source_graph_id,
            prediction_row_adapter=prediction_adapter,
            supervision_counts=supervision_counts,
            provenance_input=provenance,
        )


__all__ = ["HypergraphNodeDataPipeline"]
