"""The existing homogeneous TopoBench data pipeline."""

from __future__ import annotations

from types import MappingProxyType

from omegaconf import DictConfig

from topobench.data.capabilities import (
    GRAPH_DATASET_MANIFEST,
    RuntimeDataCapability,
    qualify_dataset,
)
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
    _native_provenance_fingerprints,
    is_parquet_typed_graph_config,
    native_prediction_row_adapter,
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

        qualification = qualify_dataset(cfg.dataset)
        capability = GRAPH_DATASET_MANIFEST[
            qualification.selector.partition("/")[2]
        ]
        preprocessor = self.preprocess(cfg)
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
        output_kind = (
            "graph"
            if cfg.dataset.parameters.task_level == "graph"
            else "homogeneous"
        )
        source_graph_id, prediction_adapter = native_prediction_row_adapter(
            cfg,
            output_kind=output_kind,
            sampling_strategy=(
                "graph-batch" if output_kind == "graph" else "homogeneous-full"
            ),
        )
        phase_datasets = {
            "train": train,
            "val": val,
            "test": test,
        }
        if output_kind == "graph":
            supervision_counts = {}
            for phase, dataset in phase_datasets.items():
                if dataset is None:
                    raise ValueError(f"{phase} split must not be empty")
                count = len(dataset)
                if count < 1:
                    raise ValueError(f"{phase} split must not be empty")
                supervision_counts[phase] = count
        else:
            if len(train) != 1 or val is not None or test is not None:
                raise ValueError(
                    "transductive node supervision requires one shared graph"
                )
            data = train[0]
            supervision_counts = {
                phase: int(
                    getattr(data, f"{phase}_mask").count_nonzero().item()
                )
                for phase in ("train", "val", "test")
            }

        observed_widths: set[int] = set()
        maximum_label: int | None = None
        for dataset in phase_datasets.values():
            if dataset is None:
                continue
            for data in dataset:
                observed_widths.add(int(data.x.size(-1)))
                if capability.task == "classification":
                    current = int(data.y.max().item())
                    maximum_label = (
                        current
                        if maximum_label is None
                        else max(maximum_label, current)
                    )
        if len(observed_widths) != 1:
            raise ValueError(
                "homogeneous runtime feature width must be consistent; "
                f"observed {sorted(observed_widths)!r}"
            )
        if capability.task == "classification" and maximum_label is None:
            raise ValueError("classification labels must not be empty")
        capability_spec = RuntimeDataCapability(
            selector=qualification.selector,
            data_domain="graph",
            output_kind=output_kind,
            feature_widths=(("node", next(iter(observed_widths))),),
            num_classes=(
                maximum_label + 1
                if capability.task == "classification"
                and maximum_label is not None
                else None
            ),
            target_node_type=None,
        )
        fingerprints = _native_provenance_fingerprints(
            phase_datasets,
            output_kind=output_kind,
        )

        split_declaration = MappingProxyType(
            {
                str(key): value
                for key, value in cfg.dataset.split_params.items()
                if key != "learning_setting"
            }
        )
        provenance = {
            "source_graph_id": source_graph_id,
            "split_declaration": split_declaration,
            "learning_setting": str(cfg.dataset.split_params.learning_setting),
            "task": prediction_adapter.task,
            "data_domain": str(cfg.dataset.loader.parameters.data_domain),
            "output_kind": output_kind,
            "supervision_counts": MappingProxyType(dict(supervision_counts)),
            **fingerprints,
        }
        provenance["capability_selector"] = capability.selector

        return DataPipelineOutput(
            datamodule=datamodule,
            preprocessing_time=float(preprocessor.preprocessing_time),
            capability_spec=capability_spec,
            source_graph_id=source_graph_id,
            prediction_row_adapter=prediction_adapter,
            supervision_counts=supervision_counts,
            provenance_input=MappingProxyType(provenance),
        )
