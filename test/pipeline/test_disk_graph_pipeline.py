"""Universal-store integration tests for the ordinary graph pipeline."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
import pickle
from pathlib import Path
from typing import Any

import hydra
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from hydra.core.hydra_config import HydraConfig
from lightning import Trainer
from omegaconf import DictConfig, OmegaConf, open_dict
from torch_geometric.data import Data

from test.data.stores.test_topology_only_pyg_partitioner import (
    asymmetric_typed_source,
    homogeneous_source as _homogeneous_source,
)
from test.data.stores.test_typed_graph_store import _build_qualified_store
import topobench.data.stores.typed_graph_store as typed_graph_store_module
from topobench.callbacks.dataloader_commit import DataloaderCommitCallback
from topobench.callbacks.input_pipeline import InputPipelineCallback
from topobench.data.loaders.parquet import ParquetTypedGraphSource
from topobench.data.pipelines import DataPipelineOutput
from topobench.data.stores.typed_graph_store import TypedGraphStore
from topobench.dataloader import GraphDataModule
from topobench.dataloader.input_monitor import InputMonitor
from topobench.dataloader.disk_graph import (
    DiskGraphDataModule,
    HomogeneousClusterStrategy,
)
from topobench.profiling.execution_events import ExecutionOperation
from topobench.run import run
from topobench.utils.config_resolvers import register_all_resolvers
from topobench.utils.model_instantiation import instantiate_model

PROJECT_ROOT = Path(__file__).parents[2]
CONFIG_ROOT = PROJECT_ROOT / "configs"


class QualificationRecordingMonitor(InputMonitor):
    """Collect canonical qualification results through the real monitor."""

    def __init__(self) -> None:
        super().__init__()
        self.qualification_records: list[tuple[object, Path]] = []

    def record_qualification(
        self,
        result: object,
        report_path: str | Path,
    ) -> Any:
        event = super().record_qualification(result, report_path)
        self.qualification_records.append((result, Path(report_path)))
        return event


def _binary_homogeneous_source(root: Path) -> ParquetTypedGraphSource:
    """Align the tiny source with the packaged two-class capability."""
    source = _homogeneous_source(root)
    target = source.spec.supervision.target_node_type
    node = next(item for item in source.spec.node_types if item.name == target)
    node_path = source.spec.source_root / node.paths[0]
    table = pq.read_table(node_path)
    label_name = source.spec.supervision.label_column
    label_index = table.schema.get_field_index(label_name)
    labels = pa.array(
        [int(label) % 2 for label in table[label_name].to_pylist()],
        type=table.schema.field(label_name).type,
    )
    first_feature = table[node.feature_columns[0]]
    second_feature = pa.array(
        [-float(value) for value in first_feature.to_pylist()],
        type=table.schema.field(node.feature_columns[0]).type,
    )
    table = table.append_column("feature_1", second_feature)
    pq.write_table(
        table.set_column(label_index, label_name, labels),
        node_path,
    )
    node_types = tuple(
        replace(
            item,
            feature_columns=("feature", "feature_1"),
            feature_width=2,
        )
        if item.name == target
        else item
        for item in source.spec.node_types
    )
    return ParquetTypedGraphSource(replace(source.spec, node_types=node_types))


def _compose(*overrides: str) -> DictConfig:
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    register_all_resolvers()
    with hydra.initialize_config_dir(
        version_base="1.3",
        config_dir=str(CONFIG_ROOT),
        job_name="disk_graph_pipeline",
    ):
        cfg = hydra.compose(
            config_name="run.yaml",
            overrides=list(overrides),
            return_hydra_config=True,
        )
    HydraConfig.instance().set_config(cfg)
    return cfg


def _source_parameters(
    source: ParquetTypedGraphSource,
    *,
    data_domain: str,
) -> dict[str, object]:
    """Render one real descriptor through the same public Hydra schema."""
    spec = source.spec
    node_types: dict[str, object] = {}
    for node in spec.node_types:
        features: dict[str, object] = {
            "dtype": node.feature_dtype,
            "width": node.feature_width,
            "representation": node.feature_representation,
        }
        if node.feature_representation == "fixed_size_list":
            features["column"] = node.feature_columns[0]
        else:
            features["columns"] = list(node.feature_columns)
        node_types[node.name] = {
            "paths": list(node.paths),
            "columns": {
                "id": node.id_column,
                "id_dtype": node.id_dtype,
                "features": features,
            },
        }

    edge_types = [
        {
            "type": list(relation.relation),
            "paths": list(relation.paths),
            "columns": {
                "source": relation.source_column,
                "destination": relation.destination_column,
                "edge_id": relation.edge_id_column,
                "fields": list(relation.edge_fields),
            },
        }
        for relation in spec.relations
    ]
    registry = spec.supervision.split_registry
    split_sets = {
        split.tag: {
            "train": split.train,
            "val": split.val,
            "test": split.test,
            "coverage": split.coverage,
            "qualified": split.qualified,
        }
        for split in registry.sets
    }
    return {
        "data_domain": data_domain,
        "data_type": "parquet_typed",
        "data_name": "ParquetTypedGraph",
        "source_root": str(spec.source_root),
        "output_kind": spec.output_kind,
        "node_types": node_types,
        "edge_types": edge_types,
        "supervision": {
            "target_node_type": spec.supervision.target_node_type,
            "labels": {
                "source": spec.supervision.label_source,
                "column": spec.supervision.label_column,
                "dtype": spec.supervision.label_dtype,
                "paths": (
                    list(spec.supervision.label_paths)
                    if spec.supervision.label_paths
                    else None
                ),
                "node_id": spec.supervision.label_id_column,
            },
            "splits": {
                "active": registry.active_tag,
                "cross_tag_overlap": registry.cross_tag_overlap,
                "within_phase_ids": registry.within_phase_ids,
                "within_tag_phases": registry.within_tag_phases,
                "target_id_resolution": registry.target_id_resolution,
                "sets": split_sets,
            },
        },
        "partition": {
            "strategy": spec.partition.strategy,
            "backend": spec.partition.backend,
            "num_partitions": spec.partition.num_partitions,
            "recursive": spec.partition.recursive,
            "memory_limit_bytes": spec.partition.memory_limit_bytes,
            "external_partition_map": spec.partition.external_partition_map,
        },
        "fitted_transform": {
            "name": spec.fitted_transform.name,
            "fit_on": spec.fitted_transform.fit_on,
            "state_path": spec.fitted_transform.state_path,
        },
        "profiling": {
            "enabled": spec.profiling.enabled,
            "sample_every_steps": spec.profiling.sample_every_steps,
            "emit_on_duration_delta": spec.profiling.emit_on_duration_delta,
            "emit_on_memory_delta_bytes": spec.profiling.emit_on_memory_delta_bytes,
        },
        "reproducibility": {
            "save_reproducibility_bundle": (
                spec.reproducibility.save_reproducibility_bundle
            ),
        },
        "ingestion": {
            "record_batch_rows": spec.ingestion.record_batch_rows,
            "memory_limit_bytes": spec.ingestion.memory_limit_bytes,
            "temp_directory": spec.ingestion.temp_directory,
        },
    }


def _parquet_cfg(
    source: ParquetTypedGraphSource,
    tmp_path: Path,
    *,
    model: str = "graph/gcn",
    store_path: Path | None = None,
) -> DictConfig:
    cfg = _compose(
        "dataset=graph/ParquetTypedGraph",
        f"model={model}",
        "data_pipeline=default",
        "trainer.accelerator=cpu",
        "trainer.devices=1",
        "~logger",
    )
    parameters = _source_parameters(source, data_domain="graph")
    parameters["partition"]["num_partitions"] = 2
    loader = OmegaConf.create(
        {
            "_target_": "topobench.data.loaders.parquet.ParquetTypedGraphLoader",
            "parameters": parameters,
        }
    )
    target = source.spec.supervision.target_node_type
    target_node = next(
        node for node in source.spec.node_types if node.name == target
    )
    with open_dict(cfg.dataset):
        cfg.dataset.loader = loader
        cfg.dataset.parameters.target_node_type = target
        cfg.dataset.parameters.num_features = target_node.feature_width
        cfg.dataset.parameters.num_classes = 2
        cfg.dataset.dataloader_params.clusters_per_batch = 1
        cfg.dataset.dataloader_params.partition_groups = None
        cfg.dataset.dataloader_params.train_shuffle = False
        cfg.dataset.dataloader_params.persistent_workers = False
        cfg.dataset.dataloader_params.pop("batch_size", None)
        cfg.dataset.dataloader_params.pop("pin_memory", None)
    with open_dict(cfg.data_pipeline):
        cfg.data_pipeline.parquet_store_root = str(tmp_path / "stores")
        cfg.data_pipeline.parquet_store_path = (
            None if store_path is None else str(store_path)
        )
        cfg.data_pipeline.fitted_state_root = str(tmp_path / "fit-state")
    with open_dict(cfg.paths):
        cfg.paths.data_dir = str(tmp_path / "data")
        cfg.paths.output_dir = str(tmp_path / "output")
    return cfg


def _build(cfg: DictConfig) -> DataPipelineOutput:
    pipeline = hydra.utils.instantiate(cfg.data_pipeline)
    output = pipeline.build(cfg)
    assert isinstance(output, DataPipelineOutput)
    return output


def test_materialized_graph_pipeline_remains_the_existing_data_module(
    tmp_path: Path,
) -> None:
    cfg = _compose(
        "dataset=graph/SyntheticNodeGraph",
        "model=graph/gcn",
        "data_pipeline=default",
        "trainer.accelerator=cpu",
        "trainer.devices=1",
        "~logger",
    )
    with open_dict(cfg.paths):
        cfg.paths.data_dir = str(tmp_path / "data")
        cfg.paths.output_dir = str(tmp_path / "output")

    output = _build(cfg)

    assert type(output.datamodule) is GraphDataModule
    assert output.active_split_tag is None
    assert output.prediction_identity_resolver is None
    assert output.supervision_counts == {}
    assert output.provenance_input is None


def test_hydra_graph_descriptor_builds_the_standard_disk_pipeline(
    tmp_path: Path,
) -> None:
    source = _binary_homogeneous_source(tmp_path / "source")
    cfg = _parquet_cfg(source, tmp_path)

    output = _build(cfg)

    assert type(output.datamodule) is DiskGraphDataModule
    assert output.data_spec is None
    assert output.active_split_tag == "default"
    assert output.supervision_counts == {"train": 1, "val": 1, "test": 1}
    assert output.reproducibility_policy is source.spec.reproducibility or (
        output.reproducibility_policy == source.spec.reproducibility
    )
    assert output.profiling_policy == source.spec.profiling
    assert output.qualification_report is not None
    assert output.execution_monitor is None
    assert output.provenance_input is not None
    assert output.provenance_input["source_graph_id"] == output.source_graph_id
    assert output.provenance_input["active_split_tag"] == "default"
    assert output.provenance_input["sampling_strategy"] == "homogeneous-cluster"
    assert output.provenance_input["sampler_backend"] == "pyg"
    assert output.provenance_input["fitted_transform_state_key"] is None
    assert output.provenance_input["supervision_counts"] == {
        "train": 1,
        "val": 1,
        "test": 1,
    }

    output.datamodule.setup("fit")
    batch = next(iter(output.datamodule.train_dataloader()))
    assert type(batch) is Data
    canonical = output.prediction_identity_resolver.resolve(
        batch,
        phase="train",
    )
    assert canonical == ((output.source_graph_id, 0),)
    assert output.prediction_identity_resolver.restore_external_ids(canonical) == (
        -5,
    )


def test_pipeline_and_store_emit_one_canonical_qualification_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _binary_homogeneous_source(tmp_path / "source")
    cfg = _parquet_cfg(source, tmp_path)
    monitor = QualificationRecordingMonitor()
    pipeline = hydra.utils.instantiate(
        cfg.data_pipeline,
        execution_monitor=monitor,
    )
    validate_store = typed_graph_store_module.validate_store
    qualified_paths: list[Path] = []

    def count_canonical_qualification(
        path: str | Path,
        **kwargs: object,
    ) -> object:
        if kwargs.get("require_directory_identity") is True:
            qualified_paths.append(Path(path))
        return validate_store(path, **kwargs)

    monkeypatch.setattr(
        typed_graph_store_module,
        "validate_store",
        count_canonical_qualification,
    )

    output = pipeline.build(cfg)

    assert output.qualification_report is not None
    expected_ids = tuple(
        result.check_id for result in output.qualification_report.checks
    )
    store_path = output.prediction_identity_resolver.store_path
    assert qualified_paths == [store_path]
    canonical = ((output.source_graph_id, 0),)
    assert output.prediction_identity_resolver.restore_external_ids(canonical) == (
        -5,
    )
    assert output.prediction_identity_resolver.restore_external_ids(canonical) == (
        -5,
    )
    assert qualified_paths == [store_path]
    assert tuple(
        result.check_id for result, _ in monitor.qualification_records
    ) == expected_ids
    assert {
        report_path for _, report_path in monitor.qualification_records
    } == {output.qualification_report.report_path}
    output.datamodule.setup("fit")
    assert output.datamodule.descriptors("train")
    assert output.datamodule.descriptors("val")
    assert type(next(iter(output.datamodule.train_dataloader()))) is Data
    assert type(next(iter(output.datamodule.val_dataloader()))) is Data
    assert qualified_paths == [store_path]
    output.datamodule.close()


    qualified_paths.clear()
    monitor.qualification_records.clear()
    cached_output = pipeline.build(cfg)

    assert qualified_paths == [store_path]
    assert tuple(
        result.check_id for result, _ in monitor.qualification_records
    ) == expected_ids
    cached_output.datamodule.close()

    qualified_paths.clear()
    monitor.qualification_records.clear()
    with TypedGraphStore.open(
        store_path,
        execution_monitor=monitor,
    ) as reopened:
        assert reopened.content_sha256 == output.source_graph_id
    assert qualified_paths == [store_path]
    assert tuple(
        result.check_id for result, _ in monitor.qualification_records
    ) == expected_ids

    qualified_paths.clear()
    direct = DiskGraphDataModule(
        store_path,
        HomogeneousClusterStrategy(clusters_per_batch=1, seed=42),
        train_shuffle=False,
    )
    assert qualified_paths == [store_path]
    direct.setup("fit")
    assert direct.descriptors("train")
    assert direct.descriptors("val")
    assert type(next(iter(direct.train_dataloader()))) is Data
    assert type(next(iter(direct.val_dataloader()))) is Data
    direct.teardown("fit")
    assert direct.closed is False
    direct.setup("test")
    assert type(next(iter(direct.test_dataloader()))) is Data
    assert qualified_paths == [store_path]

    qualified_paths.clear()
    assert direct._owner is not None
    worker_owner = pickle.loads(pickle.dumps(direct._owner))
    worker_store = worker_owner.get()
    assert worker_store.content_sha256 == output.source_graph_id
    assert qualified_paths == []
    worker_owner.close()
    direct.close()


def test_graph_disk_pipeline_runs_one_native_trainer_epoch(tmp_path: Path) -> None:
    source = _binary_homogeneous_source(tmp_path / "source")
    cfg = _parquet_cfg(source, tmp_path)
    output = _build(cfg)
    model = instantiate_model(cfg, data_spec=output.data_spec)
    output.datamodule.setup("fit")
    batch = next(iter(output.datamodule.val_dataloader()))
    selected = model.supervision_adapter.select(
        {"logits": torch.zeros((batch.num_nodes, 3)), "labels": batch.y},
        batch,
        "Validation",
    )
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=1,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        callbacks=[DataloaderCommitCallback()],
    )

    trainer.fit(model=model, datamodule=output.datamodule)

    assert trainer.current_epoch == 1
    assert selected.num_examples == int(batch.val_mask.sum()) == 1
    output.datamodule.setup("test")
    assert type(next(iter(output.datamodule.val_dataloader()))) is Data
    assert type(next(iter(output.datamodule.test_dataloader()))) is Data


def test_ordinary_run_shares_one_callback_owned_monitor_before_ingestion(
    tmp_path: Path,
) -> None:
    source = _binary_homogeneous_source(tmp_path / "source")
    cfg = _parquet_cfg(source, tmp_path)
    input_pipeline_cfg = OmegaConf.load(
        CONFIG_ROOT / "callbacks" / "input_pipeline.yaml"
    ).input_pipeline
    with open_dict(cfg):
        cfg.train = False
        cfg.test = False
        cfg.callbacks = OmegaConf.create(
            {"input_pipeline": input_pipeline_cfg},
        )

    _, objects = run(cfg)

    callback = objects["callbacks"][0]
    assert type(callback) is InputPipelineCallback
    assert objects["pipeline_output"].execution_monitor is callback.monitor
    assert objects["datamodule"].execution_monitor is callback.monitor
    assert objects["model"].execution_monitor is callback.monitor
    conversion_events = [
        event
        for event in callback.monitor.drain()
        if event.operation is ExecutionOperation.CONVERSION
    ]
    assert conversion_events


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("output", "output_kind"),
        ("model", "cfg.model.model_domain"),
        ("backend", "partition.backend"),
        ("partition_count", "num_partitions"),
        ("tag", "active split tag"),
        ("bundle", "save_reproducibility_bundle"),
    ],
)
def test_graph_pipeline_rejects_invalid_compositions_before_training(
    tmp_path: Path,
    mutation: str,
    expected: str,
) -> None:
    source = _binary_homogeneous_source(tmp_path / mutation / "source")
    cfg = _parquet_cfg(source, tmp_path / mutation)
    with open_dict(cfg):
        if mutation == "output":
            cfg.dataset.loader.parameters.output_kind = "heterogeneous"
        elif mutation == "model":
            cfg.model.model_domain = "heterogeneous"
        elif mutation == "backend":
            cfg.dataset.loader.parameters.partition.backend = "unsupported"
        elif mutation == "partition_count":
            cfg.dataset.loader.parameters.partition.num_partitions = 1
        elif mutation == "tag":
            cfg.data_pipeline.active_split_tag = "diagnostic"
        else:
            cfg.dataset.loader.parameters.reproducibility.save_reproducibility_bundle = False

    with pytest.raises((TypeError, ValueError), match=expected):
        _build(cfg)


def test_pipeline_rejects_a_store_from_a_different_descriptor(tmp_path: Path) -> None:
    heterogeneous = asymmetric_typed_source(tmp_path / "heterogeneous-source")
    foreign = _build_qualified_store(
        heterogeneous,
        tmp_path / "foreign-stores",
    )
    homogeneous = _binary_homogeneous_source(tmp_path / "homogeneous-source")
    cfg = _parquet_cfg(
        homogeneous,
        tmp_path / "graph-run",
        store_path=foreign.store_build.path,
    )

    with pytest.raises(ValueError, match="store/source binding mismatch"):
        _build(cfg)
