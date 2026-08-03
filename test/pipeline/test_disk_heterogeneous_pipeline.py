"""Universal-store integration tests for the ordinary heterogeneous pipeline."""

from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import hydra
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from lightning import Trainer
from omegaconf import DictConfig, OmegaConf, open_dict
from torch_geometric.data import HeteroData

import topobench.run as run_module
from test.data.stores.test_topology_only_pyg_partitioner import (
    asymmetric_typed_source,
)
from test.data.stores.test_typed_graph_store import _build_qualified_store
from test.pipeline.test_disk_graph_pipeline import (
    _compose,
    _source_parameters,
)
from topobench.callbacks.dataloader_commit import DataloaderCommitCallback
from topobench.data.loaders.parquet import (
    FittedTransformSpec,
    ParquetTypedGraphSource,
)
from topobench.data.pipelines import DataPipelineOutput
from topobench.data.stores.typed_graph_ingestion import ArtifactValidationError
from topobench.dataloader.disk_graph import DiskGraphDataModule
from topobench.dataloader.heterogeneous import HeterogeneousNodeDataModule
from topobench.nn.capabilities import validate_capability_composition
from topobench.transforms.incremental_pca import IncrementalPCATransform
from topobench.utils.model_instantiation import instantiate_model


class CountingPCATransform(IncrementalPCATransform):
    """Observe the real fit/apply lifecycle without replacing its behavior."""

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.begin_fit_calls = 0
        self.finalize_fit_calls = 0
        self.transform_calls = 0

    def begin_fit(self, context: Any) -> None:
        self.begin_fit_calls += 1
        super().begin_fit(context)

    def finalize_fit(self, state_root: str | Path) -> None:
        self.finalize_fit_calls += 1
        super().finalize_fit(state_root)

    def transform(self, batch: Any) -> Any:
        self.transform_calls += 1
        return super().transform(batch)


def _neighbor_source(
    root: Path, *, pca: bool = False
) -> ParquetTypedGraphSource:
    source = asymmetric_typed_source(root, num_partitions=2)
    author_path = source.spec.source_root / "nodes/authors.parquet"
    author_table = pq.read_table(author_path)
    author_values = author_table["f0"].to_pylist()
    author_table = author_table.append_column(
        "f1",
        pa.array(
            [float(value) + 1.0 for value in author_values],
            type=pa.float32(),
        ),
    )
    author_table = author_table.append_column(
        "f2",
        pa.array(
            [float(value) + 2.0 for value in author_values],
            type=pa.float32(),
        ),
    )
    author_label_index = author_table.schema.get_field_index("label")
    if author_label_index >= 0:
        author_table = author_table.remove_column(author_label_index)
    pq.write_table(author_table, author_path)

    paper_path = source.spec.source_root / "nodes/papers.parquet"
    original_paper = pq.read_table(paper_path)
    paper_ids = original_paper["paper_id"]
    base_values = original_paper["f0"].to_pylist()
    paper_columns: dict[str, pa.Array | pa.ChunkedArray] = {
        "paper_id": paper_ids,
    }
    for column_index in range(128):
        paper_columns[f"f{column_index}"] = pa.array(
            [float(value) + column_index / 100.0 for value in base_values],
            type=pa.float32(),
        )
    paper_columns["label"] = pa.array([0, 1, 0, 1, 0], type=pa.int64())
    pq.write_table(pa.table(paper_columns), paper_path)

    split_ids = {
        "primary": {
            "train": [10, 20],
            "val": [30],
            "test": [40, 50],
        },
        "diagnostic": {
            "train": [50],
            "val": [10],
            "test": [20],
        },
    }
    for split_set in source.spec.supervision.split_registry.sets:
        for phase in ("train", "val", "test"):
            split_path = source.spec.source_root / getattr(split_set, phase)
            pq.write_table(
                pa.table(
                    {
                        "paper_id": pa.array(
                            split_ids[split_set.tag][phase],
                            type=pa.int64(),
                        )
                    }
                ),
                split_path,
            )

    fitted = (
        FittedTransformSpec(name="pca")
        if pca
        else source.spec.fitted_transform
    )
    node_types = tuple(
        replace(
            node,
            feature_columns=tuple(f"f{index}" for index in range(128)),
            feature_dtype="float32",
            feature_width=128,
        )
        if node.name == "paper"
        else replace(
            node,
            feature_columns=("f0", "f1", "f2"),
            feature_dtype="float32",
            feature_width=3,
        )
        for node in source.spec.node_types
    )
    return ParquetTypedGraphSource(
        replace(
            source.spec,
            node_types=node_types,
            supervision=replace(
                source.spec.supervision,
                target_node_type="paper",
            ),
            partition=replace(source.spec.partition, strategy="neighbor"),
            fitted_transform=fitted,
        )
    )


def _parquet_cfg(
    source: ParquetTypedGraphSource,
    tmp_path: Path,
    *,
    store_path: Path | None = None,
    fitted_transform: bool = False,
) -> DictConfig:
    cfg = _compose(
        "dataset=heterogeneous/ParquetTypedGraph",
        "model=heterogeneous/hgt",
        "data_pipeline=heterogeneous_node",
        "trainer.accelerator=cpu",
        "trainer.devices=1",
        "~logger",
    )
    loader = OmegaConf.create(
        {
            "_target_": "topobench.data.loaders.parquet.ParquetTypedGraphLoader",
            "parameters": _source_parameters(
                source,
                data_domain="heterogeneous",
            ),
        }
    )
    target = source.spec.supervision.target_node_type
    with open_dict(cfg.dataset):
        cfg.dataset.loader = loader
        cfg.dataset.parameters.target_node_type = target
        cfg.dataset.parameters.num_classes = 2
        cfg.dataset.parameters.num_features = [
            node.feature_width for node in source.spec.node_types
        ]
        cfg.dataset.dataloader_params.mode = "neighbor"
        cfg.dataset.dataloader_params.batch_size = 2
        cfg.dataset.dataloader_params.num_neighbors = [-1, -1]
        cfg.dataset.dataloader_params.train_shuffle = False
        cfg.dataset.dataloader_params.replace = False
        cfg.dataset.dataloader_params.subgraph_type = "directional"
        cfg.dataset.dataloader_params.sample_direction = "forward"
        cfg.dataset.dataloader_params.filter_per_worker = False
        cfg.dataset.dataloader_params.pop("evaluation_protocol", None)
        cfg.dataset.dataloader_params.pop("evaluation_seed", None)
        cfg.dataset.dataloader_params.pop("pin_memory", None)
    with open_dict(cfg.data_pipeline):
        cfg.data_pipeline.parquet_store_root = str(tmp_path / "stores")
        cfg.data_pipeline.parquet_store_path = (
            None if store_path is None else str(store_path)
        )
        cfg.data_pipeline.fitted_state_root = str(tmp_path / "fit-state")
        if fitted_transform:
            cfg.data_pipeline.fitted_transform = OmegaConf.create(
                {
                    "_target_": (
                        "test.pipeline.test_disk_heterogeneous_pipeline."
                        "CountingPCATransform"
                    ),
                    "n_components": 1,
                    "max_batch_rows": 2,
                    "max_batch_bytes": 4096,
                    "target_node_type": target,
                    "target_field": "x",
                    "input_dtype": "float32",
                    "output_dtype": "float32",
                    "accumulation_dtype": "float64",
                    "whiten": False,
                }
            )
    with open_dict(cfg.paths):
        cfg.paths.data_dir = str(tmp_path / "data")
        cfg.paths.output_dir = str(tmp_path / "output")
    return cfg


def _build(cfg: DictConfig) -> DataPipelineOutput:
    pipeline = hydra.utils.instantiate(cfg.data_pipeline)
    output = pipeline.build(cfg)
    assert isinstance(output, DataPipelineOutput)
    return output


def test_materialized_heterogeneous_pipeline_remains_the_existing_data_module(
    tmp_path: Path,
) -> None:
    cfg = _compose(
        "dataset=heterogeneous/SyntheticHeterogeneous",
        "model=heterogeneous/hgt",
        "data_pipeline=heterogeneous_node",
        "trainer.accelerator=cpu",
        "trainer.devices=1",
        "~logger",
    )
    with open_dict(cfg.paths):
        cfg.paths.data_dir = str(tmp_path / "data")
        cfg.paths.output_dir = str(tmp_path / "output")

    output = _build(cfg)

    assert type(output.datamodule) is HeterogeneousNodeDataModule
    assert output.data_spec is not None
    assert output.active_split_tag is None
    assert output.prediction_row_adapter is not None
    assert output.supervision_counts == {"train": 20, "val": 6, "test": 10}
    assert output.provenance_input is not None
    assert (
        output.provenance_input["source_graph_id"] == "SyntheticHeterogeneous"
    )


def test_hydra_heterogeneous_descriptor_builds_native_neighbor_batches(
    tmp_path: Path,
) -> None:
    source = _neighbor_source(tmp_path / "source")
    cfg = _parquet_cfg(source, tmp_path)

    output = _build(cfg)

    assert type(output.datamodule) is DiskGraphDataModule
    assert output.active_split_tag == "primary"
    assert output.data_spec is not None
    assert output.data_spec.node_types == ("author", "paper")
    assert output.data_spec.target_node_type == "paper"
    assert output.data_spec.input_channels == (
        ("author", 3),
        ("paper", 128),
    )
    assert output.supervision_counts == {"train": 2, "val": 1, "test": 2}
    assert output.capability_spec is not None
    assert output.capability_spec.selector == "heterogeneous/ParquetTypedGraph"
    assert output.capability_spec.data_domain == "heterogeneous"
    assert output.capability_spec.output_kind == "heterogeneous"
    assert output.capability_spec.feature_widths == (
        ("author", 3),
        ("paper", 128),
    )
    assert output.capability_spec.num_classes == 2
    assert output.capability_spec.target_node_type == "paper"
    assert output.provenance_input is not None
    assert output.provenance_input["sampling_strategy"] == (
        "heterogeneous-neighbor"
    )
    assert output.provenance_input["sampler_backend"] == "pyg"
    assert output.provenance_input["fitted_transform_state_key"] is None

    output.datamodule.setup("fit")
    batch = next(iter(output.datamodule.train_dataloader()))
    assert type(batch) is HeteroData
    assert int(batch["paper"].batch_size) == 2
    canonical = output.prediction_row_adapter.resolve(
        batch,
        phase="train",
    )
    assert canonical == (
        (output.source_graph_id, "paper", 0),
        (output.source_graph_id, "paper", 1),
    )
    assert output.prediction_row_adapter.restore_external_ids(canonical) == (
        10,
        20,
    )


def test_heterogeneous_disk_pipeline_runs_one_native_hgt_epoch(
    tmp_path: Path,
) -> None:
    source = _neighbor_source(tmp_path / "source")
    cfg = _parquet_cfg(source, tmp_path)
    with open_dict(cfg):
        cfg.evaluator.undefined_metric_policy = "nan"
    output = _build(cfg)
    capability_validation = validate_capability_composition(
        cfg,
        observed=output.capability_spec,
    )
    model = instantiate_model(
        cfg,
        data_spec=output.data_spec,
        capability_validation=capability_validation,
    )
    output.datamodule.setup("fit")
    batch = next(iter(output.datamodule.val_dataloader()))
    target_count = int(batch["paper"].batch_size)
    selected = model.supervision_adapter.select(
        {
            "logits": torch.zeros((batch["paper"].num_nodes, 2)),
            "labels": batch["paper"].y,
        },
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
    assert selected.num_examples == target_count == 1
    output.datamodule.setup("test")
    assert type(next(iter(output.datamodule.val_dataloader()))) is HeteroData
    assert type(next(iter(output.datamodule.test_dataloader()))) is HeteroData


def test_fitted_transform_fits_once_and_applies_once_per_native_batch(
    tmp_path: Path,
) -> None:
    source = _neighbor_source(tmp_path / "source", pca=True)
    cfg = _parquet_cfg(source, tmp_path, fitted_transform=True)
    output = _build(cfg)
    transform = output.fitted_transform
    assert transform is output.datamodule.fitted_transform
    assert transform.canonical_config()["n_components"] == 1

    output.datamodule.setup("fit")

    assert transform.begin_fit_calls == 1
    assert transform.finalize_fit_calls == 1
    assert transform.transform_calls == 0
    batches = list(output.datamodule.train_dataloader())
    assert batches
    assert transform.transform_calls == len(batches)
    assert all(type(batch) is HeteroData for batch in batches)
    output.datamodule.teardown("fit")
    assert output.datamodule.closed is False
    output.datamodule.setup("test")
    assert type(next(iter(output.datamodule.val_dataloader()))) is HeteroData
    assert type(next(iter(output.datamodule.test_dataloader()))) is HeteroData


def test_run_refreshes_fitted_transform_provenance_after_training(
    tmp_path: Path,
) -> None:
    source = _neighbor_source(tmp_path / "source", pca=True)
    cfg = _parquet_cfg(source, tmp_path, fitted_transform=True)
    with open_dict(cfg):
        cfg.train = True
        cfg.test = False
        cfg.execution_profile = "experimental"
        cfg.preflight.enabled = False
        cfg.evaluation_artifacts.enabled = False
        cfg.evaluator.undefined_metric_policy = "nan"
        cfg.trainer.max_epochs = 1
        cfg.trainer.min_epochs = 1
        cfg.trainer.limit_train_batches = 1
        cfg.trainer.limit_val_batches = 1
        cfg.trainer.enable_progress_bar = False
        cfg.enable_progress_bar = False

    _, objects = run_module.run(cfg)

    transform = objects["datamodule"].fitted_transform
    state_key = transform.state_key
    assert isinstance(state_key, str)
    assert state_key
    returned_output = objects["pipeline_output"]
    assert returned_output.provenance_input is not None
    assert (
        returned_output.provenance_input["fitted_transform_state_key"]
        == state_key
    )


def test_live_fitted_state_key_changes_selected_transform_fingerprint(
    tmp_path: Path,
) -> None:
    cfg = _parquet_cfg(_neighbor_source(tmp_path / "source"), tmp_path)
    baseline_provenance = {
        "fitted_transform": "pca",
        "transform_fingerprint": "d" * 64,
        "fitted_transform_state_key": None,
    }
    first_key = "a" * 64
    second_key = "b" * 64

    first_live = run_module._live_pipeline_provenance(
        SimpleNamespace(
            fitted_transform=SimpleNamespace(state_key=first_key),
        ),
        baseline_provenance,
    )
    second_live = run_module._live_pipeline_provenance(
        SimpleNamespace(
            fitted_transform=SimpleNamespace(state_key=second_key),
        ),
        baseline_provenance,
    )
    fingerprints = [
        run_module._selected_provenance_by_split(
            cfg,
            source_graph_id=None,
            provenance_input=provenance,
            expected_num_examples={"val": 1, "test": 1},
            checkpoint_sha256="c" * 64,
        )["val"]["transform_fingerprint"]
        for provenance in (first_live, second_live)
    ]

    assert baseline_provenance["fitted_transform_state_key"] is None
    assert first_live["fitted_transform_state_key"] == first_key
    assert second_live["fitted_transform_state_key"] == second_key
    assert fingerprints[0] != fingerprints[1]


def test_live_provenance_copy_preserves_non_fitted_pipeline() -> None:
    provenance = {
        "fitted_transform_state_key": None,
        "transform_fingerprint": "d" * 64,
    }

    copied = run_module._live_pipeline_provenance(
        SimpleNamespace(),
        provenance,
    )

    assert copied == provenance
    assert copied is not provenance


@pytest.mark.parametrize("state_key", (None, "", "not-a-published-state-key"))
def test_unbound_fitted_transform_fails_before_selected_publication(
    state_key: str | None,
) -> None:
    with pytest.raises(
        RuntimeError,
        match="fitted transform.*immutable state_key",
    ):
        run_module._live_pipeline_provenance(
            SimpleNamespace(
                fitted_transform=SimpleNamespace(state_key=state_key),
            ),
            {"fitted_transform_state_key": None},
        )


def test_pipeline_rejects_declared_feature_dtype_mismatch_before_training(
    tmp_path: Path,
) -> None:
    source = _neighbor_source(tmp_path / "source")
    mismatched_nodes = tuple(
        replace(node, feature_dtype="float64")
        if node.name == "paper"
        else node
        for node in source.spec.node_types
    )
    mismatched = ParquetTypedGraphSource(
        replace(source.spec, node_types=mismatched_nodes)
    )
    cfg = _parquet_cfg(mismatched, tmp_path / "run")

    with pytest.raises(ArtifactValidationError, match="FEATURE-CAST-001"):
        _build(cfg)


def test_reusing_fitted_state_against_changed_source_fails_contextually(
    tmp_path: Path,
) -> None:
    first_source = _neighbor_source(tmp_path / "first-source", pca=True)
    first_cfg = _parquet_cfg(
        first_source, tmp_path / "run", fitted_transform=True
    )
    pipeline = hydra.utils.instantiate(first_cfg.data_pipeline)
    first_output = pipeline.build(first_cfg)
    first_output.datamodule.setup("fit")

    second_source = _neighbor_source(tmp_path / "second-source", pca=True)
    authors_path = second_source.spec.source_root / "nodes/authors.parquet"
    table = pq.read_table(authors_path)
    changed = table.set_column(
        table.schema.get_field_index("f0"),
        "f0",
        pa.array([40.0, 10.0, 30.0, 20.0], type=pa.float32()),
    )
    pq.write_table(changed, authors_path)
    second_cfg = _parquet_cfg(
        second_source,
        tmp_path / "run",
        fitted_transform=True,
    )

    second_output = pipeline.build(second_cfg)
    with pytest.raises(
        RuntimeError, match="fitted transform context identity mismatch"
    ):
        second_output.datamodule.setup("fit")


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("mode", "partition.strategy"),
        ("model", "cfg.model.model_domain"),
        ("bundle", "save_reproducibility_bundle"),
    ],
)
def test_heterogeneous_pipeline_rejects_mismatch_before_training(
    tmp_path: Path,
    mutation: str,
    expected: str,
) -> None:
    source = _neighbor_source(tmp_path / mutation / "source")
    cfg = _parquet_cfg(source, tmp_path / mutation)
    with open_dict(cfg):
        if mutation == "mode":
            cfg.dataset.dataloader_params.mode = "full_batch"
        elif mutation == "model":
            cfg.model.model_domain = "graph"
        else:
            cfg.dataset.loader.parameters.reproducibility.save_reproducibility_bundle = False

    with pytest.raises((TypeError, ValueError), match=expected):
        _build(cfg)


def test_unqualified_or_mutated_partition_store_is_rejected_before_training(
    tmp_path: Path,
) -> None:
    source = _neighbor_source(tmp_path / "source")
    built = _build_qualified_store(source, tmp_path / "qualified-stores")
    store_path = tmp_path / "mutated-stores" / built.store_build.content_sha256
    shutil.copytree(built.store_build.path, store_path)
    for path in (store_path, *store_path.rglob("*")):
        path.chmod(0o755 if path.is_dir() else 0o644)
    report_path = store_path / "qualification_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["checks"][0]["passed"] = False
    report_path.write_text(json.dumps(report), encoding="utf-8")
    cfg = _parquet_cfg(source, tmp_path / "run", store_path=store_path)

    with pytest.raises(RuntimeError, match="FILE-CHECKSUM-001|qualification"):
        _build(cfg)
