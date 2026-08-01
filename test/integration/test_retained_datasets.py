"""Executable qualification evidence for every retained dataset selector."""

from __future__ import annotations

import math
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import hydra
import pytest
import torch
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf
from torch_geometric.data import Data, HeteroData

from topobench.data import (
    DATASET_QUALIFICATION_MANIFEST,
    HYPERGRAPH_REPRESENTATION_VERSION,
    DatasetQualification,
    HypergraphData,
)
from topobench.utils.config_resolvers import (
    get_default_transform,
    register_all_resolvers,
)
from topobench.utils.model_instantiation import instantiate_model

_EVIDENCE_PREFIX = (
    "test/integration/test_retained_datasets.py::"
    "test_retained_dataset_lifecycle"
)
_DATA_NAME_OVERRIDES = {
    "graph/ZINC_OGB": "ZINC",
    "graph/cocitation_citeseer": "citeseer",
    "graph/cocitation_cora": "Cora",
    "graph/cocitation_pubmed": "PubMed",
    "graph/graphuniverse_inductive_triangle": "GraphUniverse",
    "graph/graphuniverse_transductive": "GraphUniverse",
    "hypergraph/20newsgroup": "20newsW100",
}
_SELECTOR_METADATA = {
    "graph/QM9": (
        ("dataset.loader.parameters.qm9_target_index", 0),
        ("dataset.parameters.qm9_target_index", 0),
    ),
    "graph/ZINC": (("dataset.parameters.loss_type", "mse"),),
    "graph/ZINC_OGB": (("dataset.parameters.loss_type", "mae"),),
    "graph/graphuniverse_inductive_triangle": (
        (
            "dataset.loader.parameters.generation_parameters.task",
            "triangle_counting",
        ),
    ),
    "graph/graphuniverse_transductive": (
        (
            "dataset.loader.parameters.generation_parameters.task",
            "community_detection",
        ),
    ),
    "graph/ogbg-molhiv": (
        ("dataset.parameters.preserve_edge_attr_if_lifted", True),
    ),
    "heterogeneous/OGB_MAG": (
        ("dataset.loader.parameters.preprocess", "metapath2vec"),
    ),
    "hypergraph/SyntheticHypergraph": (
        ("dataset.split_params.split_type", "fixed"),
    ),
}
_PIPELINE_BY_DOMAIN = {
    "graph": "default",
    "heterogeneous": "heterogeneous_node",
    "hypergraph": "hypergraph_node",
}


def _qualified_parameter(row: DatasetQualification) -> pytest.ParameterSet:
    marks = ()
    if row.gate == "download":
        marks = (pytest.mark.download, pytest.mark.integration)
    return pytest.param(row, marks=marks, id=row.selector)


_RETAINED_DATASET_PARAMETERS = tuple(
    _qualified_parameter(row)
    for row in DATASET_QUALIFICATION_MANIFEST.values()
)


def _compose(row: DatasetQualification, tmp_path: Path) -> DictConfig:
    """Compose an isolated production dataset/model/pipeline combination."""
    domain = row.selector.split("/", maxsplit=1)[0]
    data_root = tmp_path / "datasets"
    assert not data_root.exists()
    transform_selector = get_default_transform(
        row.selector,
        row.compatible_model,
    )
    overrides = [
        f"dataset={row.selector}",
        f"model={row.compatible_model}",
        f"data_pipeline={_PIPELINE_BY_DOMAIN[domain]}",
        f"transforms={transform_selector}",
        f"paths.data_dir={data_root}",
        f"paths.output_dir={tmp_path / 'output'}",
        "logger=[]",
    ]

    GlobalHydra.instance().clear()
    register_all_resolvers()
    config_dir = str(Path(__file__).resolve().parents[2] / "configs")
    try:
        with hydra.initialize_config_dir(
            version_base="1.3",
            config_dir=config_dir,
            job_name=f"qualify_{row.selector.replace('/', '_')}",
        ):
            return hydra.compose(
                config_name="run.yaml",
                overrides=overrides,
            )
    finally:
        GlobalHydra.instance().clear()


def _release_iterator(iterator: Iterator[Any]) -> None:
    """Release optional worker-backed PyG iterators after one real batch."""
    close = getattr(iterator, "close", None)
    if callable(close):
        close()
    worker_iterator = getattr(iterator, "iterator", iterator)
    shutdown_workers = getattr(worker_iterator, "_shutdown_workers", None)
    if callable(shutdown_workers):
        shutdown_workers()


def _take_one(loader: Any) -> Data | HeteroData:
    """Take exactly one batch without traversing the remaining epoch."""
    iterator = iter(loader)
    try:
        batch = next(iterator)
    finally:
        _release_iterator(iterator)
    assert isinstance(batch, (Data, HeteroData))
    return batch


def _assert_composed_metadata(
    row: DatasetQualification,
    cfg: DictConfig,
    tmp_path: Path,
) -> None:
    """Check selector-specific YAML metadata before any data is loaded."""
    domain, selector = row.selector.split("/", maxsplit=1)
    expected_name = _DATA_NAME_OVERRIDES.get(row.selector, selector)

    assert cfg.dataset.loader._target_ == row.loader_family
    assert cfg.dataset.loader.parameters.data_domain == domain
    assert cfg.dataset.loader.parameters.data_name == expected_name
    assert cfg.dataset.parameters.task == row.task
    assert cfg.dataset.parameters.task_level == row.task_level
    assert cfg.dataset.split_params.learning_setting == row.split_mode
    assert (
        f"{cfg.model.model_domain}/{cfg.model.model_name}"
        == row.compatible_model
    )
    assert Path(str(cfg.dataset.loader.parameters.data_dir)).is_relative_to(
        tmp_path
    )

    configured_policy = cfg.dataset.parameters.get("feature_policy")
    if configured_policy is not None:
        assert configured_policy == row.feature_policy
    for path, expected in _SELECTOR_METADATA.get(row.selector, ()):
        assert OmegaConf.select(cfg, path) == expected

    transforms = cfg.get("transforms")
    if row.edge_policy == "typed_relations_with_reverse":
        assert transforms is not None
        assert "reverse_relations" in transforms
    elif row.edge_policy == "native_typed_relations":
        assert transforms is None or "reverse_relations" not in transforms


def _assert_graph_format(row: DatasetQualification, batch: Data) -> None:
    """Check native graph feature and optional edge-field policy."""
    assert not isinstance(batch, HypergraphData)
    assert isinstance(batch.x, torch.Tensor)
    assert batch.x.ndim == 2
    assert batch.x.is_floating_point()
    assert torch.isfinite(batch.x).all()
    assert isinstance(batch.edge_index, torch.Tensor)
    assert batch.edge_index.dtype == torch.long
    assert batch.edge_index.shape[0] == 2

    edge_attr = batch.get("edge_attr")
    if row.edge_policy == "edge_attr_available":
        assert isinstance(edge_attr, torch.Tensor)
        assert edge_attr.size(0) == batch.edge_index.size(1)
    else:
        assert row.edge_policy == "structural_edges"
        assert edge_attr is None


def _assert_heterogeneous_format(
    row: DatasetQualification,
    cfg: DictConfig,
    batch: HeteroData,
) -> None:
    """Check typed-relation metadata and completed per-type features."""
    assert row.feature_policy in {
        "continuous_per_node_type",
        "continuous_with_constant_fill",
    }
    assert batch.node_types
    assert batch.edge_types
    for node_type in batch.node_types:
        features = batch[node_type].x
        assert features.ndim == 2
        assert features.is_floating_point()
        assert torch.isfinite(features).all()
    for edge_type in batch.edge_types:
        edge_index = batch[edge_type].edge_index
        assert edge_index.dtype == torch.long
        assert edge_index.shape[0] == 2

    target = cfg.dataset.parameters.target_node_type
    assert target in batch.node_types
    target_store = batch[target]
    assert target_store.y.dtype == torch.long
    assert all(
        mask in target_store
        for mask in ("train_mask", "val_mask", "test_mask")
    )


def _assert_hypergraph_format(
    row: DatasetQualification,
    cfg: DictConfig,
    batch: HypergraphData,
) -> None:
    """Check native incidence metadata and the selector's raw parser format."""
    assert row.feature_policy == "continuous"
    assert row.edge_policy == "hyperedge_incidence"
    assert (
        int(batch.representation_version) == HYPERGRAPH_REPRESENTATION_VERSION
    )
    assert batch.x.ndim == 2
    assert batch.x.is_floating_point()
    assert batch.x.shape[1] == cfg.dataset.parameters.num_features
    assert batch.hyperedge_index.dtype == torch.long
    assert batch.hyperedge_index.shape[0] == 2
    num_hyperedges = int(batch.hyperedge_index[1].max()) + 1
    assert torch.equal(
        torch.unique(batch.hyperedge_index[1]),
        torch.arange(num_hyperedges, device=batch.hyperedge_index.device),
    )

    raw_dir = (
        Path(str(cfg.dataset.loader.parameters.data_dir))
        / str(cfg.dataset.loader.parameters.data_name)
        / "raw"
    )
    if row.loader_family == (
        "topobench.data.loaders.CitationHypergraphDatasetLoader"
    ):
        assert all(
            (raw_dir / filename).is_file()
            for filename in (
                "features.pickle",
                "labels.pickle",
                "hypergraph.pickle",
            )
        )
    elif row.loader_family == "topobench.data.loaders.HypergraphDatasetLoader":
        name = str(cfg.dataset.loader.parameters.data_name)
        assert (raw_dir / f"{name}.content").is_file()
        assert (raw_dir / f"{name}.edges").is_file()
    else:
        assert row.loader_family == (
            "topobench.data.loaders.hypergraph.synthetic."
            "SyntheticHypergraphDatasetLoader"
        )


def _assert_runtime_format(
    row: DatasetQualification,
    cfg: DictConfig,
    batch: Data | HeteroData,
) -> None:
    domain = row.selector.split("/", maxsplit=1)[0]
    if domain == "graph":
        assert isinstance(batch, Data)
        _assert_graph_format(row, batch)
    elif domain == "heterogeneous":
        assert isinstance(batch, HeteroData)
        _assert_heterogeneous_format(row, cfg, batch)
    else:
        assert domain == "hypergraph"
        assert isinstance(batch, HypergraphData)
        _assert_hypergraph_format(row, cfg, batch)


def _assert_supervision_shapes(
    row: DatasetQualification,
    cfg: DictConfig,
    model_out: dict[str, Any],
) -> None:
    """Assert the exact scalar-regression or classification shape contract."""
    logits = model_out["logits"]
    targets = model_out["labels"]
    count = int(model_out["num_supervised_examples"])
    assert count > 0

    if row.task == "classification":
        assert logits.shape == (count, cfg.dataset.parameters.num_classes)
        assert targets.shape == (count,)
        assert targets.dtype == torch.long
    else:
        assert row.task == "regression"
        assert logits.shape == (count, 1)
        assert targets.shape == (count, 1)
        assert targets.is_floating_point()


@pytest.mark.parametrize(
    "qualification",
    _RETAINED_DATASET_PARAMETERS,
)
def test_retained_dataset_lifecycle(
    qualification: DatasetQualification,
    tmp_path: Path,
) -> None:
    """Load, preprocess, split, batch, forward, supervise, loss, and score."""
    cfg = _compose(qualification, tmp_path)
    _assert_composed_metadata(qualification, cfg, tmp_path)

    pipeline = hydra.utils.instantiate(cfg.data_pipeline)
    output = pipeline.build(cfg)
    batch = _take_one(output.datamodule.train_dataloader())
    _assert_runtime_format(qualification, cfg, batch)

    model = instantiate_model(cfg, data_spec=output.data_spec)
    model.train()
    model.state_str = "Training"
    model_out = model.model_step(batch)
    _assert_supervision_shapes(qualification, cfg, model_out)

    loss = model_out["loss"]
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert math.isfinite(float(loss.detach()))

    metrics = model.evaluator.compute()
    assert metrics
    for metric in metrics.values():
        assert isinstance(metric, torch.Tensor)
        assert metric.ndim == 0
        assert torch.isfinite(metric)


def test_qualification_manifest_keys_evidence_and_gates_are_consistent() -> (
    None
):
    """Keep immutable manifest identity, evidence, and release marks aligned."""
    assert len(_RETAINED_DATASET_PARAMETERS) == len(
        DATASET_QUALIFICATION_MANIFEST
    )
    assert len(DATASET_QUALIFICATION_MANIFEST) == 43
    packaged = {
        row.selector
        for row in DATASET_QUALIFICATION_MANIFEST.values()
        if row.gate == "packaged"
    }
    assert packaged == {
        "graph/SyntheticGraph",
        "graph/SyntheticGraphRegression",
        "graph/SyntheticNodeGraph",
        "heterogeneous/SyntheticHeterogeneous",
        "hypergraph/SyntheticHypergraph",
    }
    assert (
        sum(
            row.gate == "download"
            for row in DATASET_QUALIFICATION_MANIFEST.values()
        )
        == 38
    )
    for (key, row), parameter in zip(
        DATASET_QUALIFICATION_MANIFEST.items(),
        _RETAINED_DATASET_PARAMETERS,
        strict=True,
    ):
        assert key == row.selector
        assert row.evidence_test == _EVIDENCE_PREFIX
        marks = {mark.name for mark in parameter.marks}
        expected_marks = (
            {"download", "integration"} if row.gate == "download" else set()
        )
        assert marks == expected_marks
