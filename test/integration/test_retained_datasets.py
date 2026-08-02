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
from torch_geometric.data import Data, HeteroData, InMemoryDataset

from topobench.data import (
    DATASET_QUALIFICATION_MANIFEST,
    HYPERGRAPH_REPRESENTATION_VERSION,
    DatasetQualification,
    HypergraphData,
)
from topobench.data.capabilities import GRAPH_DATASET_MANIFEST
from topobench.data.loaders.graph.adme_datasets import ADMEDatasetLoader
from topobench.utils.config_resolvers import register_all_resolvers
from topobench.utils.model_instantiation import instantiate_model

_EVIDENCE_PREFIX = (
    "test/integration/test_retained_datasets.py::"
    "test_retained_dataset_lifecycle"
)
_PARQUET_EVIDENCE = (
    "test/data/stores/test_typed_graph_store.py::"
    "test_opens_homogeneous_and_heterogeneous_content_addressed_stores"
)
_DATA_NAME_OVERRIDES = {
    "graph/ZINC_OGB": "ZINC",
    "graph/cocitation_citeseer": "citeseer",
    "graph/cocitation_cora": "Cora",
    "graph/cocitation_pubmed": "PubMed",
    "graph/graphuniverse_inductive_triangle": "GraphUniverse",
    "graph/graphuniverse_transductive": "GraphUniverse",
}
_ADME_SELECTORS = (
    "graph/BBB_Martins",
    "graph/CYP3A4_Veith",
    "graph/Caco2_Wang",
    "graph/Clearance_Hepatocyte_AZ",
)
_SELECTOR_METADATA = {
    "graph/BBB_Martins": (
        ("dataset.loader.parameters.split_method", "scaffold"),
        ("dataset.loader.parameters.split_seed", 0),
        ("dataset.split_params.split_method", "scaffold"),
    ),
    "graph/CYP3A4_Veith": (
        ("dataset.loader.parameters.split_method", "scaffold"),
        ("dataset.loader.parameters.split_seed", 0),
        ("dataset.split_params.split_method", "scaffold"),
    ),
    "graph/Caco2_Wang": (
        ("dataset.loader.parameters.split_method", "scaffold"),
        ("dataset.loader.parameters.split_seed", 0),
        ("dataset.split_params.split_method", "scaffold"),
    ),
    "graph/Clearance_Hepatocyte_AZ": (
        ("dataset.loader.parameters.split_method", "scaffold"),
        ("dataset.loader.parameters.split_seed", 0),
        ("dataset.split_params.split_method", "scaffold"),
    ),
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
_LOCAL_FEATURE_POLICY_SELECTORS = (
    ("continuous", "graph/SyntheticGraph", None),
    ("categorical_one_hot", "graph/ZINC", "graph/gps"),
    ("degree", "graph/IMDB-MULTI", None),
    ("constant", "graph/REDDIT-BINARY", None),
)


def _qualified_parameter(row: DatasetQualification) -> pytest.ParameterSet:
    marks = ()
    if row.gate == "download":
        marks = (pytest.mark.download, pytest.mark.integration)
    return pytest.param(row, marks=marks, id=row.selector)


_RETAINED_DATASET_PARAMETERS = tuple(
    _qualified_parameter(row)
    for row in DATASET_QUALIFICATION_MANIFEST.values()
    if row.evidence_test == _EVIDENCE_PREFIX
)


def _compose(
    row: DatasetQualification,
    tmp_path: Path,
    *,
    model_selector: str | None = None,
) -> DictConfig:
    """Compose an isolated production dataset/model/pipeline combination."""
    domain = row.selector.split("/", maxsplit=1)[0]
    data_root = tmp_path / "datasets"
    assert not data_root.exists()
    overrides = [
        f"dataset={row.selector}",
        f"model={model_selector or row.compatible_model}",
        f"data_pipeline={_PIPELINE_BY_DOMAIN[domain]}",
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
    *,
    model_selector: str | None = None,
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
        == (model_selector or row.compatible_model)
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
    selector = row.selector.split("/", maxsplit=1)[1]
    capability = GRAPH_DATASET_MANIFEST[selector]
    cardinalities = capability.feature_cardinalities
    if row.feature_policy == "categorical_one_hot" and cardinalities:
        assert cardinalities
        assert batch.x.dtype != torch.bool
        assert not batch.x.is_floating_point()
        assert not batch.x.is_complex()
        assert batch.x.shape[1] == len(cardinalities)
        for column, cardinality in enumerate(cardinalities):
            assert torch.all(batch.x[:, column] >= 0)
            assert torch.all(batch.x[:, column] < cardinality)
    else:
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


def _local_graph_dataset(
    row: DatasetQualification,
    *,
    width: int,
    num_classes: int,
) -> InMemoryDataset:
    """Build a deterministic in-memory loader fixture in the raw policy format."""
    graphs: list[Data] = []
    for graph_index in range(12):
        num_nodes = 6
        path = torch.arange(num_nodes - 1, dtype=torch.long)
        edge_index = torch.stack(
            (
                torch.cat((path, path + 1)),
                torch.cat((path + 1, path)),
            )
        )
        data = Data(edge_index=edge_index, num_nodes=num_nodes)
        if row.feature_policy == "continuous":
            data.x = torch.arange(
                num_nodes * width,
                dtype=torch.float32,
            ).reshape(num_nodes, width)
        elif row.feature_policy == "categorical_one_hot":
            data.x = (
                torch.arange(num_nodes, dtype=torch.long) % width
            ).reshape(-1, 1)
        else:
            assert row.feature_policy in {"degree", "constant"}

        if row.edge_policy == "edge_attr_available":
            data.edge_attr = torch.ones(
                (edge_index.shape[1], 1),
                dtype=torch.float32,
            )
        if row.task == "classification":
            data.y = torch.tensor(
                [graph_index % num_classes],
                dtype=torch.long,
            )
        else:
            data.y = torch.tensor(
                [[float(graph_index) / 10.0]],
                dtype=torch.float32,
            )
        graphs.append(data)

    dataset = InMemoryDataset()
    dataset._data, dataset.slices = dataset.collate(graphs)
    dataset._data_list = None
    dataset.split_idx = {
        "train": torch.arange(0, 8),
        "valid": torch.arange(8, 10),
        "test": torch.arange(10, 12),
    }
    return dataset


def _execute_lifecycle(
    qualification: DatasetQualification,
    cfg: DictConfig,
) -> tuple[Data | HeteroData, Any, dict[str, Any]]:
    """Execute the retained load-through-forward lifecycle once."""
    pipeline = hydra.utils.instantiate(cfg.data_pipeline)
    output = pipeline.build(cfg)
    batch = _take_one(output.datamodule.train_dataloader())
    _assert_runtime_format(qualification, cfg, batch)
    feature_batch = batch.clone()

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
    return feature_batch, model, model_out


@pytest.mark.parametrize("selector", _ADME_SELECTORS)
def test_retained_adme_selector_declares_executed_split_contract(
    selector: str,
    tmp_path: Path,
) -> None:
    """Retained ADME YAML drives the exact method and seed the loader uses."""
    cfg = _compose(DATASET_QUALIFICATION_MANIFEST[selector], tmp_path)

    assert cfg.dataset.split_params.split_method == "scaffold"
    assert (
        cfg.dataset.loader.parameters.split_method
        == cfg.dataset.split_params.split_method
    )
    assert (
        cfg.dataset.loader.parameters.split_seed
        == cfg.dataset.split_params.data_seed
    )


def test_adme_loader_rejects_method_disagreeing_with_selector_metadata(
    tmp_path: Path,
) -> None:
    """A selector claiming scaffold cannot instantiate a random-split loader."""
    cfg = _compose(
        DATASET_QUALIFICATION_MANIFEST["graph/BBB_Martins"],
        tmp_path,
    )
    assert cfg.dataset.split_params.split_method == "scaffold"
    cfg.dataset.loader.parameters.split_method = "random"

    with pytest.raises(ValueError, match="split_method.*scaffold"):
        ADMEDatasetLoader(cfg.dataset.loader.parameters)



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
    _execute_lifecycle(qualification, cfg)


@pytest.mark.parametrize(
    ("policy", "selector", "model_selector"),
    [
        pytest.param(policy, selector, model_selector, id=policy)
        for policy, selector, model_selector in _LOCAL_FEATURE_POLICY_SELECTORS
    ],
)
def test_local_graph_feature_policy_lifecycle(
    policy: str,
    selector: str,
    model_selector: str | None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise each graph feature policy without downloading a dataset."""
    qualification = DATASET_QUALIFICATION_MANIFEST[selector]
    capability = GRAPH_DATASET_MANIFEST[selector.split("/", maxsplit=1)[1]]
    assert qualification.feature_policy == policy
    cfg = _compose(
        qualification,
        tmp_path,
        model_selector=model_selector,
    )
    if model_selector is not None:
        assert (
            cfg.model.feature_encoder.in_channels
            > capability.feature_width
        )
    fixture = _local_graph_dataset(
        qualification,
        width=capability.feature_width,
        num_classes=int(cfg.dataset.parameters.num_classes),
    )
    loader_type = hydra.utils.get_class(str(cfg.dataset.loader._target_))
    monkeypatch.setattr(
        loader_type,
        "load",
        lambda loader: (fixture, str(tmp_path / "local-source")),
    )

    _assert_composed_metadata(
        qualification,
        cfg,
        tmp_path,
        model_selector=model_selector,
    )
    batch, _, model_out = _execute_lifecycle(qualification, cfg)
    assert isinstance(batch, Data)
    assert batch.x.dtype == torch.float32
    assert batch.x.shape[1] == cfg.model.feature_encoder.in_channels
    assert torch.isfinite(batch.x).all()
    assert torch.isfinite(model_out["logits"]).all()


def test_qualification_manifest_keys_evidence_and_gates_are_consistent() -> (
    None
):
    """Keep immutable manifest identity, evidence, and release marks aligned."""
    lifecycle_rows = {
        key: row
        for key, row in DATASET_QUALIFICATION_MANIFEST.items()
        if row.evidence_test == _EVIDENCE_PREFIX
    }
    assert len(_RETAINED_DATASET_PARAMETERS) == len(lifecycle_rows) == 43
    assert len(DATASET_QUALIFICATION_MANIFEST) == 44
    assert (
        DATASET_QUALIFICATION_MANIFEST[
            "graph/ParquetTypedGraph"
        ].evidence_test
        == _PARQUET_EVIDENCE
    )
    packaged = {
        row.selector
        for row in DATASET_QUALIFICATION_MANIFEST.values()
        if row.gate == "packaged"
    }
    assert packaged == {
        "graph/SyntheticGraph",
        "graph/ParquetTypedGraph",
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
        lifecycle_rows.items(),
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
