"""Contracts for configuration-driven data pipelines."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, call

import hydra
import pytest
import torch
from lightning import LightningDataModule
from omegaconf import DictConfig, OmegaConf
from torch_geometric.data import Batch, Data, HeteroData
from torch_geometric.loader import NeighborLoader

from test._utils import simplified_pipeline
from topobench import run as run_module
from topobench.data.datasets import make_synthetic_heterogeneous_data
from topobench.data.heterogeneous import HeterogeneousDataSpec
from topobench.data.pipelines import (
    AbstractDataPipeline,
    DataPipelineOutput,
    DefaultDataPipeline,
    HeterogeneousNodeDataPipeline,
)
from topobench.dataloader import GraphDataModule
from topobench.utils.config_resolvers import register_all_resolvers


@pytest.fixture(autouse=True)
def _isolate_global_hydra() -> Iterator[None]:
    """Guarantee clean Hydra state before and after every pipeline test."""
    global_hydra = hydra.core.global_hydra.GlobalHydra.instance()
    global_hydra.clear()
    assert not global_hydra.is_initialized()
    try:
        yield
    finally:
        global_hydra.clear()
        assert not global_hydra.is_initialized()


class MutableDataModule(LightningDataModule):
    """Minimal real data module for runtime-boundary tests."""

    def __init__(self) -> None:
        super().__init__()
        self.marker = 0


def _heterogeneous_spec() -> HeterogeneousDataSpec:
    """Return a small valid heterogeneous runtime contract."""
    return HeterogeneousDataSpec(
        node_types=("author", "paper"),
        edge_types=(("author", "writes", "paper"),),
        target_node_type="author",
        num_classes=2,
        input_channels=(("author", 4), ("paper", 3)),
    )


def _native_heterogeneous_batch() -> HeteroData:
    """Return a faithful native sampled-batch shape without external data."""
    batch = HeteroData()
    batch["author"].x = torch.zeros(5, 4)
    batch["author"].n_id = torch.arange(5)
    batch["author"].batch_size = 3
    batch["paper"].x = torch.zeros(4, 3)
    batch["paper"].n_id = torch.arange(4)
    batch["author", "writes", "paper"].edge_index = torch.tensor(
        [[0, 1, 2, 3], [0, 1, 2, 3]],
        dtype=torch.long,
    )
    return batch


def test_data_pipeline_output_normalizes_and_freezes_boundary_values() -> None:
    """Valid outputs normalize time while retaining mutable runtime objects."""
    datamodule = MutableDataModule()
    data_spec = _heterogeneous_spec()

    output = DataPipelineOutput(
        datamodule=datamodule,
        preprocessing_time=0,
        data_spec=data_spec,
    )

    assert output.datamodule is datamodule
    assert output.preprocessing_time == 0.0
    assert type(output.preprocessing_time) is float
    assert output.data_spec is data_spec

    datamodule.marker = 1
    assert output.datamodule.marker == 1
    with pytest.raises(FrozenInstanceError):
        output.data_spec = None  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field_name", "value", "error_type", "message"),
    [
        (
            "datamodule",
            object(),
            TypeError,
            "datamodule must be a LightningDataModule",
        ),
        (
            "data_spec",
            object(),
            TypeError,
            "data_spec must be a HeterogeneousDataSpec or None",
        ),
        (
            "preprocessing_time",
            True,
            TypeError,
            "preprocessing_time must be a real numeric scalar",
        ),
        (
            "preprocessing_time",
            "1.0",
            TypeError,
            "preprocessing_time must be a real numeric scalar",
        ),
        (
            "preprocessing_time",
            torch.tensor(1.0),
            TypeError,
            "preprocessing_time must be a real numeric scalar",
        ),
        (
            "preprocessing_time",
            -0.1,
            ValueError,
            "preprocessing_time must be non-negative",
        ),
        (
            "preprocessing_time",
            float("nan"),
            ValueError,
            "preprocessing_time must be finite",
        ),
        (
            "preprocessing_time",
            float("inf"),
            ValueError,
            "preprocessing_time must be finite",
        ),
        (
            "preprocessing_time",
            float("-inf"),
            ValueError,
            "preprocessing_time must be finite",
        ),
    ],
    ids=[
        "datamodule",
        "data-spec",
        "bool-time",
        "string-time",
        "tensor-time",
        "negative-time",
        "nan-time",
        "positive-infinity-time",
        "negative-infinity-time",
    ],
)
def test_data_pipeline_output_rejects_invalid_runtime_values(
    field_name: str,
    value: object,
    error_type: type[Exception],
    message: str,
) -> None:
    """Invalid runtime values fail at the pipeline boundary."""
    kwargs: dict[str, object] = {
        "datamodule": MutableDataModule(),
        "preprocessing_time": 1.5,
        "data_spec": _heterogeneous_spec(),
    }
    kwargs[field_name] = value

    with pytest.raises(error_type, match=message):
        DataPipelineOutput(**kwargs)  # type: ignore[arg-type]


def test_observed_batch_size_uses_ordinary_pyg_graph_count() -> None:
    """Ordinary homogeneous batches report their number of graphs."""
    batch = Batch.from_data_list(
        [
            Data(x=torch.zeros(3, 2)),
            Data(x=torch.zeros(4, 2)),
        ]
    )

    observed = simplified_pipeline._infer_observed_train_batch_size(
        batch,
        data_spec=None,
    )

    assert observed == 2


def test_observed_batch_size_prefers_heterogeneous_seed_count() -> None:
    """Sampled heterogeneous batches report target seeds, not graph count."""
    data = make_synthetic_heterogeneous_data(seed=7)
    data["venue"].x = torch.ones(data["venue"].num_nodes, 1)
    data_spec = HeterogeneousDataSpec(
        node_types=tuple(data.node_types),
        edge_types=tuple(data.edge_types),
        target_node_type="author",
        num_classes=2,
        input_channels=(("author", 8), ("paper", 5), ("venue", 1)),
    )
    loader = NeighborLoader(
        data,
        num_neighbors=[2],
        input_nodes=("author", data["author"].train_mask),
        batch_size=4,
        shuffle=False,
    )
    batch = next(iter(loader))

    assert isinstance(batch, HeteroData)
    assert batch["author"].batch_size == 4
    assert "n_id" in batch["author"]
    assert not hasattr(batch, "num_graphs")

    observed = simplified_pipeline._infer_observed_train_batch_size(
        batch,
        data_spec=data_spec,
    )

    assert observed == 4


def test_observed_batch_size_falls_back_for_full_batch_hetero() -> None:
    """Full-batch heterogeneous data may use PyG's graph-count contract."""
    data = _native_heterogeneous_batch()
    del data["author"].batch_size
    batch = Batch.from_data_list([data])

    observed = simplified_pipeline._infer_observed_train_batch_size(
        batch,
        data_spec=_heterogeneous_spec(),
    )

    assert observed == 1


@pytest.mark.parametrize(
    ("value", "error_type", "message"),
    [
        (True, TypeError, "target seed batch_size.*positive integer"),
        (1.5, TypeError, "target seed batch_size.*positive integer"),
        (0, ValueError, "target seed batch_size.*greater than zero"),
        (-2, ValueError, "target seed batch_size.*greater than zero"),
    ],
    ids=["bool", "non-integral", "zero", "negative"],
)
def test_observed_batch_size_rejects_invalid_heterogeneous_seed_counts(
    value: object,
    error_type: type[Exception],
    message: str,
) -> None:
    """Malformed NeighborLoader seed metadata fails clearly."""
    batch = _native_heterogeneous_batch()
    batch["author"].batch_size = value

    with pytest.raises(error_type, match=message):
        simplified_pipeline._infer_observed_train_batch_size(
            batch,
            data_spec=_heterogeneous_spec(),
        )


@pytest.mark.parametrize(
    ("value", "error_type", "message"),
    [
        (False, TypeError, "num_graphs.*positive integer"),
        ("2", TypeError, "num_graphs.*positive integer"),
        (0, ValueError, "num_graphs.*greater than zero"),
        (-1, ValueError, "num_graphs.*greater than zero"),
    ],
    ids=["bool", "non-integral", "zero", "negative"],
)
def test_observed_batch_size_rejects_invalid_graph_counts(
    value: object,
    error_type: type[Exception],
    message: str,
) -> None:
    """Malformed ordinary batch-size metadata fails clearly."""
    batch = SimpleNamespace(num_graphs=value)

    with pytest.raises(error_type, match=message):
        simplified_pipeline._infer_observed_train_batch_size(
            batch,
            data_spec=None,
        )


def test_observed_batch_size_requires_native_target_store() -> None:
    """A heterogeneous contract requires its declared target node store."""
    batch = HeteroData()
    batch["paper"].x = torch.zeros(2, 3)

    with pytest.raises(
        ValueError,
        match="HeteroData.*missing target node store 'author'",
    ):
        simplified_pipeline._infer_observed_train_batch_size(
            batch,
            data_spec=_heterogeneous_spec(),
        )


def test_observed_batch_size_requires_native_heterogeneous_batch() -> None:
    """A heterogeneous runtime contract rejects homogeneous PyG batches."""
    batch = Batch.from_data_list([Data(x=torch.zeros(3, 2))])

    with pytest.raises(
        TypeError,
        match="DataBatch.*requires native HeteroData",
    ):
        simplified_pipeline._infer_observed_train_batch_size(
            batch,
            data_spec=_heterogeneous_spec(),
        )


def test_observed_batch_size_names_unsupported_batch_contract() -> None:
    """Unsupported objects identify their type and missing requirement."""
    with pytest.raises(
        TypeError,
        match="object.*num_graphs",
    ):
        simplified_pipeline._infer_observed_train_batch_size(
            object(),
            data_spec=None,
        )


def _pipeline_cfg(
    *,
    task_level: str = "graph",
    transforms: dict[str, str] | None = None,
) -> DictConfig:
    """Return the smallest configuration accepted by the default pipeline."""
    return OmegaConf.create(
        {
            "dataset": {
                "loader": {
                    "_target_": "tests.Loader",
                    "parameters": {"data_domain": "graph"},
                },
                "parameters": {
                    "task_level": task_level,
                    "feature_policy": "continuous",
                    "num_features": 1,
                },
                "split_params": {
                    "learning_setting": "inductive",
                    "data_seed": 17,
                },
                "dataloader_params": {
                    "batch_size": 8,
                    "num_workers": 2,
                },
            },
            "transforms": transforms,
        }
    )


def _install_pipeline_spies(
    monkeypatch: pytest.MonkeyPatch,
    cfg: DictConfig,
) -> tuple[
    list[object],
    object,
    object,
    tuple[object, object, object],
    object,
    object,
    MagicMock,
    MagicMock,
]:
    """Install faithful boundary spies around the moved orchestration."""
    import topobench.data.pipelines.base as base_module
    import topobench.data.pipelines.default as default_module

    events: list[object] = []
    dataset = object()
    dataset_dir = object()
    splits = (object(), object(), object())
    transforms = object()
    datamodule = MagicMock(spec=LightningDataModule)

    loader = MagicMock()

    def load() -> tuple[object, object]:
        events.append("load")
        return dataset, dataset_dir

    loader.load.side_effect = load

    def instantiate(config: DictConfig) -> object:
        if config is cfg.dataset.loader:
            events.append("instantiate_loader")
            return loader
        if config is cfg.transforms:
            events.append("instantiate_transforms")
            return transforms
        raise AssertionError(f"Unexpected instantiate call: {config!r}")

    instantiate_spy = MagicMock(side_effect=instantiate)
    monkeypatch.setattr(
        base_module.hydra.utils, "instantiate", instantiate_spy
    )

    preprocessor = MagicMock()
    preprocessor.preprocessing_time = 1.25

    def load_splits(split_params: DictConfig) -> tuple[object, object, object]:
        events.append(("load_splits", split_params))
        return splits

    preprocessor.load_dataset_splits.side_effect = load_splits

    def make_preprocessor(
        received_dataset: object,
        received_dir: object,
        received_transforms: object | None,
    ) -> MagicMock:
        events.append(
            (
                "preprocessor",
                received_dataset,
                received_dir,
                received_transforms,
            )
        )
        return preprocessor

    preprocessor_spy = MagicMock(side_effect=make_preprocessor)
    monkeypatch.setattr(base_module, "PreProcessor", preprocessor_spy)

    def prepare_features(
        dataset_train: object,
        dataset_val: object,
        dataset_test: object,
        **kwargs: object,
    ) -> None:
        assert (dataset_train, dataset_val, dataset_test) == splits
        assert kwargs == {
            "feature_policy": "continuous",
            "num_features": 1,
        }
        events.append("prepare_graph_features")

    monkeypatch.setattr(
        default_module,
        "prepare_graph_features",
        prepare_features,
    )

    def make_datamodule(**kwargs: object) -> MagicMock:
        events.append(("datamodule", kwargs))
        return datamodule

    datamodule_spy = MagicMock(side_effect=make_datamodule)
    monkeypatch.setattr(default_module, "GraphDataModule", datamodule_spy)

    return (
        events,
        dataset,
        dataset_dir,
        splits,
        transforms,
        datamodule,
        instantiate_spy,
        preprocessor_spy,
    )


@pytest.mark.parametrize("task_level", ["node", "graph"])
def test_default_pipeline_preserves_orchestration_and_output_contract(
    monkeypatch: pytest.MonkeyPatch,
    task_level: str,
) -> None:
    """The configurable boundary must be an exact move of the default path."""
    cfg = _pipeline_cfg(
        task_level=task_level,
        transforms={"_target_": "tests.Transform"},
    )
    (
        events,
        dataset,
        dataset_dir,
        splits,
        transforms,
        datamodule,
        instantiate_spy,
        preprocessor_spy,
    ) = _install_pipeline_spies(monkeypatch, cfg)

    output = DefaultDataPipeline().build(cfg)

    assert events == [
        "instantiate_loader",
        "load",
        "instantiate_transforms",
        ("preprocessor", dataset, dataset_dir, transforms),
        ("load_splits", cfg.dataset.split_params),
        "prepare_graph_features",
        (
            "datamodule",
            {
                "dataset_train": splits[0],
                "dataset_val": splits[1],
                "dataset_test": splits[2],
                "learning_setting": "inductive",
                "batch_size": 8,
                "num_workers": 2,
            },
        ),
    ]
    assert instantiate_spy.call_args_list == [
        call(cfg.dataset.loader),
        call(cfg.transforms),
    ]
    preprocessor_spy.assert_called_once()
    assert output.datamodule is datamodule
    assert output.preprocessing_time == pytest.approx(1.25)
    assert isinstance(output.preprocessing_time, float)
    assert output.data_spec is None

    with pytest.raises(FrozenInstanceError):
        output.preprocessing_time = 2.5  # type: ignore[misc]


def test_default_pipeline_does_not_instantiate_absent_transforms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No-transform runs pass ``None`` through without Hydra instantiation."""
    cfg = _pipeline_cfg(transforms=None)
    (
        events,
        dataset,
        dataset_dir,
        _,
        _,
        _,
        instantiate_spy,
        preprocessor_spy,
    ) = _install_pipeline_spies(monkeypatch, cfg)

    DefaultDataPipeline().build(cfg)

    assert instantiate_spy.call_args_list == [call(cfg.dataset.loader)]
    preprocessor_spy.assert_called_once_with(dataset, dataset_dir, None)
    assert "instantiate_transforms" not in events


def test_default_pipeline_retains_invalid_task_level_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The refactor must retain the existing invalid-task behavior and order."""
    cfg = _pipeline_cfg(task_level="edge", transforms=None)
    events, *_ = _install_pipeline_spies(monkeypatch, cfg)

    with pytest.raises(ValueError, match=r"^Invalid task_level$"):
        DefaultDataPipeline().build(cfg)

    event_names = [
        event if isinstance(event, str) else event[0] for event in events
    ]
    assert event_names == [
        "instantiate_loader",
        "load",
        "preprocessor",
        "load_splits",
    ]


@pytest.mark.parametrize(
    "overrides",
    [
        ["dataset=graph/MUTAG", "model=graph/gcn"],
        ["experiment=cell_hgt_mutag_debug"],
    ],
    ids=["graph", "cell-experiment"],
)
def test_default_pipeline_is_composed_for_existing_experiments(
    overrides: list[str],
) -> None:
    """Existing homogeneous runs select the default pipeline automatically."""
    global_hydra = hydra.core.global_hydra.GlobalHydra.instance()
    assert not global_hydra.is_initialized()
    register_all_resolvers()
    hydra.initialize(
        version_base="1.3",
        config_path="../../../configs",
    )
    cfg = hydra.compose(config_name="run.yaml", overrides=overrides)

    assert (
        cfg.data_pipeline._target_
        == "topobench.data.pipelines.DefaultDataPipeline"
    )


def test_composed_minesweeper_params_construct_native_graph_datamodule() -> None:
    """Minesweeper's composed loader parameters satisfy the native contract."""
    register_all_resolvers()
    hydra.initialize(
        version_base="1.3",
        config_path="../../../configs",
    )
    cfg = hydra.compose(
        config_name="run.yaml",
        overrides=["dataset=graph/minesweeper"],
    )
    dataloader_params = OmegaConf.to_container(
        cfg.dataset.dataloader_params,
        resolve=True,
    )
    assert isinstance(dataloader_params, dict)

    datamodule = GraphDataModule(
        dataset_train=[Data()],
        learning_setting=cfg.dataset.split_params.learning_setting,
        **dataloader_params,
    )

    assert datamodule.batch_size == 1
    assert datamodule.loader_kwargs == {
        "num_workers": 0,
        "pin_memory": False,
        "persistent_workers": False,
    }
    assert datamodule.dataset_train is datamodule.dataset_val
    assert datamodule.dataset_train is datamodule.dataset_test


def test_production_run_consumes_pipeline_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The production entry point builds the configured pipeline exactly once."""

    class FakeWandbLogger:
        def __init__(self) -> None:
            self.logged: list[dict[str, float]] = []

        def log_metrics(self, metrics: dict[str, float]) -> None:
            self.logged.append(metrics)

    datamodule = MagicMock(spec=LightningDataModule)
    data_spec = MagicMock(spec=HeterogeneousDataSpec)
    pipeline_output = DataPipelineOutput(
        datamodule=datamodule,
        preprocessing_time=3.5,
        data_spec=data_spec,
    )
    pipeline = MagicMock()
    pipeline.build.return_value = pipeline_output
    model = MagicMock()
    trainer = MagicMock()
    trainer.callback_metrics = {"train/loss": torch.tensor(1.0)}
    wandb_logger = FakeWandbLogger()
    callbacks = [object()]

    cfg = OmegaConf.create(
        {
            "seed": 7,
            "deterministic": False,
            "data_pipeline": {"_target_": "tests.Pipeline"},
            "model": {"_target_": "tests.Model"},
            "evaluator": {},
            "optimizer": {},
            "loss": {},
            "callbacks": {},
            "logger": {},
            "trainer": {"_target_": "tests.Trainer"},
            "paths": {"output_dir": str(tmp_path)},
            "train": False,
            "test": False,
        }
    )

    def instantiate(config: DictConfig, **kwargs: object) -> object:
        del kwargs
        if config is cfg.data_pipeline:
            return pipeline
        if config is cfg.trainer:
            return trainer
        raise AssertionError(f"Unexpected instantiate call: {config!r}")

    instantiate_spy = MagicMock(side_effect=instantiate)
    instantiate_model_spy = MagicMock(return_value=model)
    monkeypatch.setattr(
        run_module.hydra.utils,
        "instantiate",
        instantiate_spy,
    )
    monkeypatch.setattr(
        run_module,
        "instantiate_model",
        instantiate_model_spy,
    )
    monkeypatch.setattr(
        run_module,
        "instantiate_callbacks",
        MagicMock(return_value=callbacks),
    )
    monkeypatch.setattr(
        run_module,
        "instantiate_loggers",
        MagicMock(return_value=[wandb_logger]),
    )
    monkeypatch.setattr(
        run_module.L.pytorch.loggers.wandb,
        "WandbLogger",
        FakeWandbLogger,
    )
    monkeypatch.setattr(run_module, "log_hyperparameters", MagicMock())

    metrics, objects = run_module.run(cfg)

    assert instantiate_spy.call_args_list[:2] == [
        call(cfg.data_pipeline),
        call(
            cfg.trainer,
            callbacks=callbacks,
            logger=[wandb_logger],
            num_sanity_val_steps=0,
            log_every_n_steps=1,
        ),
    ]
    instantiate_model_spy.assert_called_once_with(cfg, data_spec=data_spec)
    pipeline.build.assert_called_once_with(cfg)
    assert objects["datamodule"] is datamodule
    assert objects["data_spec"] is data_spec
    assert metrics == trainer.callback_metrics
    assert wandb_logger.logged == [{"preprocessor_time": 3.5}]


def test_simplified_runner_consumes_pipeline_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The test runner delegates data construction to the same boundary."""
    first_batch = SimpleNamespace(num_graphs=6)
    datamodule = MagicMock(spec=LightningDataModule)
    datamodule.train_dataloader.return_value = iter([first_batch])
    pipeline = MagicMock()
    pipeline.build.return_value = DataPipelineOutput(
        datamodule=datamodule,
        preprocessing_time=0.5,
    )
    model = MagicMock()
    trainer = MagicMock()
    trainer.callback_metrics = {
        "train/loss": torch.tensor(1.0),
        "val/loss": torch.tensor(0.75),
    }
    trainer.current_epoch = 2
    trainer.checkpoint_callback.best_model_path = "/tmp/best.ckpt"
    trainer.test.return_value = [{"test/loss": 0.5}]

    cfg = OmegaConf.create(
        {
            "seed": 11,
            "data_pipeline": {"_target_": "tests.Pipeline"},
            "model": {"_target_": "tests.Model"},
            "evaluator": {},
            "optimizer": {},
            "loss": {},
            "callbacks": {},
            "trainer": {"_target_": "tests.Trainer"},
            "ckpt_path": None,
        }
    )

    def instantiate(config: DictConfig, **kwargs: object) -> object:
        del kwargs
        if config is cfg.data_pipeline:
            return pipeline
        if config is cfg.model:
            return model
        if config is cfg.trainer:
            return trainer
        raise AssertionError(f"Unexpected instantiate call: {config!r}")

    instantiate_spy = MagicMock(side_effect=instantiate)
    monkeypatch.setattr(
        simplified_pipeline.hydra.utils,
        "instantiate",
        instantiate_spy,
    )
    monkeypatch.setattr(
        simplified_pipeline,
        "instantiate_callbacks",
        MagicMock(return_value=[]),
    )

    result = simplified_pipeline.run(cfg)

    assert instantiate_spy.call_args_list[0] == call(cfg.data_pipeline)
    pipeline.build.assert_called_once_with(cfg)
    datamodule.train_dataloader.assert_called_once_with()
    assert result["observed_train_batch_size"] == 6
    assert result["epochs_completed"] == 2
    assert result["fit_metrics"] == {
        "train/loss": 1.0,
        "val/loss": 0.75,
    }
    assert result["test_results"] == [{"test/loss": 0.5}]


def test_pipeline_package_exposes_canonical_api() -> None:
    """Canonical package exports remain stable after normal test imports."""
    import topobench.data.pipelines as pipelines

    assert ",".join(pipelines.__all__) == (
        "AbstractDataPipeline,DataPipelineOutput,DefaultDataPipeline,"
        "HeterogeneousNodeDataPipeline"
    )
    assert issubclass(DefaultDataPipeline, AbstractDataPipeline)
    assert issubclass(HeterogeneousNodeDataPipeline, AbstractDataPipeline)


class _FakePreprocessor:
    """Small sequence-shaped preprocessor spy for pipeline contract tests."""

    def __init__(
        self,
        items: list[object],
        *,
        preprocessing_time: float = 1.75,
    ) -> None:
        self.items = items
        self.preprocessing_time = preprocessing_time
        self.load_dataset_splits = MagicMock(
            side_effect=AssertionError(
                "heterogeneous pipeline must not load homogeneous splits"
            )
        )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> object:
        return self.items[index]


def _heterogeneous_pipeline_cfg(
    *,
    dataloader_params: dict[str, object] | None = None,
) -> DictConfig:
    """Return the smallest heterogeneous pipeline configuration."""
    return OmegaConf.create(
        {
            "dataset": {
                "parameters": {
                    "target_node_type": "author",
                    "num_classes": 2,
                },
                "dataloader_params": dataloader_params
                or {
                    "mode": "neighbor",
                    "batch_size": 4,
                    "num_neighbors": [3, 2],
                    "evaluation_protocol": "sampled_neighbor_fixed",
                    "evaluation_seed": 17,
                    "num_workers": 0,
                    "pin_memory": False,
                    "persistent_workers": False,
                },
            }
        }
    )


def _fully_featured_heterogeneous_data() -> HeteroData:
    """Return transformed native data satisfying the validation contract."""
    from topobench.transforms.data_manipulations.heterogeneous import (
        HeterogeneousConstantFeatures,
        HeterogeneousToUndirected,
    )

    data = make_synthetic_heterogeneous_data(seed=7)
    data = HeterogeneousConstantFeatures(node_types="venue")(data)
    return HeterogeneousToUndirected(merge=False)(data)


def test_heterogeneous_pipeline_validates_and_passes_exact_dataloader_params(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The native pipeline validates once and constructs the separate module."""
    import topobench.data.pipelines.heterogeneous as module

    data = _fully_featured_heterogeneous_data()
    preprocessor = _FakePreprocessor([data])
    cfg = _heterogeneous_pipeline_cfg()
    datamodule = MutableDataModule()
    constructor = MagicMock(return_value=datamodule)
    monkeypatch.setattr(
        HeterogeneousNodeDataPipeline,
        "preprocess",
        MagicMock(return_value=preprocessor),
    )
    monkeypatch.setattr(
        module,
        "HeterogeneousNodeDataModule",
        constructor,
    )

    output = HeterogeneousNodeDataPipeline().build(cfg)

    assert output.datamodule is datamodule
    assert output.preprocessing_time == 1.75
    assert output.data_spec is not None
    assert output.data_spec.pyg_metadata() == data.metadata()
    assert output.data_spec.target_node_type == "author"
    assert output.data_spec.input_channels_dict == {
        "author": 8,
        "paper": 5,
        "venue": 1,
    }
    constructor.assert_called_once_with(
        data=data,
        spec=output.data_spec,
        mode="neighbor",
        batch_size=4,
        num_neighbors=[3, 2],
        evaluation_protocol="sampled_neighbor_fixed",
        evaluation_seed=17,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
    )
    preprocessor.load_dataset_splits.assert_not_called()
    assert not hasattr(module, "TBDataloader")


@pytest.mark.parametrize(
    ("items", "error_type", "message"),
    [
        ([], ValueError, "requires exactly one processed graph; received 0"),
        (
            [HeteroData(), HeteroData()],
            ValueError,
            "requires exactly one processed graph; received 2",
        ),
        ([Data(x=torch.ones(2, 3))], TypeError, "requires native HeteroData"),
    ],
    ids=["zero-graphs", "multiple-graphs", "homogeneous-data"],
)
def test_heterogeneous_pipeline_rejects_invalid_processed_output(
    monkeypatch: pytest.MonkeyPatch,
    items: list[object],
    error_type: type[Exception],
    message: str,
) -> None:
    """Graph cardinality and native family fail before datamodule creation."""
    import topobench.data.pipelines.heterogeneous as module

    preprocessor = _FakePreprocessor(items)
    constructor = MagicMock()
    monkeypatch.setattr(
        HeterogeneousNodeDataPipeline,
        "preprocess",
        MagicMock(return_value=preprocessor),
    )
    monkeypatch.setattr(
        module,
        "HeterogeneousNodeDataModule",
        constructor,
    )

    with pytest.raises(error_type, match=message):
        HeterogeneousNodeDataPipeline().build(_heterogeneous_pipeline_cfg())

    constructor.assert_not_called()
    preprocessor.load_dataset_splits.assert_not_called()


def test_heterogeneous_pipeline_propagates_validation_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Schema validation remains the sole actionable contract boundary."""
    import topobench.data.pipelines.heterogeneous as module

    preprocessor = _FakePreprocessor([_fully_featured_heterogeneous_data()])
    validator = MagicMock(
        side_effect=ValueError("target train_mask must be non-empty")
    )
    constructor = MagicMock()
    monkeypatch.setattr(
        HeterogeneousNodeDataPipeline,
        "preprocess",
        MagicMock(return_value=preprocessor),
    )
    monkeypatch.setattr(
        module,
        "validate_heterogeneous_node_data",
        validator,
    )
    monkeypatch.setattr(
        module,
        "HeterogeneousNodeDataModule",
        constructor,
    )

    with pytest.raises(ValueError, match="train_mask must be non-empty"):
        HeterogeneousNodeDataPipeline().build(_heterogeneous_pipeline_cfg())

    validator.assert_called_once()
    constructor.assert_not_called()
    preprocessor.load_dataset_splits.assert_not_called()


def test_real_synthetic_heterogeneous_pipeline_uses_shared_preprocessor(
    tmp_path: Path,
) -> None:
    """Real loader and default transforms produce one validated full batch."""
    register_all_resolvers()
    hydra.initialize(version_base="1.3", config_path="../../../configs")
    cfg = hydra.compose(
        config_name="run.yaml",
        overrides=[
            "dataset=heterogeneous/SyntheticHeterogeneous",
            "model=cell/hgt",
            "data_pipeline=heterogeneous_node",
            f"paths.data_dir={tmp_path}",
            "train=false",
            "test=false",
        ],
    )

    output = HeterogeneousNodeDataPipeline().build(cfg)
    batch = next(iter(output.datamodule.train_dataloader()))

    assert isinstance(batch, HeteroData)
    assert batch.num_graphs == 1
    assert output.data_spec is not None
    assert output.data_spec.pyg_metadata() == batch.metadata()
    assert output.data_spec.target_node_type == "author"
    assert "x" in batch["venue"]
    assert ("paper", "rev_writes", "author") in batch.edge_types


def test_heterogeneous_hydra_composition_resolves_pipeline_and_transforms() -> (
    None
):
    """Synthetic native data composes the dedicated pipeline and transforms."""
    register_all_resolvers()
    hydra.initialize(version_base="1.3", config_path="../../../configs")
    cfg = hydra.compose(
        config_name="run.yaml",
        overrides=[
            "dataset=heterogeneous/SyntheticHeterogeneous",
            "model=cell/hgt",
            "data_pipeline=heterogeneous_node",
            "train=false",
            "test=false",
        ],
    )

    assert cfg.data_pipeline._target_ == (
        "topobench.data.pipelines.HeterogeneousNodeDataPipeline"
    )
    assert cfg.dataset.dataloader_params.evaluation_protocol == "full_graph"
    assert cfg.dataset.dataloader_params.evaluation_seed == cfg.seed
    assert cfg.transforms.venue_features.transform_name == (
        "HeterogeneousConstantFeatures"
    )
    assert cfg.transforms.reverse_relations.transform_name == (
        "HeterogeneousToUndirected"
    )


def test_default_pipeline_uses_native_graph_datamodule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The default pipeline keeps homogeneous PyG data native through batching."""
    cfg = _pipeline_cfg(task_level="graph", transforms=None)
    events, *_ = _install_pipeline_spies(monkeypatch, cfg)

    output = DefaultDataPipeline().build(cfg)

    assert output.data_spec is None
    assert [
        event if isinstance(event, str) else event[0] for event in events
    ] == [
        "instantiate_loader",
        "load",
        "preprocessor",
        "load_splits",
        "prepare_graph_features",
        "datamodule",
    ]


def test_neighbor_mode_override_requires_explicit_protocol_override(
    tmp_path: Path,
) -> None:
    """Changing only mode cannot silently mislabel evaluation semantics."""
    register_all_resolvers()
    hydra.initialize(version_base="1.3", config_path="../../../configs")
    cfg = hydra.compose(
        config_name="run.yaml",
        overrides=[
            "dataset=heterogeneous/SyntheticHeterogeneous",
            "model=cell/hgt",
            "data_pipeline=heterogeneous_node",
            "dataset.dataloader_params.mode=neighbor",
            f"paths.data_dir={tmp_path}",
            "train=false",
            "test=false",
        ],
    )

    assert cfg.dataset.dataloader_params.evaluation_protocol == "full_graph"
    with pytest.raises(
        ValueError,
        match=r"evaluation_protocol.*mode",
    ):
        HeterogeneousNodeDataPipeline().build(cfg)
