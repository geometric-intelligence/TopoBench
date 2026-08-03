"""Tests for runtime-aware Hydra model construction."""

from __future__ import annotations

import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import hydra
import pytest
import torch
from lightning import LightningDataModule
from omegaconf import DictConfig, OmegaConf, open_dict
from torch_geometric.data import HeteroData

from topobench.data import HeterogeneousDataSpec
from topobench.model import (
    HeterogeneousNodeSupervisionAdapter,
    TBModel,
)
from topobench.nn.backbones.heterogeneous import (
    HeteroSAGEBackbone,
    HGTBackbone,
)
from topobench.nn.encoders import HeterogeneousNodeFeatureEncoder
from topobench.nn.readouts import HeterogeneousNodeReadout
from topobench.nn.wrappers.heterogeneous import HeterogeneousWrapper
from topobench.utils.config_resolvers import register_all_resolvers
from topobench.utils.model_instantiation import instantiate_model

PROJECT_ROOT = Path(__file__).parents[2]


@pytest.fixture
def heterogeneous_spec() -> HeterogeneousDataSpec:
    """Return immutable metadata with unequal feature widths."""
    return HeterogeneousDataSpec(
        node_types=("author", "paper"),
        edge_types=(
            ("author", "writes", "paper"),
            ("paper", "written_by", "author"),
        ),
        target_node_type="author",
        num_classes=3,
        input_channels=(("author", 5), ("paper", 7)),
    )


def _compose(
    model: str,
    *,
    overrides: list[str] | None = None,
) -> DictConfig:
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    register_all_resolvers()
    with hydra.initialize(version_base="1.3", config_path="../../configs"):
        return hydra.compose(
            config_name="run.yaml",
            overrides=[
                f"model={model}",
                "dataset=heterogeneous/SyntheticHeterogeneous",
                *(overrides or []),
            ],
        )


def _snapshot(cfg: DictConfig) -> object:
    """Return a primitive, unresolved structural snapshot."""
    return OmegaConf.to_container(cfg, resolve=False)


@pytest.mark.parametrize(
    ("model_name", "backbone_type"),
    [
        ("heterogeneous/hgt", HGTBackbone),
        ("heterogeneous/heterosage", HeteroSAGEBackbone),
    ],
)
def test_heterogeneous_runtime_metadata_reaches_real_components(
    model_name: str,
    backbone_type: type,
    heterogeneous_spec: HeterogeneousDataSpec,
) -> None:
    """Runtime metadata must configure every real heterogeneous component."""
    cfg = _compose(model_name)
    before = _snapshot(cfg)

    model = instantiate_model(cfg, data_spec=heterogeneous_spec)

    assert isinstance(model, TBModel)
    assert isinstance(model.feature_encoder, HeterogeneousNodeFeatureEncoder)
    assert model.feature_encoder.input_channels == {"author": 5, "paper": 7}
    assert isinstance(model.backbone, HeterogeneousWrapper)
    assert isinstance(model.backbone.backbone, backbone_type)
    assert (
        model.backbone.backbone.metadata == heterogeneous_spec.pyg_metadata()
    )
    assert model.backbone.target_node_type == "author"
    assert isinstance(model.readout, HeterogeneousNodeReadout)
    assert model.readout.target_node_type == "author"
    assert model.readout.out_channels == 3
    assert isinstance(
        model.supervision_adapter,
        HeterogeneousNodeSupervisionAdapter,
    )
    assert model.supervision_adapter.target_node_type == "author"
    assert model.supervision_adapter.mode == "full_batch"
    assert _snapshot(cfg) == before


def test_hgt_and_heterosage_receive_identical_metadata(
    heterogeneous_spec: HeterogeneousDataSpec,
) -> None:
    """The two baselines must share exactly one runtime metadata contract."""
    hgt = instantiate_model(
        _compose("heterogeneous/hgt"),
        data_spec=heterogeneous_spec,
    )
    sage = instantiate_model(
        _compose("heterogeneous/heterosage"),
        data_spec=heterogeneous_spec,
    )

    assert hgt.backbone.backbone.metadata == sage.backbone.backbone.metadata
    assert hgt.backbone.backbone.metadata == heterogeneous_spec.pyg_metadata()


def test_default_real_homogeneous_model_is_unchanged() -> None:
    """The helper's homogeneous path must remain the direct Hydra path."""
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    register_all_resolvers()
    with hydra.initialize(version_base="1.3", config_path="../../configs"):
        cfg = hydra.compose(
            config_name="run.yaml",
            overrides=["model=graph/gcn", "dataset=graph/MUTAG"],
        )
    expected = hydra.utils.instantiate(
        cfg.model,
        evaluator=cfg.evaluator,
        optimizer=cfg.optimizer,
        loss=cfg.loss,
    )
    actual = instantiate_model(cfg, data_spec=None)

    assert type(actual) is type(expected) is TBModel
    assert type(actual.feature_encoder) is type(expected.feature_encoder)
    assert type(actual.backbone) is type(expected.backbone)
    assert type(actual.readout) is type(expected.readout)
    assert actual.hparams.model_name == expected.hparams.model_name == "gcn"


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("feature_encoder.input_channels", {"author": 5, "paper": 7}),
        ("backbone.metadata", [["author", "paper"], []]),
        ("backbone_wrapper.target_node_type", "author"),
        ("readout.target_node_type", "author"),
        ("readout.out_channels", 3),
        ("supervision_adapter.target_node_type", "author"),
        ("supervision_adapter.mode", "full_batch"),
    ],
)
def test_static_runtime_values_are_rejected_without_mutating_source(
    path: str,
    value: object,
    heterogeneous_spec: HeterogeneousDataSpec,
) -> None:
    """Even matching static metadata must not override validated graph data."""
    cfg = _compose("heterogeneous/hgt")
    parent, leaf = path.rsplit(".", maxsplit=1)
    with open_dict(cfg.model[parent]):
        cfg.model[parent][leaf] = value
    before = _snapshot(cfg)

    with pytest.raises(ValueError, match=f"model\\.{path}"):
        instantiate_model(cfg, data_spec=heterogeneous_spec)

    assert _snapshot(cfg) == before


@pytest.mark.parametrize(
    ("cfg_factory", "data_spec", "error"),
    [
        (
            lambda: OmegaConf.create({"model": {"model_domain": "graph"}}),
            "invalid",
            TypeError,
        ),
        (
            lambda: _compose("heterogeneous/hgt"),
            None,
            ValueError,
        ),
        (
            lambda: _compose("graph/gcn"),
            "valid",
            ValueError,
        ),
    ],
)
def test_domain_and_data_spec_mismatches_fail_without_mutation(
    cfg_factory,
    data_spec: str | None,
    error: type[Exception],
    heterogeneous_spec: HeterogeneousDataSpec,
) -> None:
    """Model domain and validated data-family metadata must agree."""
    cfg = cfg_factory()
    before = _snapshot(cfg)
    supplied = heterogeneous_spec if data_spec == "valid" else data_spec

    with pytest.raises(error):
        instantiate_model(cfg, data_spec=supplied)  # type: ignore[arg-type]

    assert _snapshot(cfg) == before


@pytest.mark.parametrize(
    ("overrides", "message_parts"),
    [
        (
            [
                "dataset.dataloader_params.mode=neighbor",
                "+dataset.dataloader_params.num_neighbors=[10]",
                "+dataset.dataloader_params.subgraph_type=directional",
            ],
            ("model depth=2", "observed fanout depths=[1]", "remedy"),
        ),
        (
            [
                "dataset.dataloader_params.mode=neighbor",
                "+dataset.dataloader_params.num_neighbors=[10,10]",
                "+dataset.dataloader_params.subgraph_type=bidirectional",
            ],
            ("directional",),
        ),
    ],
)
def test_neighbor_depth_and_subgraph_contracts_are_diagnostic_and_immutable(
    overrides: list[str],
    message_parts: tuple[str, ...],
    heterogeneous_spec: HeterogeneousDataSpec,
) -> None:
    """Neighbor sampling topology must agree with model message-passing depth."""
    cfg = _compose("heterogeneous/hgt", overrides=overrides)
    before = _snapshot(cfg)

    with pytest.raises(ValueError) as error:
        instantiate_model(cfg, data_spec=heterogeneous_spec)

    for part in message_parts:
        assert part in str(error.value)
    assert _snapshot(cfg) == before


@pytest.mark.parametrize(
    "fanout",
    [None, "10,10", [], {}, {"author__writes__paper": []}],
)
def test_malformed_neighbor_fanouts_are_rejected(
    fanout: object,
    heterogeneous_spec: HeterogeneousDataSpec,
) -> None:
    """The helper validates fanout shape, leaving fanout values to loaders."""
    cfg = _compose("heterogeneous/hgt")
    with open_dict(cfg.dataset.dataloader_params):
        cfg.dataset.dataloader_params.mode = "neighbor"
        cfg.dataset.dataloader_params.subgraph_type = "directional"
        cfg.dataset.dataloader_params.num_neighbors = fanout
    before = _snapshot(cfg)

    with pytest.raises((TypeError, ValueError), match="num_neighbors|fanout"):
        instantiate_model(cfg, data_spec=heterogeneous_spec)

    assert _snapshot(cfg) == before


def test_relation_specific_fanout_depths_are_all_reported(
    heterogeneous_spec: HeterogeneousDataSpec,
) -> None:
    """Every relation-specific depth participates in the compatibility error."""
    cfg = _compose("heterogeneous/hgt")
    with open_dict(cfg.dataset.dataloader_params):
        cfg.dataset.dataloader_params.mode = "neighbor"
        cfg.dataset.dataloader_params.subgraph_type = "directional"
        cfg.dataset.dataloader_params.num_neighbors = {
            "writes": [15, 10],
            "written_by": [15, 10, 5],
        }
    before = _snapshot(cfg)

    with pytest.raises(ValueError) as error:
        instantiate_model(cfg, data_spec=heterogeneous_spec)

    message = str(error.value)
    assert "model depth=2" in message
    assert "observed fanout depths=[2, 3]" in message
    assert "remedy" in message
    assert _snapshot(cfg) == before


def test_relation_specific_neighbor_fanout_instantiates_neighbor_adapter(
    heterogeneous_spec: HeterogeneousDataSpec,
) -> None:
    """Relation-specific ListConfig values support a matching model depth."""
    cfg = _compose("heterogeneous/heterosage")
    with open_dict(cfg.dataset.dataloader_params):
        cfg.dataset.dataloader_params.mode = "neighbor"
        cfg.dataset.dataloader_params.subgraph_type = "directional"
        cfg.dataset.dataloader_params.num_neighbors = {
            "writes": [15, 10],
            "written_by": [20, 5],
        }
    before = _snapshot(cfg)

    model = instantiate_model(cfg, data_spec=heterogeneous_spec)

    assert model.supervision_adapter.mode == "neighbor"
    assert _snapshot(cfg) == before


@pytest.mark.parametrize("depth", [True, 0, -1, 2.0])
def test_neighbor_model_depth_must_be_nonboolean_positive_integer(
    depth: object,
    heterogeneous_spec: HeterogeneousDataSpec,
) -> None:
    """Sampling compatibility never silently coerces invalid model depth."""
    cfg = _compose("heterogeneous/hgt")
    with open_dict(cfg.dataset.dataloader_params):
        cfg.dataset.dataloader_params.mode = "neighbor"
        cfg.dataset.dataloader_params.subgraph_type = "directional"
        cfg.dataset.dataloader_params.num_neighbors = [15, 10]
    with open_dict(cfg.model.backbone):
        cfg.model.backbone.num_layers = depth
    before = _snapshot(cfg)

    with pytest.raises(
        (TypeError, ValueError),
        match="model.backbone.num_layers",
    ):
        instantiate_model(cfg, data_spec=heterogeneous_spec)

    assert _snapshot(cfg) == before


def test_missing_runtime_placeholder_is_rejected_with_exact_path(
    heterogeneous_spec: HeterogeneousDataSpec,
) -> None:
    """Runtime injection requires deliberate null fields, not absent keys."""
    cfg = _compose("heterogeneous/hgt")
    with open_dict(cfg.model.readout):
        del cfg.model.readout["out_channels"]
    before = _snapshot(cfg)

    with pytest.raises(ValueError, match="model.readout.out_channels"):
        instantiate_model(cfg, data_spec=heterogeneous_spec)

    assert _snapshot(cfg) == before


@pytest.mark.parametrize("resolver_value", [None, "statically-wrong"])
def test_runtime_placeholder_interpolation_is_rejected_without_resolution(
    resolver_value: object,
    heterogeneous_spec: HeterogeneousDataSpec,
) -> None:
    """Even null-resolving interpolation is not a literal null placeholder."""
    resolver_name = "task14_placeholder_probe"
    calls: list[object] = []
    OmegaConf.register_new_resolver(
        resolver_name,
        lambda: calls.append(resolver_value) or resolver_value,
        replace=True,
    )
    cfg = _compose("heterogeneous/hgt")
    with open_dict(cfg.model.readout):
        cfg.model.readout.out_channels = f"${{{resolver_name}:}}"
    before = _snapshot(cfg)

    try:
        with pytest.raises(ValueError, match="model.readout.out_channels"):
            instantiate_model(cfg, data_spec=heterogeneous_spec)
    finally:
        OmegaConf.clear_resolver(resolver_name)

    assert calls == []
    assert _snapshot(cfg) == before


def test_missing_placeholder_and_environment_interpolation_do_not_leak(
    heterogeneous_spec: HeterogeneousDataSpec,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing and environment-derived runtime fields fail before resolution."""
    secret = "must-never-appear-in-errors"
    monkeypatch.setenv("TOPOBENCH_TASK14_SECRET", secret)
    values = ["???", "${oc.env:TOPOBENCH_TASK14_SECRET}"]
    for value in values:
        cfg = _compose("heterogeneous/hgt")
        with open_dict(cfg.model.supervision_adapter):
            cfg.model.supervision_adapter.mode = value
        before = _snapshot(cfg)

        with pytest.raises(ValueError) as error:
            instantiate_model(cfg, data_spec=heterogeneous_spec)

        assert "model.supervision_adapter.mode" in str(error.value)
        assert secret not in str(error.value)
        assert _snapshot(cfg) == before


def test_instantiation_failure_does_not_mutate_source(
    heterogeneous_spec: HeterogeneousDataSpec,
) -> None:
    """Errors after injection remain isolated inside the copied root."""
    cfg = _compose("heterogeneous/hgt")
    with open_dict(cfg.model.readout):
        cfg.model.readout._target_ = "missing.package.Readout"
    before = _snapshot(cfg)

    with pytest.raises(Exception, match="missing.package.Readout"):
        instantiate_model(cfg, data_spec=heterogeneous_spec)

    assert _snapshot(cfg) == before


def test_helper_requires_dictconfig_root() -> None:
    """Plain mappings cannot preserve Hydra interpolation parent context."""
    with pytest.raises(TypeError, match="cfg must be a DictConfig"):
        instantiate_model({}, data_spec=None)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "model_name",
    ["heterogeneous/hgt", "heterogeneous/heterosage"],
)
def test_readonly_structured_source_is_unchanged(
    model_name: str,
    heterogeneous_spec: HeterogeneousDataSpec,
) -> None:
    """Construction supports immutable composed roots without changing flags."""
    cfg = _compose(model_name)
    OmegaConf.set_struct(cfg, True)
    OmegaConf.set_readonly(cfg, True)
    before = _snapshot(cfg)

    model = instantiate_model(cfg, data_spec=heterogeneous_spec)

    assert isinstance(model, TBModel)
    assert OmegaConf.is_struct(cfg) is True
    assert OmegaConf.is_readonly(cfg) is True
    assert _snapshot(cfg) == before


def test_readonly_structured_source_flags_survive_error(
    heterogeneous_spec: HeterogeneousDataSpec,
) -> None:
    """Validation errors cannot relax immutable source configuration flags."""
    cfg = _compose("heterogeneous/hgt")
    with open_dict(cfg.model.readout):
        cfg.model.readout.out_channels = 3
    OmegaConf.set_struct(cfg, True)
    OmegaConf.set_readonly(cfg, True)
    before = _snapshot(cfg)

    with pytest.raises(ValueError, match="model.readout.out_channels"):
        instantiate_model(cfg, data_spec=heterogeneous_spec)

    assert OmegaConf.is_struct(cfg) is True
    assert OmegaConf.is_readonly(cfg) is True
    assert _snapshot(cfg) == before


@pytest.mark.parametrize(
    "literal",
    [
        "${model.model_name}",
        "${oc.env:TOPOBENCH_TASK14_SECRET}",
        "???",
        r"back\\slash",
        "节点",
        "surrogate-\udcff",
    ],
)
@pytest.mark.parametrize(
    "model_name",
    ["heterogeneous/hgt", "heterogeneous/heterosage"],
)
def test_data_derived_names_remain_literal_and_forward(
    model_name: str,
    literal: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Runtime graph strings never cross Hydra's interpolation parser."""
    monkeypatch.setenv(
        "TOPOBENCH_TASK14_SECRET",
        "must-not-replace-literal-node-name",
    )
    other_type = "other"
    reverse_relation = f"reverse::{literal}"
    spec = HeterogeneousDataSpec(
        node_types=(literal, other_type),
        edge_types=(
            (literal, literal, other_type),
            (other_type, reverse_relation, literal),
        ),
        target_node_type=literal,
        num_classes=3,
        input_channels=((literal, 5), (other_type, 7)),
    )
    cfg = _compose(model_name)
    model = instantiate_model(cfg, data_spec=spec)
    data = HeteroData()
    data[literal].x = torch.randn(4, 5)
    data[literal].y = torch.tensor([0, 1, 2, 0])
    data[other_type].x = torch.randn(3, 7)
    data[literal, literal, other_type].edge_index = torch.tensor(
        [[0, 1, 2, 3], [0, 1, 2, 0]],
    )
    data[other_type, reverse_relation, literal].edge_index = torch.tensor(
        [[0, 1, 2, 0], [0, 1, 2, 3]],
    )

    result = model(data)

    assert model.feature_encoder.input_channels == {literal: 5, other_type: 7}
    assert model.backbone.target_node_type == literal
    assert model.backbone.backbone.metadata == spec.pyg_metadata()
    assert model.readout.target_node_type == literal
    assert model.supervision_adapter.target_node_type == literal
    assert result["logits"].shape == (4, 3)


def test_data_derived_interpolation_syntax_never_invokes_resolver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A resolver-shaped graph name remains opaque through construction."""
    resolver_name = "task14_data_probe"
    calls: list[str] = []
    OmegaConf.register_new_resolver(
        resolver_name,
        lambda: calls.append("called") or "resolved",
        replace=True,
    )
    literal = f"${{{resolver_name}:}}"
    spec = HeterogeneousDataSpec(
        node_types=(literal, "other"),
        edge_types=((literal, literal, "other"),),
        target_node_type=literal,
        num_classes=2,
        input_channels=((literal, 3), ("other", 4)),
    )
    cfg = _compose("heterogeneous/hgt")
    try:
        model = instantiate_model(cfg, data_spec=spec)
    finally:
        OmegaConf.clear_resolver(resolver_name)

    assert calls == []
    assert model.readout.target_node_type == literal
    assert model.backbone.backbone.metadata == spec.pyg_metadata()


def test_copy_isolation_preserves_absolute_interpolation_context(
    heterogeneous_spec: HeterogeneousDataSpec,
) -> None:
    """Absolute model interpolations resolve on the copied root only."""
    cfg = _compose("heterogeneous/hgt")
    before = deepcopy(_snapshot(cfg))
    model = instantiate_model(cfg, data_spec=heterogeneous_spec)

    assert model.feature_encoder.hidden_channels == 64
    assert model.backbone.backbone.hidden_channels == 64
    assert model.readout.hidden_channels == 64
    assert _snapshot(cfg) == before


@pytest.mark.parametrize(
    "model_name",
    ["heterogeneous/hgt", "heterogeneous/heterosage"],
)
def test_clean_process_resolves_and_instantiates_every_target(
    model_name: str,
) -> None:
    """Canonical package exports must work without import-order side effects."""
    script = f"""
import hydra
from topobench.data import HeterogeneousDataSpec
from topobench.utils.config_resolvers import register_all_resolvers
from topobench.utils.model_instantiation import instantiate_model
register_all_resolvers()
with hydra.initialize(version_base="1.3", config_path="configs"):
    cfg = hydra.compose(
        config_name="run.yaml",
        overrides=[
            "model={model_name}",
            "dataset=heterogeneous/SyntheticHeterogeneous",
        ],
    )
spec = HeterogeneousDataSpec(
    node_types=("author", "paper"),
    edge_types=(
        ("author", "writes", "paper"),
        ("paper", "written_by", "author"),
    ),
    target_node_type="author",
    num_classes=3,
    input_channels=(("author", 5), ("paper", 7)),
)
model = instantiate_model(cfg, data_spec=spec)
print(type(model).__module__ + "." + type(model).__name__)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip().endswith("topobench.model.model.TBModel")


def test_run_passes_pipeline_data_spec_to_central_helper(
    monkeypatch: pytest.MonkeyPatch,
    heterogeneous_spec: HeterogeneousDataSpec,
) -> None:
    """The application entry point must not duplicate metadata injection."""
    import topobench.run as run_module

    datamodule = LightningDataModule()
    pipeline_output = SimpleNamespace(
        datamodule=datamodule,
        preprocessing_time=0.0,
        data_spec=heterogeneous_spec,
    )
    pipeline = MagicMock()
    pipeline.build.return_value = pipeline_output
    trainer = MagicMock()
    trainer.callback_metrics = {}
    model = MagicMock()
    instantiate_model_mock = MagicMock(return_value=model)

    def instantiate_side_effect(config, **kwargs):
        del kwargs
        return pipeline if config is cfg.data_pipeline else trainer

    cfg = OmegaConf.create(
        {
            "seed": 7,
            "data_pipeline": {"_target_": "tests.FakePipeline"},
            "model": {"_target_": "tests.FakeModel"},
            "evaluator": {},
            "optimizer": {},
            "loss": {},
            "trainer": {"_target_": "tests.FakeTrainer"},
            "paths": {"output_dir": "test-output"},
            "callbacks": None,
            "logger": None,
            "train": False,
            "test": False,
        }
    )
    monkeypatch.setattr(
        run_module.hydra.utils,
        "instantiate",
        instantiate_side_effect,
    )
    monkeypatch.setattr(
        run_module,
        "instantiate_model",
        instantiate_model_mock,
        raising=False,
    )
    preflight_result = SimpleNamespace(passed=True, qualified=True)
    preflight_runner = MagicMock()
    preflight_runner.return_value.validate_static.return_value = preflight_result
    preflight_runner.return_value.run_probe.return_value = preflight_result
    monkeypatch.setattr(run_module, "PreflightRunner", preflight_runner)
    monkeypatch.setattr(
        run_module,
        "instantiate_callbacks",
        lambda _, **kwargs: [],
    )
    monkeypatch.setattr(run_module, "instantiate_loggers", lambda _: [])

    _, objects = run_module.run(cfg)

    instantiate_model_mock.assert_called_once_with(
        cfg,
        data_spec=heterogeneous_spec,
    )
    assert objects["model"] is model
