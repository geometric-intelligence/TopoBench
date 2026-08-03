"""Tests for runtime-aware Hydra model construction."""

from __future__ import annotations

import subprocess
import sys
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, call

import hydra
import pytest
import torch
from lightning import LightningDataModule
from omegaconf import DictConfig, OmegaConf, open_dict
from torch_geometric.data import HeteroData
from torch_geometric.nn.models import GraphSAGE

from topobench.data import HeterogeneousDataSpec
from topobench.data.capabilities import RuntimeDataCapability
from topobench.model import (
    HeterogeneousNodeSupervisionAdapter,
    TBModel,
)
from topobench.nn.backbones.heterogeneous import (
    HeteroSAGEBackbone,
    HGTBackbone,
)
from topobench.nn.capabilities import validate_capability_composition
from topobench.nn.encoders import HeterogeneousNodeFeatureEncoder
from topobench.nn.readouts import HeterogeneousNodeReadout
from topobench.nn.wrappers.heterogeneous import HeterogeneousWrapper
from topobench.utils.config_resolvers import register_all_resolvers
from topobench.utils.instantiators import (
    validate_execution_profile,
    validate_profile_capability,
)
from topobench.utils.model_instantiation import instantiate_model

PROJECT_ROOT = Path(__file__).parents[2]


@pytest.fixture
def heterogeneous_spec() -> HeterogeneousDataSpec:
    """Return immutable metadata with unequal feature widths."""
    return HeterogeneousDataSpec(
        node_types=("author", "paper", "venue"),
        edge_types=(
            ("author", "writes", "paper"),
            ("paper", "published_in", "venue"),
        ),
        target_node_type="author",
        num_classes=2,
        input_channels=(("author", 8), ("paper", 5), ("venue", 1)),
    )


def _compose_capability_pair(
    dataset_selector: str,
    model_selector: str,
) -> DictConfig:
    domain = dataset_selector.partition("/")[0]
    pipeline = {
        "graph": "default",
        "heterogeneous": "heterogeneous_node",
        "hypergraph": "hypergraph_node",
    }[domain]
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    register_all_resolvers()
    with hydra.initialize(version_base="1.3", config_path="../../configs"):
        return hydra.compose(
            config_name="run.yaml",
            overrides=[
                f"dataset={dataset_selector}",
                f"model={model_selector}",
                f"data_pipeline={pipeline}",
            ],
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
                "data_pipeline=heterogeneous_node",
                *(overrides or []),
            ],
        )


def _snapshot(cfg: DictConfig) -> object:
    """Return a primitive, unresolved structural snapshot."""
    return OmegaConf.to_container(cfg, resolve=False)


def _observed_validation(cfg: DictConfig):
    """Return exact runtime evidence for tests that bypass a data pipeline."""
    profile_record = validate_execution_profile(cfg)
    static = validate_profile_capability(
        cfg,
        profile_record=profile_record,
    )
    qualification = static.dataset
    domain = qualification.selector.partition("/")[0]
    output_kind = (
        "graph"
        if domain == "graph" and qualification.task_level == "graph"
        else "homogeneous"
        if domain == "graph"
        else domain
    )
    observed = RuntimeDataCapability(
        selector=qualification.selector,
        data_domain=domain,
        output_kind=output_kind,
        feature_widths=qualification.feature_widths,
        num_classes=qualification.num_classes,
        target_node_type=qualification.target_node_type,
    )
    return replace(static, observed=observed)


def _instantiate_observed_model(
    cfg: DictConfig,
    *,
    data_spec: HeterogeneousDataSpec | None,
) -> TBModel:
    """Instantiate through the same post-load capability gate as production."""
    model = instantiate_model(
        cfg,
        data_spec=data_spec,
        capability_validation=_observed_validation(cfg),
    )
    assert isinstance(model, TBModel)
    return model


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

    model = _instantiate_observed_model(cfg, data_spec=heterogeneous_spec)

    assert isinstance(model, TBModel)
    assert isinstance(model.feature_encoder, HeterogeneousNodeFeatureEncoder)
    assert model.feature_encoder.input_channels == {
        "author": 8,
        "paper": 5,
        "venue": 1,
    }
    assert isinstance(model.backbone, HeterogeneousWrapper)
    assert isinstance(model.backbone.backbone, backbone_type)
    assert (
        model.backbone.backbone.metadata == heterogeneous_spec.pyg_metadata()
    )
    assert model.backbone.target_node_type == "author"
    assert isinstance(model.readout, HeterogeneousNodeReadout)
    assert model.readout.target_node_type == "author"
    assert model.readout.out_channels == 2
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
    hgt = _instantiate_observed_model(
        _compose("heterogeneous/hgt"), data_spec=heterogeneous_spec
    )
    sage = _instantiate_observed_model(
        _compose("heterogeneous/heterosage"), data_spec=heterogeneous_spec
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
    actual = _instantiate_observed_model(cfg, data_spec=None)

    assert type(actual) is type(expected) is TBModel
    assert type(actual.feature_encoder) is type(expected.feature_encoder)
    assert type(actual.backbone) is type(expected.backbone)
    assert type(actual.readout) is type(expected.readout)
    assert actual.hparams.model_name == expected.hparams.model_name == "gcn"


def test_experimental_compatible_backbone_override_executes_original_config() -> (
    None
):
    """Experimental validation must not canonicalize the instantiated config."""
    cfg = _compose_capability_pair("graph/SyntheticGraph", "graph/gcn")
    custom_target = "torch_geometric.nn.models.GraphSAGE"
    with open_dict(cfg):
        cfg.execution_profile = "experimental"
    with open_dict(cfg.model.backbone):
        cfg.model.backbone._target_ = custom_target
    before = _snapshot(cfg)

    profile_record = validate_execution_profile(cfg)
    model = instantiate_model(
        cfg,
        data_spec=None,
        capability_validation=_observed_validation(cfg),
        profile_record=profile_record,
    )

    assert isinstance(model.backbone.backbone, GraphSAGE)
    assert _snapshot(cfg) == before
    assert profile_record.profile == "experimental"
    assert profile_record.qualified is False
    assert (
        "model.backbone._target_",
        custom_target,
    ) in profile_record.custom_targets


def test_run_qualified_target_override_precedes_every_hydra_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Qualified execution rejects even targets outside capability metadata."""
    import topobench.run as run_module

    cfg = _compose_capability_pair("graph/SyntheticGraph", "graph/gcn")
    with open_dict(cfg.trainer):
        cfg.trainer._target_ = "builtins.dict"
    with open_dict(cfg.paths):
        cfg.paths.output_dir = "/tmp/topobench-execution-profile-test"
    seed = MagicMock()
    factory = MagicMock()
    monkeypatch.setattr(run_module.L, "seed_everything", seed)
    monkeypatch.setattr(run_module.hydra.utils, "instantiate", factory)

    with pytest.raises(ValueError, match=r"trainer\._target_"):
        run_module.run(cfg)

    seed.assert_not_called()
    factory.assert_not_called()


def test_run_experimental_record_is_unqualified_and_reaches_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Runtime outputs and provenance retain the complete profile record."""
    import topobench.run as run_module

    cfg = _compose_capability_pair("graph/SyntheticGraph", "graph/gcn")
    custom_target = "torch_geometric.nn.models.GraphSAGE"
    with open_dict(cfg):
        cfg.execution_profile = "experimental"
        cfg.callbacks = None
        cfg.logger = None
        cfg.train = False
        cfg.test = True
    with open_dict(cfg.evaluation_artifacts):
        cfg.evaluation_artifacts.enabled = False
    with open_dict(cfg.model.backbone):
        cfg.model.backbone._target_ = custom_target
    with open_dict(cfg.paths):
        cfg.paths.output_dir = "/tmp/topobench-execution-profile-test"

    observed = _observed_validation(cfg).observed
    pipeline_output = SimpleNamespace(
        datamodule=LightningDataModule(),
        preprocessing_time=0.0,
        data_spec=None,
        capability_spec=observed,
        prediction_row_adapter=None,
        supervision_counts={},
        provenance_input={"source_graph_id": "synthetic"},
        source_graph_id="synthetic",
    )
    pipeline = MagicMock()
    pipeline.build.return_value = pipeline_output
    trainer = MagicMock(callback_metrics={})
    hydra_factory = MagicMock(side_effect=[pipeline, trainer])
    model_factory = MagicMock(return_value=MagicMock())
    preflight = MagicMock()
    preflight.validate_static.return_value = object()
    preflight.run_probe.return_value = SimpleNamespace(qualified=True)
    preflight_factory = MagicMock(return_value=preflight)
    checkpoint_runner = MagicMock(return_value={})
    warning = MagicMock()
    monkeypatch.setattr(run_module.L, "seed_everything", MagicMock())
    monkeypatch.setattr(run_module.hydra.utils, "instantiate", hydra_factory)
    monkeypatch.setattr(run_module, "instantiate_model", model_factory)
    monkeypatch.setattr(run_module, "PreflightRunner", preflight_factory)
    monkeypatch.setattr(
        run_module,
        "instantiate_callbacks",
        MagicMock(return_value=[]),
    )
    monkeypatch.setattr(
        run_module,
        "instantiate_loggers",
        MagicMock(return_value=[]),
    )
    monkeypatch.setattr(
        run_module,
        "rerun_best_model_checkpoint",
        checkpoint_runner,
    )
    monkeypatch.setattr(run_module.log, "warning", warning)

    metrics, objects = run_module.run(cfg)

    profile_record = objects["execution_profile"]
    assert metrics["qualified"] is False
    assert profile_record.qualified is False
    assert profile_record.custom_targets == (
        ("model.backbone._target_", custom_target),
    )
    warning.assert_called_once_with(
        "Experimental execution profile selected; outputs are unqualified."
    )
    provenance = checkpoint_runner.call_args.kwargs["provenance_input"]
    assert provenance["source_graph_id"] == "synthetic"
    assert provenance["execution_profile"] == {
        "profile": "experimental",
        "qualified": False,
        "targets": tuple(
            {"path": path, "import_path": import_path}
            for path, import_path in profile_record.targets
        ),
        "custom_targets": (
            {
                "path": "model.backbone._target_",
                "import_path": custom_target,
            },
        ),
    }


@pytest.mark.parametrize(
    ("dataset_selector", "model_selector"),
    [
        ("graph/SyntheticGraph", "graph/gcn"),
        (
            "heterogeneous/SyntheticHeterogeneous",
            "heterogeneous/hgt",
        ),
        ("hypergraph/SyntheticHypergraph", "hypergraph/edgnn"),
    ],
)
def test_capability_failure_precedes_every_model_factory(
    dataset_selector: str,
    model_selector: str,
    heterogeneous_spec: HeterogeneousDataSpec,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invalid component targets fail before Hydra imports any component."""
    import topobench.utils.model_instantiation as model_instantiation_module

    cfg = _compose_capability_pair(dataset_selector, model_selector)
    with open_dict(cfg.model.backbone):
        cfg.model.backbone._target_ = "tests.UnqualifiedBackbone"
    factory = MagicMock()
    monkeypatch.setattr(
        model_instantiation_module.hydra.utils,
        "instantiate",
        factory,
    )
    data_spec = (
        heterogeneous_spec
        if dataset_selector.startswith("heterogeneous/")
        else None
    )

    with pytest.raises(ValueError) as error:
        _instantiate_observed_model(cfg, data_spec=data_spec)

    assert "model.backbone._target_" in str(error.value)
    factory.assert_not_called()


def test_model_helper_requires_post_load_capability_before_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Model construction cannot proceed from static qualification alone."""
    import topobench.utils.model_instantiation as model_instantiation_module

    cfg = _compose_capability_pair("graph/SyntheticGraph", "graph/gcn")
    factory = MagicMock()
    monkeypatch.setattr(
        model_instantiation_module.hydra.utils,
        "instantiate",
        factory,
    )

    with pytest.raises(ValueError, match="observed runtime capability"):
        instantiate_model(cfg, data_spec=None)

    factory.assert_not_called()


def test_model_helper_reconciles_prior_observation_before_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A supplied validation result cannot bypass post-load reconciliation."""
    import topobench.utils.model_instantiation as model_instantiation_module

    cfg = _compose_capability_pair("graph/SyntheticGraph", "graph/gcn")
    mismatched_observation = RuntimeDataCapability(
        selector="graph/SyntheticGraph",
        data_domain="graph",
        output_kind="graph",
        feature_widths=(("node", 5),),
        num_classes=2,
        target_node_type=None,
    )
    prior_validation = replace(
        validate_capability_composition(cfg),
        observed=mismatched_observation,
    )
    factory = MagicMock()
    monkeypatch.setattr(
        model_instantiation_module.hydra.utils,
        "instantiate",
        factory,
    )

    with pytest.raises(ValueError) as error:
        instantiate_model(
            cfg,
            data_spec=None,
            capability_validation=prior_validation,
        )
    assert "observed.feature_widths" in str(error.value)
    factory.assert_not_called()


def test_model_helper_reconciles_data_spec_with_observed_capability(
    heterogeneous_spec: HeterogeneousDataSpec,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Injected heterogeneous metadata must match the observed data evidence."""
    import topobench.utils.model_instantiation as model_instantiation_module

    cfg = _compose("heterogeneous/hgt")
    mismatched_spec = replace(
        heterogeneous_spec,
        input_channels=(("author", 9), ("paper", 5), ("venue", 1)),
    )
    factory = MagicMock()
    monkeypatch.setattr(
        model_instantiation_module.hydra.utils,
        "instantiate",
        factory,
    )

    with pytest.raises(ValueError, match=r"data_spec\.input_channels"):
        instantiate_model(
            cfg,
            data_spec=mismatched_spec,
            capability_validation=_observed_validation(cfg),
        )

    factory.assert_not_called()


def test_graph_mlp_nested_loss_target_is_qualified() -> None:
    """Every nested Hydra constructor must belong to the model capability."""
    cfg = _compose_capability_pair(
        "graph/roman_empire",
        "graph/graph_mlp",
    )
    with open_dict(cfg.model.backbone.loss):
        cfg.model.backbone.loss._target_ = "tests.UnqualifiedGraphMLPLoss"

    with pytest.raises(ValueError, match=r"model\.backbone\.loss\._target_"):
        validate_capability_composition(cfg)


def test_run_static_capability_failure_precedes_seed_and_pipeline_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The first executable boundary rejects config before any side effect."""
    import topobench.run as run_module

    cfg = _compose_capability_pair("graph/SyntheticGraph", "graph/gcn")
    with open_dict(cfg.data_pipeline):
        cfg.data_pipeline._target_ = "tests.UnqualifiedPipeline"
    with open_dict(cfg.paths):
        cfg.paths.output_dir = "/tmp/topobench-capability-test"
    seed = MagicMock()
    pipeline_factory = MagicMock()
    monkeypatch.setattr(run_module.L, "seed_everything", seed)
    monkeypatch.setattr(
        run_module.hydra.utils,
        "instantiate",
        pipeline_factory,
    )

    with pytest.raises(ValueError) as error:
        run_module.run(cfg)

    assert "cfg.data_pipeline._target_" in str(error.value)
    seed.assert_not_called()
    pipeline_factory.assert_not_called()


def test_run_observation_failure_precedes_preflight_and_model_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Post-load evidence is reconciled before any model construction."""
    import topobench.run as run_module

    cfg = _compose_capability_pair("graph/SyntheticGraph", "graph/gcn")
    with open_dict(cfg):
        cfg.callbacks = None
        cfg.logger = None
        cfg.train = False
        cfg.test = False
    with open_dict(cfg.paths):
        cfg.paths.output_dir = "/tmp/topobench-capability-test"
    observed = RuntimeDataCapability(
        selector="graph/SyntheticGraph",
        data_domain="graph",
        output_kind="graph",
        feature_widths=(("node", 5),),
        num_classes=2,
        target_node_type=None,
    )
    pipeline_output = SimpleNamespace(
        datamodule=LightningDataModule(),
        preprocessing_time=0.0,
        data_spec=None,
        capability_spec=observed,
    )
    pipeline = MagicMock()
    pipeline.build.return_value = pipeline_output
    pipeline_factory = MagicMock(return_value=pipeline)
    model_factory = MagicMock()
    preflight_runner = MagicMock()
    monkeypatch.setattr(run_module.L, "seed_everything", MagicMock())
    monkeypatch.setattr(
        run_module.hydra.utils,
        "instantiate",
        pipeline_factory,
    )
    monkeypatch.setattr(run_module, "instantiate_model", model_factory)
    monkeypatch.setattr(run_module, "PreflightRunner", preflight_runner)

    with pytest.raises(ValueError) as error:
        run_module.run(cfg)

    assert "observed.feature_widths" in str(error.value)
    pipeline.build.assert_called_once_with(cfg)
    preflight_runner.assert_not_called()
    model_factory.assert_not_called()


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (
            "feature_encoder.input_channels",
            {"author": 8, "paper": 5, "venue": 1},
        ),
        (
            "backbone.metadata",
            [["author", "paper", "venue"], []],
        ),
        ("backbone_wrapper.target_node_type", "author"),
        ("readout.target_node_type", "author"),
        ("readout.out_channels", 2),
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
        _instantiate_observed_model(cfg, data_spec=heterogeneous_spec)

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
            lambda: _compose_capability_pair(
                "graph/SyntheticGraph",
                "graph/gcn",
            ),
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

    model_factory = (
        instantiate_model
        if data_spec == "invalid"
        else _instantiate_observed_model
    )
    with pytest.raises(error):
        model_factory(cfg, data_spec=supplied)  # type: ignore[arg-type]

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
        _instantiate_observed_model(cfg, data_spec=heterogeneous_spec)

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
        _instantiate_observed_model(cfg, data_spec=heterogeneous_spec)

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
        _instantiate_observed_model(cfg, data_spec=heterogeneous_spec)

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

    model = _instantiate_observed_model(cfg, data_spec=heterogeneous_spec)

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
        _instantiate_observed_model(cfg, data_spec=heterogeneous_spec)

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
        _instantiate_observed_model(cfg, data_spec=heterogeneous_spec)

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
            _instantiate_observed_model(cfg, data_spec=heterogeneous_spec)
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
            _instantiate_observed_model(cfg, data_spec=heterogeneous_spec)

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
        _instantiate_observed_model(cfg, data_spec=heterogeneous_spec)

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

    model = _instantiate_observed_model(cfg, data_spec=heterogeneous_spec)

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
        _instantiate_observed_model(cfg, data_spec=heterogeneous_spec)

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
    reverse_relation = f"reverse::{literal}"
    spec = HeterogeneousDataSpec(
        node_types=("author", "paper", "venue"),
        edge_types=(
            ("author", literal, "paper"),
            ("paper", reverse_relation, "venue"),
        ),
        target_node_type="author",
        num_classes=2,
        input_channels=(("author", 8), ("paper", 5), ("venue", 1)),
    )
    cfg = _compose(model_name)
    model = _instantiate_observed_model(cfg, data_spec=spec)
    data = HeteroData()
    data["author"].x = torch.randn(4, 8)
    data["author"].y = torch.tensor([0, 1, 0, 1])
    data["paper"].x = torch.randn(3, 5)
    data["venue"].x = torch.randn(2, 1)
    data["author", literal, "paper"].edge_index = torch.tensor(
        [[0, 1, 2, 3], [0, 1, 2, 0]],
    )
    data["paper", reverse_relation, "venue"].edge_index = torch.tensor(
        [[0, 1, 2], [0, 1, 0]],
    )

    result = model(data)

    assert model.feature_encoder.input_channels == {
        "author": 8,
        "paper": 5,
        "venue": 1,
    }
    assert model.backbone.target_node_type == "author"
    assert model.backbone.backbone.metadata == spec.pyg_metadata()
    assert model.readout.target_node_type == "author"
    assert model.supervision_adapter.target_node_type == "author"
    assert result["logits"].shape == (4, 2)


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
        node_types=("author", "paper", "venue"),
        edge_types=(("author", literal, "paper"),),
        target_node_type="author",
        num_classes=2,
        input_channels=(("author", 8), ("paper", 5), ("venue", 1)),
    )
    cfg = _compose("heterogeneous/hgt")
    try:
        model = _instantiate_observed_model(cfg, data_spec=spec)
    finally:
        OmegaConf.clear_resolver(resolver_name)

    assert calls == []
    assert model.readout.target_node_type == "author"
    assert model.backbone.backbone.metadata == spec.pyg_metadata()


def test_copy_isolation_preserves_absolute_interpolation_context(
    heterogeneous_spec: HeterogeneousDataSpec,
) -> None:
    """Absolute model interpolations resolve on the copied root only."""
    cfg = _compose("heterogeneous/hgt")
    before = deepcopy(_snapshot(cfg))
    model = _instantiate_observed_model(cfg, data_spec=heterogeneous_spec)

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
from dataclasses import replace
from topobench.data import HeterogeneousDataSpec, RuntimeDataCapability
from topobench.nn.capabilities import validate_capability_composition
from topobench.utils.config_resolvers import register_all_resolvers
from topobench.utils.model_instantiation import instantiate_model
register_all_resolvers()
with hydra.initialize(version_base="1.3", config_path="configs"):
    cfg = hydra.compose(
        config_name="run.yaml",
        overrides=[
            "model={model_name}",
            "dataset=heterogeneous/SyntheticHeterogeneous",
            "data_pipeline=heterogeneous_node",
        ],
    )
spec = HeterogeneousDataSpec(
    node_types=("author", "paper", "venue"),
    edge_types=(
        ("author", "writes", "paper"),
        ("paper", "published_in", "venue"),
    ),
    target_node_type="author",
    num_classes=2,
    input_channels=(("author", 8), ("paper", 5), ("venue", 1)),
)
static = validate_capability_composition(cfg)
observed = RuntimeDataCapability(
    selector="heterogeneous/SyntheticHeterogeneous",
    data_domain="heterogeneous",
    output_kind="heterogeneous",
    feature_widths=(("author", 8), ("paper", 5), ("venue", 1)),
    num_classes=2,
    target_node_type="author",
)
model = instantiate_model(
    cfg,
    data_spec=spec,
    capability_validation=replace(static, observed=observed),
)
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


def test_run_passes_observed_capability_to_central_model_helper(
    monkeypatch: pytest.MonkeyPatch,
    heterogeneous_spec: HeterogeneousDataSpec,
) -> None:
    """The run boundary forwards one reconciled post-load validation result."""
    import topobench.run as run_module

    cfg = _compose_capability_pair(
        "heterogeneous/SyntheticHeterogeneous",
        "heterogeneous/hgt",
    )
    with open_dict(cfg):
        cfg.callbacks = None
        cfg.logger = None
        cfg.train = False
        cfg.test = False
        cfg.execution_profile = "experimental"
    with open_dict(cfg.evaluation_artifacts):
        cfg.evaluation_artifacts.enabled = False
    with open_dict(cfg.paths):
        cfg.paths.output_dir = "/tmp/topobench-capability-test"
    profile_record = validate_execution_profile(cfg)

    observed_capability = RuntimeDataCapability(
        selector="heterogeneous/SyntheticHeterogeneous",
        data_domain="heterogeneous",
        output_kind="heterogeneous",
        feature_widths=(("author", 8), ("paper", 5), ("venue", 1)),
        num_classes=2,
        target_node_type="author",
    )
    datamodule = LightningDataModule()
    pipeline_output = SimpleNamespace(
        datamodule=datamodule,
        preprocessing_time=0.0,
        data_spec=heterogeneous_spec,
        capability_spec=observed_capability,
    )
    pipeline = MagicMock()
    pipeline.build.return_value = pipeline_output
    trainer = MagicMock()
    trainer.callback_metrics = {}
    model = MagicMock()
    instantiate_model_mock = MagicMock(return_value=model)
    static_validation = MagicMock(name="static_validation")
    observed_validation = MagicMock(name="observed_validation")
    capability_validator = MagicMock(
        side_effect=[static_validation, observed_validation]
    )

    def instantiate_side_effect(config, **kwargs):
        del kwargs
        return pipeline if config is cfg.data_pipeline else trainer

    monkeypatch.setattr(
        run_module.hydra.utils,
        "instantiate",
        instantiate_side_effect,
    )
    monkeypatch.setattr(
        run_module,
        "validate_profile_capability",
        capability_validator,
    )
    monkeypatch.setattr(
        run_module,
        "instantiate_model",
        instantiate_model_mock,
        raising=False,
    )
    preflight_result = SimpleNamespace(passed=True, qualified=True)
    preflight_runner = MagicMock()
    preflight_runner.return_value.validate_static.return_value = (
        preflight_result
    )
    preflight_runner.return_value.run_probe.return_value = preflight_result
    monkeypatch.setattr(run_module, "PreflightRunner", preflight_runner)
    monkeypatch.setattr(
        run_module,
        "instantiate_callbacks",
        lambda _, **kwargs: [],
    )
    monkeypatch.setattr(run_module, "instantiate_loggers", lambda _: [])

    _, objects = run_module.run(cfg)

    assert capability_validator.call_args_list == [
        call(cfg, profile_record=profile_record),
        call(
            cfg,
            profile_record=profile_record,
            observed=observed_capability,
        ),
    ]
    instantiate_model_mock.assert_called_once_with(
        cfg,
        data_spec=heterogeneous_spec,
        capability_validation=observed_validation,
        profile_record=profile_record,
    )
    assert objects["model"] is model
