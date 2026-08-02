"""Composition and construction gates for qualified graph models."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import hydra
import pytest
from hydra.core.hydra_config import HydraConfig
from hydra.errors import MissingConfigException
from omegaconf import DictConfig, open_dict

from topobench.data.capabilities import GRAPH_DATASET_MANIFEST
from topobench.nn.capabilities import (
    GRAPH_MODEL_CAPABILITIES,
    compatible_graph_models,
    validate_graph_composition,
)
from topobench.utils.config_resolvers import register_all_resolvers
from topobench.utils.model_instantiation import instantiate_model

PROJECT_ROOT = Path(__file__).parents[2]
REMOVED_DATASETS = (
    "US-county-demos",
    "graphuniverse_inductive",
    "ogbg-molpcba",
    "manual_dataset",
)
VALID_PAIRS = tuple(
    (dataset.selector, model.selector)
    for dataset in GRAPH_DATASET_MANIFEST.values()
    for model in compatible_graph_models(dataset)
)


def _compose(dataset: str, model: str) -> DictConfig:
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    register_all_resolvers()
    with hydra.initialize(
        version_base="1.3",
        config_path="../../configs",
        job_name="qualified_graph_config",
    ):
        cfg = hydra.compose(
            config_name="run.yaml",
            overrides=[
                f"dataset=graph/{dataset}",
                f"model=graph/{model}",
                "trainer.accelerator=cpu",
                "trainer.devices=1",
            ],
            return_hydra_config=True,
        )
        HydraConfig.instance().set_config(cfg)
        return cfg


@pytest.mark.parametrize(
    ("dataset_selector", "model_selector"),
    VALID_PAIRS,
    ids=lambda value: value,
)
def test_every_declared_graph_pair_resolves_and_instantiates_without_data_spec(
    dataset_selector: str,
    model_selector: str,
) -> None:
    cfg = _compose(dataset_selector, model_selector)

    assert cfg.model.model_domain == "graph"
    assert (
        cfg.model.feature_encoder._target_
        == "topobench.nn.encoders.GraphNodeFeatureEncoder"
    )
    assert cfg.dataset.parameters.task in {"classification", "regression"}
    assert cfg.dataset.split_params.learning_setting in {
        "inductive",
        "transductive",
    }
    assert cfg.model.backbone_wrapper.edge_attr_mode in {
        "consume",
        "ignore",
        "reject",
    }
    assert cfg.model.backbone_wrapper.edge_weight_mode in {
        "consume",
        "ignore",
        "reject",
    }

    model = instantiate_model(cfg, data_spec=None)
    assert model.backbone.edge_modes == {
        "edge_attr": GRAPH_MODEL_CAPABILITIES[model_selector].edge_attr_mode,
        "edge_weight": GRAPH_MODEL_CAPABILITIES[
            model_selector
        ].edge_weight_mode,
    }


def test_packaged_nsd_uses_an_exact_stalk_width_composition() -> None:
    cfg = _compose("SyntheticGraph", "nsd")

    assert cfg.model.feature_encoder.out_channels == 64
    assert cfg.model.backbone.hidden_dim == 64
    assert cfg.model.backbone.d == 4

    model = instantiate_model(cfg, data_spec=None)
    encoder = model.backbone.backbone
    assert encoder.sheaf_config["hidden_channels"] == 16
    assert encoder.sheaf_model.hidden_dim == 64


def test_model_capability_matrix_and_entries_are_immutable() -> None:
    capability = GRAPH_MODEL_CAPABILITIES["gcn"]

    with pytest.raises(TypeError):
        GRAPH_MODEL_CAPABILITIES["new"] = capability  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        capability.edge_attr_mode = "consume"  # type: ignore[misc]


@pytest.mark.parametrize("selector", REMOVED_DATASETS)
def test_removed_graph_dataset_selectors_are_not_composable(
    selector: str,
) -> None:
    assert not (
        PROJECT_ROOT / "configs" / "dataset" / "graph" / f"{selector}.yaml"
    ).exists()
    with pytest.raises(MissingConfigException):
        _compose(selector, "gcn")


def test_unsupported_task_pair_fails_with_dataset_path() -> None:
    cfg = _compose("SyntheticGraphRegression", "gcn")
    with open_dict(cfg):
        cfg.model.model_name = "graph_mlp"

    with pytest.raises(ValueError, match=r"dataset\.parameters\.task"):
        validate_graph_composition(cfg.dataset, cfg.model)


def test_unsupported_learning_setting_fails_with_dataset_path() -> None:
    cfg = _compose("SyntheticGraph", "gcn")
    with open_dict(cfg.dataset.split_params):
        cfg.dataset.split_params.learning_setting = "transductive"

    with pytest.raises(
        ValueError,
        match=r"dataset\.split_params\.learning_setting",
    ):
        validate_graph_composition(cfg.dataset, cfg.model)


def test_unsupported_feature_policy_fails_with_dataset_path() -> None:
    cfg = _compose("SyntheticNodeGraph", "graph_mlp")
    with open_dict(cfg.dataset.parameters):
        cfg.dataset.parameters.feature_policy = "categorical_one_hot"

    with pytest.raises(
        ValueError, match=r"dataset\.parameters\.feature_policy"
    ):
        validate_graph_composition(cfg.dataset, cfg.model)


def test_rejected_edge_field_pair_fails_before_model_construction() -> None:
    cfg = _compose("MUTAG", "gin")

    with pytest.raises(ValueError, match=r"dataset.*edge_attr.*model"):
        instantiate_model(cfg, data_spec=None)


def test_missing_explicit_edge_mode_fails_with_model_path() -> None:
    cfg = _compose("SyntheticGraph", "gcn")
    with open_dict(cfg.model.backbone_wrapper):
        del cfg.model.backbone_wrapper["edge_attr_mode"]

    with pytest.raises(
        ValueError,
        match=r"model\.backbone_wrapper\.edge_attr_mode",
    ):
        instantiate_model(cfg, data_spec=None)


def test_cross_domain_default_transform_is_rejected() -> None:
    cfg = _compose("SyntheticGraph", "gcn")
    with open_dict(cfg.model):
        cfg.model.model_domain = "hypergraph"

    with pytest.raises(
        ValueError, match="Cross-domain lifting is unsupported"
    ):
        validate_graph_composition(cfg.dataset, cfg.model)
