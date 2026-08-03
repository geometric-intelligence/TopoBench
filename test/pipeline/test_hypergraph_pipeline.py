"""Network-free composition gates for native hypergraph models."""

from __future__ import annotations

from collections.abc import Callable

import hydra
import pytest
import torch
from omegaconf import DictConfig, open_dict
from torch import Tensor, nn

import topobench.data.pipelines.hypergraph as hypergraph_pipeline_module
from test._utils.simplified_pipeline import run
from topobench.data.datasets.synthetic_hypergraph_dataset import (
    make_synthetic_hypergraph_data,
)
from topobench.data.pipelines.hypergraph import HypergraphNodeDataPipeline
from topobench.nn.capabilities import validate_capability_composition
from topobench.nn.encoders.graph_node_encoder import GraphNodeFeatureEncoder
from topobench.nn.wrappers.hypergraph import HypergraphWrapper
from topobench.utils.config_resolvers import register_all_resolvers
from topobench.utils.model_instantiation import instantiate_model


def _compose(model_selector: str) -> DictConfig:
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    register_all_resolvers()
    with hydra.initialize(
        version_base="1.3",
        config_path="../../configs",
        job_name="hypergraph_model_composition",
    ):
        return hydra.compose(
            config_name="run.yaml",
            overrides=[
                "dataset=hypergraph/SyntheticHypergraph",
                f"model=hypergraph/{model_selector}",
                "data_pipeline=hypergraph_node",
                "trainer.max_epochs=1",
                "trainer.min_epochs=1",
                "trainer.check_val_every_n_epoch=1",
                "trainer.accelerator=cpu",
                "trainer.devices=1",
                "callbacks=model_checkpoint",
                "paths=test",
            ],
        )


class _PreprocessorFixture(list):
    """Minimal singleton preprocessor result for ownership tests."""

    preprocessing_time = 0.0


class _IdentityHypergraphBackbone(nn.Module):
    """Return node features while accepting the native incidence argument."""

    def forward(self, x: Tensor, hyperedge_index: Tensor) -> Tensor:
        del hyperedge_index
        return x


def _build_from_source(
    monkeypatch: pytest.MonkeyPatch,
    source: object,
) -> tuple[DictConfig, object]:
    cfg = _compose("hypergraph_conv")
    monkeypatch.setattr(
        HypergraphNodeDataPipeline,
        "preprocess",
        staticmethod(lambda _cfg: _PreprocessorFixture([source])),
    )
    return cfg, HypergraphNodeDataPipeline().build(cfg)


def test_hypergraph_pipeline_emits_observed_capability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Capability metadata comes from validated feature and label tensors."""
    source = make_synthetic_hypergraph_data(seed=19)

    _, output = _build_from_source(monkeypatch, source)
    capability = output.capability_spec

    assert capability is not None
    assert capability.selector == "hypergraph/SyntheticHypergraph"
    assert capability.data_domain == "hypergraph"
    assert capability.output_kind == "hypergraph"
    assert capability.feature_widths == (("node", 4),)
    assert capability.num_classes == 2
    assert capability.target_node_type is None


def test_pipeline_aliases_immutable_hypergraph_tensors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pipeline masks and batch metadata never copy feature or incidence stores."""
    source = make_synthetic_hypergraph_data(seed=23)
    source_fields = set(source.keys())
    source_masks = {
        name: source[name] for name in ("train_mask", "val_mask", "test_mask")
    }

    _, output = _build_from_source(monkeypatch, source)
    runtime = output.datamodule.dataset_train[0]

    assert runtime is not source
    for field in ("x", "y", "hyperedge_index"):
        assert runtime[field] is source[field]
    assert set(source.keys()) == source_fields | {"global_nid"}
    assert torch.equal(
        source.global_nid,
        torch.arange(source.num_nodes, dtype=torch.long),
    )
    assert "batch" not in source
    for name, mask in source_masks.items():
        assert source[name] is mask


def test_singleton_pipeline_keeps_sparse_features_until_encoder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = make_synthetic_hypergraph_data(seed=27)
    sparse_x = source.x.to_sparse_coo().coalesce()
    source.x = sparse_x

    _, output = _build_from_source(monkeypatch, source)
    runtime = output.datamodule.dataset_train[0]
    batch = next(iter(output.datamodule.train_dataloader()))

    assert runtime.x is sparse_x
    assert runtime.x.layout == torch.sparse_coo
    assert batch.x is sparse_x
    assert batch.x.layout == torch.sparse_coo
    assert batch.x.is_coalesced()


def test_exhaustive_hypergraph_validation_runs_once_at_pipeline_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repeated wrapper forwards reuse the pipeline validation marker."""
    source = make_synthetic_hypergraph_data(seed=29)
    validation_calls = 0
    validate_source = hypergraph_pipeline_module.validate_hypergraph_source

    def counted_validation(*args: object, **kwargs: object) -> object:
        nonlocal validation_calls
        validation_calls += 1
        return validate_source(*args, **kwargs)

    monkeypatch.setattr(
        hypergraph_pipeline_module,
        "validate_hypergraph_source",
        counted_validation,
    )
    _, output = _build_from_source(monkeypatch, source)
    batch = next(iter(output.datamodule.train_dataloader()))
    wrapper = HypergraphWrapper(_IdentityHypergraphBackbone())

    wrapper(batch)
    wrapper(batch)

    assert validation_calls == 1


@pytest.mark.parametrize(
    "mutate",
    (
        lambda data: data.x.__setitem__((0, 0), float("nan")),
        lambda data: data.y.__setitem__(0, -1),
        lambda data: data.hyperedge_index.__setitem__(
            (0, 0),
            data.num_nodes,
        ),
        lambda data: data.train_mask.__setitem__(
            0,
            ~data.train_mask[0],
        ),
    ),
)
def test_pipeline_revalidates_stale_source_marker_before_reuse(
    monkeypatch: pytest.MonkeyPatch,
    mutate: Callable[[object], object],
) -> None:
    """In-place tensor writes cannot reuse stale boundary evidence."""
    source = make_synthetic_hypergraph_data(seed=37)
    _build_from_source(monkeypatch, source)
    mutate(source)

    with pytest.raises((TypeError, ValueError)):
        _build_from_source(monkeypatch, source)


def test_feature_encoder_rebinds_validation_to_projected_x(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Trusted shallow feature encoding refreshes x evidence for the wrapper."""
    source = make_synthetic_hypergraph_data(seed=41)
    _, output = _build_from_source(monkeypatch, source)
    batch = next(iter(output.datamodule.train_dataloader()))
    torch.manual_seed(41)
    encoder = GraphNodeFeatureEncoder(
        in_channels=batch.x.size(1),
        out_channels=7,
        dropout=0.0,
    ).eval()

    encoded = encoder(batch)
    result = HypergraphWrapper(_IdentityHypergraphBackbone())(encoded)

    assert encoded is not batch
    assert encoded.x.shape == (batch.num_nodes, 7)
    assert encoded.y is batch.y
    assert encoded.hyperedge_index is batch.hyperedge_index
    assert result["x"] is encoded.x
    assert result["labels"] is encoded.y


@pytest.mark.parametrize("model_selector", ("edgnn", "hypergraph_conv"))
def test_synthetic_hypergraph_composes_to_finite_node_logits(
    model_selector: str,
) -> None:
    """Both selectors compose over native fields with one logit row per label."""
    cfg = _compose(model_selector)
    pipeline_output = hydra.utils.instantiate(cfg.data_pipeline).build(cfg)
    batch = next(iter(pipeline_output.datamodule.train_dataloader()))
    capability_validation = validate_capability_composition(
        cfg,
        observed=pipeline_output.capability_spec,
    )
    model = instantiate_model(
        cfg,
        data_spec=None,
        capability_validation=capability_validation,
    )
    model.eval()

    model_out = model.forward(batch)

    assert set(model_out) == {"x", "labels", "batch", "logits"}
    assert model_out["labels"] is batch.y
    assert model_out["batch"] is batch.batch
    assert model_out["logits"].shape == (
        batch.y.size(0),
        int(cfg.dataset.parameters.num_classes),
    )
    assert torch.isfinite(model_out["logits"]).all()


def test_synthetic_hypergraph_meandeg_returns_finite_full_node_logits() -> (
    None
):
    """The configured MeanDeg path preserves every nonisolated fixture node."""
    cfg = _compose("edgnn")
    with open_dict(cfg.model.backbone):
        cfg.model.backbone.edconv_type = "MeanDeg"
        cfg.model.backbone.input_dropout = 0.0
        cfg.model.backbone.dropout = 0.0
    pipeline_output = hydra.utils.instantiate(cfg.data_pipeline).build(cfg)
    batch = next(iter(pipeline_output.datamodule.train_dataloader()))
    capability_validation = validate_capability_composition(
        cfg,
        observed=pipeline_output.capability_spec,
    )
    model = instantiate_model(
        cfg,
        data_spec=None,
        capability_validation=capability_validation,
    )
    model.eval()

    model_out = model.forward(batch)

    assert model_out["logits"].shape[0] == batch.x.size(0)
    assert torch.isfinite(model_out["x"]).all()
    assert torch.isfinite(model_out["logits"]).all()


@pytest.mark.parametrize("model_selector", ("edgnn", "hypergraph_conv"))
def test_synthetic_hypergraph_runs_one_epoch_and_final_test(
    model_selector: str,
) -> None:
    """Both native models complete one bounded epoch and final test."""
    result = run(_compose(model_selector))

    assert result["epochs_completed"] == 1
    assert result["observed_train_batch_size"] == 1
    assert {"train/loss", "val/loss"} <= result["fit_metrics"].keys()
    assert all(
        torch.isfinite(torch.tensor(value))
        for value in result["fit_metrics"].values()
    )
    assert result["test_results"]
    assert all(
        torch.isfinite(torch.tensor(value))
        for result_row in result["test_results"]
        for value in result_row.values()
    )
