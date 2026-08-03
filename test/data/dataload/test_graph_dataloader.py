"""Preflight contracts for native homogeneous graph data modules."""

from __future__ import annotations
from typing import Any

import pytest
import torch
from torch.utils.data import Subset
from torch_geometric.data import Data

from topobench.data.datasets.synthetic_hypergraph_dataset import (
    make_synthetic_hypergraph_data,
)
from topobench.dataloader.graph import GraphDataModule, mark_hypergraph_validated
from topobench.preflight import PreflightError

from test.preflight.test_data_probe import (
    ProbeModel,
    make_observations,
    qualified_runner,
    run_probe,
)


def _inductive_module() -> GraphDataModule:
    source = [
        Data(
            x=torch.tensor([[float(index), 1.0]]),
            y=torch.tensor([index % 2]),
            sample_id=torch.tensor([index]),
        )
        for index in range(8)
    ]
    return GraphDataModule(
        Subset(source, [0, 1, 2, 3]),
        Subset(source, [4, 5]),
        Subset(source, [6, 7]),
        learning_setting="inductive",
        batch_size=2,
        num_workers=0,
    )


def test_preflight_preserves_canonical_first_inductive_training_batch() -> None:
    control = _inductive_module()
    probed = _inductive_module()

    torch.manual_seed(90210)
    rng_before = torch.random.get_rng_state().clone()

    result = run_probe(probed, make_observations())

    assert result.passed
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    torch.manual_seed(90210)
    expected = next(iter(control.train_dataloader())).sample_id.clone()
    torch.manual_seed(90210)
    actual = next(iter(probed.train_dataloader())).sample_id
    assert torch.equal(actual, expected)


def test_preflight_obtains_each_hypergraph_phase_without_mutating_source() -> None:
    source = mark_hypergraph_validated(make_synthetic_hypergraph_data(seed=13))
    module = GraphDataModule(
        [source],
        learning_setting="transductive",
        batch_size=1,
        num_workers=0,
    )
    tensor_state = {
        name: value.clone()
        for name, value in source.to_dict().items()
        if isinstance(value, torch.Tensor)
    }
    observations = make_observations()

    result = run_probe(module, observations)

    assert result.passed
    assert [
        event.removeprefix("forward:")
        for event in observations["events"]
        if event.startswith("forward:")
    ] == ["train", "val", "test"]
    assert all(
        torch.equal(source[name], expected)
        for name, expected in tensor_state.items()
    )


def test_transductive_probe_mutations_are_isolated_from_canonical_hypergraph() -> None:
    source = mark_hypergraph_validated(make_synthetic_hypergraph_data(seed=29))
    module = GraphDataModule(
        [source],
        learning_setting="transductive",
        batch_size=1,
        num_workers=0,
    )
    canonical = module.dataset_train[0]
    canonical_keys = set(canonical.keys())
    tensor_state = {
        name: value.clone()
        for name, value in canonical.to_dict().items()
        if isinstance(value, torch.Tensor)
    }
    observations = make_observations()

    class MutatingProbeModel(ProbeModel):
        def model_step(self, batch: Data) -> dict[str, Any]:
            batch.model_state = "throwaway-preflight"
            batch.x.add_(1000)
            return dict(super().model_step(batch))

    runner, static_result, _ = qualified_runner(module)
    result = runner.run_probe(
        model_factory=lambda: MutatingProbeModel(observations),
        static_result=static_result,
    )

    assert result.passed
    assert module.dataset_train[0] is canonical
    assert set(canonical.keys()) == canonical_keys
    assert "model_state" not in canonical
    assert all(
        torch.equal(canonical[name], expected)
        for name, expected in tensor_state.items()
    )


def test_preflight_restores_graph_datamodule_state_on_execution_failure() -> None:
    module = _inductive_module()
    before = module.state_dict()

    with pytest.raises(PreflightError, match="intentional val"):
        run_probe(
            module,
            make_observations(),
            fail_phase="val",
        )

    assert module.state_dict() == before
