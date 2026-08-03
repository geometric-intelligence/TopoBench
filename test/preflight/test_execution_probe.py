"""Full throwaway-runtime execution contracts for automatic preflight."""

from __future__ import annotations

import gc
import weakref
from typing import Any

import pytest
import torch
from torch import nn
from torch_geometric.data import Data

from topobench.preflight import PreflightError

from .test_data_probe import (
    ProbeModel,
    StatefulPhaseDataModule,
    make_observations,
    qualified_runner,
)


def test_probe_executes_complete_training_validation_and_test_flow() -> None:
    datamodule = StatefulPhaseDataModule()
    observations = make_observations()
    runner, static_result, _ = qualified_runner(datamodule)

    result = runner.run_probe(
        model_factory=lambda: ProbeModel(observations),
        static_result=static_result,
    )

    events = observations["events"]
    assert events.count("transfer") == 3
    for phase in ("train", "val", "test"):
        assert f"evaluator.begin:{phase}" in events
        assert f"forward:{phase}" in events
        assert f"supervision:{phase}" in events
        assert f"loss:{phase}" in events
        assert f"evaluator.update:{phase}" in events
        assert f"evaluator.snapshot:{phase}" in events
        assert f"evaluator.abort:{phase}" in events
    assert events.count("optimizer.construct") == 1
    assert events.count("optimizer.step") == 1
    assert events.count("scheduler.step") == 1
    assert observations["gradient_nonempty"] is True
    assert observations["gradient_finite"] is True
    assert observations["parameter_changed"] is True
    assert result.passed


def test_probe_routes_raw_step_through_model_optimizer_hook_once() -> None:
    datamodule = StatefulPhaseDataModule()
    observations = make_observations()

    class HookRecordingModel(ProbeModel):
        def __init__(self) -> None:
            super().__init__(observations)
            self.optimizer_success_token = 0

        def optimizer_step(
            self,
            epoch: int,
            batch_idx: int,
            optimizer: Any,
            optimizer_closure: Any = None,
        ) -> None:
            observations["events"].append("model.optimizer_step")
            observations["optimizer_hook_args"] = (epoch, batch_idx)
            raw_steps_before = observations["events"].count("optimizer.step")
            optimizer.step(closure=optimizer_closure)
            raw_steps_after = observations["events"].count("optimizer.step")
            if raw_steps_after == raw_steps_before + 1:
                self.optimizer_success_token += 1
            observations["optimizer_success_token"] = (
                self.optimizer_success_token
            )

    runner, static_result, _ = qualified_runner(datamodule)
    result = runner.run_probe(
        model_factory=HookRecordingModel,
        static_result=static_result,
    )

    assert result.passed
    assert observations["events"].count("model.optimizer_step") == 1
    assert observations["events"].count("optimizer.step") == 1
    assert observations["optimizer_hook_args"] == (0, 0)
    assert observations["optimizer_success_token"] == 1


def test_probe_rejects_nonfinite_gradients_even_when_forward_loss_is_finite() -> (
    None
):
    datamodule = StatefulPhaseDataModule()
    observations = make_observations()
    runner, static_result, _ = qualified_runner(datamodule)

    with pytest.raises(PreflightError, match="gradient|finite"):
        runner.run_probe(
            model_factory=lambda: ProbeModel(
                observations,
                nan_gradient=True,
            ),
            static_result=static_result,
        )


@pytest.mark.parametrize("compile_enabled", [False, True])
def test_probe_uses_configured_compile_path_without_production_compile_cost(
    monkeypatch: pytest.MonkeyPatch,
    compile_enabled: bool,
) -> None:
    datamodule = StatefulPhaseDataModule()
    observations = make_observations()
    compile_calls: list[nn.Module] = []

    def compile_boundary(
        module: nn.Module, *args: Any, **kwargs: Any
    ) -> nn.Module:
        del args, kwargs
        compile_calls.append(module)
        return module

    monkeypatch.setattr(torch, "compile", compile_boundary)

    def configure(cfg: Any) -> None:
        cfg.model.compile = compile_enabled

    runner, static_result, _ = qualified_runner(
        datamodule,
        configure=configure,
    )
    result = runner.run_probe(
        model_factory=lambda: ProbeModel(
            observations,
            compile_enabled=compile_enabled,
        ),
        static_result=static_result,
    )

    assert result.passed
    assert len(compile_calls) == int(compile_enabled)


def test_test_only_probe_uses_test_setup_without_optimizer_or_compile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    datamodule = StatefulPhaseDataModule()
    observations = make_observations()
    compile_calls: list[nn.Module] = []

    def compile_boundary(
        module: nn.Module, *args: Any, **kwargs: Any
    ) -> nn.Module:
        del args, kwargs
        compile_calls.append(module)
        return module

    monkeypatch.setattr(torch, "compile", compile_boundary)

    def configure(cfg: Any) -> None:
        cfg.model.compile = True

    runner, static_result, _ = qualified_runner(
        datamodule,
        train=False,
        test=True,
        configure=configure,
    )
    result = runner.run_probe(
        model_factory=lambda: ProbeModel(
            observations,
            compile_enabled=True,
        ),
        static_result=static_result,
    )

    assert result.passed
    assert "optimizer.construct" not in observations["events"]
    assert "model.setup:fit" not in observations["events"]
    assert observations["events"].count("model.setup:test") == 1
    assert compile_calls == []
    assert [
        event.removeprefix("forward:")
        for event in observations["events"]
        if event.startswith("forward:")
    ] == ["test"]
    assert observations["events"].count("evaluator.begin:test") == 1
    assert observations["events"].count("evaluator.update:test") == 1
    assert observations["events"].count("evaluator.snapshot:test") == 1
    assert observations["events"].count("evaluator.abort:test") == 1
    check_ids = {check.check_id for check in result.checks}
    assert "execution.prediction_payload" in check_ids
    assert "execution.optimizer" not in check_ids
    assert "execution.scheduler" not in check_ids
    assert "execution.gradient" not in check_ids


def test_compile_graph_execution_error_fails_preflight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    datamodule = StatefulPhaseDataModule()
    observations = make_observations()

    class _BrokenCompiledModule(nn.Module):
        def forward(self, count: int) -> torch.Tensor:
            del count
            raise RuntimeError("qualified compile graph execution failed")

    monkeypatch.setattr(
        torch,
        "compile",
        lambda module, *args, **kwargs: _BrokenCompiledModule(),
    )

    def configure(cfg: Any) -> None:
        cfg.model.compile = True

    runner, static_result, _ = qualified_runner(
        datamodule,
        configure=configure,
    )

    with pytest.raises(
        PreflightError,
        match="compile graph execution failed",
    ):
        runner.run_probe(
            model_factory=lambda: ProbeModel(
                observations,
                compile_enabled=True,
            ),
            static_result=static_result,
        )


def test_probe_reports_nonpublishing_payload_and_structured_check_validation() -> (
    None
):
    datamodule = StatefulPhaseDataModule()
    observations = make_observations()
    runner, static_result, _ = qualified_runner(datamodule)

    result = runner.run_probe(
        model_factory=lambda: ProbeModel(observations),
        static_result=static_result,
    )

    check_ids = {check.check_id for check in result.checks}
    assert {
        "execution.gradient",
        "execution.optimizer",
        "execution.scheduler",
        "execution.structured_checks",
        "execution.reproducibility_payload",
        "execution.prediction_payload",
    } <= check_ids
    record = result.as_record()
    assert record["passed"] is True
    assert all(isinstance(check, dict) for check in record["checks"])


def test_successful_probe_discards_throwaway_model_optimizer_and_scheduler() -> (
    None
):
    datamodule = StatefulPhaseDataModule()
    observations = make_observations()
    references: dict[str, weakref.ReferenceType[Any]] = {}

    def factory() -> ProbeModel:
        model = ProbeModel(observations)
        references["model"] = weakref.ref(model)
        references["evaluator"] = weakref.ref(model.evaluator)
        return model

    runner, static_result, _ = qualified_runner(datamodule)
    result = runner.run_probe(
        model_factory=factory,
        static_result=static_result,
    )
    assert "optimizer" in observations
    assert "scheduler" in observations
    references["optimizer"] = weakref.ref(observations.pop("optimizer"))
    references["scheduler"] = weakref.ref(observations.pop("scheduler"))

    assert result.passed
    del result
    gc.collect()
    assert all(reference() is None for reference in references.values())


def test_probe_uses_native_batch_and_does_not_replace_production_data() -> (
    None
):
    datamodule = StatefulPhaseDataModule()
    canonical = datamodule.batch
    canonical_x = canonical.x.clone()
    observations = make_observations()
    runner, static_result, _ = qualified_runner(datamodule)

    result = runner.run_probe(
        model_factory=lambda: ProbeModel(observations),
        static_result=static_result,
    )

    assert result.passed
    assert isinstance(datamodule.batch, Data)
    assert datamodule.batch is canonical
    assert torch.equal(datamodule.batch.x, canonical_x)
