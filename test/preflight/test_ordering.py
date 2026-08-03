"""Runtime ordering tests for the automatic pre-training gate."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import hydra
import pytest
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

import topobench.run as run_module
from topobench.callbacks.input_pipeline import InputPipelineCallback
from topobench.preflight import (
    PreflightCheck,
    PreflightError,
    PreflightResult,
    PreflightRunner,
)
from topobench.profiling.execution_events import ExecutionOperation
from topobench.utils.config_resolvers import register_all_resolvers
from topobench.utils.instantiators import (
    validate_execution_profile,
    validate_profile_capability,
)
from topobench.utils.model_instantiation import instantiate_model


def _run_cfg(*, train: bool = True, test: bool = False):
    return OmegaConf.create(
        {
            "seed": 17,
            "execution_profile": "qualified",
            "preflight": {
                "enabled": True,
                "execution_probe": True,
                "compile_probe": "configured",
                "artifact_payload_probe": True,
                "prefetch_memory_ceiling_bytes": 1048576,
                "monitor_nan_policy": None,
            },
            "data_pipeline": {"_target_": "tests.FakePipeline"},
            "model": {"_target_": "tests.FakeModel"},
            "evaluator": {
                "task": "classification",
                "num_classes": 2,
                "metrics": ["accuracy"],
                "policy": {"train": "online", "val": "exact", "test": "exact"},
                "exact": {"max_ranking_bytes": 1048576},
            },
            "dataset": {
                "parameters": {"task": "classification", "num_classes": 2},
                "loader": {
                    "parameters": {
                        "partition": {
                            "backend": "pyg",
                            "memory_limit_bytes": 1048576,
                        },
                        "reproducibility": {
                            "save_reproducibility_bundle": True,
                        },
                    }
                },
            },
            "optimizer": {},
            "loss": {},
            "trainer": {
                "_target_": "tests.FakeTrainer",
                "accelerator": "cpu",
                "devices": 1,
            },
            "paths": {
                "output_dir": "test-output",
                "checkpoint_dir": "test-output/checkpoints",
            },
            "callbacks": None,
            "logger": None,
            "train": train,
            "test": test,
            "ckpt_path": None,
        }
    )


def _result(*, qualified: bool = True) -> PreflightResult:
    return PreflightResult(
        enabled=True,
        qualified=qualified,
        checks=(PreflightCheck("test", True, "passed"),),
    )


def _install_runtime_fakes(
    monkeypatch: pytest.MonkeyPatch,
    events: list[str],
    *,
    failure_stage: str | None = None,
) -> tuple[MagicMock, MagicMock]:
    pipeline_output = SimpleNamespace(
        datamodule=MagicMock(),
        preprocessing_time=0.0,
        data_spec=None,
        capability_spec=MagicMock(name="observed_capability"),
    )
    pipeline = MagicMock()
    pipeline.build.side_effect = lambda cfg: (
        events.append("pipeline_build") or pipeline_output
    )
    trainer = MagicMock()
    trainer.callback_metrics = {}
    trainer.fit.side_effect = lambda **kwargs: events.append("fit")
    observed_validation = MagicMock(name="observed_capability_validation")
    model_count = 0

    def instantiate(config, **kwargs):
        target = config.get("_target_")
        if target == "tests.FakePipeline":
            events.append("pipeline_construct")
            pipeline.execution_monitor = kwargs.get("execution_monitor")
            return pipeline
        if target == "tests.FakeTrainer":
            events.append("trainer_construct")
            return trainer
        raise AssertionError(f"unexpected Hydra construction: {target!r}")

    def instantiate_model(
        cfg,
        *,
        data_spec,
        capability_validation,
        profile_record,
    ):
        nonlocal model_count
        del profile_record
        del cfg, data_spec
        assert capability_validation is observed_validation
        model_count += 1
        events.append(
            "throwaway_model" if model_count == 1 else "production_model"
        )
        return MagicMock()

    class FakeRunner:
        def __init__(self, cfg, pipeline_output):
            del cfg, pipeline_output

        def validate_static(self):
            events.append("static_preflight")
            if failure_stage == "static":
                failed = PreflightResult(
                    enabled=True,
                    qualified=False,
                    checks=(PreflightCheck("static", False, "failed"),),
                )
                raise PreflightError(failed)
            return _result()

        def run_probe(self, *, model_factory, static_result):
            del static_result
            events.append("isolated_probe")
            model_factory()
            if failure_stage == "probe":
                failed = PreflightResult(
                    enabled=True,
                    qualified=False,
                    checks=(PreflightCheck("probe", False, "failed"),),
                )
                raise PreflightError(failed)
            return _result()

    monkeypatch.setattr(
        run_module.L,
        "seed_everything",
        lambda seed, workers: events.append(f"seed:{seed}:{workers}"),
    )
    monkeypatch.setattr(run_module.hydra.utils, "instantiate", instantiate)
    monkeypatch.setattr(
        run_module,
        "validate_execution_profile",
        MagicMock(name="execution_profile_record"),
    )
    monkeypatch.setattr(
        run_module,
        "validate_profile_capability",
        MagicMock(
            side_effect=[
                MagicMock(name="static_capability_validation"),
                observed_validation,
            ]
        ),
    )
    monkeypatch.setattr(run_module, "instantiate_model", instantiate_model)
    monkeypatch.setattr(run_module, "PreflightRunner", FakeRunner)
    monkeypatch.setattr(
        run_module,
        "instantiate_callbacks",
        lambda cfg, *, input_pipeline_monitor=None: (
            events.append("callbacks_construct") or []
        ),
    )
    monkeypatch.setattr(
        run_module,
        "instantiate_loggers",
        lambda cfg: events.append("loggers_construct") or [],
    )
    monkeypatch.setattr(
        run_module,
        "rerun_best_model_checkpoint",
        lambda **kwargs: events.append("test"),
    )
    return trainer, pipeline


def test_monitor_precedes_pipeline_and_is_adopted_only_after_preflight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    _, pipeline = _install_runtime_fakes(monkeypatch, events)
    monitor = MagicMock(name="shared_monitor")
    owner = SimpleNamespace(monitor=monitor)
    cfg = _run_cfg()
    cfg.callbacks = OmegaConf.create(
        {
            "input_pipeline": {
                "_target_": (
                    "topobench.callbacks.input_pipeline.InputPipelineCallback"
                ),
                "event_log_path": "test-output/execution/events.jsonl",
            }
        }
    )

    monkeypatch.setattr(
        run_module,
        "_instantiate_execution_monitor",
        lambda callbacks: events.append("monitor_construct") or monitor,
    )

    def instantiate_callbacks(callbacks, *, input_pipeline_monitor=None):
        del callbacks
        assert input_pipeline_monitor is monitor
        events.append("callbacks_construct")
        return [owner]

    monkeypatch.setattr(
        run_module,
        "instantiate_callbacks",
        instantiate_callbacks,
    )
    monkeypatch.setattr(
        run_module,
        "_shared_execution_monitor",
        lambda callbacks: callbacks[0].monitor,
    )

    run_module.run(cfg)

    assert pipeline.call_args is None
    assert events.index("monitor_construct") < events.index(
        "pipeline_construct"
    )
    assert events.index("pipeline_build") < events.index("static_preflight")
    assert events.index("static_preflight") < events.index(
        "callbacks_construct"
    )
    assert events.index("callbacks_construct") < events.index(
        "trainer_construct"
    )
    assert pipeline.build.call_args.args == (cfg,)
    assert pipeline.execution_monitor is monitor


def test_pipeline_conversion_event_survives_callback_monitor_adoption(
    tmp_path,
) -> None:
    callbacks = OmegaConf.create(
        {
            "input_pipeline": {
                "_target_": (
                    "topobench.callbacks.input_pipeline.InputPipelineCallback"
                ),
                "event_log_path": str(tmp_path / "events.jsonl"),
                "sample_every_n": 1,
            }
        }
    )

    monitor = run_module._instantiate_execution_monitor(callbacks)
    assert monitor is not None
    token = monitor.begin(
        ExecutionOperation.CONVERSION,
        phase="pipeline_build",
    )
    monitor.finish(token, row_count=3)

    callback = InputPipelineCallback(
        tmp_path / "events.jsonl",
        sample_every_n=1,
        monitor=monitor,
    )

    assert callback.monitor is monitor
    event = monitor.drain()[0]
    assert event.operation is ExecutionOperation.CONVERSION
    assert event.phase == "pipeline_build"
    assert event.row_count == 3


def test_run_constructs_preflight_and_production_objects_in_runtime_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    _install_runtime_fakes(monkeypatch, events)

    _, objects = run_module.run(_run_cfg())

    assert events == [
        "seed:17:True",
        "pipeline_construct",
        "pipeline_build",
        "static_preflight",
        "isolated_probe",
        "throwaway_model",
        "seed:17:True",
        "production_model",
        "callbacks_construct",
        "loggers_construct",
        "trainer_construct",
        "fit",
    ]
    assert objects["preflight"] == _result()


@pytest.mark.parametrize("failure_stage", ("static", "probe"))
def test_preflight_failure_prevents_all_production_and_training_side_effects(
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
) -> None:
    events: list[str] = []
    _install_runtime_fakes(monkeypatch, events, failure_stage=failure_stage)

    with pytest.raises(PreflightError):
        run_module.run(_run_cfg(train=True, test=True))

    assert "production_model" not in events
    assert "callbacks_construct" not in events
    assert "loggers_construct" not in events
    assert "trainer_construct" not in events
    assert "fit" not in events
    assert "test" not in events


def test_disabling_preflight_requires_experimental_profile() -> None:
    cfg = OmegaConf.create(
        {
            "execution_profile": "qualified",
            "preflight": {"enabled": False, "execution_probe": False},
        }
    )

    with pytest.raises(PreflightError, match="experimental"):
        PreflightRunner(cfg, pipeline_output=None).validate_static()


def test_experimental_disable_is_structured_and_unqualified() -> None:
    cfg = OmegaConf.create(
        {
            "execution_profile": "experimental",
            "preflight": {"enabled": False, "execution_probe": False},
        }
    )

    result = PreflightRunner(cfg, pipeline_output=None).validate_static()

    assert result.enabled is False
    assert result.passed is True
    assert result.qualified is False
    assert result.as_record()["qualified"] is False
    assert result.checks[-1].check_id == "preflight.enabled"


def test_config_only_run_records_probe_as_not_requested_without_model() -> (
    None
):
    cfg = _run_cfg(train=False, test=False)
    runner = PreflightRunner(cfg, pipeline_output=None)
    static_result = runner.validate_static()
    model_requested = False

    def model_factory():
        nonlocal model_requested
        model_requested = True
        raise AssertionError(
            "config-only run must not construct a probe model"
        )

    result = runner.run_probe(
        model_factory=model_factory,
        static_result=static_result,
    )

    assert model_requested is False
    assert result.passed is True
    assert result.qualified is True
    assert result.checks[-1].check_id == "execution.not_requested"
    assert "not executed" in result.checks[-1].detail


def test_isolated_probe_uses_real_model_supervision_loss_and_typed_evaluator() -> (
    None
):
    GlobalHydra.instance().clear()
    register_all_resolvers()
    with hydra.initialize(
        version_base="1.3",
        config_path="../../configs",
        job_name="real_preflight_probe",
    ):
        cfg = hydra.compose(
            config_name="run.yaml",
            overrides=[
                "dataset=graph/SyntheticGraph",
                "model=graph/gcn",
                "paths=test",
                "trainer=cpu",
                "logger=csv",
                "callbacks=model_checkpoint",
            ],
        )
    OmegaConf.set_struct(cfg, False)
    OmegaConf.update(cfg, "paths.root_dir", "test-root", merge=False)
    OmegaConf.update(cfg, "paths.work_dir", "test-work", merge=False)
    pipeline = hydra.utils.instantiate(
        cfg.data_pipeline,
        execution_monitor=None,
    )
    pipeline_output = pipeline.build(cfg)
    runner = PreflightRunner(cfg, pipeline_output)

    static_result = runner.validate_static()
    capability_validation = validate_profile_capability(
        cfg,
        profile_record=validate_execution_profile(cfg),
        observed=pipeline_output.capability_spec,
    )
    result = runner.run_probe(
        model_factory=lambda: instantiate_model(
            cfg,
            data_spec=pipeline_output.data_spec,
            capability_validation=capability_validation,
        ),
        static_result=static_result,
    )

    assert result.passed is True
    assert result.qualified is True
    execution_checks = tuple(
        check
        for check in result.checks
        if check.check_id.startswith("execution.")
    )
    assert tuple(check.check_id for check in execution_checks) == (
        "execution.representative_batch",
        "execution.gradient",
        "execution.optimizer",
        "execution.scheduler",
        "execution.compile",
        "execution.structured_checks",
        "execution.reproducibility_payload",
        "execution.prediction_payload",
    )
    assert all(check.passed for check in execution_checks)
    assert len({check.check_id for check in execution_checks}) == len(
        execution_checks
    )
