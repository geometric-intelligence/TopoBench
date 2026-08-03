"""Success and failure isolation contracts for throwaway preflight state."""

from __future__ import annotations

import gc
import random
import weakref
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from topobench.preflight import PreflightError

from .test_data_probe import (
    ProbeModel,
    StatefulPhaseDataModule,
    make_observations,
    qualified_runner,
)


def _numpy_state_equal(left: tuple[Any, ...], right: tuple[Any, ...]) -> bool:
    return (
        left[0] == right[0]
        and np.array_equal(left[1], right[1])
        and left[2:] == right[2:]
    )


def _tree(root: Path) -> set[tuple[str, bytes]]:
    if not root.exists():
        return set()
    return {
        (str(path.relative_to(root)), path.read_bytes())
        for path in root.rglob("*")
        if path.is_file()
    }


@pytest.mark.parametrize("failure_phase", [None, "val"])
def test_probe_restores_global_runtime_datamodule_and_sampler_state(
    tmp_path: Path,
    failure_phase: str | None,
) -> None:
    datamodule = StatefulPhaseDataModule()
    observations = make_observations()

    def configure(cfg: Any) -> None:
        cfg.paths.root_dir = str(tmp_path)
        cfg.paths.work_dir = str(tmp_path / "work")
        cfg.paths.output_dir = str(tmp_path / "outputs")
        cfg.paths.log_dir = str(tmp_path / "logs")

    runner, static_result, _ = qualified_runner(
        datamodule,
        configure=configure,
    )
    random.seed(1729)
    np.random.seed(1729)
    torch.manual_seed(1729)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1729)

    python_before = random.getstate()
    numpy_before = np.random.get_state()
    cpu_before = torch.random.get_rng_state().clone()
    cuda_before = [state.clone() for state in torch.cuda.get_rng_state_all()]
    dtype_before = torch.get_default_dtype()
    benchmark_before = torch.backends.cudnn.benchmark
    deterministic_before = torch.backends.cudnn.deterministic
    algorithm_before = torch.are_deterministic_algorithms_enabled()
    datamodule_before = datamodule.state_dict()
    artifacts_before = _tree(tmp_path)

    try:
        if failure_phase is None:
            result = runner.run_probe(
                model_factory=lambda: ProbeModel(
                    observations,
                    mutate_runtime=True,
                ),
                static_result=static_result,
            )
            assert result.passed
        else:
            with pytest.raises(PreflightError, match="intentional val"):
                runner.run_probe(
                    model_factory=lambda: ProbeModel(
                        observations,
                        fail_phase=failure_phase,
                        mutate_runtime=True,
                    ),
                    static_result=static_result,
                )

        assert random.getstate() == python_before
        assert _numpy_state_equal(np.random.get_state(), numpy_before)
        assert torch.equal(torch.random.get_rng_state(), cpu_before)
        assert all(
            torch.equal(actual, expected)
            for actual, expected in zip(
                torch.cuda.get_rng_state_all(),
                cuda_before,
                strict=True,
            )
        )
        assert torch.get_default_dtype() == dtype_before
        assert torch.backends.cudnn.benchmark == benchmark_before
        assert torch.backends.cudnn.deterministic == deterministic_before
        assert torch.are_deterministic_algorithms_enabled() == algorithm_before
        assert datamodule.state_dict() == datamodule_before
        assert _tree(tmp_path) == artifacts_before
    finally:
        random.setstate(python_before)
        np.random.set_state(numpy_before)
        torch.random.set_rng_state(cpu_before)
        if cuda_before:
            torch.cuda.set_rng_state_all(cuda_before)
        torch.set_default_dtype(dtype_before)
        torch.backends.cudnn.benchmark = benchmark_before
        torch.backends.cudnn.deterministic = deterministic_before
        torch.use_deterministic_algorithms(algorithm_before, warn_only=True)


def test_failed_probe_discards_throwaway_runtime_and_leaves_no_artifacts(
    tmp_path: Path,
) -> None:
    datamodule = StatefulPhaseDataModule()
    observations = make_observations()
    references: list[weakref.ReferenceType[Any]] = []

    def configure(cfg: Any) -> None:
        cfg.paths.root_dir = str(tmp_path)
        cfg.paths.work_dir = str(tmp_path / "work")
        cfg.paths.output_dir = str(tmp_path / "outputs")
        cfg.paths.log_dir = str(tmp_path / "logs")

    def factory() -> ProbeModel:
        model = ProbeModel(observations, fail_phase="test")
        references.extend((weakref.ref(model), weakref.ref(model.evaluator)))
        return model

    runner, static_result, _ = qualified_runner(
        datamodule,
        configure=configure,
    )
    before = _tree(tmp_path)

    with pytest.raises(PreflightError, match="intentional test") as raised:
        runner.run_probe(model_factory=factory, static_result=static_result)

    assert datamodule.state_dict() == {
        "cursor": {"train": 0, "val": 0, "test": 0}
    }
    assert _tree(tmp_path) == before
    del raised
    gc.collect()
    assert all(reference() is None for reference in references)


def test_probe_never_mutates_pristine_production_runtime_state() -> None:
    datamodule = StatefulPhaseDataModule()
    probe_observations = make_observations()
    production_observations = make_observations()
    production_model = ProbeModel(production_observations)
    production_parameters = {
        name: value.detach().clone()
        for name, value in production_model.state_dict().items()
    }
    production_evaluator_state = (
        production_model.evaluator.context,
        production_model.evaluator.num_examples,
    )
    runner, static_result, _ = qualified_runner(datamodule)

    result = runner.run_probe(
        model_factory=lambda: ProbeModel(probe_observations),
        static_result=static_result,
    )

    assert result.passed
    assert all(
        torch.equal(production_model.state_dict()[name], value)
        for name, value in production_parameters.items()
    )
    assert (
        production_model.evaluator.context,
        production_model.evaluator.num_examples,
    ) == production_evaluator_state
    assert production_observations["events"] == []
