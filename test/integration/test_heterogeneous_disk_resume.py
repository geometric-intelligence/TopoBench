"""Exact production-Lightning resume for typed heterogeneous disk strategies."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from omegaconf import DictConfig, open_dict

from test.integration.test_graph_disk_resume import (
    INTERRUPTION_BOUNDARIES,
    _assert_resume_equivalent,
    _LifecycleCase,
    _run_complete,
    _run_interrupted,
)
from test.pipeline.test_disk_heterogeneous_pipeline import (
    _build,
    _neighbor_source,
    _parquet_cfg,
)
from topobench.data.loaders.parquet import ParquetTypedGraphSource


def _source(root: Path, mode: str) -> ParquetTypedGraphSource:
    source = _neighbor_source(root, pca=True)
    if mode == "neighbor":
        return source
    return ParquetTypedGraphSource(
        replace(
            source.spec,
            partition=replace(source.spec.partition, strategy="cluster"),
        )
    )


def _config(
    source: ParquetTypedGraphSource,
    run_root: Path,
    store_path: Path | None,
    mode: str,
) -> DictConfig:
    cfg = _parquet_cfg(
        source,
        run_root,
        store_path=store_path,
        fitted_transform=True,
    )
    with open_dict(cfg):
        cfg.execution_profile = "experimental"
    with open_dict(cfg.dataset.parameters):
        cfg.dataset.parameters.metrics = ["accuracy"]
    if mode == "cluster":
        with open_dict(cfg.dataset.dataloader_params):
            cfg.dataset.dataloader_params.mode = "full_batch"
            cfg.dataset.dataloader_params.clusters_per_batch = 1
            cfg.dataset.dataloader_params.partition_groups = None
    return cfg


@pytest.fixture(scope="module", params=("cluster", "neighbor"))
def heterogeneous_lifecycle(
    request: pytest.FixtureRequest,
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[str, _LifecycleCase]:
    mode = str(request.param)
    root = tmp_path_factory.mktemp(f"typed-heterogeneous-{mode}-resume")
    source = _source(root / "source", mode)
    built_cfg = _config(source, root / "build", None, mode)
    built = _build(built_cfg)
    adapter = built.prediction_row_adapter
    assert adapter is not None
    store_path = adapter.store_path
    built.datamodule.close()

    def config_factory(run_root: Path, path: Path) -> DictConfig:
        return _config(source, run_root, path, mode)

    reference = _run_complete(
        config_factory(root / "reference", store_path),
        root / "reference" / "trainer",
    )
    return mode, _LifecycleCase(
        source,
        store_path,
        root,
        reference,
        config_factory,
    )


@pytest.mark.parametrize("boundary", INTERRUPTION_BOUNDARIES)
def test_heterogeneous_cluster_and_neighbor_resume_exactly_at_every_boundary(
    heterogeneous_lifecycle: tuple[str, _LifecycleCase],
    boundary: str,
) -> None:
    mode, case = heterogeneous_lifecycle
    run_root = case.root / f"{mode}-{boundary}"
    observed, resume = _run_interrupted(
        case.config_factory(run_root / "crash-cfg", case.store_path),
        case.config_factory(run_root / "resume-cfg", case.store_path),
        run_root,
        boundary,
    )
    _assert_resume_equivalent(observed, case.reference, resume, boundary)
    assert resume.state["identity"]["sampling_strategy_type"] == (
        f"heterogeneous-{mode}"
    )
    assert resume.state["identity"]["fitted_transform_state_key"] is not None
