"""Opt-in OGB-MAG sampled-loader and optimizer preflight."""

from __future__ import annotations

import os

import pytest

if os.environ.get("TOPOBENCH_ALLOW_DOWNLOADS") != "1":
    pytest.skip(
        "Set TOPOBENCH_ALLOW_DOWNLOADS=1 to run real-data integration tests",
        allow_module_level=True,
    )

import math
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import hydra
import torch
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig
from torch_geometric.data import HeteroData

from topobench.utils.config_resolvers import register_all_resolvers
from topobench.utils.model_instantiation import instantiate_model


def _compose(experiment: str, data_dir: Path, output_dir: Path) -> DictConfig:
    """Compose one real-data preflight against a shared OGB-MAG root."""
    GlobalHydra.instance().clear()
    register_all_resolvers()
    config_path = str(Path(__file__).resolve().parents[2] / "configs")
    try:
        with hydra.initialize_config_dir(
            version_base="1.3",
            config_dir=config_path,
            job_name=f"preflight_{experiment}",
        ):
            return hydra.compose(
                config_name="run.yaml",
                overrides=[
                    f"experiment={experiment}",
                    f"paths.data_dir={data_dir}",
                    f"paths.output_dir={output_dir}",
                    "trainer.min_epochs=1",
                    "trainer.max_epochs=1",
                    "logger=[]",
                ],
            )
    finally:
        GlobalHydra.instance().clear()


def _release_iterator(iterator: Iterator[HeteroData]) -> None:
    """Best-effort release of PyG/PyTorch loader workers after one batch."""
    close = getattr(iterator, "close", None)
    if callable(close):
        close()
    worker_iterator = getattr(iterator, "iterator", iterator)
    shutdown_workers = getattr(worker_iterator, "_shutdown_workers", None)
    if callable(shutdown_workers):
        shutdown_workers()


def _take_one(loader: Any) -> HeteroData:
    """Fetch one batch and release its iterator without traversing an epoch."""
    iterator = iter(loader)
    try:
        batch = next(iterator)
    finally:
        _release_iterator(iterator)
    if not isinstance(batch, HeteroData):
        raise TypeError(
            "OGB-MAG preflight expected HeteroData; "
            f"received {type(batch).__name__}"
        )
    return batch


def _accelerator_device() -> torch.device:
    """Select the best accelerator available to the direct preflight."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _prepare_memory_report(device: torch.device) -> None:
    """Reset peak CUDA accounting before one model's allocations."""
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


def _report_accelerator_memory(
    model_name: str,
    device: torch.device,
) -> None:
    """Print peak CUDA memory or an honest platform-specific alternative."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak = torch.cuda.max_memory_allocated(device)
        print(f"{model_name} peak CUDA memory: {peak} bytes")
        return
    if device.type == "mps":
        current_memory = getattr(torch.mps, "current_allocated_memory", None)
        current = (
            current_memory() if callable(current_memory) else "unavailable"
        )
        print(
            f"{model_name} peak MPS memory: unavailable; "
            f"current allocated memory: {current}"
        )
        return
    print(f"{model_name} peak CPU accelerator memory: unavailable")


@pytest.mark.integration
@pytest.mark.download
def test_real_ogb_mag_sampled_preflight(tmp_path: Path) -> None:
    """Fetch each sampled phase and take one optimizer step per backbone."""
    # Keep the large opt-in download in TopoBench's canonical shared cache.
    data_dir = Path(__file__).resolve().parents[2] / "datasets"
    output_dir = tmp_path / "output"
    hgt_cfg = _compose(
        "heterogeneous_ogb_mag_hgt",
        data_dir,
        output_dir,
    )

    pipeline = hydra.utils.instantiate(hgt_cfg.data_pipeline)
    pipeline_output = pipeline.build(hgt_cfg)
    datamodule = pipeline_output.datamodule
    spec = pipeline_output.data_spec
    data = datamodule.data
    assert spec is not None
    assert isinstance(data, HeteroData)
    assert spec.target_node_type == "paper"
    assert spec.num_classes == 349
    assert spec.node_types == tuple(data.node_types)
    assert spec.edge_types == tuple(data.edge_types)
    assert all("x" in data[node_type] for node_type in data.node_types)
    assert all(
        mask_name in data["paper"]
        for mask_name in ("train_mask", "val_mask", "test_mask")
    )

    print(
        "processed node counts:",
        {
            node_type: int(data[node_type].num_nodes)
            for node_type in data.node_types
        },
    )
    print(
        "processed relation edge counts:",
        {
            edge_type: int(data[edge_type].num_edges)
            for edge_type in data.edge_types
        },
    )

    phase_loaders = {
        "train": datamodule.train_dataloader(),
        "val": datamodule.val_dataloader(),
        "test": datamodule.test_dataloader(),
    }
    for phase, loader in phase_loaders.items():
        batch = _take_one(loader)
        print(
            f"{phase} batch node counts:",
            {
                node_type: int(batch[node_type].num_nodes)
                for node_type in batch.node_types
            },
        )
        print(
            f"{phase} batch relation edge counts:",
            {
                edge_type: int(batch[edge_type].num_edges)
                for edge_type in batch.edge_types
            },
        )

    device = _accelerator_device()
    print(f"preflight accelerator: {device.type}")
    for experiment in (
        "heterogeneous_ogb_mag_hgt",
        "heterogeneous_ogb_mag_heterosage",
    ):
        cfg = (
            hgt_cfg
            if experiment == "heterogeneous_ogb_mag_hgt"
            else _compose(experiment, data_dir, output_dir)
        )
        model_name = str(cfg.model.model_name)
        _prepare_memory_report(device)
        model = instantiate_model(cfg, data_spec=spec).to(device)
        model.train()
        model.on_train_epoch_start()
        try:
            optimizer = model.configure_optimizers()["optimizer"]
            batch = _take_one(datamodule.train_dataloader()).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = model.model_step(batch)["loss"]

            assert math.isfinite(float(loss.detach()))
            loss.backward()
            gradients = [
                parameter.grad
                for parameter in model.parameters()
                if parameter.requires_grad and parameter.grad is not None
            ]
            assert gradients
            assert all(
                torch.isfinite(gradient).all() for gradient in gradients
            )
            assert any(
                torch.count_nonzero(gradient) > 0 for gradient in gradients
            )
            optimizer.step()
            _report_accelerator_memory(model_name, device)
        finally:
            model.abort_evaluation()
        del batch, gradients, loss, model, optimizer
        if device.type == "cuda":
            torch.cuda.empty_cache()
