"""Broad native graph pipeline sentinels."""

import hashlib
import json
import os
import stat
from pathlib import Path

import hydra
import pytest
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import DictConfig, OmegaConf, open_dict

import topobench.evaluator as evaluator_module
import topobench.run as run_module
import topobench.utils.checkpoint_io as checkpoint_io_module
from test._utils.simplified_pipeline import run as run_simplified
from topobench.data.loaders.graph.synthetic import SyntheticGraphDatasetLoader
from topobench.evaluator import EvaluationResult
from topobench.model import SupervisedBatch
from topobench.run import run as run_production
from topobench.utils.checkpoint_io import TrustedCheckpointIO
from topobench.utils.config_resolvers import register_all_resolvers


def _compose(
    dataset: str,
    *,
    epochs: int = 2,
    callbacks: str = "model_checkpoint",
):
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    register_all_resolvers()
    with hydra.initialize(
        version_base="1.3",
        config_path="../../configs",
        job_name="native_graph_pipeline",
    ):
        return hydra.compose(
            config_name="run.yaml",
            overrides=[
                "model=graph/gcn",
                f"dataset=graph/{dataset}",
                f"trainer.max_epochs={epochs}",
                "trainer.min_epochs=1",
                "trainer.check_val_every_n_epoch=1",
                "trainer.accelerator=cpu",
                "trainer.devices=1",
                "dataset.dataloader_params.batch_size="
                f"{1 if dataset == 'SyntheticNodeGraph' else 4}",
                "paths=test",
                f"callbacks={callbacks}",
            ],
        )


_NATIVE_FINGERPRINT_FIELDS = (
    "source_fingerprint",
    "dataset_fingerprint",
    "split_fingerprint",
)


def _assert_sha256(value: object) -> None:
    assert isinstance(value, str)
    assert len(value) == 64
    assert set(value) <= set("0123456789abcdef")


def _qualified_checkpoint_cfg(
    tmp_path: Path,
) -> tuple[DictConfig, Path, Path]:
    """Return a qualified graph config with one explicit checkpoint root."""
    cfg = _compose("SyntheticGraph", epochs=1, callbacks="default")
    output_dir = tmp_path / "qualified-checkpoint-run"
    checkpoint_dir = output_dir / "checkpoints"
    with open_dict(cfg):
        cfg.logger = {}
        cfg.paths.output_dir = str(output_dir)
        cfg.paths.work_dir = str(output_dir)
        cfg.trainer.default_root_dir = str(output_dir)
        cfg.train = True
        cfg.test = False
    with open_dict(cfg.paths):
        cfg.paths.checkpoint_dir = str(checkpoint_dir)
    return cfg, output_dir, checkpoint_dir


def _checkpoint_manifest_path(checkpoint_path: Path) -> Path:
    return Path(f"{checkpoint_path}.manifest.json")


def _write_trusted_resume(
    checkpoint_path: Path,
    *,
    output_dir: Path,
) -> dict[str, object]:
    """Write one private same-run checkpoint and its exact digest manifest."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": {},
            "epoch": 0,
            "global_step": 0,
        },
        checkpoint_path,
    )
    manifest = {
        "schema": "topobench.checkpoint-manifest",
        "schema_version": 1,
        "run_root_sha256": hashlib.sha256(
            str(output_dir.resolve()).encode("utf-8")
        ).hexdigest(),
        "relative_path": checkpoint_path.resolve()
        .relative_to(output_dir.resolve())
        .as_posix(),
        "sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "byte_size": checkpoint_path.stat().st_size,
    }
    manifest_path = _checkpoint_manifest_path(checkpoint_path)
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True),
        encoding="utf-8",
    )
    if os.name == "posix":
        checkpoint_path.chmod(0o600)
        manifest_path.chmod(0o600)
    return manifest


def _config_snapshot(cfg: DictConfig) -> object:
    return OmegaConf.to_container(cfg, resolve=False)


def _assert_checkpoint_rejected_before_construction(
    cfg: DictConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Require checkpoint validation to precede every runtime factory."""
    before = _config_snapshot(cfg)
    construction_calls: list[object] = []

    def construction_must_not_start(*args: object, **kwargs: object) -> object:
        construction_calls.append((args, kwargs))
        raise AssertionError("checkpoint rejection must precede construction")

    monkeypatch.setattr(
        run_module.hydra.utils,
        "instantiate",
        construction_must_not_start,
    )

    with pytest.raises(
        (ValueError, RuntimeError, FileNotFoundError, PermissionError)
    ):
        run_production(cfg)

    assert construction_calls == []
    assert _config_snapshot(cfg) == before


@pytest.mark.parametrize(
    "case",
    ("external", "root_escape", "last", "hpc"),
)
def test_qualified_run_rejects_external_and_magic_resume_paths_before_factories(
    case: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Qualified execution never delegates arbitrary resume paths to Lightning."""
    cfg, output_dir, checkpoint_dir = _qualified_checkpoint_cfg(tmp_path)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if case == "external":
        checkpoint_path = tmp_path / "external.ckpt"
        torch.save({"state_dict": {}}, checkpoint_path)
        ckpt_path = str(checkpoint_path)
    elif case == "root_escape":
        checkpoint_path = output_dir / "escaped.ckpt"
        torch.save({"state_dict": {}}, checkpoint_path)
        ckpt_path = str(checkpoint_dir / ".." / checkpoint_path.name)
    else:
        ckpt_path = case
    with open_dict(cfg):
        cfg.ckpt_path = ckpt_path

    _assert_checkpoint_rejected_before_construction(cfg, monkeypatch)


@pytest.mark.parametrize(
    "case",
    (
        "missing_checkpoint",
        "missing_manifest",
        "missing_digest",
        "stale_digest",
        "missing_path",
        "stale_path",
    ),
)
def test_qualified_run_rejects_incomplete_or_stale_resume_manifest(
    case: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A trusted location alone cannot qualify a full Lightning resume."""
    cfg, output_dir, checkpoint_dir = _qualified_checkpoint_cfg(tmp_path)
    checkpoint_path = checkpoint_dir / "resume.ckpt"
    if case != "missing_checkpoint":
        manifest = _write_trusted_resume(
            checkpoint_path,
            output_dir=output_dir,
        )
        manifest_path = _checkpoint_manifest_path(checkpoint_path)
        if case == "missing_manifest":
            manifest_path.unlink()
        else:
            if case == "missing_digest":
                manifest.pop("sha256")
            elif case == "stale_digest":
                manifest["sha256"] = "0" * 64
            elif case == "missing_path":
                manifest.pop("relative_path")
            elif case == "stale_path":
                manifest["relative_path"] = "checkpoints/other.ckpt"
            manifest_path.write_text(
                json.dumps(manifest, sort_keys=True),
                encoding="utf-8",
            )
    with open_dict(cfg):
        cfg.ckpt_path = str(checkpoint_path)

    _assert_checkpoint_rejected_before_construction(cfg, monkeypatch)


@pytest.mark.skipif(
    os.name != "posix",
    reason="checkpoint mode ownership is a POSIX contract",
)
def test_qualified_run_rejects_non_private_resume_mode_before_factories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A same-run digest cannot qualify a world-readable checkpoint."""
    cfg, output_dir, checkpoint_dir = _qualified_checkpoint_cfg(tmp_path)
    checkpoint_path = checkpoint_dir / "public.ckpt"
    _write_trusted_resume(checkpoint_path, output_dir=output_dir)
    checkpoint_path.chmod(0o644)
    assert stat.S_IMODE(checkpoint_path.stat().st_mode) == 0o644
    with open_dict(cfg):
        cfg.ckpt_path = str(checkpoint_path)

    _assert_checkpoint_rejected_before_construction(cfg, monkeypatch)


def test_qualified_run_accepts_private_same_run_digest_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A complete same-run manifest admits the concrete checkpoint to runtime."""
    cfg, output_dir, checkpoint_dir = _qualified_checkpoint_cfg(tmp_path)
    checkpoint_path = checkpoint_dir / "resume.ckpt"
    manifest = _write_trusted_resume(checkpoint_path, output_dir=output_dir)
    assert set(manifest) == {
        "schema",
        "schema_version",
        "run_root_sha256",
        "relative_path",
        "sha256",
        "byte_size",
    }
    with open_dict(cfg):
        cfg.ckpt_path = str(checkpoint_path)
    before = _config_snapshot(cfg)
    construction_calls: list[object] = []

    class ConstructionReached(Exception):
        pass

    def stop_at_pipeline(*args: object, **kwargs: object) -> object:
        construction_calls.append((args, kwargs))
        raise ConstructionReached

    monkeypatch.setattr(
        run_module.hydra.utils,
        "instantiate",
        stop_at_pipeline,
    )

    with pytest.raises(ConstructionReached):
        run_production(cfg)

    assert len(construction_calls) == 1
    assert _config_snapshot(cfg) == before


def test_full_resume_loads_the_descriptor_whose_digest_was_validated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "run"
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_path = checkpoint_dir / "resume.ckpt"
    _write_trusted_resume(checkpoint_path, output_dir=output_dir)
    replacement_path = checkpoint_dir / "replacement.ckpt"
    torch.save({"marker": "replacement"}, replacement_path)
    replacement_path.chmod(0o600)
    real_digest = checkpoint_io_module.digest_open_file

    def replace_path_after_digest(file):
        result = real_digest(file)
        os.replace(replacement_path, checkpoint_path)
        return result

    monkeypatch.setattr(
        checkpoint_io_module,
        "digest_open_file",
        replace_path_after_digest,
    )

    checkpoint = TrustedCheckpointIO(
        output_root=output_dir,
        checkpoint_root=checkpoint_dir,
    ).load_checkpoint(checkpoint_path)

    assert checkpoint["state_dict"] == {}
    assert "marker" not in checkpoint


def test_selected_cleanup_preserves_identical_replacement_inode(
    tmp_path: Path,
) -> None:
    """Digest equality cannot authorize deletion of a different file object."""
    _, output_dir, checkpoint_dir = _qualified_checkpoint_cfg(tmp_path)
    checkpoint_path = checkpoint_dir / "selected.ckpt"
    checkpoint_io = TrustedCheckpointIO(
        output_root=output_dir,
        checkpoint_root=checkpoint_dir,
    )
    checkpoint_io.save_checkpoint(
        {
            "state_dict": {"weight": torch.tensor([1.0])},
            "epoch": 0,
            "global_step": 0,
        },
        checkpoint_path,
    )
    loaded = checkpoint_io_module.load_selected_checkpoint(
        checkpoint_path,
        output_root=output_dir,
        checkpoint_root=checkpoint_dir,
    )

    replacement_path = checkpoint_dir / "replacement.ckpt"
    replacement_path.write_bytes(checkpoint_path.read_bytes())
    replacement_identity = (
        replacement_path.stat().st_dev,
        replacement_path.stat().st_ino,
    )
    assert replacement_identity != (
        checkpoint_path.stat().st_dev,
        checkpoint_path.stat().st_ino,
    )
    replacement_path.replace(checkpoint_path)

    run_module._remove_loaded_checkpoint_artifacts(
        checkpoint_path,
        loaded,
        checkpoint_root=checkpoint_dir,
    )

    assert checkpoint_path.is_file()
    assert (
        checkpoint_path.stat().st_dev,
        checkpoint_path.stat().st_ino,
    ) == replacement_identity
    assert not _checkpoint_manifest_path(checkpoint_path).exists()
    assert not Path(f"{checkpoint_path}.state.pt").exists()


def test_experimental_run_preserves_explicit_external_resume_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Experimental execution retains its explicit unqualified resume escape."""
    cfg, _, _ = _qualified_checkpoint_cfg(tmp_path)
    checkpoint_path = tmp_path / "experimental-external.ckpt"
    torch.save({"state_dict": {}}, checkpoint_path)
    with open_dict(cfg):
        cfg.execution_profile = "experimental"
        cfg.ckpt_path = str(checkpoint_path)
    before = _config_snapshot(cfg)
    construction_calls: list[object] = []

    class ConstructionReached(Exception):
        pass

    def stop_at_pipeline(*args: object, **kwargs: object) -> object:
        construction_calls.append((args, kwargs))
        raise ConstructionReached

    monkeypatch.setattr(
        run_module.hydra.utils,
        "instantiate",
        stop_at_pipeline,
    )

    with pytest.raises(ConstructionReached):
        run_production(cfg)

    assert len(construction_calls) == 1
    assert cfg.ckpt_path == str(checkpoint_path)
    assert _config_snapshot(cfg) == before


def test_qualified_full_resume_uses_manifested_same_run_checkpoint(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "qualified-resume"

    def configured_run(*, max_epochs: int, ckpt_path: str | None = None):
        cfg = _compose(
            "SyntheticGraph",
            epochs=max_epochs,
            callbacks="default",
        )
        with open_dict(cfg):
            cfg.logger = {}
            cfg.paths.output_dir = str(output_dir)
            cfg.paths.work_dir = str(output_dir)
            cfg.trainer.default_root_dir = str(output_dir)
            cfg.trainer.limit_train_batches = 1
            cfg.trainer.limit_val_batches = 1
            cfg.trainer.enable_progress_bar = False
            cfg.enable_progress_bar = False
            cfg.train = True
            cfg.test = False
            cfg.ckpt_path = ckpt_path
        return run_production(cfg)

    _, first_objects = configured_run(max_epochs=1)
    checkpoint_callbacks = [
        callback
        for callback in first_objects["callbacks"]
        if isinstance(callback, ModelCheckpoint)
    ]
    assert len(checkpoint_callbacks) == 1
    checkpoint_path = Path(checkpoint_callbacks[0].best_model_path)
    assert checkpoint_path.is_file()
    assert _checkpoint_manifest_path(checkpoint_path).is_file()
    assert Path(f"{checkpoint_path}.state.pt").is_file()
    first_global_step = first_objects["trainer"].global_step

    _, resumed_objects = configured_run(
        max_epochs=2,
        ckpt_path=str(checkpoint_path),
    )

    assert resumed_objects["trainer"].global_step > first_global_step


@pytest.mark.parametrize(
    "dataset",
    ("SyntheticGraph", "SyntheticGraphRegression"),
)
def test_native_graph_pipeline_runs_two_epochs_with_real_batches(
    dataset: str,
) -> None:
    result = run_simplified(_compose(dataset))

    assert result["epochs_completed"] >= 2
    assert result["observed_train_batch_size"] > 1
    assert {"train/loss", "val/loss"} <= result["fit_metrics"].keys()
    assert result["test_results"]
    assert result["test_results"][0]["test/num_examples"] > 0


@pytest.mark.parametrize(
    ("dataset", "output_kind"),
    [
        ("SyntheticGraph", "graph"),
        ("SyntheticNodeGraph", "homogeneous"),
    ],
)
def test_graph_pipeline_observes_task_level_output_kind(
    dataset: str,
    output_kind: str,
) -> None:
    """Graph-level and node-level outputs retain distinct runtime kinds."""
    cfg = _compose(dataset, epochs=1)

    capability = (
        hydra.utils.instantiate(cfg.data_pipeline).build(cfg).capability_spec
    )

    assert capability is not None
    assert capability.selector == f"graph/{dataset}"
    assert capability.data_domain == "graph"
    assert capability.output_kind == output_kind
    assert capability.feature_widths == (("node", 4),)
    assert capability.num_classes == 2
    assert capability.target_node_type is None


def test_graph_run_returns_results_and_cleans_selected_checkpoint(
    tmp_path: Path,
) -> None:
    """The graph run returns exact results and removes every trusted file."""
    output_dir = tmp_path / "graph-selected-checkpoint"
    cfg = _compose(
        "SyntheticGraph",
        epochs=1,
        callbacks="default",
    )
    with open_dict(cfg):
        cfg.logger = {}
        cfg.paths.output_dir = str(output_dir)
        cfg.paths.work_dir = str(output_dir)
        cfg.trainer.default_root_dir = str(output_dir)
        cfg.trainer.limit_train_batches = 2
        cfg.trainer.limit_val_batches = 1.0
        cfg.trainer.limit_test_batches = 1.0
        cfg.trainer.enable_progress_bar = False
        cfg.enable_progress_bar = False
        cfg.delete_checkpoint_after_test = True

    metrics, objects = run_production(cfg)
    checkpoint_callbacks = [
        callback
        for callback in objects["callbacks"]
        if isinstance(callback, ModelCheckpoint)
    ]
    assert len(checkpoint_callbacks) == 1
    checkpoint_path = Path(checkpoint_callbacks[0].best_model_path)
    checkpoint_manifest = _checkpoint_manifest_path(checkpoint_path)
    checkpoint_state = Path(f"{checkpoint_path}.state.pt")
    assert not checkpoint_path.exists()
    assert not checkpoint_manifest.exists()
    assert not checkpoint_state.exists()

    selected = objects["selected_checkpoint_results"]
    publications = objects["selected_checkpoint_publications"]
    assert tuple(publications) == ("val", "test")
    assert tuple(selected) == ("val", "test")
    assert all(
        isinstance(selected[split], EvaluationResult)
        for split in ("val", "test")
    )
    assert selected["val"] is not selected["test"]
    assert {
        (result.context.split, result.context.pass_kind, result.context.policy)
        for result in selected.values()
    } == {
        ("val", "selected_checkpoint", "exact"),
        ("test", "selected_checkpoint", "exact"),
    }
    assert (
        selected["val"].context.checkpoint_id
        == selected["test"].context.checkpoint_id
    )
    assert selected["val"].context.checkpoint_id is not None

    datamodule = objects["datamodule"]
    expected_counts = {
        "val": sum(
            int(batch.num_graphs) for batch in datamodule.val_dataloader()
        ),
        "test": sum(
            int(batch.num_graphs) for batch in datamodule.test_dataloader()
        ),
    }
    assert {"train/num_examples", "val/num_examples"} <= metrics.keys()
    assert not any(
        name.startswith(("val_best_rerun/", "test_best_rerun/"))
        for name in metrics
    )
    for split, result in selected.items():
        namespace = f"evaluations/best_checkpoint/{split}/"
        assert result.num_examples == expected_counts[split]
        count = metrics[f"{namespace}num_examples"]
        assert count == result.num_examples
        assert type(count) is int
        for name, value in result.metrics.items():
            torch.testing.assert_close(
                torch.as_tensor(metrics[f"{namespace}{name}"]),
                torch.as_tensor(value),
            )


def test_graph_run_emits_bounded_source_slice_metrics(tmp_path: Path) -> None:
    output_dir = tmp_path / "graph-source-slices"
    cfg = _compose("SyntheticGraph", epochs=1, callbacks="default")
    with open_dict(cfg):
        cfg.logger = {}
        cfg.paths.output_dir = str(output_dir)
        cfg.paths.work_dir = str(output_dir)
        cfg.trainer.default_root_dir = str(output_dir)
        cfg.trainer.limit_train_batches = 2
        cfg.trainer.limit_val_batches = 1.0
        cfg.trainer.limit_test_batches = 1.0
        cfg.trainer.enable_progress_bar = False
        cfg.enable_progress_bar = False
        cfg.evaluation_artifacts.metadata_fields = ["source"]
        cfg.evaluation_artifacts.evaluation_slices = {
            "source": {"max_categories": 1, "min_rows": 1}
        }

    _, objects = run_production(cfg)

    selected = objects["selected_checkpoint_results"]
    publications = objects["selected_checkpoint_publications"]
    assert tuple(publications) == ("val", "test")
    for split, publication in publications.items():
        document = json.loads(
            publication.metrics_file.path.read_text(encoding="utf-8")
        )
        source_slices = document["slices"]["source"]
        assert tuple(source_slices) == ("SyntheticGraph",)
        source_result = source_slices["SyntheticGraph"]
        assert source_result["num_examples"] == selected[split].num_examples
        assert source_result["metrics"] == selected[split].metrics


def test_graph_prediction_rows_keep_stable_sample_ids_across_shuffle() -> None:
    """Graph identities survive shuffled collation and a short final batch."""
    cfg = _compose("SyntheticGraph", epochs=1)
    with open_dict(cfg):
        cfg.dataset.dataloader_params.batch_size = 3

    pipeline_output = hydra.utils.instantiate(cfg.data_pipeline).build(cfg)
    adapter = pipeline_output.prediction_row_adapter

    def collect_one_epoch() -> tuple[list[int], list[int]]:
        batch_sizes: list[int] = []
        sample_ids: list[int] = []
        for batch in pipeline_output.datamodule.train_dataloader():
            num_examples = int(batch.num_graphs)
            supervised = SupervisedBatch(
                logits=torch.zeros((num_examples, 2)),
                targets=batch.y.reshape(-1),
                num_examples=num_examples,
                row_indices=torch.arange(num_examples),
            )
            payload = adapter.adapt(batch, supervised, phase="train")

            assert isinstance(payload, evaluator_module.PredictionPayload)
            assert payload.identity.key == ("sample_id",)
            batch_ids = [
                int(sample_id)
                for sample_id in payload.identity.columns["sample_id"].tolist()
            ]
            assert len(batch_ids) == num_examples
            assert payload.prediction.shape[0] == num_examples
            batch_sizes.append(num_examples)
            sample_ids.extend(batch_ids)
        return batch_sizes, sample_ids

    first_sizes, first_ids = collect_one_epoch()
    second_sizes, second_ids = collect_one_epoch()

    assert first_sizes == [3, 3, 2]
    assert second_sizes == first_sizes
    assert len(first_ids) == len(set(first_ids)) == 8
    assert set(first_ids) == set(range(8))
    assert sorted(second_ids) == sorted(first_ids)


def test_native_graph_provenance_contains_observed_sha256_fingerprints() -> (
    None
):
    cfg = _compose("SyntheticGraph", epochs=1)

    output = hydra.utils.instantiate(cfg.data_pipeline).build(cfg)

    assert output.provenance_input is not None
    for field in _NATIVE_FINGERPRINT_FIELDS:
        _assert_sha256(output.provenance_input[field])


def test_native_graph_content_changes_source_and_dataset_fingerprints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _compose("SyntheticGraph", epochs=1)
    original_load_dataset = SyntheticGraphDatasetLoader.load_dataset
    content_changed = False

    def load_observed_dataset(loader: SyntheticGraphDatasetLoader):
        dataset = original_load_dataset(loader)
        if content_changed:
            dataset._data.x[0, 0].add_(1.0)
        return dataset

    monkeypatch.setattr(
        SyntheticGraphDatasetLoader,
        "load_dataset",
        load_observed_dataset,
    )
    baseline = hydra.utils.instantiate(cfg.data_pipeline).build(cfg)
    content_changed = True
    changed = hydra.utils.instantiate(cfg.data_pipeline).build(cfg)

    assert baseline.provenance_input is not None
    assert changed.provenance_input is not None
    assert baseline.supervision_counts == changed.supervision_counts
    for field in ("source_fingerprint", "dataset_fingerprint"):
        assert (
            baseline.provenance_input[field] != changed.provenance_input[field]
        )


def test_native_graph_split_fingerprint_tracks_phase_membership(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _compose("SyntheticGraph", epochs=1)
    original_load_dataset = SyntheticGraphDatasetLoader.load_dataset
    membership_changed = False

    def load_observed_dataset(loader: SyntheticGraphDatasetLoader):
        dataset = original_load_dataset(loader)
        if membership_changed:
            split_idx = {
                phase: indices.copy()
                for phase, indices in dataset.split_idx.items()
            }
            split_idx["train"][0], split_idx["valid"][0] = (
                split_idx["valid"][0],
                split_idx["train"][0],
            )
            dataset.split_idx = split_idx
        return dataset

    monkeypatch.setattr(
        SyntheticGraphDatasetLoader,
        "load_dataset",
        load_observed_dataset,
    )
    baseline = hydra.utils.instantiate(cfg.data_pipeline).build(cfg)
    membership_changed = True
    changed = hydra.utils.instantiate(cfg.data_pipeline).build(cfg)

    assert baseline.provenance_input is not None
    assert changed.provenance_input is not None
    assert baseline.supervision_counts == changed.supervision_counts
    assert (
        baseline.provenance_input["split_fingerprint"]
        != changed.provenance_input["split_fingerprint"]
    )


@pytest.mark.download
@pytest.mark.parametrize("dataset", ("MUTAG", "AQSOL"))
def test_real_graph_release_lifecycle(
    dataset: str,
    tmp_path: Path,
) -> None:
    """Keep real classification and scalar-regression download gates."""
    cfg = _compose(dataset, epochs=1)
    with open_dict(cfg):
        cfg.paths.data_dir = str(tmp_path / "datasets")
        cfg.paths.output_dir = str(tmp_path / "output")
        cfg.paths.work_dir = str(tmp_path / "output")
        cfg.paths.checkpoint_dir = str(tmp_path / "output" / "checkpoints")
        cfg.trainer.default_root_dir = str(tmp_path / "output")
    result = run_simplified(cfg)

    assert result["epochs_completed"] >= 1
    assert result["observed_train_batch_size"] > 1
    assert result["test_results"]


def test_native_graph_run_configuration_enables_automatic_preflight() -> None:
    cfg = _compose("SyntheticGraph", epochs=1)

    assert cfg.execution_profile == "qualified"
    assert cfg.preflight.enabled is True
    assert cfg.preflight.execution_probe is True
