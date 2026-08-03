"""Unit tests for config instantiators and execution-profile validation."""

from pathlib import Path

import hydra
import pytest
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf, open_dict

from topobench.utils.artifact_logging import ARTIFACT_LOGGER_TARGETS
from topobench.utils.config_resolvers import register_all_resolvers
from topobench.utils.instantiators import (
    ExecutionProfileRecord,
    instantiate_callbacks,
    instantiate_loggers,
    validate_execution_profile,
    validate_profile_capability,
)

CONFIG_ROOT = Path(__file__).resolve().parents[2] / "configs"

EXPECTED_QUALIFIED_LOGGER_SELECTORS: dict[str, tuple[str, ...]] = {
    "csv": ("lightning.pytorch.loggers.csv_logs.CSVLogger",),
    "heterogeneous_wandb": (
        "lightning.pytorch.loggers.wandb.WandbLogger",
    ),
    "many_loggers": (
        "lightning.pytorch.loggers.csv_logs.CSVLogger",
        "lightning.pytorch.loggers.wandb.WandbLogger",
    ),
    "wandb": ("lightning.pytorch.loggers.wandb.WandbLogger",),
}


def _compose(*overrides: str) -> DictConfig:
    GlobalHydra.instance().clear()
    register_all_resolvers()
    try:
        with hydra.initialize_config_dir(
            version_base="1.3",
            config_dir=str(CONFIG_ROOT),
            job_name="execution_profile",
        ):
            return hydra.compose(
                config_name="run.yaml",
                overrides=list(overrides),
            )
    finally:
        GlobalHydra.instance().clear()


def _configured_targets(value: object, path: str = "") -> dict[str, str]:
    if isinstance(value, DictConfig):
        value = OmegaConf.to_container(value, resolve=False)
    if isinstance(value, dict):
        targets: dict[str, str] = {}
        for key, child in value.items():
            if not path and key == "hydra":
                continue
            child_path = f"{path}.{key}" if path else str(key)
            if key == "_target_":
                assert isinstance(child, str)
                targets[child_path] = child
            else:
                targets.update(_configured_targets(child, child_path))
        return targets
    if isinstance(value, list):
        targets = {}
        for index, child in enumerate(value):
            child_path = f"{path}.{index}" if path else str(index)
            targets.update(_configured_targets(child, child_path))
        return targets
    return {}


class TestConfigInstantiators:
    """Test callback and logger construction."""

    def setup_method(self) -> None:
        """Set up minimal callback and logger configurations."""
        self.callback = OmegaConf.load("configs/callbacks/model_summary.yaml")
        self.logger = DictConfig(
            {
                "_target_": "lightning.pytorch.loggers.wandb.WandbLogger",
                "save_dir": "/",
                "offline": False,
                "id": None,
                "anonymous": None,
                "project": "None",
                "log_model": False,
                "prefix": "",
                "group": "",
                "tags": [],
                "job_type": "",
            }
        )

    def test_instantiate_callbacks(self) -> None:
        """Instantiate the configured callback collection."""
        assert isinstance(instantiate_callbacks(self.callback), list)

    def test_instantiate_loggers(self) -> None:
        """Instantiate the configured logger collection."""
        assert isinstance(instantiate_loggers(self.logger), list)


def test_qualified_default_accepts_exact_packaged_targets() -> None:
    """The packaged default is qualified and every executable is audited."""
    cfg = _compose()

    record = validate_execution_profile(cfg)

    assert isinstance(record, ExecutionProfileRecord)
    assert record.profile == "qualified"
    assert record.qualified is True
    assert record.custom_targets == ()
    assert dict(record.targets) == _configured_targets(cfg)
    assert dict(record.targets)["dataset.loader._target_"] == (
        "topobench.data.loaders.SyntheticGraphDatasetLoader"
    )
    assert dict(record.targets)["model.backbone._target_"] == (
        "torch_geometric.nn.models.GCN"
    )


def test_qualified_rejects_disabled_evaluation_artifacts() -> None:
    """Qualified execution requires publication-ready evaluation artifacts."""
    cfg = _compose()
    with open_dict(cfg.evaluation_artifacts):
        cfg.evaluation_artifacts.enabled = False

    with pytest.raises(ValueError, match=r"evaluation_artifacts\.enabled"):
        validate_execution_profile(cfg)


def test_experimental_accepts_disabled_evaluation_artifacts() -> None:
    """Experimental execution may explicitly disable artifact publication."""
    cfg = _compose()
    with open_dict(cfg):
        cfg.execution_profile = "experimental"
    with open_dict(cfg.evaluation_artifacts):
        cfg.evaluation_artifacts.enabled = False

    record = validate_execution_profile(cfg)

    assert record.profile == "experimental"
    assert record.qualified is False


@pytest.mark.parametrize("profile", ["unknown", 7, None])
def test_execution_profile_rejects_unknown_or_non_string_values(
    profile: object,
) -> None:
    """Only the two explicitly supported execution profiles are accepted."""
    cfg = _compose()
    with open_dict(cfg):
        cfg.execution_profile = profile

    with pytest.raises((TypeError, ValueError), match="execution_profile"):
        validate_execution_profile(cfg)


def test_qualified_rejects_mutated_target_before_instantiation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A changed executable path fails before Hydra can instantiate it."""
    cfg = _compose()
    with open_dict(cfg.model.backbone):
        cfg.model.backbone._target_ = "tests.custom.UnqualifiedBackbone"
    instantiate_calls: list[object] = []
    monkeypatch.setattr(
        hydra.utils,
        "instantiate",
        lambda value: instantiate_calls.append(value),
    )

    with pytest.raises(ValueError, match=r"model\.backbone\._target_"):
        validate_execution_profile(cfg)

    assert instantiate_calls == []


def test_experimental_records_every_target_and_only_changed_targets_as_custom(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Experimental auditing is deterministic without leaking other values."""
    cfg = _compose()
    secret = "unrelated-secret-must-not-appear"
    with open_dict(cfg):
        cfg.execution_profile = "experimental"
        cfg.extension = {
            "_target_": "tests.custom.Extension",
            "credential": secret,
        }
    with open_dict(cfg.model.backbone):
        cfg.model.backbone._target_ = "tests.custom.Backbone"
    with open_dict(cfg.callbacks.model_checkpoint):
        cfg.callbacks.model_checkpoint._target_ = "tests.custom.Checkpoint"

    record = validate_execution_profile(cfg)
    configured = _configured_targets(cfg)

    assert record.qualified is False
    assert record.targets == tuple(sorted(configured.items()))
    assert record.custom_targets == (
        ("callbacks.model_checkpoint._target_", "tests.custom.Checkpoint"),
        ("extension._target_", "tests.custom.Extension"),
        ("model.backbone._target_", "tests.custom.Backbone"),
    )
    assert secret not in caplog.text


def test_experimental_records_resolved_target_import_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Target provenance must match the import path Hydra will execute."""
    cfg = _compose()
    target = "torch_geometric.nn.models.GraphSAGE"
    interpolation = "${oc.env:TOPOBENCH_TEST_BACKBONE}"
    monkeypatch.setenv("TOPOBENCH_TEST_BACKBONE", target)
    with open_dict(cfg):
        cfg.execution_profile = "experimental"
    with open_dict(cfg.model.backbone):
        cfg.model.backbone._target_ = interpolation

    record = validate_execution_profile(cfg)

    assert dict(record.targets)["model.backbone._target_"] == target
    assert record.custom_targets == (("model.backbone._target_", target),)
    unresolved = OmegaConf.to_container(cfg, resolve=False)
    assert isinstance(unresolved, dict)
    assert unresolved["model"]["backbone"]["_target_"] == interpolation


@pytest.mark.parametrize("profile", ["qualified", "experimental"])
def test_execution_profile_rejects_non_string_targets(profile: str) -> None:
    """Hydra targets must be import-path strings in both profiles."""
    cfg = _compose()
    with open_dict(cfg):
        cfg.execution_profile = profile
    with open_dict(cfg.optimizer):
        cfg.optimizer._target_ = 3

    with pytest.raises(TypeError, match=r"optimizer\._target_"):
        validate_execution_profile(cfg)


def test_experimental_capability_validation_uses_copy_without_mutating_cfg() -> (
    None
):
    """Experimental custom executables retain packaged selector validation."""
    cfg = _compose()
    custom_target = "tests.custom.Backbone"
    with open_dict(cfg):
        cfg.execution_profile = "experimental"
    with open_dict(cfg.model.backbone):
        cfg.model.backbone._target_ = custom_target
    record = validate_execution_profile(cfg)

    validation = validate_profile_capability(cfg, profile_record=record)

    assert validation.dataset.selector == "graph/SyntheticGraph"
    assert validation.model.selector == "graph/gcn"
    assert cfg.model.backbone._target_ == custom_target


def test_packaged_logger_selectors_are_exactly_artifact_capable() -> None:
    """Every shipped logger selector is dependency-backed and publishable."""
    packaged = {
        path.stem for path in (CONFIG_ROOT / "logger").glob("*.yaml")
    }

    assert packaged == set(EXPECTED_QUALIFIED_LOGGER_SELECTORS)
    assert set(ARTIFACT_LOGGER_TARGETS.values()) == {
        "lightning.pytorch.loggers.csv_logs.CSVLogger",
        "lightning.pytorch.loggers.wandb.WandbLogger",
    }
    for target in ARTIFACT_LOGGER_TARGETS.values():
        assert hydra.utils.get_class(target) is not None


@pytest.mark.parametrize(
    ("selector", "expected_targets"),
    sorted(EXPECTED_QUALIFIED_LOGGER_SELECTORS.items()),
)
def test_every_packaged_logger_selector_passes_qualified_static_validation(
    selector: str,
    expected_targets: tuple[str, ...],
) -> None:
    """Every surviving selector passes the publication-aware preflight."""
    cfg = _compose(f"logger={selector}")

    record = validate_execution_profile(cfg)

    logger_targets = tuple(
        target
        for path, target in record.targets
        if path.startswith("logger.") and path.endswith("._target_")
    )
    assert logger_targets == expected_targets


@pytest.mark.parametrize("profile", ["qualified", "experimental"])
def test_enabled_artifacts_reject_unsupported_logger_during_static_validation(
    profile: str,
) -> None:
    """An unsupported logger cannot survive until checkpoint rerun."""
    cfg = _compose()
    with open_dict(cfg):
        cfg.execution_profile = profile
    with open_dict(cfg.logger.csv):
        cfg.logger.csv._target_ = (
            "lightning.pytorch.loggers.tensorboard.TensorBoardLogger"
        )

    with pytest.raises(
        ValueError,
        match="selected-checkpoint artifact publication",
    ):
        validate_execution_profile(cfg)


def test_disabled_experimental_artifacts_allow_custom_logger() -> None:
    """Experimental runs without publication retain custom logger freedom."""
    cfg = _compose()
    target = "lightning.pytorch.loggers.tensorboard.TensorBoardLogger"
    with open_dict(cfg):
        cfg.execution_profile = "experimental"
    with open_dict(cfg.evaluation_artifacts):
        cfg.evaluation_artifacts.enabled = False
    with open_dict(cfg.logger.csv):
        cfg.logger.csv._target_ = target

    record = validate_execution_profile(cfg)

    assert ("logger.csv._target_", target) in record.custom_targets
