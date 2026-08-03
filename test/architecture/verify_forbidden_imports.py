"""Verify supported public imports without removed optional dependencies."""

from __future__ import annotations

import importlib
import importlib.abc
import sys
from pathlib import Path

FORBIDDEN_DEPENDENCIES = (
    "gudhi",
    "hypernetx",
    "spharapy",
    "topomodelx",
    "toponetx",
    "trimesh",
)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PROBES = (
    (),
    ("experiment=heterogeneous_synthetic_hgt_full",),
    ("experiment=hypergraph_synthetic_edgnn",),
)


SUPPORTED_PUBLIC_MODULES = (
    "topobench",
    "topobench.data",
    "topobench.data.loaders",
    "topobench.evaluator",
    "topobench.transforms",
    "topobench.nn.backbones",
    "topobench.nn.wrappers",
    "topobench.run",
)


class _ForbiddenDependencyFinder(importlib.abc.MetaPathFinder):
    """Fail if a supported import reaches a removed dependency root."""

    def find_spec(self, fullname, path=None, target=None):
        """Reject removed roots and leave all other imports unchanged."""
        if fullname.partition(".")[0].casefold() in FORBIDDEN_DEPENDENCIES:
            raise ModuleNotFoundError(
                f"forbidden dependency import attempted: {fullname}"
            )
        return


def _compose_supported_configs() -> None:
    """Compose and validate one network-free configuration per domain."""
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    from topobench.run import validate_domain_composition
    from topobench.utils.config_resolvers import register_all_resolvers

    register_all_resolvers()
    GlobalHydra.instance().clear()
    with initialize_config_dir(
        version_base="1.3",
        config_dir=str(PROJECT_ROOT / "configs"),
    ):
        for overrides in CONFIG_PROBES:
            cfg = compose(config_name="run.yaml", overrides=list(overrides))
            validate_domain_composition(cfg)


def main() -> None:
    """Import public packages and compose supported domain configurations."""
    sys.meta_path.insert(0, _ForbiddenDependencyFinder())
    for module_name in SUPPORTED_PUBLIC_MODULES:
        importlib.import_module(module_name)
    _compose_supported_configs()
    print("clean import verified")


if __name__ == "__main__":
    main()
