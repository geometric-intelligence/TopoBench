"""Verify supported public imports without removed optional dependencies."""

from __future__ import annotations

import importlib
import importlib.abc
import sys

FORBIDDEN_DEPENDENCIES = (
    "gudhi",
    "hypernetx",
    "spharapy",
    "topomodelx",
    "toponetx",
    "trimesh",
)

SUPPORTED_PUBLIC_MODULES = (
    "topobench",
    "topobench.data",
    "topobench.data.loaders",
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


def main() -> None:
    """Import every supported public package under the dependency blocker."""
    sys.meta_path.insert(0, _ForbiddenDependencyFinder())
    for module_name in SUPPORTED_PUBLIC_MODULES:
        importlib.import_module(module_name)


if __name__ == "__main__":
    main()
