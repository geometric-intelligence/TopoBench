"""Architecture tests for the supported native data surface."""

import ast
from pathlib import Path

import topobench
import topobench.data.utils as data_utils
import topobench.dataloader as dataloader
from topobench.data.utils.hypergraph_io import (
    load_hypergraph_content_dataset,
    load_hypergraph_npz_dataset,
    validate_hypergraph_npz_assets,
)
from topobench.dataloader import GraphDataModule, HeterogeneousNodeDataModule


PACKAGE_ROOT = Path(topobench.__file__).parent
LOADER_ROOT = PACKAGE_ROOT / "data" / "loaders"
SUPPORTED_LOADER_PACKAGES = {"graph", "heterogeneous", "hypergraph"}
FORBIDDEN_MODULES = {
    "topobench.data.utils",
    "topobench.data.utils.io_utils",
    "topobench.data.utils.utils",
    "topobench.dataloader.dataload_dataset",
    "topobench.dataloader.dataloader",
    "topobench.dataloader.utils",
}
FORBIDDEN_SYMBOLS = {
    "DataloadDataset",
    "DomainData",
    "TBDataloader",
    "collate_fn",
}
NARROW_SYMBOL_OWNERS = {
    "apply_transductive_split": {"topobench.data.splits"},
    "ArchiveLimits": {"topobench.data.utils.downloads"},
    "RemoteArchive": {"topobench.data.utils.downloads"},
    "acquire_verified_archive": {"topobench.data.utils.downloads"},
    "ensure_serializable": {"topobench.data.utils.common"},
    "incidence_pairs": {"topobench.data.utils.hypergraph_io"},
    "inductive_split_views": {"topobench.data.splits"},
    "load_hypergraph_content_dataset": {
        "topobench.data.utils.hypergraph_io"
    },
    "load_hypergraph_npz_dataset": {
        "topobench.data.utils.hypergraph_io"
    },
    "validate_hypergraph_npz_assets": {
        "topobench.data.utils.hypergraph_io"
    },
    "make_hash": {"topobench.data.utils.common"},
    "validate_transductive_masks": {"topobench.data.splits"},
}


def _production_trees():
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        yield path, ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_loader_packages_match_the_supported_domains() -> None:
    """No unsupported loader package remains importable."""
    packages = {
        path.name
        for path in LOADER_ROOT.iterdir()
        if path.is_dir() and (path / "__init__.py").is_file()
    }

    assert packages == SUPPORTED_LOADER_PACKAGES


def test_legacy_batching_symbols_are_not_imported_or_exported() -> None:
    """Production imports and public exports cannot revive legacy batching."""
    violations = []
    for path, tree in _production_trees():
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                imported = {alias.name for alias in node.names}
                if imported & FORBIDDEN_SYMBOLS:
                    violations.append((path, node.lineno, imported & FORBIDDEN_SYMBOLS))
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                if not any(
                    isinstance(target, ast.Name) and target.id == "__all__"
                    for target in targets
                ):
                    continue
                exported = {
                    child.value
                    for child in ast.walk(node.value)
                    if isinstance(child, ast.Constant)
                    and isinstance(child.value, str)
                }
                if exported & FORBIDDEN_SYMBOLS:
                    violations.append((path, node.lineno, exported & FORBIDDEN_SYMBOLS))

    assert violations == []


def test_production_imports_do_not_reference_removed_data_modules() -> None:
    """Every production import resolves without a deleted compatibility module."""
    violations = []
    for path, tree in _production_trees():
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level == 0:
                if node.module in FORBIDDEN_MODULES:
                    violations.append((path, node.lineno, node.module))
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in FORBIDDEN_MODULES:
                        violations.append((path, node.lineno, alias.name))

    assert violations == []


def test_dependency_light_data_symbols_use_their_narrow_owners() -> None:
    """Shared data helpers are imported from one dependency-light owner."""
    violations = []
    for path, tree in _production_trees():
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or node.level != 0:
                continue
            for alias in node.names:
                owners = NARROW_SYMBOL_OWNERS.get(alias.name)
                if owners is not None and node.module not in owners:
                    violations.append((path, node.lineno, alias.name, node.module))

    assert violations == []


def test_native_datamodules_and_safe_hypergraph_parsers_remain_public() -> None:
    """Pruning retains native batching and non-executable parser formats."""
    assert GraphDataModule.__name__ == "GraphDataModule"
    assert HeterogeneousNodeDataModule.__name__ == "HeterogeneousNodeDataModule"
    assert dataloader.__all__ == [
        "GraphDataModule",
        "HeterogeneousNodeDataModule",
    ]
    assert data_utils.__all__ == [
        "ArchiveLimits",
        "RemoteArchive",
        "acquire_verified_archive",
        "ContentRoleSpec",
        "SAFE_HYPERGRAPH_CONVERTER_VERSION",
        "SAFE_HYPERGRAPH_FORMAT",
        "SAFE_HYPERGRAPH_FORMAT_VERSION",
        "ensure_serializable",
        "incidence_pairs",
        "load_coauthorship_hypergraph_splits",
        "load_hypergraph_content_dataset",
        "load_hypergraph_npz_dataset",
        "validate_hypergraph_npz_assets",
        "load_inductive_splits",
        "load_transductive_splits",
        "make_hash",
    ]
    assert callable(load_hypergraph_content_dataset)
    assert callable(load_hypergraph_npz_dataset)
    assert callable(validate_hypergraph_npz_assets)
