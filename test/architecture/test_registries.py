"""Architecture tests for deterministic public registries."""

import importlib
import subprocess
import sys
from pathlib import Path

import yaml

from topobench.data import datasets, loaders
from topobench.data.loaders import graph as graph_loaders
from topobench.data.loaders import heterogeneous as heterogeneous_loaders
from topobench.data.loaders import hypergraph as hypergraph_loaders
from topobench.nn import backbones, wrappers
from topobench.nn.backbones import graph as graph_backbones
from topobench.nn.backbones import heterogeneous as heterogeneous_backbones
from topobench.nn.backbones import hypergraph as hypergraph_backbones
from topobench.nn.wrappers import graph as graph_wrappers
from topobench.nn.wrappers import heterogeneous as heterogeneous_wrappers
from topobench.nn.wrappers import hypergraph as hypergraph_wrappers

EXPECTED_DATASETS = {
    "CitationHypergraphDataset",
    "HypergraphDataset",
    "SyntheticGraphDataset",
    "SyntheticHeterogeneousDataset",
    "SyntheticHypergraphDataset",
    "USCountyDemosDataset",
}
EXPECTED_GRAPH_LOADERS = {
    "ADMEDatasetLoader",
    "GraphUniverseDatasetLoader",
    "HeterophilousGraphDatasetLoader",
    "ManualGraphDatasetLoader",
    "MoleculeDatasetLoader",
    "OGBGDatasetLoader",
    "PlanetoidDatasetLoader",
    "SyntheticGraphDatasetLoader",
    "TUDatasetLoader",
    "USCountyDemosDatasetLoader",
}
ORPHAN_GRAPH_LOADERS = {
    "ManualGraphDatasetLoader",
    "USCountyDemosDatasetLoader",
}
EXPECTED_HETEROGENEOUS_LOADERS = {
    "DBLPDatasetLoader",
    "OGBMAGDatasetLoader",
    "SyntheticHeterogeneousDatasetLoader",
}
EXPECTED_HYPERGRAPH_LOADERS = {
    "CitationHypergraphDatasetLoader",
    "HypergraphDatasetLoader",
    "SyntheticHypergraphDatasetLoader",
}


def _assert_explicit_registry(
    registry: dict[str, type], expected: set[str]
) -> None:
    assert registry.__class__ is dict
    assert tuple(registry) == tuple(sorted(registry))
    assert set(registry) == expected
    assert all(
        name == registered_class.__name__
        for name, registered_class in registry.items()
    )


def test_dataset_registry_has_only_surviving_dataset_classes() -> None:
    _assert_explicit_registry(datasets.MANUAL_DATASETS, EXPECTED_DATASETS)


def test_loader_registries_have_only_surviving_loader_classes() -> None:
    _assert_explicit_registry(graph_loaders.GRAPH_LOADERS, EXPECTED_GRAPH_LOADERS)
    _assert_explicit_registry(
        heterogeneous_loaders.HETEROGENEOUS_LOADERS,
        EXPECTED_HETEROGENEOUS_LOADERS,
    )
    _assert_explicit_registry(
        hypergraph_loaders.HYPERGRAPH_LOADERS,
        EXPECTED_HYPERGRAPH_LOADERS,
    )
    _assert_explicit_registry(
        loaders.LOADER_CLASSES,
        EXPECTED_GRAPH_LOADERS
        | EXPECTED_HETEROGENEOUS_LOADERS
        | EXPECTED_HYPERGRAPH_LOADERS,
    )


def test_surviving_yaml_loader_targets_resolve_through_explicit_registries() -> None:
    config_root = Path(__file__).parents[2] / "configs" / "dataset"
    targets = {
        document["loader"]["_target_"]
        for domain in ("graph", "heterogeneous", "hypergraph")
        for config_path in (config_root / domain).glob("*.yaml")
        if (document := yaml.safe_load(config_path.read_text(encoding="utf-8")))
    }

    assert {target.rsplit(".", 1)[1] for target in targets} == (
        set(loaders.LOADER_CLASSES) - ORPHAN_GRAPH_LOADERS
    )
    for target in targets:
        module_name, class_name = target.rsplit(".", 1)
        target_module = importlib.import_module(module_name)
        assert getattr(target_module, class_name) is loaders.LOADER_CLASSES[class_name]


def test_backbone_registry_has_only_surviving_local_models() -> None:
    _assert_explicit_registry(
        graph_backbones.BACKBONE_CLASSES,
        {"GCNDGM", "GPSEncoder", "GraphMLP", "NSDEncoder"},
    )
    _assert_explicit_registry(
        heterogeneous_backbones.BACKBONE_CLASSES,
        {"HGTBackbone", "HeteroSAGEBackbone"},
    )
    _assert_explicit_registry(
        hypergraph_backbones.BACKBONE_CLASSES,
        {"EDGNN", "HypergraphConvBackbone"},
    )
    _assert_explicit_registry(
        backbones.MODEL_CLASSES,
        {
            "GCNDGM",
            "EDGNN",
            "HypergraphConvBackbone",
            "GPSEncoder",
            "GraphMLP",
            "HGTBackbone",
            "HeteroSAGEBackbone",
            "NSDEncoder",
        },
    )


def test_wrapper_registry_has_only_surviving_adapters() -> None:
    _assert_explicit_registry(
        graph_wrappers.WRAPPER_CLASSES,
        {"GNNWrapper", "GraphMLPWrapper"},
    )
    _assert_explicit_registry(
        heterogeneous_wrappers.WRAPPER_CLASSES,
        {"HeterogeneousWrapper"},
    )
    _assert_explicit_registry(
        hypergraph_wrappers.WRAPPER_CLASSES,
        {"HypergraphWrapper"},
    )
    _assert_explicit_registry(
        wrappers.WRAPPER_CLASSES,
        {
            "GNNWrapper",
            "GraphMLPWrapper",
            "HeterogeneousWrapper",
            "HypergraphWrapper",
        },
    )


def test_registry_imports_do_not_use_dynamic_discovery() -> None:
    script = r"""
import importlib
import importlib.util
from pathlib import Path

package_spec = importlib.util.find_spec("topobench")
assert package_spec is not None and package_spec.origin is not None
package_root = Path(package_spec.origin).parent
registry_directories = {
    package_root / "data" / "datasets",
    package_root / "data" / "loaders",
    package_root / "data" / "loaders" / "graph",
    package_root / "data" / "loaders" / "heterogeneous",
    package_root / "data" / "loaders" / "hypergraph",
    package_root / "nn" / "backbones",
    package_root / "nn" / "backbones" / "graph",
    package_root / "nn" / "backbones" / "heterogeneous",
    package_root / "nn" / "backbones" / "hypergraph",
    package_root / "nn" / "wrappers",
    package_root / "nn" / "wrappers" / "graph",
    package_root / "nn" / "wrappers" / "heterogeneous",
    package_root / "nn" / "wrappers" / "hypergraph",
}
original_import_module = importlib.import_module
original_glob = Path.glob
original_iterdir = Path.iterdir
original_spec_from_file_location = importlib.util.spec_from_file_location


def guarded_import_module(name, package=None):
    if name.startswith("topobench."):
        raise AssertionError(f"dynamic importlib discovery attempted for {name}")
    return original_import_module(name, package)


def guarded_iterdir(path):
    if path in registry_directories:
        raise AssertionError(f"dynamic Path.iterdir discovery attempted in {path}")
    return original_iterdir(path)


def guarded_glob(self, pattern):
    if self in registry_directories:
        raise AssertionError(f"dynamic Path.glob discovery attempted in {self}")
    return original_glob(self, pattern)


def guarded_spec_from_file_location(name, location, *args, **kwargs):
    if Path(location).parent in registry_directories:
        raise AssertionError(
            f"dynamic spec_from_file_location discovery attempted for {location}"
        )
    return original_spec_from_file_location(name, location, *args, **kwargs)


importlib.import_module = guarded_import_module
Path.iterdir = guarded_iterdir
Path.glob = guarded_glob
importlib.util.spec_from_file_location = guarded_spec_from_file_location

import topobench.data.loaders
import topobench.nn.backbones
import topobench.nn.wrappers
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_registry_modules_have_narrow_public_exports() -> None:
    registry_modules = (
        (graph_loaders, "GRAPH_LOADERS"),
        (heterogeneous_loaders, "HETEROGENEOUS_LOADERS"),
        (hypergraph_loaders, "HYPERGRAPH_LOADERS"),
        (graph_backbones, "BACKBONE_CLASSES"),
        (heterogeneous_backbones, "BACKBONE_CLASSES"),
        (hypergraph_backbones, "BACKBONE_CLASSES"),
        (backbones, "MODEL_CLASSES"),
        (graph_wrappers, "WRAPPER_CLASSES"),
        (heterogeneous_wrappers, "WRAPPER_CLASSES"),
        (hypergraph_wrappers, "WRAPPER_CLASSES"),
        (wrappers, "WRAPPER_CLASSES"),
    )

    for registry_module, registry_name in registry_modules:
        registry = getattr(registry_module, registry_name)
        assert registry_module.__all__ == [*registry, registry_name]

    assert loaders.__all__ == [
        "AbstractLoader",
        *loaders.LOADER_CLASSES,
        "LOADER_CLASSES",
    ]
    assert datasets.__all__ == [
        "PYG_DATASETS",
        "PLANETOID_DATASETS",
        "TU_DATASETS",
        "FIXED_SPLITS_DATASETS",
        "HETEROPHILIC_DATASETS",
        *datasets.MANUAL_DATASETS,
        "MANUAL_DATASETS",
        "make_synthetic_heterogeneous_data",
        "make_synthetic_hypergraph_data",
    ]
