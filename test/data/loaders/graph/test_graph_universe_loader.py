"""Focused cache and RNG contracts for the production GraphUniverse loader."""

from __future__ import annotations

import json
import random
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch
from graph_universe import GraphUniverseDataset
from omegaconf import DictConfig, OmegaConf

from topobench.data.loaders.graph.graph_universe_loader import (
    GraphUniverseDatasetLoader,
)
from topobench.data.preprocessor import PreProcessor


def _parameters(tmp_path) -> DictConfig:
    """Return a tiny but real GraphUniverse generation configuration."""
    return DictConfig(
        {
            "data_dir": str(tmp_path / "graph-universe"),
            "data_domain": "graph",
            "data_type": "GraphUniverse",
            "data_name": "GraphUniverse",
            "generation_parameters": {
                "task": "triangle_counting",
                "universe_parameters": {
                    "K": 4,
                    "feature_dim": 3,
                    "center_variance": 0.2,
                    "cluster_variance": 0.4,
                    "edge_propensity_variance": 0.5,
                    "seed": 17,
                },
                "family_parameters": {
                    "n_graphs": 2,
                    "n_nodes_range": [8, 9],
                    "n_communities_range": [2, 3],
                    "homophily_range": [0.4, 0.6],
                    "avg_degree_range": [1.0, 1.5],
                    "degree_separation_range": [0.5, 0.8],
                    "power_law_exponent_range": [2.0, 2.5],
                    "seed": 17,
                },
            },
        }
    )


def _rng_states() -> tuple[object, tuple, torch.Tensor]:
    """Snapshot Python, NumPy, and Torch caller-global RNG streams."""
    return random.getstate(), np.random.get_state(), torch.random.get_rng_state()


def _assert_rng_states_equal(
    before: tuple[object, tuple, torch.Tensor],
    after: tuple[object, tuple, torch.Tensor],
) -> None:
    """Compare all caller-global RNG streams exactly."""
    assert before[0] == after[0]
    assert before[1][0] == after[1][0]
    assert np.array_equal(before[1][1], after[1][1])
    assert before[1][2:] == after[1][2:]
    assert torch.equal(before[2], after[2])


def _preprocess(tmp_path) -> PreProcessor:
    """Exercise production generation and processed caching end to end."""
    loader = GraphUniverseDatasetLoader(_parameters(tmp_path))
    dataset, data_dir = loader.load()
    transforms = DictConfig(
        {"identity": {"transform_name": None, "value": 1}}
    )
    return PreProcessor(dataset, data_dir, transforms)


def test_graph_universe_miss_and_hit_preserve_rng_and_cache_provenance(
    tmp_path,
) -> None:
    """Real generation is isolated and the complete source record is reused."""
    processed = []
    for _ in ("miss", "hit"):
        random.seed(701)
        np.random.seed(702)
        torch.manual_seed(703)
        before = _rng_states()

        processed.append(_preprocess(tmp_path))

        _assert_rng_states_equal(before, _rng_states())

    miss, hit = processed
    assert miss.cache_identity == hit.cache_identity
    assert len(miss) == len(hit) == 2
    for left, right in zip(miss, hit, strict=True):
        assert left.to_dict().keys() == right.to_dict().keys()
        for key in left.to_dict():
            assert torch.equal(left[key], right[key])

    expected_parameters = OmegaConf.to_container(
        _parameters(tmp_path),
        resolve=True,
    )
    assert miss.cache_record["dataset_selector"] == {
        "data_domain": "graph",
        "data_type": "GraphUniverse",
        "data_name": "GraphUniverse",
    }
    assert miss.cache_record["loader"] == {
        "target": (
            "topobench.data.loaders.graph.graph_universe_loader."
            "GraphUniverseDatasetLoader"
        ),
        "parameters": expected_parameters,
    }
    assert miss.cache_record["feature_policy"] == "continuous"
    assert miss.cache_record["versions"] == {
        "representation": "pyg-data-v1",
        "parser": "graph-universe-v1",
    }
    assert list(
        (tmp_path / "graph-universe").glob(".graph-universe-*")
    ) == []


@pytest.mark.parametrize("failure", ["error", "timeout"])
def test_partial_child_artifact_cannot_poison_canonical_root(
    tmp_path,
    monkeypatch,
    failure,
) -> None:
    """Failed isolated generation is erased before a genuine retry."""
    loader = GraphUniverseDatasetLoader(_parameters(tmp_path))

    def fail_after_partial_artifact(command, **kwargs):
        child_root = Path(command[3])
        parameters = json.loads(
            Path(command[4]).read_text(encoding="utf-8")
        )
        dataset_name = GraphUniverseDataset.get_dataset_dir(None, parameters)
        partial_dir = child_root / dataset_name
        partial_dir.mkdir(parents=True, exist_ok=True)
        (partial_dir / "data.pt").write_bytes(b"partial")
        (partial_dir / "metadata.json").write_text(
            '{"partial": true}',
            encoding="utf-8",
        )
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        raise subprocess.CalledProcessError(23, command)

    with monkeypatch.context() as child:
        child.setattr(subprocess, "run", fail_after_partial_artifact)
        with pytest.raises(RuntimeError, match="GraphUniverse generation"):
            loader.load_dataset()

    canonical_root = Path(_parameters(tmp_path).data_dir)
    assert list(canonical_root.rglob("data.pt")) == []
    assert list(canonical_root.rglob("metadata.json")) == []
    assert list(canonical_root.glob(".graph-universe-*")) == []

    temporary_roots = []
    real_run = subprocess.run

    def capture_temporary_root(command, **kwargs):
        temporary_roots.append(Path(command[3]).parent)
        return real_run(command, **kwargs)

    with monkeypatch.context() as child:
        child.setattr(subprocess, "run", capture_temporary_root)
        regenerated = loader.load_dataset()

    assert len(regenerated) == 2
    assert Path(regenerated.raw_dir) == canonical_root / "GraphUniverse"
    assert regenerated.root == str(canonical_root)
    assert regenerated.name == "GraphUniverse"
    assert Path(regenerated.processed_dir) == canonical_root / "GraphUniverse"
    assert len(temporary_roots) == 1
    assert not temporary_roots[0].exists()
    assert all(
        str(temporary_roots[0]) not in repr(value)
        for value in vars(regenerated).values()
    )
    assert list(canonical_root.rglob("data.pt")) == []
