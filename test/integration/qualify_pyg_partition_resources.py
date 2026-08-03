"""Measured topology-only partition resource and external-fallback qualification."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np

from test.data.stores.test_topology_only_pyg_partitioner import (
    asymmetric_typed_source,
)
from topobench.data.stores.pyg_partitioner import TopologyOnlyPyGPartitioner
from topobench.data.stores.typed_graph_ingestion import (
    ArtifactValidationError,
    ParquetTypedGraphIngestor,
)
from topobench.data.stores.typed_partition_book import (
    PartitionQualificationLimits,
)


def _write_external(partitioner: TopologyOnlyPyGPartitioner) -> None:
    root = partitioner.ingestor.source.spec.source_root
    path = root / "external/assignment.npy"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(
        path,
        np.array([0, 0, 1, 1, 0, 1, 0, 1, 1], dtype=np.int64),
        allow_pickle=False,
    )
    manifest = {
        "format_version": "typed-external-partition-map-v1",
        "topology_fingerprint": partitioner.topology_fingerprint,
        "num_partitions": 2,
        "node_type_offsets": {"n0000": [0, 4], "n0001": [4, 9]},
        "assignment": {
            "relative_path": "external/assignment.npy",
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "dtype": "int64",
            "shape": [9],
        },
    }
    (root / "external/manifest.json").write_text(
        json.dumps(manifest, sort_keys=True), encoding="utf-8"
    )


def _child(root: Path) -> None:
    source = asymmetric_typed_source(root / "admitted-source")
    ingestor = ParquetTypedGraphIngestor(source, root / "admitted-store")
    partitioner = TopologyOnlyPyGPartitioner(
        ingestor, ingestor.build_relations()
    )
    data = partitioner.materialize_topology()
    assert all(
        set(data[node_type].keys()) == {"num_nodes"}
        for node_type in data.node_types
    )
    assert all(
        set(data[edge_type].keys()) == {"edge_index"}
        for edge_type in data.edge_types
    )
    estimate = partitioner.estimate_resources()
    book = partitioner.generate(PartitionQualificationLimits())
    measured = dict(book.measured_resources)
    assert measured["measurement_scope"] == "isolated-worker"
    assert measured["peak_rss_bytes"] <= estimate["peak_memory_bytes"]
    assert measured["temporary_disk_bytes"] <= estimate["temporary_disk_bytes"]
    assert (
        partitioner.last_work_root is not None
        and not partitioner.last_work_root.exists()
    )
    (root / "child-evidence.json").write_text(
        json.dumps(
            {
                "canonical_edges": estimate["canonical_edge_count"],
                "estimated_peak_memory_bytes": estimate["peak_memory_bytes"],
                "estimated_temporary_disk_bytes": estimate[
                    "temporary_disk_bytes"
                ],
                "measured_peak_rss_bytes": measured["peak_rss_bytes"],
                "measured_temporary_disk_bytes": measured[
                    "temporary_disk_bytes"
                ],
                "static_attributes_in_pyg": False,
                "temporary_output_cleaned": True,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", type=Path)
    args = parser.parse_args()
    if args.child is not None:
        _child(args.child)
        return
    with tempfile.TemporaryDirectory(
        prefix="topobench-pyg-qualification-"
    ) as raw:
        root = Path(raw)
        scalable = TopologyOnlyPyGPartitioner.estimate_topology_resources(
            node_count=2_000_000_000,
            canonical_edge_count=8_000_000_000,
            relation_count=4,
            node_type_count=3,
        )
        assert scalable["peak_memory_bytes"] > 256 * 1024**3
        source = asymmetric_typed_source(
            root / "fallback-source",
            memory_limit_bytes=1,
            external_partition_map="external/manifest.json",
        )
        ingestor = ParquetTypedGraphIngestor(source, root / "fallback-store")
        partitioner = TopologyOnlyPyGPartitioner(
            ingestor, ingestor.build_relations()
        )
        try:
            partitioner.preflight()
        except ArtifactValidationError as error:
            assert "PARTITION-MEMORY-001" in str(error)
        else:
            raise AssertionError("over-budget topology was materialized")
        assert partitioner.materialization_count == 0
        _write_external(partitioner)
        assert (
            ingestor.build_partitions(
                limits=PartitionQualificationLimits()
            ).book.backend
            == "external"
        )
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--child",
                str(root),
            ],
            check=True,
            cwd=Path(__file__).resolve().parents[2],
        )
        evidence = json.loads(
            (root / "child-evidence.json").read_text(encoding="utf-8")
        )
        evidence.update(
            {
                "external_fallback": True,
                "overbudget_preflight_blocked": True,
                "scalable_estimated_peak_memory_bytes": scalable[
                    "peak_memory_bytes"
                ],
            }
        )
        print(json.dumps(evidence, sort_keys=True))


if __name__ == "__main__":
    main()
