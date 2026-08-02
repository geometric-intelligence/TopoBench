"""Qualify external string-ID mapping under a strict subprocess RSS ceiling."""

from __future__ import annotations

import json
import os
import resource
import subprocess
import sys
import tempfile
from pathlib import Path

ROWS = int(os.environ.get("TOPOBENCH_ID_RSS_ROWS", "2000000"))
RSS_DELTA_LIMIT_BYTES = int(
    os.environ.get("TOPOBENCH_ID_RSS_DELTA_LIMIT_BYTES", str(192 * 1024**2))
)
BATCH_ROWS = 4096
ID_SUFFIX = "-" + "bounded-external-id-" * 8


def _rss_bytes() -> int:
    observed = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return observed if sys.platform == "darwin" else observed * 1024


def _write_sources(root: Path) -> tuple[str, ...]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    node_path = root / "nodes.parquet"
    schema = pa.schema(
        [
            pa.field("node_id", pa.string(), nullable=False),
            pa.field("feature", pa.float32(), nullable=False),
            pa.field("label", pa.int64(), nullable=False),
        ]
    )
    with pq.ParquetWriter(node_path, schema, compression="snappy") as writer:
        for start in range(0, ROWS, BATCH_ROWS):
            size = min(BATCH_ROWS, ROWS - start)
            ids = [f"{ROWS - start - offset:016x}{ID_SUFFIX}" for offset in range(size)]
            writer.write_batch(
                pa.record_batch(
                    [
                        pa.array(ids, type=pa.string()),
                        pa.array(range(size), type=pa.float32()),
                        pa.array(range(size), type=pa.int64()),
                    ],
                    schema=schema,
                )
            )

    pq.write_table(
        pa.table({"src": [f"{1:016x}{ID_SUFFIX}"], "dst": [f"{2:016x}{ID_SUFFIX}"]}),
        root / "edges.parquet",
    )
    split_paths = []
    for phase, value in (("train", 1), ("val", 2), ("test", 3)):
        path = root / f"{phase}.parquet"
        pq.write_table(pa.table({"node_id": [f"{value:016x}{ID_SUFFIX}"]}), path)
        split_paths.append(path.name)
    return tuple(split_paths)


def _worker(root: Path) -> None:
    from topobench.data.loaders.parquet import (
        IngestionLimits,
        NodeTypeSpec,
        ParquetTypedGraphSource,
        ParquetTypedGraphSpec,
        PartitionSpec,
        RelationSpec,
        SplitRegistrySpec,
        SplitSetSpec,
        SupervisionSpec,
    )
    from topobench.data.stores.typed_graph_ingestion import ParquetTypedGraphIngestor

    split = SplitSetSpec(
        tag="rss",
        train="train.parquet",
        val="val.parquet",
        test="test.parquet",
        coverage="partial",
    )
    source = ParquetTypedGraphSource(
        ParquetTypedGraphSpec(
            source_root=root,
            output_kind="homogeneous",
            node_types=(
                NodeTypeSpec(
                    name="node",
                    paths=("nodes.parquet",),
                    id_column="node_id",
                    id_dtype="string",
                    feature_columns=("feature",),
                    feature_dtype="float32",
                    feature_width=1,
                ),
            ),
            relations=(
                RelationSpec(
                    relation=("node", "links", "node"),
                    paths=("edges.parquet",),
                    source_column="src",
                    destination_column="dst",
                ),
            ),
            supervision=SupervisionSpec(
                target_node_type="node",
                label_column="label",
                label_dtype="int64",
                split_registry=SplitRegistrySpec(active_tag="rss", sets=(split,)),
            ),
            partition=PartitionSpec(strategy="cluster"),
            ingestion=IngestionLimits(
                record_batch_rows=BATCH_ROWS,
                memory_limit_bytes=64 * 1024**2,
                temp_directory="duckdb-tmp",
            ),
        )
    )
    baseline_rss_bytes = _rss_bytes()
    result = ParquetTypedGraphIngestor(source, root / "stores", threads=1).build()
    index = result.indexes["node"]
    assert len(index) == ROWS
    assert index.lookup(f"{1:016x}{ID_SUFFIX}") == 0
    assert index.external_id(ROWS - 1) == f"{ROWS:016x}{ID_SUFFIX}"
    print(
        json.dumps(
            {
                "rows": ROWS,
                "baseline_rss_bytes": baseline_rss_bytes,
                "peak_rss_bytes": _rss_bytes(),
                "rss_delta_limit_bytes": RSS_DELTA_LIMIT_BYTES,
                "duckdb_memory_limit_bytes": 64 * 1024**2,
                "batch_rows": BATCH_ROWS,
                "full_arrow_id_payload_lower_bound": ROWS * len(ID_SUFFIX.encode("utf-8")),
                "lookup_backend": "DuckDB sorted disk table without a RAM-wide ART index",
            },
            sort_keys=True,
        )
    )


def main() -> None:
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        _worker(Path(sys.argv[2]))
        return

    with tempfile.TemporaryDirectory(prefix="topobench-id-rss-") as directory:
        root = Path(directory)
        _write_sources(root)
        completed = subprocess.run(
            [sys.executable, __file__, "--worker", str(root)],
            check=True,
            capture_output=True,
            text=True,
        )
        evidence = json.loads(completed.stdout.strip().splitlines()[-1])
        evidence["rss_delta_bytes"] = (
            evidence["peak_rss_bytes"] - evidence["baseline_rss_bytes"]
        )
        if evidence["rss_delta_bytes"] >= RSS_DELTA_LIMIT_BYTES:
            raise SystemExit(
                "RSS qualification failed: ingestion delta "
                f"{evidence['rss_delta_bytes']} >= {RSS_DELTA_LIMIT_BYTES}"
            )
        if evidence["full_arrow_id_payload_lower_bound"] <= RSS_DELTA_LIMIT_BYTES:
            raise SystemExit(
                "RSS fixture is too small to reject one RAM-wide Arrow ID column"
            )
        print(json.dumps(evidence, sort_keys=True))


if __name__ == "__main__":
    main()
