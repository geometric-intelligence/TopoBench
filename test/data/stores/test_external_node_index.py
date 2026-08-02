"""Behavioral tests for deterministic disk-backed external-node indexes."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from test.data.stores.test_typed_graph_inventory import _make_source, _sha256
import topobench.data.stores.typed_graph_ingestion as ingestion_module
from topobench.data.stores.external_node_index import ExternalNodeIndex
from topobench.data.stores.typed_graph_ingestion import (
    ArtifactValidationError,
    ParquetTypedGraphIngestor,
    SourceMutationError,
)


@pytest.mark.parametrize(
    ("dtype", "declared", "fragments", "expected"),
    [
        (
            pa.int64(),
            "int64",
            (("users/b.parquet", [2**63 - 1, -1]), ("users/a.parquet", [0, -(2**63)])),
            [-(2**63), -1, 0, 2**63 - 1],
        ),
        (
            pa.uint64(),
            "uint64",
            (("users/b.parquet", [2**64 - 1, 3]), ("users/a.parquet", [0, 2**63])),
            [0, 3, 2**63, 2**64 - 1],
        ),
        (
            pa.string(),
            "string",
            (("users/b.parquet", ["é", "A", "😀"]), ("users/a.parquet", ["é", "a", "Ω"])),
            sorted(["é", "A", "😀", "é", "a", "Ω"], key=lambda value: value.encode("utf-8")),
        ),
    ],
)
def test_exact_domains_have_deterministic_ordinals_and_roundtrip(
    tmp_path: Path,
    dtype: pa.DataType,
    declared: str,
    fragments: tuple[tuple[str, list[object]], ...],
    expected: list[object],
) -> None:
    """Each exact domain receives contiguous ordinals in its canonical order."""
    source = _make_source(
        tmp_path / "source",
        user_fragments=tuple((path, dtype, values) for path, values in fragments),
    )
    object.__setattr__(source.spec.node_types[1], "id_dtype", declared)
    result = ParquetTypedGraphIngestor(source, tmp_path / "stores").build()
    index = result.indexes["user"]

    assert isinstance(index, ExternalNodeIndex)
    assert index.id_dtype == declared
    assert len(index) == len(expected)
    assert [index.external_id(local) for local in range(len(index))] == expected
    assert [index.lookup(external) for external in expected] == list(range(len(expected)))
    with index:
        assert index.lookup(expected[-1]) == len(expected) - 1
    with pytest.raises(KeyError):
        index.lookup("missing" if declared == "string" else 17)
    with pytest.raises(IndexError):
        index.external_id(len(expected))

    table = pq.read_table(index.node_ids_path)
    assert table.schema.field("local_ordinal").type == pa.int64()
    assert table.schema.field("external_id").type == dtype
    assert table.column("local_ordinal").to_pylist() == list(range(len(expected)))
    assert table.column("external_id").to_pylist() == expected
    assert index.lookup_path.suffix == ".duckdb"
    assert index.lookup_path.is_file()
    assert not hasattr(index, "_forward")
    assert not hasattr(index, "_reverse")


def test_fragment_layout_does_not_change_type_local_ordinals(tmp_path: Path) -> None:
    """External ordering, not fragment or row order, defines every ordinal."""
    first = _make_source(
        tmp_path / "first",
        user_fragments=(
            ("users/z.parquet", pa.int64(), [20, -5]),
            ("users/a.parquet", pa.int64(), [7]),
        ),
    )
    second = _make_source(
        tmp_path / "second",
        user_fragments=(
            ("users/one.parquet", pa.int64(), [7]),
            ("users/two.parquet", pa.int64(), [-5, 20]),
        ),
    )

    first_index = ParquetTypedGraphIngestor(first, tmp_path / "stores-1").build().indexes[
        "user"
    ]
    second_index = ParquetTypedGraphIngestor(
        second, tmp_path / "stores-2"
    ).build().indexes["user"]

    assert [first_index.external_id(i) for i in range(3)] == [-5, 7, 20]
    assert [second_index.external_id(i) for i in range(3)] == [-5, 7, 20]


def test_same_external_value_in_two_types_has_independent_ordinals(
    tmp_path: Path,
) -> None:
    """There is no global cross-type namespace or collision-prone Python map."""
    source = _make_source(
        tmp_path / "source",
        user_fragments=(("users/a.parquet", pa.int64(), [7, 99]),),
        item_fragments=(("items/a.parquet", pa.int64(), [-2, 7]),),
    )
    object.__setattr__(source.spec.node_types[0], "id_dtype", "int64")
    indexes = ParquetTypedGraphIngestor(source, tmp_path / "stores").build().indexes

    assert indexes["user"].lookup(7) == 0
    assert indexes["item"].lookup(7) == 1
    assert indexes["user"].external_id(0) == 7
    assert indexes["item"].external_id(1) == 7


@pytest.mark.parametrize(
    ("fragments", "check_id"),
    [
        (
            (
                ("users/a.parquet", pa.int64(), [1, 2]),
                ("users/b.parquet", pa.int64(), [2, 3]),
            ),
            "ID-DUPLICATE-001",
        ),
        (
            (("users/a.parquet", pa.int64(), [1, None]),),
            "ID-NULL-001",
        ),
        (
            (
                ("users/a.parquet", pa.int64(), [1]),
                ("users/b.parquet", pa.string(), ["2"]),
            ),
            "SCHEMA-DRIFT-001",
        ),
    ],
)
def test_invalid_id_domains_are_rejected_across_fragments(
    tmp_path: Path,
    fragments: tuple[tuple[str, pa.DataType, list[object | None]], ...],
    check_id: str,
) -> None:
    """Null, duplicate, or physically mixed IDs never enter an index."""
    source = _make_source(tmp_path / "source", user_fragments=fragments)
    ingestor = ParquetTypedGraphIngestor(source, tmp_path / "stores")

    with pytest.raises(ArtifactValidationError, match=check_id):
        ingestor.build()


def test_per_type_completion_records_dependencies_inputs_and_outputs(
    tmp_path: Path,
) -> None:
    """Each completed type is independently checksum and dependency evidenced."""
    source = _make_source(tmp_path / "source")
    result = ParquetTypedGraphIngestor(source, tmp_path / "stores").build()

    for node_type, index in result.indexes.items():
        record = json.loads(index.completion_path.read_text())
        assert record["stage"] == "external_node_index"
        assert record["node_type"] == node_type
        assert record["behavior_version"]
        assert record["input_fingerprint"] == result.inventory.source_fingerprint
        assert record["config_fingerprint"] == result.inventory.config_fingerprint
        assert set(record["dependency_versions"]) == {"duckdb", "pyarrow"}
        assert set(record["outputs"]) == {"lookup.duckdb", "node_ids.parquet"}


def _rewrite_tamper_evidence(result, index: ExternalNodeIndex) -> None:
    record = json.loads(index.completion_path.read_text())
    record["outputs"] = {
        "lookup.duckdb": _sha256(index.lookup_path),
        "node_ids.parquet": _sha256(index.node_ids_path),
    }
    index.completion_path.write_text(json.dumps(record))
    build_path = result.stage_root / "build.complete.json"
    build = json.loads(build_path.read_text())
    for path in (index.lookup_path, index.node_ids_path, index.completion_path):
        relative = path.relative_to(result.stage_root).as_posix()
        build["outputs"][relative] = _sha256(path)
    build_path.write_text(json.dumps(build))


@pytest.mark.parametrize(
    ("reverse_consistent", "check_id"),
    [
        (True, "INDEX-CANONICAL-001"),
        (False, "INDEX-REVERSE-001"),
    ],
)
def test_resume_rejects_dense_permutations_even_with_rewritten_checksums(
    tmp_path: Path,
    reverse_consistent: bool,
    check_id: str,
) -> None:
    """Dense ordinals must retain canonical order and exact reverse parity."""
    import duckdb

    source = _make_source(tmp_path / "source")
    ingestor = ParquetTypedGraphIngestor(source, tmp_path / "stores")
    result = ingestor.build()
    index = result.indexes["user"]
    index.close()
    connection = duckdb.connect(str(index.lookup_path))
    try:
        connection.execute(
            "UPDATE mapping SET local_ordinal = CASE local_ordinal "
            "WHEN 0 THEN 1 WHEN 1 THEN 0 ELSE local_ordinal END"
        )
        connection.execute("CHECKPOINT")
    finally:
        connection.close()

    reverse = pq.read_table(index.node_ids_path)
    external_ids = reverse.column("external_id").to_pylist()
    if reverse_consistent:
        rewritten = [external_ids[1], external_ids[0], *external_ids[2:]]
    else:
        rewritten = [external_ids[2], external_ids[0], external_ids[1]]
    pq.write_table(
        pa.table(
            {
                "local_ordinal": reverse.column("local_ordinal"),
                "external_id": pa.array(rewritten, type=pa.int64()),
            }
        ),
        index.node_ids_path,
    )
    _rewrite_tamper_evidence(result, index)

    with pytest.raises(ArtifactValidationError, match=check_id):
        ingestor.build_external_node_indexes(result.inventory)


def test_mapping_reads_immutable_snapshots_when_sources_swap_between_queries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A transient source swap after copying cannot alter the mapped domain."""
    import duckdb

    source = _make_source(tmp_path / "source")
    source_path = source.spec.source_root / "items/a.parquet"
    original_bytes = source_path.read_bytes()
    real_connect = duckdb.connect
    observed_read_paths: list[Path] = []

    class SwappingConnection:
        def __init__(self, connection) -> None:
            self.connection = connection
            self.pending_restore = False
            self.swapped = False

        def read_parquet(self, paths):
            observed_read_paths.extend(Path(path) for path in paths)
            if not self.swapped:
                self.swapped = True
                pq.write_table(
                    pa.table(
                        {
                            "node_id": pa.array([7], type=pa.uint64()),
                            "feature": pa.array([0], type=pa.float32()),
                        }
                    ),
                    source_path,
                )
                self.pending_restore = True
            return self.connection.read_parquet(paths)

        def execute(self, *args, **kwargs):
            try:
                return self.connection.execute(*args, **kwargs)
            finally:
                if self.pending_restore:
                    source_path.write_bytes(original_bytes)
                    self.pending_restore = False

        def __getattr__(self, name):
            return getattr(self.connection, name)

    monkeypatch.setattr(
        duckdb,
        "connect",
        lambda *args, **kwargs: SwappingConnection(
            real_connect(*args, **kwargs)
        ),
    )
    result = ParquetTypedGraphIngestor(source, tmp_path / "stores").build()

    assert source_path.read_bytes() == original_bytes
    assert source_path not in observed_read_paths
    assert [
        result.indexes["item"].external_id(ordinal)
        for ordinal in range(3)
    ] == [0, 7, 2**64 - 1]


def test_snapshot_copy_rejects_source_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every copied fragment must match the inventoried bytes exactly."""
    source = _make_source(tmp_path / "source")
    ingestor = ParquetTypedGraphIngestor(source, tmp_path / "stores")
    inventory = ingestor.inventory()
    target = source.spec.source_root / "users/a.parquet"
    original_bytes = target.read_bytes()
    real_copy = getattr(ingestion_module, "_copy_inventory_file", None)
    called = False

    def mutating_copy(entry, destination):
        nonlocal called
        if entry.absolute_path == target:
            called = True
            target.write_bytes(original_bytes + b"mutated")
        try:
            assert real_copy is not None
            return real_copy(entry, destination)
        finally:
            if entry.absolute_path == target:
                target.write_bytes(original_bytes)

    monkeypatch.setattr(
        ingestion_module,
        "_copy_inventory_file",
        mutating_copy,
        raising=False,
    )
    with pytest.raises(SourceMutationError, match="SOURCE-MUTATION-001"):
        ingestor.build_external_node_indexes(inventory)
    assert called
