"""Contract tests for immutable typed Parquet graph source descriptors."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import importlib
from pathlib import Path
import subprocess
import sys
from types import ModuleType
from typing import Any

import pytest


@pytest.fixture
def parquet_api() -> ModuleType:
    """Import the optional-dependency-free schema module under test."""
    return importlib.import_module("topobench.data.loaders.parquet")


def _node(
    api: ModuleType,
    name: str = "paper",
    *,
    paths: tuple[str, ...] = ("nodes/z.parquet", "nodes/a.parquet"),
    id_dtype: str = "int64",
    feature_columns: tuple[str, ...] = ("feature_0", "feature_1"),
    feature_width: int = 2,
    feature_representation: str = "scalar_columns",
) -> Any:
    return api.NodeTypeSpec(
        name=name,
        paths=paths,
        id_column=f"{name}_id",
        id_dtype=id_dtype,
        feature_columns=feature_columns,
        feature_dtype="float32",
        feature_width=feature_width,
        feature_representation=feature_representation,
    )


def _split_set(
    api: ModuleType,
    tag: str = "split_01_01_1990",
    *,
    coverage: str = "partial",
) -> Any:
    return api.SplitSetSpec(
        tag=tag,
        train=f"splits/train_{tag}.parquet",
        val=f"splits/val_{tag}.parquet",
        test=f"splits/test_{tag}.parquet",
        coverage=coverage,
    )


def _supervision(
    api: ModuleType,
    target: str = "paper",
    *,
    sets: tuple[Any, ...] | None = None,
) -> Any:
    split_sets = sets or (_split_set(api),)
    return api.SupervisionSpec(
        target_node_type=target,
        label_column="label",
        label_dtype="int64",
        split_registry=api.SplitRegistrySpec(
            active_tag="split_01_01_1990",
            sets=split_sets,
            cross_tag_overlap="allowed",
            within_phase_ids="unique",
            within_tag_phases="disjoint",
            target_id_resolution="required",
        ),
    )


def _relation(
    api: ModuleType,
    relation: tuple[str, str, str] = ("paper", "cites", "paper"),
    *,
    paths: tuple[str, ...] = ("edges/z.parquet", "edges/a.parquet"),
) -> Any:
    return api.RelationSpec(
        relation=relation,
        paths=paths,
        source_column="source_id",
        destination_column="destination_id",
        edge_id_column="edge_id",
        edge_fields=("weight", "timestamp"),
    )


def _homogeneous_spec(
    api: ModuleType,
    source_root: str | Path = ".",
    *,
    node: Any | None = None,
    relation: Any | None = None,
    supervision: Any | None = None,
    partition: Any | None = None,
) -> Any:
    return api.ParquetTypedGraphSpec(
        source_root=source_root,
        output_kind="homogeneous",
        node_types=(node or _node(api),),
        relations=(relation or _relation(api),),
        supervision=supervision or _supervision(api),
        partition=partition or api.PartitionSpec(strategy="cluster"),
    )


def _loader_parameters() -> dict[str, Any]:
    tag = "split_01_01_1990"
    return {
        "data_domain": "graph",
        "data_type": "parquet_typed",
        "data_name": "ParquetTypedGraph",
        "source_root": ".",
        "output_kind": "homogeneous",
        "node_types": {
            "paper": {
                "paths": ["nodes/paper.parquet"],
                "columns": {
                    "id": "paper_id",
                    "id_dtype": "int64",
                    "features": {
                        "columns": ["feature_0", "feature_1"],
                        "dtype": "float32",
                        "width": 2,
                        "representation": "scalar_columns",
                    },
                },
            }
        },
        "edge_types": [
            {
                "type": ["paper", "cites", "paper"],
                "paths": ["edges/cites.parquet"],
                "columns": {
                    "source": "source_id",
                    "destination": "destination_id",
                    "edge_id": "edge_id",
                    "fields": ["weight"],
                },
            }
        ],
        "supervision": {
            "target_node_type": "paper",
            "labels": {
                "source": "nodes",
                "column": "label",
                "dtype": "int64",
                "paths": None,
                "node_id": None,
            },
            "splits": {
                "active": tag,
                "cross_tag_overlap": "allowed",
                "within_phase_ids": "unique",
                "within_tag_phases": "disjoint",
                "target_id_resolution": "required",
                "sets": {
                    tag: {
                        "train": f"splits/train_{tag}.parquet",
                        "val": f"splits/val_{tag}.parquet",
                        "test": f"splits/test_{tag}.parquet",
                        "coverage": "partial",
                    }
                },
            },
        },
        "partition": {
            "strategy": "cluster",
            "backend": "pyg",
            "num_partitions": 1500,
            "recursive": False,
            "memory_limit_bytes": 256 * 1024**3,
            "external_partition_map": None,
        },
        "fitted_transform": {"name": "identity", "fit_on": "train"},
        "profiling": {
            "enabled": True,
            "sample_every_steps": 10,
            "emit_on_duration_delta": 0.1,
            "emit_on_memory_delta_bytes": 256 * 1024**2,
        },
        "reproducibility": {"save_reproducibility_bundle": True},
        "ingestion": {
            "record_batch_rows": 65_536,
            "memory_limit_bytes": 4 * 1024**3,
            "temp_directory": "tmp",
        },
    }


def test_one_type_homogeneous_schema_is_canonical_and_fixed_width(
    parquet_api: ModuleType,
) -> None:
    spec = _homogeneous_spec(parquet_api)

    assert spec.output_kind == "homogeneous"
    assert tuple(node.name for node in spec.node_types) == ("paper",)
    assert spec.node_types[0].id_dtype == "int64"
    assert spec.node_types[0].feature_width == 2
    assert spec.node_types[0].paths == (
        "nodes/a.parquet",
        "nodes/z.parquet",
    )
    assert spec.relations[0].paths == (
        "edges/a.parquet",
        "edges/z.parquet",
    )


def test_multi_type_heterogeneous_schema_preserves_typed_relations_and_fields(
    parquet_api: ModuleType,
) -> None:
    spec = parquet_api.ParquetTypedGraphSpec(
        source_root=".",
        output_kind="heterogeneous",
        node_types=(
            _node(
                parquet_api,
                "paper",
                paths=("nodes/paper.parquet",),
                id_dtype="uint64",
                feature_columns=("embedding",),
                feature_width=128,
                feature_representation="fixed_size_list",
            ),
            _node(
                parquet_api,
                "author",
                paths=("nodes/author.parquet",),
                id_dtype="string",
                feature_columns=("feature_0", "feature_1", "feature_2"),
                feature_width=3,
            ),
        ),
        relations=(
            _relation(
                parquet_api,
                ("paper", "written_by", "author"),
                paths=("edges/written_by.parquet",),
            ),
            _relation(
                parquet_api,
                ("author", "writes", "paper"),
                paths=("edges/writes.parquet",),
            ),
        ),
        supervision=_supervision(parquet_api),
        partition=parquet_api.PartitionSpec(strategy="neighbor"),
    )

    assert tuple(node.name for node in spec.node_types) == ("author", "paper")
    assert tuple(relation.relation for relation in spec.relations) == (
        ("author", "writes", "paper"),
        ("paper", "written_by", "author"),
    )
    assert spec.node_types[0].id_dtype == "string"
    assert spec.node_types[1].id_dtype == "uint64"
    assert spec.node_types[1].feature_width == 128
    assert spec.relations[0].source_column == "source_id"
    assert spec.relations[0].destination_column == "destination_id"
    assert spec.relations[0].edge_id_column == "edge_id"
    assert spec.relations[0].edge_fields == ("timestamp", "weight")


def test_semantically_equal_declarations_have_canonical_equality_and_hashes(
    parquet_api: ModuleType,
) -> None:
    first = _homogeneous_spec(parquet_api)
    second = _homogeneous_spec(
        parquet_api,
        node=_node(
            parquet_api,
            paths=("nodes/a.parquet", "nodes/z.parquet"),
        ),
        relation=_relation(
            parquet_api,
            paths=("edges/a.parquet", "edges/z.parquet"),
        ),
    )

    assert first == second
    assert hash(first) == hash(second)


def test_all_nested_schema_state_is_frozen_and_tuple_backed(
    parquet_api: ModuleType,
) -> None:
    spec = _homogeneous_spec(parquet_api)

    with pytest.raises(FrozenInstanceError):
        spec.output_kind = "heterogeneous"
    with pytest.raises(FrozenInstanceError):
        spec.node_types[0].feature_width = 3
    assert isinstance(spec.node_types, tuple)
    assert isinstance(spec.relations, tuple)
    assert isinstance(spec.supervision.split_registry.sets, tuple)
    assert isinstance(spec.node_types[0].feature_columns, tuple)


def test_split_registry_names_triplets_and_declares_overlap_invariants(
    parquet_api: ModuleType,
) -> None:
    older = _split_set(parquet_api, "split_01_01_1990")
    newer = _split_set(parquet_api, "split_01_01_2000")
    registry = parquet_api.SplitRegistrySpec(
        active_tag="split_01_01_1990",
        sets=(newer, older),
        cross_tag_overlap="allowed",
        within_phase_ids="unique",
        within_tag_phases="disjoint",
        target_id_resolution="required",
    )

    assert tuple(split.tag for split in registry.sets) == (
        "split_01_01_1990",
        "split_01_01_2000",
    )
    assert registry.sets[0].train == (
        "splits/train_split_01_01_1990.parquet"
    )
    assert registry.sets[0].val == "splits/val_split_01_01_1990.parquet"
    assert registry.sets[0].test == "splits/test_split_01_01_1990.parquet"
    assert registry.sets[0].qualified is True
    assert registry.cross_tag_overlap == "allowed"
    assert registry.within_phase_ids == "unique"
    assert registry.within_tag_phases == "disjoint"
    assert registry.target_id_resolution == "required"


@pytest.mark.parametrize("coverage", ["complete", "partial"])
def test_split_coverage_policy_is_explicit_and_closed(
    parquet_api: ModuleType,
    coverage: str,
) -> None:
    assert _split_set(parquet_api, coverage=coverage).coverage == coverage


def test_default_partition_admission_is_exactly_256_gib(
    parquet_api: ModuleType,
) -> None:
    partition = parquet_api.PartitionSpec(strategy="cluster")

    assert partition.memory_limit_bytes == 256 * 1024**3


def test_existing_manifest_registers_both_output_strategy_capabilities(
    parquet_api: ModuleType,
) -> None:
    from topobench.data.capabilities import GRAPH_DATASET_MANIFEST

    capability = GRAPH_DATASET_MANIFEST["ParquetTypedGraph"]

    assert capability.descriptor_only is True
    assert capability.supports_source(
        domain="graph",
        output_kind="homogeneous",
        strategy="cluster",
        backend="pyg",
    )
    assert capability.supports_source(
        domain="heterogeneous",
        output_kind="heterogeneous",
        strategy="neighbor",
        backend="pyg",
    )

    mutable_capabilities = set(capability.source_capabilities)
    immutable_copy = replace(
        capability,
        source_capabilities=mutable_capabilities,  # type: ignore[arg-type]
    )
    mutable_capabilities.clear()
    assert isinstance(immutable_copy.source_capabilities, frozenset)
    assert hash(immutable_copy)
    assert immutable_copy.supports_source(
        domain="graph",
        output_kind="homogeneous",
        strategy="cluster",
        backend="pyg",
    )


def test_loader_exposes_a_descriptor_without_dataset_or_row_io(
    parquet_api: ModuleType,
    tmp_path: Path,
) -> None:
    parameters = _loader_parameters()
    parameters["source_root"] = str(tmp_path / "source-does-not-exist")

    loader = parquet_api.ParquetTypedGraphLoader(parameters=parameters)

    assert isinstance(loader.source, parquet_api.ParquetTypedGraphSource)
    assert isinstance(loader.source.spec, parquet_api.ParquetTypedGraphSpec)
    assert not hasattr(loader, "load_dataset")
    assert "Dataset" not in {base.__name__ for base in type(loader).__mro__}
    assert loader.source.files
    assert all(path.suffix == ".parquet" for path in loader.source.files)


def test_schema_import_succeeds_without_duckdb_or_pyarrow() -> None:
    command = [
        sys.executable,
        "-c",
        (
            "import sys; "
            "sys.modules['duckdb'] = None; "
            "sys.modules['pyarrow'] = None; "
            "import topobench.data.loaders.parquet; "
            "assert sys.modules['duckdb'] is None; "
            "assert sys.modules['pyarrow'] is None; "
            "print('optional-imports-clean')"
        ),
    ]

    result = subprocess.run(command, check=True, capture_output=True, text=True)

    assert result.stdout.strip() == "optional-imports-clean"


@pytest.mark.parametrize("id_dtype", ["float64", "bool", "object", "utf8"])
def test_node_id_dtype_rejects_noncanonical_domains(
    parquet_api: ModuleType,
    id_dtype: str,
) -> None:
    with pytest.raises(ValueError, match=r"node_types\[paper\]\.id_dtype"):
        _node(parquet_api, id_dtype=id_dtype)


@pytest.mark.parametrize(
    ("representation", "columns", "width"),
    [
        ("variable_list", ("embedding",), 4),
        ("fixed_size_list", ("left", "right"), 4),
        ("scalar_columns", ("left", "right"), 3),
        ("scalar_columns", ("left", "left"), 2),
    ],
)
def test_node_features_reject_variable_or_width_ambiguous_mappings(
    parquet_api: ModuleType,
    representation: str,
    columns: tuple[str, ...],
    width: int,
) -> None:
    with pytest.raises(ValueError, match=r"node_types\[paper\]\.features"):
        _node(
            parquet_api,
            feature_representation=representation,
            feature_columns=columns,
            feature_width=width,
        )


def test_split_rejects_a_missing_triplet_member(
    parquet_api: ModuleType,
) -> None:
    parameters = _loader_parameters()
    del parameters["supervision"]["splits"]["sets"][
        "split_01_01_1990"
    ]["val"]

    with pytest.raises(
        ValueError,
        match=r"supervision\.splits\.sets\[split_01_01_1990\]\.val",
    ):
        parquet_api.ParquetTypedGraphLoader(parameters=parameters)


def test_split_rejects_duplicate_phase_filenames(
    parquet_api: ModuleType,
) -> None:
    path = "splits/phase.parquet"

    with pytest.raises(ValueError, match="phase filenames must be distinct"):
        parquet_api.SplitSetSpec(
            tag="split_01_01_1990",
            train=path,
            val=path,
            test="splits/test.parquet",
            coverage="partial",
        )


@pytest.mark.parametrize("tag", ["", "has space", "../escape", "train/tag"])
def test_split_rejects_illegal_tags(
    parquet_api: ModuleType,
    tag: str,
) -> None:
    with pytest.raises(ValueError, match="split tag"):
        _split_set(parquet_api, tag)


def test_split_registry_rejects_duplicate_tags(
    parquet_api: ModuleType,
) -> None:
    split = _split_set(parquet_api)

    with pytest.raises(ValueError, match="duplicate split tag"):
        parquet_api.SplitRegistrySpec(
            active_tag=split.tag,
            sets=(split, split),
        )


def test_split_registry_rejects_unresolved_target_id_policy(
    parquet_api: ModuleType,
) -> None:
    with pytest.raises(ValueError, match="target_id_resolution"):
        parquet_api.SplitRegistrySpec(
            active_tag="split_01_01_1990",
            sets=(_split_set(parquet_api),),
            target_id_resolution="allow_unresolved",
        )


def test_supervision_rejects_an_unresolved_target_node_type(
    parquet_api: ModuleType,
) -> None:
    with pytest.raises(ValueError, match="target_node_type.*missing"):
        _homogeneous_spec(
            parquet_api,
            supervision=_supervision(parquet_api, target="missing"),
        )

@pytest.mark.parametrize(
    ("label_column", "target_role"),
    [
        ("paper_id", "id_column"),
        ("feature_0", "feature_columns"),
    ],
)
def test_node_sourced_labels_reject_target_column_role_overlap(
    parquet_api: ModuleType,
    label_column: str,
    target_role: str,
) -> None:
    supervision = parquet_api.SupervisionSpec(
        target_node_type="paper",
        label_column=label_column,
        label_dtype="int64",
        label_source="nodes",
        split_registry=_supervision(parquet_api).split_registry,
    )

    with pytest.raises(
        ValueError,
        match=(
            rf"supervision\.labels\.column.*{label_column}"
            rf".*node_types\[paper\]\.{target_role}"
        ),
    ):
        _homogeneous_spec(parquet_api, supervision=supervision)


def test_node_sourced_labels_accept_a_distinct_target_column(
    parquet_api: ModuleType,
) -> None:
    spec = _homogeneous_spec(
        parquet_api,
        supervision=_supervision(parquet_api),
    )

    assert spec.supervision.label_column == "label"


@pytest.mark.parametrize("label_column", ["paper_id", "feature_0"])
def test_separate_label_files_keep_independent_column_roles(
    parquet_api: ModuleType,
    label_column: str,
) -> None:
    supervision = parquet_api.SupervisionSpec(
        target_node_type="paper",
        label_column=label_column,
        label_dtype="int64",
        label_source="dataset",
        label_paths=("labels/labels.parquet",),
        label_id_column="source_paper_id",
        split_registry=_supervision(parquet_api).split_registry,
    )

    spec = _homogeneous_spec(parquet_api, supervision=supervision)

    assert spec.supervision.label_column == label_column


def test_split_rejects_invalid_coverage_policy(
    parquet_api: ModuleType,
) -> None:
    with pytest.raises(ValueError, match="coverage"):
        _split_set(parquet_api, coverage="best_effort")


@pytest.mark.parametrize("owner", ["node", "relation"])
def test_schema_rejects_a_source_with_zero_declared_files(
    parquet_api: ModuleType,
    owner: str,
) -> None:
    with pytest.raises(ValueError, match="paths must contain at least one file"):
        if owner == "node":
            _node(parquet_api, paths=())
        else:
            _relation(parquet_api, paths=())


@pytest.mark.parametrize(
    "path",
    ["/absolute/nodes.parquet", "../escape.parquet", "nodes/../../escape"],
)
def test_schema_rejects_absolute_and_traversal_paths(
    parquet_api: ModuleType,
    path: str,
) -> None:
    with pytest.raises(ValueError, match="safe relative path"):
        _node(parquet_api, paths=(path,))

def test_schema_rejects_node_relation_file_reuse(
    parquet_api: ModuleType,
) -> None:
    with pytest.raises(
        ValueError,
        match=(
            r"shared\.parquet.*node_types\[paper\]\.paths"
            r".*relations\["
        ),
    ):
        _homogeneous_spec(
            parquet_api,
            node=_node(parquet_api, paths=("shared.parquet",)),
            relation=_relation(parquet_api, paths=("shared.parquet",)),
        )

def test_schema_rejects_cross_role_hard_links(
    parquet_api: ModuleType,
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    node_path = source_root / "node.parquet"
    relation_path = source_root / "relation.parquet"
    node_path.touch()
    relation_path.hardlink_to(node_path)

    with pytest.raises(
        ValueError,
        match=(
            r"node\.parquet.*node_types\[paper\]\.paths"
            r".*relation\.parquet.*relations\["
        ),
    ):
        _homogeneous_spec(
            parquet_api,
            source_root=source_root,
            node=_node(parquet_api, paths=("node.parquet",)),
            relation=_relation(
                parquet_api,
                paths=("relation.parquet",),
            ),
        )


def test_schema_rejects_split_label_file_reuse(
    parquet_api: ModuleType,
) -> None:
    tag = "split_01_01_1990"
    split = parquet_api.SplitSetSpec(
        tag=tag,
        train="shared.parquet",
        val="splits/val.parquet",
        test="splits/test.parquet",
        coverage="partial",
    )
    supervision = parquet_api.SupervisionSpec(
        target_node_type="paper",
        label_column="label",
        label_dtype="int64",
        label_source="dataset",
        label_paths=("shared.parquet",),
        label_id_column="paper_id",
        split_registry=parquet_api.SplitRegistrySpec(
            active_tag=tag,
            sets=(split,),
        ),
    )

    with pytest.raises(
        ValueError,
        match=(
            r"shared\.parquet.*supervision\.labels\.paths"
            r".*supervision\.splits"
        ),
    ):
        _homogeneous_spec(parquet_api, supervision=supervision)


@pytest.mark.parametrize("symlink_kind", ["leaf", "intermediate"])
def test_schema_rejects_in_root_symlink_components(
    parquet_api: ModuleType,
    tmp_path: Path,
    symlink_kind: str,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    if symlink_kind == "leaf":
        target = source_root / "target.parquet"
        target.touch()
        declared = "alias.parquet"
        (source_root / declared).symlink_to(target)
    else:
        target = source_root / "target"
        target.mkdir()
        declared = "alias/nodes.parquet"
        (source_root / "alias").symlink_to(
            target,
            target_is_directory=True,
        )

    with pytest.raises(
        ValueError,
        match=r"node_types\[paper\]\.paths.*symlink component.*alias",
    ):
        _homogeneous_spec(
            parquet_api,
            source_root=source_root,
            node=_node(parquet_api, paths=(declared,)),
        )


def test_schema_rejects_a_symlink_escape_from_the_source_root(
    parquet_api: ModuleType,
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    outside = tmp_path / "outside"
    source_root.mkdir()
    outside.mkdir()
    (source_root / "escape").symlink_to(outside, target_is_directory=True)

    with pytest.raises(
        ValueError,
        match=r"node_types\[paper\]\.paths.*symlink component.*escape",
    ):
        _homogeneous_spec(
            parquet_api,
            source_root=source_root,
            node=_node(parquet_api, paths=("escape/nodes.parquet",)),
        )


def test_schema_rejects_duplicate_node_types(parquet_api: ModuleType) -> None:
    paper = _node(parquet_api)

    with pytest.raises(ValueError, match="duplicate node type"):
        parquet_api.ParquetTypedGraphSpec(
            source_root=".",
            output_kind="heterogeneous",
            node_types=(paper, paper),
            relations=(_relation(parquet_api),),
            supervision=_supervision(parquet_api),
            partition=parquet_api.PartitionSpec(strategy="neighbor"),
        )


def test_schema_rejects_duplicate_relations(parquet_api: ModuleType) -> None:
    relation = _relation(parquet_api)

    with pytest.raises(ValueError, match="duplicate relation"):
        parquet_api.ParquetTypedGraphSpec(
            source_root=".",
            output_kind="homogeneous",
            node_types=(_node(parquet_api),),
            relations=(relation, relation),
            supervision=_supervision(parquet_api),
            partition=parquet_api.PartitionSpec(strategy="cluster"),
        )


@pytest.mark.parametrize(
    "relation",
    [("paper", "cites"), ("paper", "", "paper"), ("paper", "cites", "paper", "extra")],
)
def test_relation_rejects_malformed_canonical_triples(
    parquet_api: ModuleType,
    relation: tuple[str, ...],
) -> None:
    with pytest.raises(ValueError, match="relation must be a canonical triple"):
        _relation(parquet_api, relation=relation)  # type: ignore[arg-type]


def test_relation_rejects_unresolved_source_or_destination_type(
    parquet_api: ModuleType,
) -> None:
    with pytest.raises(ValueError, match="relation.*unknown node type"):
        _homogeneous_spec(
            parquet_api,
            relation=_relation(parquet_api, ("author", "writes", "paper")),
        )


@pytest.mark.parametrize("executable_key", ["sql", "python"])
def test_loader_rejects_executable_sql_or_python_configuration(
    parquet_api: ModuleType,
    executable_key: str,
) -> None:
    parameters = _loader_parameters()
    parameters[executable_key] = (
        "SELECT * FROM nodes"
        if executable_key == "sql"
        else "lambda row: row"
    )

    with pytest.raises(
        ValueError,
        match=f"unsupported configuration key.*{executable_key}",
    ):
        parquet_api.ParquetTypedGraphLoader(parameters=parameters)


@pytest.mark.parametrize(
    ("output_kind", "strategy", "backend"),
    [
        ("tensor", "cluster", "pyg"),
        ("homogeneous", "neighbor", "pyg"),
        ("heterogeneous", "full_graph", "pyg"),
        ("homogeneous", "cluster", "duckdb"),
    ],
)
def test_schema_rejects_unsupported_output_strategy_backend_combinations(
    parquet_api: ModuleType,
    output_kind: str,
    strategy: str,
    backend: str,
) -> None:
    nodes = (_node(parquet_api),)
    if output_kind == "heterogeneous":
        nodes = (_node(parquet_api, "author"), _node(parquet_api))
    relation = _relation(parquet_api)

    with pytest.raises(ValueError, match="unsupported output/strategy/backend"):
        parquet_api.ParquetTypedGraphSpec(
            source_root=".",
            output_kind=output_kind,
            node_types=nodes,
            relations=(relation,),
            supervision=_supervision(parquet_api),
            partition=parquet_api.PartitionSpec(
                strategy=strategy,
                backend=backend,
            ),
        )
