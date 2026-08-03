# Universal Typed Parquet Graph Store and GPU Streaming Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to
> implement this plan task-by-task.

**Goal:** Add one bounded-memory typed Parquet-to-CSC store to TopoBench that
supports homogeneous cluster-union `Data` batches and heterogeneous
relation-aware neighbor-sampled `HeteroData` batches without duplicating the
framework's training, evaluation, logging, or lifecycle stacks.

**Architecture:** A shared Parquet ingestor writes one immutable universal
physical store. Homogeneous graphs are represented as one node type and one
relation, then exposed through a type-erasing `Data` adapter and deterministic
cluster strategy. Heterogeneous graphs retain typed node maps and relation CSC
and are consumed by PyG `NeighborLoader` through `FeatureStore`/`GraphStore`
views. One TopoBench disk data module owns lazy workers, ordered host/device
prefetch, issued-versus-committed checkpoint state, monitoring, provenance
inputs, and external-ID resolution.

**Tech stack:** Python 3.11, PyTorch 2.3+, PyTorch Geometric 2.8+, Lightning
2.4+, Hydra/OmegaConf, NumPy memory maps, DuckDB and PyArrow in the `parquet`
extra, pytest, Ruff, uv.

## Execution rules

- Work directly on `topobench_graph_hetero`; do not create another worktree.
- Read and follow
  `docs/plans/2026-07-31-parquet-graph-ingestion-streaming-design.md` and
  `docs/plans/2026-07-31-selected-checkpoint-prediction-artifacts-design.md`.
- Preserve TopoBench's loader -> preprocessor -> explicit data pipeline ->
  Lightning data module -> `TBModel` architecture. Do not create a second
  trainer, evaluator, callback stack, logger stack, or CLI entry point.
- Use TDD for every behavior change: failing focused test, smallest production
  change, focused and adjacent regressions, then commit.
- Keep `run.py` data-domain agnostic. Strategy and output-view selection belong
  in configuration and data-pipeline construction.
- One physical store, one manifest schema, one loader/data-module boundary, and
  two explicit sampling strategies. Do not duplicate homogeneous and
  heterogeneous ingestion, store validation, prefetch, monitoring, identity,
  or checkpoint code.
- Homogeneous cluster, heterogeneous cluster, and heterogeneous neighbor
  sampling have distinct scientific contracts. Never hide them behind runtime
  `isinstance` branches or claim cluster and neighbor estimators are
  interchangeable.
- DuckDB/PyArrow imports remain lazy and optional. No ordinary graph,
  heterogeneous, or hypergraph import may require the Parquet extra.
- Parquet conversion and disk runtime never materialize complete features,
  mapped edge tables, or authoritative native graph copies. The explicitly
  admitted materialized-reference partition step may construct topology-only
  `Data`/`HeteroData` under its separate memory ceiling.
- Every committed fixture is independently generated. Real-data qualification
  remains download-gated and never uploads IDs, rows, predictions, or caches.
- Tasks 1-15 implement and qualify the shared materialized-reference and disk
  paths. Remediation Task 21 owns final selected-checkpoint artifact
  publication.

## Contract fixed before implementation

### Source and store

- YAML declares `output_kind`, typed node datasets, canonical relation triples,
  per-type feature schemas, one target node type, an explicit registry of named
  split triplets, execution strategy, resource caps, partition settings,
  fitted transforms, profiling, and reproducibility policy.
- IDs are unique within node type; the same integer/string in different types
  is valid.
- Each relation is stored once as destination-oriented CSC sorted by
  `(destination_local, source_local, edge_id)`.
- Duplicate endpoint pairs require a stable edge identity/order field or fail.
- Reverse flow requires an explicit reverse relation.
- Stable internal directory keys prevent arbitrary type names from becoming
  unchecked paths.
- Promotion from `stores/.staging/<build-id>` to
  `stores/<content-sha256>` is atomic and only follows every checksum,
  cross-reference, partition, split, and qualification check.

### Runtime views

- Homogeneous and heterogeneous cluster strategies emit writable native `Data`
  or `HeteroData` with exact induced directed edges, active-tag supervision,
  and canonical identities.
- Heterogeneous neighbor strategy exposes memory-mapped selected features and
  relation CSC through PyG protocols and emits writable native `HeteroData`.
- Neighbor supervision/export uses only target-type `n_id[:batch_size]`.
- Disk evaluation is deterministic under the qualified cluster or exhaustive
  target-seed neighbor descriptor contract.
- External IDs never enter GPU batches. A validated store resolver restores
  them for selected-checkpoint prediction export.

### Shared lifecycle

- Every batch has a monotonic sequence ID and a strategy descriptor.
- Checkpoints persist committed cursor/RNG/evaluator/strategy state plus
  immutable store, partition-book, active split, and fitted-transform
  identities, never prefetched issued state.
- Gradient accumulation commits a sequence group only after a successful
  global-step advance.
- One generic host/device prefetch implementation handles `Data` and
  `HeteroData` with explicit byte budgets.
- One structured event/check stream records conversion, partitioning,
  validation, fitted transforms, wait/read/assembly/H2D/compute, resources, and
  actionable failures locally and through bounded logger summaries.
- `save_reproducibility_bundle` defaults to true and is mandatory for qualified
  runs.

---

### Task 1: Define the typed Parquet schema and capability contract

**Files:**

- Create: `topobench/data/loaders/parquet.py`
- Modify: `topobench/data/capabilities.py`
- Create: `configs/dataset/graph/ParquetTypedGraph.yaml`
- Create: `configs/dataset/heterogeneous/ParquetTypedGraph.yaml`
- Create: `test/data/loaders/test_parquet_typed_schema.py`
- Modify: `test/config/test_all_surviving_configs.py`

**Step 1: Write failing schema tests**

Cover one-type homogeneous and multi-type/multi-relation heterogeneous specs;
per-type ID dtypes and feature widths; stable relation triples; source/
destination maps; edge fields and duplicate-edge IDs; canonical path ordering;
output/strategy capability; 256 GiB default partition admission; and explicit
split registries such as `train_split_01_01_1990`, `val_split_01_01_1990`, and
`test_split_01_01_1990`.

Assert multiple tags may overlap with each other while train/validation/test IDs
within one tag are unique and pairwise disjoint. Reject missing triplet members,
duplicate phase filenames, illegal tags, unresolved target IDs, invalid
complete/partial coverage, zero files, unsafe paths, duplicate type/relation
names, malformed triples, variable feature lists, configuration SQL/Python,
and unsupported output/strategy/backend combinations.

**Step 2: Run the red tests**

```bash
uv run pytest test/data/loaders/test_parquet_typed_schema.py test/config/test_all_surviving_configs.py -q
```

**Step 3: Implement frozen specs and capability declarations**

Add immutable `ParquetTypedGraphSpec`, `NodeTypeSpec`, `RelationSpec`,
`SupervisionSpec`, `SplitRegistrySpec`, `SplitSetSpec`, `PartitionSpec`,
`FittedTransformSpec`, `ProfilingSpec`, `ReproducibilitySpec`, and
`IngestionLimits`. Normalize exact type, relation, file, and tag ordering once.
Expose a lightweight source descriptor from the loader; do not force it
through the in-memory `InMemoryDataset` path.

Register graph and heterogeneous selectors explicitly through the existing
capability manifest. Keep the two YAML selectors thin: same loader/schema,
different explicit data-pipeline/output strategy.

**Step 4: Run focused and clean-import tests**

```bash
uv run pytest test/data/loaders/test_parquet_typed_schema.py test/config/test_all_surviving_configs.py test/architecture/test_domain_contract.py -q
```

**Step 5: Commit**

```bash
git add topobench/data/loaders/parquet.py topobench/data/capabilities.py configs/dataset/graph/ParquetTypedGraph.yaml configs/dataset/heterogeneous/ParquetTypedGraph.yaml test/data/loaders/test_parquet_typed_schema.py test/config/test_all_surviving_configs.py
git commit -m "feat: define typed Parquet graph sources"
```

---

### Task 2: Build bounded inventory and per-type external-ID indexes

**Files:**

- Create: `topobench/data/stores/typed_graph_ingestion.py`
- Create: `topobench/data/stores/external_node_index.py`
- Create: `topobench/data/stores/__init__.py`
- Create: `test/data/stores/test_typed_graph_inventory.py`
- Create: `test/data/stores/test_external_node_index.py`
- Create: `test/integration/qualify_typed_graph_id_rss.py`

**Step 1: Write failing inventory and ID tests**

Use generated Parquet fragments with unsorted arbitrary `int64`, `uint64`, and
UTF-8 IDs. Assert deterministic dense local ordinals per node type, exact
round-trip maps, collision safety for the same external ID in two types,
source-byte digests, canonical path ordering, row counts, schema fingerprints,
and preflight final/temp disk estimates.

Reject null/mixed IDs, duplicate IDs within a type, schema drift, mutation
between inventory and mapping, insufficient declared disk, and unknown
artifacts. The RSS harness must fail if implementation holds all IDs in a
Python dictionary or RAM-wide Arrow table.

**Step 2: Run the red tests**

```bash
uv run pytest test/data/stores/test_typed_graph_inventory.py test/data/stores/test_external_node_index.py -q
```

**Step 3: Implement bounded DuckDB/Arrow inventory and maps**

Create a locked staging root keyed by canonical source/config fingerprint.
Configure DuckDB memory and spill directory explicitly. Stream file digests.
Externally order each node type within its exact ID domain, assign dense local
`int64` ordinals, and write disk lookup state plus per-type
`node_ids.parquet`. Do not construct a global cross-type ID map.

Every stage writes a checksum-validated completion record. Resume accepts it
only when source bytes, dependency versions, semantic schema, and behavior
options match exactly.

**Step 4: Run focused tests and RSS harness**

```bash
uv run pytest test/data/stores/test_typed_graph_inventory.py test/data/stores/test_external_node_index.py -q
uv run python test/integration/qualify_typed_graph_id_rss.py
```

**Step 5: Commit**

```bash
git add topobench/data/stores test/data/stores/test_typed_graph_inventory.py test/data/stores/test_external_node_index.py test/integration/qualify_typed_graph_id_rss.py
git commit -m "feat: map typed graph IDs out of core"
```

---

### Task 3: Stream per-type features and target supervision

**Files:**

- Modify: `topobench/data/stores/typed_graph_ingestion.py`
- Create: `topobench/data/stores/typed_graph_arrays.py`
- Create: `test/data/stores/test_typed_graph_features.py`
- Create: `test/data/stores/test_typed_graph_supervision.py`

**Step 1: Write failing feature and supervision tests**

Exercise different feature widths/dtypes across node types, fixed-size-list and
ordered scalar-column representations, labels in target-node columns or keyed
datasets, generated deterministic splits, and multiple explicit split triplets.
Assert exact arrays, target dtype/shape, alignment by external ID, unique phase
files, uniqueness within each phase, pairwise disjointness within each tag,
declared complete/partial coverage, legal overlap across tags, and stable active
tag/checksum identity.

Reject variable width, non-finite features, unsupported casts, labels on a
non-target type, missing/extra/duplicate supervision rows, implicit positional
joins, phase overlap within a tag, incomplete required supervision, unresolved
target IDs, missing tag phases, and malformed classification/regression targets.

**Step 2: Run the red tests**

```bash
uv run pytest test/data/stores/test_typed_graph_features.py test/data/stores/test_typed_graph_supervision.py -q
```

**Step 3: Implement chunked array writing**

Stream record batches through the per-type indexes into preallocated
memory-mapped arrays or bounded disk staging runs. Preserve exact qualified
feature dtype/width and target semantics. Resolve every registered split source
through the target-type canonical map and write sorted IDs under
`splits/<tag>/`, not full masks for non-target types. Persist tag coverage,
phase/source checksums, shapes, dtypes, counts, and disjointness evidence in
staged metadata.

**Step 4: Run focused tests**

```bash
uv run pytest test/data/stores/test_typed_graph_features.py test/data/stores/test_typed_graph_supervision.py test/data/test_graph_feature_contract.py -q
```

**Step 5: Commit**

```bash
git add topobench/data/stores/typed_graph_ingestion.py topobench/data/stores/typed_graph_arrays.py test/data/stores/test_typed_graph_features.py test/data/stores/test_typed_graph_supervision.py
git commit -m "feat: stream typed graph features and supervision"
```

---

### Task 4: Build exact per-relation destination CSC

**Files:**

- Modify: `topobench/data/stores/typed_graph_ingestion.py`
- Create: `topobench/data/stores/typed_graph_csc.py`
- Create: `test/data/stores/test_typed_graph_csc.py`
- Create: `test/data/stores/test_typed_graph_edge_rejections.py`

**Step 1: Write failing relation round-trip tests**

Generate directed homogeneous and heterogeneous relations, explicit reverse
relations, self-loops, relation-specific weights/attributes, shuffled files,
and duplicate endpoint pairs with stable edge IDs. Assert exact direction,
source/destination type, canonical `(destination, source, edge_id)` order,
`colptr`/`row` values, edge-field alignment, and layout-independent semantic
digests.

Reject unresolved typed endpoints, a source ID looked up in the destination
map, endpoint overflow, null endpoints, implicit reverse edges, ambiguous
endpoint duplicates, duplicate edge IDs, malformed `colptr`, and relation field
length mismatch.

**Step 2: Run the red tests**

```bash
uv run pytest test/data/stores/test_typed_graph_csc.py test/data/stores/test_typed_graph_edge_rejections.py -q
```

**Step 3: Implement external joins and ordered CSC emission**

Use two bounded external joins per relation. Stream the externally sorted
result directly to `colptr.npy`, `row.npy`, stable `edge_id.npy`, and aligned
field arrays. Never materialize mapped edges in Python, silently coalesce
rows, symmetrize the runtime relation, or write a second full adjacency layout.

**Step 4: Run focused tests**

```bash
uv run pytest test/data/stores/test_typed_graph_csc.py test/data/stores/test_typed_graph_edge_rejections.py test/data/test_heterogeneous_spec.py -q
```

**Step 5: Commit**

```bash
git add topobench/data/stores/typed_graph_ingestion.py topobench/data/stores/typed_graph_csc.py test/data/stores/test_typed_graph_csc.py test/data/stores/test_typed_graph_edge_rejections.py
git commit -m "feat: write typed relation CSC out of core"
```

---

### Task 5: Preserve and correct the homogeneous ClusterData baseline

**Files:**

- Inspect and pin: `dleko11:on_disk_transductive` commit
  `b55b876a3bb227fdc5a79b776b6e8d337ff5fe02`
- Create: `topobench/data/stores/materialized_partition.py`
- Create: `test/data/stores/test_materialized_homogeneous_partition.py`
- Create: `test/data/dataload/test_materialized_homogeneous_cluster.py`

**Step 1: Write failing characterization tests**

Recreate independently invented branch fixtures and assert PyG `ClusterData`
partition membership, split-aware selected partition unions, exact directed
induced edges, features, labels, masks, edge fields, and writable native
`Data`. Compare canonical node identity against `perm_to_global`; add a hard
regression that rejects the branch defect where permuted row positions were
reported as `global_nid`.

**Step 2: Run the red tests**

```bash
uv run pytest test/data/stores/test_materialized_homogeneous_partition.py test/data/dataload/test_materialized_homogeneous_cluster.py -q
```

**Step 3: Port the minimum qualified baseline**

Reuse PyG `ClusterData`/partition representations and the branch's useful
selected-partition induced-union behavior. Replace SQLite-plus-memmap
duplication, unsafe pickle/download handling, eager parent-process maps, device
transfer in collation, noncanonical identities, and non-resumable shuffle
rather than copying them. Keep the ordinary TopoBench pipeline and native
`Data` output.

**Step 4: Run focused parity**

```bash
uv run pytest test/data/stores/test_materialized_homogeneous_partition.py test/data/dataload/test_materialized_homogeneous_cluster.py test/data/test_graph_feature_contract.py -q
```

**Step 5: Commit**

```bash
git add topobench/data/stores/materialized_partition.py test/data/stores/test_materialized_homogeneous_partition.py test/data/dataload/test_materialized_homogeneous_cluster.py
git commit -m "refactor: preserve qualified PyG cluster baseline"
```

---

### Task 6: Generate and qualify the typed PyG partition book

**Files:**

- Create: `topobench/data/stores/typed_partition_book.py`
- Create: `topobench/data/stores/pyg_partitioner.py`
- Modify: `topobench/data/stores/typed_graph_ingestion.py`
- Create: `test/data/stores/test_topology_only_pyg_partitioner.py`
- Create: `test/data/stores/test_typed_partition_book.py`
- Create: `test/data/stores/test_partition_qualification.py`
- Create: `test/integration/qualify_pyg_partition_resources.py`

**Step 1: Write failing topology-only and adapter tests**

Build asymmetric typed fixtures with different feature widths, missing and
explicit reverse relations, isolated nodes, parallel relations, edge fields,
and multiple split tags. Assert the PyG input contains only per-type node
counts and relation topology. Run `Partitioner.generate_partition()` in a
trusted temporary directory and prove typed assignment completeness,
permutation/inverse round-trip, relation direction/edge ownership, topology
fingerprint, and absence of leaked temporary reverse arcs.

**Step 2: Write hard qualification failures**

Reject missing/duplicate/out-of-range typed IDs, malformed PyG output,
fingerprint mismatch, empty/pathological partitions, and configured per-type,
per-phase for every qualified split tag, per-relation, feature-byte, total-size,
and cut/locality limits by exact stable check ID. Prove a validated external
partition map can replace a rejected or over-budget PyG candidate.

**Step 3: Run the red tests**

```bash
uv run pytest test/data/stores/test_topology_only_pyg_partitioner.py test/data/stores/test_typed_partition_book.py test/data/stores/test_partition_qualification.py -q
```

**Step 4: Implement generation, adaptation, and qualification**

Construct a temporary undirected METIS scoring view without mutating canonical
relations. Estimate topology RAM/temp disk against the 256 GiB default, invoke
PyG unchanged, record measured resources, adapt the trusted local output into
an immutable typed partition book, and validate before publication. Treat
independent nondeterministic METIS reruns as new partition identities.

**Step 5: Run focused and resource checks**

```bash
uv run pytest test/data/stores/test_topology_only_pyg_partitioner.py test/data/stores/test_typed_partition_book.py test/data/stores/test_partition_qualification.py -q
uv run python test/integration/qualify_pyg_partition_resources.py
```

**Step 6: Commit**

```bash
git add topobench/data/stores test/data/stores/test_topology_only_pyg_partitioner.py test/data/stores/test_typed_partition_book.py test/data/stores/test_partition_qualification.py test/integration/qualify_pyg_partition_resources.py
git commit -m "feat: qualify typed PyG partition books"
```

---

### Task 7: Finalize, distribute, and expose the immutable universal store

**Files:**

- Create: `topobench/data/stores/typed_graph_store.py`
- Create: `topobench/data/stores/pyg_store.py`
- Create: `topobench/data/stores/store_bundle.py`
- Create: `topobench/data/stores/qualification_checks.py`
- Modify: `topobench/data/stores/__init__.py`
- Create: `test/data/stores/test_typed_graph_store.py`
- Create: `test/data/stores/test_pyg_store.py`
- Create: `test/data/stores/test_typed_store_promotion.py`
- Create: `test/data/stores/test_prepartitioned_store_bundle.py`
- Create: `test/data/stores/test_qualification_checks.py`

**Step 1: Write failing store and PyG protocol tests**

Assert one physical store opens through homogeneous and heterogeneous cluster/
neighbor views; selected feature reads are bounded; relation CSC remains
memory-mapped; PyG `FeatureStore`/`GraphStore` methods return exact selected
slices/layout metadata; arbitrary type names map through collision-safe keys;
external IDs restore per type; and all split/partition identities round-trip.

Assert staging is invisible, atomic promotion creates only
`stores/<content-sha256>`, validated identity is a cache hit, changed input is a
miss, and corruption is quarantined. Package, download, safely extract, verify,
promote, move, and reopen a pre-partitioned bundle without pickle. Reject path
traversal, unknown versions, missing files, checksum/shape/dtype/CSC errors,
invalid phase IDs, partition inconsistency, stale identity, and incomplete
staging by exact check ID with evidence, remediation, and local report path.

**Step 2: Run the red tests**

```bash
uv run pytest test/data/stores/test_typed_graph_store.py test/data/stores/test_pyg_store.py test/data/stores/test_typed_store_promotion.py test/data/stores/test_prepartitioned_store_bundle.py test/data/stores/test_qualification_checks.py -q
```

**Step 3: Implement atomic writer and lazy protocol views**

Write stable `nNNNN`/`rNNNN` paths, typed partition/split directories, versioned
non-executable manifests, build environment, and qualification report. Build
under `stores/.staging/<id>`, validate all arrays/cross-references/checksums,
derive the content hash, and atomically rename on the same filesystem.
Implement process-local lazy maps, explicit close, safe archive containment,
digest-pinned download, and selected protocol reads without hidden full-layout
conversion.

Record Python/PyTorch/PyG/partition-backend versions, dependency-lock and source
state digests, OS/hardware/CUDA/container details when available, and the
partition/split fingerprints. Do not import DuckDB/PyArrow in the hot path
except explicit external-ID restoration.

**Step 4: Run focused and clean-import tests**

```bash
uv run pytest test/data/stores/test_typed_graph_store.py test/data/stores/test_pyg_store.py test/data/stores/test_typed_store_promotion.py test/data/stores/test_prepartitioned_store_bundle.py test/data/stores/test_qualification_checks.py test/architecture/test_domain_contract.py -q
```

**Step 5: Commit**

```bash
git add topobench/data/stores test/data/stores/test_typed_graph_store.py test/data/stores/test_pyg_store.py test/data/stores/test_typed_store_promotion.py test/data/stores/test_prepartitioned_store_bundle.py test/data/stores/test_qualification_checks.py
git commit -m "feat: expose immutable typed graph store"
```

---

### Task 8: Add materialized and disk cluster/neighbor strategies

**Files:**

- Create: `topobench/dataloader/disk_graph.py`
- Modify: `topobench/dataloader/graph.py`
- Modify: `topobench/dataloader/heterogeneous.py`
- Modify: `topobench/dataloader/__init__.py`
- Create: `test/data/dataload/test_disk_graph_datamodule.py`
- Create: `test/data/dataload/test_homogeneous_cluster_strategy.py`
- Create: `test/data/dataload/test_heterogeneous_cluster_strategy.py`
- Create: `test/data/dataload/test_heterogeneous_neighbor_strategy.py`
- Create: `test/data/dataload/test_disk_neighbor_parity.py`

**Step 1: Write failing exact cluster-oracle tests**

Select single, noncontiguous, and all-partition unions. For homogeneous `Data`
and typed `HeteroData`, assert exact selected canonical IDs, features, labels,
active-tag masks, induced directed relations/edges, edge fields, participant
counts, bounded reads, and deterministic descriptors. Compare every
heterogeneous materialized and disk union to
`HeteroData.subgraph(subset_dict)`.

**Step 2: Write hard neighbor-oracle tests**

Compare materialized and disk PyG `NeighborLoader` with identical target seed
order, per-relation fanout, hop count, replacement/direction settings, and RNG.
Require exact ordered target IDs, sampled nodes by type, edges by relation, hop
membership, fields, `batch_size`, supervision, and participants. Cover
asymmetric/zero fanout, isolated and high-degree seeds, explicit reverse
relations, duplicate endpoint pairs, final short batches, multi-worker
prefetch, reload, moved stores, and resume. Reject an incapable nondeterministic
backend instead of weakening assertions to set equality.

**Step 3: Run the red tests**

```bash
uv run pytest test/data/dataload/test_disk_graph_datamodule.py test/data/dataload/test_homogeneous_cluster_strategy.py test/data/dataload/test_heterogeneous_cluster_strategy.py test/data/dataload/test_heterogeneous_neighbor_strategy.py test/data/dataload/test_disk_neighbor_parity.py -q
```

**Step 4: Implement one strategy-driven data module**

Define a small `GraphSamplingStrategy` protocol for descriptors, native batch
materialization, phase setup, state, and capability validation. Implement
`HomogeneousClusterStrategy`, `HeterogeneousClusterStrategy`, and
`HeterogeneousNeighborStrategy` over materialized and disk views as supported.
Reuse the current graph/heterogeneous data-module public lifecycle and put
shared store ownership in one `DiskGraphDataModule`, not parallel frameworks.

PyG owns `HeteroData.subgraph` reference semantics and typed neighbor
traversal/filtering. TopoBench owns partition selection, active split, store
validation, descriptor order, transfer, lifecycle, and supervision.

**Step 5: Run focused and adjacent tests**

```bash
uv run pytest test/data/dataload/test_disk_graph_datamodule.py test/data/dataload/test_homogeneous_cluster_strategy.py test/data/dataload/test_heterogeneous_cluster_strategy.py test/data/dataload/test_heterogeneous_neighbor_strategy.py test/data/dataload/test_disk_neighbor_parity.py test/data/dataload/test_Dataloaders.py test/data/dataload/test_heterogeneous_dataloader.py -q
```

**Step 6: Commit**

```bash
git add topobench/dataloader test/data/dataload/test_disk_graph_datamodule.py test/data/dataload/test_homogeneous_cluster_strategy.py test/data/dataload/test_heterogeneous_cluster_strategy.py test/data/dataload/test_heterogeneous_neighbor_strategy.py test/data/dataload/test_disk_neighbor_parity.py
git commit -m "feat: add disk graph sampling strategies"
```

---

### Task 9: Fit and apply immutable training-only transforms

**Files:**

- Create: `topobench/transforms/fittable.py`
- Create: `topobench/transforms/incremental_pca.py`
- Modify: `topobench/dataloader/disk_graph.py`
- Create: `test/transforms/test_fittable_transform.py`
- Create: `test/transforms/test_incremental_pca.py`
- Create: `test/integration/test_fitted_transform_lifecycle.py`

**Step 1: Write failing leakage and state tests**

Assert a canonical fit view visits every active-tag training target exactly
once despite cluster context duplication, never reads validation/test rows,
uses bounded updates, and produces immutable JSON/array state. Cover
complete/partial splits, several tags, different feature dtypes, empty fitting
input, non-finite values, interrupted fit, state corruption, and every cache-key
component.

**Step 2: Write failing PCA application tests**

Fit a small known PCA fixture and assert the declared mean, components,
explained variance, projected width/dtype, deterministic projected batches, and
identical state reuse after process restart, store move, download, and training
resume. Numeric assertions use explicit dtype-derived tolerances.

**Step 3: Run the red tests**

```bash
uv run pytest test/transforms/test_fittable_transform.py test/transforms/test_incremental_pca.py test/integration/test_fitted_transform_lifecycle.py -q
```

**Step 4: Implement the protocol and bounded PCA**

Implement `begin_fit`, bounded `update_fit`, `finalize_fit`, and `transform`.
Key state by store, active split/train checksum, transform code/config, input
schema/dtype, implementation versions, and precision. Atomically publish
validated non-executable state and apply it after canonical batch assembly but
before pin/transfer. Label-consuming transforms require an explicit supervised
capability and can read training labels only.

**Step 5: Run focused tests and commit**

```bash
uv run pytest test/transforms/test_fittable_transform.py test/transforms/test_incremental_pca.py test/integration/test_fitted_transform_lifecycle.py -q
git add topobench/transforms topobench/dataloader/disk_graph.py test/transforms test/integration/test_fitted_transform_lifecycle.py
git commit -m "feat: fit bounded training-only graph transforms"
```

---

### Task 10: Make prefetch checkpoint state commit-safe

**Files:**

- Create: `topobench/dataloader/sequence_state.py`
- Modify: `topobench/dataloader/disk_graph.py`
- Modify: `topobench/model/model.py`
- Create: `topobench/callbacks/dataloader_commit.py`
- Create: `configs/callbacks/dataloader_commit.yaml`
- Create: `test/data/dataload/test_sequence_state.py`
- Create: `test/callbacks/test_dataloader_commit.py`

**Step 1: Write failing state-machine tests**

Cover monotonic issue/prepare/delivery, out-of-order worker completion with
ordered delivery, gradient accumulation, successful/skipped optimizer steps,
exceptions before and after backward, checkpoints before issue, after issue,
after collation, after backward, immediately before/after optimizer/global-step
commit, after evaluator update, and after a committed accumulation group.
Exercise homogeneous and heterogeneous partition descriptors plus typed target-
seed sampler descriptors.

Assert checkpoints contain store, partition-book, active split, fitted-
transform, committed cursor, RNG, evaluator, and strategy identities only.
Issued/prepared/consumed but uncommitted descriptors regenerate after resume;
validation/test never mutate training commit state.

**Step 2: Run the red tests**

```bash
uv run pytest test/data/dataload/test_sequence_state.py test/callbacks/test_dataloader_commit.py -q
```

**Step 3: Implement one sequence protocol and Lightning commit hook**

Keep issue/delivery in the data module and observe successful global-step
advance through one callback compatible with Lightning 2.4. Accumulate pending
IDs across gradient accumulation and commit the sampler cursor, evaluator
sequence/count, and model global step at one boundary only after the optimizer
advances. Implement versioned `state_dict`/`load_state_dict` with strict store,
partition, split, transform, strategy, and sequence validation.

**Step 4: Run focused tests**

```bash
uv run pytest test/data/dataload/test_sequence_state.py test/callbacks/test_dataloader_commit.py test/pipeline/test_pipeline.py test/pipeline/test_heterogeneous_pipeline.py -q
```

**Step 5: Commit**

```bash
git add topobench/dataloader/sequence_state.py topobench/dataloader/disk_graph.py topobench/model/model.py topobench/callbacks/dataloader_commit.py configs/callbacks/dataloader_commit.yaml test/data/dataload/test_sequence_state.py test/callbacks/test_dataloader_commit.py
git commit -m "feat: commit disk sampler state after optimizer steps"
```

---

### Task 11: Add bounded generic host and CUDA prefetch

**Files:**

- Create: `topobench/dataloader/device_prefetch.py`
- Modify: `topobench/dataloader/disk_graph.py`
- Create: `test/data/dataload/test_device_prefetch.py`
- Create: `test/integration/qualify_device_prefetch_cuda.py`

**Step 1: Write failing CPU lifecycle and budget tests**

For both `Data` and `HeteroData`, assert lazy worker opening, ordered output,
bounded host queue, writable pinned tensors, per-type/per-relation byte
estimates, explicit CPU/MPS host-only behavior, early shutdown, worker error
propagation, no leaked processes/maps, and rejection when a batch or configured
queue exceeds node/edge/host/device caps.

**Step 2: Write CUDA qualification harness**

Exercise depths 1 and 3, already-device-resident delivery to Lightning,
dedicated-stream event ordering, no second framework copy, bounded device
bytes, error cleanup, and profiler evidence that H2D overlaps model compute.
The harness fails explicitly without a CUDA runner; absence is not a pass.

**Step 3: Run CPU red tests**

```bash
uv run pytest test/data/dataload/test_device_prefetch.py -q
```

**Step 4: Implement generic transfer and ownership**

Recursively validate/pin/move native PyG stores without copying tensors twice.
Use one producer and a configurable CUDA-ready ring ordered by sequence ID.
Derive worst-case bytes from store field schemas, output view, and batch caps.
Make teardown idempotent after completion, exceptions, cancellation, or trainer
shutdown.

**Step 5: Run CPU and mandatory CUDA checks**

```bash
uv run pytest test/data/dataload/test_device_prefetch.py -q
uv run python test/integration/qualify_device_prefetch_cuda.py
```

**Step 6: Commit**

```bash
git add topobench/dataloader/device_prefetch.py topobench/dataloader/disk_graph.py test/data/dataload/test_device_prefetch.py test/integration/qualify_device_prefetch_cuda.py
git commit -m "feat: prefetch native graph batches to CUDA"
```

---

### Task 12: Add structured execution profiling and check reporting

**Files:**

- Create: `topobench/profiling/execution_events.py`
- Create: `topobench/profiling/local_event_log.py`
- Create: `topobench/dataloader/input_monitor.py`
- Modify: `topobench/data/stores/qualification_checks.py`
- Modify: `topobench/dataloader/device_prefetch.py`
- Modify: `topobench/dataloader/disk_graph.py`
- Modify: `topobench/model/model.py`
- Create: `topobench/callbacks/input_pipeline.py`
- Create: `configs/callbacks/input_pipeline.yaml`
- Create: `test/profiling/test_execution_events.py`
- Create: `test/profiling/test_local_event_log.py`
- Create: `test/data/dataload/test_input_monitor.py`
- Create: `test/callbacks/test_input_pipeline.py`

**Step 1: Write failing monitor tests**

Use a controllable clock and fake CUDA events. Assert stable operation/check
IDs; partition/conversion/validation/fitted-transform/read/assembly/H2D/
compute/checkpoint/artifact events; wall and monotonic time; counts/bytes;
RSS/pinned/GPU/temp-disk deltas; p50/p95/p99; queue state; active split and
descriptor identity; sampling cadence; rotation/bounds; and warn/error policy.

Assert local evidence is authoritative, W&B/logger output is bounded and
aggregated, raw IDs and secrets never appear, `system/*` cannot select a
checkpoint, no batch tensors are retained, no hot-path CUDA synchronize occurs,
and hard check failures propagate with evidence/remediation/report path.

**Step 2: Run the red tests**

```bash
uv run pytest test/profiling/test_execution_events.py test/profiling/test_local_event_log.py test/data/dataload/test_input_monitor.py test/callbacks/test_input_pipeline.py -q
```

**Step 3: Implement asynchronous monitoring**

Emit one schema-versioned structured stream from conversion through training,
resolve completed CUDA timings asynchronously, rotate the bounded local log,
and derive immutable summaries. Route sampled aggregates and failed-check
artifacts through existing logger adapters. Feed the existing run-provenance
contract; do not create another provenance serializer or unbounded W&B stream.

**Step 4: Run focused tests**

```bash
uv run pytest test/profiling/test_execution_events.py test/profiling/test_local_event_log.py test/data/dataload/test_input_monitor.py test/callbacks/test_input_pipeline.py test/callbacks/test_best_epoch_metrics.py -q
```

**Step 5: Commit**

```bash
git add topobench/profiling topobench/dataloader/input_monitor.py topobench/dataloader/device_prefetch.py topobench/dataloader/disk_graph.py topobench/data/stores/qualification_checks.py topobench/model/model.py topobench/callbacks/input_pipeline.py configs/callbacks/input_pipeline.yaml test/profiling test/data/dataload/test_input_monitor.py test/callbacks/test_input_pipeline.py
git commit -m "feat: record typed graph execution evidence"
```

---

### Task 13: Integrate the universal store through TopoBench pipelines

**Files:**

- Modify: `topobench/data/pipelines/default.py`
- Modify: `topobench/data/pipelines/heterogeneous.py`
- Modify: `topobench/data/pipelines/base.py`
- Modify: `topobench/data/pipelines/__init__.py`
- Modify: `configs/data_pipeline/default.yaml`
- Modify: `configs/data_pipeline/heterogeneous_node.yaml`
- Modify: `configs/dataset/graph/ParquetTypedGraph.yaml`
- Modify: `configs/dataset/heterogeneous/ParquetTypedGraph.yaml`
- Modify: `configs/model/graph/gcn.yaml`
- Modify: `configs/model/heterogeneous/hgt.yaml`
- Create: `test/pipeline/test_disk_graph_pipeline.py`
- Create: `test/pipeline/test_disk_heterogeneous_pipeline.py`
- Modify: `test/config/test_all_surviving_configs.py`

**Step 1: Write failing pipeline tests**

Compose materialized-reference and disk cluster/neighbor configs through Hydra
and ordinary `run()` setup. Assert standard data-module/spec, active split tag,
canonical prediction resolver, fitted-transform state, reproducibility policy,
profiling/check sinks, and provenance input. GCN receives native `Data`; HGT
receives native `HeteroData`; exact supervision counts own loss weighting; a
fitted transform runs once before training and its immutable transform runs
once per batch; no `run.py` type dispatch or alternate loop is added.

Negative tests cover output/model/backend/store mismatch, illegal split tag,
unqualified partition, stale transform state, and
`save_reproducibility_bundle: false` under a qualified profile.

**Step 2: Run the red tests**

```bash
uv run pytest test/pipeline/test_disk_graph_pipeline.py test/pipeline/test_disk_heterogeneous_pipeline.py test/config/test_all_surviving_configs.py -q
```

**Step 3: Implement thin pipeline adapters**

Teach existing graph and heterogeneous pipelines to recognize the validated
Parquet source descriptor and build the strategy-driven disk data module.
Reuse `DataPipelineOutput`, data specs, supervision adapters, models,
evaluator, callbacks, and `TBModel`. Keep configuration strategy-explicit and
store-capability validated.

Provide identity resolvers required by the selected-checkpoint artifact design:
homogeneous `(source_graph_id, global_nid)` and heterogeneous
`(source_graph_id, target_node_type, n_id)` with exact external-ID restoration.
Do not write prediction files in this task.

**Step 4: Run focused lifecycle smokes**

```bash
uv run pytest test/pipeline/test_disk_graph_pipeline.py test/pipeline/test_disk_heterogeneous_pipeline.py test/pipeline/test_pipeline.py test/pipeline/test_heterogeneous_pipeline.py -q
uv run python -m topobench dataset=graph/ParquetTypedGraph model=graph/gcn trainer.max_epochs=1
uv run python -m topobench dataset=heterogeneous/ParquetTypedGraph model=heterogeneous/hgt trainer.max_epochs=1
```

**Step 5: Commit**

```bash
git add topobench/data/pipelines configs/data_pipeline configs/dataset/graph/ParquetTypedGraph.yaml configs/dataset/heterogeneous/ParquetTypedGraph.yaml configs/model/graph/gcn.yaml configs/model/heterogeneous/hgt.yaml test/pipeline/test_disk_graph_pipeline.py test/pipeline/test_disk_heterogeneous_pipeline.py test/config/test_all_surviving_configs.py
git commit -m "feat: integrate typed disk graphs into TopoBench"
```

---

### Task 14: Qualify disk lifecycle, artifacts, and exact resume

**Files:**

- Create: `test/integration/test_typed_graph_conversion_resume.py`
- Create: `test/integration/test_graph_disk_resume.py`
- Create: `test/integration/test_heterogeneous_disk_resume.py`
- Create: `test/integration/test_typed_graph_lifecycle.py`
- Modify only for proven defects: store/data-module/callback files owned by
  Tasks 1-13

**Step 1: Write conversion crash/resume tests**

Interrupt after every staged boundary. Assert exact-source resume reuses only
checksum-validated stages; any source/schema/dependency/split/partition/
transform/output/strategy change invalidates staging; no partial final store is
visible; atomic promotion is idempotent; and concurrent builders respect the
lock. Exercise fresh build, cache hit, pre-partitioned download, moved store,
corruption quarantine, and a second clean process consuming the same bundle.

**Step 2: Write training interruption and reproducibility controls**

For homogeneous cluster, heterogeneous cluster, and heterogeneous neighbor
modes, compare uninterrupted deterministic runs with checkpoints at every
issue/prepare/consume/optimizer/evaluator/commit boundary. Assert exact
remaining descriptors, partition/seed IDs, sampler/evaluator counts and
sequence, store/partition/split/transform identity, selected checkpoint, and
prediction identities; model/optimizer/scheduler/final metrics meet the
declared bitwise or numeric-equivalence profile. Use production `ckpt_path` and
real Lightning hooks.

**Step 3: Run the red tests**

```bash
uv run pytest test/integration/test_typed_graph_conversion_resume.py test/integration/test_graph_disk_resume.py test/integration/test_heterogeneous_disk_resume.py test/integration/test_typed_graph_lifecycle.py -q
```

**Step 4: Fix only observed lifecycle gaps**

Do not add a second checkpoint callback. Repair the shared store, data module,
sequence state, or existing best-epoch callback at the source of any failing
invariant. Record exact tolerance where a qualified backend is not bitwise
identical.

**Step 5: Run focused and adjacent lifecycle tests**

```bash
uv run pytest test/integration/test_typed_graph_conversion_resume.py test/integration/test_graph_disk_resume.py test/integration/test_heterogeneous_disk_resume.py test/integration/test_typed_graph_lifecycle.py test/callbacks/test_best_epoch_metrics.py test/pipeline/test_pipeline.py test/pipeline/test_heterogeneous_pipeline.py -q
```

**Step 6: Commit**

```bash
git add test/integration/test_typed_graph_conversion_resume.py test/integration/test_graph_disk_resume.py test/integration/test_heterogeneous_disk_resume.py test/integration/test_typed_graph_lifecycle.py topobench
git commit -m "test: qualify typed disk graph lifecycle"
```

---

### Task 15: Prove bounded memory, reference parity, and CUDA overlap

**Files:**

- Create: `test/integration/qualify_typed_graph_rss.py`
- Create: `test/integration/qualify_typed_graph_cuda.py`
- Create: `test/integration/test_real_parquet_graph.py`
- Create: `test/integration/test_real_parquet_heterogeneous.py`
- Modify: `test/integration/test_retained_datasets.py`
- Modify: `.github/workflows/test.yml`
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Modify only after all gates pass: current README/docs sections owned by the
  parent remediation release task

**Step 1: Add subprocess RSS qualification**

Generate large homogeneous and heterogeneous Parquet fixtures whose mapped
edges plus feature arrays exceed the child RSS ceiling by a declared factor.
Assert bounded DuckDB memory/spill, no complete feature or mapped-edge table,
admitted topology-only partition RSS below its configured ceiling, selected
runtime reads, exact semantic round-trip, measured/final/temp sizes, and
validated content-addressed cache reuse.

**Step 2: Add mandatory CUDA qualification**

For GCN homogeneous cluster plus HGT heterogeneous cluster and neighbor modes,
compare synchronous, host-only, device depth 1, and device depth 3. Require
finite loss, exact canonical descriptor order, profiler-confirmed disk/CPU/H2D
overlap, bounded pinned/GPU bytes, and at most 5% steady-state input stall.
Missing CUDA/Parquet/partition/sampling backend, missing evidence, or threshold
breach fails the release job.

Add exact functional-oracle runs: selecting all partitions in one cluster union
and exhaustive relation fanout must reproduce full materialized target logits
and metrics under the declared numeric profile. For realistic sampled
training, run paired seeds and persist each result, mean, standard deviation,
confidence interval, paired difference, and a predeclared maximum degradation
against the same strategy's materialized reference; never choose tolerances
after seeing results.

**Step 3: Add download-gated real Parquet qualification**

Exercise representative real homogeneous and heterogeneous multi-file Parquet
sources through fresh conversion, several named split triplets, partition
production, fitted PCA, and native model steps in both heterogeneous
strategies. Validate semantic roles, relation directions, per-type features,
target supervision, external-ID restoration, artifact replay, and absence of
raw IDs in generic logs.

**Step 4: Run focused qualification**

```bash
uv run pytest test/integration/test_real_parquet_graph.py test/integration/test_real_parquet_heterogeneous.py test/integration/test_retained_datasets.py -q
uv run python test/integration/qualify_typed_graph_rss.py
uv run python test/integration/qualify_typed_graph_cuda.py
```

**Step 5: Wire optional dependencies and CI**

Pin DuckDB/PyArrow in the explicit `parquet` extra and lock file. Add separate
bounded-RSS, mandatory-CUDA, and live-data jobs. Persist only approved aggregate
qualification evidence. Provenance serialization remains owned by remediation
Task 30; provide it store/source fingerprints, schema roles, representation,
strategy state, queue budgets, timing quantiles, starvation, memory, and disk
summaries rather than creating another module.

**Step 6: Run the complete companion suite**

```bash
uv sync --frozen --all-extras
uv run pytest test/data/loaders/test_parquet_typed_schema.py test/data/stores test/data/dataload/test_disk_graph_datamodule.py test/data/dataload/test_homogeneous_cluster_strategy.py test/data/dataload/test_heterogeneous_cluster_strategy.py test/data/dataload/test_heterogeneous_neighbor_strategy.py test/data/dataload/test_disk_neighbor_parity.py test/transforms/test_fittable_transform.py test/transforms/test_incremental_pca.py test/data/dataload/test_sequence_state.py test/data/dataload/test_device_prefetch.py test/profiling test/data/dataload/test_input_monitor.py test/pipeline/test_disk_graph_pipeline.py test/pipeline/test_disk_heterogeneous_pipeline.py test/integration/test_fitted_transform_lifecycle.py test/integration/test_typed_graph_conversion_resume.py test/integration/test_graph_disk_resume.py test/integration/test_heterogeneous_disk_resume.py test/integration/test_typed_graph_lifecycle.py -q
uv run ruff check topobench test
```

**Step 7: Commit**

```bash
git add test/integration .github/workflows/test.yml pyproject.toml uv.lock README.md docs
git commit -m "test: qualify universal typed graph streaming"
```

---

## Completion criteria

This companion plan is complete only when all are true:

- one immutable typed store and accepted partition book serve homogeneous and
  heterogeneous materialized/disk cluster and heterogeneous neighbor views;
- Parquet conversion never materializes complete features or mapped edge
  tables; topology-only PyG partition generation obeys the independent 256 GiB
  default admission and reports measured RSS/temp disk;
- every named split triplet is explicit, internally unique/disjoint, coverage-
  checked, fingerprinted, and selectable without forbidding overlap across
  different tags;
- typed IDs, features, supervision, directed CSC, edge fields, partition maps,
  and inverse identities round-trip exactly;
- temporary METIS reverse arcs never leak; hard type/phase/relation/byte/size
  qualification gates promotion; the accepted map, not a rerun, defines
  reproducibility;
- materialized and disk cluster unions match `HeteroData.subgraph`; qualified
  disk neighbor batches match materialized `NeighborLoader` exactly in ordered
  canonical identities and fields;
- fitted PCA and other declared transforms fit each training entity once,
  never inspect validation/test, and replay from immutable checked state;
- pre-partitioned download verifies, safely stages, atomically promotes, moves,
  and reopens without executable artifacts;
- qualified runs save a complete reproducibility bundle by default and cannot
  disable it; fresh/cache/download/resume/second-process cases are tested;
- every hard check has a stable ID, structured local evidence, remediation, and
  bounded logger publication; profiling covers conversion through training;
- lazy ownership, writable batches, bounded queues, ordered delivery, teardown,
  committed-cursor resume, and evaluator/global-step agreement are proven;
- GCN and HGT complete finite steps; exact exhaustive oracles and predeclared
  paired-seed sampled-metric bounds pass;
- bounded conversion/partition RSS, mandatory CUDA overlap, at-most-5% strict
  stall, and real Parquet source gates pass with immutable evidence;
- TopoBench remains the only lifecycle and exposes canonical prediction/
  external-ID resolution to the selected-checkpoint artifact contract;
- no ordinary core import requires DuckDB/PyArrow, and no unsupported backend
  or unqualified artifact is silently accepted.
