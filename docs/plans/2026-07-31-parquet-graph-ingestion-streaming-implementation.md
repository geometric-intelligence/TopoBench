# Out-of-Core Parquet Graph Ingestion and GPU Streaming Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Use superpowers:test-driven-development for each behavior change, superpowers:systematic-debugging for unexpected failures, and superpowers:verification-before-completion before each commit and release claim.

**Goal:** Convert one large node/edge Parquet graph into a canonical disk-backed CSR store without materializing the graph or embedding matrix in RAM, then keep a single GPU supplied through bounded host/device prefetch with continuous input-starvation telemetry.

**Architecture:** A configurable Parquet schema feeds staged DuckDB and Arrow scans, a disk-backed external-ID index, temporary CSR construction, deterministic streaming Fennel partitioning, and an immutable partition-ordered array store. Worker processes assemble selected native PyG `Data` batches; an ordered CUDA ring prefetches three batches ahead by default, while an asynchronous monitor records and warns on input starvation without changing scientific metrics or runtime parameters.

**Tech Stack:** Python 3.11, PyTorch 2.3, PyTorch Geometric 2.8, Lightning 2.4, DuckDB, PyArrow, NumPy memory maps, Hydra/OmegaConf, pytest, Ruff, uv.

**Design:** `docs/plans/2026-07-31-parquet-graph-ingestion-streaming-design.md`

**Execution position:** Run this plan after Tasks 1–5 of `2026-07-31-research-production-remediation.md` and before its Task 9. This plan is the authoritative decomposition of remediation Tasks 6–8: Tasks 1–6 below complete remediation Task 6, Tasks 7–10 complete remediation Task 7, and Tasks 11–13 complete remediation Task 8.

---

## Execution rules

- Work only in the existing `graph-hetero-core-impl` worktree.
- Do not port the draft branch's rank-aware collator, `x_0`, `batch_0`, `cell_statistics`, SQLite cluster copies, batch lifting, or compatibility aliases.
- Keep all Parquet imports lazy outside the Parquet modules. A clean core import without the Parquet extra must still succeed.
- Keep `GraphDataModule` as the ordinary in-memory path. The new path is selected explicitly by `data_pipeline=graph_disk` plus the Parquet loader.
- Never materialize all node features, all mapped edges, a full `edge_index`, or a PyG `Data` during Parquet conversion.
- Never execute configuration-provided Python, SQL, or expressions. YAML maps declared semantic roles only.
- Every test fixture must be independently generated and contain no source-derived or confidential data.
- Run the existing heterogeneous sentinel tests after any change to shared loader, model, callback, or run lifecycle code.
- Do not run project-wide formatters, linters, or the full suite between tasks. The final remediation release task owns those gates.

---

### Task 1: Pin Parquet dependencies and validate semantic schemas

**Files:**

- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Create: `topobench/data/parquet_graph.py`
- Modify: `topobench/data/__init__.py`
- Create: `test/data/test_parquet_graph_spec.py`

**Step 1: Write failing schema tests**

Add tests for frozen `ParquetGraphSpec`, `NodeParquetSpec`, `EdgeParquetSpec`, `SupervisionSpec`, and `IngestionLimits`. Cover:

- node and edge path lists are non-empty strings;
- semantic roles name physical columns;
- features accept exactly one fixed-size list column or an ordered non-empty scalar-column list;
- node IDs support integral and UTF-8 string Arrow types but reject float, null, nested, and mixed types;
- labels may come from nodes or a separately keyed dataset;
- splits accept categorical, three masks, a separate dataset, or generated policy;
- positive batch rows, memory limit, partition count, and hard node/edge caps;
- unknown keys and contradictory states fail before any file opens;
- spec serialization is canonical and stable under input mapping order.

Use constructors rather than source-text assertions. Include one exact minimal valid spec:

```python
spec = ParquetGraphSpec.from_mapping(
    {
        "nodes": {
            "paths": ["nodes/*.parquet"],
            "columns": {
                "id": "node_id",
                "features": {
                    "column": "embedding",
                    "representation": "fixed_size_list",
                },
            },
        },
        "edges": {
            "paths": ["edges/*.parquet"],
            "columns": {"source": "src", "destination": "dst"},
        },
        "supervision": {
            "labels": {"source": "nodes", "column": "label"},
            "split": {
                "source": "nodes",
                "mode": "categorical",
                "column": "split",
                "values": {
                    "train": "train",
                    "val": "validation",
                    "test": "test",
                },
            },
        },
        "ingestion": {
            "record_batch_rows": 65_536,
            "memory_limit": "4GB",
            "num_partitions": 128,
            "partition_seed": 0,
        },
    }
)
assert spec.nodes.id_column == "node_id"
assert spec.canonical_digest == expected_digest
```

**Step 2: Run the red tests**

```bash
uv run pytest test/data/test_parquet_graph_spec.py -q
```

Expected: FAIL because the schema types do not exist.

**Step 3: Implement the narrow schema boundary**

Use frozen dataclasses and closed `Literal` unions. Normalize lists to tuples and paths to POSIX-style relative patterns without resolving them yet. Reject booleans wherever an integer is required. Parse byte-size strings once into positive byte counts.

Expose only the public schema types from `topobench.data`. Do not import DuckDB or PyArrow from `topobench/data/__init__.py`; physical schema inspection belongs behind lazy helper imports in `parquet_graph.py`.

**Step 4: Declare and lock the optional dependencies**

Add a `parquet` optional dependency group with direct, compatible pins for DuckDB and PyArrow. Regenerate the lock without upgrading unrelated packages:

```bash
uv lock
uv sync --frozen --extra parquet
```

Expected: both commands succeed and the lock includes direct project edges for the two packages.

**Step 5: Run focused and clean-import tests**

```bash
uv run pytest test/data/test_parquet_graph_spec.py test/architecture/verify_forbidden_imports.py -q
uv run python -c "import topobench; import topobench.data"
```

Expected: PASS both with the extra installed and in the later clean-core probe where the extra is unavailable.

**Step 6: Commit**

```bash
git add pyproject.toml uv.lock topobench/data/parquet_graph.py topobench/data/__init__.py test/data/test_parquet_graph_spec.py
git commit -m "feat: define parquet graph source contracts"
```

---

### Task 2: Inventory source files and build the external-node index

**Files:**

- Modify: `topobench/data/parquet_graph.py`
- Create: `topobench/data/graph_store.py`
- Create: `test/data/fixtures/__init__.py`
- Create: `test/data/fixtures/parquet_graph.py`
- Create: `test/data/test_parquet_graph_ingestion.py`

**Step 1: Create independent chunked fixtures**

Add a helper that writes equivalent node and edge datasets under different file and row-group layouts. Include:

- non-contiguous signed integer IDs;
- UTF-8 string IDs in a separate parametrized case;
- fixed-size float embeddings;
- shuffled physical row order;
- labels and split values;
- directed edges crossing community boundaries;
- optional edge weights and attributes.

The helper returns expected canonical rows from values constructed before Parquet serialization. Never derive expected output through the implementation under test.

**Step 2: Write failing inventory and mapping tests**

Assert that `ParquetGraphIngestor.inventory()`:

- resolves and sorts globs canonically;
- rejects empty globs, schema drift, changed ID types, missing columns, and source mutation;
- streams SHA-256 digests without reading a whole file in one call;
- estimates final and temporary disk requirements;
- includes source digests, schema, dependency versions, and ingestion limits in one canonical fingerprint.

Assert that `build_external_node_index()` creates an on-disk DuckDB table with exactly `(external_id, global_nid)`, where `global_nid` is dense `int64`, unique, deterministic, and independent of Parquet file/row-group layout. Reject duplicate and null external IDs.

**Step 3: Run the red tests**

```bash
uv run pytest test/data/test_parquet_graph_ingestion.py -q
```

Expected: FAIL because inventory, staging workspace, and node indexing are absent.

**Step 4: Implement locked staged ingestion**

Create a staging workspace whose identity is the canonical source/config fingerprint. Configure DuckDB's `memory_limit` and `temp_directory` before executing scans. Keep SQL static in source; bind paths and values as parameters or verified identifiers derived from the validated role mapping.

Assign global ordinals with an explicitly ordered query over the exact supported ID type. Write stage metadata and checksums only after a stage completes and fsync the metadata required to resume safely.

Expose stage methods rather than a generic query API:

```python
class ParquetGraphIngestor:
    def inventory(self) -> SourceInventory: ...
    def build_external_node_index(self) -> ExternalNodeIndex: ...
    def validate_stage(self, name: StageName) -> StageMetadata | None: ...
```

**Step 5: Prove bounded reads and deterministic mapping**

Instrument Python file reads and Arrow batch sizes. Assert no requested Python read exceeds the configured buffer and no Arrow batch exceeds `record_batch_rows`. Rebuild from alternate chunk layouts and compare node-index digests.

```bash
uv run pytest test/data/test_parquet_graph_ingestion.py -q
```

Expected: PASS.

**Step 6: Commit**

```bash
git add topobench/data/parquet_graph.py topobench/data/graph_store.py test/data/fixtures test/data/test_parquet_graph_ingestion.py
git commit -m "feat: index parquet graph nodes out of core"
```

---

### Task 3: Stream embeddings, labels, and split ownership

**Files:**

- Modify: `topobench/data/parquet_graph.py`
- Modify: `topobench/data/graph_store.py`
- Modify: `test/data/test_parquet_graph_ingestion.py`
- Create: `test/data/test_parquet_supervision.py`

**Step 1: Write failing node-field tests**

Parametrize:

- one fixed-size-list embedding column;
- explicit ordered scalar feature columns;
- labels in node rows;
- labels in a separate keyed Parquet dataset;
- categorical split values;
- three mask columns;
- separate keyed split data;
- existing deterministic generated split policy.

Assert exact row alignment by `global_nid`, stable dtype/shape, finite rank-2 `x`, integral rank-1 `y`, and full-length non-empty pairwise-disjoint boolean masks. Prove a shuffled supervision file cannot trigger a positional join.

Reject variable embedding widths, non-finite values, missing/duplicate/extra supervision IDs outside the declared policy, invalid labels, unknown split categories, overlapping masks, and empty phases.

**Step 2: Run the red tests**

```bash
uv run pytest test/data/test_parquet_supervision.py test/data/test_parquet_graph_ingestion.py -q
```

Expected: FAIL because node fields are not streamed.

**Step 3: Implement batch-wise field writers**

Count and validate shape before preallocating memory-mappable arrays. Consume Arrow record batches and write each output range without concatenating batches. For separate supervision, perform an external join through the node index and emit rows ordered by `global_nid`.

Keep static embeddings immutable. Do not register them as model parameters or include them in checkpoint payloads. Persist stage shape/dtype/checksum metadata for resume.

**Step 4: Verify memory behavior and split identity**

Run tests with total embedding bytes larger than a deliberately small Python allocation guard. Snapshot NumPy and Torch global RNG state around generated-split conversion.

```bash
uv run pytest test/data/test_parquet_supervision.py test/data/test_parquet_graph_ingestion.py -q
```

Expected: PASS; generated split conversion leaves global RNG unchanged.

**Step 5: Commit**

```bash
git add topobench/data/parquet_graph.py topobench/data/graph_store.py test/data/test_parquet_graph_ingestion.py test/data/test_parquet_supervision.py
git commit -m "feat: stream parquet node features and supervision"
```

---

### Task 4: Map edge endpoints and construct temporary CSR

**Files:**

- Modify: `topobench/data/parquet_graph.py`
- Modify: `topobench/data/graph_store.py`
- Create: `test/data/test_parquet_edge_ingestion.py`

**Step 1: Write failing edge-contract tests**

Use directed fixture edges with self-loops, duplicate endpoint pairs, cross-community edges, edge weights, and vector edge attributes. Assert:

- both endpoints map through the external-node index;
- mapped rows are ordered deterministically by source, destination, and declared stable duplicate order;
- `indptr`, `indices`, weights, and attributes reconstruct the exact declared edge multiset;
- alternate Parquet chunk/row-group layouts produce the same semantic CSR digest;
- no complete mapped edge table or PyG `edge_index` appears in Python memory.

Reject null IDs, missing endpoints, ID type mismatch, non-finite edge fields, inconsistent attribute width, and a configured duplicate/self-loop policy that the source violates.

**Step 2: Run the red tests**

```bash
uv run pytest test/data/test_parquet_edge_ingestion.py -q
```

Expected: FAIL because external edge joins and CSR writing are absent.

**Step 3: Implement external joins and ordered CSR emission**

Use two static DuckDB joins from edge endpoints to `global_nid`. Order through DuckDB's spillable execution. Stream ordered record batches into preallocated `indptr`/`indices` and aligned edge arrays; update row pointers incrementally.

Do not infer nodes from endpoints. Do not coalesce duplicates unless the schema declares a duplicate policy. Preserve original direction in the stored CSR.

**Step 4: Run focused tests**

```bash
uv run pytest test/data/test_parquet_edge_ingestion.py test/data/test_parquet_graph_ingestion.py -q
```

Expected: PASS with exact edge-field alignment.

**Step 5: Commit**

```bash
git add topobench/data/parquet_graph.py topobench/data/graph_store.py test/data/test_parquet_edge_ingestion.py
git commit -m "feat: build parquet graph csr out of core"
```

---

### Task 5: Partition CSR with deterministic bounded memory

**Files:**

- Create: `topobench/data/graph_partition.py`
- Modify: `topobench/data/__init__.py`
- Create: `test/data/test_graph_partition.py`
- Create: `test/data/verify_graph_partition_rss.py`

**Step 1: Write failing partition invariants**

Create community-structured directed graphs whose expected community cut is independent of production code. Assert that the partitioner:

- returns one assignment per node in `[0, num_partitions)`;
- honors the hard partition-size cap;
- is identical for the same seed and deterministic input;
- uses seed-controlled tie-breaking;
- treats direction as undirected only for scoring;
- does not mutate or reorder the original directed CSR;
- records balance, cut, and algorithm metadata;
- beats a declared deterministic hash baseline on the qualification fixture;
- rejects impossible balance constraints rather than falling back silently.

**Step 2: Run the red tests**

```bash
uv run pytest test/data/test_graph_partition.py -q
```

Expected: FAIL because the streaming partitioner does not exist.

**Step 3: Implement the initial Fennel-style partitioner**

Use a memory-mapped integer assignment array initialized to `-1`, sequential CSR row scans, bounded neighbor buffers, and $O(K)$ in-memory partition counts. Apply a documented Fennel score and hard balance cap. Use deterministic seed-based tie-breaking without a global RNG mutation.

Expose one closed algorithm selector whose only qualified initial value is `fennel_streaming`; reject unknown values. Never invoke PyG `ClusterData`, METIS, or a full adjacency materialization on this path.

**Step 4: Prove the RSS boundary in a subprocess**

The helper builds or reuses a disk CSR much larger than its allowed RSS margin, runs partitioning, reports peak RSS and output digests, and exits cleanly.

```bash
uv run pytest test/data/test_graph_partition.py::test_partition_subprocess_is_bounded_and_deterministic -q
```

Expected: PASS below the declared RSS ceiling.

**Step 5: Run focused tests**

```bash
uv run pytest test/data/test_graph_partition.py -q
```

Expected: PASS including balance and cut-quality evidence.

**Step 6: Commit**

```bash
git add topobench/data/graph_partition.py topobench/data/__init__.py test/data/test_graph_partition.py test/data/verify_graph_partition_rss.py
git commit -m "feat: partition disk csr with bounded memory"
```

---

### Task 6: Finalize the immutable partition-ordered graph store

**Files:**

- Modify: `topobench/data/graph_store.py`
- Modify: `topobench/data/parquet_graph.py`
- Modify: `topobench/data/__init__.py`
- Create: `test/data/test_graph_store.py`
- Create: `test/data/test_parquet_conversion_resume.py`

**Step 1: Write failing final-store tests**

Assert that conversion writes:

- canonical `manifest.json`;
- `partptr.npy`;
- partition-ordered CSR `indptr.npy` and `indices.npy`;
- `perm_to_global.npy` of canonical `int64` ordinals;
- `node_ids.parquet` mapping ordinals back to exact external IDs;
- aligned `x`, `y`, masks, weights, attributes, and declared fields.

Reload without source Parquet, DuckDB, temporary CSR, or a full `Data` reference. Reconstruct the exact semantic source graph and map predictions back to both integer and string external IDs.

Reject malformed version, missing arrays, wrong dtype/shape, non-monotone pointers, out-of-range columns, incomplete permutations, checksum mismatch, stale source/config fingerprints, and partially promoted directories.

**Step 2: Write failing staged-resume tests**

Interrupt after node index, node fields, temporary CSR, partitioning, and final rewrite. An exact rerun must reuse only complete validated stages. Any source byte, schema role, dependency version, memory-relevant conversion option, or partition parameter change must invalidate incompatible state. Source mutation during an active scan must fail.

**Step 3: Run the red tests**

```bash
uv run pytest test/data/test_graph_store.py test/data/test_parquet_conversion_resume.py -q
```

Expected: FAIL because final rewrite, atomic promotion, and stage resume are incomplete.

**Step 4: Implement partition-order rewrite and promotion**

Compute the inverse permutation on disk. Reorder nodes by `(partition_id, global_nid)`, remap edge endpoints, externally sort the final directed edges, and write aligned arrays sequentially. Build in a locked temporary sibling directory, fsync required files and directory entries, validate the full artifact, then atomically rename it into place.

The reader exposes validated metadata, lazy array opening, selected row/CSR reads, and external-ID export only. Keep `node_ids.parquet` off the batch hot path.

**Step 5: Add disk-space preflight**

Estimate source, DuckDB spill, temporary arrays, rewrite, and final-store peaks conservatively. Fail before conversion when available capacity is below the estimate plus an explicit margin. Do not delete source files or unrelated staging directories.

**Step 6: Run focused conversion tests**

```bash
uv run pytest test/data/test_graph_store.py test/data/test_parquet_conversion_resume.py test/data/test_parquet_graph_ingestion.py test/data/test_parquet_supervision.py test/data/test_parquet_edge_ingestion.py test/data/test_graph_partition.py -q
```

Expected: PASS. Equivalent Parquet layouts produce equal semantic store digests.

**Step 7: Commit**

```bash
git add topobench/data/graph_store.py topobench/data/parquet_graph.py topobench/data/__init__.py test/data/test_graph_store.py test/data/test_parquet_conversion_resume.py
git commit -m "feat: finalize immutable parquet graph stores"
```

---

### Task 7: Assemble exact selected cluster batches in workers

**Files:**

- Create: `topobench/dataloader/graph_disk.py`
- Modify: `topobench/dataloader/__init__.py`
- Modify: `topobench/transforms/data_transform.py`
- Modify: `topobench/transforms/data_manipulations/identity_transform.py`
- Modify: `topobench/transforms/data_manipulations/node_features_to_float.py`
- Modify: `topobench/transforms/data_manipulations/infere_knn_connectivity.py`
- Modify: `topobench/transforms/data_manipulations/infere_radius_connectivity.py`
- Create: `test/data/dataload/test_graph_disk.py`
- Create: `test/data/dataload/verify_graph_disk_workers.py`

**Step 1: Write failing selected-read and exact-union tests**

Select non-contiguous partitions from the Task 6 fixture. Assert exact `x`, `y`, masks, edge fields, canonical `global_nid`, directed `edge_index`, and every edge whose endpoints are both in the union, including cross-partition edges.

Instrument store methods and assert no unselected feature or CSR rows are read. Returned tensors must be writable and must not alias read-only arrays. Reject sampled unions above `max_batch_nodes` or `max_batch_edges` before pinning.

**Step 2: Write failing transform-boundary tests**

Add the immutable `BatchTransformSpec` contract from the parent remediation plan. Assert a counting transform runs once after identity/supervision fields exist and before pinning/device transfer. Reject stochastic, device-requiring, graph-to-hypergraph, node-count-changing, label/mask/global-ID-changing, or incompatible-width transforms.

Assert no `x_0`, `batch_0`, `cell_statistics`, `supervised_mask`, rank collator, lifting, or device transfer appears.

**Step 3: Run the red tests**

```bash
uv run pytest test/data/dataload/test_graph_disk.py -q
```

Expected: FAIL because the selected-cluster loader does not exist.

**Step 4: Implement CPU assembly**

Use a lightweight sequence-ID/partition-ID dataset. Each spawned worker opens store arrays lazily, sorts selected CSR ranges, copies selected node fields, filters destinations to the selected union, remaps endpoints locally, and applies the identical edge mask to every edge field.

Attach all phase masks and `perm_to_global[selected]` as `global_nid`, then invoke the optional qualified transform exactly once. Return CPU PyG `Data`; pinning and device movement stay outside the collator.

**Step 5: Prove spawn ownership and cleanup**

The subprocess helper uses two spawn workers, consumes full and early-stopped iterators, reports distinct lazy-open events, and terminates without children, inherited array handles, or pinned/device state.

```bash
uv run pytest test/data/dataload/test_graph_disk.py::test_spawn_workers_open_store_lazily_and_release -q
```

Expected: PASS.

**Step 6: Run focused and adjacent tests**

```bash
uv run pytest test/data/dataload/test_graph_disk.py test/data/dataload/test_dataload_dataset.py test/data/pipelines/test_data_pipelines.py -q
```

Expected: PASS.

**Step 7: Commit**

```bash
git add topobench/dataloader/graph_disk.py topobench/dataloader/__init__.py topobench/transforms/data_transform.py topobench/transforms/data_manipulations/identity_transform.py topobench/transforms/data_manipulations/node_features_to_float.py topobench/transforms/data_manipulations/infere_knn_connectivity.py topobench/transforms/data_manipulations/infere_radius_connectivity.py test/data/dataload/test_graph_disk.py test/data/dataload/verify_graph_disk_workers.py
git commit -m "feat: assemble native graph batches from disk"
```

---

### Task 8: Make prefetched sampler state checkpoint-safe

**Files:**

- Modify: `topobench/dataloader/graph_disk.py`
- Create: `test/data/dataload/test_graph_disk_sampler.py`
- Create: `test/integration/test_graph_disk_resume.py`

**Step 1: Write failing issued-versus-committed tests**

Run more partitions than clusters per batch and force workers to issue several future batches. Assert:

- every batch has a monotonic sequence ID and exact partition IDs;
- `issued_cursor` advances when work enters queues;
- `committed_cursor` advances only after an explicit successful-step acknowledgment;
- checkpoint state records the committed cursor and RNG state at that cursor;
- issued-but-uncommitted batches are regenerated after restore;
- worker timing does not change delivery order;
- train epochs change deterministically, while validation/test order remains fixed.

**Step 2: Run the red tests**

```bash
uv run pytest test/data/dataload/test_graph_disk_sampler.py test/integration/test_graph_disk_resume.py -q
```

Expected: FAIL because issued and committed state are not separated.

**Step 3: Implement transactional sampler state**

Keep deterministic batch descriptors reconstructible from epoch, seed, and cursor. Store committed state independently from worker queue state. Expose one `commit(sequence_id)` method that rejects gaps, duplicates, and out-of-order commits.

Data-module checkpoint state contains only immutable store/transform fingerprints plus the committed sampler state. Never pickle workers, iterators, queues, memory maps, or CUDA state.

**Step 4: Prove exact resume sequence**

Interrupt after work has been issued beyond the committed step. Restore in a fresh process and compare every subsequent batch descriptor with the uninterrupted control.

```bash
uv run pytest test/data/dataload/test_graph_disk_sampler.py test/integration/test_graph_disk_resume.py -q
```

Expected: PASS.

**Step 5: Commit**

```bash
git add topobench/dataloader/graph_disk.py test/data/dataload/test_graph_disk_sampler.py test/integration/test_graph_disk_resume.py
git commit -m "feat: checkpoint committed cluster sampler state"
```

---

### Task 9: Add bounded host and CUDA prefetch queues

**Files:**

- Create: `topobench/dataloader/prefetch.py`
- Modify: `topobench/dataloader/graph_disk.py`
- Modify: `topobench/dataloader/__init__.py`
- Create: `test/data/dataload/test_prefetch.py`
- Create: `test/data/dataload/verify_cuda_prefetch.py`

**Step 1: Write failing resource-bound tests**

Test normalized loader settings for positive `num_workers`, `prefetch_factor`, `device_prefetch_batches`, host/device budgets, and node/edge caps. Reject booleans, negative values, persistent workers with zero workers, and any conservative queue footprint above its budget.

Use field dtypes/shapes plus aggregate `BatchTransformSpec` to calculate maximum batch bytes. Assert the default large-graph profile resolves to three device-prefetched batches plus the currently computing batch.

**Step 2: Write failing ordered-ring tests**

With a fake asynchronous device adapter, delay copies out of order and assert the consumer still receives monotonically ordered sequence IDs. Cover:

- queue fill and drain;
- one, three, and larger configured depths;
- producer exception propagation;
- early shutdown and context-manager exit;
- buffer release after normal and exceptional completion;
- no second transfer for an already-device-resident batch.

**Step 3: Run the red CPU tests**

```bash
uv run pytest test/data/dataload/test_prefetch.py -q
```

Expected: FAIL because the bounded prefetch ring does not exist.

**Step 4: Implement the host/device pipeline**

Forward `prefetch_factor`, `persistent_workers`, and `pin_memory` to the ordered PyTorch loader. Implement `DevicePrefetchLoader` as a narrow extension of PyG's prefetch behavior:

- one dedicated CUDA copy stream per process/rank;
- configurable ordered ring depth;
- `Data.pin_memory()` followed by `Data.to(device, non_blocking=True)`;
- event/stream dependency before yielding;
- no extra copy when Lightning receives the resident batch;
- explicit host-only status on CPU/MPS;
- deterministic teardown and buffer release.

Do not call CUDA from workers. Do not place device transfer in `collate_fn`.

**Step 5: Run the CUDA subprocess when available**

The helper compares device depth 0, 1, and 3; verifies device placement, sequence order, dedicated-stream copy, and buffer release. It skips only outside the mandatory CUDA qualification job.

```bash
uv run pytest test/data/dataload/test_prefetch.py -q
uv run pytest test/data/dataload/verify_cuda_prefetch.py -q
```

Expected: CPU contract PASS; CUDA helper PASS on the qualified runner.

**Step 6: Commit**

```bash
git add topobench/dataloader/prefetch.py topobench/dataloader/graph_disk.py topobench/dataloader/__init__.py test/data/dataload/test_prefetch.py test/data/dataload/verify_cuda_prefetch.py
git commit -m "feat: prefetch graph batches onto cuda"
```

---

### Task 10: Measure and warn on GPU input starvation

**Files:**

- Modify: `topobench/dataloader/prefetch.py`
- Create: `topobench/callbacks/input_pipeline.py`
- Modify: `topobench/callbacks/__init__.py`
- Create: `configs/callbacks/input_pipeline_monitor.yaml`
- Create: `test/data/dataload/test_input_pipeline_monitor.py`
- Create: `test/callbacks/test_input_pipeline.py`

**Step 1: Write failing monitor-statistics tests**

Feed deterministic event samples and assert exact rolling:

- ready-batch wait;
- disk read and CPU assembly time;
- host/device queue depths;
- H2D and compute time;
- p50/p95/p99 summaries;
- starvation event count;
- input-stall fraction.

Exclude configured warm-up steps. Trigger warnings after three consecutive starved steps or two bad 100-step windows above 5%. `action: warn` must not abort; explicit `action: error` must raise after the same evidence boundary.

**Step 2: Write failing lifecycle and metric-isolation tests**

Assert:

- CUDA events are queried asynchronously after completion;
- no hot-path `torch.cuda.synchronize()` occurs;
- metrics use only `system/input/*` names;
- monitor metrics never enter evaluator, callback best metrics, checkpoint monitor selection, or returned scientific metrics;
- warning payload includes queue depths, I/O rate, H2D/compute timing, batch shape, sequence ID, and partition IDs;
- actual worker death, sequence gaps, queue corruption, invalid batch, and copy failure raise regardless of warning policy.

**Step 3: Run the red tests**

```bash
uv run pytest test/data/dataload/test_input_pipeline_monitor.py test/callbacks/test_input_pipeline.py -q
```

Expected: FAIL because the monitor and callback do not exist.

**Step 4: Implement asynchronous telemetry**

The loader records CPU monotonic timings and queue state. CUDA paths record copy events; the callback records compute events around the batch step. Resolve only completed event pairs and retain a bounded rolling sample window.

Log aggregated values at the configured interval through Lightning without adding them to callback-best state. Emit structured warnings through the established rank-zero logger. Never auto-change workers, queue depth, cluster count, or batch caps.

**Step 5: Run focused callback and lifecycle tests**

```bash
uv run pytest test/data/dataload/test_input_pipeline_monitor.py test/callbacks/test_input_pipeline.py test/callbacks/test_best_epoch_metrics.py test/pipeline/test_pipeline.py -q
```

Expected: PASS; best-epoch and return-metric dictionaries contain no `system/input/*` keys.

**Step 6: Commit**

```bash
git add topobench/dataloader/prefetch.py topobench/callbacks/input_pipeline.py topobench/callbacks/__init__.py configs/callbacks/input_pipeline_monitor.yaml test/data/dataload/test_input_pipeline_monitor.py test/callbacks/test_input_pipeline.py
git commit -m "feat: monitor graph input starvation"
```

---

### Task 11: Integrate the Parquet disk pipeline and configuration

**Files:**

- Create: `topobench/data/loaders/graph/parquet.py`
- Modify: `topobench/data/loaders/graph/__init__.py`
- Create: `topobench/data/pipelines/graph_disk.py`
- Modify: `topobench/data/pipelines/base.py`
- Modify: `topobench/data/pipelines/__init__.py`
- Create: `configs/data_pipeline/graph_disk.yaml`
- Create: `configs/dataset/graph/ParquetNodeGraph.yaml`
- Create: `configs/experiment/graph_parquet_disk_gcn.yaml`
- Modify: `topobench/data/capabilities.py`
- Modify: `topobench/model/supervision.py`
- Modify: `topobench/run.py`
- Create: `test/pipeline/test_graph_disk_pipeline.py`
- Modify: `test/config/test_surviving_graph_configs.py`

**Step 1: Write failing composition tests**

Compose the Parquet selector with explicit temporary node/edge paths and native GCN node classification. Assert YAML role mapping reaches `ParquetGraphSpec` exactly, `data_pipeline=graph_disk` is required, and empty default paths fail with a clear override message before a scan.

Reject graph-level, inductive, multi-graph, heterogeneous, hypergraph, incompatible model-channel, stochastic transform, invalid prefetch-budget, and unavailable Parquet-extra combinations at configuration preflight.

**Step 2: Write failing one-step pipeline test**

Build a small chunked Parquet fixture, convert it, drop all source/temporary handles, build the disk data module, and run one finite train/validation/test model step. Assert phase supervision uses only sliced canonical masks and prediction export maps `global_nid` back to external IDs.

The runtime transform must remain absent from store identity, present in run identity, and visible exactly once in model input.

**Step 3: Run the red tests**

```bash
uv run pytest test/pipeline/test_graph_disk_pipeline.py test/config/test_surviving_graph_configs.py -q
```

Expected: FAIL because the loader, pipeline, and configs are absent.

**Step 4: Implement the explicit pipeline**

The Parquet loader returns a source specification, not a `Data` object. The disk pipeline validates capabilities, runs/resumes conversion, builds the qualified CPU transform, resolves model input width, constructs the disk data module and prefetch settings, and exposes immutable typed metadata.

Do not pass Parquet input through the ordinary `PreProcessor`, `GraphDataModule`, or PyG `pre_transform`. Do not import the Parquet extra until this loader is instantiated.

Expose source/store digests, representation and partition versions, schema roles, partition seed/statistics, sampler committed state, transform digest, queue settings, and monitor thresholds for Task 30 provenance.

**Step 5: Run focused and adjacent tests**

```bash
uv run pytest test/pipeline/test_graph_disk_pipeline.py test/config/test_surviving_graph_configs.py test/pipeline/test_graph_model_capabilities.py test/data/pipelines/test_data_pipelines.py -q
```

Expected: PASS.

**Step 6: Commit**

```bash
git add topobench/data/loaders/graph/parquet.py topobench/data/loaders/graph/__init__.py topobench/data/pipelines/graph_disk.py topobench/data/pipelines/base.py topobench/data/pipelines/__init__.py configs/data_pipeline/graph_disk.yaml configs/dataset/graph/ParquetNodeGraph.yaml configs/experiment/graph_parquet_disk_gcn.yaml topobench/data/capabilities.py topobench/model/supervision.py topobench/run.py test/pipeline/test_graph_disk_pipeline.py test/config/test_surviving_graph_configs.py
git commit -m "feat: integrate parquet graph disk training"
```

---

### Task 12: Prove process lifecycle and exact checkpoint resume

**Files:**

- Modify: `test/integration/test_graph_disk_resume.py`
- Create: `test/integration/test_parquet_graph_lifecycle.py`
- Modify: `test/data/dataload/verify_graph_disk_workers.py`
- Modify: `test/data/dataload/verify_cuda_prefetch.py`
- Modify: `topobench/run.py`

**Step 1: Write the interrupted-versus-uninterrupted lifecycle test**

Train a deterministic fixture for several steps with two workers, host prefetch greater than one, and three device-prefetched batches when CUDA is present. Checkpoint after future batches have been issued but not committed. Resume in a fresh process.

Assert equality of:

- committed partition sequence;
- model and optimizer state;
- scheduler/global step/epoch;
- metric and callback state;
- selected checkpoint and rerun metrics;
- final external-ID prediction mapping.

Queued but uncommitted batches must be regenerated rather than skipped.

**Step 2: Exercise every shutdown path**

Cover full iteration, early iterator deletion, worker exception, CUDA-copy exception, trainer exception, and normal `fit`/`validate`/`test` teardown. Assert no child process, DuckDB connection, memory map, pinned tensor, CUDA event, or prefetched device batch remains reachable.

**Step 3: Run focused lifecycle tests**

```bash
uv run pytest test/integration/test_graph_disk_resume.py test/integration/test_parquet_graph_lifecycle.py -q
```

Expected: PASS on CPU; CUDA-specific cases PASS in the mandatory CUDA job.

**Step 4: Run adjacent shared-lifecycle sentinels**

```bash
uv run pytest test/pipeline/test_pipeline.py test/pipeline/test_heterogeneous_pipeline.py test/callbacks/test_best_epoch_metrics.py -q
```

Expected: PASS with no cross-domain callback, checkpoint, or loader regression.

**Step 5: Commit**

```bash
git add test/integration/test_graph_disk_resume.py test/integration/test_parquet_graph_lifecycle.py test/data/dataload/verify_graph_disk_workers.py test/data/dataload/verify_cuda_prefetch.py topobench/run.py
git commit -m "test: prove parquet graph lifecycle and resume"
```

---

### Task 13: Qualify out-of-core conversion and GPU overlap

**Files:**

- Create: `test/integration/test_real_parquet_graph.py`
- Create: `test/integration/qualify_parquet_graph_rss.py`
- Create: `test/integration/qualify_parquet_graph_cuda.py`
- Modify: `test/integration/test_retained_datasets.py`
- Modify: `.github/workflows/test.yml`
- Modify: `test/pipeline/test_graph_disk_pipeline.py`

**Step 1: Add a large synthetic out-of-core qualification**

Generate chunked node and edge Parquet whose uncompressed graph plus embedding bytes exceed the subprocess RSS ceiling by a substantial declared factor. Convert, reload, consume batches, and report:

- peak RSS;
- DuckDB memory limit and spill bytes;
- temporary disk peak;
- final-store bytes;
- conversion throughput;
- exact semantic output digest.

```bash
uv run pytest test/integration/test_real_parquet_graph.py::test_conversion_is_strictly_out_of_core -q
```

Expected: PASS below the RSS ceiling; no skip is accepted in the mandatory qualification job.

**Step 2: Add real-source qualification**

Use one approved real large transductive graph that can be represented as pinned node/edge Parquet fixtures or generated from an authenticated approved download. Verify source schema, build/reload behavior, selected reads, exact supervision, finite native GCN step, and external-ID export. Gate downloads only with `TOPOBENCH_ALLOW_DOWNLOADS=1`.

```bash
TOPOBENCH_ALLOW_DOWNLOADS=1 uv run pytest test/integration/test_real_parquet_graph.py::test_real_parquet_graph -q
```

Expected: PASS in the mandatory live-data job; a skip is not release evidence.

**Step 3: Add CUDA overlap qualification**

Compare synchronous transfer, host-only prefetch, device depth 1, and device depth 3 after warm-up. Capture profiler/CUDA-event evidence that:

- worker disk/assembly overlaps compute;
- H2D for a future batch overlaps current compute;
- device delivery order remains exact;
- the representative depth-3 profile has steady-state input-stall fraction at or below 5%;
- warnings fire when an injected storage delay exceeds the threshold;
- monitor collection does not synchronize the hot path.

```bash
uv run pytest test/integration/qualify_parquet_graph_cuda.py -q
```

Expected: PASS on the mandatory CUDA runner. Packaged runtime uses warning policy, but this qualification threshold is a release gate.

**Step 4: Prove the provenance handoff**

Extend the pipeline test to assert its immutable metadata exposes every value
later consumed by remediation Task 30: source/store fingerprints, resolved
schema roles, partition algorithm/options/seed and quality statistics,
dependency versions, committed sampler state, queue depths/budgets, monitor
thresholds, p50/p95/p99 timings, starvation count, peak
RSS/pinned/GPU/temp-disk bytes, and final-store size. Do not create a second
provenance module in this companion plan; Task 30 owns serialization.

**Step 5: Wire mandatory jobs**

Add an offline Parquet-extra job, bounded-RSS qualification, mandatory live-data gate, and CUDA qualification. Pin actions by full SHA under the parent remediation security task. A skip, missing extra, absent CUDA runner, threshold breach, or missing evidence fails its corresponding release job.

**Step 6: Run the complete focused Parquet suite**

```bash
uv run pytest test/data/test_parquet_graph_spec.py test/data/test_parquet_graph_ingestion.py test/data/test_parquet_supervision.py test/data/test_parquet_edge_ingestion.py test/data/test_graph_partition.py test/data/test_graph_store.py test/data/test_parquet_conversion_resume.py test/data/dataload/test_graph_disk.py test/data/dataload/test_graph_disk_sampler.py test/data/dataload/test_prefetch.py test/data/dataload/test_input_pipeline_monitor.py test/callbacks/test_input_pipeline.py test/pipeline/test_graph_disk_pipeline.py test/integration/test_graph_disk_resume.py test/integration/test_parquet_graph_lifecycle.py test/integration/test_real_parquet_graph.py -q
```

Expected: PASS. Run CUDA and download-marked commands separately in their mandatory environments.

**Step 7: Commit**

```bash
git add test/integration/test_real_parquet_graph.py test/integration/qualify_parquet_graph_rss.py test/integration/qualify_parquet_graph_cuda.py test/integration/test_retained_datasets.py test/pipeline/test_graph_disk_pipeline.py .github/workflows/test.yml
git commit -m "test: qualify parquet graph streaming"
```

---

## Completion criteria

This companion plan is complete only when all of the following are true:

- node, edge, label, and split Parquet schemas are mapped explicitly in YAML;
- arbitrary supported integer/string source IDs map deterministically to canonical `int64` ordinals and back;
- conversion never materializes the complete graph, feature matrix, mapped edge table, or PyG `Data` in RAM;
- DuckDB/Arrow work respects declared memory and temporary-disk boundaries;
- the streaming partitioner is deterministic, balanced, graph-aware, and meets its declared cut-quality gate without fallback;
- the final store is immutable, non-executable, checksum-validated, atomic, and independent of source file/row-group layout;
- workers read only selected graph rows, preserve all induced-union edges and aligned fields, and emit writable native PyG batches;
- qualified runtime transforms execute exactly once on CPU before pinning and device transfer;
- issued and committed sampler state produce exact interrupted/resumed batch sequences;
- host prefetch and a default three-batch-ahead CUDA ring are bounded by explicit byte budgets;
- every step records input wait, assembly, H2D, compute, and queue telemetry without hot-path synchronization;
- packaged runs warn on persistent input starvation, while CUDA release qualification fails above 5% steady-state input stall;
- monitor metrics remain outside scientific metrics, checkpoint selection, and returned result dictionaries;
- all process, memory-map, pinned-buffer, CUDA-event, and device-buffer resources are released after normal and exceptional teardown;
- mandatory bounded-RSS, real-data, CUDA-overlap, lifecycle, and resume gates pass with immutable evidence.
