# Graph, Heterogeneous Graph, and Hypergraph Core Design

## Status

Approved on 2026-07-31 for the `topobench_graph_hetero` branch and amended
after implementation-plan, audit, confidentiality, split-contract, and
disk-streaming reviews on the same date.

## Objective

Reduce TopoBench to three first-class, native PyTorch Geometric domains:

- homogeneous graphs;
- heterogeneous graphs;
- lightweight hypergraphs.

The surviving runtime must not depend on cell-rank representations or on the
removed topological domains. In particular, homogeneous graphs and
hypergraphs must stop passing through `DomainData`, rank-indexed feature
fields, and the custom topological collator.

This is an intentionally breaking branch for cell, simplicial,
combinatorial, point-cloud, and non-relational APIs. Surviving graph and
heterogeneous selector names remain stable where practical, but checkpoints
that depend on the old rank-based module structure are not compatibility
targets.

## Supported Domains

### Homogeneous graphs

Homogeneous graph datasets remain native PyG `Data` objects and are batched as
PyG `Batch` objects. The public structural field contract is:

```text
x
edge_index
edge_attr              optional, with explicit dataset/model handling
edge_weight            optional, with explicit dataset/model handling
y
batch                  for inductive mini-batches
train_mask             for transductive node tasks
val_mask
test_mask
global_nid             required for disk-sampled transductive batches
```

The supported homogeneous task set is closed: graph-level binary or multiclass
classification, graph-level scalar regression, single-graph transductive node
classification, and inductive node classification over explicitly separated
training, validation, and test graph datasets. Each inductive node phase may
contain one or more graphs and supervises all labeled nodes in its graphs.
Node regression and multilabel graph classification are not supported in the
first reduced release. Their selectors, including `graph/US-county-demos` and
`graph/ogbg-molpcba`, are removed rather than left composition-only. The legacy
`graph/graphuniverse_inductive` and `graph/manual_dataset` selectors remain
removed; supporting an explicit phase-dataset contract does not qualify those
legacy loaders automatically.

Each graph-level example stores a rank-one, length-one `y`. Classification
labels are integral; scalar-regression labels are floating point. PyG batching
therefore produces rank-one `[B]` labels. The supervision adapter preserves
classification targets as `[B]`, normalizes scalar-regression targets exactly
once to `[B, 1]`, and requires regression logits to have the same `[B, 1]`
shape. Node-classification labels are integral `[N]`. Other target ranks,
dtypes, non-finite regression targets, or prediction/target broadcasting fail
before loss or metric evaluation.

Generated and fixed inductive splits are index-backed dataset views, not
materialized `list[Data]` copies. Explicitly phase-separated inductive inputs
remain separate datasets. Transductive `train_mask`, `val_mask`, and
`test_mask` are full-length boolean tensors. The split builder owns mask
construction; the supervision adapter only validates and consumes them.

### Transductive execution modes

Transductive node classification has two explicit execution modes over the
same one-source-graph contract:

- `full_graph` reuses the one in-memory `Data` object for every phase and
  requires graph `batch_size == 1`;
- `cluster_disk` converts either one native in-memory source graph or one
  logical graph stored as chunked node/edge Parquet datasets into a versioned,
  partition-ordered global CSR representation, then reads only the selected
  cluster union for each step.

The native-source adapter may materialize its one graph during preprocessing.
The Parquet adapter is strictly out of core: schema roles are mapped in YAML;
DuckDB/Arrow scans, external ID joins and sorts, streaming graph partitioning,
and final array writes remain within declared memory and temporary-disk bounds.
Neither the complete graph nor the complete embedding matrix becomes a PyG
`Data` or an in-memory tensor. The authoritative detailed contract is
`docs/plans/2026-07-31-parquet-graph-ingestion-streaming-design.md`.

The manifest records source and split fingerprints, resolved semantic schema,
partition algorithm and seed, representation version, dependency versions,
tensor dtypes/shapes, checksums, and conversion resource limits. The stored
artifact contains CSR row pointers and column indices, cluster boundaries,
canonical numeric global-node permutation maps, a Parquet mapping back to
external integer/string node IDs, features, labels, all three phase masks, and
declared edge fields. It contains no transformed batches and uses
non-executable metadata.

Each loader worker opens arrays lazily. A step samples cluster IDs, copies only
their selected read-only slices into writable tensors, reconstructs the induced
union including edges between co-sampled clusters, and emits one native PyG
`Data`. The batch carries all sliced phase masks and canonical `global_nid`;
external source IDs are restored only at prediction-export/audit boundaries.
Validation and test cluster orders are deterministic. Training uses explicit
issued and committed sampler state so prefetched-but-uncommitted work is
regenerated after checkpoint resume.

### Split inputs and phase ownership

Every homogeneous graph loader returns exactly one explicit split input:

```text
UnsplitDataset(dataset)
IndexedDataset(dataset, SplitIndices(train, valid, test))
PhaseDatasets(train, valid, test)
```

`UnsplitDataset` requests an ordinary generated train/validation/test split.
Its configuration names `random` or `stratified`, declares positive
`train_prop`, `val_prop`, and `test_prop` values that sum to one, and declares
one split seed. Generation uses local random generators and never mutates
process-global PyTorch or NumPy RNG state. K-fold is not a qualified split
mode: cross-validation must not expose its held-out fold as an independent
test result.

`IndexedDataset` lets an adapter provide indices directly without serializing
them. For one node-task graph, the indices select nodes and are converted once
to canonical boolean masks. For a multi-graph dataset, the indices select
examples and become lazy phase views. The boundary validates integral rank-one
indices, bounds, non-empty phases, within-phase uniqueness, pairwise
disjointness, and complete coverage.

`PhaseDatasets` accepts already separated training, validation, and test
datasets. This is an inductive protocol. Graph-level tasks supervise each graph
as one example; node-level tasks supervise all labeled nodes in each phase
graph and do not require phase masks. One or many graphs per phase are valid.
The framework validates structural contracts and non-empty phases, while an
adapter remains responsible for domain-specific entity separation such as
patient, customer, scaffold, source-family, or temporal isolation.

Every retained dataset must satisfy an explicit input-feature policy before
model execution: floating features pass through, categorical integer features
are encoded, and featureless graphs receive deterministic constant or degree
features. Each dataset declares `edge_attr` and `edge_weight` availability as
absent, optional, or required; the selected model separately declares whether
it consumes, explicitly ignores, or rejects each field. A dataset is not
considered supported merely because its YAML composes.

### Heterogeneous graphs

The existing native `HeteroData` implementation remains the reference design.
It keeps its dedicated full-graph and `NeighborLoader` data module,
`HeterogeneousDataSpec`, metadata-driven models, seed-node supervision, fixed
sampled evaluation, and shared `TBModel` lifecycle.

Neighbor sampling is a required capability. The installed environment must
retain at least one PyG sampling backend (`torch-sparse` initially, or
`pyg-lib` after a separately tested migration); source-level import searches
are not sufficient evidence that the backend can be removed.

### Hypergraphs

Hypergraphs use a small native PyG `HypergraphData` subclass with this field
contract:

```text
x
hyperedge_index         shape [2, number_of_incident_pairs]
num_hyperedges
y
train_mask
val_mask
test_mask
```

The first row of `hyperedge_index` contains node indices and the second row
contains contiguous hyperedge indices. `HypergraphData.__inc__()` must offset
the two rows independently during batching: node indices by `num_nodes` and
hyperedge indices by `num_hyperedges`.

Empty hyperedges are rejected in the first reduced release because incidence
pairs cannot represent them. Batched `num_hyperedges` is per-example metadata;
models infer the total from the canonical contiguous incidence indices or sum
validated per-example counts explicitly.

Version 1 of the reduced branch supports hypergraph node classification. The
surviving models are the local EDGNN implementation and a PyG
`HypergraphConv` baseline. AllSet, AllSetTransformer, UniGCN, and UniGCNII are
removed with TopoModelX.

Hypergraph processed artifacts use a versioned cache filename/schema. Legacy
rank-based `data.pt` files are never silently interpreted as native
`HypergraphData`.

## Runtime Architecture

All domains share configuration, preprocessing, optimization, evaluation,
callbacks, logging, checkpointing, and best-checkpoint reruns. Data-specific
logic is selected through explicit data-pipeline configuration rather than
runtime type checks in `run.py`.

```text
loader
  -> shared representation-preserving preprocessor
  -> explicit data pipeline
       -> graph data module
            -> in-memory native PyG batching
            -> disk-backed transductive cluster sampling
                 -> native PyG Data batch
                 -> optional graph-to-graph batch transform
       -> heterogeneous node data module
       -> hypergraph node data module
  -> optional immutable runtime data specification
  -> domain feature encoder
  -> domain backbone and wrapper
  -> domain readout
  -> supervision adapter
  -> TBModel loss, metrics, optimizer, callbacks, logging, checkpoint reruns
```

### Graph pipeline

The graph pipeline uses native PyG objects in both storage modes. Inductive
graph-level datasets batch multiple examples through PyG `DataLoader` and use
`batch` for pooling. Inductive node-level phase datasets emit node logits and
supervise every labeled node in the current phase graphs without phase masks.

Transductive mode is explicit rather than inferred from missing
validation/test datasets and always requires exactly one source graph.
`full_graph` requires graph `batch_size == 1` and reuses that graph for every
phase. `cluster_disk` uses the versioned partition store and a dedicated native
collator; its cluster count per step is not graph-example `batch_size`. Both
modes expose the same `x`, `edge_index`, optional edge fields, `y`, and boolean
phase-mask semantics. Only sampled disk batches additionally require
`global_nid`.

The qualified large-graph profile overlaps lazy disk reads, worker-side CPU
assembly, pinned-host staging, asynchronous H2D copies, and model compute. Host
prefetch and an ordered CUDA-ready ring are independently byte-bounded; the
initial device depth is configurable and defaults to three batches ahead.
Every step records input wait, assembly, H2D, compute, and queue telemetry
without hot-path synchronization. Packaged runs warn on persistent starvation;
release qualification fails when representative steady-state input stall
exceeds the declared 5% boundary.

The graph encoder consumes `data.x`; wrappers consume `x`, `edge_index`,
`batch` when graph pooling requires it, and only optional edge fields permitted
by the selected model capability. Loss reductions are weighted by the number
of supervised examples: graph batches use their label count, transductive node
batches use the selected phase-mask count, and heterogeneous sampled batches
use the target seed count. Loss and metric boundaries require exact
prediction/target shapes and never rely on PyTorch broadcasting.

### Heterogeneous pipeline

The heterogeneous path remains separate from ordinary example batching. No
behavioral rewrite is planned beyond adapting shared interfaces that change
when the old homogeneous topological path is removed.

### Hypergraph pipeline

The hypergraph data module uses PyG batching over `HypergraphData`. Its
supervision adapter selects phase masks on node logits. Hypergraph models
consume `x` and `hyperedge_index`; no rank-one cell tensor or
`incidence_hyperedges` sparse matrix is part of the public runtime contract.

The pipeline creates or loads boolean phase masks from dataset split
parameters before validation. Synthetic fixtures exercise the same production
factory, while release validation additionally covers one real pickle-format
and one real content/edges-format dataset.

## Capability Matrix

Configuration composition and runtime support are separate claims. An explicit
surviving-dataset manifest is the product boundary. Every entry records the
selector, loader family, feature policy, edge-feature policy, task kind, task
level, split mode, required transform, compatible models, and named
qualification evidence. Every surviving model declares task kinds, task
levels, batching modes, and edge-field handling. Architecture tests require
every surviving dataset to have at least one valid model pairing and reject
selectors absent from the manifest.

GCN, GAT, GIN, GPS, and NSD are target graph models. GraphMLP and GCN-DGM are
conditional: they remain only if dedicated native-batching lifecycle tests
prove their intended task modes. Otherwise their source, configs, registry
entries, capability rows, and tests are removed together.

Every retained selector must pass a selector-specific config/metadata check and
cite executable loader, feature, split, forward, loss, and metric evidence.
Selectors may share a loader-family integration test only when a
selector-specific assertion proves that they use the same parser and contract.
Download-marked qualification may run outside ordinary CI, but it is a
mandatory release gate; a skipped or unavailable result is not evidence of
support.

## Explicit Registries

Filesystem-wide dynamic discovery is removed from loaders, backbones,
wrappers, encoders, readouts, and transforms. Each surviving package exports
an explicit, deterministic registry. This prevents deleted modules and
optional dependencies from being imported as a side effect of importing
`topobench`.

The only registered domains are `graph`, `heterogeneous`, and `hypergraph`.
Selecting a removed config group fails during Hydra composition because the
config no longer exists. The branch does not retain deprecation shims for
removed topological APIs.

## Transform Boundary

All topology liftings and feature liftings are removed. The remaining
transform registry contains only operations that accept at least one
supported native representation and declare that compatibility explicitly.

The branch retains:

- graph-native feature, positional, and structural encodings that do not
  import removed dependencies;
- general PyG data manipulations used by graph datasets;
- `HeterogeneousConstantFeatures` and `HeterogeneousToUndirected`;
- any narrowly required hypergraph normalization transform implemented
  without TopoModelX, TopoNetX, or HyperNetX.

HOPSE, SANN, cell-rank encodings, barycentric subdivision, simplicial
curvature, and other higher-order-only transforms are removed.

The disk-backed transductive graph path may apply one optional
`batch_transform` after the sampled induced graph has been assembled and before
Lightning transfers it to the model device. This name is intentionally
different from PyG `pre_transform`, which denotes once-before-storage work.
The runtime transform accepts and returns native homogeneous `Data`; graph to
hypergraph or rank-indexed output is unsupported. Eligibility is explicit:
each qualified transform exposes an immutable `BatchTransformSpec` declaring
graph input/output, determinism, node-identity preservation, and either
feature-width preservation or one fixed output width. Missing metadata is a
rejection, not an inferred capability.

The transform may change graph-native features or edges, but it must preserve
node count, labels, all phase masks, and `global_nid`. These identity fields are
attached before invocation and validated afterward. The transform runs exactly
once per batch, never mutates the immutable store, and is not cached. Qualified
runtime transforms are deterministic; stochastic augmentations remain
experimental until their RNG and resume state have an explicit lifecycle
contract.

## Dependency Boundary

The core project removes direct dependencies that exist only for deleted
domains, including:

- TopoModelX;
- TopoNetX;
- GUDHI when no surviving graph transform requires it;
- HyperNetX;
- trimesh and spharapy when no surviving graph loader requires them.

The lock file is regenerated after the source and configuration audit. A
clean-process test blocks imports of removed libraries and verifies that all
surviving configs compose and instantiate without them.

`torch-sparse` remains while heterogeneous `NeighborLoader` depends on it.
The dependency audit includes runtime capability probes, not only static
searches.

DuckDB and PyArrow are pinned direct dependencies of an explicit Parquet
extra. They are imported lazily: ordinary graph, heterogeneous, and hypergraph
imports and runs remain independent of the extra. Parquet conversion and its
mandatory bounded-memory qualification install it explicitly.

## Configuration and Product Surface

The default `run.yaml` selects a small graph configuration rather than a
simplicial dataset and model. Config groups for removed domains and unsupported
task kinds are deleted. In particular, node-regression and multilabel graph
selectors are absent. Inductive node classification is available through the
explicit phase-dataset contract; legacy selectors remain absent until they
independently satisfy the qualification boundary. An exact dataset manifest,
not directory presence, defines the supported product. Model/dataset
default-transform resolvers are simplified to the three supported domains.

The README and documentation describe TopoBench on this branch as a focused
graph, heterogeneous-graph, and hypergraph benchmarking framework. Historical
topological scripts, tutorials, examples, and generated documentation are
removed when they expose unsupported commands. License and attribution files
remain unchanged.

## Error Handling

Failures must occur at the narrowest boundary:

- invalid graph batches fail before backbone, loss, or metric execution for
  missing fields, unsupported task/split combinations, non-singleton
  transductive datasets, incompatible edge-field handling, or target
  dtype/shape mismatches;
- invalid disk-backed graph artifacts fail before loader construction for
  unsupported schema versions, incomplete or inconsistent arrays, invalid CSR
  bounds, bad permutations, checksum mismatch, or split/partition identity
  mismatch;
- invalid sampled batches fail before model execution for cross-batch node
  references, missing true global IDs, empty phase supervision, or a runtime
  transform that changes node or supervision identity;
- invalid heterogeneous graphs continue to fail through
  `HeterogeneousDataSpec`;
- invalid hypergraphs fail validation for malformed incidence pairs,
  non-contiguous or out-of-bounds hyperedge indices, feature/label mismatch,
  invalid masks, or missing `num_hyperedges`;
- references to removed Hydra groups fail at composition;
- importing the core package must never fail because a deleted topological
  dependency is unavailable.

## Testing Strategy

The migration is test-driven and proceeds from new contracts to deletion.

1. Add architecture tests defining the allowed domains, exact surviving
   dataset manifest, package/config trees, dependencies, supported task kinds,
   and forbidden rank-based fields.
2. Establish native synthetic graph classification, scalar-regression,
   transductive node-classification, phase-separated inductive
   node-classification, and hypergraph fixtures. All committed fixtures are
   independently invented and contain no confidential or data-derived values.
3. Qualify generated three-way splits, direct node/example indices, and
   separately loaded phase datasets through one normalization boundary.
4. Migrate homogeneous preprocessing, splitting, batching, encoders,
   wrappers, readouts, supervision, loss, and metrics to exact native
   `Data`/`Batch` contracts.
5. Prove that explicit transductive mode rejects multiple source graphs, all
   split modes reject overlap or incomplete coverage where applicable, and
   scalar-regression prediction/target shapes cannot broadcast.
6. Qualify both native-source and out-of-core Parquet disk-backed transductive
   execution: schema mapping, arbitrary external IDs, bounded conversion RSS
   and temporary disk, deterministic graph-aware partitioning, cache
   miss/hit/resume equivalence, selected-read instrumentation, cross-cluster
   edges, canonical global IDs plus external-ID export, writable tensors,
   deterministic committed sampler resume, multi-worker lazy opening,
   exactly-once graph-to-graph transforms, bounded host/CUDA prefetch, and
   continuous input-starvation telemetry.
7. Migrate hypergraph data and both surviving models.
8. Qualify target and conditional graph models against task, batching, and
   edge-field capabilities.
9. Qualify every retained dataset selector through its manifest evidence,
   including one real scalar-regression lifecycle and each retained loader
   family; delete selectors that fail or cannot be exercised.
10. Replace dynamic discovery with explicit registries.
11. Delete unsupported neural, data, configuration, and documentation surfaces
    in separate reviewable commits. Remove the US County node-regression path
    rather than migrating its train-only feature/target standardization.
12. Remove dependencies and regenerate the lock file while preserving the
    heterogeneous sampling backend.
13. Run graph, heterogeneous, and hypergraph lifecycle tests, compatibility
    tests for surviving selector names, negative architecture tests, Ruff, and
    the broad network-free regression suite.

The final audit must prove that `topobench`, its default configuration, and
every selector in the surviving manifest import, compose, load, preprocess,
split, execute one compatible model batch, and produce finite loss/metrics in
the appropriate network-free or mandatory download-marked gate. It must prove
ordinary generated three-way splitting, direct fixed node/example indices,
phase-separated inductive node graphs, exact `[B, 1]` scalar-regression
prediction/target tensors, disk-backed transductive cluster sampling from both
native and chunked Parquet sources, bounded conversion and selected reads, an
exactly-once graph-to-graph batch transform, measured CUDA overlap with at most
5% qualified steady-state input stall, heterogeneous neighbor batching after a
clean dependency sync, and representative real hypergraph raw formats before
release. It must also prove
that unsupported task selectors are absent and that TopoModelX, TopoNetX,
GUDHI, and HyperNetX are unavailable to the clean process.

Confidential adapters are qualified only inside the approved data boundary.
Repository tests use independently generated artificial data; they never
commit sampled, transformed, anonymized, or statistically matched records.
Secure qualification validates real data and repeatability in memory, emits
only approved non-sensitive status, and does not upload raw data, split
indices, tensors, embeddings, predictions, caches, or checkpoints.

## Non-goals

- Checkpoint compatibility with rank-based graph or hypergraph modules.
- Compatibility stubs for deleted domains.
- Reimplementation of TopoModelX AllSet or UniGCN families.
- Hypergraph graph-level prediction in the first reduced release.
- Hypergraph neighbor sampling or distributed sampling.
- Heterogeneous link prediction or graph-level prediction.
- Multi-graph disk batching, remote object-store memory mapping, distributed
  Cluster-GCN/conversion, mutable Parquet embeddings, arbitrary
  configuration-provided SQL/Python expressions, and stochastic qualified
  runtime transforms.
- Renaming the Python package or CLI.
