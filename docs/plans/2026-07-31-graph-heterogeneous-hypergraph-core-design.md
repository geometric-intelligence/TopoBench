# Graph, Heterogeneous Graph, and Hypergraph Core Design

## Status

Approved on 2026-07-31 for the `topobench_graph_hetero` branch and amended
after two design/implementation-plan reviews on the same date.

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
```

The supported homogeneous task set is closed: graph-level binary or multiclass
classification, graph-level scalar regression, and single-graph transductive
node classification. Node regression, inductive node prediction over multiple
graphs, and multilabel graph classification are not supported in the first
reduced release. Their selectors, including `graph/US-county-demos`,
`graph/graphuniverse_inductive`, and `graph/ogbg-molpcba`, are removed rather
than left composition-only. The legacy `graph/manual_dataset` selector and its
custom input path are also removed from the focused product.

Each graph-level example stores a rank-one, length-one `y`. Classification
labels are integral; scalar-regression labels are floating point. PyG batching
therefore produces rank-one `[B]` labels. The supervision adapter preserves
classification targets as `[B]`, normalizes scalar-regression targets exactly
once to `[B, 1]`, and requires regression logits to have the same `[B, 1]`
shape. Node-classification labels are integral `[N]`. Other target ranks,
dtypes, non-finite regression targets, or prediction/target broadcasting fail
before loss or metric evaluation.

Inductive splits are index-backed dataset views, not materialized
`list[Data]` copies. Transductive `train_mask`, `val_mask`, and `test_mask`
are full-length boolean tensors. The split builder owns mask construction;
the supervision adapter only validates and consumes them.

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

The graph pipeline uses PyG `DataLoader` directly. Inductive datasets batch
multiple examples and use `batch` for graph-level pooling. Transductive mode
is explicit rather than inferred from missing validation/test datasets: it
requires exactly one source graph, `batch_size == 1`, and reuses that graph for
every phase while selecting supervision with the phase mask. The graph encoder
consumes `data.x`; wrappers consume `x`, `edge_index`, `batch`, and only the
optional edge fields permitted by the selected model capability.

Loss reductions are weighted by the number of supervised examples: graph
batches use their label count, transductive node batches use the selected mask
count, and heterogeneous sampled batches use the target seed count. Loss and
metric boundaries require exact prediction/target shapes and never rely on
PyTorch broadcasting.

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

## Configuration and Product Surface

The default `run.yaml` selects a small graph configuration rather than a
simplicial dataset and model. Config groups for removed domains and unsupported
task kinds are deleted. In particular, node-regression, inductive-node, and
multilabel graph selectors are absent. An exact dataset manifest, not directory
presence, defines the supported product. Model/dataset default-transform
resolvers are simplified to the three supported domains.

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
   transductive node-classification, and hypergraph fixtures.
3. Migrate homogeneous preprocessing, splitting, batching, encoders,
   wrappers, readouts, supervision, loss, and metrics to exact native
   `Data`/`Batch` contracts.
4. Prove that explicit transductive mode rejects multiple graphs and that
   scalar-regression prediction/target shapes cannot broadcast.
5. Migrate hypergraph data and both surviving models.
6. Qualify target and conditional graph models against task, batching, and
   edge-field capabilities.
7. Qualify every retained dataset selector through its manifest evidence,
   including one real scalar-regression lifecycle and each retained loader
   family; delete selectors that fail or cannot be exercised.
8. Replace dynamic discovery with explicit registries.
9. Delete unsupported neural, data, configuration, and documentation surfaces
   in separate reviewable commits. Remove the US County node-regression path
   rather than migrating its train-only feature/target standardization.
10. Remove dependencies and regenerate the lock file while preserving the
    heterogeneous sampling backend.
11. Run graph, heterogeneous, and hypergraph lifecycle tests, compatibility
    tests for surviving selector names, negative architecture tests, Ruff, and
    the broad network-free regression suite.

The final audit must prove that `topobench`, its default configuration, and
every selector in the surviving manifest import, compose, load, preprocess,
split, execute one compatible model batch, and produce finite loss/metrics in
the appropriate network-free or mandatory download-marked gate. It must prove
that unsupported task selectors are absent, scalar regression uses exact
`[B, 1]` prediction/target tensors, heterogeneous neighbor batching works after
a clean dependency sync, and representative real hypergraph raw formats work
before release. It must also prove that TopoModelX, TopoNetX, GUDHI, and
HyperNetX are unavailable to the clean process.

## Non-goals

- Checkpoint compatibility with rank-based graph or hypergraph modules.
- Compatibility stubs for deleted domains.
- Reimplementation of TopoModelX AllSet or UniGCN families.
- Hypergraph graph-level prediction in the first reduced release.
- Hypergraph neighbor sampling or distributed sampling.
- Heterogeneous link prediction or graph-level prediction.
- Renaming the Python package or CLI.
