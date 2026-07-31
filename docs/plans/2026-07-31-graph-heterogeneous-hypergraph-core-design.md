# Graph, Heterogeneous Graph, and Hypergraph Core Design

## Status

Approved on 2026-07-31 for the `topobench_graph_hetero` branch and amended
after the implementation-plan review on the same date.

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
PyG `Batch` objects. The public field contract is:

```text
x
edge_index
edge_attr              optional
y
batch                  for inductive mini-batches
train_mask             for transductive node tasks
val_mask
test_mask
```

Both graph-level classification/regression and single-graph node
classification remain supported.

Inductive splits are index-backed dataset views, not materialized
`list[Data]` copies. Transductive `train_mask`, `val_mask`, and `test_mask`
are full-length boolean tensors. The split builder owns mask construction;
the supervision adapter only validates and consumes them.

Every retained dataset must satisfy an explicit input-feature policy before
model execution: floating features pass through, categorical integer features
are encoded, and featureless graphs receive deterministic constant or degree
features. A dataset is not considered supported merely because its YAML
composes.

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
multiple examples and use `batch` for graph-level pooling. Transductive node
datasets expose the same graph to every phase and select supervision with the
phase mask. The graph encoder consumes `data.x`; wrappers consume `x`,
`edge_index`, optional edge features, and `batch`.

Loss reductions are weighted by the number of supervised examples: graph
batches use their label count, transductive node batches use the selected mask
count, and heterogeneous sampled batches use the target seed count.

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

Configuration composition and runtime support are separate claims. Every
surviving dataset declares feature type, task level, split type, and required
transform. Every surviving model declares supported task levels and batching
modes.

GCN, GAT, GIN, GPS, and NSD are target graph models. GraphMLP and GCN-DGM are
conditional: they remain only if dedicated native-batching lifecycle tests
prove their intended task modes. Otherwise their code and configs are removed
rather than advertised as nominally supported.

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
simplicial dataset and model. Config groups for removed datasets, models,
liftings, and experiments are deleted. Model/dataset default-transform
resolvers are simplified to the three supported domains.

The README and documentation describe TopoBench on this branch as a focused
graph, heterogeneous-graph, and hypergraph benchmarking framework. Historical
topological scripts, tutorials, examples, and generated documentation are
removed when they expose unsupported commands. License and attribution files
remain unchanged.

## Error Handling

Failures must occur at the narrowest boundary:

- invalid graph batches fail with missing-field or shape errors in the graph
  encoder/wrapper;
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

1. Add architecture tests defining the allowed domains, package tree, config
   tree, dependencies, and forbidden rank-based fields.
2. Establish native synthetic graph and hypergraph fixtures.
3. Migrate homogeneous preprocessing, splitting, batching, encoders,
   wrappers, readouts, and supervision to `Data`/`Batch`.
4. Migrate hypergraph data and both surviving models.
5. Qualify conditional graph models against the capability matrix.
6. Replace dynamic discovery with explicit registries.
7. Delete unsupported neural, data, configuration, and documentation surfaces
   in separate reviewable commits.
8. Remove dependencies and regenerate the lock file while preserving the
   heterogeneous sampling backend.
9. Run graph, heterogeneous, and hypergraph lifecycle tests, compatibility
   tests for surviving selector names, negative architecture tests, Ruff, and
   the broad network-free regression suite.

The final audit must prove that `topobench`, its default configuration, and
all surviving config groups import and compose in a clean process where
TopoModelX, TopoNetX, GUDHI, and HyperNetX are unavailable.
It must also prove heterogeneous neighbor batching after a clean dependency
sync and exercise representative real hypergraph raw formats before release.

## Non-goals

- Checkpoint compatibility with rank-based graph or hypergraph modules.
- Compatibility stubs for deleted domains.
- Reimplementation of TopoModelX AllSet or UniGCN families.
- Hypergraph graph-level prediction in the first reduced release.
- Hypergraph neighbor sampling or distributed sampling.
- Heterogeneous link prediction or graph-level prediction.
- Renaming the Python package or CLI.
