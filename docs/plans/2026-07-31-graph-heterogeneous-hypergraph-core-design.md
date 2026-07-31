# Graph, Heterogeneous Graph, and Hypergraph Core Design

## Status

Approved on 2026-07-31 for the `topobench_graph_hetero` branch.

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

### Heterogeneous graphs

The existing native `HeteroData` implementation remains the reference design.
It keeps its dedicated full-graph and `NeighborLoader` data module,
`HeterogeneousDataSpec`, metadata-driven models, seed-node supervision, fixed
sampled evaluation, and shared `TBModel` lifecycle.

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

Version 1 of the reduced branch supports hypergraph node classification. The
surviving models are the local EDGNN implementation and a PyG
`HypergraphConv` baseline. AllSet, AllSetTransformer, UniGCN, and UniGCNII are
removed with TopoModelX.

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

### Heterogeneous pipeline

The heterogeneous path remains separate from ordinary example batching. No
behavioral rewrite is planned beyond adapting shared interfaces that change
when the old homogeneous topological path is removed.

### Hypergraph pipeline

The hypergraph data module uses PyG batching over `HypergraphData`. Its
supervision adapter selects phase masks on node logits. Hypergraph models
consume `x` and `hyperedge_index`; no rank-one cell tensor or
`incidence_hyperedges` sparse matrix is part of the public runtime contract.

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
5. Replace dynamic discovery with explicit registries.
6. Delete unsupported source, configs, tests, scripts, tutorials, and docs.
7. Remove dependencies and regenerate the lock file.
8. Run graph, heterogeneous, and hypergraph lifecycle tests, compatibility
   tests for surviving selector names, negative architecture tests, Ruff, and
   the broad network-free regression suite.

The final audit must prove that `topobench`, its default configuration, and
all surviving config groups import and compose in a clean process where
TopoModelX, TopoNetX, GUDHI, and HyperNetX are unavailable.

## Non-goals

- Checkpoint compatibility with rank-based graph or hypergraph modules.
- Compatibility stubs for deleted domains.
- Reimplementation of TopoModelX AllSet or UniGCN families.
- Hypergraph graph-level prediction in the first reduced release.
- Hypergraph neighbor sampling or distributed sampling.
- Heterogeneous link prediction or graph-level prediction.
- Renaming the Python package or CLI.
