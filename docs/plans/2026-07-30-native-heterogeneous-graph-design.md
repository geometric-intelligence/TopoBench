# Native Heterogeneous Graph Support Design

## Status

Approved on 2026-07-30. Implementation is staged through synthetic support,
DBLP integration, and OGB-MAG readiness configurations; real-data gates remain
explicitly opt-in. The public configuration and extension contract is
maintained in the
[native heterogeneous graph guide](../heterogeneous_graphs.md); this document
retains architectural rationale and scope decisions.

## Objective

Introduce native heterogeneous node-classification support into TopoBench.
PyTorch Geometric `HeteroData` must remain native from dataset loading through
model execution. HGT is the primary heterogeneous model and HeteroSAGE is the
second model used to prove that the new data infrastructure is not
HGT-specific.

Development is deliberately staged:

1. deterministic synthetic heterogeneous data;
2. full-batch DBLP;
3. neighbor-sampled OGB-MAG.

## Non-goals

The first version does not include:

- heterogeneous link prediction;
- heterogeneous graph-level prediction;
- arbitrary generated train/validation/test splits;
- distributed sampling;
- HGT-specific sampling with `HGTLoader`;
- automatic conversion of every existing TopoBench transform to
  `HeteroData`;
- flattening heterogeneous graphs into `DomainData`;
- a universal rewrite of every TopoBench data domain.

The existing cell-complex HGT remains supported. It is an adapter from
rank-indexed cell-complex data to typed dictionaries, not a native
heterogeneous dataset pipeline.

## Existing Constraints

TopoBench currently assumes homogeneous PyG `Data` at several boundaries:

- `PreProcessor.process()` asserts that its collated output is `Data`;
- transductive split functions read top-level `x`, `y`, and masks;
- `DataloadDataset` converts samples to flat key/value lists;
- `collate_fn` rebuilds samples as `DomainData`;
- `TBDataloader` always uses that custom collation;
- `TBModel.process_outputs()` reads top-level masks;
- model configuration is instantiated before any runtime heterogeneous
  metadata is injected.

The file `topobench/data/loaders/graph/hetero_datasets.py` loads
*heterophilous* homogeneous graphs. It is not native heterogeneous-graph
support and should not be reused or renamed for this feature.

## Architectural Decision

Use a native heterogeneous path under TopoBench's shared training shell.

```text
dataset loader
    -> shared transform/cache preprocessor
    -> configured data pipeline
        -> default TopoBench split + TBDataloader
        -> heterogeneous mask validation + HeterogeneousNodeDataModule
    -> runtime data specification
    -> typed feature encoder
    -> HGTBackbone or HeteroSAGEBackbone
    -> heterogeneous wrapper
    -> target-node readout
    -> supervision adapter
    -> shared loss, evaluator, optimizer, callbacks, logging, and final test
```

The boundary is selected through Hydra configuration. `run.py` must not grow
scattered checks such as `if isinstance(data, HeteroData)` or
`if data_domain == "heterogeneous"`.

## Shared Preprocessing

The existing preprocessing pipeline should be reused where its abstraction is
valid:

- Hydra transform selection;
- `DataTransform`;
- PyG `Compose`;
- transform-parameter hashing;
- file locking;
- preprocessing cache;
- preprocessing-device transitions;
- serialization;
- preprocessing-time logging.

`PreProcessor` will accept both `Data` and `HeteroData` and preserve the
concrete class through processing, caching, and loading. This is not
implemented by simply deleting the current assertion. The preprocessor must
validate input/output representation consistency and have serialization
round-trip tests for both data classes.

Transforms remain in the existing TopoBench transform registry. New wrappers
around PyG transforms provide the operations needed by heterogeneous datasets:

- constant features for selected featureless node types;
- explicit conversion to bidirectional relations.

Each transform has a declared `HeteroData` compatibility capability.
`DataTransform` rejects a heterogeneous input early when the configured
transform does not support it. Existing graph liftings remain homogeneous-only
unless they are deliberately upgraded and tested later.

Directionality is never changed implicitly. Dataset transform configuration
must request reverse relations explicitly. DBLP already contains relations in
both directions and only needs constant conference features. OGB-MAG needs
reverse relations for bidirectional message flow.

Featureless node types are handled in preprocessing, not hidden inside a
model. After configured transforms run, every represented node type must have
an `x` tensor. Version 1 raises a clear validation error otherwise. Learned
featureless-node embeddings are deferred.

## Configurable Data Pipelines

Add a small data-pipeline interface that returns:

- a Lightning data module;
- preprocessing time;
- an optional runtime data specification.

### Default data pipeline

The default implementation wraps the current behavior:

1. instantiate `PreProcessor`;
2. call the existing inductive/transductive split utilities;
3. instantiate `TBDataloader`.

This keeps current graph and topological experiments behaviorally unchanged.

### Heterogeneous node pipeline

The heterogeneous implementation:

1. runs the shared preprocessor;
2. requires exactly one native `HeteroData` graph;
3. constructs and validates `HeterogeneousDataSpec`;
4. instantiates `HeterogeneousNodeDataModule`;
5. returns the specification for model construction.

It never uses `DataloadDataset`, `DomainData`, `collate_fn`, or
`TBDataloader`.

## Runtime Heterogeneous Data Specification

`HeterogeneousDataSpec` is an immutable record derived after preprocessing.
It contains:

- ordered node types;
- canonical edge triples `(source_type, relation_type, destination_type)`;
- target node type;
- configured number of classes.

Construction validates:

- the object is `HeteroData`;
- the target node type exists;
- every node type has a rank-2 `x` tensor with one row per node;
- the target store contains one-dimensional `y`;
- `y` has one entry per target node;
- `train_mask`, `val_mask`, and `test_mask` are boolean;
- every mask has one entry per target node;
- each split is non-empty;
- masks are pairwise disjoint;
- labels selected by masks are within the configured class range;
- the metadata contains at least one node type and one edge type;
- every edge index has shape `[2, num_edges]` and valid source/destination
  bounds.

The masks need not cover every target node because some datasets legitimately
leave nodes unsupervised.

Model metadata must come from this runtime specification. Dataset relation
names are never hard-coded in HGT or HeteroSAGE. A centralized model
instantiation helper copies the model configuration, injects node types and
edge types into declared placeholder fields, and then invokes Hydra. Existing
models receive no runtime overrides.

## Separate Heterogeneous Data Module

`HeterogeneousNodeDataModule` has two explicit modes.

### Full-batch mode

Use PyG's ordinary `DataLoader` with the single `HeteroData` graph. Train,
validation, and test loaders see the same transductive graph. The supervision
adapter applies the appropriate target-node mask for each phase.

This mode is used for the synthetic prototype and DBLP.

### Neighbor mode

Use a separate PyG `NeighborLoader` for every phase:

```text
input_nodes = (target_node_type, phase_mask)
```

Training shuffles seed nodes. Validation and test do not. Loss and metrics use
only the first `batch[target_node_type].batch_size` target nodes because PyG
places seed nodes first in sampled batches.

`num_neighbors` must contain at least as many hops as the model has message
passing layers. Version 1 uses generic `NeighborLoader` for both HGT and
HeteroSAGE so their sampling conditions remain identical.

Validation and test receive separate configurable fan-outs and a fixed
evaluation seed. Sampled evaluation is explicitly labeled as sampled rather
than exact full-graph inference. Benchmark-quality layer-wise exact inference
is outside version 1 and must be designed separately before a publication
claim that requires it.

The environment must have a supported sampling backend. The current
environment has `torch-sparse`; `pyg-lib` may be added later for performance.
Startup validation reports an actionable error if neither backend is usable.

## Synthetic Dataset

Provide a deterministic, offline `SyntheticHeterogeneousDataset` rather than
depending directly on PyG's randomly generated `FakeHeteroDataset`.

The graph contains:

- target type `author`, with features, labels, and fixed masks;
- type `paper`, with a different input feature dimension;
- featureless type `venue`;
- directed `author -> paper` and `paper -> venue` relations;
- reverse relations added through the shared transform pipeline;
- balanced train, validation, and test masks.

The fixture is intentionally small enough for unit tests and two-epoch CPU
pipeline tests. Target features contain a simple learnable signal so a short
overfit run can establish that loss decreases. The same production dataset
class is used by debug experiments; test helpers may request smaller sizes but
must not maintain a second schema implementation.

## Models

### Typed feature encoder

A `HeterogeneousNodeFeatureEncoder` owns one eagerly constructed linear
projection per node type and maps all type-specific feature dimensions to a
shared hidden width. The validated data specification supplies the input
widths before optimizer construction. The encoder follows the existing
TopoBench contract: it accepts `HeteroData`, updates each typed `x` store, and
returns the encoded `HeteroData`. It applies activation and dropout
consistently across types. Missing features are an input-validation error.

### Shared HGT core

Extract the dictionary-processing portion of `CellHGT` into
`HGTBackbone` under the heterogeneous backbone package. It accepts:

- runtime metadata;
- `x_dict`;
- `edge_index_dict`;
- hidden width;
- number of layers and heads;
- activation and dropout.

`CellHGT` becomes a thin subclass/adapter that:

1. creates rank-based metadata;
2. converts sparse incidence matrices to edge-index dictionaries;
3. delegates message passing to `HGTBackbone`;
4. maps typed output back to integer ranks.

Existing `CellHGT` public attributes and tests remain valid.

### HeteroSAGE

`HeteroSAGEBackbone` builds a `HeteroConv` layer for each message-passing
layer and one eager bipartite
`SAGEConv((hidden_channels, hidden_channels), hidden_channels)` per canonical
edge type. The typed feature encoder has already projected every input into
the common hidden width, so lazy relation parameters are unnecessary.
Relation outputs aggregate by sum. Each layer applies normalization,
activation, and dropout by node type. A node type that receives no relation
output carries its previous representation forward.

This mirrors PyG's official DBLP heterogeneous convolution example while
remaining metadata-driven.

### Wrapper and readout

A heterogeneous wrapper receives the encoded `HeteroData`, passes
`batch.x_dict` and `batch.edge_index_dict` to either backbone, and returns the
complete output dictionary plus unfiltered target labels.

A target-node readout applies a shared linear classifier to the selected node
type. Both HGT and HeteroSAGE use the same encoder and readout.

## Shared Training and Typed Supervision

Do not create a second Lightning training model.

Add a supervision-adapter interface to `TBModel`:

- the default adapter preserves current top-level mask behavior;
- the heterogeneous adapter selects a configured target node type.

For a full graph, the heterogeneous adapter selects the mask associated with
the current phase. For a sampled graph, it slices logits and labels to the
first `batch_size` seed nodes. It also reports the number of supervised
examples so Lightning loss logging is weighted correctly.

The sampling mode is passed explicitly from configuration. The adapter does
not infer full versus sampled execution from incidental batch attributes.
With directional neighbor sampling, the fanout depth must equal the model
depth; configuration validation rejects a mismatch before training.

Loss, evaluator, optimizer, callbacks, checkpoint selection, W&B integration,
and best-checkpoint validation/test reruns remain shared.

## Configuration

Add configuration groups:

- `data_pipeline/default.yaml`;
- `data_pipeline/heterogeneous_node.yaml`;
- `dataset/heterogeneous/SyntheticHeterogeneous.yaml`;
- `dataset/heterogeneous/DBLP.yaml`;
- later `dataset/heterogeneous/OGB_MAG.yaml`;
- `model/heterogeneous/hgt.yaml`;
- `model/heterogeneous/heterosage.yaml`.

Add experiment configurations for:

- synthetic HGT full-batch debug;
- synthetic HeteroSAGE full-batch debug;
- synthetic HGT sampled debug;
- synthetic HeteroSAGE sampled debug;
- DBLP HGT;
- DBLP HeteroSAGE;
- later OGB-MAG HGT and HeteroSAGE sampled runs.

Every experiment uses the normal `python -m topobench` entry point.

## Promotion Gates

### Gate 1: synthetic unit correctness

- exact metadata and relation direction;
- different feature widths map to one hidden width;
- featureless `venue` receives configured constant features;
- mask and schema validation errors are precise;
- both backbones preserve node-type shapes;
- gradients are finite;
- evaluation is deterministic;
- existing `CellHGT` tests remain green.

### Gate 2: synthetic full-batch integration

- Hydra composes both models;
- two CPU epochs train, validate, and test;
- losses and metrics are finite;
- the best checkpoint is reloaded;
- `test_best_rerun/*` is logged when `test=true`;
- a short overfit run reduces training loss;
- no network download is required.

### Gate 3: synthetic sampled integration

- all target seed nodes occur exactly once per evaluation pass;
- only seed nodes contribute to loss and metrics;
- sampled edges stay within the returned subgraph;
- HGT and HeteroSAGE receive identical loader settings;
- two CPU epochs complete with finite metrics;
- repeated validation with the same seed is reproducible.

### Gate 4: DBLP

- official author masks are preserved;
- conference constant features are added by the shared transform pipeline;
- both models complete a one-batch forward/backward smoke test;
- both models complete a short full-batch training run;
- validation and best-checkpoint test metrics are logged;
- no performance threshold is used as a correctness gate.

### Gate 5: OGB-MAG readiness

- use official paper masks and 349 classes;
- use processed structural features for otherwise featureless node types;
- add reverse relations explicitly;
- one sampled batch passes schema validation;
- both models complete forward/backward on the same sampled batch;
- a one-epoch sampled smoke run completes;
- memory use, throughput, fan-outs, batch size, and sampling backend are
  recorded.

No OGB-MAG performance claim is made until sampled-versus-exact inference
semantics are explicitly chosen and documented.

## Reliability and Compatibility

The implementation must follow test-driven development. Every production
change starts with a focused failing test, followed by the smallest passing
implementation and a narrow commit.

Required regression protection:

- all pre-existing preprocessing tests;
- all `TBDataloader` and `DomainData` collation tests;
- all model masking tests;
- all cell-complex HGT tests;
- representative graph, cell, simplicial, and hypergraph config composition;
- the complete test suite before handoff.

PyG must become an explicit project dependency rather than remaining only a
transitive dependency. Version `2.8.0.post1`, already resolved by the lock
file, is the initial supported baseline and must be exercised in CI. Broaden
the range only with a compatibility matrix. The local editable/development
installation must not be treated as the compatibility baseline.

## Primary References

- PyG heterogeneous graph tutorial:
  <https://pytorch-geometric.readthedocs.io/en/2.5.0/tutorial/heterogeneous.html>
- PyG neighbor sampling tutorial:
  <https://pytorch-geometric.readthedocs.io/en/latest/tutorial/neighbor_loader.html>
- PyG DBLP dataset:
  <https://pytorch-geometric.readthedocs.io/en/2.8.0/generated/torch_geometric.datasets.DBLP.html>
- PyG OGB-MAG dataset:
  <https://pytorch-geometric.readthedocs.io/en/2.6.1/generated/torch_geometric.datasets.OGB_MAG.html>
- OGB node property prediction:
  <https://ogb.stanford.edu/docs/nodeprop/>
