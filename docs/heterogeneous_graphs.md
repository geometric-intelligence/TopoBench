# Native heterogeneous graphs

TopoBench supports native PyTorch Geometric `HeteroData` for a deliberately
narrow first use case: node classification on one heterogeneous graph. HGT and
HeteroSAGE share the same preprocessing, batching, supervision, evaluation,
checkpoint, and logging path.

Here, **heterogeneous** means that a graph has named node types and typed
relations, such as `("author", "writes", "paper")`, with a separate feature
matrix for each node type. This is different from a **heterophilous
homogeneous** graph: the datasets loaded by
`HeterophilousGraphDatasetLoader` are ordinary PyG `Data` objects whose
connected nodes often have different labels. They do not use this pipeline.

The architectural rationale and staged promotion gates are recorded in the
[approved design](plans/2026-07-30-native-heterogeneous-graph-design.md). This
guide is the public configuration and extension contract.

## Offline synthetic checks

The deterministic synthetic graph is the quickest way to validate a setup. It
uses no downloaded data. These commands compose the production configuration,
build the data and model objects, and intentionally skip training, testing, and
external logging:

```bash
uv run python -m topobench experiment=heterogeneous_synthetic_hgt_full train=false test=false logger=[]
uv run python -m topobench experiment=heterogeneous_synthetic_heterosage_full train=false test=false logger=[]
uv run python -m topobench experiment=heterogeneous_synthetic_hgt_neighbor train=false test=false logger=[]
uv run python -m topobench experiment=heterogeneous_synthetic_heterosage_neighbor train=false test=false logger=[]
```

The executable documentation test also builds each documented synthetic
production pipeline and runs one model step:

```bash
uv run pytest test/docs/test_heterogeneous_examples.py -q
```

Remove `train=false test=false logger=[]` to run a configured experiment with
training, best-checkpoint validation/test reruns, and W&B logging. For example:

```bash
uv run python -m topobench experiment=heterogeneous_synthetic_hgt_neighbor
```

## Version 1 data contract

The heterogeneous-node pipeline accepts exactly one processed `HeteroData`
object. Multiple graphs, graph-level labels, and link targets are rejected. The
processed graph must satisfy all of these invariants before model construction:

- `data.node_types` is non-empty and every node store has a floating-point,
  rank-two `x` tensor with one row per node.
- Each edge type is a canonical `(source, relation, destination)` triple. An
  `edge_index`, when present, is a `torch.long` tensor with shape `[2, E]` and
  indices within the source and destination stores.
- `dataset.parameters.target_node_type` names an existing node store. Only
  this store is supervised.
- The target store has one-dimensional `torch.long` labels `y`, with one label
  per target node. Only labels selected by the union of `train_mask`,
  `val_mask`, and `test_mask` must be in
  `[0, dataset.parameters.num_classes - 1]`. Unsupervised nodes may retain
  sentinel or out-of-range labels; TopoBench ignores them.
- The target store has non-empty, boolean `train_mask`, `val_mask`, and
  `test_mask` tensors. They have one entry per target node and are pairwise
  disjoint. They may leave some nodes unsupervised.

Set the target type in the dataset configuration, not in the model:

```yaml
parameters:
  task: classification
  task_level: node
  target_node_type: author
  num_classes: 4
```

The pipeline derives ordered metadata and per-type input widths after
preprocessing. It injects that validated runtime specification into model
fields declared as literal `null`; relation names and feature widths are not
hard-coded in a backbone.

### Official masks

DBLP and OGB-MAG retain the train, validation, and test masks supplied by PyG.
TopoBench neither regenerates nor repartitions them. In full-batch mode, each
phase sees the same transductive graph and its corresponding target-store mask
selects supervised nodes. In neighbor mode, each phase constructs its input
seeds from that same official mask.

### Transforms and featureless node types

Heterogeneous data reuses the ordinary TopoBench `PreProcessor`, transform
registry, cache, locking, and PyG serialization. A transform may run on
`HeteroData` only when its implementation declares
`supports_heterodata = True`; incompatible graph liftings and data
manipulations fail early instead of flattening typed stores.

Every node type must have features after preprocessing. Use the shared
`HeterogeneousConstantFeatures` transform to create or append a constant
channel for selected featureless stores. Synthetic data uses it for `venue`;
DBLP uses it for `conference`. OGB-MAG is loaded with
`preprocess: metapath2vec`, which supplies structural features for all node
types. Learned embeddings for featureless stores are not part of version 1.

Directionality is explicit. `HeterogeneousToUndirected` adds typed reverse
relations and defaults to `merge: false`. Synthetic data and OGB-MAG request
this transform; DBLP already supplies both directions and must not receive it
again.

## Full-graph and neighbor batching

`dataset.dataloader_params.mode` is part of the experiment protocol.

`full_batch` uses PyG `DataLoader` with the one native graph. The phase mask on
the target store determines the loss and metrics. This is the synthetic
full-batch and DBLP protocol.

`neighbor` uses a separate generic PyG `NeighborLoader` for train, validation,
and test. It is intentionally separate from TopoBench's topological collation
path. Sampling is directional, and every relation fan-out list must have
exactly `model.backbone.num_layers` entries. HGT and HeteroSAGE use identical
loader settings.

PyG places target seed nodes first in a sampled target store. TopoBench uses

```python
seed_count = int(batch[target_node_type].batch_size)
supervised_logits = logits[:seed_count]
supervised_labels = labels[:seed_count]
```

before computing loss or metrics. The remaining target nodes and all other
node types are message-passing context and never contribute directly to
supervision.

Validation and test use the configured `evaluation_seed` and
`evaluation_protocol: sampled_neighbor_fixed`, so repeated runs reproduce the
same sampled traversal. This is still sampled evaluation, not exact full-graph
inference. Report it as such; do not compare it to exact results without
recording the protocol. A supported `pyg-lib` or `torch-sparse` sampling
backend is required.

## Adding a dataset

Keep the extension small and metadata-driven:

1. Add an `AbstractLoader` under
   `topobench/data/loaders/heterogeneous/`. It must return a PyG dataset of
   length one whose item is native `HeteroData`; do not convert it to
   `DomainData`.
2. Export the loader from the heterogeneous loader package.
3. Add `configs/dataset/heterogeneous/<Dataset>.yaml` with
   `target_node_type`, `num_classes`, `source: official_masks`, and an explicit
   `dataloader_params.mode`.
4. Add a dataset-default transform config only for necessary, declared
   `HeteroData` operations: feature creation and/or explicit reverse
   relations.
5. Test the loader without network by substituting a schema-accurate PyG
   dataset. Assert native store order, labels, masks, feature handling, and
   relation directions. Put any real download test behind
   `TOPOBENCH_ALLOW_DOWNLOADS=1`.
6. Promote from a full-batch smoke test to neighbor sampling only when graph
   size requires it. For neighbor mode, align fan-out depth with model depth
   and record the evaluation protocol.

Do not create new random splits for a dataset that provides official masks.

## Adding a model

A native heterogeneous model must consume runtime metadata rather than dataset
names:

1. Add an eager backbone under `topobench/nn/backbones/heterogeneous/`.
   Construct every trainable per-type and per-relation module in `__init__`;
   `forward()` must not create parameters.
2. Accept the ordered PyG metadata plus `x_dict` and `edge_index_dict`, and
   return a representation for every node type. Define how node types with no
   incoming relation are carried forward.
3. Reuse `HeterogeneousNodeFeatureEncoder`,
   `HeterogeneousWrapper`, `HeterogeneousNodeReadout`, and
   `HeterogeneousNodeSupervisionAdapter`.
4. Add `configs/model/heterogeneous/<model>.yaml`. Leave
   `feature_encoder.input_channels`, `backbone.metadata`, target-type fields,
   classifier output width, and supervision mode as literal `null` runtime
   placeholders.
5. Test arbitrary safe and unsafe type names, metadata order, finite
   gradients, serialization, no parameter creation during forward, full-batch
   supervision, and seed-only neighbor supervision.

If the backbone has message-passing depth `L`, every neighbor fan-out list must
also have length `L`; TopoBench validates this before training.

## DBLP and OGB-MAG

The following commands may download real datasets. The
`TOPOBENCH_ALLOW_DOWNLOADS=1` prefix is an explicit operator opt-in and is
required by the integration tests; the dataset loader itself follows PyG and
may download whenever files are absent. Review the target data directory and
available storage before running them.

DBLP uses full-graph training with `author` labels and official masks:

```bash
TOPOBENCH_ALLOW_DOWNLOADS=1 uv run python -m topobench experiment=heterogeneous_dblp_hgt
TOPOBENCH_ALLOW_DOWNLOADS=1 uv run python -m topobench experiment=heterogeneous_dblp_heterosage
```

Its bounded real-data optimizer smoke test is:

```bash
TOPOBENCH_ALLOW_DOWNLOADS=1 uv run pytest test/integration/test_dblp_heterogeneous.py -q
```

OGB-MAG is large and sampled-only. Promote it in three stages. First, execute
the opt-in preflight, which fetches one train, validation, and test batch and
takes one optimizer step for each model:

```bash
TOPOBENCH_ALLOW_DOWNLOADS=1 uv run pytest \
  test/integration/test_ogb_mag_preflight.py -q -s
```

The preflight may download and preprocess OGB-MAG. It must pass on the intended
machine before any Lightning run. The `-s` flag leaves output capture disabled
so the processed counts, sampled counts, and accelerator-memory report remain
visible.

Second, run bounded Lightning smoke experiments. These exercise training,
checkpointing, fixed sampled validation/test reruns, and the shared W&B project
without traversing every seed.

HGT:

```bash
uv run python -m topobench.run \
  experiment=heterogeneous_ogb_mag_hgt \
  seed=0 \
  trainer.min_epochs=1 \
  trainer.max_epochs=1 \
  trainer.limit_train_batches=10 \
  trainer.limit_val_batches=5 \
  trainer.limit_test_batches=5 \
  logger=heterogeneous_wandb \
  logger.wandb.name=ogb-mag-hgt-neighbor-bounded-seed0
```

HeteroSAGE:

```bash
uv run python -m topobench.run \
  experiment=heterogeneous_ogb_mag_heterosage \
  seed=0 \
  trainer.min_epochs=1 \
  trainer.max_epochs=1 \
  trainer.limit_train_batches=10 \
  trainer.limit_val_batches=5 \
  trainer.limit_test_batches=5 \
  logger=heterogeneous_wandb \
  logger.wandb.name=ogb-mag-heterosage-neighbor-bounded-seed0
```

Bounded metrics are diagnostics, not benchmark results. They cover only a
prefix of the fixed sampled evaluation loaders.

Third, only after a model's bounded run passes, remove all three batch-limit
overrides and change `bounded` to `epoch1` in its W&B name. This performs one
complete sampled train, validation, best-checkpoint validation rerun, and
best-checkpoint test traversal.

HGT:

```bash
uv run python -m topobench.run \
  experiment=heterogeneous_ogb_mag_hgt \
  seed=0 \
  trainer.min_epochs=1 \
  trainer.max_epochs=1 \
  logger=heterogeneous_wandb \
  logger.wandb.name=ogb-mag-hgt-neighbor-epoch1-seed0
```

HeteroSAGE:

```bash
uv run python -m topobench.run \
  experiment=heterogeneous_ogb_mag_heterosage \
  seed=0 \
  trainer.min_epochs=1 \
  trainer.max_epochs=1 \
  logger=heterogeneous_wandb \
  logger.wandb.name=ogb-mag-heterosage-neighbor-epoch1-seed0
```

Even a complete epoch uses fixed neighbor-sampled evaluation, not exact
full-graph inference. Record that protocol with any reported metric.

Only after the complete one-epoch traversal succeeds should the default
50-epoch starting point be attempted:

```bash
uv run python -m topobench.run \
  experiment=heterogeneous_ogb_mag_hgt \
  seed=0 \
  logger=heterogeneous_wandb \
  logger.wandb.name=ogb-mag-hgt-neighbor-epoch50-seed0

uv run python -m topobench.run \
  experiment=heterogeneous_ogb_mag_heterosage \
  seed=0 \
  logger=heterogeneous_wandb \
  logger.wandb.name=ogb-mag-heterosage-neighbor-epoch50-seed0
```

The defaults are neighbor fan-outs `[15, 10]`, target seed batch size `128`,
two message-passing layers, four loader workers, and fixed sampled evaluation.
The preflight reports processed and sampled counts plus accelerator memory
where the platform exposes it. Passing it proves resource readiness, not model
quality; the 50-epoch defaults are an operational baseline rather than tuned
hyperparameters.

## W&B identity

Heterogeneous experiments use the shared W&B project
`topobench-heterogeneous`. The configured run identity is deterministic:

```text
{dataset_name}-{model_name}-{mode}-seed{seed}
```

For example, `DBLP-hgt-full_batch-seed0` and
`OGB_MAG-heterosage-neighbor-seed0` are grouped by dataset, tagged with
dataset/model/mode, and use the mode as W&B `job_type`. Change `seed=<n>` one
run at a time; keep the project and naming scheme unchanged when comparing
models. Use `logger=[]` for checks that must not contact W&B.

## Current non-goals

Version 1 does not support heterogeneous link prediction, heterogeneous graph
classification, multiple heterogeneous examples, distributed sampling,
`HGTLoader`, exact layer-wise sampled inference, automatic conversion of
existing liftings, learned featureless-node embeddings, or flattening
`HeteroData` into the topological data path. The existing cell-complex HGT is a
separate rank-to-type adapter and remains supported.
