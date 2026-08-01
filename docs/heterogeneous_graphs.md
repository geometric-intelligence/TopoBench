# Native heterogeneous graphs

TopoBench uses PyTorch Geometric `HeteroData` directly for node classification on one typed graph. A heterogeneous graph has named node stores and canonical relation triples such as `("author", "writes", "paper")`. Each node type keeps its own feature matrix, and each relation keeps its own `edge_index`.

HGT and HeteroSAGE share the same validation, preprocessing, loader, target supervision, evaluator, checkpoint, and logging contracts. The main operating choice is whether a phase sees the full graph or a sampled neighborhood around target seeds.

## Run the four offline experiments

The deterministic synthetic graph requires no download. The following four commands are the quickest complete checks of both models and both loader modes:

```bash
uv run python -m topobench experiment=heterogeneous_synthetic_hgt_full
uv run python -m topobench experiment=heterogeneous_synthetic_heterosage_full
uv run python -m topobench experiment=heterogeneous_synthetic_hgt_neighbor
uv run python -m topobench experiment=heterogeneous_synthetic_heterosage_neighbor
```

Use the full variants to check typed message passing without a sampling backend. The neighbor variants additionally require a PyG-supported sampling package such as `pyg-lib` or `torch-sparse`.

## Data contract

A heterogeneous dataset contains exactly one processed `HeteroData` object. Only one configured target node type contributes labels, loss, and metrics. The object must satisfy these invariants before model construction:

- `data.node_types` is non-empty.
- Every node store has a floating `x` matrix with shape `[N_type, F_type]`, one feature row per node, and finite values.
- Every edge type is a `(source, relation, destination)` triple. Its `edge_index` has shape `[2, E]`, uses `torch.long`, and contains IDs within the source and destination stores.
- `dataset.parameters.target_node_type` names an existing node store.
- The target store has one-dimensional `torch.long` labels `y`, one per target node.
- The target store has boolean `train_mask`, `val_mask`, and `test_mask` vectors, each with one entry per target node. The masks are non-empty and pairwise disjoint. Nodes outside their union are context only.
- Supervised labels are within `[0, num_classes - 1]`. Unsupervised target nodes may retain a sentinel label because they are never selected.

The dataset configuration owns the target declaration:

```yaml
parameters:
  task: classification
  task_level: node
  target_node_type: author
  num_classes: 4
```

After preprocessing, the pipeline derives ordered PyG metadata and each node type's feature width. Model config fields for metadata, target type, input widths, and output width remain `null` until that validated runtime specification is available.

## Preserve official masks

DBLP and OGB-MAG keep the train, validation, and test masks provided by PyG. TopoBench does not repartition them.

In full mode, all three phases reuse the same transductive graph, and the corresponding target-store mask selects supervised nodes. In neighbor mode, the same masks define the phase-specific target seed IDs. This makes the split independent of the sampling traversal.

## Preprocessing typed stores

Heterogeneous data uses the shared preprocessor, transform registry, cache, locking, and PyG serialization. A transform may receive `HeteroData` only when it declares `supports_heterodata = True`; an incompatible transform fails before it can flatten or discard typed stores.

Every node type must have features after preprocessing. `HeterogeneousConstantFeatures` creates or appends a constant channel for selected featureless stores. Synthetic data uses it for `venue`, and DBLP uses it for `conference`. OGB-MAG requests its configured structural preprocessing so every store reaches the model with features.

Relation direction is explicit. `HeterogeneousToUndirected` adds typed reverse relations and defaults to `merge: false`. Synthetic data and OGB-MAG request it. DBLP already provides both directions and does not request it again.

## Full-graph mode

`dataset.dataloader_params.mode: full_batch` uses a PyG `DataLoader` containing the one validated graph. Its operating invariants are:

- loader batch size is one graph;
- train, validation, and test reuse the same typed structure;
- the phase mask on the target store selects loss and metric rows;
- every non-target store and every unselected target node remains available as message-passing context;
- no target-seed metadata is needed because the mask is the supervision boundary.

Synthetic full experiments and both DBLP experiments use this mode.

## Neighbor mode

`dataset.dataloader_params.mode: neighbor` creates a separate `NeighborLoader` for train, validation, and test. It does not use homogeneous or hypergraph collation.

Every relation fan-out list must contain one entry per backbone message-passing layer. HGT and HeteroSAGE use the same loader settings. PyG places target seed nodes first in the sampled target store and records their count in `batch[target_node_type].batch_size`. TopoBench validates that metadata, then applies the equivalent of:

```python
seed_count = int(batch[target_node_type].batch_size)
supervised_logits = logits[:seed_count]
supervised_labels = batch[target_node_type].y[:seed_count]
```

Only those leading seed rows contribute directly to loss and metrics. Other sampled target nodes and all non-target stores are context. Missing, non-integer, zero, or oversized target `batch_size` metadata is an error rather than a reason to supervise the complete sampled store.

Validation and test use the configured `evaluation_seed` and `sampled_neighbor_fixed` protocol. Each fresh traversal reproduces the same sampled sequence without retaining all sampled batches in memory. This protocol is sampled evaluation, not exact full-graph inference, and should be recorded with reported results.

## Run DBLP in full mode

These commands may download DBLP. The environment variable is an explicit operator opt-in used by the project's integration workflow:

```bash
TOPOBENCH_ALLOW_DOWNLOADS=1 uv run python -m topobench experiment=heterogeneous_dblp_hgt
TOPOBENCH_ALLOW_DOWNLOADS=1 uv run python -m topobench experiment=heterogeneous_dblp_heterosage
```

Both experiments supervise `author` nodes through official masks and use the complete graph for each phase.

## Preflight OGB-MAG in neighbor mode

OGB-MAG is large and sampled-only. Start with bounded one-epoch runs before removing the batch limits. They exercise training, checkpoint selection, and fixed sampled validation and test reruns without traversing every seed:

```bash
uv run python -m topobench experiment=heterogeneous_ogb_mag_hgt seed=0 trainer.min_epochs=1 trainer.max_epochs=1 trainer.limit_train_batches=10 trainer.limit_val_batches=5 trainer.limit_test_batches=5 logger=heterogeneous_wandb logger.wandb.name=ogb-mag-hgt-neighbor-bounded-seed0
uv run python -m topobench experiment=heterogeneous_ogb_mag_heterosage seed=0 trainer.min_epochs=1 trainer.max_epochs=1 trainer.limit_train_batches=10 trainer.limit_val_batches=5 trainer.limit_test_batches=5 logger=heterogeneous_wandb logger.wandb.name=ogb-mag-heterosage-neighbor-bounded-seed0
```

A bounded metric describes only the configured prefix of a fixed sampled loader and is not a benchmark result. After the bounded run succeeds on the intended machine, remove the three `limit_*_batches` overrides to traverse every configured seed. Keep the sampled evaluation protocol attached to any reported metric.

## Checkpoint and W&B identity

Best-checkpoint validation and test reruns use `val_best_rerun/<metric>` and `test_best_rerun/<metric>`. These are distinct from the last epoch's phase metrics.

Heterogeneous W&B runs use project `topobench-heterogeneous` and the deterministic identity:

```text
{dataset_name}-{model_name}-{mode}-seed{seed}
```

Dataset, model, and mode tags remain stable across seed comparisons. Use `logger=csv` for local runs that must not contact W&B.

## Add a heterogeneous dataset

1. Implement an `AbstractLoader` under `topobench/data/loaders/heterogeneous/`. It must return a PyG dataset of length one whose item is native `HeteroData`.
2. Export the class from that package and include it in `HETEROGENEOUS_LOADERS`. The top-level `LOADER_CLASSES` registry then exposes the class under `topobench.data.loaders`.
3. Add `configs/dataset/heterogeneous/<Dataset>.yaml` with `target_node_type`, `num_classes`, split source, and an explicit loader mode.
4. Add only necessary transforms that declare typed-data support.
5. Preserve official masks. Promote to neighbor mode only when size requires it, align every fan-out depth with model depth, and declare the sampled evaluation protocol.

## Add a heterogeneous model

1. Implement an eager backbone under `topobench/nn/backbones/heterogeneous/`. Construct trainable modules in `__init__`; `forward()` must not create parameters.
2. Accept ordered PyG metadata, `x_dict`, and `edge_index_dict`, and return a representation for every node type. Define how stores with no incoming relation retain a representation.
3. Export the backbone from its domain package, then add its public name to `topobench.nn.backbones.MODEL_CLASSES`.
4. Reuse `HeterogeneousNodeFeatureEncoder`, `HeterogeneousWrapper`, `HeterogeneousNodeReadout`, and `HeterogeneousNodeSupervisionAdapter` when their contracts fit.
5. Add `configs/model/heterogeneous/<model>.yaml`, leaving dataset-derived metadata, dimensions, target type, output width, and supervision mode as runtime values.

The model must behave identically under full and neighbor supervision except for how supervised rows are selected.

## Related documentation

- [Graph data and batching](graph_data.md) explains homogeneous `Data` collation and split contracts.
- [Native hypergraphs](hypergraphs.md) explains incidence batching and node masks.
- [API reference](api/index.rst) lists the public loader, data-module, backbone, wrapper, and supervision classes.
