# Graph data and batching

TopoBench keeps homogeneous graphs as native PyTorch Geometric `Data` objects. The graph pipeline validates their task and split contracts, then lets PyG create a `Batch`; it does not copy the graph into a framework-specific container.

If you want to run first, use the default synthetic classification pair:

```bash
uv run python -m topobench dataset=graph/SyntheticGraph model=graph/gcn
```

For scalar graph regression, use the fixed experiment:

```bash
uv run python -m topobench experiment=graph_synthetic_regression
```

## Mental model

A graph example carries three core runtime fields:

- `x` has shape `[N, F]` and stores one feature row for each node.
- `edge_index` has shape `[2, E]` and stores source and destination node IDs.
- `batch` has shape `[N_total]` and maps every node in a PyG batch to its graph example.

A dataset returns `Data` objects with `x` and `edge_index`. PyG adds `batch` during collation. For example, if one graph has three nodes and the next has two, their feature matrices become one five-row matrix, the second graph's node IDs in `edge_index` are increased by three, and `batch` is `[0, 0, 0, 1, 1]`. Graph-level readouts use that vector to pool each example independently.

Node features must have two dimensions, one row per node, and a numeric representation compatible with the dataset's declared feature policy. Connectivity uses integer IDs within the graph's node range.

## Inductive graph contracts

Graph classification and graph regression are inductive: train, validation, and test contain different graph examples.

The split boundary creates three non-empty, index-backed `Subset` views over one source dataset. Fixed split metadata uses the keys `train`, `valid`, and `test`. The views must share that source dataset; copying examples into three unrelated lists would discard the identity and laziness guarantees of the split.

The training loader may shuffle its view. Validation and test loaders do not. `batch_size` counts graphs, so a batch may concatenate graphs with different node and edge counts.

### Graph classification targets

Each graph has one integer class label. After batching:

- logits have shape `[B, C]`, where `C` is `num_classes`;
- targets have shape `[B]` and use `torch.long`;
- loss and metrics compare one class decision per graph.

### Scalar graph regression targets

Each graph has one floating target. The supervision boundary normalizes accepted per-example targets to shape `[B, 1]`; model output and target must then have exactly that shape. The evaluator rejects scalar `[B]` predictions, extra output columns, integer targets, and mismatched batch sizes instead of relying on broadcasting.

The `graph_synthetic_regression` experiment is the smallest runnable example of this contract.

## Transductive node-classification contracts

Node classification is transductive: every phase reuses exactly one source graph. The data loader requires `batch_size=1` and does not accept separate phase copies.

The graph has one label per node plus boolean `train_mask`, `val_mask`, and `test_mask` vectors of length `N`. Every mask must be non-empty. The masks must be pairwise disjoint and their union must cover all labeled nodes. Each phase computes loss and metrics only for the nodes selected by its mask while message passing can use the complete graph.

For node classification, logits have shape `[N, C]` and labels have shape `[N]` with integer class IDs.

## Edge-policy behavior

Dataset capabilities declare whether `edge_attr` and `edge_weight` are present. Every graph model declares one policy for each field:

- **`consume`** means the wrapper passes the field to the backbone and the backbone must use it correctly.
- **`ignore`** means the dataset/model pair is allowed, but the wrapper deliberately omits that field.
- **`reject`** means composition fails before training when the dataset supplies the field.

This check is field-specific. For example, GCN ignores `edge_attr` but consumes `edge_weight`; GAT ignores `edge_attr` and rejects `edge_weight`; GIN rejects both. NSD explicitly ignores both. A dataset with an edge field is therefore not assumed compatible merely because a backbone accepts `edge_index`.

Use the capability table as the source of truth when adding a model. Do not silently drop a field in `forward()` or infer behavior from a dataset name.

## Adding a graph dataset

1. Implement an `AbstractLoader` under `topobench/data/loaders/graph/` that returns native `Data` objects.
2. Export the loader in `topobench.data.loaders.graph` and include it in `GRAPH_LOADERS`; the top-level `LOADER_CLASSES` registry then publishes it.
3. Add `configs/dataset/graph/<Dataset>.yaml` with the registered loader target, task, task level, feature policy, split setting, and data-loader parameters.
4. Add the dataset capability entry so model composition checks its task, feature, and edge fields before construction.
5. For inductive data, preserve official graph indices when available. For transductive data, preserve official node masks when available.

See [Native heterogeneous graphs](heterogeneous_graphs.md) and [Native hypergraphs](hypergraphs.md) for the other native representations, or open the [API reference](api/index.rst) for public classes.
