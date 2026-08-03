# Native hypergraphs

TopoBench represents a hypergraph as node-to-hyperedge incidence in `HypergraphData`, a PyTorch Geometric `Data` subclass. The current pipeline supports transductive node classification with EDGNN and PyG HypergraphConv.

Run both models on the deterministic synthetic fixture:

```bash
uv run python -m topobench experiment=hypergraph_synthetic_edgnn
uv run python -m topobench experiment=hypergraph_synthetic_hypergraph_conv
```

## Incidence convention

`HypergraphData.hyperedge_index` is a `torch.long` tensor with shape `[2, M]`, where `M` is the number of node-to-hyperedge memberships:

- `hyperedge_index[0]` contains node IDs in `[0, num_nodes)`;
- `hyperedge_index[1]` contains hyperedge IDs in `[0, num_hyperedges)`.

The hyperedge IDs must be contiguous from zero, and every ID below `num_hyperedges` must occur at least once. `num_hyperedges` is an explicit positive integer; it is not inferred from a batch. Node IDs and hyperedge IDs must be nonnegative and within their independent ranges.

For example, memberships `(node 0, hyperedge 0)`, `(node 2, hyperedge 0)`, and `(node 1, hyperedge 1)` are stored as:

```python
hyperedge_index = torch.tensor(
    [
        [0, 2, 1],
        [0, 0, 1],
    ],
    dtype=torch.long,
)
```

The orientation is exact: row 0 is always nodes and row 1 is always hyperedges. Loaders must emit this convention directly; the validation boundary does not transpose, renumber, or repair malformed incidence.

## Node data and valid masks

A complete node-classification example carries:

- floating, finite `x` with shape `[N, F]`;
- `hyperedge_index` and explicit `num_hyperedges` as defined above;
- one label per node in `y`;
- boolean `train_mask`, `val_mask`, and `test_mask`, each with shape `[N]`.

Every phase mask must select at least one node. The masks must be pairwise disjoint, and their union must cover every labeled node exactly once. The pipeline rejects partial coverage, overlap, non-boolean masks, and masks with the wrong length.

All phases reuse the same hypergraph structure. The phase mask selects which node logits contribute to loss and metrics; other nodes remain available as context. This is node classification only: graph-level targets, hyperedge-level targets, and regression targets are outside the current hypergraph contract.

## Independent batching offsets

Node IDs and hyperedge IDs live in different index spaces. `HypergraphData.__inc__` therefore tells PyG to offset the two incidence rows independently.

Suppose example A has `N_A` nodes and `H_A` hyperedges. When example B is appended, PyG adds:

```text
[N_A]
[H_A]
```

to B's incidence rows. In other words, B's node IDs increase by `N_A`, while B's hyperedge IDs increase by `H_A`. Later examples receive cumulative node and hyperedge counts from all preceding examples.

This differs from `edge_index`, whose two rows both index nodes and therefore share the same node offset. A generic `Data` object would apply the wrong semantics to hyperedge IDs. Dataset loaders must return `HypergraphData` and set both `num_nodes` through `x` and explicit `num_hyperedges` so PyG can batch safely.

The current transductive pipeline uses one source hypergraph with data-loader batch size one. The independent offset contract still belongs to the public representation and protects direct PyG batching, fixtures, and future loaders.

## Processed-cache version

The native representation version is **2**, and its processed filename is:

```text
hypergraph_data_v2.pt
```

An older `processed/data.pt` cache is incompatible. When only that older file exists, the dataset warns, ignores it, and regenerates `hypergraph_data_v2.pt` from the authenticated raw data. It never deserializes the older object as native data.

`hypergraph_data_v2.pt` is a tensor/primitive-only, versioned payload paired with `hypergraph_data_v2.pt.manifest.json`. TopoBench opens both beneath the configured per-principal cache root without following links, verifies file ownership, permissions, path, byte size, cache identity, and SHA-256, then calls `torch.load(..., weights_only=True)`. Only after closed-schema validation does it reconstruct the statically selected `HypergraphData` class and validate `representation_version == 2` plus the incidence structure. Missing, stale, or mismatched metadata raises an error rather than guessing how to migrate the cache. Remove the invalid payload and its manifest to regenerate them from retained raw files.

Changing the incidence layout requires a new representation number and processed filename. Reusing version 2 for incompatible data would allow stale caches to cross the validation boundary.

## Adding a hypergraph dataset

1. Implement an `AbstractLoader` under `topobench/data/loaders/hypergraph/` that returns `HypergraphData` in the documented incidence orientation.
2. Export the class from that package and include it in `HYPERGRAPH_LOADERS`. The top-level `LOADER_CLASSES` registry then exposes it under `topobench.data.loaders`.
3. Add `configs/dataset/hypergraph/<Dataset>.yaml` with node classification, transductive splitting, the registered loader target, feature policy, and data-loader parameters.
4. Preserve official node masks when available. Otherwise create one deterministic, complete, non-overlapping partition through the shared split utilities.
5. Write processed data with the current representation number and cache filename. Validate structure before it reaches model construction.

## Adding a hypergraph model

1. Implement the backbone under `topobench/nn/backbones/hypergraph/` against `x` and the exact node-to-hyperedge incidence orientation.
2. Export it from the domain package and add its public name to `topobench.nn.backbones.MODEL_CLASSES`.
3. Reuse `HypergraphWrapper` and the registered node readout when their contracts fit.
4. Add `configs/model/hypergraph/<model>.yaml` with registered targets and a capability entry limited to the supported node-classification contract.

See [Graph data and batching](graph_data.md), [Native heterogeneous graphs](heterogeneous_graphs.md), and the [API reference](api/index.rst) for neighboring contracts.
