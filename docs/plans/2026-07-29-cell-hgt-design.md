# Cell-Complex HGT Design

## Objective

Add a batched Heterogeneous Graph Transformer (HGT) backbone to TopoBench for
inductive graph-level learning on graphs lifted to cell complexes. The first
version targets the existing default graph-to-cell cycle lifting and uses only
unsigned, bidirectional incidence relations.

## Scope

Version 1 models ranks 0, 1, and 2 as three heterogeneous node types:

- `rank_0`: graph vertices;
- `rank_1`: graph edges;
- `rank_2`: lifted 2-cells.

It models four directed edge types:

- `rank_0 --up_incidence-0--> rank_1`;
- `rank_1 --down_incidence-1--> rank_0`;
- `rank_1 --up_incidence-1--> rank_2`;
- `rank_2 --down_incidence-2--> rank_1`.

Signed incidence values, adjacency relations, long-range relations, a
`HeteroData` conversion, and ports of the historical `acbull/pyHGT`
implementation are explicitly out of scope.

## Architecture

The existing TopoBench pipeline remains intact:

```text
batched lifted DomainData
    -> AllCellFeatureEncoder
    -> CellHGT
    -> TuneWrapper
    -> PropagateSignalDown
    -> graph-level pooling and prediction
```

`CellHGT` converts `x_0`, `x_1`, and `x_2` to a PyG `x_dict`. It converts the
four sparse TopoBench neighborhood tensors to a typed `edge_index_dict` and
applies a stack of PyG `HGTConv` layers. The backbone returns the rank-keyed
dictionary expected by `TuneWrapper`.

TopoBench sparse incidence neighborhoods use matrix rows for destinations and
columns for sources. PyG `edge_index` uses the first row for sources and the
second for destinations. Every sparse neighborhood index must therefore be
converted with `coalesce().indices().flip(0).long()`.

## Batching

Batching is mandatory. TopoBench's collation produces block-diagonal sparse
incidence matrices plus `batch_0`, `batch_1`, and `batch_2`. HGT processes that
disjoint union in one call, without looping over examples and without
materializing individual `HeteroData` objects. The existing readout continues
to use `batch_0` for graph pooling.

The implementation must handle:

- more than one graph per mini-batch;
- graphs with different numbers of cells at each rank;
- a mini-batch in which some graphs have no 2-cells;
- a mini-batch with no 2-cells at all;
- empty relations and node types that receive no messages.

If PyG omits a node type or returns `None` because it receives no message, that
rank retains its previous representation for that layer. This is a defined
fallback, not an error.

## Configuration

Add `configs/model/cell/hgt.yaml`. It reuses:

- `AllCellFeatureEncoder`;
- `TuneWrapper`;
- `PropagateSignalDown`;
- the default `CellCycleLifting`.

The initial controlled configuration uses:

- hidden width `64`;
- two HGT layers;
- four attention heads;
- dropout `0.1`;
- the four bidirectional incidence relations above.

The hidden width must be divisible by the number of heads. Neither the
backbone nor its configuration disables dataset batching.

## Test Strategy and Promotion Gates

Development follows test-driven stages.

1. Unit tests establish relation names, index direction, shape preservation,
   validation errors, empty-rank behavior, finite gradients, and deterministic
   evaluation.
2. A real `collate_fn` test batches two different lifted complexes, proves the
   sparse relations remain block diagonal, and compares per-graph output with
   output from the same graph processed alone.
3. Hydra composition and model-forward tests establish compatibility with the
   normal TopoBench encoder, wrapper, and readout.
4. Two-epoch CPU smoke runs on MUTAG and PROTEINS establish that multi-example
   batches train and validate without exceptions or non-finite losses.
5. A short overfit/debug run establishes that training loss decreases on a
   small dataset slice or, if a slice is not exposed by the existing loader,
   that the full small-dataset training loss decreases over a fixed short run.
6. Only after all correctness gates pass is the full ZINC run enabled.

There is deliberately no accuracy threshold for promotion to ZINC. MUTAG and
PROTEINS are debugging datasets here, not evidence of model quality. ZINC MAE,
runtime, memory use, parameter count, and comparison against an equivalently
budgeted TopoTune baseline are experimental results.

## Risks and Controls

- **Incorrect edge direction:** covered by exact expected-index tests.
- **Cross-graph leakage:** covered by block-diagonal and alone-versus-batched
  equivalence tests.
- **Empty 2-cell failures:** covered by explicit zero-row fixtures.
- **Silent rank loss from HGTConv:** covered by output-key and fallback tests.
- **Configuration drift:** covered by Hydra composition and instantiated
  forward tests.
- **Unfair performance claims:** controlled by fixed seeds, the same lift,
  shared training protocol, reported parameter counts, and a TopoTune
  reference run.

## Acceptance Criteria

The implementation is ready for ZINC when:

- all new unit and configuration tests pass;
- the relevant existing combinatorial-backbone tests still pass;
- MUTAG and PROTEINS complete batched two-epoch CPU smoke runs;
- all logged losses and predictions are finite;
- gradients reach HGT parameters;
- there is no per-graph loop in `CellHGT.forward`;
- the ZINC configuration composes and a one-batch forward/backward probe
  succeeds.
