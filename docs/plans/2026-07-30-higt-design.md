# HIGT Design

## Objective

Add a standalone **Heterogeneous Isomorphism Graph Transformer (HIGT)**
backbone to TopoBench. HIGT combines HGT's type- and relation-conditioned
attention with a cardinality-preserving heterogeneous injective-sum branch.
Both branches operate on the same primitive rank-incidence relations, so the
model does not require manually specified meta-paths or materialized composed
adjacencies.

The existing `CellHGT` backbone remains unchanged as the attention-only
baseline. The new public class is named exactly `HIGT`.

## Scope

HIGT version 1 uses the same cell-complex schema as the existing HGT model:

- `rank_0`: graph vertices;
- `rank_1`: graph edges;
- `rank_2`: lifted 2-cells;
- `rank_0 --up_incidence-0--> rank_1`;
- `rank_1 --down_incidence-1--> rank_0`;
- `rank_1 --up_incidence-1--> rank_2`;
- `rank_2 --down_incidence-2--> rank_1`.

Version 1 is deliberately incidence-only. It does not construct CIN's
same-rank atom--atom-through-bond or bond--bond-through-ring neighborhoods.
This keeps the HGT and injective branches on the same computation graph and
isolates the effect of their aggregation mechanisms.

Signed incidence values, manually configured meta-paths, composed adjacency
matrices, temporal encoding, custom heterogeneous sampling, and additional
fusion variants are out of scope.

## Architectural alternatives

Three organizations were considered.

1. Parallel HGT and injective-sum branches with concatenation and a
   rank-specific projection.
2. Parallel branches combined by a learned convex gate.
3. A serial injective-sum-then-HGT pipeline.

The parallel concatenation design was selected. It preserves the two branch
representations separately, permits their arbitrary joint use, and allows the
fusion projection to ignore either branch. Gated addition forces a trade-off
between branches, while serial composition prevents a controlled comparison
on the same layer input.

## Heterogeneous injective-sum branch

Let a primitive heterogeneous relation be

\[
\rho=(q,\operatorname{relation},r),
\]

where \(q\) is the source rank and \(r\) is the target rank. For target cell
\(i\), the relation-specific aggregate at layer \(\ell\) is

\[
S_{\rho,i}^{\ell}
=
\sum_{j\in\mathcal N_{\rho}(i)}
\phi_{\rho}^{\ell}(H_{q,j}^{\ell}).
\]

The aggregation is an unnormalized sum. Each relation has a distinct message
network \(\phi_{\rho}^{\ell}\), so messages from different incidence
directions are not treated as samples from the same distribution.

The aggregates for all relations entering target rank \(r\) remain separate
until the target-rank update:

\[
C_{r,i}^{\ell}
=
U_r^{\ell}\left(
(1+\epsilon_r^{\ell})H_{r,i}^{\ell}
\;\Vert\;
S_{\rho_1,i}^{\ell}
\;\Vert\cdots\Vert\;
S_{\rho_k,i}^{\ell}
\right).
\]

Here, \(U_r^\ell\) is a rank-specific two-layer MLP, \(\epsilon_r^\ell\) is a
learnable scalar, and \(\Vert\) denotes concatenation. Keeping relation sums
in distinct concatenation slots prevents, for example, atom-to-bond and
face-to-bond messages from collapsing into the same bucket.

An empty relation contributes an all-zero aggregate of the configured hidden
width. A rank with no incoming relation still receives its learnable
self-update.

## HGT attention branch

In parallel, a standard PyG `HGTConv` receives the same pre-layer feature
dictionary and the same four primitive incidence edge types:

\[
A^\ell=\operatorname{HGTConv}^{\ell}(H^\ell,E).
\]

The branch retains HGT's node-type-specific projections, relation-conditioned
attention and message transformations, multi-head aggregation, and
target-type output transformation. If `HGTConv` omits a node type because it
receives no messages, its attention-branch value is defined as the previous
representation for that rank.

HIGT provides only the primitive heterogeneous schema. Stacking HIGT layers
allows the HGT branch to learn implicit soft meta-paths such as
rank-0--rank-1--rank-2 without declaring those paths or creating additional
edges. The injective branch likewise composes primitive relations only by
stacking layers.

## Fusion and layer update

For every target rank, HIGT normalizes the two branch outputs independently,
concatenates them, and applies a rank-specific projection:

\[
H_r^{\ell+1}
=
H_r^\ell
+
\operatorname{Dropout}\left(
W_r^{F,\ell}
\left[
\operatorname{LN}_{r,C}^{\ell}(C_r^\ell)
\;\Vert\;
\operatorname{LN}_{r,A}^{\ell}(A_r^\ell)
\right]
\right).
\]

The projection maps from twice the hidden width back to the hidden width.
The resulting tensor is passed through the configured activation. Fusion is
performed separately for each rank; parameters are never shared across
rank-0, rank-1, and rank-2 targets.

The model preserves type-respecting cell permutation equivariance because the
HGT branch, relation-wise sum branch, rank-specific MLPs, and rank-specific
fusion are all equivariant under independent permutations within each rank.

## Pipeline integration

The standard TopoBench pipeline remains:

```text
batched lifted DomainData
    -> AllCellFeatureEncoder
    -> HIGT
    -> TuneWrapper
    -> PropagateSignalDown
    -> graph-level pooling and prediction
```

HIGT consumes `x_0`, `x_1`, and `x_2` plus the four configured sparse
incidence neighborhoods and returns a dictionary keyed by integer cell rank.
It processes the block-diagonal disjoint union produced by TopoBench batching
without looping over examples and without converting individual examples to
`HeteroData`.

The stable metadata and sparse-index conversion shared with `CellHGT` will be
extracted into a small combinatorial-backbone utility module. TopoBench sparse
neighborhood rows are destinations and columns are sources, while PyG
`edge_index` uses sources followed by destinations. Conversion therefore
continues to use:

```python
matrix.coalesce().indices().flip(0).contiguous().long()
```

The refactor must preserve the public behavior and tests of `CellHGT`.

## Public configuration

The public artifacts are:

- class `HIGT`;
- module `topobench/nn/backbones/combinatorial/higt.py`;
- model config `configs/model/cell/higt.yaml`;
- experiment configs `higt_mutag_debug`, `higt_proteins_debug`, and
  `higt_zinc`;
- model name `higt`.

The initial controlled configuration mirrors HGT:

- hidden width 64;
- two layers;
- four attention heads;
- dropout 0.1;
- ReLU activation;
- the same four bidirectional incidence relations.

No fusion-choice option is exposed in version 1.

## ZINC launcher

Provide `scripts/higt/zinc_higt_search.sh`, analogous to
`scripts/hgt/zinc_hgt_search.sh`. It supports:

```text
zinc_higt_search.sh depth [seed]
zinc_higt_search.sh heads <best_depth> [seed]
zinc_higt_search.sh width <best_depth> <best_heads> [seed]
zinc_higt_search.sh lr <best_depth> <best_heads> <best_width> [seed]
zinc_higt_search.sh final <best_depth> <best_heads> <best_width> <best_lr> [seed]
```

The launcher uses `experiment=higt_zinc`, defaults to W&B project
`higt-zinc`, and uses `zinc-higt-*` groups and run names. Search phases keep
`test=false`; only the final selected configuration uses `test=true`.

It preserves strict argument validation, width/head divisibility checks,
`DRY_RUN=1`, W&B project/entity overrides, accelerator/device overrides, and
macOS `caffeinate` behavior. Contract tests inspect dry-run command strings
without executing TopoBench, loading ZINC, or contacting W&B.

## Development and verification policy

ZINC must not be executed during HIGT development. Specifically, development
must not:

- load a ZINC dataset or dataloader;
- run a ZINC model forward or backward pass;
- train or evaluate on ZINC;
- execute the `higt_zinc` experiment.

The ZINC configuration and launcher are delivered unexecuted for the user.
Runtime validation is limited to MUTAG and PROTEINS.

Unit tests establish:

1. exact typed relation metadata and edge directions;
2. separate relation aggregation buckets;
3. cardinality sensitivity, including doubling an identical incoming
   neighbor doubling its pre-update relation sum;
4. type-preserving permutation equivariance;
5. use of primitive incidence relations only;
6. finite gradients through attention, injective, and fusion parameters;
7. deterministic evaluation;
8. empty-rank and empty-relation behavior;
9. no cross-graph leakage after batching;
10. equality between processing a graph alone and inside a batch.

Pipeline tests establish Hydra composition, public `HIGT` instantiation, and
two-epoch CPU smoke runs on MUTAG and PROTEINS. All existing HGT tests must
continue to pass.

Launcher verification is limited to dry-run subprocess tests and `bash -n`.
It must not execute any generated ZINC command.

## Experimental comparison

The user will run the ZINC launcher. The first scientific comparison should
use identical lifting, width, depth, optimizer, seed set, and training
protocol for HGT and HIGT. Because HIGT adds parameters, results should report
parameter counts and include a subsequent parameter-matched HIGT width when
interpreting performance.

## Acceptance criteria

HIGT is ready for handoff when:

- the public class is named `HIGT`;
- HIGT uses only the four primitive incidence relations;
- no meta-path input or composed adjacency is required;
- relation-wise sum aggregates remain separate until target-rank update;
- all gradients are finite and reach both branches and fusion;
- batched graphs remain isolated;
- empty relations and rank-2 tensors are safe;
- MUTAG and PROTEINS smoke runs pass;
- existing `CellHGT` behavior and tests remain intact;
- the ZINC launcher passes dry-run contract and shell-syntax tests;
- no ZINC workload was executed during development.
