# ETNN Design Notes

Participant: Gaurav Khanal
Track: Track 2 - Topological Neural Networks
Team: E(n)igma
Model config: `combinatorial/etnn`
Status: Full GraphUniverse evaluation completed

## Scope

This submission adds a TopoBench-native combinatorial ETNN backbone for lifted
GraphUniverse inputs. The implementation follows the rank-wise topological
feature-message-passing structure of E(n)-Equivariant Topological Neural
Networks (ETNNs), while omitting coordinate-dependent geometric terms because
GraphUniverse does not provide physical Euclidean coordinates.

The original ETNN formulation combines:

- combinatorial-complex neighborhood message passing over cells;
- geometric invariants computed from node coordinates;
- an E(n)-equivariant coordinate update.

GraphUniverse does not provide physical coordinates. For that reason, this
implementation is intentionally coordinate-free: it implements the ETNN
combinatorial message-passing core over TopoBench neighborhoods, but it does not
claim to implement the full coordinate-update component of ETNN on these
datasets. This adaptation keeps the coordinate extension point explicit while
remaining compatible with TopoBench's graph-to-combinatorial lifting pipeline.

## Design Choices

### Topological Domain

Track 2 submissions are expected to operate on a topological domain. ETNN is
naturally defined over cells and typed cell neighborhoods, so this
implementation uses TopoBench's `combinatorial` domain.

The model config applies `GraphTriangleInducedCC`, which lifts graph datasets
into combinatorial complexes with:

- rank-0 cells: original graph vertices;
- rank-1 cells: graph edges;
- rank-2 cells: induced triangles / 2-cells.

This lift is a practical match for the GraphUniverse tasks: community detection
can use rank-0 and rank-1 relational structure, while triangle counting benefits
from explicit rank-2 cells.

### Feature Representation

TopoBench stores one feature matrix per cell rank:

```text
x_0: rank-0 cell features
x_1: rank-1 cell features
x_2: rank-2 cell features
```

`AllCellFeatureEncoder` projects the selected ranks to a common hidden
dimension before the ETNN backbone. This keeps the backbone simple: each
relation-specific message MLP can assume that sender and receiver states live in
the same feature dimension.

### Typed ETNN Relations

The ETNN paper updates cell states by aggregating messages over typed
neighborhoods. In TopoBench, these neighborhoods are sparse matrices. The
submitted config uses:

```text
up_adjacency-0    rank 0 -> rank 0
up_adjacency-1    rank 1 -> rank 1
up_adjacency-2    rank 2 -> rank 2
up_incidence-0    rank 0 -> rank 1
down_incidence-1  rank 1 -> rank 0
up_incidence-1    rank 1 -> rank 2
down_incidence-2  rank 2 -> rank 1
```

Each relation has its own message MLP. Each destination rank has its own update
MLP. This mirrors ETNN's typed neighborhood design while fitting TopoBench's
existing sparse-neighborhood interface.

### Equation-to-Code Mapping

The coordinate-free backbone implements the ETNN feature update in the following
form. For a sender cell `d`, receiver cell `c`, and neighborhood type `N`:

```text
z_{d,c,N} = [h_d, h_c, a_{d,c,N}]
m_{c,N}   = sum_{d in N(c)} phi_N(z_{d,c,N})
h'_c      = h_c + psi_rank(c)(h_c, m_{c,N_1}, ..., m_{c,N_k})
```

where:

- `h_d` and `h_c` are sender and receiver cell states;
- `a_{d,c,N}` is the scalar value stored in the sparse TopoBench neighborhood;
- `phi_N` is the relation-specific message MLP;
- `psi_rank(c)` is the rank-specific update MLP.

This corresponds to the rank-wise topological feature-message-passing structure
of ETNN. The coordinate-dependent geometric invariant and coordinate-update
terms are omitted because GraphUniverse does not provide physical coordinates.

In implementation, the concatenation in `psi_rank(c)` is performed rank-wise
using the configured incoming relation list for each destination rank. Empty
relations contribute zero tensors, so the update MLP input dimension remains
fixed across cells and mini-batches.

### Sparse Direction Convention

TopoBench stores sparse neighborhoods as matrices whose rows index receiver
cells and columns index sender cells. PyG-style message passing expects
`edge_index = [sender, receiver]`. The backbone therefore flips sparse matrix
indices when constructing explicit edge indices.

This direction convention is covered by unit tests because reversing it would
silently change the model semantics.

### Empty-Rank and Placeholder Handling

GraphUniverse mini-batches can contain complexes with no cells at a requested
rank, especially no rank-2 cells. During batching, sparse tensors may still
carry placeholder axes or explicitly stored zero values. The implementation:

- removes stored-zero sparse entries before message passing;
- compacts placeholder sparse axes to the real feature rows when batch metadata
  is available;
- raises a clear error when sparse-axis compaction is ambiguous.

This keeps the model robust on lifted graph batches without silently truncating
or misaligning cell features.

### Wrapper and Readout

The backbone returns a dictionary keyed by cell rank:

```text
{0: x_0_out, 1: x_1_out, 2: x_2_out}
```

That output contract is compatible with TopoBench's existing combinatorial
`TuneWrapper`. The submitted config then uses `PropagateSignalDown` so
higher-rank information can contribute to rank-0 predictions for graph-level
tasks.

### Device Behavior

All tensors created inside the backbone follow the active feature tensor's
device and dtype. The implementation therefore supports CPU execution for local
development and CUDA execution for larger GraphUniverse evaluations without a
separate code path.

### GraphUniverse Setting

GraphUniverse provides graph families generated from controlled community,
homophily, degree, degree-separation, graph-size, and feature-distribution
parameters. These inputs are structural rather than physical-geometric. The
coordinate-free ETNN baseline is therefore a conservative adaptation for this
evaluation setting: it preserves the typed topological message-passing structure
of ETNN without inventing physical coordinates that are absent from the data.

## Evaluation

The official GraphUniverse evaluation completed all 72 runs.

Metadata:

- study id: `2026-06-12_10-14-24`
- model config: `combinatorial/etnn`
- seeds: `42`, `43`, `44`
- output directory: `2026_tdl_challenge/outputs/etnn`

Headline in-distribution metrics from `results.json`:

| Task | Metric | Mean +/- std | Runs |
| --- | --- | ---: | ---: |
| Community detection | Accuracy | `0.4534 +/- 0.1308` | 36 |
| Triangle counting | MSE / total triangles | `0.1213 +/- 0.1121` | 36 |

Homophily slices:

| Task | h_lo | h_mid | h_hi |
| --- | ---: | ---: | ---: |
| Community accuracy | `0.3195 +/- 0.0045` | `0.4205 +/- 0.0316` | `0.6203 +/- 0.0584` |
| Triangle MSE / triangles | `0.0259 +/- 0.0289` | `0.1141 +/- 0.0394` | `0.2239 +/- 0.1251` |

## Tests

The implementation includes unit and integration coverage for:

- rank-wise output shapes;
- no-coordinate GraphUniverse compatibility;
- `TuneWrapper` compatibility;
- sparse neighborhood direction;
- empty-rank sparse placeholder compaction;
- Hydra composition with graph-to-combinatorial lifting;
- required `test/pipeline/test_pipeline.py` end-to-end pipeline test;
- optional one-epoch ETNN-specific smoke test.

Verified locally before submission:

```bash
uv run pytest test/nn/backbones/combinatorial/test_etnn.py test/pipeline/test_etnn_pipeline.py -q
```

Result:

```text
8 passed, 1 skipped, 1 warning
```

The skipped test is an explicit one-epoch pipeline smoke test that may
download/process data and is intended for manual execution when the environment
is available.

The required challenge pipeline test in `test/pipeline/test_pipeline.py` uses
`dataset=graph/MUTAG` and `model=combinatorial/etnn`.

## References

- Claudio Battiloro, Ege Karaismailoglu, Mauricio Tec, George Dasoulas,
  Michelle Audirac, Francesca Dominici. E(n) Equivariant Topological Neural
  Networks. arXiv:2405.15429.
- Official ETNN implementation:
  https://github.com/NSAPH-Projects/topological-equivariant-networks
