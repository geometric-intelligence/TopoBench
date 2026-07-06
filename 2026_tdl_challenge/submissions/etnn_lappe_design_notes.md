# Track 2 Submission: ETNN-LapPE Design Notes

Participant: Gaurav Khanal
Track: Track 2 - Topological Neural Networks
Team: E(n)igma
Model config: `combinatorial/etnn_lappe`
Related baseline: PR #320, `combinatorial/etnn`
Status: Full GraphUniverse evaluation completed

## Scope

This coordinate-enabled follow-up builds on the TopoBench-native ETNN baseline
from PR #320. The baseline implements the rank-wise topological
feature-message-passing structure of E(n)-Equivariant Topological Neural
Networks (ETNNs) for GraphUniverse data, while omitting coordinate-dependent
geometric terms because GraphUniverse does not provide physical Euclidean
coordinates.

This submission adds fixed Laplacian positional encodings (LapPE) as structural
pseudo-coordinates before graph-to-combinatorial lifting. The ETNN feature
update still follows the neighborhood aggregation and rank-wise update used in
the coordinate-free baseline, but each relation message also receives a
structural distance:

```text
p_0 = structural rank-0 coordinates
p_r = mean_{d incident to c} p_{r-1,d}
z_{d,c,N} = concat(h_d, h_c, a_{d,c,N}, ||p_d - p_c||^2)
m_{c,N} = sum_{d in N(c)} psi_N(z_{d,c,N})
h'_c = h_c + beta_rank(c)(h_c, concat_N m_{c,N})
```

This corresponds to the ETNN/CCMPN neighborhood aggregation in Eq. 1-3 and the
feature update in Eq. 6 of the ETNN paper. The coordinate update from Eq. 7 is
not applied. Coordinates are fixed structural embeddings of the graph, not
physical positions.

In implementation, incoming relation messages are concatenated rank-wise
according to the configured neighborhood order. If a relation is empty in a
mini-batch, it contributes a zero tensor so that the rank-specific update MLP
input dimension remains fixed across cells and batches.

This design is motivated by the GraphUniverse benchmark. GraphUniverse
generates inductive graph families from a hierarchical degree-corrected
stochastic block model with persistent semantic communities and controlled
homophily, degree, degree-separation, graph-size, and feature-noise parameters.
The benchmark does not provide physical coordinates, and its generator is
structural/community-based rather than physical-geometric. LapPE is therefore
used as a conservative graph-derived coordinate frame, not as a claim that the
original GraphUniverse graphs have physical Euclidean geometry.

## Coordinate Policy

`combinatorial/etnn_lappe` uses normalized graph Laplacian eigenvectors:

```text
p_v = LapPE(v)
```

Rank-0 coordinates are lifted to higher-rank cells by recursive incidence
averaging:

```text
p_r(c) = mean_{d incident to c} p_{r-1}(d)
```

The squared-distance feature `||p_d - p_c||^2` is invariant to rigid
transformations of the chosen structural coordinate frame. This makes the
message update insensitive to translations, rotations, and reflections applied
consistently to all structural coordinates within the same graph. It does not
imply that the original GraphUniverse graphs possess an underlying physical
E(n) action.

This distance-only use of LapPE also handles common spectral-coordinate
ambiguities. Global sign flips of individual Laplacian eigenvectors correspond
to reflections of the structural coordinate frame, and any orthogonal change of
basis preserves pairwise squared distances when applied consistently within the
graph:

```text
||Q p_d - Q p_c||^2 = ||p_d - p_c||^2, where Q^T Q = I.
```

## Design Choices

- Coordinates are computed before graph-to-combinatorial lifting and stored
  separately from node features with `concat_to_x=false`.
- Rank-wise cell features remain handled by `AllCellFeatureEncoder`.
- The backbone uses the same TopoBench neighborhoods as the coordinate-free
  ETNN baseline.
- Relation message MLPs receive two scalar relation features: the sparse
  TopoBench neighborhood value and the squared structural distance.
- Coordinates are fixed auxiliary inputs; only cell features are updated.
- LapPE is used as a conservative default coordinate policy, not as a claim
  that it is universally optimal. Other structural coordinates, such as
  diffusion coordinates, shortest-path/MDS embeddings, SCORE-style spectral
  coordinates, or learned coordinate refinements, may be preferable for other
  datasets.
- The implementation is intentionally conservative: no learned coordinate
  generator, no coordinate update, and no multi-view coordinate aggregation.

## Evaluation

ETNN-LapPE completed the official 72-run GraphUniverse evaluation protocol with
seeds `42`, `43`, and `44`.

Headline in-distribution metrics:

| Model config | Community accuracy | Triangle MSE / triangles |
| --- | ---: | ---: |
| `combinatorial/etnn_lappe` | `0.454471 +- 0.130901` | `0.114699 +- 0.082613` |

Homophily slices:

| Model config | Task | h_lo | h_mid | h_hi |
| --- | --- | ---: | ---: | ---: |
| `combinatorial/etnn_lappe` | Community accuracy | `0.319887 +- 0.005597` | `0.422062 +- 0.029232` | `0.621463 +- 0.058902` |
| `combinatorial/etnn_lappe` | Triangle MSE / triangles | `0.028061 +- 0.033176` | `0.130757 +- 0.054071` | `0.185280 +- 0.060866` |

Compared with the coordinate-free baseline from PR #320, ETNN-LapPE keeps
community detection essentially tied while improving global triangle-counting
error and reducing triangle-counting variance.

## Tests

The coordinate-enabled implementation includes tests for:

- structural-coordinate ETNN forward execution without physical `pos`;
- missing coordinate attributes and malformed rank-0 coordinate rows;
- incidence-based coordinate lifting;
- empty higher-rank coordinate tensors;
- rigid-motion invariance of squared structural distances;
- Hydra composition for the LapPE model config.

Verified locally:

```bash
uv run pytest \
  test/nn/backbones/combinatorial/test_etnn.py \
  test/nn/backbones/combinatorial/test_etnn_lappe.py \
  test/pipeline/test_etnn_pipeline.py -q
```

Result:

```text
focused tests passed locally; full GraphUniverse evaluation completed
```

## References

- Claudio Battiloro, Ege Karaismailoglu, Mauricio Tec, George Dasoulas,
  Michelle Audirac, Francesca Dominici. E(n) Equivariant Topological Neural
  Networks. arXiv:2405.15429.
- Official ETNN implementation:
  https://github.com/NSAPH-Projects/topological-equivariant-networks
- Louis Van Langendonck, Guillermo Bernardez, Nina Miolane, Pere Barlet-Ros.
  GraphUniverse: Synthetic Graph Generation for Evaluating Inductive
  Generalization. ICLR 2026. arXiv:2509.21097.
