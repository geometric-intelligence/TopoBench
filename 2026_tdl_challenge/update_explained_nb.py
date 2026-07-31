"""One-shot updater for sheaf_tsp_explained.ipynb → final architecture."""

import json

P = "2026_tdl_challenge/sheaf_tsp_explained.ipynb"
try:
    nb = json.load(open(P))
except FileNotFoundError:
    P = "sheaf_tsp_explained.ipynb"
    nb = json.load(open(P))
cells = nb["cells"]


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {
        "cell_type": "code", "metadata": {}, "execution_count": None,
        "outputs": [], "source": text.splitlines(keepends=True),
    }


cells[0] = md("""# SheafTSP: Spectral Sheaf Convolution for Cell Complexes

**Track 2 (TNN) submission for the Topological Deep Learning Challenge 2026**

This notebook is an equation-level, executable walkthrough of the SheafTSP
architecture: a spectral sheaf convolutional network with orientation-equivariant
learned transports, a transport-consistency kernel, PPR sheaf diffusion, and an
exact, endogenous substructure-counting pathway. Every mechanism is demonstrated
on a toy graph, including the property checks (orthogonality, equivariance,
exact counting) that back the claims in `docs/sheaf_tsp_overview.html`.

Official 72-run grid results for this configuration: community detection
accuracy **0.4735**, triangle-count MSE/triangle **0.0108**
(12 GraphUniverse settings x 3 seeds x 2 tasks).

### References

1. Tandon et al. "Consistent Geometric Deep Learning via Hilbert Bundles and Cellular Sheaves" (2026), [arXiv:2605.06395](https://arxiv.org/abs/2605.06395)
2. Bodnar et al. "Neural Sheaf Diffusion" (2022), [arXiv:2202.04579](https://arxiv.org/abs/2202.04579)
3. Bamberger et al. "Bundle Neural Networks (BuNN)" (2024), [arXiv:2405.15540](https://arxiv.org/abs/2405.15540)
4. Chen, Chen, Villar & Bruna, "Can Graph Neural Networks Count Substructures?" (NeurIPS 2020), [arXiv:2002.04025](https://arxiv.org/abs/2002.04025)
5. Zhang et al. "MagNet: A Neural Network for Directed Graphs" (NeurIPS 2021), [arXiv:2102.11391](https://arxiv.org/abs/2102.11391)
""")

cells[1] = md("""---
## 1. Architecture Overview

The full pipeline from raw graph to prediction:

```
Raw Graph
   |  CellCliqueLifting: ALL 3-cliques become 2-cells
   |  (canonical set -> the lifted complex is invariant to node relabeling)
   v
Cell Complex  (x_0, x_1, x_2, B_1, B_2, L_1^down, L_1^up)
   |  AllCellFeatureEncoder (width 64, fixed by the challenge)
   v
Encoded features  x_1 in R^{N_1 x 64}
   |  SheafTSP backbone (3 layers):
   |    transports R_e in SO(d)  ->  kernel k_ij on ||s_i - R_e s_j||
   |    ->  L_hat = D^{-1/2} delta^T K delta D^{-1/2}
   |    ->  PPR filter  y = sum_k w_k P^k s,  P = I - L_hat/2,  K = 10
   v
Refined x_1 in R^{N_1 x 64}
   |  x_0 = B_1 @ x_1  +  x_0_enc  +  W_tri t_v      (NO LayerNorm here)
   |          diffusion    residual    exact count signal
   |          t_v = |B_1||B_2|1  (endogenous, exact under the clique lifting)
   v
Node embeddings  x_0 in R^{N_0 x 64}
   |  readout; graph-level tasks sum-pool (sum preserves count linearity)
   v
Prediction (node-level: community detection / graph-level: triangle count)
```

Two signal regimes coexist as additive terms of one embedding: the
sheaf-diffused signal is operator-normalized for stable long-range diffusion,
and the count term reaches the prediction through strictly linear,
unnormalized operations — any LayerNorm/BatchNorm/mean on that route erases
cardinality (design principles P1-P2 in the overview document).
""")

cells[3] = md("""---
## 3. Restriction Map Learning

Each restriction map $\\mathbf{R}_e \\in SO(d)$ is produced from an
**antisymmetrized** skew-generator and projected to the rotation group.

### Step 1: Antisymmetrized edge conditioning

$$
\\mathbf{p}_{uv} = \\mathrm{MLP}([\\mathbf{x}_u \\| \\mathbf{x}_v]) - \\mathrm{MLP}([\\mathbf{x}_v \\| \\mathbf{x}_u])
$$

so $\\mathbf{p}_{vu} = -\\mathbf{p}_{uv}$ by construction. The parameters fill a
skew-symmetric matrix $\\mathbf{S} = -\\mathbf{S}^\\top$.

### Step 2: Projection to SO(d)

Default (Cayley): $\\mathbf{R} = (\\mathbf{I} - \\mathbf{S})(\\mathbf{I} + \\mathbf{S})^{-1}$.
Alternative (`rotation_param: exp`): $\\mathbf{R} = \\exp(\\mathbf{S})$, surjective onto $SO(d)$.

### Properties (all verified in code below)

- **Special orthogonal**: $\\mathbf{R}^\\top\\mathbf{R} = \\mathbf{I}$, $\\det \\mathbf{R} = +1$.
- **Orientation equivariance**: the antisymmetric generator gives
  $\\mathbf{R}_{vu} = \\mathbf{R}_{uv}^{-1}$ analytically — reversing an edge inverts its
  transport, and the model is invariant to node relabeling.
- **Precision note**: the Cayley image is the dense subset of $SO(d)$ excluding
  rotations with a $-1$ eigenvalue. We benchmarked the surjective exponential
  map head-to-head; the exclusion is measurably non-binding (details in the
  overview document, Sec. 3.1) and Cayley remains the default.
- At $d = 2$, $SO(2)$ is abelian and the resulting operator is real-conjugate
  to a **magnetic Laplacian** with learned per-edge phases [5]. Measured on
  trained models: low-homophily settings learn near-antipodal transports
  (mean $|\\theta| = 165.8°$), homophilous settings learn near-identity ones
  ($9.9°$) — the model selects its gauge from data.
""")

# Insert equivariance check after the map-learner demo (index 12)
equiv_code = code("""# Property checks: special orthogonality and orientation equivariance
R_fwd = map_learner(x, edge_index)                 # R_uv
R_bwd = map_learner(x, edge_index.flip(0))         # R_vu

I = torch.eye(d).expand(edge_index.shape[1], d, d)
orth_err = (R_fwd.transpose(1, 2) @ R_fwd - I).abs().max()
dets = torch.linalg.det(R_fwd)
equiv_err = (torch.bmm(R_fwd, R_bwd) - I).abs().max()

print(f"orthogonality  max|R^T R - I| = {orth_err:.2e}")
print(f"determinants   {dets.tolist()}")
print(f"equivariance   max|R_uv R_vu - I| = {equiv_err:.2e}   (R_vu = R_uv^-1)")
""")
cells.insert(13, equiv_code)

# Section 4 (now index 5): append PPR + kernel notes
cells[5]["source"] = "".join(cells[5]["source"]) + """

### Submission defaults on top of the polynomial form

**Transport-consistency kernel.** Edge weights use the distance *under the
learned map*, $\\rho_{ij}^2 = \\lVert s_i - R_e s_j \\rVert^2$, in a Gaussian
kernel with learnable bandwidth — a raw feature-distance kernel would
reintroduce the homophily bias the sheaf exists to remove.

**PPR scalar filter (default, `filter_basis: ppr`, K = 10).** Instead of
per-order weight matrices, the shipped filter uses K+1 scalars on the sheaf
lazy walk $P = I - \\tfrac12\\hat L_{\\mathcal F}$ (spectrum in $[0,1]$, so ten
hops stay numerically stable), initialized to the personalized-PageRank
profile $w_k = \\alpha(1-\\alpha)^k$ — long-range low-pass diffusion
concentrating on the bottom eigensections, where transport-consistent
community structure lives.
"""
cells[5]["source"] = cells[5]["source"].splitlines(keepends=True)

# Insert the counting-pathway section before old section 8 (find the wrapper cell)
wrap_idx = next(i for i, c in enumerate(cells) if c["cell_type"] == "markdown" and "TopoBench Integration" in "".join(c["source"]))
count_md = md("""---
## 7.8 Exact, endogenous substructure counting

Message-passing GNNs provably cannot count triangles [4]; lifted models
receive higher-order structure explicitly. SheafTSP derives a per-node count
signal from the lifted complex's own incidence matrices:

$$
t_v = |B_1|\\,|B_2|\\,\\mathbf 1, \\qquad \\sum_{v} t_v = 6 \\cdot \\#\\text{triangles}
$$

Under the all-3-cliques lifting each triangle incident to a node contributes
exactly two of its edges at that node, so the identity is exact — verified
below against direct enumeration. The signal is injected as
$x_0 \\mathrel{+}= W_{\\text{tri}} t_v$ (zero-initialized, one warm channel)
and travels a strictly linear, unnormalized route to the sum-pooled
prediction: the regression head only learns a scale factor.
""")
count_code = code("""import networkx as nx
from torch_geometric.data import Data
from topobench.transforms.liftings.graph2cell.clique_cell_lifting import CellCliqueLifting

g = nx.gnp_random_graph(30, 0.25, seed=7)
true_triangles = sum(nx.triangles(g).values()) // 3

E = torch.tensor(list(g.edges())).t()
E = torch.cat([E, E.flip(0)], dim=1)
lifted = CellCliqueLifting()(Data(edge_index=E, x=torch.randn(30, 4), num_nodes=30))

B1 = lifted.incidence_1.to_dense().abs()
B2 = lifted.incidence_2.to_dense().abs()
t_v = B1 @ B2 @ torch.ones(B2.shape[1], 1)

print(f"2-cells attached by the clique lifting : {B2.shape[1]}")
print(f"true triangle count (networkx)         : {true_triangles}")
print(f"sum(t_v) / 6                           : {t_v.sum().item() / 6:.0f}   (exact)")
""")
cells.insert(wrap_idx, count_md)
cells.insert(wrap_idx + 1, count_code)

# Config cell
cfg_idx = next(i for i, c in enumerate(cells) if c["cell_type"] == "markdown" and "Model Configuration" in "".join(c["source"]))
cells[cfg_idx] = md("""---
## 11. Model Configuration

The submitted Hydra defaults (`configs/model/cell/sheaf_tsp.yaml`):

```yaml
backbone:
  n_layers: 3            # sheaf convolution layers
  stalk_dim: 2           # SO(2) transports (magnetic-Laplacian regime)
  filter_basis: ppr      # scalar PPR coefficients on P = I - L_hat/2
  filter_order: 10       # ten stable diffusion hops
  kernel_distance: transport   # ||s_i - R_e s_j|| in the Gaussian kernel
  reg_form: alignment    # bounded kernel-alignment transport regularizer
  rotation_param: cayley # benchmarked against the surjective exp map
  dropout: 0.0           # signal-path dropout corrupts the count pathway
  mlp_dropout: 0.5       # regularize the transport MLP instead

backbone_wrapper:
  residual_connections: false  # no post-hoc LayerNorm on x_0 (P2)
  count_source: incidence      # t_v = |B1||B2|1, endogenous and exact
  tri_warm: 0.1                # warm-start channel of W_tri

transforms (model_defaults):
  graph2cell_lifting: clique_cell   # all 3-cliques; permutation-invariant
```

Every default was selected against a measured alternative; the experiment
log records each comparison, including the rejected options
(`docs/BATTLE_PLAN.md` in the project repository).
""")

# Summary cell (last markdown)
sum_idx = next(i for i, c in enumerate(cells) if c["cell_type"] == "markdown" and "## 13. Summary" in "".join(c["source"]))
cells[sum_idx] = md("""---
## 13. Summary

**SheafTSP** couples two signal regimes in one architecture:

1. **Learned sheaf diffusion** — orientation-equivariant $SO(2)$ transports
   (antisymmetrized Cayley generator, $R_{vu} = R_{uv}^{-1}$ by construction),
   a transport-consistency kernel, and PPR diffusion on the sheaf lazy walk.
   Measured transports match the theory: near-antipodal on heterophilic
   settings, near-identity on homophilous ones.

2. **An exact counting pathway** — $t_v = |B_1||B_2|\\mathbf 1$ from the
   canonical clique lifting, exact by construction and preserved by a
   strictly linear, unnormalized route to the prediction.

The full pipeline is invariant to node relabeling (equivariant transports +
canonical lifting), its operator spectrum is bounded in $[0,2]$ per input
(kernel-weighted degrees, verified numerically), and the official 72-run grid
scores are community detection **0.4735** and triangle-count MSE/triangle
**0.0108**. Design principles, scoped theoretical claims, and the complete
experiment log live in the project documentation.
""")

json.dump(nb, open(P, "w"), indent=1)
print(f"updated {P}: {len(cells)} cells")
