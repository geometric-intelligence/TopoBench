# DPHGNN component ablation

Supplementary analysis for the 2026 TDL Challenge, Track 2 (team: *Oversmooth
operators*, model: DPHGNN, arXiv:2405.16616). This file is the
human-readable summary: question, protocol, and results. The full
authoritative spec is `conf/E2_CLAUDE.md` (repo root).

## Question

DPHGNN's claimed contribution is a **dual spatial-spectral** formulation over
hypergraphs, with structural fusion across expansions (clique, star,
HyperGCN). This is a mechanistic follow-up to the lifting-confounding study
(`../lifting_confounding_study/`, Experiment E1): E1 asked whether TNN-vs-GNN
comparisons are confounded by the lifting; E2 asks, given the architecture,
**which component actually produces the capability, and does it do so in
every structural regime or only some?**

- **H1.** Removing either branch (spatial or spectral TAA), the SIB block,
  or two of the three expansions degrades performance -- i.e. the duality is
  real, not decorative.
- **H2 (the interesting one).** The *fusion mechanism* matters, not merely
  the presence of both branches. Formally: `A3` (naive sum fusion) is
  meaningfully worse than `A0` (learned gate). If `A0` ≈ `A3`, the paper's
  contribution reduces to "use both branches", and the gate is unjustified
  complexity.
- **H3.** Component importance is **regime-dependent** -- e.g. the spectral
  branch matters under heterophily but not homophily. This ties E2 back to
  the challenge's Structural Sensitivity question.

A null result on any of these is a legitimate, reportable finding. This
study does not tune until a hypothesis is confirmed.

## Ablation arms

Five keyword-only flags were added to `DPHGNN.__init__`
(`topobench/nn/backbones/hypergraph/dphgnn.py`), all defaulting to the
paper-faithful architecture:

| Arm | Name | Flag(s) | What is disabled | Hypothesis |
|---|---|---|---|---|
| `A0` | full | (defaults) | nothing -- paper-faithful reference | -- |
| `A1` | spatial_only | `use_spectral=False` | spectral TAA branch output (Eq. 5.4, `x_Z`) zeroed before fusion | H1 |
| `A2` | spectral_only | `use_spatial=False` | spatial TAA branch output (Eq. 5.3, `x_kappa`) zeroed before fusion | H1 |
| `A3` | fusion_sum | `fusion="sum"` | learned multiplicative gate (Eq. 2) replaced by elementwise sum | **H2** |
| `A4` | no_sib | `use_sib=False` | SIB block's Laplacian term (Eq. 6.1) disabled -- reduces to its `lambda -> 0` residual | H1 |
| `A5` | clique_only | `expansions=("clique",)` | star + HyperGCN expansions dropped; clique view stands in for both | H1 |

### Masking strategy (E2_CLAUDE.md Sec. 2.1) -- decision and caveats

**Default to output masking, not deletion.** Every submodule (`clique_conv`,
`star_conv`, `hyp_conv`, `taa_spatial`, `taa_spectral`, `sib`,
`feature_mixture`, `dff_layers`, `output_layer`) is constructed regardless of
the ablation flags. Only the *forward pass* branches differently. Consequence:
**`n_params` is identical across every arm** (verified:
`52,064` trainable parameters for `A0`-`A5` alike, at `hidden_channels=64`,
via the preflight check below). A reader must not infer that a masked arm is
a smaller model -- some of its parameters simply receive no gradient for that
arm.

Two consequences flagged explicitly by `E2_CLAUDE.md` and handled as follows:

- **A1/A2 and the gate.** In this implementation the multiplicative gate in
  Eq. (2) is derived entirely from the SIB block's output (`X_spectral`), not
  from either TAA branch. So `use_spectral=False` / `use_spatial=False`
  zeroing `x_Z`/`x_kappa` does **not** halve the gate the way the generic
  `sigma(W . x_spec) (*) x_spat` example in `E2_CLAUDE.md` Sec. 2.1 warns
  about -- that formula does not describe this codebase's actual wiring. The
  real (smaller) caveat: `MLP_1` in Eq. (7.1) has a learned bias and shares
  weights across both halves of its concatenated `[x_kappa | x_Z]` input, so
  a small bias-driven signal from the "disabled" branch can still reach
  `X_eqv`. This is documented in the `use_spectral`/`use_spatial` docstrings.
- **A5 is not maskable.** Implemented as a genuine branch in `forward`
  (`DPHGNN.forward`, `topobench/nn/backbones/hypergraph/dphgnn.py`): when
  `"star"` is excluded from `expansions`, the star-conv branch is skipped
  entirely and both the TAA query view and the Dynamic Feature Fusion
  `S -> E` term fall back to / drop out in favor of the clique view (`X_E =
  M_V` alone, no supernode term). When `"hypergcn"` is excluded, the TAA
  value view falls back to the clique view likewise. `"clique"` is
  structurally required (it defines the TAA neighborhood, decision D-3) and
  is rejected with `ValueError` if omitted.

## Experimental design

**Data cells.** Two official cells, contrasted on homophily -- the axis H3
predicts will matter:

| Cell key | Why |
|---|---|
| `h_lo__d_lo__pl_lo` | severe heterophily |
| `h_hi__d_lo__pl_lo` | strong homophily |

Obtained via `build_generation_parameters()` from `2026_tdl_challenge/
utils.py`. These are the *same two cells* used by Experiment E1 for its
`gcn` arm, so `fig4_component_by_regime.png` can plot the E1 GCN baseline as
a like-for-like reference line.

**Budget.**
- **Phase 1 (must complete first):** 6 arms x 2 cells x seed 42 = **12
  runs**. Qualitative only -- no significance testing, raw deltas only.
- **Phase 2 (only once Phase 1 is fully green):** adds seeds 43 and 44 = **36
  runs total**. Enables bootstrap CIs.

**Task:** community detection only (`dataset=graph/graphuniverse_inductive`),
metric: accuracy (higher is better). Triangle counting is out of scope for
E2.

**Model:** always `hypergraph/dphgnn`, i.e. every arm shares the *same*
model config; only the ablation-flag Hydra overrides differ. Unlike
Experiment E1 (which held the model fixed and swept the lifting), E2 holds
the lifting fixed (repo default: `HypergraphKHopLifting`, `k=1` -- confirmed
empirically below, and identical to what the official `results.json` uses)
and sweeps the model's internal ablation flags.

## Comparability with the official pipeline

`run_e2.py` reuses `utils.CHALLENGE_GRID_HYDRA_OVERRIDES`, `utils.MAX_EPOCHS`
and `utils.apply_challenge_feature_encoder_out_channels` unchanged, so
`max_epochs`, early-stopping patience, `check_val_every_n_epoch`, and
`feature_encoder.out_channels` exactly match the official run. The
`graph2hypergraph` lifting is **never overridden**: it is left at whatever
`get_default_transform` resolves to for a `hypergraph/*` model on a
`graph/*` dataset, which was confirmed (by direct `hydra.compose`, see
"Verified Hydra override strings" below) to be `HypergraphKHopLifting`,
`k=1` -- the same lifting the official submission's `results.json` uses.
Three deliberate divergences (identical to Experiment E1's, for the same
reasons):

- **`logger=csv` instead of `wandb`** -- this is an internal supplementary
  analysis, not the official submission.
- **`dataset.dataloader_params.num_workers=0` / `persistent_workers=False`,
  always** -- avoids DataLoader worker subprocesses outliving a single run
  across a long sequential sweep (see Experiment E1's README for the crash
  this caused there). Does not change trained weights.
- **OOM fallback** -- if a run OOMs, `run_e2.py` automatically retries with
  the dataloader `batch_size` reduced (64 -> 32 -> 16). The affected record
  gets a `batch_size_override` field and a `note` explaining the change
  breaks strict comparability for that one run.

## Verified Hydra override strings

**None of the five ablation flags exist as keys in `configs/model/hypergraph/
dphgnn.yaml`** (that file lists `hidden_channels`, `n_gnn_layers`,
`taa_heads`, `dropout`, `sib_lambda`, `n_dff_layers`, `supernode_init`,
`with_mediators`, `taa_neighborhood` -- not the new flags, since it was left
unmodified per `E2_CLAUDE.md` Sec. 3: "Create no new files under
`configs/model/`", read together with the instruction not to modify that
file at all). Consequently, a plain `model.backbone.use_spectral=false`-style
override -- the literal form `E2_CLAUDE.md` Sec. 3 illustrates -- **fails**
under OmegaConf's struct mode with `Could not override 'model.backbone.
use_spectral'. To append to your config use +model.backbone.use_spectral=
false`. This was verified empirically (see item 6 in the report to the
session's caller) before trusting it in `run_e2.py`; the leading `+` is
required for all five flags:

```text
A0 full:           (no override)
A1 spatial_only:    +model.backbone.use_spectral=false
A2 spectral_only:   +model.backbone.use_spatial=false
A3 fusion_sum:       +model.backbone.fusion=sum
A4 no_sib:           +model.backbone.use_sib=false
A5 clique_only:      +model.backbone.expansions=[clique]
```

`run_e2.py`'s `preflight_check()` re-verifies this automatically before every
sweep, the same way Experiment E1's `preflight_check()` did for its lifting
overrides -- except cheaper: since the ablation flags live on the backbone
constructor itself (not on a `transforms` block that needs a full dataset to
observe), the preflight only needs a toy 6-node/4-hyperedge incidence matrix
(the same fixture used by `test_dphgnn_ablation.py`) and never touches the
GPU or builds a GraphUniverse dataset. It checks, per arm:

1. `hydra.utils.instantiate(cfg.model.backbone)` produces a `DPHGNN` whose
   ablation-flag attributes match the arm's intended values.
2. On the same toy hypergraph, every non-`A0` arm's forward output differs
   from `A0`'s (the same "did the override silently no-op" guard E1 used for
   its lifting overrides, applied here to the ablation flags instead).

Confirmed (`python run_e2.py --skip-preflight` replaced by running
`preflight_check()` standalone during development): all six arms produce
`n_params=52064` and pairwise-distinct outputs from `A0` on the toy fixture.

## Local testing

**Never run this script locally expecting real numbers without `--cpu
--smoke-test` first**, for the same reason as Experiment E1:
`configs/trainer/default.yaml` hardcodes `accelerator: gpu, devices: [0]`
with no CPU/"auto" fallback.

```bash
# Tiny synthetic data (20 graphs), 2 epochs, CPU-forced. Validates the
# script mechanics in well under a few minutes. Never persists to
# e2_ablation_results.json.
python run_e2.py --smoke-test --cpu
```

Verified during development: a full 12-job `--smoke-test --cpu` sweep
completed end to end (subprocess-per-job, resumable JSON append logic,
OOM-fallback ladder wiring, RAM logging before/after each job) with every
job returning `status: "ok"` in a few seconds each. Real Phase-1/Phase-2
runs (`python run_e2.py` / `--phase2`, no `--smoke-test`) were deliberately
**not** launched from this development environment -- no GPU was available
locally, and the repo's own accelerator default assumes one. That sweep is
reserved for a Kaggle GPU session; see `E2_kaggle_run.ipynb`.

## Known risks

1. **Silent no-op flag.** The single highest-probability failure mode
   (E2_CLAUDE.md Sec. 7). Covered by `test_dphgnn_ablation.py`'s
   `test_arm_output_differs_from_a0` (unit level, toy fixture) and
   `run_e2.py`'s `preflight_check()` (integration level, through the real
   Hydra/`hydra.utils.instantiate` path) -- both currently pass.
2. **Default-path drift.** The highest-*cost* failure: it would damage the
   primary Correctness score. Covered by
   `test_dphgnn_ablation.py::TestDefaultPathRegression`, which locks
   `DPHGNN`'s default-flags forward output (and `n_params`) to values
   generated from the pre-ablation code, before any flag was added. Still
   passes after every flag was implemented.
3. **Arm/architecture mismatch.** Ruled out for all six arms during design
   (see "Masking strategy" above) -- each maps onto a distinct, documented
   piece of the implementation (`DPHGNN_SPEC.md` Sec. 4-9). No arm required
   improvising an interpretation of the paper.
4. **A1/A2 gate interaction.** Decided and documented above and in the
   `use_spectral`/`use_spatial` docstrings: this implementation's gate does
   not depend on either TAA branch, so the generic warning in
   `E2_CLAUDE.md` Sec. 2.1 does not literally apply here; the real (smaller)
   caveat is `MLP_1`'s shared bias.
5. **Runtime.** A single real (non-smoke-test) run was not measured locally
   (no GPU). Watch the first few Kaggle runs' wall-clock time against the
   ~15-minute budget in `E2_CLAUDE.md` Sec. 7 before trusting an unattended
   sweep.

## Statistics

- **Phase 1 (n=1 seed):** no significance testing. *Single seed; differences
  under ~2 accuracy points are not interpretable.* Report raw deltas only.
- **Phase 2 (n=3 seeds):** paired comparison of each arm against `A0`
  *within cell*; bootstrap 95% CIs on the deltas (not ±std -- three seeds do
  not support an SD estimate). For H3, the quantity of interest is the
  **difference of deltas** across cells: `Δ_arm(h_lo) − Δ_arm(h_hi)`,
  reported with a bootstrap CI -- a CI excluding zero is the evidence for
  regime-dependence. No t-test per arm is run: with 6 arms, multiplicity
  correction would leave n=3 underpowered; effect sizes with CIs are
  reported instead, and the design is plainly underpowered for formal
  testing. All of the above is implemented in `02_component_ablation.ipynb`
  and will populate automatically once `e2_ablation_results.json` has
  Phase-2 records.

## Figures

`figures/fig3_ablation_delta.png` (main) -- horizontal bar chart of
`Δ = accuracy(arm) − accuracy(A0)`, one bar per arm (`A1`-`A5`), two panels
side by side (left = `h_lo`, right = `h_hi`), shared x axis, `A0` marked as
a zero reference line. Bars left of zero mean the ablated component helps
the full model (removing it hurts accuracy); the *difference in bar
magnitude between the two panels* is the evidence for H3.

`figures/fig4_component_by_regime.png` -- grouped bars: x = arms (`A0`-`A5`),
y = raw test accuracy, one colour per cell, with the Experiment-E1 GCN
baseline plotted as a dashed horizontal reference line per cell (from
`../lifting_confounding_study/lifting_ablation_results.json`, arm `gcn`,
same two cell keys) -- shows whether an ablated DPHGNN is still above the
Track-1 baseline.

Both figures are produced by `02_component_ablation.ipynb`, which reads
`e2_ablation_results.json` only and never trains. As of this commit, no
Phase-1 records exist yet (see Finding below), so both figures were not yet
generated -- the notebook prints `"No data yet -- skipping ..."` for each
and exits cleanly rather than fabricating a plot from nothing.

## Reproducing

```bash
cd 2026_tdl_challenge/extra_analysis_oversmooth_operators/E2_ablation

# Sanity check first -- see "Local testing" above.
python run_e2.py --smoke-test --cpu

# Phase 1 -- 12 runs, seed 42 only. Resumable: safe to kill and re-run,
# already-present (arm, cell, seed) triples are skipped. Requires a GPU
# (or drop --cpu manually if you accept CPU wall-clock times); intended
# to be run from E2_kaggle_run.ipynb on Kaggle.
python run_e2.py

# Phase 2 -- once Phase 1 is fully green, adds seeds 43/44 (36 runs total).
python run_e2.py --phase2

# Analysis notebook only reads e2_ablation_results.json and plots -- never
# trains.
jupyter nbconvert --to notebook --execute --inplace 02_component_ablation.ipynb
```

`e2_ablation_results.json` is written incrementally (one record appended per
run, atomically) so a Kaggle session hitting the 12h cap loses at most the
in-flight run. Per-run checkpoints/logs land in `runs/`, which is
git-ignored.

## Finding

**Status: Phase 1 not yet run.** All code for this experiment -- the five
ablation flags in `topobench/nn/backbones/hypergraph/dphgnn.py`, the unit
tests in `test_dphgnn_ablation.py` (regression + per-arm, all green), the
`preflight_check()` verifying every Hydra override takes effect, and the
`run_e2.py` runner's mechanics (verified end to end via `--smoke-test
--cpu`, all 12 smoke jobs `status: "ok"`) -- is implemented and locally
validated. The real Phase-1 sweep (12 GPU training runs) was deliberately
**not** launched from this development session: no GPU was available
locally, real DPHGNN training runs are not "fast, CPU/tiny-data only"
validation, and the task explicitly reserves that sweep for a Kaggle GPU
session via `E2_kaggle_run.ipynb`.

Once `e2_ablation_results.json` has 12 `status: "ok"` Phase-1 records (run
`E2_kaggle_run.ipynb` on Kaggle, or `python run_e2.py` locally with a GPU),
re-run `02_component_ablation.ipynb` -- it will populate this section's
figures and the notebook's `print_finding()` cell automatically, and this
paragraph should be replaced with the real three-line H1/H2/H3 summary,
stated honestly including any nulls.
