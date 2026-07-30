# Cell-HGT ZINC Capacity Search Design

## Goal

Provide a small, interpretable ZINC hyperparameter search for CellHGT that
runs one seed at a time, logs every run to one Weights & Biases project, and
uses meaningful names that encode the complete model configuration.

## Experimental protocol

The search keeps the choices that are not currently under investigation
fixed:

- the standard fixed ZINC split;
- the default rank-0/1/2 cell-complex lift;
- batch size 128;
- encoder and backbone dropout 0.1;
- Adam with the repository's StepLR schedule (`step_size=50`, `gamma=0.5`);
- minimum 50 and maximum 500 epochs;
- validation every 5 epochs;
- early-stopping patience 10 and minimum delta 0.005.

This matches the final graph-level TopoTune training budget. The old
development search scripts contain some 1,000-epoch commands, but the
published ZINC protocol and reproduction commands use the 500-epoch cap.

## Staged search

The search is deliberately staged instead of Cartesian so that each result
has a clear interpretation and the laptop does not spend time on combinations
that have already lost in an earlier stage.

1. `depth`: run 2, 4, and 8 HGT layers with 4 heads, width 64, and learning
   rate 0.001.
2. `heads`: after choosing the best depth by validation MAE, run 2 and 8
   heads. The 4-head result already exists from the depth stage.
3. `width`: after choosing depth and heads, run width 128. The width-64
   result already exists.
4. `lr`: after choosing the architecture, run learning rates 0.0005 and
   0.002. The 0.001 result already exists.

Each invocation uses exactly one seed. Candidates within a stage run
sequentially.

## W&B organization

All runs default to the W&B project `cell-hgt-zinc`. The project can be
changed with the `WANDB_PROJECT` environment variable, and `WANDB_ENTITY` is
honored when set.

Runs are grouped by stage and seed, for example
`zinc-hgt-depth-s0`. A run name records every varied value, for example:

`zinc-hgt-depth-d04-h04-w064-lr1e-3-s0`

Tags record `cell`, `hgt`, `zinc`, `hpo`, and the search stage. W&B receives
the best-checkpoint validation and test metrics through TopoBench's
`val_best_rerun/*` and `test_best_rerun/*` logging path.

## Interface and safety

The launcher lives at `scripts/hgt/zinc_hgt_search.sh` and supports:

```text
zinc_hgt_search.sh depth [seed]
zinc_hgt_search.sh heads <best_depth> [seed]
zinc_hgt_search.sh width <best_depth> <best_heads> [seed]
zinc_hgt_search.sh lr <best_depth> <best_heads> <best_width> [seed]
```

`DRY_RUN=1` prints the exact commands without training. The launcher validates
positive integer arguments and the HGT requirement that width be divisible by
the number of heads. On macOS it uses `caffeinate -i`; elsewhere it invokes
the project Python executable directly.

