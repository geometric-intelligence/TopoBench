# TRAWL in TopoBench

TRAWL lifts input data to cells, computes structural encodings, samples random
walks over the cells, and encodes the walk sequences with configurable neural
layers. It uses TopoBench's loaders, transform cache, collator, `TBModel`,
losses, evaluators, optimizers, Lightning trainer, callbacks and Hydra sweeps.
Every behavior is selected by configuration; no code branches on dataset names.

## Repository layout

- `nn/backbones/combinatorial/trawl.py`: the backbone in one module, the
  sequence layers (Mamba, SISA, GRU, transformer, MLP, graph adapters) and
  `TRAWL`.
- `transforms/data_manipulations/trawl.py`: builds walk states, relations and
  TopoBench's HOPSE positional/structural encodings after any lifting.
- The backbone module also holds CPU walk sampling and guidance (with an
  optional `numba` kernel), the feature adapter `TRAWLFeatureEncoder` and the
  task head `TRAWLReadout`.
- `model/trawl_pretraining.py`, `evaluator/checkpoint.py`,
  `callbacks/model_checkpoint.py`: optional pretraining and top-K checkpoint
  evaluation.
- Configs: `configs/model/combinatorial/trawl.yaml`, `configs/model/graph/trawl.yaml`,
  `configs/transforms/trawl_*.yaml`, `configs/experiment/trawl/`.

## Start here

```bash
python -m topobench model=combinatorial/trawl dataset=graph/PROTEINS
python -m topobench model=combinatorial/trawl dataset=graph/NCI1 model.backbone.architecture=sisa
python -m topobench model=graph/trawl dataset=graph/cocitation_cora
```

`combinatorial/trawl` applies TopoBench's cycle lifting and then the TRAWL
transform to graph inputs; already-lifted datasets use `trawl_existing`.
`graph/trawl` walks the original graph without higher cells. To use another
lifting, compose it first and `/transforms/data_manipulations@trawl: trawl` last,
and set `model.backbone.max_rank` to the highest rank you want.

Pure-PyTorch Mamba is the default and runs on CPU. The official kernel is
selected with `model.backbone.layer_options.mamba.backend=mamba_ssm` (validated
with `mamba-ssm==2.2.6.post3`, `transformers==4.44.2`); selection never falls
back silently.

## Relations and walks

`transforms.trawl.graph` chooses the walk graph:

- `hasse`: immediate incidences only; every step changes rank.
- `augmented_hasse` (default): any selected incidences and same-rank adjacencies.
- `cell_overlap`: cells in `overlap_ranks` connect when they share a vertex.

`transforms.trawl.neighborhoods` uses TopoBench names (`up_incidence-0`,
`down_adjacency-1`, `2-up_adjacency-0`, ...); the default is
`[up_incidence-0, up_incidence-1]`, traversed in both directions.
`experiment=trawl/adjacency` and `experiment=trawl/mixed` are examples of other
relation sets.

Walk options live under `model.backbone.walks`: `k`, `length` (default 32/32),
`start_policy` (`coverage` favours rarely visited start states until coverage
plateaus), `epsilon`, `reverse` (adds reversed copies), and `guidance`
(`gamma`, `diffusion_t`, `dense_limit`). Walks are non-backtracking unless a
state has no other exit.

- `walk_scope=union` samples K walks over all relations together; `separate`
  samples K per relation and fuses them (`fusion=mean|concat|learned|attention`,
  `encoder_sharing=shared|independent`).
- `walk_refresh=train` resamples every training step; `fixed` keeps one sample.
- `eval_views` averages several walk samples during validation;
  `evaluation.walk_views` does the same for the final test.

## Encodings

`transforms.trawl.encodings` adds per-state positional/structural encodings
computed on each relation's graph (or on the union of relations with
`encoding_scope=union`). Apart from `local` (`log(1 + degree)` and neighbor
count), every key names one of TopoBench's HOPSE encodings and holds its
arguments: `LapPE`, `RWSE`, `ElectrostaticPE` and `HKdiagSE`. They are computed
on the unweighted, active part of each relation; states outside a relation get
zeros. The default is

```yaml
encodings:
  local: true
  LapPE: {max_pe_dim: 8}
  RWSE: {max_pe_dim: 8}
  ElectrostaticPE: {}
  HKdiagSE: {kernel_param_HKdiagSE: [1, 6]}
```

`walks.guidance` reweights transitions by `|exp(-tL)_ij|^γ`, with L the
symmetric Chung Laplacian of the weighted relation. It uses an exact dense
eigendecomposition limited by `dense_limit=2048`; exceeding it raises an error
rather than switching to an approximation.

Optional categorical colors come from `color_key` (set `model.backbone.num_colors`
to cover them); `color_refinement=true` adds encodings for color-pair subgraphs.
Absent ranks stay empty and empty relations get zero encodings. Unvisited cells
fall back to their projected input features.

## Layers and outputs

`architecture` selects `mamba`, `sisa`, `hybrid` (M–S–M–S–M at depth 5),
`transformer`, `gru` or `mlp`; `layer_options` configures each kind. For any
other stack, set `model.backbone.layers` to a list of layer configs
(`experiment=trawl/custom_layers`). A layer may be a Hydra `_target_` mapping
`[walk, time, hidden]` to the same shape, or `kind: graph` wrapping a PyG module
that runs on each walk as a path graph.

The backbone returns contextual cell features `x_r`/`batch_r` for any TopoBench
readout, plus walk and graph embeddings. `TRAWLReadout` supports graph and node
tasks with `aggregation=embedding` (head on the pooled graph embedding) or
`walk_logits` (head per walk, logits averaged). Time pooling is `mean`, `max` or
`mean_max`; `occurrence_pooling` and `graph_readout=walks|cells` control how
walk states become cell and graph embeddings.

Input widths are inferred from the training data before the optimizer is built.
To use a native feature encoder, wrap it in
`topobench.nn.backbones.combinatorial.trawl.TRAWLFeatureEncoder` and set
`model.backbone.in_channels`.

## Pretraining

```bash
python -m topobench model=combinatorial/trawl dataset=graph/PROTEINS pretraining.enabled=true
```

Self-supervised pretraining reconstructs masked cell features and can also
predict colors and masked connections (`pretraining.objectives`). It uses only
the training split, never labels, and hides encodings that would leak masked
structure. The best validation encoder is restored before supervised training;
`reset_head` reinitializes the head. Resume with
`pretraining.ckpt_path=.../pretraining/last.ckpt`; `pretraining.min_delta` sets
the improvement needed by both checkpointing and early stopping.

## Recipes

The configs in `configs/experiment/trawl/` are ordinary compositions of the
options above:

```bash
python -m topobench experiment=trawl/proteins_mamba
python -m topobench experiment=trawl/proteins_hybrid
python -m topobench experiment=trawl/nci1_hybrid    # also nci1_sisa, nci1_mamba
python -m topobench -m experiment=trawl/proteins_hybrid seed=40,41,42,43,44
```

`recipe.yaml` holds the shared PROTEINS/NCI1 settings: cycles up to six nodes,
the default HOPSE encodings, hidden width 224,
one-logit BCE with walk-logit averaging, seeded 50/25/25 stratified splits,
batch 2 with gradient accumulation, SSL pretraining, three evaluation views and
`deterministic=strict`. `proteins_hybrid` adds top-5 checkpoint weight
averaging. To reproduce the scores reported for the original implementation
exactly, use the `trawl-integration` branch of the fork.

## Reproducibility and performance

Walks, splits, epoch order, pretraining masks and initialization are seeded.
`deterministic=strict` also enables PyTorch's deterministic kernels (raising on
any without one), passes the setting to every Lightning trainer, and sets
`CUBLAS_WORKSPACE_CONFIG=:4096:8` (launchers should export it before CUDA
starts). Under this mode two runs on separate GPUs produced bit-identical
checkpoints, metric histories and evaluations, for about 9% extra wall time.
Exact repeats assume the same software stack, GPU model, device count and
evaluation batch size.

Speedups that leave results unchanged: an exact reimplementation of NumPy's
weighted choice (compiled when `numba` is installed, `pip install topobench[trawl]`),
cached transition distributions and evaluation walks, host-side walk bookkeeping
with non-blocking batch transfer (`TBModel` stashes the backbone's `host_fields`),
pinned memory and `evaluator.validate_args=false`.

## Splits and evaluation

- `split_type=seeded_stratified` performs two stratified splits with
  `random_state=data_seed`; `split_type=imported` reads `train`/`valid`/`test`
  index arrays from `split_file`.
- `dataset.dataloader_params.order_seed` shuffles the training set with
  `EpochRandomSampler` (`dataloader/samplers.py`): epoch `e` uses a permutation
  seeded by `order_seed + e + 1`, independent of other random draws. Without
  it, the usual seeded shuffle is used. Under several processes, evaluation
  uses `UnpaddedDistributedSampler`, which visits every sample exactly once
  instead of padding uneven shards with repeats.
- `evaluation.checkpoint` is `best`, `weight_average` or `logit_ensemble`
  (`evaluator/checkpoint.py`): the latter two average the weights, or the
  logits, of the `evaluation.top_k` best checkpoints. They are ranked by
  `RankedModelCheckpoint` (`callbacks/model_checkpoint.py`), which evicts the
  latest of tied checkpoints and indexes the top K in `checkpoint_index.json`,
  so a run directory can be moved and replayed with
  `train=false ckpt_path=...`. With `save_top_k: 1` it behaves like
  Lightning's `ModelCheckpoint`.
- One-logit binary outputs are scored as two-class probabilities, so AUROC does
  not depend on the evaluation batch size. Plateau schedulers step only on
  validated epochs.
- After evaluation each run writes `trawl_evaluation.json`.

## Development checks

```bash
python -m pytest test/nn/trawl -q
python -m ruff check topobench test/nn/trawl
```

`test/nn/trawl/fixtures/reference.npz` holds SISA outputs from the original
implementation.
