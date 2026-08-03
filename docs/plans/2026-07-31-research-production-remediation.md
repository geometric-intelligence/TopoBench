# Research-Production Remediation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task. Use `superpowers:test-driven-development` for every behavior change, `superpowers:systematic-debugging` for any unexpected failure, and `superpowers:verification-before-completion` before claiming a task or release gate is complete.

**Goal:** Make the reduced graph, heterogeneous-graph, and hypergraph TopoBench core scientifically trustworthy for an internal applied-research team: qualified runs must mean what their configs say, reproduce from recorded inputs, reject malformed data before training, report the selected checkpoint consistently, and fail predictably at declared scale boundaries.

**Architecture:** Keep native PyG `Data`/`Batch`,
`HeteroData`/`NeighborLoader`, native incidence-index hypergraphs, explicit
domain registries, and easy custom-component development. Full-graph
transductive runs retain native in-memory paths. Large homogeneous and
heterogeneous single-graph runs share one immutable universal typed CSC store,
bounded selected reads, generic host/device prefetch, committed-cursor resume,
telemetry, provenance inputs, and prediction identity; homogeneous graph-aware
cluster unions and heterogeneous relation-aware target-seed neighbor sampling
remain explicit strategies. Selected-checkpoint validation and test each
publish independent final metrics and sharded per-sample artifacts. Packaged
selectors run in strict **qualified** mode. Custom Hydra targets remain
available only in explicit **experimental** mode and are recorded as
unqualified; raw Hydra configuration remains trusted local research code, not
an untrusted job-submission API.

**Tech Stack:** Python 3.11, PyTorch 2.3, PyTorch Geometric, Lightning 2.4, Hydra/OmegaConf, TorchMetrics, pytest, Ruff, uv.

**Source plans:**

- `docs/plans/2026-07-31-graph-heterogeneous-hypergraph-core-design.md`
- `docs/plans/2026-07-31-graph-heterogeneous-hypergraph-core-implementation.md`
- `docs/plans/2026-07-31-parquet-graph-ingestion-streaming-design.md`
- `docs/plans/2026-07-31-parquet-graph-ingestion-streaming-implementation.md`
- `docs/plans/2026-07-31-selected-checkpoint-prediction-artifacts-design.md`
- `docs/plans/2026-07-31-scalable-evaluator-design.md`
- `docs/plans/2026-07-31-scalable-evaluator-implementation.md`

This is a remediation plan for the current implementation candidate. It does not replace the core architecture plan. Where this plan is stricter, this plan controls release qualification.

---

## Execution rules

- Work directly in the active `topobench_graph_hetero` implementation worktree. Do not create another worktree.
- Preserve unrelated user changes. Never reset or overwrite files to recover a clean tree.
- One task at a time. Every behavior change begins with a focused failing test, followed by the smallest source change and adjacent regression tests.
- A declaration-only test is not evidence. Each production contract needs an output-sensitive, failure-sensitive, or lifecycle test.
- Use local deterministic fixtures in ordinary CI. Live downloads supplement them; they never replace them.
- Never regenerate splits, caches, or model state silently. Reject stale or ambiguous artifacts with an actionable message.
- No compatibility shims for deleted domains or unsupported multioutput/multilabel paths. Remove stale public surfaces instead.
- Do not recursively ban Hydra `_target_` in the trusted research CLI. Qualified mode validates packaged targets; experimental mode permits custom targets and records that the run is outside the qualified matrix.
- Do not add network-facing job submission, sandboxing, multi-tenant storage, or distributed training. Those remain non-goals.
- Do not weaken deterministic mode with `warn_only=True` while describing it as deterministic.
- Do not use Python `assert` as the only guard for user data, Hydra configuration, cache integrity, split integrity, target shape/range, or a public model constructor.
- Commit after each task only after its focused and adjacent tests pass.

---

## Risk model and priorities

The framework serves a trusted internal team running comparative experiments and adding case-specific adapters. The dominant risks are therefore:

1. **Scientific correctness:** leakage, wrong split semantics, reversed edges, ignored hyperparameters, target coercion, and misleading metrics.
2. **Reproducibility and provenance:** stale caches, global RNG mutation, incomplete split/data fingerprints, resume divergence, and ambiguous qualification.
3. **Model/data robustness:** malformed data accepted too late, non-finite values, cross-example influence, isolated-node failures, and wrong class vocabularies.
4. **Lifecycle accounting:** best-checkpoint metrics, callback state, and returned results disagreeing with dashboards or resumed runs.
5. **Resource predictability:** hidden quadratic work, avoidable densification/cloning, and unbounded metric state.
6. **Disk-streaming correctness:** stale/malformed typed stores, relation
   direction or fanout loss, incorrect homogeneous cluster unions, hidden
   full-array reads/conversions, and cluster/seed sampler state that diverges
   after resume.
7. **Evaluation auditability:** selected-checkpoint metrics disagree with
   retained prediction rows, identities/targets drift, or validation/test
   artifacts collide or omit logger registrations.
8. **Trusted-workflow hardening:** secret redaction, authenticated downloads,
   frozen environments, and pinned CI actions.

A network service accepting untrusted Hydra configs is explicitly outside this release. If that use case appears, build a separate target-free submission schema and isolated worker; do not reinterpret the research CLI as safe for hostile input.

---

## Release gates

### Gate A: Scientific validity

All are mandatory:

- train/validation/test phases are non-empty, pairwise disjoint, and complete whenever the contract says they partition one source;
- every named split tag has exactly one explicitly registered train/validation/
  test source; phase IDs and filenames are unique and pairwise disjoint within
  that tag, while overlap across different tags is permitted and provenance
  records the active tag plus all three source digests;
- ordinary fixed or generated train/validation/test is the only qualified split interface; `k-fold` is removed until an outer held-out test protocol exists;
- ADME selectors use declared scaffold splits with a recorded seed and provenance;
- every packaged dataset's task, task level, learning setting, feature policy, class count, and target vocabulary match its qualification manifest and runtime data;
- no graph batch contains a cross-graph edge;
- homogeneous disk batches preserve canonical node identity, phase ownership,
  edge direction, and every edge induced by the selected cluster union;
- heterogeneous disk batches preserve typed relation direction/fields and
  configured fanout, and supervise/export only target seeds;
- GCN-DGM uses incoming selected neighbors, has a trainable supported `k`, and
  cannot enter unbounded dense all-pairs construction under a qualified
  selector;
- EDGNN uses every exposed regularization parameter and supports or explicitly
  rejects isolated nodes per convolution;
- best-epoch train/validation metrics come from one epoch; one
  validation-selected checkpoint owns returned metrics plus independent
  validation/test per-sample artifacts with exact identities and targets.

### Gate B: Reproducibility and provenance

All are mandatory:

- generated splits and synthetic data use local generators and do not mutate global NumPy/Torch RNG state;
- processed-cache identity includes every loader parameter that changes raw data, transform configuration, representation version, and relevant parser policy;
- qualified runs default `save_reproducibility_bundle` to true and cannot
  disable it; an explicit experimental override marks all outputs unqualified;
- each run records dataset/source/cache, active split tag and phase digests,
  resolved config/model, fitted-transform state, environment, RNG/determinism,
  accepted partition-book/store, selected checkpoint, check evidence, and
  artifact-manifest identities;
- cache hit/miss and pre-partitioned download produce the same validated
  semantic content for one identity;
- typed-store identity records source/node/relation/split schemas, every array
  checksum, output/strategy capability, accepted partition map and hard-balance
  report, backend/build/options, producer environment, content hash, and
  committed sampler/evaluator state;
- the accepted partition map—not an independent METIS rerun—reproduces
  partition membership;
- each selected-checkpoint artifact records checkpoint/data/split/model/
  transform fingerprints, row/shard coverage/digests, identity schema, and
  distinct logger registrations;
- fresh/cache/download/moved-store/second-process and interrupted/resumed
  qualification runs satisfy their declared bitwise or numeric-equivalence
  profile for state, metrics, and prediction artifacts.

### Gate C: Robustness and scale

All are mandatory:

- graph, heterogeneous, and hypergraph features are floating, rank-correct, non-empty where required, and finite before model construction;
- classification labels are integral before conversion, have exact supported shape, and lie in the qualified vocabulary;
- ambiguous content rows are preserved or rejected by explicit metadata; payload values never decide whether a node exists;
- sparse raw features remain sparse or hit an explicit bounded densification gate;
- full transductive hypergraphs are not deep-cloned or re-collated needlessly;
- both disk strategies read only selected CSC/feature/supervision slices,
  produce writable native batches, and retain no full graph or feature matrix
  in runtime memory;
- homogeneous graph transforms run exactly once before transfer and preserve
  node/supervision identity; heterogeneous store/view normalization is
  validated before relation sampling;
- exhaustive structure validation runs once at ingestion/cache boundaries,
  not on every forward;
- selected-checkpoint prediction writing is bounded and sharded; it never
  retains an epoch of outputs in RAM or saves arbitrary batch/input state;
- default metrics do not retain unbounded per-example multiclass AUROC state on
  large node tasks;
- remote reads, archive extraction, local store promotion, and final prediction
  artifact promotion are bounded and atomic.

### Gate D: Trusted workflow

All are mandatory:

- no credential value is printed, persisted in `config_tree.log`, or forwarded to another logger;
- published remote data has pinned digest and size metadata; runtime never executes remote pickle payloads;
- state-dict-only checkpoint reruns use safe loading; full resume is restricted to verified same-run artifacts;
- setup and CI consume a committed frozen lock without deleting or rewriting it;
- third-party GitHub Actions are pinned to reviewed full commit SHAs and permissions are job-minimal.

### Gate E: Qualification evidence

All are mandatory:

- ordinary offline CI exercises malformed inputs, semantic model oracles, cache hit/miss equivalence, and lifecycle accounting;
- one explicit download opt-in controls all live-data tests;
- representative real OGB-MAG, DBLP, ADME, citation-hypergraph, and content-hypergraph qualifications exercise all phases and supervision ownership;
- representative real large homogeneous and heterogeneous typed Parquet graphs
  qualify conversion, cache reload, bounded selected reads, exact partition
  unions or relation fanout, all target phases, and finite GCN/HGT steps;
- selected-checkpoint validation/test artifacts qualify exact row identity,
  targets, raw/exported outputs, source metadata, sharding, separate paths, and
  per-file logger registration;
- skipped mandatory release qualifications fail the release job;
- the final seven end-to-end smokes pass from a clean frozen environment.

---

## Finding disposition ledger

Every validated audit finding maps to one implementation task. Duplicate reports share one ID.

| ID | Finding | Disposition | Task |
|---|---|---|---|
| F01 | Inductive phases can overlap or omit examples | Open; release blocker | 1 |
| F02 | `k-fold` reports validation as test | Remove from qualified product | 1 |
| F03 | ADME silently uses PyTDC random split | Open; release blocker | 2 |
| F04 | Split/cache generation mutates or depends on global RNG | Open | 2, 5 |
| F05 | Default graph transform selection can contradict feature policy | Open; release blocker | 3 |
| F06 | Graph/heterogeneous features can admit NaN/Inf | Open | 4 |
| F07 | Zero-node graph items pass preprocessing | Open | 4 |
| F08 | Configured and observed class vocabularies can drift | Open; release blocker | 4, 23 |
| F09 | Fractional hypergraph labels truncate silently | Open; release blocker | 9 |
| F10 | Value-based placeholder filtering deletes legitimate nodes | Open | 9 |
| F11 | Sparse hypergraph features densify without a bound | Open; scale blocker | 10 |
| F12 | Remote pickle assets execute before validation | Open; block affected selectors until migrated | 11 |
| F13 | Downloads/extraction lack timeout, size, and atomicity bounds | Open | 12 |
| F14 | Cross-graph edges can influence another example | Open; release blocker | 13 |
| F15 | Consumed edge-weight/edge-attribute behavior needs semantic oracle | Partly covered; strengthen | 13 |
| F16 | GCN-DGM learned-edge direction is reversed | Open; release blocker | 14 |
| F17 | GCN-DGM `k=1` has no structure-learning gradient | Open | 14 |
| F18 | GCN-DGM dense all-pairs work is unbounded | Open; scale blocker | 15 |
| F19 | NSD default width is not divisible by stalk dimension | Open | 16 |
| F20 | EDGNN ignores `input_dropout` | Open | 17 |
| F21 | EDGNN MeanDeg crashes or emits `-inf` for isolated nodes | Open; release blocker | 17 |
| F22 | OGB atom features are eagerly expanded for the full dataset | Open | 18 |
| F23 | Hypergraph pipeline deep-clones/re-collates the singleton graph | Open | 19 |
| F24 | Hypergraph wrapper repeats exhaustive full-graph validation | Open | 19 |
| F25 | Default multiclass AUROC can retain unbounded epoch state | Open | 20 |
| F26 | Best-epoch train metrics can come from another epoch | Open; release blocker | 21 |
| F27 | Best/rerun metrics do not reliably cross the returned metric API | Open; release blocker | 21 |
| F28 | Callback best state is not qualified across resume | Open | 22 |
| F29 | Full resume versus uninterrupted training is unqualified | Open | 22 |
| F30 | Non-graph capability/manifest validation is incomplete | Open; release blocker | 23 |
| F31 | Custom Hydra targets can look qualified | Open; solve with qualified/experimental profiles | 24 |
| F32 | Secrets are resolved into console, files, and logger payloads | Open | 25 |
| F33 | Writable caches/checkpoints use executable deserialization | Open; trusted-boundary hardening | 26 |
| F34 | Installer deletes the committed lock and re-resolves dependencies | Open | 27 |
| F35 | GitHub Actions use mutable tags with write credentials | Open | 27 |
| F36 | Production invariants rely on removable Python assertions | Open | 28 |
| F37 | Unsupported multioutput/multilabel and stale split APIs remain public | Open; delete | 28 |
| F38 | Real OGB-MAG/DBLP tests under-check phase supervision | Open | 29 |
| F39 | Live-download gates are inconsistent | Open | 29 |
| F40 | Existing tests overfit declarations and tiny shapes | Open; addressed by semantic/scale tests throughout | 1–31 |
| F41 | Deterministic mode has two authorities and permits warnings | Open | 30 |
| F42 | Run artifacts lack complete scientific provenance | Open; release blocker | 30 |
| F43 | Multi-worker neighbor replay did not force genuine choices for every relation/hop | Resolved in current candidate; preserve regression | 29, 31 |
| F44 | Native domain/pipeline target matching was incomplete | Resolved in current candidate; preserve regression | 23, 31 |
| F45 | Disk partitions can be stale, partial, executable, or schema-inconsistent | New approved scope; release blocker | 6 |
| F46 | Selected cluster unions can lose cross-cluster edges or expose permuted IDs as source IDs | New approved scope; release blocker | 7 |
| F47 | Runtime transforms can run twice, mutate stored views, or change supervision identity | New approved scope; release blocker | 7 |
| F48 | Multi-worker and resumed cluster sampling can diverge or reopen full graph state | New approved scope | 8 |
| F49 | One disk graph cannot select several named, independently validated split triplets | New approved scope; release blocker | 6, 8, 30 |
| F50 | Independent METIS reruns cannot reproduce an accepted partition | New approved scope; persist accepted map/environment | 6, 30 |
| F51 | Pre-partitioned download/cache promotion can expose partial or unsafe artifacts | New approved scope; release blocker | 6 |
| F52 | Conversion/training checks and temporal resource signals lack one actionable evidence stream | New approved scope | 7, 30 |
| F53 | Fitted preprocessing can leak validation/test data or repeat cluster context rows | New approved scope; release blocker | 7 |
| F54 | Materialized/disk sampling and final-metric parity lack hard reference oracles | New approved scope; release blocker | 7, 8, 31 |

---

## Phase 1: Scientific split and data contracts

### Task 1: Enforce one honest train/validation/test partition

**Files:**

- Modify: `topobench/data/splits.py`
- Modify: `topobench/data/utils/split_utils.py`
- Modify: `topobench/data/pipelines/default.py`
- Modify: surviving `configs/dataset/graph/*.yaml` that still advertise `k-fold`
- Modify: `test/data/utils/test_split_utils.py`
- Modify: `test/data/pipelines/test_data_pipelines.py`
- Modify: `test/config/test_all_surviving_configs.py`

**Step 1: Write failing partition tests**

Add cases for overlap, omission, duplicate indices within a phase, out-of-range indices, empty phases, unordered valid indices, and one valid partition. Assert exact `ValueError` messages name the offending phase or cross-phase invariant. Preserve the lazy `Subset`/shared-source assertion.

**Step 2: Write a failing `k-fold` rejection test**

Compose every surviving selector that currently advertises `split_type=k-fold`. Assert qualification rejects it before any dataset load. Add a direct test proving no helper returns `test == valid` as an ordinary result.

**Step 3: Run the focused red tests**

```bash
uv run pytest test/data/utils/test_split_utils.py test/data/pipelines/test_data_pipelines.py test/config/test_all_surviving_configs.py -q
```

Expected: new overlap/omission and `k-fold` tests fail for the intended reasons.

**Step 4: Implement canonical partition validation**

Normalize each phase once. Require:

- integral, unique, in-range indices;
- non-empty phases;
- pairwise disjointness;
- sorted union exactly `range(len(dataset))` for a complete fixed/generated split.

Return lazy views only after all validation succeeds. Do not modify source data.

**Step 5: Remove misleading `k-fold` behavior**

Delete the path that aliases validation indices into `test`. Remove `k-fold` from surviving product configs and resolver choices. Error text must explain that nested cross-validation needs an outer held-out test and is not yet qualified.

**Step 6: Run focused and adjacent tests**

```bash
uv run pytest test/data/utils/test_split_utils.py test/data/pipelines/test_data_pipelines.py test/config/test_all_surviving_configs.py test/data/dataload/test_Dataloaders.py -q
```

**Step 7: Commit**

```bash
git add topobench/data/splits.py topobench/data/utils/split_utils.py topobench/data/pipelines/default.py configs/dataset/graph test/data/utils/test_split_utils.py test/data/pipelines/test_data_pipelines.py test/config/test_all_surviving_configs.py
git commit -m "fix: enforce honest phase partitions"
```

### Task 2: Make ADME split semantics explicit and reproducible

**Files:**

- Modify: `topobench/data/loaders/graph/adme_datasets.py`
- Modify: retained `configs/dataset/graph/*_*.yaml` ADME selectors as needed
- Modify: `topobench/data/loaders/base.py`
- Modify: `test/data/loaders/graph/test_adme_loader.py`
- Modify: `test/integration/test_retained_datasets.py`

**Step 1: Write failing call-contract tests**

Mock PyTDC and assert `get_split(method="scaffold", seed=<configured seed>)` is called exactly. Add a negative test for a selector whose YAML claims scaffold while the loader method differs.

**Step 2: Write failing RNG-isolation tests**

Capture global NumPy and Torch RNG states before cache miss and cache hit. Assert both are unchanged. Use two configured seeds and assert distinct split fingerprints; repeat one seed and assert exact equality.

**Step 3: Run the red tests**

```bash
uv run pytest test/data/loaders/graph/test_adme_loader.py -q
```

**Step 4: Implement explicit split provenance**

Pass method and seed explicitly. Store split method, seed, PyTDC dataset name/version when available, exact phase-index digests, and source-data digest beside the cache. Use local generators only.

**Step 5: Reject stale caches**

Cache load must compare the complete provenance record. A mismatch is a cache miss or an actionable integrity error; it must never silently return old splits under new metadata.

**Step 6: Run focused and real-adapter tests**

```bash
uv run pytest test/data/loaders/graph/test_adme_loader.py test/integration/test_retained_datasets.py -q
```

**Step 7: Commit**

```bash
git add topobench/data/loaders/graph/adme_datasets.py topobench/data/loaders/base.py configs/dataset/graph test/data/loaders/graph/test_adme_loader.py test/integration/test_retained_datasets.py
git commit -m "fix: make ADME scaffold splits explicit"
```

### Task 3: Restore deterministic feature-policy composition

**Files:**

- Modify: `configs/run.yaml`
- Modify: `topobench/utils/config_resolvers.py`
- Modify: `topobench/data/capabilities.py`
- Modify: `topobench/nn/capabilities.py`
- Modify: `test/config/test_dataset_feature_policies.py`
- Modify: `test/config/test_all_surviving_configs.py`
- Modify: `test/integration/test_retained_datasets.py`

**Step 1: Write failing composition tests**

For every packaged graph selector, compose the default command without a transform override. Assert the resolved transform produces the declared policy and width. Include degree, constant, categorical, and continuous datasets. Add negative tests for manual transform overrides that contradict the manifest.

**Step 2: Run the red tests**

```bash
uv run pytest test/config/test_dataset_feature_policies.py test/config/test_all_surviving_configs.py -q
```

**Step 3: Implement one resolver path**

Restore dataset/model-aware default transform selection or an equivalent explicit manifest mapping. Validate policy/width compatibility during composition, before download. Do not maintain a second transform registry.

**Step 4: Exercise raw-to-batch behavior**

For one dataset per feature policy, load a local fixture, transform it, batch it, and assert exact dtype/width/finiteness and one forward pass.

**Step 5: Run focused and adjacent tests**

```bash
uv run pytest test/config/test_dataset_feature_policies.py test/config/test_all_surviving_configs.py test/integration/test_retained_datasets.py test/pipeline/test_graph_model_capabilities.py -q
```

**Step 6: Commit**

```bash
git add configs/run.yaml topobench/utils/config_resolvers.py topobench/data/capabilities.py topobench/nn/capabilities.py test/config/test_dataset_feature_policies.py test/config/test_all_surviving_configs.py test/integration/test_retained_datasets.py
git commit -m "fix: resolve graph feature policies before loading"
```

### Task 4: Enforce finite, non-empty, qualified feature and target vocabularies

**Files:**

- Modify: `topobench/data/features.py`
- Modify: `topobench/data/heterogeneous.py`
- Modify: `topobench/data/hypergraph.py`
- Modify: `topobench/data/capabilities.py`
- Modify: `topobench/data/pipelines/default.py`
- Modify: `topobench/data/pipelines/heterogeneous.py`
- Modify: `topobench/data/pipelines/hypergraph.py`
- Modify: `test/data/test_graph_feature_contract.py`
- Modify: `test/data/test_heterogeneous_spec.py`
- Modify: `test/data/test_hypergraph_data.py`
- Modify: `test/config/test_surviving_dataset_manifest.py`

**Step 1: Write failing malformed-data tests**

Cover NaN and Inf in every domain, zero-node homogeneous graphs, wrong label dtype/rank, negative labels, labels above configured range, configured extra classes, and a valid full source whose one phase omits a class.

**Step 2: Run the red tests**

```bash
uv run pytest test/data/test_graph_feature_contract.py test/data/test_heterogeneous_spec.py test/data/test_hypergraph_data.py test/config/test_surviving_dataset_manifest.py -q
```

**Step 3: Implement boundary validation**

Validate full-source data before splitting/model construction. Qualification must compare manifest class count with config and runtime vocabulary. Phase subsets may omit classes; the full qualified source may not invent or silently omit a declared class unless the manifest explicitly records that exception.

**Step 4: Keep errors contextual**

Messages identify selector, item or node type, field, observed shape/dtype/range, and expected contract. No repair or coercion.

**Step 5: Run focused and pipeline tests**

```bash
uv run pytest test/data/test_graph_feature_contract.py test/data/test_heterogeneous_spec.py test/data/test_hypergraph_data.py test/config/test_surviving_dataset_manifest.py test/data/pipelines/test_data_pipelines.py -q
```

**Step 6: Commit**

```bash
git add topobench/data/features.py topobench/data/heterogeneous.py topobench/data/hypergraph.py topobench/data/capabilities.py topobench/data/pipelines test/data/test_graph_feature_contract.py test/data/test_heterogeneous_spec.py test/data/test_hypergraph_data.py test/config/test_surviving_dataset_manifest.py
git commit -m "fix: validate qualified feature and label contracts"
```

### Task 5: Make processed-cache identity complete and RNG-independent

**Files:**

- Modify: `topobench/data/preprocessor/preprocessor.py`
- Modify: `topobench/data/loaders/base.py`
- Modify: `topobench/data/datasets/synthetic_graph_dataset.py`
- Modify: `topobench/data/datasets/synthetic_hypergraph_dataset.py`
- Modify: `test/data/preprocess/test_preprocessor.py`
- Modify: `test/data/datasets/test_synthetic_graph_dataset.py`
- Modify: `test/data/datasets/test_synthetic_hypergraph_dataset.py`

**Step 1: Write failing identity tests**

Within one cache root, vary each seed/size/generation parameter independently and assert a different cache identity. Repeat the same config and assert identical processed tensors and identical identity. Assert cache hit/miss preserve global RNG state.

**Step 2: Run the red tests**

```bash
uv run pytest test/data/preprocess/test_preprocessor.py test/data/datasets/test_synthetic_graph_dataset.py test/data/datasets/test_synthetic_hypergraph_dataset.py -q
```

**Step 3: Implement canonical cache identity**

Hash a versioned canonical record containing selector, exact loader target, all resolved loader parameters that affect raw data, transform target/parameters, feature policy, and representation/parser version. Do not hash only transform parameters. Store the readable record beside the digest.

**Step 4: Remove global seed calls from loaders**

Use `numpy.random.Generator`, `torch.Generator`, or scoped deterministic library APIs. A cache miss and cache hit must leave callers' RNG streams untouched.

**Step 5: Run focused tests**

```bash
uv run pytest test/data/preprocess/test_preprocessor.py test/data/datasets/test_synthetic_graph_dataset.py test/data/datasets/test_synthetic_hypergraph_dataset.py -q
```

**Step 6: Commit**

```bash
git add topobench/data/preprocessor/preprocessor.py topobench/data/loaders/base.py topobench/data/datasets/synthetic_graph_dataset.py topobench/data/datasets/synthetic_hypergraph_dataset.py test/data/preprocess/test_preprocessor.py test/data/datasets/test_synthetic_graph_dataset.py test/data/datasets/test_synthetic_hypergraph_dataset.py
git commit -m "fix: fingerprint processed data completely"
```

---

## Phase 2: Universal typed Parquet store and disk-backed execution

This phase starts only after Tasks 1–5 make source, split, feature, and cache
identity authoritative. Its approved architecture is
`docs/plans/2026-07-31-parquet-graph-ingestion-streaming-design.md`.
Its authoritative TDD steps, exact file ownership, commands, and commit
boundaries are
`docs/plans/2026-07-31-parquet-graph-ingestion-streaming-implementation.md`.

The companion plan contributes supported materialized-reference and disk modes,
one typed partition book/store, explicit named split triplets, fitted
training-only transforms, content-addressed pre-partitioned distribution,
reproducibility bundles, structured check evidence, profiling, prefetch, and
committed resume. Homogeneous cluster, heterogeneous cluster, and
heterogeneous neighbor strategies emit native `Data`/`HeteroData` without
rank-indexed collators, per-cluster SQLite copies, parent-open arrays, CUDA in
`collate_fn`, or compatibility fields.

### Task 6: Build the bounded universal typed store

**Files:**

- Execute companion-plan Tasks 1–7 exactly.
- Do not duplicate their schema/split registry, ingestion, ID maps, CSC writer,
  corrected homogeneous baseline, typed PyG partition adapter/validator,
  content-addressed store, bundle, or PyG protocol views.

**Step 1: Establish the red contracts**

Follow companion Tasks 1–7: typed YAML and named split triplets; inventory and
per-type IDs/features/supervision; canonical relation CSC; corrected pinned
`ClusterData` baseline; topology-only PyG typed partition book with 256 GiB
default admission and hard balance; then safe atomic store/bundle promotion.

The boundary accepts homogeneous and heterogeneous sources. Parquet conversion
never materializes complete features or mapped edges; the explicit reference
partition step may materialize topology only under its independent ceiling.

**Step 2: Run the Phase 2 store gate**

```bash
uv run pytest \
  test/data/loaders/test_parquet_typed_schema.py \
  test/data/stores/test_typed_graph_inventory.py \
  test/data/stores/test_external_node_index.py \
  test/data/stores/test_typed_graph_features.py \
  test/data/stores/test_typed_graph_supervision.py \
  test/data/stores/test_typed_graph_csc.py \
  test/data/stores/test_typed_graph_edge_rejections.py \
  test/data/stores/test_materialized_homogeneous_partition.py \
  test/data/stores/test_topology_only_pyg_partitioner.py \
  test/data/stores/test_typed_partition_book.py \
  test/data/stores/test_partition_qualification.py \
  test/data/stores/test_typed_graph_store.py \
  test/data/stores/test_pyg_store.py \
  test/data/stores/test_typed_store_promotion.py \
  test/data/stores/test_prepartitioned_store_bundle.py \
  test/data/stores/test_qualification_checks.py -q
```

Expected: exact split and typed-ID contracts, semantic digests across physical
layouts, canonical relation direction/fields, corrected homogeneous identity,
typed partition qualification, bounded conversion/reference resources,
selected memory-mapped reads, safe distribution, and atomic promotion.

**Step 3: Respect companion commit boundaries**

Companion Tasks 1–7 own seven commits. Do not add a redundant summary commit
or mark remediation Task 6 complete until their criteria pass.

### Task 7: Sample, transform, prefetch, resume, and profile disk views

**Files:**

- Execute companion-plan Tasks 8–12 exactly.
- Do not duplicate their data module, strategies, fitted transforms, sequence
  state, prefetch, event/check stream, callbacks, or monitor.

**Step 1: Establish exact native sampling behavior**

Execute companion Task 8. Homogeneous and heterogeneous partition descriptors
reconstruct exact induced unions against materialized PyG oracles.
Heterogeneous target-seed descriptors use exact deterministic relation-specific
neighbor parity and supervise only `n_id[:batch_size]`.

**Step 2: Fit training-only transforms**

Execute companion Task 9. The canonical fit view visits each active-tag
training entity once, never reads validation/test, and atomically publishes
bounded PCA/transform state keyed by all scientific inputs.

**Step 3: Make queued work checkpoint-safe**

Execute companion Task 10. One sequence protocol separates issued/prepared/
consumed work from committed optimizer/evaluator/global-step state, handles
gradient accumulation, and regenerates all uncommitted descriptors.

**Step 4: Add bounded generic prefetch**

Execute companion Task 11. Host and CUDA queues accept `Data`/`HeteroData`, use
independent budgets, and default to three device-ready batches plus the current
batch. CPU/MPS use explicit host-only mode; workers never call CUDA.

**Step 5: Add structured profiling and check evidence**

Execute companion Task 12. Profile conversion through training with bounded
local evidence and sampled W&B/logger aggregates. Every hard failure carries a
stable check ID, expected/observed evidence, remediation, and report path;
`system/*` never contaminates scientific metrics.

**Step 6: Run the Phase 2 streaming gate**

```bash
uv run pytest \
  test/data/dataload/test_disk_graph_datamodule.py \
  test/data/dataload/test_homogeneous_cluster_strategy.py \
  test/data/dataload/test_heterogeneous_neighbor_strategy.py \
  test/data/dataload/test_heterogeneous_cluster_strategy.py \
  test/data/dataload/test_disk_neighbor_parity.py \
  test/transforms/test_fittable_transform.py \
  test/transforms/test_incremental_pca.py \
  test/data/dataload/test_sequence_state.py \
  test/data/dataload/test_device_prefetch.py \
  test/profiling/test_execution_events.py \
  test/profiling/test_local_event_log.py \
  test/data/dataload/test_input_monitor.py \
  test/callbacks/test_dataloader_commit.py \
  test/callbacks/test_input_pipeline.py -q
```

Expected: both native output views are exact; spawn workers open lazily and
close cleanly; delivery and committed resume are ordered; telemetry does not
contaminate scientific results.

**Step 7: Respect companion commit boundaries**

Companion Tasks 8–12 own five commits. Do not add a summary commit.

### Task 8: Integrate and qualify universal disk training end to end

**Files:**

- Execute companion-plan Tasks 13–15 exactly.
- Preserve existing graph and heterogeneous in-memory data modules.

**Step 1: Wire thin TopoBench pipeline adapters**

Execute companion Task 13. Existing pipelines select materialized/disk and
cluster/neighbor strategies explicitly, reuse standard outputs and `TBModel`,
and provide active split, transform, reproducibility, profiling, and canonical
prediction identity inputs. `run.py` does not branch on native data type.
Parquet imports remain lazy outside this path.

**Step 2: Prove lifecycle and resume**

Execute companion Task 14 in fresh processes. Fresh/cache/download/moved-store
and interrupted/uninterrupted homogeneous cluster, heterogeneous cluster, and
neighbor runs agree on immutable identities, remaining descriptors,
sampler/evaluator/model state, selected checkpoint, metrics, and predictions
under the declared reproduction profile.

**Step 3: Run bounded-memory, real-data, and CUDA qualification**

Execute companion Task 15. Generated conversion and topology-partition fixtures
prove their independent memory ceilings. Real multi-split Parquet gates
exercise fitted transforms and both typed strategies. Exact exhaustive oracles,
predeclared paired-seed metric bounds, CUDA H2D/compute overlap, and at most 5%
strict steady-state input stall all pass.

```bash
uv run pytest \
  test/integration/test_typed_graph_conversion_resume.py \
  test/integration/test_graph_disk_resume.py \
  test/integration/test_heterogeneous_disk_resume.py \
  test/integration/test_typed_graph_lifecycle.py \
  test/integration/test_real_parquet_graph.py \
  test/integration/test_real_parquet_heterogeneous.py -q

uv run python test/integration/qualify_typed_graph_rss.py
uv run python test/integration/qualify_typed_graph_cuda.py
```

Expected: PASS in each mandatory environment. A skip, missing Parquet extra,
missing neighbor backend, absent CUDA runner, RSS/stall breach, or missing
evidence is not a passing release.

**Step 4: Run focused and adjacent framework suites**

```bash
uv run pytest \
  test/data/stores \
  test/data/dataload/test_disk_graph_datamodule.py \
  test/pipeline/test_disk_graph_pipeline.py \
  test/pipeline/test_disk_heterogeneous_pipeline.py \
  test/pipeline/test_graph_model_capabilities.py \
  test/pipeline/test_heterogeneous_pipeline.py \
  test/config/test_all_surviving_configs.py -q
```

Expected: PASS.

**Step 5: Respect companion commit boundaries**

Companion Tasks 13–15 own three commits. Do not add a redundant summary commit.
Resume this remediation plan at Task 9 only after every companion completion
criterion passes.

---

## Phase 3: Hypergraph input integrity and memory

### Task 9: Parse hypergraph labels and node roles without coercion

**Files:**

- Modify: `topobench/data/utils/hypergraph_io.py`
- Modify: `topobench/data/datasets/hypergraph_datasets.py`
- Modify: `test/data/load/test_hypergraph_dataset_loaders.py`
- Modify: `test/integration/test_real_hypergraph_formats.py`

**Step 1: Write failing parser fixtures**

Add content and legacy-format fixtures with fractional, NaN, and Inf labels; a legitimate isolated node with all-zero features and label 0; an explicit padding row; mixed raw IDs; duplicate incidence; and malformed node/hyperedge role ambiguity.

**Step 2: Run the red tests**

```bash
uv run pytest test/data/load/test_hypergraph_dataset_loaders.py -q
```

**Step 3: Implement explicit parsing**

Parse labels separately as numeric values, require finite integer-valued entries before `torch.long`, and canonicalize only after validation. Determine node/hyperedge roles from format metadata or dataset-specific declared rules, never from whether payload values are zero.

**Step 4: Remove global heuristic filtering**

Replace `filter_zero_placeholders=True` with a dataset-specific schema field or explicit known sentinel/count. If a source is ambiguous, fail with an explanation rather than deleting rows.

**Step 5: Run parser and format tests**

```bash
uv run pytest test/data/load/test_hypergraph_dataset_loaders.py test/integration/test_real_hypergraph_formats.py -q
```

**Step 6: Commit**

```bash
git add topobench/data/utils/hypergraph_io.py topobench/data/datasets/hypergraph_datasets.py test/data/load/test_hypergraph_dataset_loaders.py test/integration/test_real_hypergraph_formats.py
git commit -m "fix: parse hypergraph roles and labels exactly"
```

### Task 10: Preserve sparse hypergraph features or bound densification

**Files:**

- Modify: `topobench/data/hypergraph.py`
- Modify: `topobench/data/utils/hypergraph_io.py`
- Modify: `topobench/nn/encoders/graph_node_encoder.py` or create one shared sparse-compatible node encoder only if existing encoders cannot consume the chosen representation
- Modify: `test/data/test_hypergraph_data.py`
- Modify: `test/data/load/test_hypergraph_dataset_loaders.py`
- Modify: `test/pipeline/test_hypergraph_pipeline.py`

**Step 1: Choose one explicit contract**

Preferred: preserve scipy CSR as a PyTorch sparse CSR/COO tensor through ingestion and use a sparse-compatible projection before dense message passing. Fallback: allow densification only when a configurable byte estimate is below a conservative ceiling. Do not silently call `toarray()`/`todense()`.

**Step 2: Write failing scale-bound tests**

Use a tiny high-shape sparse fixture whose dense estimate exceeds the test ceiling. Assert no dense allocation occurs and either sparse projection succeeds or a clear pre-allocation error reports shape, dtype, estimated bytes, and limit.

**Step 3: Run the red tests**

```bash
uv run pytest test/data/test_hypergraph_data.py test/data/load/test_hypergraph_dataset_loaders.py test/pipeline/test_hypergraph_pipeline.py -q
```

**Step 4: Implement the selected contract**

Keep representation support narrow. Do not add sparse support to unrelated graph paths. Validate sparse layout, indices, shape, dtype, and finiteness of stored values.

**Step 5: Run focused tests and one real-format parser**

```bash
uv run pytest test/data/test_hypergraph_data.py test/data/load/test_hypergraph_dataset_loaders.py test/pipeline/test_hypergraph_pipeline.py -q
TOPOBENCH_ALLOW_DOWNLOADS=1 uv run pytest test/integration/test_real_hypergraph_formats.py -q
```

**Step 6: Commit**

```bash
git add topobench/data/hypergraph.py topobench/data/utils/hypergraph_io.py topobench/nn/encoders test/data/test_hypergraph_data.py test/data/load/test_hypergraph_dataset_loaders.py test/pipeline/test_hypergraph_pipeline.py
git commit -m "fix: bound hypergraph feature materialization"
```

### Task 11: Eliminate runtime deserialization of remote pickle datasets

**Files:**

- Modify: `topobench/data/utils/hypergraph_io.py`
- Modify: `topobench/data/datasets/citation_hypergraph_dataset.py`
- Modify: `topobench/data/datasets/hypergraph_datasets.py`
- Modify: `configs/dataset/hypergraph/cocitation_cora.yaml`
- Modify: other retained citation/coauthorship hypergraph YAMLs
- Modify: `test/data/load/test_hypergraph_dataset_loaders.py`
- Modify: `test/integration/test_real_hypergraph_formats.py`
- Create only if needed for one-time maintainers: `scripts/topobench/convert_legacy_hypergraph_asset.py`

**Step 1: Define a non-executable published format**

Use NPZ with `allow_pickle=False` for tensor arrays plus JSON for primitive metadata and a plain incidence array/table. Include format version, shapes, dtypes, raw-source digest, and conversion-tool version.

**Step 2: Write failing hostile-format tests**

Spy on `pickle.load` and assert packaged runtime paths never call it. Assert object-dtype NPZ, unknown schema versions, digest mismatches, and malformed incidence fail before cache creation. Keep a valid local safe-format fixture.

**Step 3: Convert assets outside the runtime path**

Run any legacy pickle conversion only as an explicit maintainer operation in an isolated environment. Record source and derived SHA-256 values. Runtime code must not retain a fallback to pickle.

**Step 4: Block selectors until safe assets exist**

If a safe authenticated asset is unavailable, remove that selector from the qualified manifest instead of shipping a pickle fallback or skipping its gate.

**Step 5: Run focused tests**

```bash
uv run pytest test/data/load/test_hypergraph_dataset_loaders.py test/integration/test_real_hypergraph_formats.py -q
```

**Step 6: Commit**

```bash
git add topobench/data/utils/hypergraph_io.py topobench/data/datasets configs/dataset/hypergraph test/data/load/test_hypergraph_dataset_loaders.py test/integration/test_real_hypergraph_formats.py scripts/topobench
git commit -m "security: remove remote pickle runtime loading"
```

### Task 12: Bound and authenticate remote dataset acquisition

**Files:**

- Modify: `topobench/data/utils/downloads.py`
- Modify: hypergraph dataset adapters under `topobench/data/datasets/`
- Modify: dataset capability/manifest metadata
- Create or modify: `test/data/utils/test_downloads.py`
- Modify: `test/integration/test_real_hypergraph_formats.py`

**Step 1: Write failing downloader tests**

Mock connect timeout, read timeout, non-2xx response, oversized compressed body, digest mismatch, partial interruption, unsafe archive member path/type, too many members, excessive per-member/total expanded size, and excessive expansion ratio. Assert cleanup leaves no promoted archive or cache.

**Step 2: Run the red tests**

```bash
uv run pytest test/data/utils/test_downloads.py -q
```

**Step 3: Implement bounded atomic download**

Use finite connect/read timeouts, `stream=True`, chunked writes to a private temporary file, compressed-byte ceiling, `raise_for_status()`, expected size and SHA-256 verification, `fsync` where appropriate, then atomic promotion.

**Step 4: Implement bounded extraction**

Validate all member paths and types before extraction. Enforce member count, per-member bytes, aggregate uncompressed bytes, and expansion ratio. Extract to a temporary directory and atomically promote only after format validation.

**Step 5: Run focused and live positive tests**

```bash
uv run pytest test/data/utils/test_downloads.py -q
TOPOBENCH_ALLOW_DOWNLOADS=1 uv run pytest test/integration/test_real_hypergraph_formats.py -q
```

**Step 6: Commit**

```bash
git add topobench/data/utils/downloads.py topobench/data/datasets topobench/data/capabilities.py test/data/utils/test_downloads.py test/integration/test_real_hypergraph_formats.py
git commit -m "fix: bound and authenticate dataset downloads"
```

---

## Phase 4: Model semantic correctness

### Task 13: Enforce graph batch isolation and edge-field semantics

**Files:**

- Modify: `topobench/nn/wrappers/graph/gnn_wrapper.py`
- Modify: `test/nn/wrappers/graph/test_graph_wrappers.py`
- Modify: `test/pipeline/test_graph_model_capabilities.py`

**Step 1: Write failing wrapper tests**

Create `batch=[0,0,1,1]` with an edge crossing graph IDs. Assert rejection before a recording backbone executes. Add wrong-rank, wrong-length, wrong-device, and non-finite `edge_weight`/`edge_attr` cases for consume modes.

**Step 2: Write an output-sensitive GCN test**

With dropout disabled and fixed parameters, compare non-uniform weights with all-ones weights and assert logits differ. Zeroing a bridge weight should match removing that message within tolerance.

**Step 3: Run the red tests**

```bash
uv run pytest test/nn/wrappers/graph/test_graph_wrappers.py test/pipeline/test_graph_model_capabilities.py -q
```

**Step 4: Implement pre-backbone validation**

Require both edge endpoints to share one graph ID. Validate consumed edge fields against `E = edge_index.size(1)`, dtype, device, and finiteness. Do not mutate the input batch on failure.

**Step 5: Run focused and graph-model tests**

```bash
uv run pytest test/nn/wrappers/graph/test_graph_wrappers.py test/pipeline/test_graph_model_capabilities.py test/nn/backbones/graph -q
```

**Step 6: Commit**

```bash
git add topobench/nn/wrappers/graph/gnn_wrapper.py test/nn/wrappers/graph/test_graph_wrappers.py test/pipeline/test_graph_model_capabilities.py
git commit -m "fix: enforce graph batch isolation"
```

### Task 14: Correct GCN-DGM neighbor direction and trainability

**Files:**

- Modify: `topobench/nn/backbones/graph/gcn_dgm.py`
- Modify: `configs/model/graph/gcn_dgm.yaml`
- Modify: `test/nn/backbones/graph/test_gcn_dgm.py`
- Modify: `test/nn/encoders/test_dgm.py`

**Step 1: Write a failing direction oracle**

Use asymmetric node features/distances where selected neighbors are known. Assert every query node has exactly `effective_k` incoming learned edges from its selected neighbors and weights normalize over that incoming set.

**Step 2: Write failing gradient/boundary tests**

Assert supported `k` values produce nonzero gradients for the structure learner and temperature. Assert `k=1`, booleans, non-integral values, and impossible values fail at construction/composition before a forward pass.

**Step 3: Run the red tests**

```bash
uv run pytest test/nn/backbones/graph/test_gcn_dgm.py test/nn/encoders/test_dgm.py -q
```

**Step 4: Fix edge orientation**

Build PyG source-to-target edges as `(selected_neighbor, query)`, retaining aligned scores. Normalize weights over messages entering each query.

**Step 5: Reject `k=1`**

Use `k >= 2` for the current deterministic top-k implementation. Do not claim differentiable topology learning for a one-choice softmax.

**Step 6: Run focused and capability tests**

```bash
uv run pytest test/nn/backbones/graph/test_gcn_dgm.py test/nn/encoders/test_dgm.py test/pipeline/test_graph_model_capabilities.py -q
```

**Step 7: Commit**

```bash
git add topobench/nn/backbones/graph/gcn_dgm.py configs/model/graph/gcn_dgm.yaml test/nn/backbones/graph/test_gcn_dgm.py test/nn/encoders/test_dgm.py
git commit -m "fix: correct GCN-DGM learned neighborhoods"
```

### Task 15: Put an explicit scale boundary around GCN-DGM

**Files:**

- Modify: `topobench/nn/backbones/graph/gcn_dgm.py`
- Modify: `topobench/nn/capabilities.py`
- Modify: `topobench/data/capabilities.py`
- Modify: `configs/model/graph/gcn_dgm.yaml`
- Modify: `test/nn/backbones/graph/test_gcn_dgm.py`
- Modify: `test/pipeline/test_graph_model_capabilities.py`

**Step 1: Decide the qualified implementation**

Preferred: exact chunked top-k that bounds peak pairwise storage while preserving results. If training autograd through chunked distances is still impractical, add a conservative qualified node-count/byte-workspace bound and reject larger datasets before allocation. Approximate ANN is a separate model and must not silently replace exact semantics.

**Step 2: Write failing equivalence and bound tests**

On small fixtures, compare chunked and dense neighbor indices/scores exactly or within declared tolerance. On a large-shape synthetic input, instrument `torch.cdist` calls and assert no full `N x N` matrix is requested. Test the preflight rejection path without allocating the matrix.

**Step 3: Run the red tests**

```bash
uv run pytest test/nn/backbones/graph/test_gcn_dgm.py test/pipeline/test_graph_model_capabilities.py -q
```

**Step 4: Implement and expose limits**

Make chunk size/workspace limit explicit in config and provenance. Capability qualification must reject infeasible dataset/model pairs before training.

**Step 5: Run focused tests**

```bash
uv run pytest test/nn/backbones/graph/test_gcn_dgm.py test/pipeline/test_graph_model_capabilities.py -q
```

**Step 6: Commit**

```bash
git add topobench/nn/backbones/graph/gcn_dgm.py topobench/nn/capabilities.py topobench/data/capabilities.py configs/model/graph/gcn_dgm.yaml test/nn/backbones/graph/test_gcn_dgm.py test/pipeline/test_graph_model_capabilities.py
git commit -m "fix: bound GCN-DGM neighborhood construction"
```

### Task 16: Make NSD dimensions exact

**Files:**

- Modify: `configs/model/graph/nsd.yaml`
- Modify: `topobench/nn/backbones/graph/nsd.py`
- Modify: `test/nn/backbones/graph/test_nsd.py`
- Modify: `test/config/test_surviving_graph_configs.py`

**Step 1: Write failing constructor/config tests**

Assert non-divisible `hidden_dim % d` raises a contextual `ValueError`; assert the packaged config's internal width equals its declared width.

**Step 2: Run the red tests**

```bash
uv run pytest test/nn/backbones/graph/test_nsd.py test/config/test_surviving_graph_configs.py -q
```

**Step 3: Implement exact validation**

Validate before integer division. Set packaged `d` to a divisor of 64, such as 4, based on the intended model rather than silently flooring.

**Step 4: Run focused tests**

```bash
uv run pytest test/nn/backbones/graph/test_nsd.py test/config/test_surviving_graph_configs.py -q
```

**Step 5: Commit**

```bash
git add configs/model/graph/nsd.yaml topobench/nn/backbones/graph/nsd.py test/nn/backbones/graph/test_nsd.py test/config/test_surviving_graph_configs.py
git commit -m "fix: require exact NSD stalk dimensions"
```

### Task 17: Honor EDGNN dropout and isolated-node contracts

**Files:**

- Modify: `topobench/nn/backbones/hypergraph/edgnn.py`
- Modify: `configs/model/hypergraph/edgnn.yaml`
- Modify: `test/nn/backbones/hypergraph/test_edgnn.py`
- Modify: `test/pipeline/test_hypergraph_pipeline.py`

**Step 1: Write a failing dropout test**

With identical parameters and fixed RNG, show `input_dropout=0` and `input_dropout>0` produce the expected train/eval distinction. Verify gradients and no effect in evaluation mode.

**Step 2: Write failing MeanDeg isolated-node tests**

Cover trailing isolated, interior isolated, no isolated, and empty-incidence rejection. Require finite output for supported isolated nodes and no shape truncation.

**Step 3: Run the red tests**

```bash
uv run pytest test/nn/backbones/hypergraph/test_edgnn.py test/pipeline/test_hypergraph_pipeline.py -q
```

**Step 4: Implement exact semantics**

Apply `self.input_drop` at the documented input point. In MeanDeg, pass `dim_size=N`, define isolated degree handling without `log(0)` (for example clamp before log with a documented zero-degree convention), and keep non-isolated behavior unchanged.

**Step 5: Run focused and wrapper tests**

```bash
uv run pytest test/nn/backbones/hypergraph/test_edgnn.py test/nn/wrappers/hypergraph/test_hypergraph_wrapper.py test/pipeline/test_hypergraph_pipeline.py -q
```

**Step 6: Commit**

```bash
git add topobench/nn/backbones/hypergraph/edgnn.py configs/model/hypergraph/edgnn.yaml test/nn/backbones/hypergraph/test_edgnn.py test/pipeline/test_hypergraph_pipeline.py
git commit -m "fix: honor EDGNN regularization and isolation"
```

---

## Phase 5: Memory and metric predictability

### Task 18: Keep molecular categorical features compact until batching

**Files:**

- Modify: `topobench/data/loaders/graph/ogbg_datasets.py`
- Modify: `topobench/data/loaders/graph/adme_datasets.py`
- Modify: `topobench/nn/encoders/graph_node_encoder.py`
- Modify: graph molecular configs and capability metadata
- Modify: `test/data/loaders/graph/test_ogbg_loader.py`
- Modify: `test/data/loaders/graph/test_adme_loader.py`
- Modify: `test/nn/encoders/test_graph_node_encoder.py`

**Step 1: Write failing compactness and equivalence tests**

Assert cached/full-dataset molecular features remain compact categorical columns. On a fixture, compare batch-local encoding to the previous canonical one-hot result or to a declared embedding contract. Reject out-of-range categories before lookup.

**Step 2: Run the red tests**

```bash
uv run pytest test/data/loaders/graph/test_ogbg_loader.py test/data/loaders/graph/test_adme_loader.py test/nn/encoders/test_graph_node_encoder.py -q
```

**Step 3: Implement batch-time encoding**

Prefer embeddings per categorical column when model semantics allow; otherwise one-hot only the current batch. Record category cardinalities and encoding mode in the data spec/provenance. Avoid a full-dataset 174-wide float copy.

**Step 4: Run focused and retained-data tests**

```bash
uv run pytest test/data/loaders/graph/test_ogbg_loader.py test/data/loaders/graph/test_adme_loader.py test/nn/encoders/test_graph_node_encoder.py test/integration/test_retained_datasets.py -q
```

**Step 5: Commit**

```bash
git add topobench/data/loaders/graph topobench/nn/encoders/graph_node_encoder.py configs/dataset/graph topobench/data/capabilities.py test/data/loaders/graph test/nn/encoders/test_graph_node_encoder.py
git commit -m "perf: encode molecular categories per batch"
```

### Task 19: Remove avoidable full-hypergraph copies and hot-path scans

**Files:**

- Modify: `topobench/data/pipelines/hypergraph.py`
- Modify: `topobench/dataloader/graph.py`
- Modify: `topobench/nn/wrappers/hypergraph/hypergraph_wrapper.py`
- Modify: `test/pipeline/test_hypergraph_pipeline.py`
- Modify: `test/data/dataload/test_hypergraph_dataloader.py`
- Modify: `test/nn/wrappers/hypergraph/test_hypergraph_wrapper.py`

**Step 1: Write failing identity/allocation tests**

Assert the pipeline does not clone `x`, `y`, or `hyperedge_index`; only split masks may be new. Assert a singleton transductive loader yields the existing validated object or a shallow batch view without recollating tensor storage.

**Step 2: Write a validation-frequency test**

Instrument exhaustive structure validation and assert it runs once per cache load/pipeline build, not once per forward. Keep cheap shape/dtype/device checks in the wrapper.

**Step 3: Run the red tests**

```bash
uv run pytest test/pipeline/test_hypergraph_pipeline.py test/data/dataload/test_hypergraph_dataloader.py test/nn/wrappers/hypergraph/test_hypergraph_wrapper.py -q
```

**Step 4: Implement shallow ownership**

Attach or replace only masks without copying immutable feature/incidence storage. Add a dedicated singleton full-batch loader only if the existing GraphDataModule cannot avoid recollation; do not fork general batching behavior.

**Step 5: Move exhaustive validation to the boundary**

Cache load and pipeline construction own finiteness, incidence contiguity, graph isolation, and full scans. Wrapper checks only fields whose validity can change during device transfer or custom user injection.

**Step 6: Run focused tests**

```bash
uv run pytest test/pipeline/test_hypergraph_pipeline.py test/data/dataload/test_hypergraph_dataloader.py test/nn/wrappers/hypergraph/test_hypergraph_wrapper.py -q
```

**Step 7: Commit**

```bash
git add topobench/data/pipelines/hypergraph.py topobench/dataloader/graph.py topobench/nn/wrappers/hypergraph/hypergraph_wrapper.py test/pipeline/test_hypergraph_pipeline.py test/data/dataload/test_hypergraph_dataloader.py test/nn/wrappers/hypergraph/test_hypergraph_wrapper.py
git commit -m "perf: avoid full hypergraph recopy and revalidation"
```

### Task 20: Implement the scalable evaluator and automatic preflight

**Authoritative companion plan:**

- Design: `docs/plans/2026-07-31-scalable-evaluator-design.md`
- Implementation:
  `docs/plans/2026-07-31-scalable-evaluator-implementation.md`

This task delegates evaluator, metric-policy, count-reporting, and pre-training
dry-run file ownership to Tasks 1–9 of the companion implementation plan. Do
not implement a second bounded-ranking patch or an ad hoc dry run in this
remediation file.

**Step 1: Execute companion Tasks 1–9**

The ordered companion work must:

- reduce evaluator tasks/metrics to the surviving core;
- introduce typed context, batch, result, and lifecycle contracts;
- introduce the automatic preflight gate early, before production logger and
  trainer construction;
- replace dynamic metric discovery with explicit TopoBench specifications and
  internal TorchMetrics adapters;
- implement exact, online, and audit policies;
- implement binary AUPRC as average precision and Somers'
  $D_{S\mid Y}=2\,\mathrm{AUROC}-1$, rejecting both for non-binary vocabularies;
- retain all exact binary/multiclass ranking observations in one guarded
  TopoBench-owned CPU backend with no stateful TorchMetrics duplicate;
- define strict/NaN undefined behavior;
- commit and report exact `num_examples` for every train/validation/test
  context;
- wire `TBModel` to the canonical selected supervision batch;
- complete non-committing data and isolated execution probes; and
- make configuration and the direct TorchMetrics dependency authoritative.

Training defaults to bounded online ranking metrics. Validation and test
default to exact in-memory ranking metrics under the declared byte ceiling.
Audit mode compares exact and thresholded values in one pass. No policy
silently falls back.

**Step 2: Run the companion focused gate**

```bash
uv run pytest \
  test/evaluator \
  test/preflight \
  test/model/test_model.py \
  test/model/test_supervision.py \
  test/config/test_all_surviving_configs.py \
  test/utils/test_config_resolvers.py \
  test/pipeline/test_pipeline.py \
  test/pipeline/test_heterogeneous_pipeline.py -q
```

Expected: PASS. Tests prove fixed state does not grow with example count,
binary exact state is shared, multiclass exact state is CPU-owned, every
reachable exact tensor and compute workspace is guarded, AUPRC/Somers' D
semantics and audit metadata are explicit, phase `num_examples` is exact,
preflight runs before training, and probe execution changes no production
state.

**Step 3: Record evaluator and preflight provenance**

Run provenance includes metric implementation, aggregation, policy,
exactness/thresholds, positive-class and Somers' D orientation, class support,
integer `num_examples`, retained/peak bytes, preflight checks, compilation
status, component targets/versions, skipped-impossible checks with reasons, and
qualification status. Disabling preflight requires an experimental override
and marks the run unqualified.

**Step 4: Use companion commit boundaries**

Use the focused commit boundaries in companion Tasks 1–9. Do not create a
duplicate Task 20 commit after those commits already own the changes.

---

## Phase 6: Training lifecycle and reported results

### Task 21: Make selected-checkpoint metrics and prediction artifacts authoritative

First execute companion evaluator Task 10 from
`docs/plans/2026-07-31-scalable-evaluator-implementation.md`. It owns explicit
selected-checkpoint evaluation contexts and the one authoritative
`EvaluationResult`; this task owns durable prediction serialization and logger
artifact publication.

**Files:**

- Modify: `topobench/model/supervision.py`
- Modify: `topobench/model/model.py`
- Modify: `topobench/data/pipelines/base.py`
- Modify: `topobench/data/splits.py`
- Modify: `topobench/callbacks/best_epoch_metrics.py`
- Create: `topobench/callbacks/prediction_artifacts.py`
- Create: `topobench/evaluator/prediction.py`
- Create: `topobench/utils/artifact_logging.py`
- Modify: `topobench/run.py`
- Modify: `configs/run.yaml`
- Create: `configs/evaluation_artifacts/default.yaml`
- Modify: `test/model/test_supervision.py`
- Create: `test/evaluator/test_prediction_payload.py`
- Create: `test/callbacks/test_prediction_artifacts.py`
- Create: `test/utils/test_artifact_logging.py`
- Modify: `test/callbacks/test_best_epoch_metrics.py`
- Modify: `test/pipeline/test_pipeline.py`
- Modify: `test/pipeline/test_heterogeneous_pipeline.py`

**Step 1: Write real selected-checkpoint lifecycle tests**

Use tiny actual Lightning trainers for at least two epochs with distinguishable
training/validation values. Assert same-epoch `best_epoch/train/*` and
`best_epoch/val/*` capture without manually injecting callback state. Assert
the checkpoint selected solely by validation is loaded for exactly one
validation rerun and one test rerun; test metrics never affect selection.
Assert train, ordinary validation, selected-checkpoint validation, and
selected-checkpoint test each log one exact `num_examples` total from the
finalized `EvaluationResult`, never a Lightning average of per-batch counts.

**Step 2: Write domain identity and payload tests**

Extend the canonical `EvaluationBatch` with aligned, immutable prediction
identity supplied by a pipeline-configured adapter. The supervision adapter
continues to select the loss-owning outputs/targets exactly once. Cover:

- graph-level `sample_id` across shuffle and unequal final batches;
- homogeneous full/disk `(source_graph_id, global_nid)`;
- heterogeneous `(source_graph_id, target_node_type, n_id)` with only
  `n_id[:batch_size]` seeds and no context nodes;
- hypergraph `(source_graph_id, global_nid)`;
- exact target, raw-output, exported prediction, optional model/normalized
  target spaces, shapes, class vocabulary/units, and source metadata.

External string IDs are restored only at export. Reject missing, duplicate,
surplus, or misaligned IDs, shape broadcasting, undeclared metadata, and the
same external ID collision across node types.

**Step 3: Write sharding, atomicity, and logger tests**

Require:

```text
evaluations/best_checkpoint/val/metrics.json
evaluations/best_checkpoint/val/predictions/manifest.json
evaluations/best_checkpoint/val/predictions/part-*.npz
evaluations/best_checkpoint/test/metrics.json
evaluations/best_checkpoint/test/predictions/manifest.json
evaluations/best_checkpoint/test/predictions/part-*.npz
```

Force several shards under tiny row/byte caps. Load every shard with
`allow_pickle=False` and assert ordered exact row coverage, equal column
lengths, dtypes/shapes, unique composite identity, checkpoint SHA-256, shard
SHA-256, source/dataset/split/model/transform fingerprints, and separate val
and test paths, callback state, manifests, and logger names. Require integer
`num_examples` in each `metrics.json` and exact equality with
`EvaluationResult.num_examples`, manifest observed rows, returned output,
provenance, and the split-qualified logger value.

Test atomic temporary-directory promotion, participant-count disagreement,
interruption before promotion, idempotent exact rerun, conflicting rerun
rejection, and explicit multi-rank failure. Every configured logger must
receive one separate artifact record/upload per `metrics.json`, manifest, and
shard. W&B uses immutable artifacts; CSV/local uses append-only URI+digest
index records. An unsupported logger fails preflight when artifacts are
enabled.

**Step 4: Write returned and source-sliced metric tests**

Without an external logger, assert `run()` returns best-epoch metrics,
selected-checkpoint validation/test metrics, each context's `num_examples`,
and both artifact manifest paths/digests. With bounded configured `source`
vocabulary, assert per-source metrics and counts come from the same
evaluator/prediction rows, are written under `slices/source/...`, and use stable
logger keys. Reject unbounded category expansion. Set `optimized_metric` to a
best-epoch key and assert lookup works.

**Step 5: Run the red tests**

```bash
uv run pytest \
  test/model/test_supervision.py \
  test/evaluator/test_prediction_payload.py \
  test/callbacks/test_prediction_artifacts.py \
  test/utils/test_artifact_logging.py \
  test/callbacks/test_best_epoch_metrics.py \
  test/pipeline/test_pipeline.py \
  test/pipeline/test_heterogeneous_pipeline.py -q
```

**Step 6: Implement one bounded artifact path**

Keep `TBModel.model_step` as the single supervision boundary. Companion
evaluator Task 10 routes the resulting `EvaluationBatch` through explicit
selected-checkpoint validation and test contexts. When artifact capture is
enabled, the writer consumes that same batch; it does not repeat the forward
pass or supervision selection. Stream completed batches to CPU and bounded
`.npz` buffers, release flushed buffers, validate coverage/finiteness/digests,
write versioned JSON, and promote each split atomically only after participant
counts agree.

The same immutable `EvaluationResult` feeds returned values, local
`metrics.json`, every logger, and provenance. Serialize
`EvaluationResult.num_examples` as an integer outside the metric mapping and
require it to equal manifest observed rows. W&B summary may mirror scalar
values but cannot be the only sink. Preserve numeric values without lossy
conversion.

**Step 7: Run focused and lifecycle tests**

```bash
uv run pytest \
  test/model/test_supervision.py \
  test/evaluator/test_prediction_payload.py \
  test/callbacks/test_prediction_artifacts.py \
  test/utils/test_artifact_logging.py \
  test/callbacks/test_best_epoch_metrics.py \
  test/pipeline/test_pipeline.py \
  test/pipeline/test_heterogeneous_pipeline.py \
  test/pipeline/test_disk_graph_pipeline.py \
  test/pipeline/test_disk_heterogeneous_pipeline.py -q
```

**Step 8: Commit**

```bash
git add topobench/model/supervision.py topobench/model/model.py topobench/data/pipelines/base.py topobench/data/splits.py topobench/callbacks/best_epoch_metrics.py topobench/callbacks/prediction_artifacts.py topobench/evaluator/prediction.py topobench/utils/artifact_logging.py topobench/run.py configs/run.yaml configs/evaluation_artifacts/default.yaml test/model/test_supervision.py test/evaluator/test_prediction_payload.py test/callbacks/test_prediction_artifacts.py test/utils/test_artifact_logging.py test/callbacks/test_best_epoch_metrics.py test/pipeline
git commit -m "feat: retain selected-checkpoint predictions"
```

---

### Task 22: Qualify checkpoint resume against uninterrupted training

First execute companion evaluator Task 11 from
`docs/plans/2026-07-31-scalable-evaluator-implementation.md`. It owns the joint
evaluator-sequence, sampler-cursor, and model-global-step checkpoint boundary;
this task proves that contract through complete production lifecycle resumes.

**Files:**

- Modify only for proven defects: `topobench/callbacks/best_epoch_metrics.py`
- Modify only for proven defects: `topobench/callbacks/prediction_artifacts.py`
- Modify only for proven defects: `topobench/evaluator/evaluator.py`
- Modify only for proven defects: `topobench/model/model.py`
- Modify only for proven defects: `topobench/run.py`
- Modify: `test/callbacks/test_best_epoch_metrics.py`
- Modify: `test/callbacks/test_prediction_artifacts.py`
- Modify: `test/evaluator/test_state_serialization.py`
- Modify: `test/pipeline/test_pipeline.py`
- Modify: `test/pipeline/test_heterogeneous_pipeline.py`
- Modify: `test/integration/test_graph_disk_resume.py`
- Modify: `test/integration/test_heterogeneous_disk_resume.py`

**Step 1: Write uninterrupted controls**

Run deterministic packaged tiny in-memory graph, homogeneous disk cluster,
in-memory heterogeneous neighbor, and heterogeneous disk neighbor experiments
for two or three epochs, including gradient accumulation greater than one.
Record model/optimizer/global-step/epoch state, best
monitor/epoch/checkpoint, callback metric state, evaluator committed
sequence/`num_examples`, split/store/strategy fingerprints, committed sampler
cursor, rerun metrics, prediction manifest digests, ordered canonical
identities, and shard contents.

**Step 2: Write resumed comparisons**

Inject production-hook checkpoint requests before an optimizer step, between
optimizer/sampler/evaluator commits, at an aligned global-step boundary, and at
epoch end. Mixed/pending requests must defer or fail without publishing a
checkpoint. Resume every valid checkpoint through `ckpt_path` and finish to the
same total epoch. Assert identical final model state, remaining cluster/seed
descriptors, evaluator metric state and `num_examples`, selected checkpoint,
validation/test metrics, prediction rows, shard semantic digests, and logger
registrations. Issued-but-uncommitted work is regenerated exactly once. An
interrupted final prediction write exposes no partial final directory and
restarts or resumes only at a validated shard boundary.

**Step 3: Run the red tests**

```bash
uv run pytest \
  test/callbacks/test_best_epoch_metrics.py \
  test/callbacks/test_prediction_artifacts.py \
  test/evaluator/test_state_serialization.py \
  test/pipeline/test_pipeline.py \
  test/pipeline/test_heterogeneous_pipeline.py \
  test/integration/test_graph_disk_resume.py \
  test/integration/test_heterogeneous_disk_resume.py -q
```

**Step 4: Persist lifecycle state correctly**

Implement or repair versioned `state_dict`/`load_state_dict` for best value,
epoch, metric record, comparison mode, selected checkpoint identity, and only
the prediction-writer state needed for validated shard-boundary recovery.
Training checkpoints contain only an aligned evaluator sequence/count state,
sampler cursor, and model global step; they contain no pending microbatch or
partial prediction tensors. Resume preserves pre-resume best state and
immutable store/artifact fingerprints.

**Step 5: Define qualification accurately**

If a backend cannot produce bitwise identity, document and test the exact
tolerance/invariant. Never label an untested resume path as equivalent to
uninterrupted training.

**Step 6: Run focused lifecycle tests**

```bash
uv run pytest \
  test/callbacks/test_best_epoch_metrics.py \
  test/callbacks/test_prediction_artifacts.py \
  test/evaluator/test_state_serialization.py \
  test/pipeline/test_pipeline.py \
  test/pipeline/test_heterogeneous_pipeline.py \
  test/integration/test_graph_disk_resume.py \
  test/integration/test_heterogeneous_disk_resume.py -q
```

**Step 7: Commit**

```bash
git add topobench/callbacks/best_epoch_metrics.py topobench/callbacks/prediction_artifacts.py topobench/evaluator/evaluator.py topobench/model/model.py topobench/run.py test/callbacks/test_best_epoch_metrics.py test/callbacks/test_prediction_artifacts.py test/evaluator/test_state_serialization.py test/pipeline/test_pipeline.py test/pipeline/test_heterogeneous_pipeline.py test/integration/test_graph_disk_resume.py test/integration/test_heterogeneous_disk_resume.py
git commit -m "fix: align selected results and metrics across resume"
```

---

## Phase 7: Qualification, extensibility, and trusted boundaries

### Task 23: Validate every domain through one capability boundary

**Files:**

- Modify: `topobench/data/capabilities.py`
- Modify: `topobench/nn/capabilities.py`
- Modify: `topobench/run.py`
- Modify: `topobench/utils/model_instantiation.py`
- Modify: `test/config/test_surviving_dataset_manifest.py`
- Modify: `test/config/test_all_surviving_configs.py`
- Modify: `test/utils/test_model_instantiation.py`

**Step 1: Write a domain mutation matrix**

For graph, heterogeneous, and hypergraph configs, mutate one field at a time: loader target, data name, task, task level, learning setting, split type, feature policy, class count, target node type, model selector, top-level model target, component target, and pipeline target. Qualified mode must reject every contradiction before loader/model construction.

**Step 2: Run the red tests**

```bash
uv run pytest test/config/test_surviving_dataset_manifest.py test/config/test_all_surviving_configs.py test/utils/test_model_instantiation.py -q
```

**Step 3: Implement complete manifests/capabilities**

Use one selector manifest and per-model capability records. Do not create separate partial gates in `run.py` and model instantiation. Compare runtime-observed metadata after loading before model construction.

**Step 4: Preserve custom research work**

Qualified packaged runs require exact registered targets. Experimental runs may use custom compatible components but are stamped `qualified=false`, include every target import path, and cannot emit benchmark-comparable status.

**Step 5: Run focused tests**

```bash
uv run pytest test/config/test_surviving_dataset_manifest.py test/config/test_all_surviving_configs.py test/utils/test_model_instantiation.py test/architecture/test_registries.py -q
```

**Step 6: Commit**

```bash
git add topobench/data/capabilities.py topobench/nn/capabilities.py topobench/run.py topobench/utils/model_instantiation.py test/config/test_surviving_dataset_manifest.py test/config/test_all_surviving_configs.py test/utils/test_model_instantiation.py
git commit -m "fix: qualify all domains through one boundary"
```

### Task 24: Separate qualified and experimental Hydra execution

**Files:**

- Modify: `configs/run.yaml`
- Modify: `topobench/run.py`
- Modify: `topobench/utils/model_instantiation.py`
- Modify: `topobench/utils/instantiators.py`
- Modify: `test/config/test_all_surviving_configs.py`
- Modify: `test/utils/test_instantiators.py`
- Modify: `test/utils/test_model_instantiation.py`

**Step 1: Write failing profile tests**

Default packaged commands are qualified. Overriding any executable target makes the run fail qualified preflight. The same compatible override under explicit `execution_profile=experimental` may instantiate, but the returned/provenance record is unqualified and names all custom targets.

**Step 2: Run the red tests**

```bash
uv run pytest test/config/test_all_surviving_configs.py test/utils/test_instantiators.py test/utils/test_model_instantiation.py -q
```

**Step 3: Implement the two profiles**

Keep trusted Hydra extensibility. Do not present experimental combinations as validated selectors. Reject unknown profile values. Print a concise warning for experimental runs without leaking the full config.

**Step 4: Document the actual trust boundary in existing docs**

Raw Hydra `_target_` values execute trusted Python. They are not an untrusted service API. External submission requires a separate target-free schema and isolated worker, which is not part of this task.

**Step 5: Run focused tests**

```bash
uv run pytest test/config/test_all_surviving_configs.py test/utils/test_instantiators.py test/utils/test_model_instantiation.py -q
```

**Step 6: Commit**

```bash
git add configs/run.yaml topobench/run.py topobench/utils/model_instantiation.py topobench/utils/instantiators.py test/config/test_all_surviving_configs.py test/utils/test_instantiators.py test/utils/test_model_instantiation.py docs
git commit -m "feat: distinguish qualified and experimental runs"
```

### Task 25: Remove credentials from all config output and logger payloads

**Files:**

- Modify: `configs/logger/comet.yaml`
- Modify: `configs/logger/neptune.yaml`
- Modify: `topobench/utils/logging_utils.py`
- Modify: `topobench/utils/rich_utils.py`
- Modify: `topobench/utils/utils.py`
- Modify: `test/utils/test_logging_utils.py`
- Modify: `test/utils/test_rich_utils.py`
- Modify: `test/utils/test_utils.py`

**Step 1: Write failing leak tests**

Set canary values for keys containing `api_key`, `token`, `password`, and `secret` at nested levels. Assert canaries do not appear in captured console output, `config_tree.log`, or any logger hyperparameter payload, including multi-logger mode.

**Step 2: Run the red tests**

```bash
uv run pytest test/utils/test_logging_utils.py test/utils/test_rich_utils.py test/utils/test_utils.py -q
```

**Step 3: Remove credentials from serializable Hydra config where possible**

Let SDKs read environment variables directly or pass a non-logged secret at construction. Add recursive key-based redaction as defense in depth before every print/save/log operation. Redaction must not mutate the source config.

**Step 4: Run focused tests**

```bash
uv run pytest test/utils/test_logging_utils.py test/utils/test_rich_utils.py test/utils/test_utils.py -q
```

**Step 5: Commit**

```bash
git add configs/logger/comet.yaml configs/logger/neptune.yaml topobench/utils/logging_utils.py topobench/utils/rich_utils.py topobench/utils/utils.py test/utils/test_logging_utils.py test/utils/test_rich_utils.py test/utils/test_utils.py
git commit -m "security: redact logger credentials everywhere"
```

### Task 26: Establish trusted cache and checkpoint deserialization boundaries

**Files:**

- Modify: `topobench/data/preprocessor/preprocessor.py`
- Modify: `topobench/data/loaders/graph/adme_datasets.py`
- Modify: `topobench/data/datasets/citation_hypergraph_dataset.py`
- Modify: `topobench/data/datasets/hypergraph_datasets.py`
- Modify: `topobench/run.py`
- Modify: `configs/paths/default.yaml`
- Modify: relevant cache and lifecycle tests

**Step 1: Write failing poisoned-artifact tests**

Use a harmless reducer canary and assert state-dict rerun and data-cache loads do not execute it. Test digest mismatch, path outside trusted root, wrong owner/mode where portable, and stale manifest rejection.

**Step 2: Run focused red tests**

```bash
uv run pytest test/data/preprocess/test_preprocessor.py test/data/loaders/graph/test_adme_loader.py test/data/load/test_hypergraph_dataset_loaders.py test/pipeline/test_pipeline.py -q
```

**Step 3: Use tensor/primitive-only cache payloads**

Serialize a versioned mapping of tensors and primitive metadata, load with `weights_only=True` or a non-executable format, validate, then reconstruct `Data`/`HypergraphData`. Write atomically under a per-principal cache root.

**Step 4: Split checkpoint policies**

Selected-checkpoint rerun loads state dict only with `weights_only=True` and `strict=True`. Full Lightning resume is accepted only from same-run trusted storage with a recorded digest and expected path/permissions. Reject arbitrary external `ckpt_path` in qualified mode.

**Step 5: Run focused and resume tests**

```bash
uv run pytest test/data/preprocess/test_preprocessor.py test/data/loaders/graph/test_adme_loader.py test/data/load/test_hypergraph_dataset_loaders.py test/pipeline/test_pipeline.py test/pipeline/test_heterogeneous_pipeline.py -q
```

**Step 6: Commit**

```bash
git add topobench/data/preprocessor/preprocessor.py topobench/data/loaders/graph/adme_datasets.py topobench/data/datasets/citation_hypergraph_dataset.py topobench/data/datasets/hypergraph_datasets.py topobench/run.py configs/paths/default.yaml test
git commit -m "security: constrain cache and checkpoint loading"
```

### Task 27: Freeze dependencies and pin CI actions

**Files:**

- Modify: `uv_env_setup.sh`
- Modify: `pyproject.toml`
- Regenerate once: `uv.lock`
- Modify: `.github/workflows/test.yml`
- Modify: `.github/workflows/lint.yml`
- Modify: `.github/workflows/docs.yml`
- Modify: `.github/workflows/update_leaderboard.yml`
- Modify: `test/dependencies/test_reduced_dependencies.py`
- Create: `test/dependencies/test_frozen_environment.py`

**Step 1: Write failing static policy tests**

Assert the setup script never deletes `uv.lock`, installation uses `uv sync --frozen`, pre-commit is in the lock, and every `uses:` value is a full 40-character commit SHA with a human-readable version comment.

**Step 2: Run the red tests**

```bash
uv run pytest test/dependencies/test_reduced_dependencies.py test/dependencies/test_frozen_environment.py -q
```

**Step 3: Simplify installation**

Encode supported platform sources in `pyproject.toml`/the lock. Stop editing `pyproject.toml` during setup. Separate production/dev extras only if necessary, but both must be frozen. Generate the lock once as a reviewed source change.

**Step 4: Pin and narrow workflows**

Pin all official and third-party actions to reviewed SHAs. Set checkout `persist-credentials: false` unless a later step needs credentials. Scope permissions at job level and keep cross-repository documentation credentials only in the deploy step/job.

**Step 5: Verify a clean frozen sync**

```bash
uv sync --frozen --all-extras
uv run pytest test/dependencies/test_reduced_dependencies.py test/dependencies/test_frozen_environment.py -q
```

**Step 6: Commit**

```bash
git add uv_env_setup.sh pyproject.toml uv.lock .github/workflows test/dependencies
git commit -m "build: freeze installs and pin CI actions"
```

### Task 28: Remove stale public APIs and replace boundary assertions

**Files:**

- Modify: `topobench/model/supervision.py`
- Modify: `topobench/loss/dataset/DatasetLoss.py`
- Modify: `topobench/evaluator/evaluator.py`
- Modify: `topobench/data/utils/split_utils.py`
- Modify: surviving `topobench/nn/backbones/graph/nsd_utils/*.py` only where assertions guard public/runtime data
- Modify: registries/exports and tests referencing unsupported APIs
- Modify: `test/architecture/test_data_surface.py`
- Modify: `test/architecture/test_domain_contract.py`
- Modify: `test/evaluator/test_evaluator.py`

**Step 1: Write failing surface tests**

Assert deleted multioutput/multilabel task values, stale coauthorship split helpers, and unsupported rank-era selectors are absent from exports/config choices and rejected at composition. Do not preserve dead aliases.

**Step 2: Write `python -O` boundary tests**

Run focused subprocesses under `python -O` for split/config/model constructor guards. Assert malformed public inputs still raise explicit `TypeError`, `ValueError`, `RuntimeError`, or `NotImplementedError` as appropriate.

**Step 3: Run the red tests**

```bash
uv run pytest test/architecture/test_data_surface.py test/architecture/test_domain_contract.py test/evaluator/test_evaluator.py -q
```

**Step 4: Delete unsupported branches**

The reduced product supports binary/multiclass classification and scalar regression only where declared. Remove multioutput/multilabel branches from supervision, loss, evaluator, configs, resolvers, tests, and public docs.

**Step 5: Replace boundary assertions selectively**

Replace assertions that are the sole guard for user/config/data invariants. Keep genuine internal impossibility assertions where a preceding validated boundary guarantees them. Preserve diagnostic context.

**Step 6: Run focused and optimized-mode tests**

```bash
uv run pytest test/architecture/test_data_surface.py test/architecture/test_domain_contract.py test/evaluator/test_evaluator.py test/data/utils/test_split_utils.py test/nn/backbones/graph/test_nsd.py -q
```

**Step 7: Commit**

```bash
git add topobench test configs
git commit -m "refactor: remove stale tasks and harden boundaries"
```

---

## Phase 8: Qualification evidence and provenance

### Task 29: Make real-data and neighbor-sampling qualification meaningful

**Files:**

- Modify: `test/integration/test_ogb_mag_preflight.py`
- Modify: `test/integration/test_dblp_heterogeneous.py`
- Modify: `test/integration/test_real_hypergraph_formats.py`
- Modify: `test/integration/test_retained_datasets.py`
- Modify: `test/integration/test_real_graph_disk.py`
- Modify: `test/data/dataload/test_heterogeneous_dataloader.py`
- Modify: `test/data/dataload/verify_fixed_heterogeneous_neighbor_workers.py`
- Modify: pytest configuration and `.github/workflows/test.yml`

**Step 1: Standardize live-data gating**

Use only `TOPOBENCH_ALLOW_DOWNLOADS=1` and a single `download` marker. Ordinary CI must skip every live request and still run local raw-format fixtures. Release CI enables the variable and fails if any mandatory selector qualification skips.

**Step 2: Strengthen OGB-MAG phase assertions**

For train/validation/test, assert positive target `batch_size`, leading seed IDs belong to the correct official mask, supervision count equals seed count, labels match source IDs, and no non-seed node contributes to loss/metrics.

**Step 3: Strengthen DBLP phase assertions**

Exercise all three full-batch phases and assert author-mask ownership, labels, counts, finite loss, and a real optimizer step on train only.

**Step 4: Preserve the strengthened fanout oracle**

Keep the existing fixture where every relation has more candidate incoming neighbors than every configured hop fanout, including multi-worker replay. Same seed must produce byte-identical batch digests; a different seed must change at least one sampled neighborhood.

**Step 5: Run offline qualification**

```bash
uv run pytest test/data/dataload/test_heterogeneous_dataloader.py test/integration/test_real_hypergraph_formats.py test/integration/test_real_graph_disk.py test/integration/test_retained_datasets.py -q
```

**Step 6: Run explicit live qualification**

```bash
TOPOBENCH_ALLOW_DOWNLOADS=1 uv run pytest test/integration/test_ogb_mag_preflight.py test/integration/test_dblp_heterogeneous.py test/integration/test_real_hypergraph_formats.py test/integration/test_real_graph_disk.py test/integration/test_retained_datasets.py -q
```

Expected: no mandatory skips.

**Step 7: Commit**

```bash
git add test/integration test/data/dataload .github/workflows/test.yml pyproject.toml
git commit -m "test: qualify real phase supervision and downloads"
```

### Task 30: Make deterministic mode and scientific provenance authoritative

**Files:**

- Modify: `configs/run.yaml`
- Modify: `configs/trainer/default.yaml`
- Create: `configs/artifacts/reproducibility.yaml`
- Modify: `topobench/run.py`
- Create: `topobench/provenance.py`
- Create: `topobench/reproducibility.py`
- Create: `test/test_provenance.py`
- Create: `test/test_reproducibility_bundle.py`
- Modify: lifecycle and pipeline tests

**Step 1: Write failing deterministic-mode tests**

Assert there is one authoritative config field. In deterministic mode, nondeterministic operations raise rather than warn. Assert global seeds, loader/sampler seeds, worker policy, and deterministic backend state are recorded.

Assert `save_reproducibility_bundle` defaults to true, cannot be false in a
qualified profile, and marks a permitted experimental run unqualified. Exercise
fresh, cache-hit, downloaded pre-partitioned, moved, resumed, multiple-split,
and second-clean-process replay.

**Step 2: Write failing provenance-schema tests**

Require a versioned record containing:

- qualification/reproduction level and reasons;
- dataset selector, loader/backend versions, raw/cache/store content digests;
- all registered split tags, active tag, exact phase source/index/mask digests,
  counts, coverage policy, and partition-balance evidence;
- feature policy plus fitted/runtime transform configuration, code/state digest,
  fit-input checksum, and leakage qualification;
- typed-store source/schema/array/checksum manifest, output/target types,
  accepted typed partition-book digest, PyG/METIS backend/build/options,
  hard-balance/cut evidence, and pre-partitioned bundle identity;
- committed cluster/seed sampler cursor and RNG plus evaluator sequence/count
  and model global step;
- host/device queue settings, structured profiling/check schema, event/summary
  digests, p50/p95/p99 timings, starvation, RSS/pinned/GPU/temp-disk/final-size;
- model selector, component targets, behavior-changing hyperparameters,
  parameter count, and initialization digest;
- loss/metric definitions, exactness/tolerances, participant counts, and
  sampled-strategy paired-seed qualification;
- selected epoch/checkpoint path/digest, validation/test rerun results, and
  prediction manifest paths/digests;
- per-artifact schema, row/shard counts, identity schema, logger registration,
  and bounded slice definitions;
- Python, OS/kernel/architecture, CPU, Torch, PyG, Lightning,
  `torch-sparse`/`pyg-lib`, METIS, CUDA runtime/driver/GPU, BLAS/OpenMP,
  container digest when present, repository revision/dirty-patch digest, and
  dependency-lock digest;
- global/component seeds, deterministic algorithms, workers, and sampler
  settings.

Do not include secrets, raw data, individual real-data IDs, tensors, predictions, or embeddings in generic logs.

**Step 3: Run the red tests**

```bash
uv run pytest test/test_provenance.py test/test_reproducibility_bundle.py test/pipeline/test_pipeline.py test/pipeline/test_heterogeneous_pipeline.py -q
```

**Step 4: Implement one immutable run record**

Build one immutable record and bundle from validated pipeline/model/lifecycle
outputs, not a Hydra-tree dump. Hash canonical non-sensitive structures, copy
the qualified lock/config/state references, atomically publish the bundle, and
return its path/digest from `run()`.

**Step 5: Enforce strict deterministic mode**

Remove duplicate root/trainer authorities. Use strict deterministic algorithms when selected. If a qualified backend cannot comply, fail preflight or mark the run experimental; do not continue with a warning while claiming reproducibility.

**Step 6: Run focused tests**

```bash
uv run pytest test/test_provenance.py test/test_reproducibility_bundle.py test/pipeline/test_pipeline.py test/pipeline/test_heterogeneous_pipeline.py test/data/dataload/test_heterogeneous_dataloader.py -q
```

**Step 7: Commit**

```bash
git add configs/run.yaml configs/trainer/default.yaml configs/artifacts/reproducibility.yaml topobench/run.py topobench/provenance.py topobench/reproducibility.py test/test_provenance.py test/test_reproducibility_bundle.py test/pipeline test/data/dataload/test_heterogeneous_dataloader.py
git commit -m "feat: record authoritative experiment provenance"
```

### Task 31: Run the final release qualification

**Files:**

- Modify only if a genuine defect is found: relevant source/test/config file
- Modify after all checks pass: existing current docs and changelog/release notes if the repository has one

**Step 1: Run focused scientific suites**

```bash
uv run pytest \
  test/data/utils/test_split_utils.py \
  test/data/test_graph_feature_contract.py \
  test/data/test_heterogeneous_spec.py \
  test/data/test_hypergraph_data.py \
  test/nn/backbones/graph/test_gcn_dgm.py \
  test/nn/backbones/graph/test_nsd.py \
  test/nn/backbones/hypergraph/test_edgnn.py \
  test/callbacks/test_best_epoch_metrics.py \
  test/callbacks/test_prediction_artifacts.py \
  test/evaluator/test_evaluator.py \
  test/evaluator/test_prediction_payload.py \
  test/utils/test_artifact_logging.py -q

```

Expected: PASS.

**Step 2: Run configuration, security, and dependency suites**

```bash
uv run pytest \
  test/config \
  test/architecture \
  test/utils \
  test/dependencies \
  test/test_provenance.py -q
```

Expected: PASS.

**Step 3: Run all offline tests**

```bash
uv run pytest test -q
```

Expected: PASS with only explicitly documented optional skips; no live download attempts.

**Step 4: Run style and import gates**

```bash
uv run ruff check .
uv run ruff format --check .
uv run python test/data/pipelines/verify_clean_import.py
```

Expected: PASS.

**Step 5: Build a clean frozen environment**

From a disposable clean checkout or CI job:

```bash
uv sync --frozen --all-extras
uv run python test/data/pipelines/verify_clean_import.py
uv run pytest test -q
uv run python test/integration/qualify_typed_graph_rss.py
uv run python test/integration/qualify_typed_graph_cuda.py
```

Expected: PASS; `uv.lock` unchanged.

**Step 6: Run the seven end-to-end CPU smokes**

```bash
WANDB_MODE=disabled uv run python -m topobench.run experiment=graph_synthetic_gcn trainer.accelerator=cpu trainer.devices=1 trainer.max_epochs=1
WANDB_MODE=disabled uv run python -m topobench.run experiment=graph_synthetic_regression trainer.accelerator=cpu trainer.devices=1 trainer.max_epochs=1
WANDB_MODE=disabled uv run python -m topobench.run experiment=graph_synthetic_inductive_node trainer.accelerator=cpu trainer.devices=1 trainer.max_epochs=1
WANDB_MODE=disabled uv run python -m topobench.run experiment=graph_synthetic_disk_gcn trainer.accelerator=cpu trainer.devices=1 trainer.max_epochs=1
WANDB_MODE=disabled uv run python -m topobench.run experiment=heterogeneous_synthetic_disk_hgt_neighbor trainer.accelerator=cpu trainer.devices=1 trainer.max_epochs=1
WANDB_MODE=disabled uv run python -m topobench.run experiment=heterogeneous_synthetic_hgt trainer.accelerator=cpu trainer.devices=1 trainer.max_epochs=1
WANDB_MODE=disabled uv run python -m topobench.run experiment=hypergraph_synthetic_edgnn trainer.accelerator=cpu trainer.devices=1 trainer.max_epochs=1
```

For each smoke, inspect returned values and the run directory and assert:

- `qualified=true` and finite train/validation/test metrics;
- binary runs include AUPRC and Somers' D with declared positive-class and
  orientation metadata;
- best epoch and validation-selected checkpoint digest;
- exact validation/test selected-checkpoint metric records;
- each phase's `num_examples` agrees across returned output, logger, final JSON,
  prediction manifest where applicable, and provenance;
- distinct `evaluations/best_checkpoint/{val,test}` metrics, manifest, and
  pickle-free prediction shards using that same checkpoint digest;
- complete data/split/model/environment/artifact provenance;
- no secret-bearing config dump.

**Step 7: Run explicit real-data qualification**

```bash
TOPOBENCH_ALLOW_DOWNLOADS=1 uv run pytest \
  test/integration/test_ogb_mag_preflight.py \
  test/integration/test_dblp_heterogeneous.py \
  test/integration/test_real_hypergraph_formats.py \
  test/integration/test_real_parquet_graph.py \
  test/integration/test_real_parquet_heterogeneous.py \
  test/integration/test_retained_datasets.py -q

uv run python test/integration/qualify_typed_graph_rss.py
uv run python test/integration/qualify_typed_graph_cuda.py
```

Expected: PASS with zero mandatory skips.

**Step 8: Review changes from the researcher’s perspective**

For one representative graph, heterogeneous, and hypergraph run, compare:

- cold cache versus warm cache;
- seed A repeat versus seed B;
- uninterrupted versus resumed training;
- current versus selected checkpoint;
- qualified packaged model versus explicit experimental custom component.

Any disagreement must be explained by a recorded contract, not hidden state.

**Step 9: Update existing docs only after all gates pass**

Document exactly:

- supported domains/tasks/selectors/models;
- split semantics and removal of `k-fold` test reporting;
- qualified versus experimental profile;
- cache and checkpoint trust boundaries;
- deterministic guarantees and limitations;
- live-data qualification command;
- disk partition identity, runtime-memory boundary, sampler-resume semantics,
  and deterministic exactly-once `batch_transform` constraints;
- scale limits for GCN-DGM, sparse hypergraph features, and metrics;
- selected-checkpoint validation/test prediction schemas, artifact directories,
  logger registration names, source-sliced analyses, and retention policy.

**Step 10: Run the final sparse-comment review**

Execute companion evaluator Task 14 from
`docs/plans/2026-07-31-scalable-evaluator-implementation.md` after every
behavioral and documentation gate above passes. One review agent inspects the
integrated runtime and adds only short rationale comments at genuinely nested
or non-obvious reliability boundaries. The task's focused tests and Ruff gates
must pass; any discovered behavioral defect returns to TDD and requires
re-running the affected release gates.

**Step 11: Final commit**

```bash
git add README.md docs configs topobench test pyproject.toml uv.lock .github/workflows
git commit -m "release: qualify the reduced research core"
```

---

## Completion criteria

Implementation is complete only when all are true:

- F01–F42 and F45–F54 are fixed, removed, or explicitly bounded as specified;
  F43–F44 remain covered by regressions.
- No qualified run uses overlap within one named split triplet,
  validation-as-test, random ADME mislabeled as scaffold, stale caches,
  silently coerced labels, or mismatched classes; distinct split tags may
  intentionally overlap and remain independently fingerprinted.
- Every exposed model hyperparameter changes executed behavior or is rejected.
- Graph and hypergraph examples remain isolated under batching/message passing.
- One typed store/partition book supports homogeneous and heterogeneous
  materialized/disk cluster plus heterogeneous neighbor `Data`/`HeteroData`
  views without a parallel framework.
- Parquet conversion never materializes complete features or mapped edge
  tables; topology-only partitioning obeys its independent 256 GiB default
  admission; no hidden layout densification/full clone/exhaustive forward scan
  or unbounded default metric state remains.
- Typed IDs/features/supervision, directed relations/fields, and partition
  inverses round-trip exactly. Temporary METIS reverse arcs never leak.
- Materialized/disk cluster unions match PyG subgraph oracles; qualified
  deterministic disk neighbor batches exactly match materialized
  `NeighborLoader`; exhaustive logits/metrics and predeclared paired-seed
  sampled-metric bounds pass.
- Training-only fitted transforms visit canonical active-tag rows once, never
  read validation/test, and replay from immutable checked state.
- Pre-partitioned bundles safely download, stage, validate, atomically promote,
  move, and reopen by digest without executable artifacts.
- Qualified runs retain the default-mandatory reproducibility bundle with full
  source/config/environment/partition/split/transform/checkpoint/artifact
  identity; METIS is not rerun for replay.
- Bounded host/CUDA queues preserve order and committed sampler/evaluator/
  global-step resume for every strategy. Structured local profiling/check
  evidence remains authoritative, W&B publication is bounded, and strict
  qualification remains at or below 5% input stall.
- Best-epoch, selected-checkpoint rerun, returned metrics, every external
  logger, final metrics JSON, prediction manifest, and resume state agree by
  construction on metric values and exact `num_examples`.
- Binary AUPRC/Somers' D semantics, online bounds, exact TopoBench-owned
  binary/multiclass CPU state, complete retained-plus-compute guards, and
  audit/error policies are qualified with independent reference values.
- resumable checkpoints align evaluator sequence/count state, sampler committed
  cursor, and model global step; gradient-accumulation microbatches cannot be
  checkpointed pending or counted twice;
- Validation and test each retain independent atomic metrics plus sharded
  per-sample identity/target/raw-output/prediction artifacts from the same
  validation-selected checkpoint; files never silently overwrite and every
  configured logger receives a separate split-qualified artifact record.
- Qualified packaged configs are closed and source-validated; custom Hydra components remain easy to add under an explicit unqualified experimental profile.
- No credential appears in stdout, `config_tree.log`, run provenance, or cross-logger hyperparameters.
- Runtime does not execute remote pickle data; downloaded artifacts are bounded, authenticated, validated, and atomically promoted.
- Cache and checkpoint deserialization follows the declared trusted boundary and safe state-dict path.
- `uv sync --frozen --all-extras` succeeds without changing the lock; all actions are full-SHA pinned.
- a final agent review leaves sparse rationale-bearing comments only at
  genuinely complex reliability boundaries and its focused/style gates pass;
- Offline suite, Ruff, clean-import probe, seven CPU smokes, bounded-RSS
  homogeneous/heterogeneous Parquet conversion, mandatory CUDA overlap,
  selected-checkpoint prediction artifacts, and live-data qualifications pass
  with recorded evidence.

## Explicit non-goals

- Hostile multi-tenant Hydra job submission.
- Arbitrary untrusted checkpoint uploads.
- Distributed or DDP metric qualification.
- Distributed prediction-artifact merge before an all-rank collection protocol
  is separately qualified.
- Heterogeneous link prediction or graph-level prediction.
- Hypergraph graph-level prediction, neighbor sampling, or distributed sampling.
- Multi-graph disk batching, remote object-store memory mapping, distributed
  Cluster-GCN/conversion, mutable Parquet embeddings, arbitrary
  configuration-provided SQL/Python expressions, and stochastic qualified
  runtime transforms.
- Reintroduction of cell, simplicial, combinatorial, point-cloud, or rank-indexed APIs.
- `k-fold` benchmark reporting without a designed outer held-out test protocol.
- Compatibility shims for removed multioutput/multilabel behavior.
