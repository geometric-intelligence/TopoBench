# Scalable Exact and Online Evaluator Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Refactor `TBEvaluator` behind typed TopoBench contracts, retain TorchMetrics internally, support exact/online/audit policies including binary AUPRC and Somers' D at scale, report the exact metric-participant count for every train/validation/test context, and automatically prove the configured pipeline with an isolated pre-training dry run.

**Architecture:** The supervision adapter emits immutable `EvaluationBatch` values into an explicit evaluator lifecycle. TopoBench-owned metric specifications adapt TorchMetrics without exporting third-party classes; policy selects exact or thresholded ranking metrics while decomposable metrics remain exact streaming accumulators. Binary AUROC/AUPRC share one exact score/target buffer and Somers' D derives from AUROC. One committed counter supplies `num_examples` to phase logs, returned results, final JSON, prediction-manifest validation, and provenance. A preflight component runs before production loggers/training, validates static contracts, obtains non-committing batches, exercises a throwaway execution path, and emits a structured qualification result.

**Tech Stack:** Python 3.11, PyTorch 2.3, PyTorch Geometric 2.8, Lightning 2.4, TorchMetrics, Hydra/OmegaConf, pytest, Ruff, uv.

**Design:** `docs/plans/2026-07-31-scalable-evaluator-design.md`

---

## Execution rules

- Work directly in the active `graph-hetero-core-impl` worktree. Do not create another worktree.
- Use @superpowers:test-driven-development for every behavior change,
  @superpowers:systematic-debugging for unexpected failures, and
  @superpowers:verification-before-completion before claiming a task or the
  plan complete.
- Keep native PyG `Data`, `Batch`, `HeteroData`, and `NeighborLoader` contracts.
- Do not reintroduce rank-indexed data, removed domains, or compatibility
  shims.
- Do not copy PICID's NumPy conversion or unconditional prediction buffer.
- Do not reimplement standard metric mathematics. Use public TorchMetrics
  APIs behind TopoBench adapters.
- Do not use TorchMetrics private functions or expose TorchMetrics classes from
  `topobench.evaluator`.
- Keep exact validation/test and online training as defaults. Never silently
  change policy after a resource or undefined-metric failure.
- Treat binary AUPRC as average precision and Somers' D as
  $D_{S\mid Y}=2\,\mathrm{AUROC}-1$ for positive-class index `1`; reject both
  metrics unless `num_classes == 2`.
- Increment `num_examples` only after an entire batch update commits. Every
  metric, logger, returned result, JSON artifact, and provenance record for one
  context must report the same count.
- The automatic preflight must run before production logger/trainer side
  effects and must leave model, optimizer, RNG, and sampler state pristine.
- Keep preflight checks in one component. Later tasks extend that component;
  they do not create independent ad hoc dry runs.
- Run focused tests while implementing. Run the complete evaluator/preflight
  suite and end-to-end smokes only at the final task.
- Commit after each task using the listed boundary. If concurrent approved
  plan work already owns one listed file, coordinate ownership before editing;
  do not duplicate implementations.

## Cross-plan ordering and ownership

This is a companion plan to:

- `docs/plans/2026-07-31-graph-heterogeneous-hypergraph-core-implementation.md`;
- `docs/plans/2026-07-31-research-production-remediation.md`; and
- `docs/plans/2026-07-31-selected-checkpoint-prediction-artifacts-design.md`.

The native supervision adapter and surviving task contracts are prerequisites.
This plan owns evaluator types, metric policy, evaluator lifecycle, preflight,
and model/evaluator integration. Remediation Task 21 continues to own durable
prediction files, logger artifact uploads, and final selected-checkpoint
artifact publication. It must consume this plan's `EvaluationBatch` and
`EvaluationResult` rather than creating another metric dictionary or evaluator.

Implement Tasks 1-9 before remediation Task 21. Task 10 provides the explicit
handoff to selected-checkpoint reruns. If artifact code is not yet present,
keep the sink protocol and preflight probe test local; do not create a second
artifact implementation. Task 14 is deliberately last and runs only after
Tasks 1-13 and remediation Task 21 are integrated and verified.

---

### Task 1: Freeze the reduced evaluator contract

**Files:**

- Modify: `test/evaluator/test_evaluator.py`
- Modify: `test/evaluator/test_graph_target_shapes.py`
- Create: `test/evaluator/test_reduced_contract.py`
- Modify: `test/utils/test_config_resolvers.py`

**Step 1: Write failing reduced-surface tests**

Add tests asserting:

- only `classification` and `regression` evaluator tasks are accepted;
- default multiclass metric keys are exactly `accuracy`, `precision`, `recall`,
  `f1`, and `auroc`;
- default binary classification additionally includes `auprc` and `somers_d`;
- regression keys are exactly `mae`, `mse`, `rmse`, and `r2`;
- `auprc` and `somers_d` reject `num_classes != 2` at construction;
- multilabel and multioutput tasks fail during evaluator construction rather
  than updating and then raising;
- `example`, `confusion_matrix`, `f1_macro`, `f1_weighted`, and suffixed metric
  aliases are not built-in core names;
- classification remains rank-2 `[N, C]` outputs plus rank-1 `torch.long`
  targets, with binary classification represented by `[N, 2]` and `{0, 1}`;
- scalar regression remains equal floating `[N, 1]` outputs and targets; and
- `get_default_metrics` uses task plus class count to return active metrics and
  rejects removed tasks or binary-only metrics on other vocabularies.

Use exact assertions, for example:

```python
@pytest.mark.parametrize(
    "task",
    ["multilabel classification", "multioutput classification"],
)
def test_removed_evaluator_tasks_fail_at_construction(task):
    with pytest.raises(ValueError, match="Supported tasks"):
        TBEvaluator(task=task, num_classes=3, metrics=["accuracy"])
```

Do not preserve old tests merely because they describe current behavior. Delete
or rewrite tests for removed APIs.

**Step 2: Run the focused tests to verify failure**

```bash
uv run pytest \
  test/evaluator/test_evaluator.py \
  test/evaluator/test_graph_target_shapes.py \
  test/evaluator/test_reduced_contract.py \
  test/utils/test_config_resolvers.py -q
```

Expected: new reduced-contract tests fail on current multilabel/multioutput,
legacy names, and resolver behavior; existing active task shape tests remain
informative.

**Step 3: Make the smallest construction/resolver cut**

Restrict current construction and `get_default_metrics` to the approved task
surface without yet redesigning lifecycle or metric storage. Preserve active
classification/regression behavior so later tasks have a stable baseline.

**Step 4: Run the focused tests**

Run the command from Step 2.

Expected: PASS.

**Step 5: Commit**

```bash
git add test/evaluator/test_evaluator.py test/evaluator/test_graph_target_shapes.py test/evaluator/test_reduced_contract.py test/utils/test_config_resolvers.py topobench/evaluator/evaluator.py topobench/utils/config_resolvers.py
git commit -m "refactor: freeze reduced evaluator contract"
```

---

### Task 2: Introduce typed evaluator lifecycle contracts

**Files:**

- Create: `topobench/evaluator/types.py`
- Rewrite: `topobench/evaluator/base.py`
- Modify: `topobench/evaluator/evaluator.py`
- Modify: `topobench/evaluator/__init__.py`
- Create: `test/evaluator/test_types.py`
- Create: `test/evaluator/test_lifecycle.py`

**Step 1: Write failing dataclass validation tests**

Cover immutable `EvaluationContext`, `EvaluationBatch`, `EvaluationResult`, and
policy/split/pass-kind literals. Assert:

- positive expected, batch, and observed counts;
- no boolean-as-integer counts;
- tensor-only outputs/targets;
- `EvaluationBatch.num_examples` equals leading tensor counts;
- `EvaluationResult.num_examples` is an integer, immutable, and separate from
  the scalar metric mapping;
- immutable ordered result mappings;
- invalid split, pass kind, policy, task, and status fail immediately; and
- constructing a batch does not clone tensor storage.

Use a storage-identity assertion rather than source-text inspection:

```python
batch = EvaluationBatch(outputs=outputs, targets=targets, num_examples=3)
assert batch.outputs.data_ptr() == outputs.data_ptr()
assert batch.targets.data_ptr() == targets.data_ptr()
```

**Step 2: Write failing lifecycle state-machine tests**

Test:

- initial idle state;
- `begin -> update -> snapshot -> finalize -> idle`;
- `begin -> update -> abort -> idle`;
- one batch, many uneven batches, and unequal final batches accumulate the
  exact `num_examples`;
- `snapshot` reports the count without resetting or incrementing it;
- a backend failure leaves the batch uncounted and abort clears partial state;
- begin while active, update/snapshot/finalize while idle, empty finalize, and
  mixed-context updates all fail;
- declared expected count mismatch fails before a result is published;
- finalize clears mutable backend state before a downstream consumer can fail;
- abort is idempotent only after a recorded active failure, not a universal
  no-op; and
- result order follows configured metric order.

Use a tiny fake backend so these tests do not depend on TorchMetrics policy yet.

**Step 3: Run the new tests to verify failure**

```bash
uv run pytest test/evaluator/test_types.py test/evaluator/test_lifecycle.py -q
```

Expected: FAIL because the typed contracts and lifecycle do not exist.

**Step 4: Implement the contracts and minimal lifecycle**

Implement public dataclasses/enums in `types.py`. Replace the no-op abstract
base with protocols/ABCs that name:

```python
begin(context: EvaluationContext) -> None
update(batch: EvaluationBatch) -> None
snapshot() -> EvaluationResult
finalize() -> EvaluationResult
abort() -> None
```

Implement the state machine in `TBEvaluator` with an injected minimal backend
mapping. Commit one batch to the observed counter only after every configured
backend update succeeds; abort the context on partial backend failure. Keep
TorchMetrics construction temporarily adapted behind that mapping; do not
expose it in `__init__.py`.

**Step 5: Run typed and adjacent evaluator tests**

```bash
uv run pytest \
  test/evaluator/test_types.py \
  test/evaluator/test_lifecycle.py \
  test/evaluator/test_graph_target_shapes.py -q
```

Expected: PASS.

**Step 6: Commit**

```bash
git add topobench/evaluator/types.py topobench/evaluator/base.py topobench/evaluator/evaluator.py topobench/evaluator/__init__.py test/evaluator/test_types.py test/evaluator/test_lifecycle.py
git commit -m "feat: add typed evaluator lifecycle"
```

---

### Task 3: Add the automatic pre-training gate early

**Files:**

- Create: `topobench/preflight.py`
- Create: `configs/preflight/default.yaml`
- Modify: `configs/run.yaml`
- Modify: `topobench/run.py`
- Create: `test/preflight/__init__.py`
- Create: `test/preflight/test_ordering.py`
- Create: `test/preflight/test_static_validation.py`
- Modify: `test/pipeline/test_pipeline.py`

**Step 1: Write failing ordering tests**

Patch construction boundaries and assert the sequence:

```text
seed -> build data pipeline -> static preflight -> isolated probe
-> restore seed -> production model/callback/logger/trainer -> fit
```

The tests must prove:

- preflight is enabled by default;
- no production logger or trainer is instantiated before preflight success;
- preflight failure prevents `trainer.fit` and `trainer.test`;
- explicit disable is accepted only under the experimental execution profile;
- disabled preflight marks returned/provenance qualification false; and
- normal successful preflight appears in `object_dict` for later provenance
  and logger publication.

**Step 2: Write failing static-validation tests**

Use composed tiny configs and reject before model execution:

- unknown task/metric/policy;
- invalid exact, topology-partition, or prefetch memory ceiling;
- malformed/internally overlapping named split tag or missing active tag;
- unqualified/stale store, partition book, sampler backend, or fitted-transform
  identity;
- exact/online policy incompatible with a custom metric;
- unsupported multi-rank qualified run;
- checkpoint monitor with possible NaN but no monitor policy;
- unsupported artifact/logger adapter;
- `save_reproducibility_bundle: false` in qualified mode; and
- unresolved OmegaConf interpolation.

No test contacts W&B or downloads data.

**Step 3: Run the new tests to verify failure**

```bash
uv run pytest \
  test/preflight/test_ordering.py \
  test/preflight/test_static_validation.py \
  test/pipeline/test_pipeline.py -q
```

Expected: FAIL because `topobench.preflight` and the config group do not exist.

**Step 4: Implement the minimal gate and structured report**

Create immutable `PreflightResult` and `PreflightCheck` values. Implement one
`PreflightRunner` with static checks and an initial isolated execution probe
over one representative batch. At this early boundary the probe must at least
instantiate a throwaway current model/evaluator, execute forward, supervision,
loss, and metric update, then discard the probe state. Tasks 4–7 migrate that
probe through the new evaluator contracts; Task 8 adds backward,
optimizer/scheduler, compilation, artifact-payload, disk-sampler, and complete
state-isolation qualification. Execution probing is never represented by a
placeholder success.

Wire the gate into `run()` after data-pipeline construction but before
production model, callback, logger, and trainer construction. Restore the
single authoritative seed after the probe boundary.

**Step 5: Run focused tests**

Run the Step 3 command.

Expected: PASS. The test proves runtime ordering, not source-string order.

**Step 6: Commit**

```bash
git add topobench/preflight.py configs/preflight/default.yaml configs/run.yaml topobench/run.py test/preflight test/pipeline/test_pipeline.py
git commit -m "feat: gate training with automatic preflight"
```

---

### Task 4: Replace dynamic discovery with explicit metric specifications

**Files:**

- Create: `topobench/evaluator/registry.py`
- Create: `topobench/evaluator/backends.py`
- Modify: `topobench/evaluator/evaluator.py`
- Modify: `topobench/evaluator/__init__.py`
- Delete: `topobench/evaluator/metrics/example.py`
- Rewrite: `topobench/evaluator/metrics/__init__.py` or delete the package if no
  surviving module needs it
- Create: `test/evaluator/test_registry.py`
- Create: `test/evaluator/test_extensions.py`
- Modify: `test/evaluator/test_evaluator.py`

**Step 1: Write failing registry tests**

Assert:

- the built-in registry is immutable and contains exactly the approved keys;
- every specification declares task, prediction view, backend group, exact
  factory, optional online factory, scalar contract, direction, and undefined
  behavior;
- binary `auroc` and `auprc` share one exact backend group;
- `auprc` declares average-precision integration and positive-class index `1`;
- `somers_d` declares $D_{S\mid Y}$ orientation, derives from `auroc`, and owns
  no backend state;
- both binary-only metrics reject any class count other than two;
- duplicate constructor-injected names, reserved `num_examples`, and generated
  audit-key collisions fail;
- unsupported task/policy combinations fail during evaluator construction;
- custom specifications can use a TopoBench backend protocol without importing
  TorchMetrics;
- global registration/mutation is unavailable; and
- `topobench.evaluator` exports no TorchMetrics class.

Do not test source text. Import public modules and inspect actual exported
objects/behavior.

**Step 2: Run tests to verify failure**

```bash
uv run pytest \
  test/evaluator/test_registry.py \
  test/evaluator/test_extensions.py \
  test/evaluator/test_evaluator.py -q
```

Expected: FAIL on missing specifications and current public `METRICS` mapping.

**Step 3: Implement explicit specs and adapters**

Create:

- an immutable `MetricSpec` with explicit backend-group/derived-metric fields;
- a structural TopoBench backend protocol;
- an internal TorchMetrics adapter that provides update/compute/reset/device
  and state behavior;
- explicit built-in factories using public task-specific TorchMetrics APIs;
  and
- a binary derived-metric specification for Somers' D with no independent
  retained state.

Reserve non-metric result/provenance field names, including `num_examples`, and
precompute all audit-expanded output keys during construction so collisions
fail before any state is allocated.

Use stateful `MulticlassAccuracy`, `MulticlassPrecision`,
`MulticlassRecall`, `MulticlassF1Score`, `MeanAbsoluteError`,
`MeanSquaredError`, and `R2Score` adapters for fixed-state metrics. Stateful
`BinaryAUROC`, `BinaryAveragePrecision`, and `MulticlassAUROC` modules are
permitted only for thresholded online state. Exact ranking factories must
target one TopoBench-owned grouped CPU-buffer backend and public
`binary_auroc`, `binary_average_precision`, and `multiclass_auroc` functional
compute calls; they must not construct per-metric stateful exact modules.
Configure aggregation explicitly. AUPRC's public key maps to TorchMetrics
average precision, not a second trapezoidal PR-area definition.

Delete `ExampleRegressionMetric`, private MSE functional imports, filesystem
scanning, `sys.path` mutation, exception-print discovery, and public `METRICS`.
Delete the metrics package entirely if it becomes empty.

**Step 4: Run focused tests**

Run the Step 2 command.

Expected: PASS.

**Step 5: Commit**

```bash
git add -A topobench/evaluator test/evaluator/test_registry.py test/evaluator/test_extensions.py test/evaluator/test_evaluator.py
git commit -m "refactor: make evaluator metrics explicit"
```

---

### Task 5: Implement exact, online, and audit policies

**Files:**

- Modify: `topobench/evaluator/types.py`
- Modify: `topobench/evaluator/registry.py`
- Modify: `topobench/evaluator/backends.py`
- Modify: `topobench/evaluator/evaluator.py`
- Create: `test/evaluator/test_metric_policies.py`
- Create: `test/evaluator/test_metric_semantics.py`

**Step 1: Write failing known-value and partition tests**

For classification, use hand-checkable logits/targets with all classes present
and assert accuracy, macro precision, macro recall, macro F1, and AUROC. On
binary fixtures also assert:

- `auprc` against `sklearn.metrics.average_precision_score`;
- `somers_d` against both $2\,\mathrm{AUROC}-1$ and
  `scipy.stats.somersd(targets, scores).statistic`;
- positive-class index `1`, tied-score behavior, and class-label reversal; and
- rejection under a three-class vocabulary.

For regression, assert MAE, MSE, RMSE, and R2. For every metric, compare one
all-at-once update with several uneven updates preserving order. Require equal
results within an explicit tolerance. Do not merely assert a value lies in
`[0, 1]`.

**Step 2: Write failing policy tests**

Assert:

- training context resolves to online;
- validation and test resolve to exact;
- explicit binary audit constructs one exact grouped ranking backend plus
  thresholded AUROC/AUPRC state while Somers' D derives from each AUROC;
- online ranking metrics use the configured shared 512-threshold grid;
- exact binary AUROC/AUPRC retain one score/target buffer;
- exact multiclass AUROC retains one `[N, C]` probability/target buffer;
- no stateful exact TorchMetrics ranking module is reachable from evaluator
  state;
- exact ranking metrics record exact status and no threshold grid;
- binary audit returns canonical `auroc`, `auprc`, and `somers_d`, plus
  `<metric>_online` and `<metric>_online_abs_error` for each;
- multiclass audit retains only the AUROC triplet;
- decomposable and derived metrics are not duplicated in audit mode;
- hard classes, full probabilities, and positive-class probabilities are
  derived at most once per update when required;
- no softmax is computed for regression or classification without a
  probability-based metric; and
- policy never changes after `begin`.

Use call-count instrumentation at the prediction-view boundary, not patches to
Torch internals.

**Step 3: Run tests to verify failure**

```bash
uv run pytest \
  test/evaluator/test_metric_policies.py \
  test/evaluator/test_metric_semantics.py -q
```

Expected: FAIL on absent policy backends and audit outputs.

**Step 4: Implement policy resolution and views**

Implement thresholded online AUROC/AUPRC with public stateful TorchMetrics
modules. Implement one grouped exact-ranking backend that owns detached CPU
chunks and invokes public TorchMetrics functional compute APIs sequentially.
Binary AUROC and AUPRC consume one positive-score/target buffer; multiclass
AUROC consumes one `[N, C]` probability/target buffer. No stateful exact
TorchMetrics ranking module may retain observations. Derive Somers' D from the
already computed AUROC. Keep standard metrics as streaming state under every
policy. Derive probability, positive-probability, and hard-class views once per
batch, only when requested.

Normal modes keep stable public keys. Audit keeps exact values canonical and
adds only the approved online and absolute-error suffixes.

**Step 5: Run focused tests**

Run the Step 3 command.

Expected: PASS.

**Step 6: Commit**

```bash
git add topobench/evaluator test/evaluator/test_metric_policies.py test/evaluator/test_metric_semantics.py
git commit -m "feat: add exact and online metric policies"
```

---

### Task 6: Enforce memory, device, and undefined-value contracts

**Files:**

- Modify: `topobench/evaluator/types.py`
- Modify: `topobench/evaluator/backends.py`
- Modify: `topobench/evaluator/evaluator.py`
- Create: `test/evaluator/test_scale_state.py`
- Create: `test/evaluator/test_undefined_metrics.py`
- Create: `test/evaluator/test_device_policy.py`

**Step 1: Write failing bounded-state tests**

Update online metrics with increasing synthetic populations and inspect actual
state tensors. Assert:

- decomposable state bytes do not grow with example count;
- thresholded AUROC/AUPRC state is bounded by classes and threshold count;
- exact binary and multiclass retained bytes grow monotonically according to
  their recorded layouts;
- enabling binary AUROC, AUPRC, and Somers' D retains exactly one shared
  score/target observation buffer;
- every tensor recursively reachable from exact backends is CPU-owned and
  included in retained-byte accounting;
- no stateful exact TorchMetrics ranking object retains hidden observations;
- exact snapshots do not duplicate retained storage; and
- `reset`, `finalize`, and `abort` release retained references.

Do not use process RSS for unit assertions; use owned tensor storage. Reserve RSS
checks for integration qualification.

**Step 2: Write failing exact-memory guard tests**

Cover known and unknown expected counts for binary and multiclass ranking
layouts. Account for one shared binary buffer or one `[N, C]` multiclass buffer
plus the largest sequential concatenation/sort compute workspace, not the sum
of all sequential metric workspaces. Assert preflight rejection when the
expected peak exceeds the ceiling and runtime rejection before the next append
would exceed it. The exception must name split, observed/projected examples,
projected bytes, configured limit, and the explicit online-policy remedy.
Prove no partial offending chunk or `num_examples` increment was committed.

**Step 3: Write failing undefined-policy tests**

Cover single-class binary AUROC/AUPRC/Somers' D, insufficient multiclass AUROC
support, absent macro class support, one-sample/constant-target R2, and empty
evaluation. Assert:

- qualified default `error` raises structured exceptions;
- explicit `nan` returns NaN plus reason/support metadata;
- neither mode coerces to zero; and
- a NaN-capable checkpoint monitor fails static preflight unless its monitor
  policy is explicit.

**Step 4: Write failing device tests**

Assert fixed/online state follows the evaluation device and every exact binary
or multiclass ranking chunk is detached CPU storage. On CUDA, use the mandatory
runner to prove no exact observation or TorchMetrics-owned state retains CUDA
storage, binary AUROC/AUPRC share observations, guarded multiclass finalization
succeeds, and measured peak memory stays within the declared estimate.
CPU-only unit tests validate the same policy metadata without pretending to
qualify CUDA.

**Step 5: Run tests to verify failure**

```bash
uv run pytest \
  test/evaluator/test_scale_state.py \
  test/evaluator/test_undefined_metrics.py \
  test/evaluator/test_device_policy.py -q
```

Expected: FAIL on missing guards/metadata/device separation.

**Step 6: Implement guards and policies**

Add one memory estimator shared by static preflight and runtime checks. Base it
on recursively owned tensor bytes, binary/multiclass layout, and the largest
sequential compute-workspace factor measured by integration tests; record
estimate inputs in `EvaluationResult`.

Move every exact ranking chunk to the grouped CPU backend. Keep online/fixed
state on the active metric device. Normalize undefined backend behavior into
TopoBench's `error` or `nan` contract before producing a result.

**Step 7: Run focused tests**

Run the Step 5 command.

Expected: PASS on CPU; CUDA-specific test is marked for the mandatory CUDA job,
not reported as locally qualified when unavailable.

**Step 8: Commit**

```bash
git add topobench/evaluator test/evaluator/test_scale_state.py test/evaluator/test_undefined_metrics.py test/evaluator/test_device_policy.py test/preflight/test_static_validation.py
git commit -m "feat: bound evaluator state and undefined metrics"
```

---

### Task 7: Wire typed evaluation through `TBModel`

**Files:**

- Modify: `topobench/model/model.py`
- Modify only if needed: `topobench/model/supervision.py`
- Modify: `test/model/test_model.py`
- Modify: `test/model/test_supervision.py`
- Modify: `test/evaluator/test_lifecycle.py`
- Modify: `test/pipeline/test_pipeline.py`
- Modify: `test/pipeline/test_heterogeneous_pipeline.py`

**Step 1: Write failing model-boundary tests**

Assert:

- supervision selection occurs exactly once per batch;
- `EvaluationBatch.outputs` and `.targets` alias the loss-owning selected
  tensors rather than cloned/reselected values;
- `num_examples` agrees across supervision, loss logging, evaluator, and batch
  size;
- the phase total is logged exactly once at finalize as
  `{train,val,test}/num_examples`, including unequal final batches;
- resumable batches propagate their canonical `sequence_id` without deriving
  it from dataloader worker arrival order;
- a failed batch does not contribute to the total;
- model steps no longer pass mutable `model_out` into the evaluator;
- train/validation/test hooks call begin/finalize with the correct split and
  pass kind;
- an exception in forward, supervision, loss, metric update, or logging aborts
  evaluator state; and
- one phase cannot leak into another.

**Step 2: Write failing dead-state test**

Instantiate `TBModel` and assert it has no `val_acc_best`,
`metric_collector_val`, `metric_collector_val2`, or `metric_collector_test`.
Assert there is no direct TorchMetrics import in `topobench.model` through an
import-isolation test.

**Step 3: Run tests to verify failure**

```bash
uv run pytest \
  test/model/test_model.py \
  test/model/test_supervision.py \
  test/evaluator/test_lifecycle.py \
  test/pipeline/test_pipeline.py \
  test/pipeline/test_heterogeneous_pipeline.py -q
```

Expected: FAIL because `TBModel` still calls `update(model_out)` and uses the
old hooks/state.

**Step 4: Implement one model/evaluator boundary**

Create the typed batch immediately after the supervision adapter selects the
loss-owning tensors. Pass the same batch to evaluator and, when enabled later,
the prediction sink. Keep `model_out["logits"]` and `model_out["labels"]` only
where the loss/readout runtime still requires them; do not let evaluator read
the dictionary.

Replace ad hoc reset/log methods with explicit begin/finalize/abort hook paths.
Remove the unused TorchMetrics `MeanMetric` and collectors. Preserve stable
Lightning metric keys and log `EvaluationResult.num_examples` once under the
same finalized phase namespace; never let Lightning average per-batch counts.

**Step 5: Run focused tests**

Run the Step 3 command.

Expected: PASS.

**Step 6: Commit**

```bash
git add topobench/model/model.py topobench/model/supervision.py test/model/test_model.py test/model/test_supervision.py test/evaluator/test_lifecycle.py test/pipeline/test_pipeline.py test/pipeline/test_heterogeneous_pipeline.py
git commit -m "refactor: route typed batches through evaluator"
```

---

### Task 8: Complete the isolated data and execution preflight

**Files:**

- Modify: `topobench/preflight.py`
- Modify: `topobench/data/pipelines/base.py`
- Modify: `topobench/dataloader/graph.py`
- Modify: `topobench/dataloader/heterogeneous.py`
- Modify only if present after companion plan execution:
  `topobench/dataloader/disk_graph.py`
- Create: `test/preflight/test_data_probe.py`
- Create: `test/preflight/test_execution_probe.py`
- Create: `test/preflight/test_state_isolation.py`
- Create: `test/data/dataload/test_graph_dataloader.py`
- Modify: `test/data/dataload/test_heterogeneous_dataloader.py`

**Step 1: Write failing non-committing batch tests**

For homogeneous, heterogeneous full-batch, heterogeneous cluster/neighbor,
hypergraph, and available materialized/disk strategies, assert preflight obtains
one representative batch for each enabled phase of the active split without
committing sampler state. Probe `FittableTransform` begin/update/finalize/apply
without publishing state or reading another phase.

Capture data-module/sampler/evaluator/transform state before and after. Recreate
production loaders and assert their first canonical partition/seed descriptor
matches a control run without preflight.

**Step 2: Write failing execution-flow tests**

With tiny deterministic fixtures, prove the throwaway probe executes:

- device transfer;
- model forward;
- supervision selection;
- loss and finiteness checks;
- evaluator begin/update/snapshot/abort;
- backward and nonempty finite gradients;
- optimizer construction and one step;
- scheduler construction and a semantically valid step where applicable;
- validation/test update paths;
- fitted-transform, structured check/profiling, and reproducibility payload
  validation without publication; and
- prediction-payload schema validation without final publication.

Patch production constructor boundaries and prove the probed model/optimizer
are discarded and newly instantiated for training.

**Step 3: Write failing compilation tests**

When `compile=false`, assert no compile call. When `compile=true`, execute the
configured compile path on the throwaway model and fail preflight on a real
compile/graph execution error. Provide an explicit test backend so unit tests
do not pay production compile cost; retain a separate integration test for the
qualified backend.

**Step 4: Write failing isolation tests**

Capture Python, NumPy, CPU/CUDA Torch, model, optimizer, evaluator, data module,
sampler, fitted-transform cache, event/check sinks, and artifact state. Assert
successful and failed preflights restore or discard all probe state before
production construction. Assert no W&B/CSV log, fitted state, checkpoint,
reproducibility bundle, promoted prediction directory, or final metric file is
created by the probe.

**Step 5: Run tests to verify failure**

```bash
uv run pytest \
  test/preflight/test_data_probe.py \
  test/preflight/test_execution_probe.py \
  test/preflight/test_state_isolation.py \
  test/data/dataload/test_graph_dataloader.py \
  test/data/dataload/test_heterogeneous_dataloader.py -q
```

Expected: FAIL on missing probe protocols and isolation behavior.

**Step 6: Implement one preflight probe protocol**

Extend the data-pipeline/data-module boundary with an explicit non-committing
probe API or a tested snapshot/restore protocol. Do not special-case concrete
classes in `run.py`.

Instantiate throwaway runtime objects from the same resolved configuration,
execute the approved data/fit/model/evaluator/artifact flow, release them,
restore every state authority, and only then construct production objects.
Produce stable structured check/timing/device/memory metadata without raw IDs
or tensors.

**Step 7: Run focused tests**

Run the Step 5 command.

Expected: PASS.

**Step 8: Commit**

```bash
git add topobench/preflight.py topobench/data/pipelines/base.py topobench/dataloader test/preflight test/data/dataload/test_graph_dataloader.py test/data/dataload/test_heterogeneous_dataloader.py
git commit -m "feat: exercise isolated pre-training dry run"
```

---

### Task 9: Make configuration and dependency semantics authoritative

**Files:**

- Modify: `configs/evaluator/default.yaml`
- Modify: `configs/evaluator/classification.yaml`
- Modify: `configs/evaluator/regression.yaml`
- Modify: `configs/preflight/default.yaml`
- Modify: `configs/run.yaml`
- Modify: `topobench/utils/config_resolvers.py`
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Modify: `test/config/test_all_surviving_configs.py`
- Modify: `test/utils/test_config_resolvers.py`
- Create: `test/evaluator/test_config.py`
- Modify: `test/architecture/verify_forbidden_imports.py`

**Step 1: Write failing composition tests**

Compose every surviving graph, heterogeneous, and hypergraph experiment and
assert resolved evaluator/preflight values:

```yaml
policy:
  train: online
  val: exact
  test: exact
online.ranking_thresholds: 512
exact.max_ranking_bytes: 536870912
exact.buffer_device: cpu
undefined_metric_policy: error
preflight.enabled: true
```

Assert dataset-provided metric overrides are validated against the active
registry, binary defaults depend on `num_classes == 2`, binary-only names fail
for other class counts, removed tasks/names fail, and no stale
`multioutput_classes`, `auroc_thresholds`, or `max_auroc_bytes` field survives.

**Step 2: Write failing dependency/import tests**

Assert TorchMetrics is a direct constrained project dependency and the locked
version imports with pinned Torch/Lightning in a clean process. Assert public
`topobench.evaluator` imports without optional removed-domain dependencies and
exports no TorchMetrics class.

**Step 3: Run tests to verify failure**

```bash
uv run pytest \
  test/evaluator/test_config.py \
  test/config/test_all_surviving_configs.py \
  test/utils/test_config_resolvers.py -q
uv run python test/architecture/verify_forbidden_imports.py
```

Expected: FAIL on old config surface and absent direct dependency.

**Step 4: Update configs, resolver, and lock**

Add the approved policy/resource/undefined fields and preflight group. Remove
multioutput configuration. Declare a TorchMetrics constraint qualified against
Python 3.11, Torch 2.3, and Lightning 2.4; let `uv sync` lock the exact version.
Do not change unrelated dependencies.

**Step 5: Run clean dependency and composition checks**

```bash
uv lock --check
uv sync --frozen --all-extras
uv run pytest \
  test/evaluator/test_config.py \
  test/config/test_all_surviving_configs.py \
  test/utils/test_config_resolvers.py -q
uv run python test/architecture/verify_forbidden_imports.py
```

Expected: PASS; `uv.lock` changes only as required by the explicit direct
constraint.

**Step 6: Commit**

```bash
git add configs/evaluator configs/preflight configs/run.yaml topobench/utils/config_resolvers.py pyproject.toml uv.lock test/evaluator/test_config.py test/config/test_all_surviving_configs.py test/utils/test_config_resolvers.py test/architecture/verify_forbidden_imports.py
git commit -m "build: qualify evaluator policy configuration"
```

---

### Task 10: Integrate selected-checkpoint results and artifact handoff

**Files:**

- Modify: `topobench/run.py`
- Modify: `topobench/model/model.py`
- Modify: `topobench/callbacks/best_epoch_metrics.py`
- Modify only if present after remediation Task 21:
  `topobench/callbacks/prediction_artifacts.py`
- Modify only if present after remediation Task 21:
  `topobench/evaluator/prediction.py`
- Modify: `test/callbacks/test_best_epoch_metrics.py`
- Modify only if present: `test/callbacks/test_prediction_artifacts.py`
- Create: `test/evaluator/test_selected_checkpoint_context.py`
- Modify: `test/pipeline/test_pipeline.py`
- Modify: `test/pipeline/test_heterogeneous_pipeline.py`

**Step 1: Write failing context and authority tests**

Assert ordinary fit validation uses `(split=val, pass_kind=fit_epoch)` and
selected-checkpoint reruns use independent
`(val, selected_checkpoint)` and `(test, selected_checkpoint)` contexts.

Prove:

- both reruns load the checkpoint selected only by validation;
- validation/test states and results are independent;
- test metrics never affect checkpoint selection;
- exact policy applies by default to both final contexts;
- one `EvaluationResult` feeds returned values, local metric serialization,
  every logger, and provenance;
- `EvaluationResult.num_examples` is identical in returned output,
  `metrics.json`, provenance, and selected-checkpoint logger keys;
- the artifact sink receives the same canonical `EvaluationBatch`;
- `num_examples` equals the prediction manifest's observed row count before
  atomic publication; and
- no monkey-patched lifecycle hook remains.

**Step 2: Write failing logger-key tests**

Assert stable ordinary keys, `{train,val,test}/num_examples`, and
selected-checkpoint namespaces. In binary audit mode, assert exact `auroc`,
`auprc`, and `somers_d` remain canonical while online/error keys are auxiliary.
Assert exactness/threshold metadata is logged separately and cannot collide
with scalar metric names.

**Step 3: Run tests to verify failure**

```bash
uv run pytest \
  test/evaluator/test_selected_checkpoint_context.py \
  test/callbacks/test_best_epoch_metrics.py \
  test/callbacks/test_prediction_artifacts.py \
  test/pipeline/test_pipeline.py \
  test/pipeline/test_heterogeneous_pipeline.py -q
```

If the artifact test file is not yet present, omit only that path; do not create
artifact persistence in this task.

Expected: FAIL on current rerun hook monkey-patch and metric authority split.

**Step 4: Implement explicit selected-checkpoint contexts**

Route validation and test reruns through the same typed evaluator lifecycle with
explicit pass kind. Replace direct metric-dictionary recomputation and hook
monkey-patching. Hand the canonical batch/result to remediation Task 21's sink
and serialization contracts when available.

**Step 5: Run focused tests**

Run the applicable Step 3 command.

Expected: PASS.

**Step 6: Commit**

```bash
git add topobench/run.py topobench/model/model.py topobench/callbacks test/evaluator/test_selected_checkpoint_context.py test/callbacks test/pipeline/test_pipeline.py test/pipeline/test_heterogeneous_pipeline.py
git commit -m "feat: unify selected-checkpoint evaluation results"
```

---

### Task 11: Prove checkpoint and failure recovery semantics

**Files:**

- Modify: `topobench/evaluator/backends.py`
- Modify: `topobench/evaluator/evaluator.py`
- Modify: `topobench/model/model.py`
- Modify only if present after companion disk-plan execution:
  `topobench/dataloader/disk_graph.py`
- Modify only if present after companion disk-plan execution:
  `topobench/dataloader/disk_heterogeneous.py`
- Modify: `topobench/preflight.py`
- Create: `test/evaluator/test_state_serialization.py`
- Create: `test/evaluator/test_failure_recovery.py`
- Modify: `test/pipeline/test_pipeline.py`
- Modify: `test/pipeline/test_heterogeneous_pipeline.py`
- Modify only if present: `test/pipeline/test_disk_graph_pipeline.py`
- Modify only if present: `test/pipeline/test_disk_heterogeneous_pipeline.py`
- Modify: `test/preflight/test_static_validation.py`

**Step 1: Write failing serialization tests**

Use a resumable descriptor loader with gradient accumulation greater than one.
Inject checkpoint requests:

- after a metric update but before the optimizer step;
- after the optimizer step but before sampler/evaluator commit;
- after sampler cursor, evaluator sequence prefix, and global step all agree;
  and
- at epoch completion.

Pending-boundary requests must defer or fail explicitly and publish no
checkpoint. Restore each valid checkpoint into a fresh runtime, regenerate only
uncommitted descriptors, process the remainder, and compare with uninterrupted
control:

- metric values and exact `num_examples`;
- ordered descriptor/`sequence_id` coverage;
- evaluator last committed sequence, sampler cursor, and model global step;
- exactness metadata; and
- no duplicated or omitted supervised examples.

Also prove ordinary in-memory checkpoint policy is epoch-boundary-only when no
stable sequence contract exists. Static preflight rejects qualified mid-epoch
resume combined with exact/audit training policy. Do not checkpoint active
exact validation/test buffers; assert the selected-checkpoint restart contract.

**Step 2: Write failing injected-failure tests**

Inject failures during forward, supervision, loss, each metric update,
count-commit, optimizer/global-step advance, sampler commit, evaluator sequence
commit, checkpoint serialization, snapshot, finalize, logger publication, and
artifact handoff. Assert the evaluator aborts, a partially failed batch is
never counted in an authoritative result, no mixed-boundary checkpoint is
published, retained buffers release, the next context begins cleanly, and no
partial result becomes authoritative.

**Step 3: Run tests to verify failure**

```bash
uv run pytest \
  test/evaluator/test_state_serialization.py \
  test/evaluator/test_failure_recovery.py \
  test/preflight/test_static_validation.py \
  test/pipeline/test_pipeline.py \
  test/pipeline/test_heterogeneous_pipeline.py \
  test/pipeline/test_disk_graph_pipeline.py \
  test/pipeline/test_disk_heterogeneous_pipeline.py -q
```

Expected: FAIL until live versus committed state and cleanup are coordinated
across evaluator, optimizer/global step, and resumable sampler.

**Step 4: Implement joint commit and cleanup hooks**

Track pending sequence IDs/counts after live metric updates. After a successful
optimizer step and sampler-cursor commit, mark exactly the same sequence prefix
committed in the evaluator. TopoBench-owned `state_dict` serialization must
refuse or defer while a microbatch is pending and must verify evaluator
sequence, sampler cursor, and global step alignment before returning state.
Restore only a context whose fingerprint and committed cursor match the current
task/policy/metrics/split. No unbounded rollback buffer is needed because
qualified checkpoints exist only at aligned boundaries.

Ensure finalization creates the immutable result before clearing mutable state
and that consumer failure cannot resurrect or overwrite it.

**Step 5: Run focused tests**

Run the Step 3 command.

Expected: PASS.

**Step 6: Commit**

```bash
git add topobench/evaluator topobench/model/model.py topobench/preflight.py topobench/dataloader/disk_graph.py topobench/dataloader/disk_heterogeneous.py test/evaluator/test_state_serialization.py test/evaluator/test_failure_recovery.py test/preflight/test_static_validation.py test/pipeline
git commit -m "feat: align evaluator and sampler resume state"
```

---

### Task 12: Remove obsolete evaluator and run surfaces

**Files:**

- Delete any surviving empty `topobench/evaluator/metrics/` files
- Modify: `topobench/evaluator/__init__.py`
- Modify: `topobench/evaluator/base.py`
- Modify: `topobench/evaluator/evaluator.py`
- Modify: `topobench/model/model.py`
- Modify: `topobench/utils/config_resolvers.py`
- Modify: `test/evaluator/test_evaluator.py`
- Modify: `test/loss/test_dataset_loss.py` only if removed task behavior still
  leaks into the surviving runtime contract
- Modify: `test/architecture/test_domain_contract.py`
- Modify: `test/architecture/verify_forbidden_imports.py`

**Step 1: Add negative architecture tests**

Prove no surviving evaluator/model/config runtime contains or exports:

- dynamic metric discovery;
- `LoadManager` for evaluator metrics;
- `METRICS` public mapping;
- `ExampleRegressionMetric`;
- TorchMetrics private functional imports;
- evaluator-owned generic prediction buffers;
- multilabel/multioutput evaluator branches or config fields;
- unused `MeanMetric` or metric collector fields; or
- `update(model_out)` compatibility behavior.

Use imports, symbols, construction, and behavior where possible. Token scans are
reserved for the explicit forbidden-surface architecture gate.

**Step 2: Run negative tests to verify any remaining failures**

```bash
uv run pytest \
  test/evaluator \
  test/architecture/test_domain_contract.py -q
uv run python test/architecture/verify_forbidden_imports.py
```

Expected: any stale surface fails with its exact path/symbol.

**Step 3: Delete stale code and tests**

Remove obsolete modules, comments, aliases, config keys, tests, and exports. Do
not leave deprecation shims in this breaking branch. Do not broaden the deletion
to unrelated loss/model discovery without a separately proven requirement.

**Step 4: Run focused tests**

Run the Step 2 command.

Expected: PASS.

**Step 5: Commit**

```bash
git add -A topobench/evaluator topobench/model/model.py topobench/utils/config_resolvers.py test/evaluator test/architecture test/loss/test_dataset_loss.py
git commit -m "refactor: remove legacy evaluator surfaces"
```

---

### Task 13: Run evaluator, preflight, and end-to-end qualification

**Files:**

- Modify only for proven defects: files owned by Tasks 1-12
- Modify: `README.md` only if current evaluator/dry-run usage is documented
  incorrectly
- Modify: relevant surviving source documentation only if it contains a real
  stale contract
- Modify: `docs/plans/2026-07-31-graph-heterogeneous-hypergraph-core-design.md`
  only to record implemented status after evidence exists
- Modify: `docs/plans/2026-07-31-research-production-remediation.md` only to
  record completed ownership/evidence after evidence exists
- Create: `test/integration/qualify_evaluator_cuda.py`
- Create: `test/integration/qualify_preflight_compile_cuda.py`

**Step 1: Run the complete focused suite**

```bash
uv run pytest \
  test/evaluator \
  test/preflight \
  test/model/test_model.py \
  test/model/test_supervision.py \
  test/callbacks/test_best_epoch_metrics.py \
  test/config/test_all_surviving_configs.py \
  test/utils/test_config_resolvers.py \
  test/architecture/test_domain_contract.py \
  test/pipeline/test_pipeline.py \
  test/pipeline/test_heterogeneous_pipeline.py -q
uv run python test/architecture/verify_forbidden_imports.py
```

Expected: PASS.

**Step 2: Run automatic-preflight smokes before training**

Run one-epoch CPU smokes for:

```bash
WANDB_MODE=disabled uv run python -m topobench dataset=graph/SyntheticGraph model=graph/gcn trainer.accelerator=cpu trainer.devices=1 trainer.max_epochs=1
WANDB_MODE=disabled uv run python -m topobench experiment=graph_synthetic_regression trainer.accelerator=cpu trainer.devices=1 trainer.max_epochs=1
WANDB_MODE=disabled uv run python -m topobench dataset=graph/SyntheticNodeGraph model=graph/gcn trainer.accelerator=cpu trainer.devices=1 trainer.max_epochs=1
WANDB_MODE=disabled uv run python -m topobench experiment=heterogeneous_synthetic_hgt_neighbor trainer.accelerator=cpu trainer.devices=1 trainer.max_epochs=1
WANDB_MODE=disabled uv run python -m topobench experiment=hypergraph_synthetic_edgnn trainer.accelerator=cpu trainer.devices=1 trainer.max_epochs=1
```

Use the exact surviving experiment selector names present at execution time; do
not invent aliases. For every run inspect structured output and assert:

- automatic preflight completed before training;
- production model state started pristine;
- train policy is online;
- validation/test policy is exact;
- binary runs report finite AUPRC and Somers' D with declared semantics;
- `num_examples` equals the known split participation count in phase logs,
  returned output, final JSON, prediction manifest, and provenance;
- selected-checkpoint validation/test are distinct; and
- preflight/evaluator exactness metadata is in local provenance and logger
  output.

**Step 3: Run scale and CUDA qualification**

On the mandatory CUDA runner, execute:

```bash
uv run pytest \
  test/evaluator/test_device_policy.py \
  test/integration/qualify_evaluator_cuda.py \
  test/integration/qualify_preflight_compile_cuda.py -q
```

Use representative synthetic large-count inputs to prove bounded online
ranking state, shared exact binary CPU buffering, guarded exact multiclass CPU
buffering, recursively measured owned state, measured compute peaks, configured
`torch.compile`, and no unintended CUDA retention. A missing CUDA runner or
skipped mandatory test fails the release job; local development may report the
qualification as not run, never as passed.

**Step 4: Run clean-environment and style gates**

```bash
uv lock --check
uv sync --frozen --all-extras
uv run ruff check .
uv run ruff format --check .
uv run python test/data/pipelines/verify_clean_import.py
```

Expected: PASS; clean import does not require removed topological packages and
TorchMetrics resolves through the direct qualified constraint.

**Step 5: Update only genuinely stale documentation**

After all behavior works, document:

- active metric keys, aggregation, binary positive class, AUPRC integration,
  and Somers' D orientation;
- train/validation/test defaults;
- exact ranking-memory ceiling and online ranking thresholds;
- audit mode keys;
- `num_examples` meaning and its phase/logger/JSON namespaces;
- undefined-metric policy;
- automatic preflight, experimental opt-out, and qualification effect; and
- the fact that TorchMetrics remains internal and directly declared.

Do not create a parallel evaluator guide if an existing relevant document can
be corrected.

**Step 6: Run the complete network-free suite and release smokes**

Run the repository's complete network-free suite plus the seven end-to-end
release smokes named by the integrated core/remediation plan. Then run opted-in
representative real-data qualification under the existing download gate.

Expected: no mandatory skip; every run records preflight, evaluator policy, and
participant-count evidence. Any exact-memory overrun, undefined metric,
count disagreement, unavailable mandatory probe, or stale config fails rather
than downgrading silently.

**Step 7: Commit qualification evidence**

```bash
git add README.md docs/plans/2026-07-31-scalable-evaluator-design.md docs/plans/2026-07-31-scalable-evaluator-implementation.md docs/plans/2026-07-31-graph-heterogeneous-hypergraph-core-design.md docs/plans/2026-07-31-graph-heterogeneous-hypergraph-core-implementation.md docs/plans/2026-07-31-research-production-remediation.md test/integration/qualify_evaluator_cuda.py test/integration/qualify_preflight_compile_cuda.py
git commit -m "test: qualify scalable evaluator and preflight"
```

---

### Task 14: Add sparse maintainer comments after integration

Run this task only after Tasks 1-13 and remediation Task 21 are integrated and
their behavioral qualification passes. Assign one review agent the final
integrated source, not this plan in isolation.

**Files:**

- Modify only where the review finds a real need: evaluator, preflight,
  `TBModel`, selected-checkpoint callback/artifact writer, and artifact-logging
  source files changed by the integrated implementation
- Do not modify tests, configuration, or documentation unless a comment reveals
  a real behavioral defect; route such a defect back through TDD before
  continuing this task

**Step 1: Review the final implementation for comprehension hazards**

Ask the agent to identify nested or non-obvious code where a maintainer cannot
recover the invariant or rationale from names and types alone. Prioritize:

- evaluator lifecycle transitions and the post-success `num_examples` commit;
- shared exact binary ranking state and the Somers' D derivation;
- projected-memory rejection before buffer mutation;
- joint evaluator/sampler/global-step checkpoint commit ordering, especially
  under gradient accumulation;
- absence of stateful exact ranking modules and complete recursive memory
  accounting for binary and multiclass CPU buffers;
- preflight snapshot/restore and non-committing sampler behavior;
- selected-checkpoint authority and val/test isolation; and
- atomic artifact promotion, resume, and logger-publication boundaries.

**Step 2: Add only rationale-bearing comments**

Add short comments immediately above the smallest relevant block. Each comment
must explain an invariant, ordering constraint, resource bound, failure
atomicity rule, or mathematically non-obvious choice. Do not restate syntax,
types, function names, or obvious control flow. Do not add module tours,
line-by-line narration, speculative notes, `TODO`s, commented-out code, or
comments solely to increase coverage. Remove or correct any stale comment the
review exposes.

**Step 3: Inspect the comment-only diff**

Require every added comment to answer both: “what would a reasonable
maintainer otherwise misunderstand?” and “why must this code remain ordered or
structured this way?” Revert comments that fail either test. Confirm no
executable token, public API, configuration, or test expectation changed.

**Step 4: Re-run focused and style gates**

```bash
uv run pytest test/evaluator test/preflight \
  test/model/test_model.py test/model/test_supervision.py \
  test/callbacks/test_best_epoch_metrics.py \
  test/callbacks/test_prediction_artifacts.py \
  test/utils/test_artifact_logging.py -q
uv run ruff check topobench
uv run ruff format --check topobench
```

Expected: PASS. The implementation has sparse rationale comments at the
identified comprehension hazards and no behavior change.

**Step 5: Commit the reviewed comments**

```bash
git add topobench/evaluator topobench/preflight.py topobench/model/model.py \
  topobench/callbacks topobench/utils/artifact_logging.py
git commit -m "docs: explain evaluator reliability invariants"
```

---

## Completion criteria

Implementation is complete only when all of the following are true:

- `TBEvaluator` accepts typed `EvaluationBatch` values through the approved
  lifecycle and returns typed `EvaluationResult` values;
- no public TopoBench evaluator API exposes TorchMetrics classes;
- TorchMetrics is directly declared, locked, and clean-sync qualified;
- core metric semantics are explicit and tested with known values;
- binary AUPRC is average precision, Somers' D is $D_{S\mid Y}$, and both reject
  non-binary vocabularies;
- online state is bounded in supervised count;
- exact ranking state is in-memory, TopoBench-owned and CPU-retained by default,
  shared for binary AUROC/AUPRC, guarded for binary and multiclass layouts by a
  declared retained-plus-compute byte ceiling, and never silently approximated;
- train defaults online and validation/test default exact;
- audit reports exact, online, and approximation error in one pass;
- undefined metrics default to an actionable failure and explicit `nan` mode is
  provenance-visible;
- every stage reports exact `num_examples`, and supervision, loss, evaluator,
  artifact handoff, logger output, returned output, final JSON, prediction
  manifest, and provenance agree;
- active training checkpoints serialize only a jointly committed evaluator
  sequence, sampler cursor, and model global step; pending gradient-accumulation
  batches are never checkpointed or double-counted on resume;
- automatic static/data/execution preflight runs before production training;
- the probe performs no logger/checkpoint/final-artifact/sampler commit and
  production state is pristine afterward;
- disabling preflight requires an experimental override and marks the run
  unqualified;
- multilabel/multioutput evaluator paths, dynamic metric discovery, example
  metric, private TorchMetrics APIs, and dead model metric fields are absent;
- a final agent review adds only sparse rationale-bearing comments at genuinely
  complex invariants after all behavior is qualified; and
- focused, clean-import, CPU smoke, mandatory CUDA, network-free, release, and
  final comment-only qualification gates pass with recorded evidence.
