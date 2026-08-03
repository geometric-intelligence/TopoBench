# Graph, Heterogeneous Graph, and Hypergraph Core Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reduce TopoBench to a high-quality native PyG core for homogeneous graphs, heterogeneous node classification, and lightweight hypergraph node classification, with no TopoModelX/TopoNetX dependency or rank-indexed runtime contract.

**Architecture:** Homogeneous graphs use native PyG `Data`/`Batch` and
ordinary datasets use `DataLoader`. Large homogeneous and heterogeneous
single-graph inputs are converted into one immutable, content-addressed
universal typed store with destination-oriented per-relation CSC and an
accepted typed partition book. The homogeneous adapter emits native `Data`
after erasing its synthetic node/relation type; heterogeneous cluster mode
emits native `HeteroData`, while heterogeneous neighbor mode exposes the same
arrays through PyG `FeatureStore`/`GraphStore` protocols to `NeighborLoader`.
Materialized PyG partition/subgraph/sampler paths are supported reference
implementations and behavioral oracles for disk strategies. Both strategies
share lazy worker reads, bounded pinned-host/CUDA prefetch, structured
profiling, committed-cursor resume, reproducibility evidence, the canonical
`EvaluationBatch`/`EvaluationResult`, and selected-checkpoint artifact
callbacks. Hypergraphs use a small `HypergraphData` subclass whose two
incidence rows batch with independent node and hyperedge offsets. Every domain
retains an explicit pipeline, model adapter, and validation boundary while
training, evaluation, callbacks, provenance, and logging remain in `TBModel`
and `topobench.run`.

**Tech Stack:** Python 3.11, PyTorch 2.3, PyTorch Geometric, Lightning 2.4, Hydra/OmegaConf, DuckDB/PyArrow in the explicit Parquet extra, NumPy memory maps, pytest, Ruff, uv.

---

## Execution rules

- Work directly on `topobench_graph_hetero`; do not create a worktree.
- Use test-driven development for every behavior change: one failing focused test, the smallest implementation, then focused and adjacent regression tests.
- Keep the existing native heterogeneous contracts green after every task. Run the heterogeneous sentinel suite listed below before each commit that touches shared runtime code.
- Do not add compatibility adapters for `x_0`, `batch_0`, `DomainData`, or `incidence_hyperedges`. This branch is a deliberate clean break.
- Do not delete a package or dependency until the surviving graph, heterogeneous, and hypergraph paths no longer import it.
- Use `git rm` only for paths named by this plan, inspect `git status --short` before every commit, and never discard unrelated user changes.

The heterogeneous sentinel suite is:

```bash
uv run pytest \
  test/data/test_heterogeneous_spec.py \
  test/data/dataload/test_heterogeneous_dataloader.py \
  test/nn/encoders/test_heterogeneous_node_encoder.py \
  test/nn/backbones/heterogeneous \
  test/nn/wrappers/heterogeneous \
  test/nn/readouts/test_heterogeneous_node_readout.py \
  test/model/test_supervision.py \
  test/pipeline/test_heterogeneous_pipeline.py -q
```

Expected during Tasks 1-14: all selected tests pass; tests that require an
optional sampler extension may retain their existing skip while unrelated
code is being migrated. At the Task 15 dependency gate and Task 18 release
gate, the real `NeighborLoader` consumption test is mandatory and must not
skip.

## Target runtime contracts

Homogeneous `Data`/`Batch`:

```text
x, edge_index, edge_attr?, edge_weight?, y,
batch (inductive batches), train_mask/val_mask/test_mask (transductive),
global_nid (disk-sampled transductive batches only)
```

Each graph-level example stores rank-one, length-one `y`. Classification uses
integral labels and produces `[B]` targets after batching. Scalar regression
uses floating labels and the supervision boundary normalizes batched `[B]`
labels exactly once to `[B, 1]`, matching `[B, 1]` logits. Transductive node
classification uses integral `[N]` labels. Other task kinds, target ranks,
dtypes, or non-finite regression targets are rejected before loss evaluation.

Heterogeneous `HeteroData`:

```text
per-type x/y/masks, edge_index_dict, metadata,
target-store batch_size and n_id for sampled batches
```

Hypergraph `HypergraphData`:

```text
x, hyperedge_index, num_hyperedges, y,
train_mask, val_mask, test_mask, batch
```

No surviving production module or configuration may use `x_0`, `batch_0`, `x_1`, `incidence_hyperedges`, `num_cell_dimensions`, or a lifting selector.

## Reviewed implementation decisions

The following decisions close gaps found during the critical plan review and
override any legacy behavior encountered during implementation:

- The supported homogeneous task set is graph-level binary/multiclass
  classification, graph-level scalar regression, single-graph transductive
  node classification, and inductive node classification over explicit
  training, validation, and test graph datasets. Node regression and
  multilabel graph classification remain unsupported.
- `graph/US-county-demos`, `graph/graphuniverse_inductive`, and
  `graph/ogbg-molpcba` remain removed. The new phase-dataset contract does not
  qualify a deleted legacy selector without its own manifest evidence.
- Every graph loader returns one of `UnsplitDataset`, `IndexedDataset`, or
  `PhaseDatasets`. One normalization boundary produces the datasets consumed
  by the graph data module.
- Generated splits are ordinary disjoint train/validation/test splits. They
  require explicit positive `train_prop`, `val_prop`, and `test_prop` values
  summing to one and use only local RNG state. K-fold and k-fold-fixed are not
  qualified split modes.
- Fixed indices remain in memory unless an approved adapter persists them.
  They select nodes for one node-task graph and examples for a multi-graph
  dataset. The boundary validates type, shape, bounds, uniqueness,
  non-emptiness, pairwise disjointness, and complete coverage.
- Generated and fixed multi-graph splits are index-backed `Subset` views and
  are never materialized as `list[Data]`. Already separated phase datasets
  remain separate and may contain one or more graphs per phase.
- Transductive mode requires exactly one source graph, no separate
  validation/test datasets, and full-length, mutually disjoint boolean masks.
  `full_graph` additionally requires graph `batch_size == 1`; `cluster_disk`
  samples cluster IDs from a versioned partition and does not reinterpret that
  count as graph-example batch size. Raw index arrays never cross the split
  boundary under a `*_mask` name.
- Graph classification targets remain `[B]`; scalar-regression targets and
  logits are exactly `[B, 1]`; node-classification targets are `[N]`.
  Supervision, loss, and metrics reject broadcasting-compatible mismatches.
- Epoch loss reductions are weighted by the number of supervised labels:
  graph labels for inductive graph tasks, selected nodes for transductive node
  tasks, and target seeds for sampled heterogeneous tasks.
- Every optional `edge_attr` or `edge_weight` has explicit dataset and model
  handling: `consume`, `ignore`, or `reject`. Unknown or implicit handling is
  invalid, and wrappers forward only fields a backbone declares it consumes.
- Hypergraph split generation occurs before hypergraph validation. Existing
  raw loaders are not expected to provide phase masks.
- Native hypergraph caches use a new schema/versioned filename. Old rank-based
  `data.pt` artifacts are rejected or bypassed, never silently reused.
- `torch-sparse` remains a required dependency until a separately tested
  `pyg-lib` migration replaces it; heterogeneous `NeighborLoader` batching is
  a release gate.
- Empty hyperedges are unsupported in v1. Batched hypergraph models infer the
  total hyperedge count from contiguous `hyperedge_index` or explicitly sum
  validated per-example counts.
- GCN, GAT, GIN, GPS, and NSD are target graph models. GraphMLP and GCN-DGM are
  conditional and remain only if their dedicated native-batching lifecycle
  tests pass.


### Approved universal typed-store follow-on

The native graph contract in Task 3 is a prerequisite, not the final
large-graph implementation. After the current candidate exists, execute
remediation Tasks 1–5, then all 15 tasks in
`docs/plans/2026-07-31-parquet-graph-ingestion-streaming-implementation.md`
before resuming remediation Task 9. Its authoritative design is
`docs/plans/2026-07-31-parquet-graph-ingestion-streaming-design.md`.

The follow-on contributes supported `materialized_reference` and `disk` modes,
one immutable typed partition book/store, and homogeneous cluster,
heterogeneous cluster, and heterogeneous neighbor strategies inside existing
TopoBench pipelines. The homogeneous reference ports and corrects pinned
`dleko11:on_disk_transductive` behavior. The heterogeneous reference invokes
PyG `Partitioner` on topology-only `HeteroData`; its default admission is
256 GiB, and an external map is required when the candidate is over budget or
fails hard balance.

Several explicit `{phase}_{unique_tag}` split triplets may coexist. Each tag's
train/validation/test IDs are unique and pairwise disjoint; tags may overlap.
Every run records one active tag. A typed partition adapter proves complete
identity round-trip, canonical relation direction, temporary-reverse isolation,
and hard per-type/phase/relation/byte/size balance before promotion.
Nondeterministic METIS reruns are not reproduction; the checksummed accepted
map is.

Disk cluster unions exactly match materialized `Data`/`HeteroData.subgraph`
oracles. Qualified deterministic disk neighbor batches exactly match
materialized `NeighborLoader` in ordered seeds, typed nodes, relations, hops,
fields, and supervision. Explicitly exhaustive cluster/neighbor configurations
must reproduce full-graph logits/metrics; realistically sampled final metrics
use predeclared paired-seed degradation bounds rather than an unjustified
equality claim.

A training-only `FittableTransform` pass supports bounded incremental PCA and
enumerates canonical active-split training entities once without validation/
test leakage. Immutable state is fingerprinted, atomically published, and
checkpointed. Batch transforms then run once after canonical assembly and
before transfer.

The store is built in staging and atomically renamed to its content hash.
Producer machines may publish a digest-pinned, non-executable pre-partitioned
bundle; consumers safely extract, validate, and promote it before memory-mapped
training. Qualified runs require the default-enabled reproducibility bundle
covering resolved config, dependency/source/environment state, partition,
split, transform, RNG, checkpoint, check evidence, and artifact digests.

One committed descriptor protocol aligns sampler, evaluator, and model global
step across prefetch and gradient accumulation. One structured event/check
stream profiles conversion through training, saves authoritative bounded local
evidence, and sends sampled aggregates/failures to W&B/logger adapters. Every
hard check exposes a stable ID, expected/observed evidence, remediation, and
local report path.

### Approved selected-checkpoint artifact follow-on

Remediation Task 21 implements
`docs/plans/2026-07-31-selected-checkpoint-prediction-artifacts-design.md`.
The existing validation-selected checkpoint is rerun once on validation and
once on test. Each split publishes independent final metrics and bounded,
pickle-free prediction shards containing canonical entity identity, targets,
raw outputs, exported predictions, optional target-transform values, and only
allowlisted lightweight metadata.

The writer is a shared TopoBench callback fed by the existing supervision
boundary and a pipeline-provided identity adapter. It neither reruns the model
nor introduces a domain-specific evaluator. Graph-level examples carry stable
sample IDs; homogeneous/hypergraph node predictions use canonical global IDs;
heterogeneous neighbor predictions include only target seeds and qualify IDs
by node type. Every file is registered separately with every configured logger
under distinct validation/test artifact names. Local run files remain
authoritative and are atomically promoted without silent overwrite.

### Approved scalable evaluator and automatic-preflight follow-on

Before remediation Task 21, execute Tasks 1–9 of
`docs/plans/2026-07-31-scalable-evaluator-implementation.md`. Its authoritative
design is `docs/plans/2026-07-31-scalable-evaluator-design.md`. The companion
plan refactors the existing `TBEvaluator`; it retains TorchMetrics internally
behind typed TopoBench contracts, adds exact/online/audit policies including
binary AUPRC and Somers' D, guards all exact binary/multiclass ranking state in
one TopoBench-owned CPU backend, commits exact per-context `num_examples`, and
adds the automatic isolated pre-training gate.

The dry-run framework is introduced in companion Task 3, before the metric and
selected-checkpoint integration work, and completed in Task 8. It is the one
preflight path later artifact/provenance work extends. Companion Task 10 is the
handoff into remediation Task 21: the artifact callback consumes the same
`EvaluationBatch` and `EvaluationResult` and does not create another evaluator
or authoritative metric dictionary. Companion Task 11 aligns evaluator
sequence/count state with sampler cursor and model global step at resumable
checkpoint boundaries. Then execute Tasks 12–14; Task 14 is the final sparse
maintainer-comment review after integrated qualification.

### Dataset and edge-feature policy

Every retained dataset is listed in an immutable surviving-dataset manifest
and is assigned exactly one pre-model node-feature policy:

| Input feature condition | Required policy |
| --- | --- |
| Rank-2 floating `x` | Validate and project |
| Integer categorical `x` | Deterministic one-hot encoding or a model-owned embedding |
| Missing `x` | Deterministic constant or degree features |
| Sparse `x` | Explicitly densify with a size guard, or reject with a documented reason |

Each manifest entry declares `edge_attr` and `edge_weight` availability as
`absent`, `optional`, or `required`. Each model capability declares `consume`,
`ignore`, or `reject` for both fields. A required dataset field may pair only
with a consuming model; an optional field may be ignored only when the pair
records that decision explicitly.


The manifest records selector, loader family, task kind, task level, split
mode, node/edge feature policies, required transform, compatible models, and
named qualification evidence. Every selector passes a selector-specific
metadata/config assertion plus loader, preprocessing, split, forward, loss,
and metric evidence. A loader-family integration test may be shared only when
the selector-specific assertion proves the same parser and contract.
Configuration composition alone is not evidence that a dataset is supported.

### Initial graph-model capability matrix

| Model selector | Graph classification | Scalar regression | Transductive node classification | Phase-separated inductive node classification | Native batching decision |
| --- | --- | --- | --- | --- | --- |
| `graph/gcn` | required | required | required | required | retain |
| `graph/gat` | required | required | required | required | retain |
| `graph/gin` | required | required | required | required | retain |
| `graph/gps` | required | required | required | required | retain after focused smoke |
| `graph/nsd` | required | required | required | required | retain after focused smoke |
| `graph/graph_mlp` | conditional | conditional | required | required | retain only after disjoint-batch loss test |
| `graph/gcn_dgm` | conditional | conditional | required | required | retain only after mask/batch isolation test |

Task 6 must replace every model's initially unknown edge-field behavior with
tested `consume`, `ignore`, or `reject` values; no capability entry may remain
unknown. If a conditional model fails its gate, remove its source, config,
special loss, registry entry, capability row, and registry/capability tests in
the same task. Do not weaken the gate or publish configuration-only support.

### Stop/go gates and known risks

| Risk | Required evidence | Decision if evidence fails |
| --- | --- | --- |
| Shared trainer changes regress heterogeneous learning | Full/full-batch and neighbor-sampled heterogeneous lifecycle sentinels | Stop the current task and repair before committing |
| A surviving dataset has no valid model or executable evidence | Exact manifest/config equality plus selector-specific qualification record | Delete the selector and orphaned loader/docs before release |
| Unsupported node-regression or multilabel selectors remain | Negative architecture test for the removed selectors and supported task set | Stop configuration pruning and remove the residue |
| Graph config composes but model is incompatible with task, batching, or edge fields | Capability-matrix lifecycle and edge-field forwarding/rejection tests | Reject the pair; remove a conditional model if its required gate fails |
| Regression targets broadcast silently | Exact `[B, 1]` logits/targets through supervision, loss, and metrics, including a smaller final batch | Stop graph migration and fix the target boundary |
| A transductive data module accepts more than one graph | Explicit-mode negative test with a two-item dataset | Stop graph migration and enforce singleton cardinality |
| A disk-streamed path retains a full source graph, full feature matrix, mapped edge table, or converted adjacency at runtime | Selected-read instrumentation, weak-reference/full-source absence checks, memory-map identity, and multi-worker lazy-open tests | Stop scale qualification and repair the universal store boundary |
| A homogeneous sampled union loses a directed edge or node identity | Exact multi-partition induced-union oracle with canonical `global_nid` | Stop graph qualification and repair CSC filtering/remapping |
| A heterogeneous batch changes relation direction/fanout or supervises context nodes | Per-relation CSC/fanout oracle and target `n_id[:batch_size]` supervision/export test | Stop heterogeneous qualification and repair the store/sampler boundary |
| Validation and test prediction artifacts collide, omit rows, or use different checkpoints | Real selected-checkpoint lifecycle with exact IDs/counts/digests, separate paths/logger names, and same checkpoint SHA-256 | Stop lifecycle qualification and repair the artifact callback/identity adapter |
| Split input modes disagree about ownership or permit leakage | Lifecycle tests for generated indices, supplied node/example indices, and independent phase datasets | Stop graph migration and repair the normalization boundary |
| Featureless/categorical datasets silently change semantics | Dataset-policy audit plus deterministic repeat test | Reject the dataset/config until an explicit policy exists |
| Hypergraph batching offsets only work for equal-size examples | Unequal-node/unequal-hyperedge batch test with reconstruction of both examples | Stop before model migration |
| Old rank-based cache is loaded as native data | Cache-version test with an old `data.pt` sentinel | Bypass/regenerate; never add a compatibility guess |
| A parser passes synthetic fixtures but fails actual archives | Mandatory selector/loader-family download gate, including pickle and content/edges hypergraphs | Release is blocked; local parser unit tests are insufficient |
| Dependency pruning leaves `NeighborLoader` importable but unusable | Consume a target-seed mini-batch after a fresh `uv sync` | Keep `torch-sparse` or install a separately qualified backend |
| Mass deletion hides an accidental public-surface loss | Separate neural, data, config, and docs commits with sentinel suites between them | Revert only the offending deletion commit and narrow its scope |

These are implementation gates, not aspirational checks. A failed required
model gate changes the published capability matrix. A failed dataset,
regression-shape, transductive-cardinality, shared-runtime, batching,
dependency, or real-format gate blocks completion.

### Task 1: Establish the reduced-domain architecture contract

**Files:**
- Create: `topobench/domains.py`
- Create: `test/architecture/__init__.py`
- Create: `test/architecture/test_domain_contract.py`
- Modify: `topobench/__init__.py`

**Step 1: Write the failing public-contract tests**

Add tests that require a closed domain set and reject unsupported names:

```python
import pytest

from topobench.domains import SUPPORTED_DOMAINS, require_supported_domain


def test_supported_domains_are_closed_and_ordered() -> None:
    assert SUPPORTED_DOMAINS == ("graph", "heterogeneous", "hypergraph")


@pytest.mark.parametrize("domain", ["cell", "simplicial", "combinatorial", "pointcloud"])
def test_removed_domains_are_rejected(domain: str) -> None:
    with pytest.raises(ValueError, match=f"Unsupported domain {domain!r}"):
        require_supported_domain(domain)
```

**Step 2: Verify the tests fail**

Run: `uv run pytest test/architecture/test_domain_contract.py -q`

Expected: FAIL because `topobench.domains` does not exist.

**Step 3: Implement the closed contract**

Use an immutable tuple and a narrow validator:

```python
SUPPORTED_DOMAINS = ("graph", "heterogeneous", "hypergraph")


def require_supported_domain(domain: str) -> str:
    if not isinstance(domain, str):
        raise TypeError("domain must be a string")
    if domain not in SUPPORTED_DOMAINS:
        raise ValueError(f"Unsupported domain {domain!r}; expected one of {SUPPORTED_DOMAINS}")
    return domain
```

Export only `SUPPORTED_DOMAINS` and `require_supported_domain` from this module. Do not yet assert that old directories are absent; that assertion belongs after deletion.

**Step 4: Run the focused test**

Run: `uv run pytest test/architecture/test_domain_contract.py -q`

Expected: PASS.

**Step 5: Commit**

```bash
git add topobench/domains.py topobench/__init__.py test/architecture
git commit -m "test: define reduced domain contract"
```

### Task 2: Replace filesystem discovery with explicit surviving registries

**Files:**
- Modify: `topobench/data/datasets/__init__.py`
- Modify: `topobench/data/loaders/__init__.py`
- Modify: `topobench/data/loaders/graph/__init__.py`
- Modify: `topobench/data/loaders/heterogeneous/__init__.py`
- Modify: `topobench/data/loaders/hypergraph/__init__.py`
- Modify: `topobench/nn/backbones/__init__.py`
- Modify: `topobench/nn/backbones/graph/__init__.py`
- Modify: `topobench/nn/backbones/heterogeneous/__init__.py`
- Modify: `topobench/nn/backbones/hypergraph/__init__.py`
- Modify: `topobench/nn/wrappers/__init__.py`
- Modify: `topobench/nn/wrappers/graph/__init__.py`
- Modify: `topobench/nn/wrappers/heterogeneous/__init__.py`
- Modify: `topobench/nn/wrappers/hypergraph/__init__.py`
- Create: `test/architecture/test_registries.py`

**Step 1: Write failing registry tests**

Assert that registries are ordinary explicit dictionaries, are deterministically ordered, and expose only intentional public classes from the three target domains. Include the current surviving classes first; add `HypergraphConvBackbone` in the hypergraph-model task.

```python
from topobench.nn import backbones
from topobench.nn import wrappers


def test_backbone_registry_has_only_surviving_local_models() -> None:
    assert tuple(backbones.MODEL_CLASSES) == tuple(sorted(backbones.MODEL_CLASSES))
    assert set(backbones.MODEL_CLASSES) == {
        "EDGNN", "GPSEncoder", "GraphMLP", "HGTBackbone",
        "HeteroSAGEBackbone", "NSDEncoder",
    }


def test_wrapper_registry_has_only_surviving_adapters() -> None:
    assert set(wrappers.WRAPPER_CLASSES) == {
        "GNNWrapper", "GraphMLPWrapper", "HeterogeneousWrapper", "HypergraphWrapper"
    }
```

Also test that importing `topobench.data.loaders`, `topobench.nn.backbones`, and `topobench.nn.wrappers` does not call `importlib`, `Path.iterdir`, or `spec_from_file_location`. A subprocess import test is preferable to implementation mocking.

**Step 2: Verify failure against dynamic discovery**

Run: `uv run pytest test/architecture/test_registries.py -q`

Expected: FAIL because removed-domain classes are discovered.

**Step 3: Replace each manager with explicit imports and maps**

Use this pattern in each registry:

```python
from .graph import GPSEncoder, GraphMLP, NSDEncoder
from .heterogeneous import HGTBackbone, HeteroSAGEBackbone
from .hypergraph import EDGNN

MODEL_CLASSES = dict(sorted({
    cls.__name__: cls
    for cls in (GPSEncoder, GraphMLP, NSDEncoder, HGTBackbone, HeteroSAGEBackbone, EDGNN)
}.items()))

__all__ = [*MODEL_CLASSES, "MODEL_CLASSES"]
globals().update(MODEL_CLASSES)
```

For loaders and datasets, import every class referenced by a surviving YAML `_target_`; do not construct targets by scanning filenames. Preserve public selector class names used by surviving configs.

Do not register helpers (`RedrawProjection`, EDGNN convolution internals, or
private metadata adapters). The four identity graph classes are removed unless
a surviving config or lifecycle test establishes a concrete use; there is no
generic `IdentityGNN` class in the current source.

**Step 4: Run focused and heterogeneous registry tests**

Run:

```bash
uv run pytest test/architecture/test_registries.py \
  test/nn/backbones/heterogeneous \
  test/nn/wrappers/heterogeneous -q
```

Expected: PASS.

**Step 5: Commit**

```bash
git add topobench/data/datasets/__init__.py topobench/data/loaders \
  topobench/nn/backbones topobench/nn/wrappers test/architecture/test_registries.py
git commit -m "refactor: make domain registries explicit"
```

### Task 3: Introduce explicit homogeneous split inputs and native batching

**Files:**
- Modify: `topobench/data/loaders/base.py`
- Modify: `topobench/data/splits.py`
- Modify: `topobench/data/utils/split_utils.py`
- Modify: `topobench/data/preprocessor/preprocessor.py`
- Modify: `topobench/data/pipelines/base.py`
- Modify: `topobench/data/pipelines/default.py`
- Modify: `topobench/dataloader/graph.py`
- Modify: retained homogeneous loaders under `topobench/data/loaders/graph/`
- Modify: `topobench/data/datasets/synthetic_graph_dataset.py`
- Create: `configs/dataset/graph/SyntheticInductiveNodeGraph.yaml`
- Create: `configs/experiment/graph_synthetic_inductive_node.yaml`
- Modify: `configs/dataset/graph/SyntheticGraph.yaml`
- Modify: `configs/dataset/graph/SyntheticNodeGraph.yaml`
- Modify: `configs/dataset/graph/SyntheticGraphRegression.yaml`
- Modify: `topobench/dataloader/__init__.py`
- Replace tests in: `test/data/dataload/test_Dataloaders.py`
- Replace tests in: `test/data/dataload/test_dataload_dataset.py`
- Modify: `test/data/utils/test_split_utils.py`
- Modify: `test/data/pipelines/test_data_pipelines.py`
- Modify: `test/model/test_supervision.py`
- Modify: `test/data/datasets/test_synthetic_graph_dataset.py`

All committed test data in this task must be independently invented. Do not
sample, perturb, anonymize, summarize, or otherwise derive fixtures from a
confidential dataset.

**Step 1: Write failing split-input contract tests**

Define tests around three mutually exclusive loader results:

```python
@dataclass(frozen=True)
class SplitIndices:
    train: IndexLike
    valid: IndexLike
    test: IndexLike


@dataclass(frozen=True)
class UnsplitDataset:
    dataset: Dataset[Data]


@dataclass(frozen=True)
class IndexedDataset:
    dataset: Dataset[Data]
    indices: SplitIndices


@dataclass(frozen=True)
class PhaseDatasets:
    train: Dataset[Data]
    valid: Dataset[Data]
    test: Dataset[Data]
```

Cover these observable contracts:

- generated, indexed, and phase-dataset inputs normalize to one train/valid/test
  result;
- index tensors must be non-boolean integral rank-one tensors;
- every phase is non-empty and internally unique;
- phases are pairwise disjoint and cover the source exactly;
- node indices are bounded by the one source graph's node count;
- example indices are bounded by the source dataset length;
- `PhaseDatasets` rejects empty phases but does not require a shared backing
  dataset;
- invalid input combinations fail before preprocessing or loader iteration;
- validation failures state only the invariant and counts, never index values.

Add a confidentiality sentinel asserting that exception text does not contain
the supplied index values or data tensor representations.

**Step 2: Verify the split-input tests fail**

Run:

```bash
uv run pytest test/data/dataload/test_dataload_dataset.py \
  test/data/utils/test_split_utils.py -q
```

Expected: FAIL because the explicit input types and common normalization
boundary do not exist.

**Step 3: Implement the typed input and validation boundary**

Add the frozen input types to `topobench/data/splits.py`; expose a closed
`GraphDatasetInput` union. Update `AbstractLoader` typing and every retained
homogeneous loader to return exactly one member of that union:

- loaders without an authoritative split return `UnsplitDataset`;
- OGB, molecule, ADME, and synthetic loaders with fixed example indices return
  `IndexedDataset`;
- an adapter that independently loads train, validation, and test data returns
  `PhaseDatasets` directly.

Do not use optional fields that allow contradictory states. Do not retain
`dataset.split_idx`, `dataset.split_idx_list`, or another ad hoc phase
attribute as a compatibility path. Preserve the canonical phase key `valid`
inside `SplitIndices`; use `val` only for Lightning method and metric names.

Implement one validator that normalizes accepted index sequences to CPU
`torch.long` tensors and checks shape, bounds, uniqueness, non-empty phases,
pairwise disjointness, and complete coverage. Complete coverage is mandatory
for this qualified product; a future partial-supervision policy would require
a separate explicit contract.

**Step 4: Replace generated splitting with ordinary three-way splits**

Generated `random` and `stratified` modes require:

```yaml
split_params:
  learning_setting: inductive
  split_type: stratified
  train_prop: 0.70
  val_prop: 0.15
  test_prop: 0.15
  data_seed: 0
```

Reject booleans, non-finite values, non-positive values, or a sum other than
one within an explicit tolerance. Derive integer phase sizes deterministically
with a documented largest-remainder rule and reject datasets too small to
produce three non-empty phases. Classification stratification must either
preserve every feasible class across phases or fail with a contextual error;
it must never silently fall back to random splitting.

Use a local `numpy.random.Generator` or deterministic sklearn
`random_state`; never call `torch.manual_seed`, `np.random.seed`, or mutate
Python's global RNG. Generate the inexpensive indices on every run rather than
persisting split files. Remove `k_fold_split`, `k_fold_split_fixed`, their
configuration choices, and tests that treat a held-out fold as both
validation and test.

Tests must prove:

- exact disjoint coverage;
- deterministic same-seed output;
- at least one different-seed assignment;
- declared phase sizes;
- classification balance where mathematically feasible;
- global NumPy and Torch RNG states are unchanged;
- no cache-hit/cache-miss branch exists for generated split identity.

**Step 5: Normalize split ownership through preprocessing**

Refactor the graph pipeline so preprocessing preserves the loader result:

```text
UnsplitDataset
  -> preprocess source
  -> generate and validate indices
  -> Subset phase views or transductive masks

IndexedDataset
  -> preprocess source without reordering
  -> validate supplied indices
  -> Subset phase views or transductive masks

PhaseDatasets
  -> preprocess train/valid/test in distinct phase cache namespaces
  -> validate each phase
  -> keep the three datasets separate
```

For one homogeneous node-task source, generated or supplied indices select
nodes and become canonical full-length `train_mask`, `val_mask`, and
`test_mask`. The data module then reuses that one graph in transductive mode.
For a multi-graph source, indices select examples and remain lazy `Subset`
views. `PhaseDatasets` is inductive: graph tasks supervise each graph, while
node tasks supervise every labeled node and do not create phase masks.

Accumulate preprocessing time across phase datasets. Ensure train, validation,
and test cannot reuse the same processed cache filename accidentally. Split
indices supplied in memory must not be serialized merely because preprocessing
is cached.

**Step 6: Generalize native PyG batching**

Keep the initial in-memory transductive constraints unchanged: one graph,
`batch_size == 1`, and no separate phase datasets. Build the explicit
`cluster_disk`/Parquet follow-on only after this native contract is green by
executing
`docs/plans/2026-07-31-parquet-graph-ingestion-streaming-implementation.md`.
For inductive mode, require three non-empty datasets
but remove the `Subset` and shared-source restrictions from `GraphDataModule`.
Training alone shuffles; validation and test remain deterministic.

Add batching tests for:

- fixed graph indices over one shared source;
- generated ordinary graph splits;
- one independently loaded node-classification graph per phase;
- multiple node-classification graphs in a phase;
- graph-level phase datasets with unrelated backing objects;
- transductive node indices converted to boolean masks;
- rejection of phase datasets in transductive mode.

For `task_level=node` plus `learning_setting=inductive`, verify the existing
`node_inductive` supervision path consumes every label in the current phase
batch and never asks for `train_mask`, `val_mask`, or `test_mask`.

**Step 7: Add one phase-separated node lifecycle**

Create a deterministic synthetic loader/config containing three independently
constructed graphs, one for each phase. Run one GCN training, validation, and
test step. Assert:

- each loader sees only its owned graph;
- node logits and labels have matching leading dimension;
- every phase loss is finite;
- validation and test do not mutate or acquire phase masks;
- train/validation/test graph objects and storage are distinct.

**Step 8: Run focused split, pipeline, and supervision tests**

Run:

```bash
uv run pytest test/data/dataload/test_Dataloaders.py \
  test/data/dataload/test_dataload_dataset.py \
  test/data/utils/test_split_utils.py \
  test/data/pipelines/test_data_pipelines.py \
  test/data/datasets/test_synthetic_graph_dataset.py \
  test/model/test_supervision.py -q
```

Expected: PASS.

**Step 9: Commit**

```bash
git add topobench/data/loaders topobench/data/splits.py \
  topobench/data/utils/split_utils.py topobench/data/preprocessor \
  topobench/data/pipelines topobench/dataloader \
  configs/dataset/graph configs/experiment/graph_synthetic_inductive_node.yaml \
  test/data test/model/test_supervision.py
git commit -m "feat: make graph split ownership explicit"
```

### Task 4: Migrate homogeneous feature encoding to `data.x`

**Files:**
- Create: `topobench/nn/encoders/graph_node_encoder.py`
- Create: `topobench/data/features.py`
- Create: `topobench/transforms/data_manipulations/constant_node_features.py`
- Modify: `topobench/nn/encoders/__init__.py`
- Modify: `topobench/data/pipelines/default.py`
- Modify: all retained files under `configs/dataset/graph/`; exclude the four
  selectors deleted in Task 6
- Modify: `configs/transforms/dataset_defaults/REDDIT-BINARY.yaml`
- Create: `test/nn/encoders/test_graph_node_encoder.py`
- Create: `test/data/test_graph_feature_contract.py`
- Create: `test/config/test_dataset_feature_policies.py`
- Modify: `test/nn/encoders/test_dgm.py`
- Modify: `topobench/nn/encoders/dgm_encoder.py`

**Step 1: Write failing encoder contract tests**

Test eager parameter creation, graph-aware normalization, floating rank-2
feature validation, missing `batch` fallback for a single graph, and in-place
`data.x` replacement. Explicitly assert no `x_0` is created.

```python
def test_graph_node_encoder_replaces_native_features() -> None:
    data = Batch.from_data_list([graph_a, graph_b])
    encoder = GraphNodeFeatureEncoder(3, 8, dropout=0.0)
    result = encoder(data)
    assert result is data
    assert result.x.shape == (graph_a.num_nodes + graph_b.num_nodes, 8)
    assert "x_0" not in result
```

**Step 2: Verify failure**

Run: `uv run pytest test/nn/encoders/test_graph_node_encoder.py -q`

Expected: FAIL because the encoder does not exist.

**Step 3: Implement the native encoder**

Use `GraphNorm`, `Linear`, activation, and dropout, but accept a scalar `in_channels` rather than a rank list:

```python
class GraphNodeFeatureEncoder(AbstractFeatureEncoder):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm = GraphNorm(in_channels)
        self.projection = torch.nn.Linear(in_channels, out_channels)
        self.activation = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, data: Data) -> Data:
        if not isinstance(data, Data) or isinstance(data, HeteroData):
            raise TypeError("GraphNodeFeatureEncoder requires homogeneous Data")
        if not isinstance(data.x, Tensor) or data.x.ndim != 2:
            raise ValueError("data.x must be a rank-2 tensor")
        batch = data.get("batch")
        data.x = self.dropout(self.activation(
            self.projection(self.norm(data.x, batch=batch))
        ))
        return data
```

Adapt `DGMStructureFeatureEncoder` only where required by `gcn_dgm.yaml`, changing its graph feature access from `x_0`/`batch_0` to `x`/`batch`. Do not preserve general rank support.

**Step 4: Make dataset feature preparation explicit**

Add a required `parameters.feature_policy` to every retained graph dataset
config. Accepted initial values are `continuous`, `categorical_one_hot`,
`degree`, and `constant`. The graph pipeline validates the post-transform
result against this contract before constructing loaders.

Implement `ConstantNodeFeatures` as a deterministic PyG transform for
featureless graphs and replace REDDIT-BINARY's random
`EqualGausFeatures` default. Keep ZINC's deterministic categorical encoding
and IMDB's degree encoding. Add a config audit that fails if any graph dataset
has no policy or if a policy lacks its required default transform.

Simplify `infer_in_channels()` so the no-lifting graph/hypergraph path returns
a scalar integer, not the legacy one-element rank list. Add resolver tests for
continuous, one-hot, degree, constant, and positional-encoding cases.

**Step 5: Run encoder and feature-policy tests**

Run:

```bash
uv run pytest test/nn/encoders/test_graph_node_encoder.py \
  test/nn/encoders/test_dgm.py \
  test/nn/encoders/test_heterogeneous_node_encoder.py \
  test/data/test_graph_feature_contract.py \
  test/config/test_dataset_feature_policies.py -q
```

Expected: PASS.

**Step 6: Commit**

```bash
git add topobench/nn/encoders topobench/data/features.py \
  topobench/data/pipelines/default.py \
  topobench/transforms/data_manipulations/constant_node_features.py \
  configs/dataset/graph configs/transforms/dataset_defaults/REDDIT-BINARY.yaml \
  test/nn/encoders test/data/test_graph_feature_contract.py \
  test/config/test_dataset_feature_policies.py
git commit -m "refactor: encode native graph features"
```

### Task 5: Migrate graph wrappers and readouts to `x` and `batch`

**Files:**
- Modify: `topobench/nn/wrappers/graph/gnn_wrapper.py`
- Modify: `topobench/nn/wrappers/graph/graph_mlp_wrapper.py`
- Modify: `topobench/nn/wrappers/graph/__init__.py`
- Create: `test/nn/wrappers/graph/test_graph_wrappers.py`
- Modify: `topobench/nn/readouts/base.py`
- Modify: `topobench/nn/readouts/identical.py`
- Modify: `topobench/nn/readouts/mlp_readout.py`
- Modify: `topobench/nn/readouts/__init__.py`
- Modify: `topobench/model/supervision.py`
- Modify: `test/nn/readouts/test_identical.py`
- Modify: `test/nn/readouts/test_mlp_readout.py`
- Modify: `test/model/test_supervision.py`
- Modify: `topobench/loss/dataset/DatasetLoss.py`
- Modify: `topobench/evaluator/evaluator.py`
- Create: `test/loss/test_graph_target_shapes.py`
- Create: `test/evaluator/test_graph_target_shapes.py`

**Step 1: Write failing node-, graph-, target-, and edge-field adapter tests**

Require `GNNWrapper` to return `{"x", "labels", "batch"}` and graph readouts
to consume that contract. Cover batched graph classification, batched scalar
regression, and single-graph node classification.

Add table-driven wrapper tests for `edge_attr` and `edge_weight`: `consume`
forwards the tensor, `ignore` omits it deliberately, and `reject` raises before
the backbone runs. Add negative tests for missing/invalid `x`, `edge_index`,
`y`, batched graph-level data without `batch`, and unknown edge modes.

Add supervision, loss, and evaluator tests requiring classification targets
`[B]`, scalar-regression logits/targets `[B, 1]`, and node-classification
targets `[N]`. Include `[B]` versus `[B, 1]`, `[B, 1, 1]`, non-floating or
non-finite regression targets, and a final batch of size one. Each mismatch
must raise rather than broadcast.

**Step 2: Verify failure**

Run:

```bash
uv run pytest test/nn/wrappers/graph/test_graph_wrappers.py \
  test/nn/readouts/test_identical.py test/nn/readouts/test_mlp_readout.py -q
```

Expected: FAIL because current components use rank-indexed keys.

**Step 3: Make graph wrappers independent of `AbstractWrapper`**

Use a plain `torch.nn.Module`; the wrapper owns backbone argument translation
and explicit optional-edge-field handling:

```python
from typing import Literal

class GNNWrapper(torch.nn.Module):
    def __init__(
        self,
        backbone: torch.nn.Module,
        edge_attr_mode: Literal["consume", "ignore", "reject"],
        edge_weight_mode: Literal["consume", "ignore", "reject"],
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.edge_modes = {
            "edge_attr": edge_attr_mode,
            "edge_weight": edge_weight_mode,
        }

    def forward(self, batch: Data) -> dict[str, Tensor | None]:
        kwargs = {"batch": batch.get("batch")}
        for field, mode in self.edge_modes.items():
            value = batch.get(field)
            if value is not None and mode == "reject":
                raise ValueError(f"{field} is unsupported by this model")
            if value is not None and mode == "consume":
                kwargs[field] = value
        x = self.backbone(batch.x, batch.edge_index, **kwargs)
        return {"x": x, "labels": batch.y, "batch": batch.get("batch")}
```

Validate both modes during construction. Adapt `GraphMLPWrapper` to the same
output and edge-mode contract. Keep residual connections inside a
graph-specific wrapper only if a focused test demonstrates an existing
surviving model needs them; do not retain rank iteration.

**Step 4: Refactor the graph readout base**

Rename internal concepts from zero-cells to nodes. `NoReadOut` remains the
configured public class name, but computes logits from `model_out["x"]`.
Graph pooling uses `model_out["batch"]` and `torch_geometric.utils.scatter`.
Supported node classification applies the linear head without pooling.
`MLPReadout` follows the same keys.

The heterogeneous readout remains untouched because it already consumes
`x_dict`. Refactor `MLPReadout` to own its `torch.nn.Sequential` layers instead
of importing `topobench.nn.backbones.non_relational.MLP`; this makes deletion
of the non-relational domain in Task 12 safe.

**Step 5: Enforce target shapes and homogeneous batch weighting**

Change `DefaultSupervisionAdapter` so graph classification reports `[B]`
targets and scalar regression normalizes source `[B]` labels once to `[B, 1]`.
Require prediction/target shape equality before `DatasetLoss` or `TBEvaluator`
runs; remove their unconditional target `unsqueeze` calls. Graph-level batches
report `num_examples=B`; transductive node classification reports
`mask.sum()`. Add a full batch plus smaller final batch and assert Lightning's
epoch loss uses supervised-example weighting.

**Step 6: Run focused, model, and supervision tests**

Run:

```bash
uv run pytest test/nn/wrappers/graph test/nn/readouts \
  test/model/test_model.py test/model/test_supervision.py \
  test/loss/test_graph_target_shapes.py \
  test/evaluator/test_graph_target_shapes.py -q
```

Expected: PASS after updating graph-only fixtures; heterogeneous readout,
loss, and evaluator tests remain green.

**Step 7: Commit**

```bash
git add topobench/nn/wrappers/graph topobench/nn/readouts \
  topobench/model/supervision.py topobench/loss/dataset/DatasetLoss.py \
  topobench/evaluator/evaluator.py test/nn/wrappers/graph test/nn/readouts \
  test/model test/loss/test_graph_target_shapes.py \
  test/evaluator/test_graph_target_shapes.py
git commit -m "refactor: enforce native graph model contracts"
```

### Task 6: Migrate all surviving homogeneous model configs and resolvers

**Files:**
- Modify: `configs/model/graph/gat.yaml`
- Modify: `configs/model/graph/gcn.yaml`
- Modify: `configs/model/graph/gcn_dgm.yaml`
- Modify: `configs/model/graph/gin.yaml`
- Modify: `configs/model/graph/gps.yaml`
- Modify: `configs/model/graph/graph_mlp.yaml`
- Modify: `configs/model/graph/nsd.yaml`
- Delete: `configs/dataset/graph/US-county-demos.yaml`
- Delete: `configs/dataset/graph/graphuniverse_inductive.yaml`
- Delete: `configs/dataset/graph/ogbg-molpcba.yaml`
- Delete: `configs/dataset/graph/manual_dataset.yaml`
- Modify: `topobench/utils/config_resolvers.py`
- Modify: `topobench/utils/model_instantiation.py`
- Create: `topobench/nn/capabilities.py`
- Create: `topobench/data/capabilities.py`
- Create: `test/config/test_surviving_graph_configs.py`
- Create: `test/pipeline/test_graph_model_capabilities.py`
- Create: `test/config/test_surviving_dataset_manifest.py`
- Modify: `test/pipeline/test_pipeline.py`

**Step 1: Write failing composition and instantiation tests**

Parametrize over the seven candidate graph configs and the three synthetic
task contracts. Compose each only with dataset/task/edge modes declared in the
capability matrix, resolve the config, and assert:

- `model.model_domain == "graph"`;
- feature encoder is `GraphNodeFeatureEncoder`;
- no resolved key or string contains `num_cell_dimensions`,
  `AllCellFeatureEncoder`, `x_0`, or a lifting target;
- task kind, learning setting, and edge modes are explicit;
- model instantiation succeeds without a runtime `data_spec`.

Use `OmegaConf.to_container(cfg, resolve=True)` and a recursive key/string
walker; do not rely on YAML text search alone.

Represent model capabilities in `topobench/nn/capabilities.py` and the exact
surviving-dataset manifest in `topobench/data/capabilities.py` as immutable
data used by config validation and tests. Require equality between manifest
selectors and surviving YAML files, require at least one compatible model per
dataset, and reject task, learning-setting, feature, or edge-mode pairings
absent from the matrices with a path-rich error before model construction.

**Step 2: Verify failure**

Run:
`uv run pytest test/config/test_surviving_graph_configs.py test/config/test_surviving_dataset_manifest.py -q`

Expected: FAIL because graph configs still select rank fields, the four
unsupported or legacy selectors still exist, and no exact manifest is defined.

**Step 3: Remove unsupported task selectors and rewrite surviving graph configs**

Delete `graph/US-county-demos`, `graph/graphuniverse_inductive`,
`graph/ogbg-molpcba`, and `graph/manual_dataset` before constructing the exact
manifest. Do not add feature policies or compatibility rows for them. Their
orphaned source paths are removed in Task 13.

Use scalar input widths and simple native adapters:

```yaml
feature_encoder:
  _target_: topobench.nn.encoders.GraphNodeFeatureEncoder
  in_channels: ${infer_in_channels:${dataset},${oc.select:transforms,null}}
  out_channels: 64
  dropout: 0.0

backbone_wrapper:
  _target_: topobench.nn.wrappers.GNNWrapper
  _partial_: true
  edge_attr_mode: ${validated_edge_attr_mode:${dataset},${model}}
  edge_weight_mode: ${validated_edge_weight_mode:${dataset},${model}}

readout:
  _target_: topobench.nn.readouts.NoReadOut
  hidden_dim: ${model.feature_encoder.out_channels}
  out_channels: ${dataset.parameters.num_classes}
  task_level: ${define_task_level:${dataset.parameters.task_level},${dataset.split_params.learning_setting}}
  pooling_type: sum
```

Keep each backbone's actual model-specific fields. Remove `wrapper_name`,
`readout_name`, `num_cell_dimensions`, and interpolation through those names
unless Hydra requires the public selector for an existing override.

**Step 4: Remove topological-only resolvers**

Delete registration and implementation for `infer_num_cell_dimensions`, `infer_topotune_num_cell_dimensions`, `get_required_lifting`, HOPSE/SANN dimension resolvers, and duplicated registrations. Simplify `get_default_transform()`:

```python
if data_domain != model_domain:
    raise ValueError(
        f"Cross-domain lifting is unsupported: dataset={data_domain!r}, model={model_domain!r}"
    )
return dataset_or_model_specific_default_if_present_else_no_transform
```

Do not allow graph-to-hypergraph or hypergraph-to-graph model composition in this branch.

**Step 5: Prove the model capability matrix**

Add focused one-epoch CPU lifecycle tests:

- GCN, GAT, and GIN on `SyntheticGraph` and `SyntheticNodeGraph`;
- GCN on `SyntheticGraphRegression`, including exact `[B, 1]` loss and metric
  assertions and a smaller final batch;
- regression forward/loss shape tests for every other model that declares
  scalar-regression support;
- GPS and NSD on both classification task levels with model-specific forward
  assertions;
- GraphMLP on transductive nodes plus an explicit disjoint-batch contrastive
  loss test that proves whether cross-graph pairs are handled intentionally;
- GCN-DGM on transductive nodes plus a batch-isolation test proving learned
  auxiliary edges and masks never mix examples;
- for every model, edge-attribute and edge-weight tests that replace unknown
  behavior with `consume`, `ignore`, or `reject`.

If GraphMLP or GCN-DGM fails its gate, remove its config, source, special loss,
registry entry, capability row, and corresponding expectations in
`test/architecture/test_registries.py` during this task. Do not postpone the
decision to final verification.

**Step 6: Make native graph pipelines the broad graph sentinels**

Change `test/pipeline/test_pipeline.py` to cover `graph/gcn` on both
`graph/SyntheticGraph` and `graph/SyntheticGraphRegression`, two epochs, CPU,
real batch sizes greater than one, and final testing. Keep download-marked
MUTAG classification and ZINC or AQSOL scalar-regression lifecycle tests.
They are mandatory release gates even when excluded from ordinary network-free
CI. Add separate model-config unit coverage instead of looping downloaded
models in the ordinary lifecycle test.

**Step 7: Run configuration and graph model tests**

Run:

```bash
uv run pytest test/config/test_surviving_graph_configs.py \
  test/config/test_surviving_dataset_manifest.py \
  test/nn/backbones/graph test/pipeline/test_graph_model_capabilities.py \
  test/pipeline/test_pipeline.py -q
```

Expected: PASS; classification and scalar-regression lifecycle tests report
observed training batch sizes greater than one and exact target shapes.

**Step 8: Commit**

```bash
git add configs/model/graph configs/dataset/graph topobench/utils \
  topobench/nn/capabilities.py topobench/data/capabilities.py test/config \
  test/pipeline/test_graph_model_capabilities.py \
  test/pipeline/test_pipeline.py test/nn/backbones/graph \
  test/architecture/test_registries.py
git commit -m "refactor: qualify native graph capabilities"
```

### Task 7: Define and validate native hypergraph data

**Files:**
- Create: `topobench/data/hypergraph.py`
- Create: `topobench/data/datasets/synthetic_hypergraph_dataset.py`
- Create: `topobench/data/loaders/hypergraph/synthetic.py`
- Create: `configs/dataset/hypergraph/SyntheticHypergraph.yaml`
- Modify: `topobench/data/__init__.py`
- Create: `test/data/test_hypergraph_data.py`
- Create: `test/data/datasets/test_synthetic_hypergraph_dataset.py`
- Modify: `test/conftest.py`

**Step 1: Write failing batching and validation tests**

Create two synthetic hypergraphs with different node and hyperedge counts. Batch them and assert the second graph receives independent row offsets:

```python
def test_hypergraph_batch_offsets_nodes_and_hyperedges_independently() -> None:
    batch = Batch.from_data_list([first, second])
    second_incidence = batch.hyperedge_index[:, first.hyperedge_index.size(1):]
    assert torch.equal(second_incidence[0], second.hyperedge_index[0] + first.num_nodes)
    assert torch.equal(second_incidence[1], second.hyperedge_index[1] + first.num_hyperedges)
```

Add validation failures for wrong shape/dtype, negative indices, out-of-bounds
nodes, missing/invalid `num_hyperedges`, non-contiguous hyperedge IDs, empty
hyperedges, label count mismatch, overlapping or incomplete masks, and empty
supervised splits. Test that batching stores per-example hyperedge counts and
that callers must not interpret the collated tensor as a scalar total.

**Step 2: Verify failure**

Run: `uv run pytest test/data/test_hypergraph_data.py -q`

Expected: FAIL because `HypergraphData` does not exist.

**Step 3: Implement `HypergraphData.__inc__`**

```python
class HypergraphData(Data):
    representation_version = 2

    def __inc__(self, key: str, value: Tensor, *args, **kwargs):
        if key == "hyperedge_index":
            if self.num_nodes is None or self.num_hyperedges is None:
                raise ValueError("hypergraph batching requires node and hyperedge counts")
            return torch.tensor([[self.num_nodes], [self.num_hyperedges]], device=value.device)
        return super().__inc__(key, value, *args, **kwargs)
```

**Step 4: Implement transactional validation**

`validate_hypergraph_node_data(data)` returns the same object only after all checks pass. It must not mutate or renumber malformed input. Require a rank-2 floating `x`, long `hyperedge_index` of shape `[2, M]`, contiguous hyperedge IDs `0..num_hyperedges-1`, rank-1 node labels, and rank-1 boolean masks whose length equals `num_nodes` and which partition labeled nodes.

Require every hyperedge identifier in `0..num_hyperedges-1` to occur at least
once; empty hyperedges are unsupported in v1. Validate each individual graph
before batching. Define a shared native cache filename such as
`hypergraph_data_v2.pt` and a representation-version constant for dataset
loaders to use in Task 8.

Add a small deterministic `synthetic_hypergraph` fixture in `test/conftest.py`; do not import any lifting or topological library to construct it.

Promote the same generator into a small production
`SyntheticHypergraphDataset` and loader so CLI and lifecycle tests exercise
the real loader/preprocessor boundary without network access. The test fixture
must call the production factory and then clone its result; do not maintain a
second synthetic topology in the test tree.

**Step 5: Run tests**

Run: `uv run pytest test/data/test_hypergraph_data.py test/data/datasets/test_synthetic_hypergraph_dataset.py -q`

Expected: PASS.

**Step 6: Commit**

```bash
git add topobench/data/hypergraph.py topobench/data/__init__.py \
  topobench/data/datasets/synthetic_hypergraph_dataset.py \
  topobench/data/loaders/hypergraph/synthetic.py \
  configs/dataset/hypergraph/SyntheticHypergraph.yaml \
  test/data/test_hypergraph_data.py \
  test/data/datasets/test_synthetic_hypergraph_dataset.py test/conftest.py
git commit -m "feat: add native hypergraph data contract"
```

### Task 8: Convert hypergraph datasets and parsing to native incidence indices

**Files:**
- Create: `topobench/data/utils/common.py`
- Create: `topobench/data/utils/downloads.py`
- Create: `topobench/data/utils/hypergraph_io.py`
- Modify: `topobench/data/utils/__init__.py`
- Modify: `topobench/data/datasets/citation_hypergraph_dataset.py`
- Modify: `topobench/data/datasets/hypergraph_datasets.py`
- Modify: `topobench/data/loaders/hypergraph/citation_hypergraph_dataset_loader.py`
- Modify: `topobench/data/loaders/hypergraph/hypergraph_dataset_loader.py`
- Modify: `test/data/utils/test_io_utils.py`
- Create: `test/data/load/test_hypergraph_dataset_loaders.py`
- Create: `test/integration/test_real_hypergraph_formats.py`

**Step 1: Write failing parser tests using local temporary raw fixtures**

Cover both content-style and pickle-style hypergraph inputs without downloading. Require each parser to return `HypergraphData` with canonical, sorted, duplicate-free `hyperedge_index`, contiguous hyperedge IDs, and explicit `num_hyperedges`.

Include a regression test where raw hyperedge identifiers are sparse strings or integers; the parser must remap them deterministically while leaving node indices aligned with features and labels.

Test isolated-node behavior explicitly. Preserve the pickle loader's existing
singleton-hyperedge policy or document and test a deliberate replacement; do
not silently drop isolated nodes during canonicalization.

**Step 2: Verify failure**

Run:

```bash
uv run pytest test/data/utils/test_io_utils.py \
  test/data/load/test_hypergraph_dataset_loaders.py -q
```

Expected: FAIL because current parsers emit sparse `incidence_hyperedges` and import the topological utility umbrella.

**Step 3: Split dependency-free common utilities**

Move only `make_hash` and `ensure_serializable` into `topobench/data/utils/common.py`. Import these narrow modules from `PreProcessor` and graph transforms. Do not let `topobench.data.utils.__init__` import removed complex utilities as a side effect.

Move `download_file_from_drive` and `download_file_from_link` into
`downloads.py`. Do not migrate `read_us_county_demos`; it remains isolated with
the unsupported US County loader until both are deleted in Task 13. Update
surviving dataset modules to import the narrow module that owns each symbol.
This permits complete deletion of mixed `io_utils.py`, whose module-level
TopoModelX/TopoNetX imports currently poison otherwise native graph and
hypergraph loaders.

**Step 4: Implement canonical hyperedge conversion**

Centralize the conversion:

```python
def incidence_pairs(hyperedges: Mapping[Hashable, Iterable[int]], num_nodes: int) -> tuple[Tensor, int]:
    ordered_ids = sorted(hyperedges, key=lambda value: (type(value).__name__, repr(value)))
    hyperedge_id = {raw: index for index, raw in enumerate(ordered_ids)}
    pairs = sorted({(int(node), hyperedge_id[raw])
                    for raw in ordered_ids for node in hyperedges[raw]})
    if any(node < 0 or node >= num_nodes for node, _ in pairs):
        raise ValueError("hyperedge contains an out-of-bounds node")
    index = torch.tensor(pairs, dtype=torch.long).t().contiguous()
    return index if pairs else torch.empty((2, 0), dtype=torch.long), len(ordered_ids)
```

Make dataset `process()` methods save the canonical class through PyG's normal
`InMemoryDataset.save()`/`load()` path. Set their processed filename to the
versioned native name from Task 7 and persist `representation_version=2`.
Remove backward handling for rank-based processed artifacts. If an old
`data.pt` is present, bypass it and emit one concise regeneration notice; never
load it as native data.

**Step 5: Add gated real-format integration smokes**

Add `download` and `integration` marked tests for one small pickle-format
dataset (for example cocitation Cora) and one content/edges-format dataset
(Zoo or Mushroom). Verify archive extraction, feature width, class range,
contiguous incidence IDs, and the declared cache version. These tests are
excluded from normal CI but are mandatory before release.

**Step 6: Run parser and loader tests**

Run:

```bash
uv run pytest test/data/utils/test_io_utils.py \
  test/data/load/test_hypergraph_dataset_loaders.py \
  test/data/test_hypergraph_data.py -q
```

Expected: PASS, with no network access.

When network access is explicitly available, run:

```bash
uv run pytest test/integration/test_real_hypergraph_formats.py \
  -m "download and integration" -q
```

Expected: both raw-format smokes PASS.

**Step 7: Commit**

```bash
git add topobench/data/utils topobench/data/datasets \
  topobench/data/loaders/hypergraph test/data \
  test/integration/test_real_hypergraph_formats.py
git commit -m "refactor: load native hypergraph incidence data"
```

### Task 9: Add an explicit hypergraph node pipeline with native batching

**Files:**
- Create: `topobench/data/pipelines/hypergraph.py`
- Modify: `topobench/data/pipelines/__init__.py`
- Create: `topobench/dataloader/hypergraph.py`
- Modify: `topobench/dataloader/__init__.py`
- Create: `configs/data_pipeline/hypergraph_node.yaml`
- Modify: all files under `configs/dataset/hypergraph/`
- Create: `test/data/dataload/test_hypergraph_dataloader.py`
- Modify: `test/data/pipelines/test_data_pipelines.py`

**Step 1: Write failing pipeline tests**

Mock the loader with one structural `HypergraphData` that has no phase masks.
Verify the pipeline creates masks from `cfg.dataset.split_params` before full
validation, then require train/validation/test loaders to return native
`Batch` objects. Verify phase masks remain boolean on the batch and
`batch_size=1` is enforced for this transductive v1.

**Step 2: Verify failure**

Run:

```bash
uv run pytest test/data/dataload/test_hypergraph_dataloader.py \
  test/data/pipelines/test_data_pipelines.py -q
```

Expected: FAIL because no hypergraph-specific pipeline exists.

**Step 3: Implement the data module and pipeline**

`HypergraphNodeDataModule` may reuse the same internal PyG `DataLoader` construction as `GraphDataModule`, but it must expose a separate public class and validate `HypergraphData`. `HypergraphNodeDataPipeline` must:

1. preprocess exactly one graph;
2. require `HypergraphData` rather than generic `Data`;
3. validate representation-only invariants that do not require phase masks;
4. generate or load split indices with the shared split algorithms;
5. apply full-length boolean masks through
   `topobench.data.splits.apply_transductive_split`;
6. call `validate_hypergraph_node_data` on the completed object;
7. build the hypergraph data module;
8. return `DataPipelineOutput` with no heterogeneous `data_spec`.

Add a failure-atomicity test: invalid split parameters or invalid masks must
not partially mutate the cached/source hypergraph. Apply masks to a cloned
runtime object after all split indices validate.

Normalize every hypergraph dataset's `dataloader_params.batch_size` to `1`
(replace current `-1`). Dataset YAML remains scoped under `dataset`; it must
not try to override a Hydra sibling group. Select
`data_pipeline=hypergraph_node` explicitly in every hypergraph experiment and
documented command, and add a config test that rejects a hypergraph dataset
paired with the default or heterogeneous pipeline.

**Step 4: Run pipeline and hetero sentinels**

Run:

```bash
uv run pytest test/data/dataload/test_hypergraph_dataloader.py \
  test/data/pipelines/test_data_pipelines.py -q
```

Then run the heterogeneous sentinel suite.

Expected: PASS.

**Step 5: Commit**

```bash
git add topobench/data/pipelines topobench/dataloader \
  configs/data_pipeline configs/dataset/hypergraph test/data
git commit -m "feat: add native hypergraph data pipeline"
```

### Task 10: Migrate EDGNN and add a PyG HypergraphConv baseline

**Files:**
- Modify: `topobench/nn/backbones/hypergraph/edgnn.py`
- Create: `topobench/nn/backbones/hypergraph/hypergraph_conv.py`
- Modify: `topobench/nn/backbones/hypergraph/__init__.py`
- Modify: `topobench/nn/backbones/__init__.py`
- Modify: `topobench/nn/wrappers/hypergraph/hypergraph_wrapper.py`
- Modify: `topobench/nn/wrappers/hypergraph/__init__.py`
- Modify: `configs/model/hypergraph/edgnn.yaml`
- Create: `configs/model/hypergraph/hypergraph_conv.yaml`
- Modify: `test/nn/backbones/hypergraph/test_edgnn.py`
- Create: `test/nn/backbones/hypergraph/test_hypergraph_conv.py`
- Create: `test/nn/wrappers/hypergraph/test_hypergraph_wrapper.py`
- Create: `test/pipeline/test_hypergraph_pipeline.py`

**Step 1: Write failing model and wrapper tests**

Require both backbones to accept `(x, hyperedge_index)` and return node
embeddings of shape `[num_nodes, hidden_channels]`. Require
`HypergraphWrapper` to emit the same `{"x", "labels", "batch"}` contract as
graph wrappers. Add invalid incidence and output-shape tests. Batch two
hypergraphs and prove the backbone does not pass the collated per-example
`num_hyperedges` tensor as a scalar `num_edges` argument.

**Step 2: Verify failure**

Run:

```bash
uv run pytest test/nn/backbones/hypergraph \
  test/nn/wrappers/hypergraph -q
```

Expected: FAIL because EDGNN returns a tuple and the wrapper consumes sparse rank fields.

**Step 3: Refactor EDGNN narrowly**

- Keep its local EquivSet implementation.
- Accept only dense long `hyperedge_index` shaped `[2, M]`; remove sparse-incidence compatibility.
- Return node embeddings only.
- Replace direct `torch_scatter` calls with the already available PyG scatter helper only if parity tests pass; otherwise retain `torch-scatter` because NSD also uses it.

**Step 4: Implement the PyG baseline**

```python
class HypergraphConvBackbone(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int,
                 num_layers: int, dropout: float = 0.0) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be positive")
        widths = [in_channels] + [hidden_channels] * num_layers
        self.convs = torch.nn.ModuleList(
            HypergraphConv(widths[i], widths[i + 1]) for i in range(num_layers)
        )
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x: Tensor, hyperedge_index: Tensor) -> Tensor:
        for layer_index, conv in enumerate(self.convs):
            x = conv(x, hyperedge_index)
            if layer_index + 1 < len(self.convs):
                x = self.dropout(F.relu(x))
        return x
```

The pinned PyG 2.8 signature accepts optional `num_edges`. Prefer omitting it
and allow inference from the validated contiguous incidence IDs. If an
explicit value is needed, compute the sum of validated per-example counts;
never use `batch.num_hyperedges` as though it were already a scalar total.

**Step 5: Rewrite hypergraph configs**

Both models use `GraphNodeFeatureEncoder`, `HypergraphWrapper`, and the native node readout. Remove `AllCellFeatureEncoder`, `PropagateSignalDown`, `num_cell_dimensions`, and rank terminology. Keep selector `hypergraph/edgnn`; add selector `hypergraph/hypergraph_conv`.

**Step 6: Add a network-free lifecycle test**

Run one epoch for each hypergraph model on the synthetic fixture through `HypergraphNodeDataPipeline`, then execute final testing using the best checkpoint. Assert `train/loss`, `val/loss`, and `test/loss` are finite.

**Step 7: Run hypergraph and shared-model tests**

Run:

```bash
uv run pytest test/nn/backbones/hypergraph test/nn/wrappers/hypergraph \
  test/pipeline/test_hypergraph_pipeline.py \
  test/model/test_model.py test/model/test_supervision.py -q
```

Expected: PASS.

**Step 8: Commit**

```bash
git add topobench/nn/backbones/hypergraph topobench/nn/wrappers/hypergraph \
  topobench/nn/backbones/__init__.py configs/model/hypergraph \
  test/nn/backbones/hypergraph test/nn/wrappers/hypergraph \
  test/pipeline/test_hypergraph_pipeline.py
git commit -m "feat: add native hypergraph models"
```

### Task 11: Reduce transforms to native graph and heterogeneous operations

**Files:**
- Modify: `topobench/transforms/__init__.py`
- Modify: `topobench/transforms/data_transform.py`
- Modify: `topobench/transforms/data_manipulations/__init__.py`
- Modify surviving files under: `topobench/transforms/data_manipulations/`
- Delete: `topobench/transforms/feature_liftings/`
- Delete: `topobench/transforms/liftings/`
- Delete: `configs/transforms/feature_liftings/`
- Delete: `configs/transforms/liftings/`
- Delete topological-only files under: `configs/transforms/data_manipulations/`
- Delete: `test/transforms/feature_liftings/`
- Delete: `test/transforms/liftings/`
- Delete tests for removed data manipulations under: `test/transforms/data_manipulations/`
- Create: `test/architecture/test_transform_registry.py`

**Step 1: Define the surviving transform allowlist in a failing test**

Keep these native operations, subject to their existing focused tests:

```text
CombinedEncodings, SelectDestinationEncodings,
CombinedFEs, SelectDestinationFEs, CombinedPSEs,
ConstantNodeFeatures,
ElectrostaticPE, HeterogeneousConstantFeatures, HeterogeneousToUndirected,
HKFE, HKdiagSE, IdentityTransform,
InfereKNNConnectivity, InfereRadiusConnectivity,
KeepOnlyConnectedComponent, KeepSelectedDataFields,
KeepSelectedTargetIndices, KHopFE, LapPE, NodeDegrees,
NodeFeaturesToFloat, OneHotDegreeFeatures, PPRFE, RWSE,
RenameFields, SheafConnLapPE
```

Treat this as the exact public allowlist after adding
`ConstantNodeFeatures`; helpers imported from PyG and module-private classes do
not become registry entries. Explicitly reject every lifting and: barycentric
subdivision, simplicial curvature, HOPSE preprocessing, SANN feature
generators, random Gaussian replacement features, rank feature
duplication/concatenation, and hypergraph incidence homophily transforms that
still expect `incidence_hyperedges`.

**Step 2: Verify failure**

Run: `uv run pytest test/architecture/test_transform_registry.py -q`

Expected: FAIL because discovery exposes topological transforms and liftings.

**Step 3: Build an explicit transform map**

Replace discovery with direct imports and a sorted `TRANSFORMS` dictionary. Remove `LIFTINGS` and `FEATURE_LIFTINGS` exports. Make `DataTransform` accept `Data` (including `HypergraphData`) and `HeteroData`; preserve the explicit `supports_heterodata` opt-in.

`AddGPSEInformation` is removed: its neighborhood-route input is the
rank-based HOPSE contract, not a native graph transform.

**Step 4: Delete transforms and configs outside the allowlist**

Use `git rm` for the two entire topological transform trees and their tests. Remove corresponding Hydra configs plus `combined_fe.yaml`, `combined_pe.yaml`, `sheaf_pe.yaml`, or `custom_example.yaml` only when their contained transform has been rejected. Preserve dataset/model defaults required by GPS/NSD and all three heterogeneous defaults.

Rewrite surviving dataset/model defaults to remove every
`get_required_lifting` interpolation and `liftings@_here_` entry. Verify the
REDDIT-BINARY constant-feature default, ZINC categorical encoding, IMDB degree
encoding, and GPS positional encodings still compose and produce the scalar
width expected by the graph encoder.

**Step 5: Run surviving transform tests and config composition**

Run:

```bash
uv run pytest test/architecture/test_transform_registry.py \
  test/transforms/data_manipulations -q
uv run pytest test/config/test_surviving_graph_configs.py \
  test/pipeline/test_heterogeneous_pipeline.py \
  test/pipeline/test_hypergraph_pipeline.py -q
```

Expected: PASS.

**Step 6: Commit**

```bash
git add -A topobench/transforms configs/transforms test/transforms \
  test/architecture/test_transform_registry.py
git commit -m "refactor: limit transforms to native graph data"
```

### Task 12: Prune unsupported neural and model surfaces

**Files:**
- Delete: `topobench/nn/backbones/cell/`
- Delete: `topobench/nn/backbones/combinatorial/`
- Delete: `topobench/nn/backbones/non_relational/`
- Delete: `topobench/nn/backbones/simplicial/`
- Delete: `topobench/nn/wrappers/cell/`
- Delete: `topobench/nn/wrappers/combinatorial/`
- Delete: `topobench/nn/wrappers/pointcloud/`
- Delete: `topobench/nn/wrappers/simplicial/`
- Delete: `topobench/nn/encoders/all_cell_encoder.py`
- Delete: `topobench/nn/encoders/flat_encoder.py`
- Delete: `topobench/nn/encoders/hopse_encoder.py`
- Delete: `topobench/nn/encoders/kdgm.py` if GCN-DGM failed its gate
- Delete: `topobench/nn/readouts/hopse.py`
- Delete: `topobench/nn/readouts/propagate_signal_down.py`
- Delete corresponding removed-model tests under: `test/nn/`
- Modify: all surviving `topobench/nn/**/__init__.py`
- Modify: `test/architecture/test_domain_contract.py`

**Step 1: Write the failing neural-surface architecture test**

Assert that backbone and wrapper domain directories are exactly `graph`,
`heterogeneous`, and `hypergraph`, and that surviving encoders/readouts import
no rank-based base class. Assert the registry equals the public classes proven
by Tasks 2, 6, and 10.

**Step 2: Verify failure**

Run: `uv run pytest test/architecture/test_domain_contract.py -q`

Expected: FAIL on unsupported neural directories and rank components.

**Step 3: Delete neural code and its tests**

Use explicit `git rm` targets above. Also remove GraphMLP or GCN-DGM
source/config-specific losses only if Task 6 recorded a failed capability
gate. Never delete a test for GCN/GAT/GIN/GPS/NSD, heterogeneous models,
EDGNN, HypergraphConv, shared model lifecycle, loss, or evaluator behavior.

**Step 4: Run surviving neural suites**

Run:

```bash
uv run pytest test/architecture/test_domain_contract.py \
  test/nn test/model test/loss test/evaluator -q
```

Then run the heterogeneous sentinel suite.

Expected: PASS.

**Step 5: Commit**

```bash
git add -A topobench/nn topobench/loss test/nn test/model test/loss \
  test/evaluator test/architecture/test_domain_contract.py
git commit -m "refactor: remove unsupported neural domains"
```

### Task 13: Prune unsupported data and legacy batching surfaces

**Files:**
- Delete: `topobench/data/loaders/pointcloud/`
- Delete: `topobench/data/loaders/simplicial/`
- Delete: `topobench/data/loaders/graph/mantra_dataset.py`
- Delete: `topobench/data/loaders/graph/manual_graph_dataset_loader.py`
- Delete: `topobench/data/loaders/graph/us_county_demos_dataset_loader.py`
- Delete: `topobench/data/datasets/mantra_dataset.py`
- Delete: `topobench/data/datasets/us_county_demos_dataset.py`
- Delete: US County-only parser and standardization tests
- Delete: `topobench/data/utils/utils.py`
- Delete: `topobench/data/utils/io_utils.py`
- Delete: `topobench/dataloader/dataload_dataset.py`
- Delete: `topobench/dataloader/dataloader.py`
- Delete: `topobench/dataloader/utils.py`
- Delete corresponding removed-data tests under: `test/data/`
- Modify: `topobench/data/loaders/__init__.py`
- Modify: `topobench/data/datasets/__init__.py`
- Modify: `topobench/data/utils/__init__.py`
- Modify: `topobench/dataloader/__init__.py`

**Step 1: Write a failing data-surface test**

Assert loaders expose only graph, heterogeneous, and hypergraph packages;
assert no import/export references `DataloadDataset`, `TBDataloader`,
`DomainData`, or `collate_fn`; and assert the dependency-free narrow utility
modules own every surviving import.

**Step 2: Verify failure**

Run: `uv run pytest test/architecture test/data -q`

Expected: FAIL on legacy data packages and batching symbols.

**Step 3: Delete the explicit data paths and clean exports**

Use only the explicit removal targets above. Delete the US County loader,
dataset, parser, train-only feature/target standardization path, and tests as
one unsupported node-regression surface. Confirm surviving graph downloads and
both hypergraph raw formats import from `downloads.py` or `hypergraph_io.py`
before removing mixed `io_utils.py`.

**Step 4: Run data and lifecycle sentinels**

Run:

```bash
uv run pytest test/architecture test/data test/pipeline/test_pipeline.py \
  test/pipeline/test_heterogeneous_pipeline.py \
  test/pipeline/test_hypergraph_pipeline.py -q
```

Expected: PASS without network access.

**Step 5: Commit**

```bash
git add -A topobench/data topobench/dataloader test/data test/architecture \
  test/pipeline
git commit -m "refactor: remove legacy data domains and batching"
```

### Task 14: Prune unsupported configuration and test surfaces

**Files:**
- Delete: `configs/dataset/pointcloud/`
- Delete: `configs/dataset/simplicial/`
- Delete: `configs/model/cell/`
- Delete: `configs/model/combinatorial/`
- Delete: `configs/model/non_relational/`
- Delete: `configs/model/pointcloud/`
- Delete: `configs/model/simplicial/`
- Delete: `configs/model/graph/hopse_gat.yaml`
- Delete: `configs/model/graph/hopse_gcn.yaml`
- Delete: `configs/model/graph/hopse_gin.yaml`
- Delete: `configs/model/hypergraph/alldeepset.yaml`
- Delete: `configs/model/hypergraph/allsettransformer.yaml`
- Delete: `configs/model/hypergraph/unignn.yaml`
- Delete: `configs/model/hypergraph/unignn2.yaml`
- Delete topological experiment configs under: `configs/experiment/`
- Delete corresponding pipeline/script tests under: `test/`
- Modify: `test/architecture/test_domain_contract.py`

**Step 1: Extend the architecture test before configuration deletion**

Assert exact allowed config directories, exact equality between dataset YAML
selectors and the surviving-dataset manifest, and absence of unsupported task
kinds. Assert that `graph/US-county-demos`,
`graph/graphuniverse_inductive`, `graph/ogbg-molpcba`, and
`graph/manual_dataset` are rejected and have no source-only product path. Walk
production Python plus surviving YAML
for forbidden tokens:

```python
FORBIDDEN_TOKENS = (
    "x_0", "batch_0", "incidence_hyperedges", "num_cell_dimensions",
    "topomodelx", "toponetx", "gudhi", "hypernetx",
)
```

Historical `docs/plans/` are the only exemption.

**Step 2: Verify failure**

Run: `uv run pytest test/architecture/test_domain_contract.py -q`

Expected: FAIL and list unsupported config/test references.

**Step 3: Delete unsupported configs and tests**

Delete tests only when the corresponding production behavior was intentionally
removed. Preserve shared trainer, callback, final-evaluation, graph,
heterogeneous, and hypergraph tests.

**Step 4: Compose every surviving selector and run architecture tests**

Run:

```bash
uv run pytest test/architecture test/config test/pipeline -q
```

Expected: PASS.

**Step 5: Commit**

```bash
git add -A configs test
git commit -m "refactor: remove unsupported configuration surfaces"
```

### Task 15: Remove topological dependencies and regenerate the lock

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Create: `test/dependencies/test_reduced_dependencies.py`
- Modify: `test/dependencies/test_torch_geometric_dependency.py`
- Create: `test/dependencies/test_neighbor_sampler_backend.py`
- Modify: `.github/workflows/test.yml`
- Modify: `.github/workflows/lint.yml`

**Step 1: Write failing dependency-boundary tests**

Parse `pyproject.toml` with `tomllib` and require that direct dependencies exclude:

```text
topomodelx, toponetx, gudhi, hypernetx, trimesh, spharapy
```

Add a clean subprocess test that installs a meta-path finder blocking those module names and imports:

```text
topobench
topobench.data
topobench.data.loaders
topobench.transforms
topobench.nn.backbones
topobench.nn.wrappers
topobench.run
```

Add a separate runtime probe that constructs the packaged synthetic
`HeteroData`, instantiates PyG `NeighborLoader`, and consumes one mini-batch.
The probe must assert that the target node store exposes both `batch_size` and
`n_id`, and that the seed batch has more than one node. This is a dependency
test, not merely a heterogeneous-pipeline unit test: it proves a supported
sampling backend is actually installed.

**Step 2: Verify failure**

Run: `uv run pytest test/dependencies -q`

Expected: FAIL because forbidden direct dependencies remain.

**Step 3: Audit every remaining dependency before editing**

For each candidate run `rg -n` over `topobench`, scripts, and current tests.
Remove the six forbidden dependencies unconditionally after source pruning.
Also remove `networkx`, `matplotlib`, `decorator`, `yacs`, `einops`,
`tabulate`, `pandas`, or `torch-cluster` only if no surviving production path
uses them. Keep `torch-scatter` when NSD or EDGNN still imports it.

Keep `torch-sparse` as an explicit runtime dependency even if static source
search finds no direct import: PyG `NeighborLoader` delegates sampling to an
optional compiled backend. Removing it is allowed only in a later, separately
scoped migration that adds `pyg-lib`, repeats the clean-sync probe, and proves
full behavior parity. Do not treat the absence of a direct Python import as
evidence that the backend is unused.

Move Jupyter applications (`ipykernel`, `notebook`, `jupyterlab`) out of core dependencies into `doc` or `dev`. Do not broaden dependency cleanup into unrelated version upgrades.

**Step 4: Regenerate the lock**

Run: `uv lock`

Expected: exit 0; the lock graph contains no forbidden packages, including transitive copies formerly pulled by TopoModelX.

Run: `rg -n "topomodelx|toponetx|gudhi|hypernetx" uv.lock pyproject.toml`

Expected: no output.

**Step 5: Sync and execute clean import tests**

Run:

```bash
uv sync --extra test
uv run pytest test/dependencies test/architecture \
  test/pipeline/test_heterogeneous_pipeline.py -q
```

Expected: PASS, including consumption of a real neighbor-sampled mini-batch
after the fresh sync.

**Step 6: Update CI caches/commands only as required**

Keep CI using the regenerated lock and the supported Python 3.11 environment. Remove install steps for deleted topology packages; do not weaken the existing test invocation.

**Step 7: Commit**

```bash
git add pyproject.toml uv.lock test/dependencies .github/workflows
git commit -m "build: remove topological dependencies"
```

### Task 16: Qualify the surviving product and set coherent CLI defaults

**Files:**
- Modify: `configs/run.yaml`
- Modify: `configs/experiment/example.yaml`
- Preserve and verify: `configs/experiment/heterogeneous_*.yaml`
- Create: `configs/experiment/graph_synthetic_regression.yaml`
- Create: `configs/experiment/hypergraph_synthetic_edgnn.yaml`
- Create: `configs/experiment/hypergraph_synthetic_hypergraph_conv.yaml`
- Modify: `topobench/run.py` only if domain validation is not already centralized
- Create: `test/config/test_all_surviving_configs.py`
- Create: `test/integration/test_retained_datasets.py`
- Modify: `test/pipeline/test_heterogeneous_pipeline.py`
- Modify: `test/pipeline/test_hypergraph_pipeline.py`
- Modify: `test/callbacks/test_best_epoch_metrics.py`

**Step 1: Write a failing full product-manifest test**

Discover YAML selectors from only:

```text
configs/dataset/{graph,heterogeneous,hypergraph}
configs/model/{graph,heterogeneous,hypergraph}
configs/experiment
```

Require exact equality between dataset YAML selectors and the immutable
surviving-dataset manifest. Every dataset must declare a supported task kind,
feature and edge policies, split mode, at least one compatible model, and named
qualification evidence. Compose every declared valid pairing, resolve
interpolation, and instantiate components that do not require downloads.
Assert every experiment refers only to existing selectors. Assert undeclared
same-domain pairs, unsupported task kinds, and every cross-domain pair fail
with the path-rich resolver error from Task 6.

**Step 2: Write selector qualification tests**

For every retained graph, heterogeneous, and hypergraph selector, add a
qualification row naming:

```text
selector, loader family, fixture/download gate, expected task,
expected split mode, feature policy, edge policy, compatible model
```

Each row must load fresh data, apply production preprocessing and splitting,
obtain one real loader batch, run one compatible model forward/loss/metric
update, and assert finite results and exact target shapes. A row may cite an
existing heterogeneous or real-format hypergraph integration test, but the
manifest test must resolve that test name and the cited test must make a
selector-specific metadata/parser assertion. Mark network-dependent rows
`download` and `integration`; these may be excluded from ordinary CI but are
mandatory before release.

**Step 3: Verify failure**

Run:

```bash
uv run pytest test/config/test_all_surviving_configs.py \
  test/integration/test_retained_datasets.py \
  -m "not download and not integration" -q
```

Expected: FAIL until the manifest, configs, and network-free qualification rows
agree exactly.

**Step 4: Set network-free graph defaults**

Use packaged `graph/SyntheticGraph` plus `graph/gcn` as the default. Add
`graph_synthetic_regression.yaml` using `graph/SyntheticGraphRegression` plus
`graph/gcn`. Do not make default execution download data or retain the legacy
manual loader. Set `data_pipeline: default`, `transforms: no_transform`, and
retain `train: true`, `test: true`; best-checkpoint validation/test reruns
remain mandatory.

**Step 5: Rewrite experiments**

- Make `configs/experiment/example.yaml` a native graph classification example.
- Preserve all eight heterogeneous experiments and their meaningful names.
- Add the scalar-regression synthetic experiment.
- Add small hypergraph EDGNN and HypergraphConv experiments using
  `hypergraph/SyntheticHypergraph`; both explicitly override
  `/data_pipeline: hypergraph_node`.
- Remove cell-HGT, HOPSE, SANN, TopoTune, simplicial, and combinatorial experiments.

**Step 6: Verify final evaluation behavior in every supported task family**

Add callback/run tests for graph classification, graph scalar regression,
heterogeneous node classification, and hypergraph node classification. Each
tiny run reruns the validation-selected checkpoint once per split, publishes
`evaluations/best_checkpoint/{val,test}/metrics.json` with exact
`num_examples`, writes complete prediction manifests plus bounded pickle-free
shards, and registers every file with split-qualified logger names. Mock W&B;
do not contact the service.

**Step 7: Run network-free product suites**

Run:

```bash
uv run pytest test/config test/pipeline \
  test/integration/test_retained_datasets.py \
  test/callbacks/test_best_epoch_metrics.py \
  -m "not download and not integration" -q
```

Expected: PASS, with every network-free manifest entry qualified.

**Step 8: Commit**

```bash
git add -A configs topobench/run.py topobench/data/capabilities.py \
  test/config test/pipeline test/integration/test_retained_datasets.py \
  test/callbacks/test_best_epoch_metrics.py
git commit -m "test: qualify the reduced product surface"
```

### Task 17: Rewrite current documentation and remove obsolete examples

**Files:**
- Modify: `README.md`
- Modify: `docs/index.rst`
- Modify: `docs/heterogeneous_graphs.md`
- Create: `docs/graph_data.md`
- Create: `docs/hypergraphs.md`
- Modify: `docs/conf.py`
- Regenerate or prune: `docs/api/`
- Delete obsolete notebooks under: `tutorials/`
- Delete obsolete scripts under: `scripts/hgt/`, `scripts/hopse/`, `scripts/topotune/`
- Modify or remove: `scripts/topobench/reproduce.sh`
- Delete obsolete tests: `test/scripts/test_zinc_hgt_search.py`, `test/test_tutorials.py`
- Modify: `test/docs/test_heterogeneous_examples.py`
- Create: `test/docs/test_current_examples.py`

**Step 1: Write failing executable-documentation tests**

Extract documented Hydra commands and assert their dataset/model/experiment selectors exist and compose. Add text assertions that current docs do not advertise cell, simplicial, combinatorial, lifting, TopoModelX, or TopoNetX support. Historical plan documents are exempt.

**Step 2: Verify failure**

Run: `uv run pytest test/docs -q`

Expected: FAIL on obsolete README/tutorial/API references.

**Step 3: Rewrite the product documentation**

README structure:

1. scope: graph, heterogeneous graph, hypergraph;
2. installation with uv;
3. smallest homogeneous command;
4. generated, explicit-index, and separate-phase graph inputs;
5. heterogeneous full and neighbor-sampled commands;
6. hypergraph EDGNN and HypergraphConv commands;
7. batching and supervision semantics by domain;
8. best-checkpoint rerun and W&B metric names;
9. adding a dataset/model using the explicit registries.

Keep `docs/heterogeneous_graphs.md` detailed and update only links/scope.
`docs/hypergraphs.md` must document the exact incidence convention and cache
incompatibility. `docs/graph_data.md` must document native `x`/`batch` fields,
the three graph split input types, explicit split proportions, and the
confidential-adapter rule that no real or derived fixture is committed.

**Step 4: Prune generated API pages and obsolete artifacts**

Remove API pages for deleted modules and regenerate the index if the repository tooling is deterministic. Remove topology-specific tutorials and generated tutorial checkpoints/logs. Preserve licensing, attribution, contribution policy, and the approved historical plan documents.

**Step 5: Run documentation and link/config tests**

Run:

```bash
uv run pytest test/docs test/config -q
uv run python docs/generate_api_index.py
git diff --check
```

Expected: tests PASS; generator exits 0; no whitespace errors.

**Step 6: Commit**

```bash
git add -A README.md docs tutorials scripts test/docs \
  test/scripts/test_zinc_hgt_search.py test/test_tutorials.py
git commit -m "docs: document native graph benchmark core"
```

### Task 18: Final clean-environment and lifecycle verification

**Files:**
- Modify only if a genuine defect is found: relevant surviving source/test file
- Create: `test/architecture/verify_forbidden_imports.py`

**Step 1: Run static residue checks**

Run:

```bash
rg -n "x_0|x_1|batch_0|incidence_hyperedges|num_cell_dimensions|liftings@_here_|get_required_lifting" \
  topobench configs test scripts docs \
  -g '!docs/plans/**'
rg -n "topomodelx|toponetx|gudhi|hypernetx|trimesh|spharapy" \
  topobench configs test scripts docs pyproject.toml uv.lock \
  -g '!docs/plans/**'
```

Expected: no output. If any match is an intentional error message in an architecture test, keep its forbidden token list in one allowlisted test constant only and make the shell check exclude that file.

**Step 2: Run formatting and lint**

Run:

```bash
uv run ruff format --check topobench
uv run ruff check topobench
git diff --check
```

Expected: all exit 0.

**Step 3: Run the complete network-free suite**

Run:

```bash
uv run pytest -m "not download and not integration" -q
```

Expected: PASS with only documented optional-extension skips.

Also run the contract-focused suites without marker exclusions:

```bash
uv run pytest test/architecture test/dependencies test/config \
  test/data/test_graph_feature_contract.py \
  test/config/test_dataset_feature_policies.py \
  test/data/test_hypergraph_data.py \
  test/model/test_supervision.py -q
```

Expected: PASS with no skipped dependency, feature-policy, mask, supervision,
or batching contract test.

**Step 4: Run mandatory real-dataset qualification**

With network access enabled, run:

```bash
uv run pytest test/integration/test_retained_datasets.py \
  -m "download and integration" -q

uv run pytest test/integration/test_real_hypergraph_formats.py \
  -m "download and integration" -q

TOPOBENCH_ALLOW_DOWNLOADS=1 uv run pytest \
  test/integration/test_real_parquet_graph.py \
  test/integration/test_real_parquet_heterogeneous.py -q

uv run python test/integration/qualify_typed_graph_rss.py
uv run python test/integration/qualify_typed_graph_cuda.py

```

Expected: every download-marked surviving selector, both selected hypergraph
raw formats, representative real homogeneous and heterogeneous Parquet graphs,
and mandatory bounded-RSS/CUDA-overlap gates PASS against fresh raw/processed
directories. This includes graph classification and scalar-regression
lifecycles with exact `[B, 1]` targets, strictly out-of-core typed conversion,
bounded selected reads, exact directed cluster unions, exact relation fanout,
all target-supervision phases, at most 5% qualified input stall, exact
external-ID restoration, and selector-specific real parser assertions. A
network outage may postpone the release step, but a skip or postponed result is
not a passing release.

**Step 5: Run seven end-to-end smoke tests**

Run graph classification, graph scalar regression, phase-separated inductive
node classification, disk-streamed homogeneous transductive node
classification, disk-streamed heterogeneous neighbor classification,
in-memory heterogeneous neighbor batching, and synthetic hypergraph
classification on CPU with W&B disabled:
```bash
WANDB_MODE=disabled uv run python -m topobench.run \
  experiment=example trainer.accelerator=cpu trainer.devices=1 trainer.max_epochs=1

WANDB_MODE=disabled uv run python -m topobench.run \
  experiment=graph_synthetic_regression \
  trainer.accelerator=cpu trainer.devices=1 trainer.max_epochs=1

WANDB_MODE=disabled uv run python -m topobench.run \
  experiment=graph_synthetic_inductive_node \
  trainer.accelerator=cpu trainer.devices=1 trainer.max_epochs=1

WANDB_MODE=disabled uv run python -m topobench.run \
  experiment=graph_synthetic_disk_gcn \
  trainer.accelerator=cpu trainer.devices=1 trainer.max_epochs=1

WANDB_MODE=disabled uv run python -m topobench.run \
  experiment=heterogeneous_synthetic_disk_hgt_neighbor \
  trainer.accelerator=cpu trainer.devices=1 trainer.max_epochs=1

WANDB_MODE=disabled uv run python -m topobench.run \
  experiment=heterogeneous_synthetic_hgt_neighbor \
  trainer.accelerator=cpu trainer.devices=1 trainer.max_epochs=1

WANDB_MODE=disabled uv run python -m topobench.run \
  experiment=hypergraph_synthetic_hypergraph_conv \
  trainer.accelerator=cpu trainer.devices=1 trainer.max_epochs=1
```

Expected for each: training completes, a checkpoint is selected solely by
validation, and both validation and test selected-checkpoint metrics plus
distinct `evaluations/best_checkpoint/{val,test}` prediction manifests/shards
are produced. Every shard round-trips with `allow_pickle=False`; the two
manifests record the same checkpoint SHA-256 and separate logger names. The
regression run reports exact `[B, 1]` outputs/targets without broadcasting. The
inductive-node run consumes distinct phase graphs without masks. Both
heterogeneous neighbor runs export each target seed once and no context node.

**Step 6: Verify clean forbidden imports in a subprocess**

`test/architecture/verify_forbidden_imports.py` installs a meta-path finder that raises for the four removed module roots, imports every public TopoBench package, composes the default plus one config per domain, and exits zero. Run:

```bash
uv run python test/architecture/verify_forbidden_imports.py
```

Expected: exit 0 with a concise `clean import verified` message.

**Step 7: Inspect repository state and commit final fixes**

Run:

```bash
git status --short
git diff --stat HEAD~1
git log --oneline --decorate -15
```

If verification required code fixes, commit only those verified fixes:

```bash
git add <exact-fixed-paths>
git commit -m "test: complete reduced core verification"
```

If the tree is already clean, do not create an empty commit.

## Completion criteria

Implementation is complete only when all of the following are true:

- homogeneous inductive training uses real PyG mini-batches and native
  `x`/`batch`;
- supported homogeneous tasks are graph binary/multiclass classification,
  graph scalar regression, singleton transductive node classification, and
  phase-separated inductive node classification; unsupported selectors are
  absent;
- graph loaders use the explicit unsplit, indexed, or phase-dataset input
  contract;
- generated splits are ordinary three-way splits with explicit proportions
  and local RNG state;
- supplied node/example indices are validated as complete disjoint
  partitions;
- generated and fixed multi-graph splits remain lazy index-backed views, while
  phase datasets may have independent backing storage;
- explicit transductive mode requires one source graph and disjoint target
  phases; `full_graph` requires graph `batch_size == 1`,
  homogeneous `cluster_disk` uses partition descriptors, and heterogeneous
  `neighbor_disk` uses target-seed descriptors with relation fanout;
- one versioned non-executable universal typed store represents homogeneous
  graphs as one node type/relation and heterogeneous graphs with exact typed
  maps and relation triples;
- chunked typed Parquet conversion is schema-mapped and strictly out of core:
  no complete graph, feature matrix, mapped edge table, `Data`, or
  `HeteroData` is materialized;
- per-type external IDs/features/supervision and per-relation directed CSC/edge
  fields round-trip exactly; the same external ID may occur in two node types;
- homogeneous disk batches perform bounded selected reads, reconstruct exact
  induced directed cluster unions, emit writable `Data`, and carry canonical
  `global_nid` plus exact external-ID restoration;
- heterogeneous disk batches keep relation CSC memory-mapped, use exact
  relation-specific fanout, emit writable `HeteroData`, and supervise/export
  only target `n_id[:batch_size]`;
- qualified `batch_transform` accepts and returns native homogeneous `Data`,
  runs exactly once before pinning/device transfer, and preserves
  node/supervision identity;
- bounded generic host prefetch and a configurable CUDA ring defaulting to
  three batches ahead preserve ordered delivery and committed-cursor resume
  for both disk strategies;
- continuous asynchronous input telemetry remains outside scientific metrics,
  packaged runs warn on starvation, and mandatory CUDA qualification proves at
  most 5% steady-state input stall for both native output views;
- scalar-regression logits and targets are exactly `[B, 1]` through
  supervision, loss, and metrics, including a smaller final batch;
- every retained graph dataset is present in the exact manifest, has explicit
  node/edge feature policies and at least one compatible model, and passes its
  named network-free or mandatory download-marked qualification evidence;
- every model declares tested `consume`, `ignore`, or `reject` behavior for
  `edge_attr` and `edge_weight`; wrappers never silently forward or drop them;
- heterogeneous in-memory full/neighbor modes and disk neighbor mode retain
  target-seed lifecycle behavior, and real `NeighborLoader` mini-batches are
  consumed after a clean dependency sync without a skip;
- hypergraphs batch correctly with independent node/hyperedge offsets, reject
  unsupported empty hyperedges, and run both EDGNN and HypergraphConv;
- native hypergraph caches are versioned and cannot silently reuse legacy
  rank-based `data.pt` artifacts;
- the validation-selected checkpoint is the sole checkpoint used for one
  validation rerun and one test rerun; each publishes independent metrics,
  integer `num_examples`, and bounded versioned per-sample prediction shards
  with exact identity, target, raw output, exported/normalized values, and
  allowlisted metadata;
- graph-level sample IDs, homogeneous/hypergraph global node IDs, and
  heterogeneous typed target-seed IDs remain stable, unique, complete, and
  externally resolvable at export;
- every result, phase logger, returned output, final metrics JSON, prediction
  manifest, and provenance record agrees on the metric-participant count;
- every configured logger receives separate validation/test metrics and one
  split-qualified artifact record/upload per local artifact file; final
  directories promote atomically and never silently overwrite;
- evaluator input/output and lifecycle use typed TopoBench contracts while
  TorchMetrics remains an internal directly declared implementation detail;
- training metrics are bounded online by default, validation/test metrics are
  exact by default, binary AUPRC uses average precision, Somers' D derives from
  AUROC, audit mode compares exact and online ranking policies, all exact
  binary/multiclass state is TopoBench-owned CPU storage under one
  retained-plus-compute guard, and resource or undefined-value failures never
  downgrade silently;
- mid-epoch checkpoints align evaluator sequence/count state, sampler committed
  cursor, and model global step so pending gradient-accumulation batches cannot
  be counted twice after resume;
- every normal training run passes automatic static/data/execution preflight
  before production side effects, and the probe leaves production
  model/optimizer/RNG/sampler/logger/checkpoint/artifact state pristine;
- every mandatory real selector, real homogeneous/heterogeneous Parquet,
  bounded-RSS, CUDA-overlap, and both real hypergraph-format gates pass;
- only graph, heterogeneous, and hypergraph source/config groups remain;
- no surviving runtime uses a rank-indexed field or a lifting;
- TopoModelX, TopoNetX, GUDHI, HyperNetX, trimesh, and spharapy are absent from
  source, direct dependencies, and lockfile, while the qualified heterogeneous
  sampler backend remains installed;
- epoch losses are weighted by supervised examples for inductive,
  transductive, and sampled heterogeneous tasks;
- the default CLI is network-free and executes final selected-checkpoint
  validation/test evaluation and artifact publication;
- a final post-qualification agent review leaves sparse rationale comments only
  at nested or non-obvious reliability boundaries and passes focused/style
  verification without executable changes;
- the complete network-free suite, Ruff, clean-import probe, all seven
  end-to-end smokes, bounded-RSS typed Parquet conversion, mandatory
  homogeneous/heterogeneous CUDA overlap, and prediction-artifact
  qualification pass.
