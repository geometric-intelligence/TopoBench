# Native Heterogeneous Graph Support Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add first-class, batched heterogeneous node-classification support to TopoBench with native PyG `HeteroData`, reusable HGT and HeteroSAGE models, deterministic synthetic tests, and staged promotion to DBLP and OGB-MAG.

**Architecture:** Keep TopoBench's shared experiment, preprocessing, model, evaluator, callback, logging, and best-checkpoint evaluation shell. Introduce a configurable data-pipeline boundary: the existing default pipeline remains behaviorally unchanged, while a heterogeneous-node pipeline preserves native `HeteroData`, validates a typed data specification, and selects either PyG `DataLoader` for full-graph training or `NeighborLoader` for sampled mini-batches. Models consume metadata-driven dictionaries through heterogeneous-specific encoder, wrapper, backbone, and readout components; `TBModel` delegates selection of supervised predictions and targets to a small adapter rather than branching throughout the training loop.

**Tech Stack:** Python 3.11, PyTorch 2.3, PyTorch Geometric, Lightning 2.4, Hydra 1.3, pytest, Ruff, W&B.

---

## Execution Rules

- Execute in `/Users/leone/dev/topobench` on the current main checkout. Do not create a worktree.
- Use `@test-driven-development` for every implementation task below: add one failing test, run it and confirm the intended failure, implement only enough to pass, then rerun the focused test.
- Preserve all existing homogeneous, topological, and cell-complex behavior. The heterogeneous path must never send `HeteroData` through `DomainData`, `DataloadDataset`, `collate_fn`, or `TBDataloader`.
- Use the existing transform registry and `PreProcessor`; extend their contracts instead of creating a duplicate preprocessing framework.
- Treat the official split masks in each target node store as authoritative. Do not regenerate DBLP or OGB-MAG splits.
- Run `@verification-before-completion` before claiming a stage or the whole feature is complete.
- Commit after every task. The commit commands are part of the plan, but inspect `git diff` before each commit and do not include unrelated user changes.

## Scope and Promotion Gates

The first release supports one native heterogeneous graph and node classification on one configured `target_node_type`.

| Gate | Dataset / mode | Required evidence before promotion |
|---|---|---|
| G0 | Existing suite | Baseline focused tests pass before changes |
| G1 | Synthetic / data only | Deterministic schema, transforms, serialization, spec validation |
| G2 | Synthetic / full graph | HGT and HeteroSAGE train, validate, and rerun the best checkpoint on test |
| G3 | Synthetic / neighbor sampled | Both models use true seed-node mini-batches and complete the same lifecycle |
| G4 | DBLP / full graph | Official masks, typed features, both models, reproducible smoke run |
| G5 | OGB-MAG / sampled | Configuration and preflight pass; a manual one-epoch sampled run fits memory |

Do not claim large-graph performance or compare accuracy until G5 completes and the evaluation protocol has been recorded. Exact full-graph and sampled inference are different protocols and must be named separately in experiment metadata.

## Task 1: Freeze the Existing Behavior and Declare PyG Directly

**Files:**

- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Test: `test/dependencies/test_torch_geometric_dependency.py`

**Step 1: Record the baseline**

Run:

```bash
uv run pytest \
  test/data/preprocess/test_preprocessor.py \
  test/data/dataload/test_Dataloaders.py \
  test/model/test_model.py \
  test/nn/backbones/combinatorial/test_hgt.py \
  test/pipeline/test_hgt_pipeline.py -q
```

Expected: all selected tests pass. Save the test count in the implementation log or commit message notes.

**Step 2: Add a failing dependency declaration test**

Create `test/dependencies/test_torch_geometric_dependency.py`:

```python
from pathlib import Path

import tomllib


def test_torch_geometric_is_a_direct_runtime_dependency() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())
    dependencies = pyproject["project"]["dependencies"]

    assert any(
        dependency.startswith("torch-geometric") for dependency in dependencies
    )
```

**Step 3: Run the new test and confirm the failure**

Run:

```bash
uv run pytest test/dependencies/test_torch_geometric_dependency.py -q
```

Expected: FAIL because PyG currently arrives only transitively.

**Step 4: Add a compatible direct dependency**

Add `"torch-geometric>=2.5,<3"` to `dependencies` in `pyproject.toml`. This lower bound covers the APIs used in the approved design; the lock file determines the tested concrete version.

Regenerate the lock:

```bash
uv lock
```

Do not upgrade unrelated dependencies intentionally. Inspect the lock diff and stop if it contains a broad, unexplained resolution change.

**Step 5: Verify**

Run:

```bash
uv run pytest test/dependencies/test_torch_geometric_dependency.py -q
uv run python -c "import torch_geometric; print(torch_geometric.__version__)"
```

Expected: the test passes and the installed PyG version prints successfully.

**Step 6: Commit**

```bash
git add pyproject.toml uv.lock test/dependencies/test_torch_geometric_dependency.py
git commit -m "build: declare torch geometric dependency"
```

## Task 2: Create a Deterministic Native Heterogeneous Fixture

**Files:**

- Create: `topobench/data/datasets/synthetic_heterogeneous_dataset.py`
- Modify: `topobench/data/datasets/__init__.py`
- Test: `test/data/datasets/test_synthetic_heterogeneous_dataset.py`

**Step 1: Write the schema contract test**

Create a test that builds the dataset twice and checks:

```python
import torch
from torch_geometric.data import HeteroData

from topobench.data.datasets import make_synthetic_heterogeneous_data


def test_synthetic_heterogeneous_schema_is_native_and_deterministic() -> None:
    first = make_synthetic_heterogeneous_data(seed=7)
    second = make_synthetic_heterogeneous_data(seed=7)

    assert isinstance(first, HeteroData)
    assert first.node_types == ["author", "paper", "venue"]
    assert set(first.edge_types) == {
        ("author", "writes", "paper"),
        ("paper", "published_in", "venue"),
    }
    assert first["author"].x.shape[1] != first["paper"].x.shape[1]
    assert "x" not in first["venue"]
    assert torch.equal(first["author"].x, second["author"].x)
    assert torch.equal(
        first["author", "writes", "paper"].edge_index,
        second["author", "writes", "paper"].edge_index,
    )
```

Add a second test asserting that:

- `author` is the only labeled node type;
- `train_mask`, `val_mask`, and `test_mask` are boolean, non-empty, and disjoint;
- every author belongs to exactly one split;
- all classes appear in the training mask;
- the features contain a learnable class signal.

**Step 2: Run and confirm import failure**

Run:

```bash
uv run pytest test/data/datasets/test_synthetic_heterogeneous_dataset.py -q
```

Expected: FAIL because the factory does not exist.

**Step 3: Implement the smallest deterministic factory**

Implement:

```python
def make_synthetic_heterogeneous_data(
    *,
    seed: int = 0,
    num_authors: int = 36,
    num_papers: int = 24,
    num_venues: int = 6,
) -> HeteroData:
    ...
```

Use a local `torch.Generator().manual_seed(seed)`; never mutate global RNG state. Give:

- `author`: 8-dimensional features, two balanced classes, `y`, and official-style masks;
- `paper`: 5-dimensional features;
- `venue`: `num_nodes` only, deliberately featureless;
- forward-only `writes` and `published_in` relations.

Construct edges so every node participates, indices stay in bounds, and author labels have a simple signal in `author.x[:, :2]`. This fixture is a correctness and overfitting probe, not a benchmark.

**Step 4: Verify**

Run:

```bash
uv run pytest test/data/datasets/test_synthetic_heterogeneous_dataset.py -q
```

Expected: PASS.

**Step 5: Commit**

```bash
git add topobench/data/datasets test/data/datasets/test_synthetic_heterogeneous_dataset.py
git commit -m "test: add deterministic heterogeneous graph fixture"
```

## Task 3: Load the Synthetic Graph Through TopoBench

**Files:**

- Create: `topobench/data/loaders/heterogeneous/__init__.py`
- Create: `topobench/data/loaders/heterogeneous/synthetic.py`
- Modify: `topobench/data/loaders/__init__.py`
- Create: `configs/dataset/heterogeneous/SyntheticHeterogeneous.yaml`
- Test: `test/data/load/test_heterogeneous_dataset_loaders.py`

**Step 1: Add a failing loader test**

```python
from omegaconf import OmegaConf
from torch_geometric.data import HeteroData

from topobench.data.loaders import SyntheticHeterogeneousDatasetLoader


def test_synthetic_loader_returns_one_native_heterogeneous_graph(tmp_path) -> None:
    parameters = OmegaConf.create(
        {
            "data_dir": str(tmp_path),
            "data_name": "SyntheticHeterogeneous",
            "seed": 11,
        }
    )

    dataset, data_dir = SyntheticHeterogeneousDatasetLoader(parameters).load()

    assert len(dataset) == 1
    assert isinstance(dataset[0], HeteroData)
    assert data_dir.endswith("SyntheticHeterogeneous")
```

**Step 2: Run and confirm failure**

Run:

```bash
uv run pytest test/data/load/test_heterogeneous_dataset_loaders.py -q
```

Expected: FAIL because the loader is not exported.

**Step 3: Implement the loader**

Subclass `AbstractLoader`, return a one-element PyG-compatible in-memory dataset, and expose only fixture-generation arguments from configuration. Keep the loader free of model or batching decisions.

The Hydra dataset config must declare:

```yaml
loader:
  _target_: topobench.data.loaders.SyntheticHeterogeneousDatasetLoader
  parameters:
    data_domain: heterogeneous
    data_type: synthetic
    data_name: SyntheticHeterogeneous
    data_dir: ${paths.data_dir}/${dataset.loader.parameters.data_domain}/${dataset.loader.parameters.data_type}
    seed: ${seed}

parameters:
  task: classification
  task_level: node
  num_classes: 2
  target_node_type: author

split_params:
  learning_setting: transductive
  source: official_masks

dataloader_params:
  mode: full_batch
  num_workers: 0
  pin_memory: false
```

Do not add fake top-level `num_features`; heterogeneous input widths belong to the data stores.

**Step 4: Verify loader and config composition**

Run:

```bash
uv run pytest test/data/load/test_heterogeneous_dataset_loaders.py -q
uv run python -m topobench.run --cfg job \
  dataset=heterogeneous/SyntheticHeterogeneous \
  model=cell/hgt \
  train=false test=false
```

Expected: the loader unit test passes. Config composition prints successfully; execution is not expected to support the heterogeneous pipeline yet.

**Step 5: Commit**

```bash
git add \
  topobench/data/loaders/heterogeneous \
  topobench/data/loaders/__init__.py \
  configs/dataset/heterogeneous/SyntheticHeterogeneous.yaml \
  test/data/load/test_heterogeneous_dataset_loaders.py
git commit -m "feat: add synthetic heterogeneous dataset loader"
```

## Task 4: Preserve `HeteroData` in the Existing PreProcessor

**Files:**

- Modify: `topobench/data/preprocessor/preprocessor.py`
- Modify: `topobench/data/preprocessor/__init__.py`
- Test: `test/data/preprocess/test_preprocessor.py`

**Step 1: Add round-trip tests before changing production code**

Add tests that pass a one-graph dataset containing `HeteroData` to `PreProcessor` and assert:

```python
def test_preprocessor_preserves_heterodata_on_process_and_reload(
    synthetic_heterogeneous_dataset,
    tmp_path,
) -> None:
    first = PreProcessor(
        synthetic_heterogeneous_dataset,
        tmp_path,
        transform_config=None,
    )
    reloaded = PreProcessor(
        synthetic_heterogeneous_dataset,
        tmp_path,
        transform_config=None,
    )

    assert isinstance(first[0], HeteroData)
    assert isinstance(reloaded[0], HeteroData)
    assert first[0].metadata() == reloaded[0].metadata()
    assert torch.equal(
        first[0]["author"].train_mask,
        reloaded[0]["author"].train_mask,
    )
```

Retain an explicit homogeneous regression test asserting processed `Data` remains `Data`.

**Step 2: Run and confirm the intended assertion failure**

Run:

```bash
uv run pytest test/data/preprocess/test_preprocessor.py \
  -k "heterodata or preserves_data" -q
```

Expected: the heterogeneous case fails at the current homogeneous-only assertion.

**Step 3: Generalize the preprocessor contract**

Introduce a local type alias such as:

```python
SupportedData = torch_geometric.data.Data | torch_geometric.data.HeteroData
```

Update `process()` and annotations to:

- accept either native type;
- preserve the input representation exactly;
- apply the configured transform to each example;
- reject a transform result that is neither `Data` nor `HeteroData`;
- reject a representation change (`Data` to `HeteroData` or the reverse) unless a future transform explicitly declares such a conversion;
- use PyG's supported save/collation mechanism so reload retains stores, metadata, masks, and edge indices.

Do not add splitting behavior here yet.

**Step 4: Verify focused and existing preprocessing tests**

Run:

```bash
uv run pytest test/data/preprocess/test_preprocessor.py -q
```

Expected: all preprocessing tests pass, including homogeneous regressions.

**Step 5: Commit**

```bash
git add topobench/data/preprocessor test/data/preprocess/test_preprocessor.py
git commit -m "feat: preserve native heterodata in preprocessing"
```

## Task 5: Reuse the Transform Pipeline With Explicit Compatibility

**Files:**

- Modify: `topobench/transforms/data_transform.py`
- Create: `topobench/transforms/data_manipulations/heterogeneous.py`
- Modify: `topobench/transforms/data_manipulations/__init__.py`
- Create: `configs/transforms/data_manipulations/heterogeneous_constant_features.yaml`
- Create: `configs/transforms/data_manipulations/heterogeneous_to_undirected.yaml`
- Create: `configs/transforms/dataset_defaults/SyntheticHeterogeneous.yaml`
- Test: `test/transforms/test_heterogeneous_transforms.py`

**Step 1: Specify compatibility and transform behavior in failing tests**

Cover four cases:

```python
def test_heterogeneous_constant_features_fills_only_selected_store(): ...
def test_to_undirected_adds_reverse_typed_relations_without_merging(): ...
def test_heterogeneous_transforms_compose_in_declared_order(): ...
def test_data_transform_rejects_unmarked_transform_for_heterodata(): ...
```

The transformed synthetic graph must have:

- unchanged `author.x` and `paper.x`;
- a one-channel constant `venue.x` with the configured value;
- reverse edge types generated by `ToUndirected(merge=False)`;
- unchanged labels and masks.

**Step 2: Run and confirm failure**

Run:

```bash
uv run pytest test/transforms/test_heterogeneous_transforms.py -q
```

Expected: FAIL because compatibility markers and wrappers do not exist.

**Step 3: Add a minimal capability contract**

Add `supports_heterodata: bool = False` to the transform base/registry contract used by `DataTransform`. Before applying a transform to `HeteroData`, raise a readable `TypeError` when the selected transform does not opt in. Include the transform class name and node/edge types in the error.

Create thin registered wrappers around:

```python
torch_geometric.transforms.Constant(node_types=[...])
torch_geometric.transforms.ToUndirected(merge=False)
```

Mark only these wrappers as heterogeneous-compatible. Do not blanket-mark existing topological liftings.

**Step 4: Configure the synthetic default transform**

Compose:

1. constant features for `venue`;
2. `ToUndirected(merge=False)`.

The final graph must contain all required `x_dict` entries before model construction. Directionality is configuration, not an implicit model side effect.

**Step 5: Verify transform and preprocessor integration**

Run:

```bash
uv run pytest \
  test/transforms/test_heterogeneous_transforms.py \
  test/data/preprocess/test_preprocessor.py -q
```

Expected: PASS.

**Step 6: Commit**

```bash
git add \
  topobench/transforms \
  configs/transforms/data_manipulations \
  configs/transforms/dataset_defaults/SyntheticHeterogeneous.yaml \
  test/transforms/test_heterogeneous_transforms.py
git commit -m "feat: support compatible heterogeneous transforms"
```

## Task 6: Validate a Typed Heterogeneous Data Specification

**Files:**

- Create: `topobench/data/heterogeneous.py`
- Modify: `topobench/data/__init__.py`
- Test: `test/data/test_heterogeneous_spec.py`

**Step 1: Write validation tests**

Define the expected immutable result:

```python
@dataclass(frozen=True)
class HeterogeneousDataSpec:
    metadata: tuple[tuple[str, ...], tuple[tuple[str, str, str], ...]]
    target_node_type: str
    num_classes: int
    input_channels: Mapping[str, int]
```

Tests must cover:

- a valid transformed synthetic graph produces exact metadata and per-type widths;
- missing target node type;
- missing target `y`;
- missing or non-boolean masks;
- empty masks;
- overlapping masks;
- labels outside `[0, num_classes)`;
- a featureless node type after transforms;
- malformed edge index shape;
- source or destination edge index out of bounds.

Use parametrization and require actionable error messages naming the node or edge type.

**Step 2: Run and confirm import failure**

Run:

```bash
uv run pytest test/data/test_heterogeneous_spec.py -q
```

Expected: FAIL.

**Step 3: Implement one validation entry point**

Implement:

```python
def validate_heterogeneous_node_data(
    data: HeteroData,
    *,
    target_node_type: str,
    num_classes: int,
) -> HeterogeneousDataSpec:
    ...
```

Return tuples or immutable mapping proxies so later code cannot silently mutate the specification. This function owns all schema and split-mask validation; loaders, datamodules, and models must consume the result instead of repeating checks.

Do not require every node to be in a split for real datasets; require masks to be pairwise disjoint and non-empty. The stricter exhaustive split property belongs only to the synthetic fixture test.

**Step 4: Verify**

Run:

```bash
uv run pytest test/data/test_heterogeneous_spec.py -q
```

Expected: PASS.

**Step 5: Commit**

```bash
git add topobench/data/heterogeneous.py topobench/data/__init__.py test/data/test_heterogeneous_spec.py
git commit -m "feat: validate heterogeneous node data contracts"
```

## Task 7: Introduce the Configurable Data-Pipeline Boundary

**Files:**

- Create: `topobench/data/pipelines/__init__.py`
- Create: `topobench/data/pipelines/base.py`
- Create: `topobench/data/pipelines/default.py`
- Create: `topobench/data/pipelines/heterogeneous.py`
- Create: `configs/data_pipeline/default.yaml`
- Create: `configs/data_pipeline/heterogeneous_node.yaml`
- Modify: `configs/run.yaml`
- Modify: `topobench/run.py`
- Modify: `test/_utils/simplified_pipeline.py`
- Test: `test/data/pipelines/test_data_pipelines.py`
- Test: `test/pipeline/test_pipeline.py`

**Step 1: Test the result contract and default equivalence**

Specify:

```python
@dataclass(frozen=True)
class DataPipelineOutput:
    datamodule: LightningDataModule
    preprocessing_time: float
    data_spec: object | None = None
```

Add tests that:

- the default pipeline calls the same loader, `PreProcessor.load_dataset_splits`, and `TBDataloader` path as today;
- the default output has `data_spec is None`;
- existing graph and cell experiment composition selects `data_pipeline=default`;
- the heterogeneous pipeline receives one processed `HeteroData`, validates it once, and never instantiates `TBDataloader`;
- zero or multiple heterogeneous graphs fail with a clear transductive-v1 error.

**Step 2: Run and confirm failure**

Run:

```bash
uv run pytest test/data/pipelines/test_data_pipelines.py -q
```

Expected: FAIL because the pipeline abstraction does not exist.

**Step 3: Move existing orchestration without changing it**

Create a `DefaultDataPipeline` whose `build()` contains the current sequence from `topobench/run.py`:

1. instantiate dataset loader;
2. load dataset and directory;
3. instantiate transforms;
4. construct `PreProcessor`;
5. call the existing split method;
6. create `TBDataloader`.

The move must be behavior-preserving. Do not rename split keys or change defaults in this task.

**Step 4: Add the heterogeneous pipeline shell**

`HeterogeneousNodeDataPipeline.build()` must:

1. reuse the same loader, transform instantiation, and `PreProcessor`;
2. obtain the single processed native graph without calling homogeneous split utilities;
3. validate it with `validate_heterogeneous_node_data`;
4. delegate loader construction to the dedicated data module introduced in Task 8.

It must not inspect model names.

**Step 5: Make `run.py` configuration-driven**

Add to `configs/run.yaml` defaults:

```yaml
- data_pipeline: default
```

Replace the hard-coded data block with:

```python
pipeline = hydra.utils.instantiate(cfg.data_pipeline)
pipeline_output = pipeline.build(cfg)
datamodule = pipeline_output.datamodule
```

Log `pipeline_output.preprocessing_time`. Put `data_spec` in `object_dict`. Update `simplified_pipeline.py` to use the same pipeline construction so tests do not maintain a second orchestration implementation.

**Step 6: Verify the default pipeline did not regress**

Run:

```bash
uv run pytest \
  test/data/pipelines/test_data_pipelines.py \
  test/pipeline/test_pipeline.py \
  test/data/dataload/test_Dataloaders.py -q
```

Expected: PASS. Existing tests must not require heterogeneous configuration.

**Step 7: Commit**

```bash
git add \
  topobench/data/pipelines \
  configs/data_pipeline \
  configs/run.yaml \
  topobench/run.py \
  test/_utils/simplified_pipeline.py \
  test/data/pipelines/test_data_pipelines.py \
  test/pipeline/test_pipeline.py
git commit -m "refactor: add configurable data pipeline boundary"
```

## Task 8: Add a Separate Batched Heterogeneous Data Module

**Files:**

- Create: `topobench/dataloader/heterogeneous.py`
- Modify: `topobench/dataloader/__init__.py`
- Modify: `topobench/data/pipelines/heterogeneous.py`
- Test: `test/data/dataload/test_heterogeneous_dataloader.py`

**Step 1: Specify full-graph loading**

Test `mode="full_batch"`:

- `train_dataloader`, `val_dataloader`, and `test_dataloader` use PyG `DataLoader`;
- each yields one native `HeteroData`;
- labels and phase masks remain on the target store;
- batch size is exactly one graph regardless of a stray graph `batch_size` setting.

**Step 2: Specify sampled loading**

Test `mode="neighbor"` with:

```python
num_neighbors = [3, 2]
batch_size = 4
shuffle = True  # train only
```

For each phase assert:

```python
batch["author"].batch_size <= 4
seed_global_ids = batch["author"].n_id[: batch["author"].batch_size]
assert seed_global_ids belong to the configured phase mask
```

Also assert:

- validation and test loaders are not shuffled;
- `input_nodes=("author", phase_mask)` is used;
- `num_neighbors` is passed for every relation;
- the same generic loader class is used for HGT and HeteroSAGE;
- the sampled batch contains `n_id` and target-store `batch_size`;
- no `DataloadDataset`, `collate_fn`, or `TBDataloader` constructor is called.

Skip the neighbor-specific test with a precise dependency message only if the installed PyG sampling backend is unavailable. CI should install the declared sparse dependencies, so a skip is diagnostic, not the desired normal result.

**Step 3: Run and confirm failure**

Run:

```bash
uv run pytest test/data/dataload/test_heterogeneous_dataloader.py -q
```

Expected: FAIL.

**Step 4: Implement `HeterogeneousNodeDataModule`**

Constructor inputs:

```python
HeterogeneousNodeDataModule(
    data: HeteroData,
    spec: HeterogeneousDataSpec,
    *,
    mode: Literal["full_batch", "neighbor"],
    batch_size: int = 128,
    num_neighbors: list[int] | dict[EdgeType, list[int]] = [15, 10],
    num_workers: int = 0,
    pin_memory: bool = False,
)
```

Implementation rules:

- construct one loader per phase;
- neighbor input nodes come only from the target store's phase mask;
- train shuffles; validation and test do not;
- seed nodes are always the first `batch[target_type].batch_size` target nodes, per the PyG `NeighborLoader` contract;
- keep evaluation sampled and fixed in v1; expose the protocol in config and logs;
- reject unknown mode, empty fanout, nonpositive batch size, or target types absent from the spec.

Do not subclass or modify `TBDataloader`.

**Step 5: Wire the heterogeneous pipeline**

Build the data module from `cfg.dataset.dataloader_params` and the validated spec. Add the spec and sampling protocol to `DataPipelineOutput`.

**Step 6: Verify**

Run:

```bash
uv run pytest \
  test/data/dataload/test_heterogeneous_dataloader.py \
  test/data/pipelines/test_data_pipelines.py \
  test/data/dataload/test_Dataloaders.py -q
```

Expected: PASS, including the unchanged topological loader tests.

**Step 7: Commit**

```bash
git add \
  topobench/dataloader/heterogeneous.py \
  topobench/dataloader/__init__.py \
  topobench/data/pipelines/heterogeneous.py \
  test/data/dataload/test_heterogeneous_dataloader.py
git commit -m "feat: add batched heterogeneous data module"
```

## Task 9: Isolate Typed Supervision in `TBModel`

**Files:**

- Create: `topobench/model/supervision.py`
- Modify: `topobench/model/model.py`
- Modify: `topobench/model/__init__.py`
- Test: `test/model/test_supervision.py`
- Test: `test/model/test_model.py`

**Step 1: Test the adapter contract**

Define:

```python
@dataclass(frozen=True)
class SupervisedBatch:
    logits: Tensor
    targets: Tensor
    num_examples: int
```

Test the legacy adapter against current graph, transductive-node, and inductive-node behavior. Its output must be byte-for-byte/equal-tensor compatible with the current `TBModel.process_outputs`.

Test the heterogeneous adapter:

- full-batch selects the phase mask from `batch[target_node_type]`;
- neighbor mode selects only `[: batch[target_node_type].batch_size]`;
- sampled validation/test do not accidentally supervise context nodes;
- empty supervised slices raise a readable error;
- `num_examples` is the number of selected labels.

**Step 2: Run and confirm failure**

Run:

```bash
uv run pytest test/model/test_supervision.py -q
```

Expected: FAIL.

**Step 3: Extract legacy logic before adding heterogeneous logic**

Move the current selection rules into `DefaultSupervisionAdapter` without semantic changes. Make `TBModel` accept an optional adapter:

```python
supervision_adapter: SupervisionAdapter | None = None
```

Default to the legacy adapter so every existing config continues to work.

Add `HeterogeneousNodeSupervisionAdapter(target_node_type=...)`.

Have `model_step()` call the adapter after the wrapper/readout returns logits. Use `num_examples` as Lightning's `batch_size` for loss and metric logging instead of hard-coded `1`.

Do not add `isinstance(HeteroData)` branches to every train/validation/test method.

**Step 4: Verify legacy and heterogeneous behavior**

Run:

```bash
uv run pytest \
  test/model/test_supervision.py \
  test/model/test_model.py \
  test/pipeline/test_pipeline.py -q
```

Expected: PASS.

**Step 5: Commit**

```bash
git add topobench/model test/model/test_supervision.py test/model/test_model.py
git commit -m "refactor: isolate typed supervision selection"
```

## Task 10: Add Per-Type Feature Encoding

**Files:**

- Create: `topobench/nn/encoders/heterogeneous_node_encoder.py`
- Modify: `topobench/nn/encoders/__init__.py`
- Test: `test/nn/encoders/test_heterogeneous_node_encoder.py`

**Step 1: Write failing unit tests**

Tests must assert:

- different source widths map to the same hidden width;
- every metadata node type has a projection;
- the encoder accepts and returns native `HeteroData`, matching the existing
  `TBModel.feature_encoder(batch)` contract;
- encoded `x_dict` keys exactly equal the original `x_dict` keys;
- a missing node type raises an error naming it;
- unknown extra node types are rejected;
- gradients reach every projection;
- no parameters are created during `forward()` after initialization.

Example:

```python
encoder = HeterogeneousNodeFeatureEncoder(
    input_channels={"author": 8, "paper": 5, "venue": 1},
    hidden_channels=16,
    dropout=0.0,
)
encoded_data = encoder(data)
assert encoded_data is data
assert {
    key: value.shape for key, value in encoded_data.x_dict.items()
} == {
    "author": (36, 16),
    "paper": (24, 16),
    "venue": (6, 16),
}
```

**Step 2: Run and confirm failure**

Run:

```bash
uv run pytest test/nn/encoders/test_heterogeneous_node_encoder.py -q
```

Expected: FAIL.

**Step 3: Implement deterministic module construction**

Use `nn.ModuleDict` of per-type `nn.Linear(input_width, hidden_channels)`, plus
configured activation and dropout. Build all layers in `__init__` from the
validated data spec. In `forward`, replace each `data[node_type].x` with its
encoded value and return the same `HeteroData` object, just as existing
TopoBench feature encoders update and return the batch. Do not use
uncontrolled lazy creation in `forward`; optimizer construction happens
before the first batch.

**Step 4: Verify**

Run:

```bash
uv run pytest test/nn/encoders/test_heterogeneous_node_encoder.py -q
```

Expected: PASS.

**Step 5: Commit**

```bash
git add \
  topobench/nn/encoders/heterogeneous_node_encoder.py \
  topobench/nn/encoders/__init__.py \
  test/nn/encoders/test_heterogeneous_node_encoder.py
git commit -m "feat: add heterogeneous node feature encoder"
```

## Task 11: Extract a Reusable HGT Backbone Without Breaking Cell HGT

**Files:**

- Create: `topobench/nn/backbones/heterogeneous/__init__.py`
- Create: `topobench/nn/backbones/heterogeneous/hgt.py`
- Modify: `topobench/nn/backbones/combinatorial/hgt.py`
- Test: `test/nn/backbones/heterogeneous/test_hgt.py`
- Test: `test/nn/backbones/combinatorial/test_hgt.py`

**Step 1: Test the generic HGT contract**

Instantiate with transformed synthetic metadata and assert:

- the module accepts `x_dict` and `edge_index_dict`;
- every node type returns `[num_nodes, hidden_channels]`;
- all configured layers execute;
- invalid metadata and `hidden_channels % heads != 0` fail early;
- gradients reach every HGT layer;
- both full and neighbor-sampled `HeteroData` can be forwarded.

**Step 2: Run and confirm failure**

Run:

```bash
uv run pytest test/nn/backbones/heterogeneous/test_hgt.py -q
```

Expected: FAIL.

**Step 3: Extract the metadata-driven backbone**

Create:

```python
class HGTBackbone(nn.Module):
    def __init__(
        self,
        metadata: Metadata,
        hidden_channels: int,
        num_layers: int,
        heads: int,
        dropout: float,
        activation: str,
    ): ...

    def forward(
        self,
        x_dict: dict[str, Tensor],
        edge_index_dict: dict[EdgeType, Tensor],
    ) -> dict[str, Tensor]: ...
```

Each layer applies `HGTConv`, activation, normalization, and dropout consistently. Preserve node types that receive no update in a layer through an explicit residual/carry-forward rule, and test that rule.

**Step 4: Adapt `CellHGT`**

Make `CellHGT` delegate to or subclass the generic implementation while retaining:

- `edge_types`;
- `metadata`;
- `to_heterogeneous_inputs`;
- constructor arguments and config targets;
- state-dict compatibility where practical;
- current output structure by cell rank.

Do not change the existing cell-complex HGT experiment semantics.

**Step 5: Verify generic and compatibility tests**

Run:

```bash
uv run pytest \
  test/nn/backbones/heterogeneous/test_hgt.py \
  test/nn/backbones/combinatorial/test_hgt.py \
  test/pipeline/test_hgt_pipeline.py -q
```

Expected: PASS.

**Step 6: Commit**

```bash
git add \
  topobench/nn/backbones/heterogeneous \
  topobench/nn/backbones/combinatorial/hgt.py \
  test/nn/backbones/heterogeneous/test_hgt.py \
  test/nn/backbones/combinatorial/test_hgt.py
git commit -m "refactor: extract reusable heterogeneous hgt backbone"
```

## Task 12: Add a Generic HeteroSAGE Baseline

**Files:**

- Create: `topobench/nn/backbones/heterogeneous/heterosage.py`
- Modify: `topobench/nn/backbones/heterogeneous/__init__.py`
- Test: `test/nn/backbones/heterogeneous/test_heterosage.py`

**Step 1: Write failing relation-wise message-passing tests**

Assert:

- one `SAGEConv((-1, -1), hidden_channels)` exists per edge type per layer;
- relations are combined through `HeteroConv(..., aggr="sum")`;
- output shapes match every node type;
- changing one relation's edges changes the appropriate destination embeddings;
- node types with no incoming relation survive through the carry-forward rule;
- full and sampled batches forward successfully;
- gradients reach relation-specific convolutions.

**Step 2: Run and confirm failure**

Run:

```bash
uv run pytest test/nn/backbones/heterogeneous/test_heterosage.py -q
```

Expected: FAIL.

**Step 3: Implement the baseline**

Build every relation module in `__init__` from metadata. For every layer:

1. run `HeteroConv`;
2. merge untouched node types from the previous dictionary;
3. apply per-type normalization, activation, and dropout;
4. optionally apply a same-width residual when configured.

Keep the public input/output contract identical to `HGTBackbone`, allowing the wrapper and data module to remain model-agnostic.

**Step 4: Verify**

Run:

```bash
uv run pytest \
  test/nn/backbones/heterogeneous/test_heterosage.py \
  test/nn/backbones/heterogeneous/test_hgt.py -q
```

Expected: PASS.

**Step 5: Commit**

```bash
git add \
  topobench/nn/backbones/heterogeneous/heterosage.py \
  topobench/nn/backbones/heterogeneous/__init__.py \
  test/nn/backbones/heterogeneous/test_heterosage.py
git commit -m "feat: add heterogeneous graphsage baseline"
```

## Task 13: Connect Encoder, Backbone, and Target-Node Readout

**Files:**

- Create: `topobench/nn/wrappers/heterogeneous/__init__.py`
- Create: `topobench/nn/wrappers/heterogeneous/heterogeneous_wrapper.py`
- Modify: `topobench/nn/wrappers/__init__.py`
- Create: `topobench/nn/readouts/heterogeneous_node.py`
- Modify: `topobench/nn/readouts/__init__.py`
- Test: `test/nn/wrappers/heterogeneous/test_heterogeneous_wrapper.py`
- Test: `test/nn/readouts/test_heterogeneous_node_readout.py`

**Step 1: Test the wrapper boundary**

The wrapper must:

- require native `HeteroData`;
- receive the already encoded `HeteroData` returned by
  `TBModel.feature_encoder(batch)`;
- call either backbone with `batch.edge_index_dict`;
- return typed embeddings plus
  `labels=batch[target_node_type].y`, without filtering labels or masks;
- retain `n_id` and seed-count information on the batch for the supervision adapter;
- produce identical interface shape for HGT and HeteroSAGE.

**Step 2: Test the readout**

The readout follows TopoBench's existing
`readout(model_out=model_out, batch=batch)` call contract, selects
`model_out["x_dict"][target_node_type]`, and adds one classifier result under
`model_out["logits"]`:

```python
readout = HeterogeneousNodeReadout(
    target_node_type="author",
    hidden_channels=16,
    out_channels=2,
)
model_out = readout(
    model_out={"x_dict": x_dict, "labels": data["author"].y},
    batch=data,
)
assert model_out["logits"].shape == (num_authors, 2)
```

It must fail clearly if the target embedding is absent and advertise `task_level = "node"`.

**Step 3: Run and confirm failures**

Run:

```bash
uv run pytest \
  test/nn/wrappers/heterogeneous/test_heterogeneous_wrapper.py \
  test/nn/readouts/test_heterogeneous_node_readout.py -q
```

Expected: FAIL.

**Step 4: Implement the two narrow components**

The wrapper may copy the unfiltered target labels into `model_out`, matching
the current TopoBench wrapper contract. Keep phase-mask and sampled seed
selection out of both wrapper and readout; that remains the supervision
adapter's job. Avoid checking specific backbone classes.

**Step 5: Verify**

Run:

```bash
uv run pytest \
  test/nn/wrappers/heterogeneous/test_heterogeneous_wrapper.py \
  test/nn/readouts/test_heterogeneous_node_readout.py \
  test/model/test_supervision.py -q
```

Expected: PASS.

**Step 6: Commit**

```bash
git add \
  topobench/nn/wrappers/heterogeneous \
  topobench/nn/wrappers/__init__.py \
  topobench/nn/readouts/heterogeneous_node.py \
  topobench/nn/readouts/__init__.py \
  test/nn/wrappers/heterogeneous/test_heterogeneous_wrapper.py \
  test/nn/readouts/test_heterogeneous_node_readout.py
git commit -m "feat: connect heterogeneous model components"
```

## Task 14: Inject Runtime Metadata Into Hydra Model Construction

**Files:**

- Create: `topobench/utils/model_instantiation.py`
- Modify: `topobench/utils/__init__.py`
- Modify: `topobench/run.py`
- Create: `configs/model/heterogeneous/hgt.yaml`
- Create: `configs/model/heterogeneous/heterosage.yaml`
- Test: `test/utils/test_model_instantiation.py`

**Step 1: Write configuration-safety tests**

Tests must assert:

- a heterogeneous data spec supplies exact `input_channels` and `metadata`;
- `target_node_type` and `num_classes` reach the readout and supervision adapter;
- HGT and HeteroSAGE receive the same metadata;
- the source `DictConfig` is unchanged;
- default models receive no injected fields and instantiate as before;
- metadata from config cannot silently disagree with the validated graph.

**Step 2: Run and confirm failure**

Run:

```bash
uv run pytest test/utils/test_model_instantiation.py -q
```

Expected: FAIL.

**Step 3: Add one centralized construction helper**

Implement a helper such as:

```python
def instantiate_model(
    cfg: DictConfig,
    *,
    data_spec: HeterogeneousDataSpec | None,
) -> LightningModule:
    ...
```

Copy the model config before injecting runtime-only values. Do not mutate Hydra's composed configuration. Only the helper knows how a data spec maps to component arguments; `run.py`, models, and loaders must not duplicate that mapping.

**Step 4: Define both model configs**

Both configs use:

- `_target_: topobench.model.TBModel`;
- `HeterogeneousNodeFeatureEncoder`;
- `HeterogeneousWrapper`;
- either `HGTBackbone` or `HeteroSAGEBackbone`;
- `HeterogeneousNodeReadout`;
- `HeterogeneousNodeSupervisionAdapter`.

Keep comparable defaults:

```yaml
hidden_channels: 64
num_layers: 2
dropout: 0.1
```

HGT additionally uses `heads: 4`. Do not make input widths or metadata Hydra resolvers; they are runtime properties of the validated processed graph.

**Step 5: Replace direct model instantiation in `run.py`**

Call the helper with `pipeline_output.data_spec`. All callback, logger, trainer, checkpoint rerun, and W&B behavior stays shared.

**Step 6: Verify**

Run:

```bash
uv run pytest \
  test/utils/test_model_instantiation.py \
  test/model/test_model.py \
  test/pipeline/test_pipeline.py \
  test/pipeline/test_hgt_pipeline.py -q
```

Expected: PASS.

**Step 7: Commit**

```bash
git add \
  topobench/utils/model_instantiation.py \
  topobench/utils/__init__.py \
  topobench/run.py \
  configs/model/heterogeneous \
  test/utils/test_model_instantiation.py
git commit -m "feat: instantiate models from heterogeneous metadata"
```

## Task 15: Pass G2 With Full-Batch Synthetic End-to-End Tests

**Files:**

- Create: `configs/experiment/heterogeneous_synthetic_hgt_full.yaml`
- Create: `configs/experiment/heterogeneous_synthetic_heterosage_full.yaml`
- Create: `configs/logger/heterogeneous_wandb.yaml`
- Create: `test/pipeline/test_heterogeneous_pipeline.py`
- Modify: `configs/dataset/heterogeneous/SyntheticHeterogeneous.yaml`

**Step 1: Configure consistent heterogeneous W&B identity**

Create a logger config based on the existing W&B logger, with:

```yaml
wandb:
  _target_: lightning.pytorch.loggers.wandb.WandbLogger
  save_dir: ${paths.output_dir}
  offline: false
  id: null
  anonymous: null
  project: topobench-heterogeneous
  name: ${dataset.loader.parameters.data_name}-${model.model_name}-${dataset.dataloader_params.mode}-seed${seed}
  group: ${dataset.loader.parameters.data_name}
  tags: ${tags}
  job_type: ${dataset.dataloader_params.mode}
  log_model: false
  prefix: ""
```

Every heterogeneous experiment config must override `/logger:
heterogeneous_wandb`. This gives all runs one project and deterministic,
meaningful dataset/model/mode/seed names. Tests may still override
`logger=null`.

**Step 2: Write an end-to-end lifecycle test for each model**

Compose the real Hydra config and run two short CPU epochs with:

```yaml
data_pipeline: heterogeneous_node
dataset: heterogeneous/SyntheticHeterogeneous
trainer:
  accelerator: cpu
  devices: 1
  max_epochs: 2
  limit_train_batches: 2
  limit_val_batches: 2
  limit_test_batches: 2
train: true
test: true
logger: null
```

Patch only filesystem output paths. Do not mock the encoder, backbone, `TBModel`, trainer, or data module.

Assert:

- `trainer.fit` completed;
- train and validation loss are finite;
- validation and test metrics exist;
- best-checkpoint rerun executed validation and test;
- the checkpoint callback reports a best model path;
- both HGT and HeteroSAGE use the same native data pipeline.

**Step 3: Add an overfit-signal test**

For the small synthetic fixture, train a fixed number of steps and assert the final training loss is lower than the first by a conservative margin. Seed all RNGs. This catches disconnected features or incorrect target selection without asserting an unstable accuracy threshold.

**Step 4: Run and inspect the first failure**

Run:

```bash
uv run pytest test/pipeline/test_heterogeneous_pipeline.py \
  -k "full_batch" -q
```

Expected: initially FAIL at the first incomplete integration point. Fix only that point and repeat until the lifecycle passes.

**Step 5: Verify Gate G2**

Run:

```bash
uv run pytest \
  test/pipeline/test_heterogeneous_pipeline.py \
  test/pipeline/test_pipeline.py \
  test/pipeline/test_hgt_pipeline.py -q
```

Expected: PASS for both heterogeneous models and existing pipelines.

**Step 6: Commit**

```bash
git add \
  configs/experiment/heterogeneous_synthetic_hgt_full.yaml \
  configs/experiment/heterogeneous_synthetic_heterosage_full.yaml \
  configs/logger/heterogeneous_wandb.yaml \
  configs/dataset/heterogeneous/SyntheticHeterogeneous.yaml \
  test/pipeline/test_heterogeneous_pipeline.py
git commit -m "test: validate full batch heterogeneous training"
```

## Task 16: Pass G3 With Neighbor-Sampled End-to-End Tests

**Files:**

- Create: `configs/experiment/heterogeneous_synthetic_hgt_neighbor.yaml`
- Create: `configs/experiment/heterogeneous_synthetic_heterosage_neighbor.yaml`
- Modify: `test/pipeline/test_heterogeneous_pipeline.py`
- Modify: `test/model/test_supervision.py`

**Step 1: Add sampled experiment configs**

Override:

```yaml
dataset:
  dataloader_params:
    mode: neighbor
    batch_size: 4
    num_neighbors: [3, 2]
    num_workers: 0
```

Record these W&B tags when logging is enabled:

```yaml
tags: [heterogeneous, synthetic, neighbor, hgt]  # heterosage in the other config
```

**Step 2: Add lifecycle and accounting tests**

For both models assert:

- one epoch consumes more than one train batch;
- each loss uses only the target seed count;
- the sum of phase seed counts covers the configured mask once when shuffle is disabled;
- context target nodes are never included in supervision;
- validation and test complete under the explicitly named `sampled` protocol;
- best-checkpoint validation/test rerun works with neighbor loaders.

**Step 3: Run and confirm the intended failure**

Run:

```bash
uv run pytest test/pipeline/test_heterogeneous_pipeline.py \
  -k "neighbor" -q
```

Expected: FAIL until sampled batch handling is fully connected.

**Step 4: Fix only integration defects**

Likely integration points are device transfer, `batch_size` discovery, mask retention, or output slicing. Do not special-case HGT or HeteroSAGE in the dataloader.

**Step 5: Verify Gate G3**

Run:

```bash
uv run pytest \
  test/data/dataload/test_heterogeneous_dataloader.py \
  test/model/test_supervision.py \
  test/pipeline/test_heterogeneous_pipeline.py -q
```

Expected: PASS for both full and neighbor modes.

**Step 6: Commit**

```bash
git add \
  configs/experiment/heterogeneous_synthetic_hgt_neighbor.yaml \
  configs/experiment/heterogeneous_synthetic_heterosage_neighbor.yaml \
  test/pipeline/test_heterogeneous_pipeline.py \
  test/model/test_supervision.py
git commit -m "test: validate sampled heterogeneous training"
```

## Task 17: Add DBLP as the First Real Heterogeneous Dataset

**Files:**

- Create: `topobench/data/loaders/heterogeneous/dblp.py`
- Modify: `topobench/data/loaders/heterogeneous/__init__.py`
- Modify: `topobench/data/loaders/__init__.py`
- Create: `configs/dataset/heterogeneous/DBLP.yaml`
- Create: `configs/transforms/dataset_defaults/DBLP.yaml`
- Create: `configs/experiment/heterogeneous_dblp_hgt.yaml`
- Create: `configs/experiment/heterogeneous_dblp_heterosage.yaml`
- Modify: `test/data/load/test_heterogeneous_dataset_loaders.py`
- Create: `test/integration/test_dblp_heterogeneous.py`

**Step 1: Test the loader without network access**

Monkeypatch `torch_geometric.datasets.DBLP` with a small local stand-in and assert:

- root path is correct;
- native metadata is preserved;
- target node type is `author`;
- four classes are configured;
- official `author` masks are kept;
- no TopoBench random split utility is called.

**Step 2: Configure DBLP-specific preprocessing**

The PyG DBLP graph already includes reverse relation types. Its `conference` node type lacks features. Configure only constant features for `conference`; do not apply `ToUndirected` again unless a metadata test proves a missing reverse relation.

Validate the processed metadata rather than hard-coding it inside either model.

**Step 3: Run offline tests**

Run:

```bash
uv run pytest \
  test/data/load/test_heterogeneous_dataset_loaders.py \
  test/data/test_heterogeneous_spec.py -q
```

Expected: PASS without downloading DBLP.

**Step 4: Add an opt-in real-data smoke test**

Mark the integration test:

```python
@pytest.mark.integration
@pytest.mark.download
```

It should load/process DBLP, validate the spec, take one full-batch train step for each model, and assert finite loss. Keep it excluded from the default suite unless the data already exists or an explicit environment flag enables downloads.

**Step 5: Run the real Gate G4 smoke test**

Run manually:

```bash
TOPOBENCH_ALLOW_DOWNLOADS=1 uv run pytest \
  test/integration/test_dblp_heterogeneous.py -q
```

Expected: PASS for HGT and HeteroSAGE.

Then run one seed for each real experiment:

```bash
uv run python -m topobench.run \
  experiment=heterogeneous_dblp_hgt \
  seed=0 \
  logger=heterogeneous_wandb
```

```bash
uv run python -m topobench.run \
  experiment=heterogeneous_dblp_heterosage \
  seed=0 \
  logger=heterogeneous_wandb
```

Expected: both runs train, select a best checkpoint, and log final validation and test reruns to the same W&B project with distinct names.

**Step 6: Commit**

```bash
git add \
  topobench/data/loaders/heterogeneous \
  topobench/data/loaders/__init__.py \
  configs/dataset/heterogeneous/DBLP.yaml \
  configs/transforms/dataset_defaults/DBLP.yaml \
  configs/experiment/heterogeneous_dblp_hgt.yaml \
  configs/experiment/heterogeneous_dblp_heterosage.yaml \
  test/data/load/test_heterogeneous_dataset_loaders.py \
  test/integration/test_dblp_heterogeneous.py
git commit -m "feat: add dblp heterogeneous node classification"
```

## Task 18: Prepare OGB-MAG for Sampled Large-Graph Runs

**Files:**

- Create: `topobench/data/loaders/heterogeneous/ogb_mag.py`
- Modify: `topobench/data/loaders/heterogeneous/__init__.py`
- Modify: `topobench/data/loaders/__init__.py`
- Create: `configs/dataset/heterogeneous/OGB_MAG.yaml`
- Create: `configs/transforms/dataset_defaults/OGB_MAG.yaml`
- Create: `configs/experiment/heterogeneous_ogb_mag_hgt.yaml`
- Create: `configs/experiment/heterogeneous_ogb_mag_heterosage.yaml`
- Modify: `test/data/load/test_heterogeneous_dataset_loaders.py`
- Create: `test/integration/test_ogb_mag_preflight.py`

**Step 1: Add an offline loader-contract test**

Monkeypatch `torch_geometric.datasets.OGB_MAG` and assert:

- `preprocess="metapath2vec"` is passed so all node types have usable features;
- target node type is `paper`;
- class count is 349;
- official paper masks are preserved;
- reverse typed relations are added exactly once;
- the dataset is configured for neighbor sampling, never ordinary example batching.

**Step 2: Add the conservative sampled configuration**

Start with:

```yaml
parameters:
  target_node_type: paper
  num_classes: 349

dataloader_params:
  mode: neighbor
  batch_size: 128
  num_neighbors: [15, 10]
  num_workers: 4
  pin_memory: true
```

Use `ToUndirected(merge=False)` because raw OGB-MAG relations are directed. Record fanout, batch size, number of layers, preprocessing variant, and `evaluation_protocol: sampled` in W&B.

Do not use `HGTLoader` in v1: the generic `NeighborLoader` is deliberately shared by HGT and HeteroSAGE for a controlled model comparison.

**Step 3: Add a preflight command/test**

The opt-in preflight must:

1. load and validate the processed graph;
2. construct train/validation/test neighbor loaders;
3. fetch one batch from each;
4. print per-type node counts and per-relation edge counts;
5. run one forward/backward/optimizer step for each model;
6. report peak accelerator memory when available;
7. avoid a full epoch by default.

Run:

```bash
TOPOBENCH_ALLOW_DOWNLOADS=1 uv run pytest \
  test/integration/test_ogb_mag_preflight.py -q
```

Expected: PASS without an out-of-memory error.

**Step 4: Tune only resource parameters if preflight fails**

Change in this order:

1. reduce `batch_size`;
2. reduce fanout;
3. reduce hidden width;
4. reduce workers if host memory is the issue.

Do not change target masks, metadata, class count, or supervision selection to make the run fit.

**Step 5: Prepare, but do not automatically launch, one-epoch commands**

HGT:

```bash
uv run python -m topobench.run \
  experiment=heterogeneous_ogb_mag_hgt \
  seed=0 \
  trainer.max_epochs=1 \
  logger=heterogeneous_wandb
```

HeteroSAGE:

```bash
uv run python -m topobench.run \
  experiment=heterogeneous_ogb_mag_heterosage \
  seed=0 \
  trainer.max_epochs=1 \
  logger=heterogeneous_wandb
```

Expected: one complete epoch plus best-checkpoint validation/test rerun under the sampled evaluation protocol.

**Step 6: Commit**

```bash
git add \
  topobench/data/loaders/heterogeneous \
  topobench/data/loaders/__init__.py \
  configs/dataset/heterogeneous/OGB_MAG.yaml \
  configs/transforms/dataset_defaults/OGB_MAG.yaml \
  configs/experiment/heterogeneous_ogb_mag_hgt.yaml \
  configs/experiment/heterogeneous_ogb_mag_heterosage.yaml \
  test/data/load/test_heterogeneous_dataset_loaders.py \
  test/integration/test_ogb_mag_preflight.py
git commit -m "feat: prepare sampled ogb mag experiments"
```

## Task 19: Document the Public Extension Contract

**Files:**

- Create: `docs/heterogeneous_graphs.md`
- Modify: `README.md`
- Modify: `docs/plans/2026-07-30-native-heterogeneous-graph-design.md`
- Test: `test/docs/test_heterogeneous_examples.py`

**Step 1: Add documentation examples as executable tests**

Extract or mirror the documented synthetic commands in a test that:

- composes both model configs;
- builds the synthetic pipeline in full and neighbor modes;
- performs one forward pass;
- uses `train=false test=false` for a fast configuration check.

**Step 2: Write user and contributor documentation**

Document:

- what “heterogeneous” means here versus the existing heterophilous homogeneous datasets;
- supported v1 task: single-graph node classification;
- native `HeteroData` invariants;
- how to select `target_node_type`;
- how official masks are consumed;
- transform compatibility and featureless node types;
- full-batch versus neighbor-sampled behavior;
- seed-node-only loss;
- sampled evaluation caveat;
- how to add a dataset;
- how to add a metadata-driven model;
- synthetic, DBLP, and OGB-MAG commands;
- W&B naming conventions;
- current non-goals: link prediction, graph classification, distributed sampling, `HGTLoader`, learned featureless-node embeddings.

Link the approved design document rather than duplicating its rationale.

**Step 3: Run documentation examples**

Run:

```bash
uv run pytest test/docs/test_heterogeneous_examples.py -q
```

Expected: PASS.

**Step 4: Commit**

```bash
git add \
  docs/heterogeneous_graphs.md \
  README.md \
  docs/plans/2026-07-30-native-heterogeneous-graph-design.md \
  test/docs/test_heterogeneous_examples.py
git commit -m "docs: explain heterogeneous graph workflows"
```

## Task 20: Final Regression, Quality, and Reproducibility Audit

**Files:**

- Modify only if a test exposes a defect in files already introduced above.

**Step 1: Run the focused heterogeneous suite**

Run:

```bash
uv run pytest \
  test/dependencies/test_torch_geometric_dependency.py \
  test/data/datasets/test_synthetic_heterogeneous_dataset.py \
  test/data/load/test_heterogeneous_dataset_loaders.py \
  test/data/preprocess/test_preprocessor.py \
  test/data/test_heterogeneous_spec.py \
  test/data/pipelines/test_data_pipelines.py \
  test/data/dataload/test_heterogeneous_dataloader.py \
  test/transforms/test_heterogeneous_transforms.py \
  test/model/test_supervision.py \
  test/model/test_model.py \
  test/nn/encoders/test_heterogeneous_node_encoder.py \
  test/nn/backbones/heterogeneous \
  test/nn/backbones/combinatorial/test_hgt.py \
  test/nn/wrappers/heterogeneous \
  test/nn/readouts/test_heterogeneous_node_readout.py \
  test/utils/test_model_instantiation.py \
  test/pipeline/test_heterogeneous_pipeline.py \
  test/pipeline/test_hgt_pipeline.py \
  test/docs/test_heterogeneous_examples.py -q
```

Expected: PASS with no unexpected skips. A neighbor-sampling skip must state the missing backend and be resolved in the supported development environment before G3 is accepted.

**Step 2: Run formatting and lint**

Run:

```bash
uv run ruff format --check topobench test
uv run ruff check topobench test
```

Expected: PASS.

If formatting is needed:

```bash
uv run ruff format topobench test
```

Then rerun both checks.

**Step 3: Run the full test suite**

Run:

```bash
uv run pytest -q
```

Expected: PASS. Investigate every failure; do not label unrelated failures without reproducing them against the pre-feature commit.

**Step 4: Run deterministic synthetic repetitions**

Run the synthetic full and sampled tests three times:

```bash
uv run pytest test/pipeline/test_heterogeneous_pipeline.py -q --count=3
```

If `pytest-repeat` is not installed, run the command three times manually rather than adding a runtime dependency.

Expected: all repetitions pass with stable schemas and finite metrics.

**Step 5: Audit architectural boundaries**

Run:

```bash
rg -n "HeteroData|NeighborLoader|HeterogeneousNodeDataModule" \
  topobench/dataloader topobench/data/pipelines topobench/run.py
rg -n "TBDataloader|DataloadDataset|collate_fn|DomainData" \
  topobench/data/pipelines/heterogeneous.py \
  topobench/dataloader/heterogeneous.py
git diff --check
git status --short
```

Expected:

- heterogeneous-specific types appear only at the intended boundaries;
- the second search returns no heterogeneous-path dependency on the old batching stack;
- `git diff --check` returns no whitespace errors;
- only intended changes remain.

**Step 6: Perform manual promotion gates**

In order:

1. G2: synthetic full-batch HGT and HeteroSAGE;
2. G3: synthetic neighbor-sampled HGT and HeteroSAGE;
3. G4: DBLP smoke plus one-seed W&B runs;
4. G5: OGB-MAG preflight, followed by the one-epoch commands only when resources permit.

Record in the implementation handoff:

- exact command;
- git commit;
- dataset preprocessing variant;
- seed;
- batch size and fanout;
- model depth, width, and heads where applicable;
- best checkpoint path;
- W&B URL;
- whether test metrics are exact/full or sampled.

**Step 7: Commit audit-only fixes, if any**

```bash
git add <only-files-fixed-during-audit>
git commit -m "test: complete heterogeneous graph regression audit"
```

Skip this commit when the audit required no code changes.

## Acceptance Criteria

Implementation is complete only when all of the following are true:

- Native `HeteroData` survives loading, preprocessing, transforms, saving, reloading, batching, model execution, and checkpoint rerun.
- The existing homogeneous/topological path remains the default and its regression tests pass unchanged.
- The heterogeneous data path uses its own full-graph or `NeighborLoader` data module and never enters `TBDataloader` or the topological collate path.
- Featureless types are resolved by explicit configured transforms before validation.
- A single immutable data spec owns metadata, feature-width, target, class-count, mask, label, and edge-bound validation.
- HGT and HeteroSAGE share the same encoder, wrapper, readout, supervision, loaders, and experiment lifecycle.
- Neighbor-sampled loss and metrics use target seed nodes only.
- `TBModel` still owns training, evaluation, logging, optimizer setup, and best-checkpoint testing for every domain.
- Synthetic full and neighbor modes pass end to end for both models.
- DBLP uses official author masks and passes its real-data smoke gate.
- OGB-MAG uses official paper masks, `metapath2vec` preprocessing, explicit reverse relations, and neighbor sampling; its preflight fits the target machine before a full run.
- W&B runs share project `topobench-heterogeneous`, use meaningful dataset/model/mode/seed names, and label sampled versus full evaluation.
- Documentation describes both the supported contract and current non-goals.

## Deferred Work

The following require separate designs after this plan is complete:

- heterogeneous link prediction and edge supervision;
- heterogeneous graph-level classification;
- multiple target node types in one task;
- learned embeddings for featureless node types;
- `HGTLoader` or metapath-aware model-specific sampling;
- exact layer-wise or full-graph inference for large validation/test graphs;
- distributed neighbor sampling;
- heterogeneous-to-topological lifts.

Keeping these out of v1 is deliberate: none is required to establish a clean, reusable native heterogeneous graph foundation.
