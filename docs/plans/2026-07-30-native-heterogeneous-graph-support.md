# Native Heterogeneous Graph Support Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add first-class, batched heterogeneous node-classification support to TopoBench with native PyG `HeteroData`, reusable HGT and HeteroSAGE models, deterministic synthetic tests, and staged promotion to DBLP and OGB-MAG.

**Architecture:** Keep TopoBench's shared experiment, preprocessing, model, evaluator, callback, logging, and best-checkpoint evaluation shell. Introduce a configurable data-pipeline boundary: the existing default pipeline remains behaviorally unchanged, while a heterogeneous-node pipeline preserves native `HeteroData`, validates a typed data specification, and selects either PyG `DataLoader` for full-graph training or `NeighborLoader` for sampled mini-batches. Models consume metadata-driven dictionaries through heterogeneous-specific encoder, wrapper, backbone, and readout components; `TBModel` delegates selection of supervised predictions and targets to a small adapter rather than branching throughout the training loop.

**Tech Stack:** Python 3.11, PyTorch 2.3, PyTorch Geometric 2.8.0.post1, Lightning 2.4, Hydra 1.3, pytest, Ruff, W&B.

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

## Normative Implementation Contracts

These contracts resolve details that would otherwise be left to the
implementer:

1. **Compatibility baseline:** implement against the locked
   `torch-geometric==2.8.0.post1`. A local editable or development PyG build is
   useful for debugging but is not the supported baseline.
2. **Native persistence:** use `InMemoryDataset.save()` and `load()`; PyG stores
   the concrete `HeteroData` class in the serialized tuple. Do not invent a
   second serialization format.
3. **Eager parameters:** build every per-type projection, HGT layer, relation
   SAGE convolution, normalization, and classifier in `__init__`. No trainable
   parameter may first appear during `forward()`.
4. **Existing `TBModel` flow:** the heterogeneous feature encoder accepts and
   returns `HeteroData`; the wrapper receives that encoded batch; the readout
   receives `model_out` plus the original batch. Do not create an alternate
   forward path.
5. **Explicit sampling mode:** the supervision adapter receives
   `mode="full_batch"` or `mode="neighbor"` from configuration. It must not
   guess the protocol from the presence of arbitrary attributes.
6. **Directional sampling depth:** v1 uses
   `NeighborLoader(subgraph_type="directional")`. Every relation fanout list
   must have exactly `model.backbone.num_layers` entries. Reject a mismatch
   before training.
7. **Seed-only supervision:** in neighbor mode, the first
   `batch[target_node_type].batch_size` target nodes are the only examples used
   by the loss and evaluator. Context nodes are never supervised.
8. **Exact default-path compatibility:** `DefaultDataPipeline` must move the
   existing loader/preprocessor/split/`TBDataloader` sequence without changing
   its semantics. Any unrelated cleanup is deferred.
9. **No accepted sampling skips:** neighbor-sampling tests may report a clear
   missing-backend diagnostic during development, but Gate G3 cannot pass
   while those tests are skipped.
10. **One graph in v1:** the heterogeneous data pipeline requires exactly one
    processed `HeteroData`. Multiple heterogeneous examples, heterogeneous
    graph classification, and heterogeneous link prediction remain deferred.

Use the following shared type vocabulary in skeletons:

```python
from typing import Literal, TypeAlias

from torch_geometric.data import Data, HeteroData
from torch_geometric.typing import EdgeType, Metadata, NodeType

DataObject: TypeAlias = Data | HeteroData
SamplingMode: TypeAlias = Literal["full_batch", "neighbor"]
```

Primary API anchors for implementation review:

- [PyG heterogeneous graph tutorial](https://pytorch-geometric.readthedocs.io/en/2.5.0/tutorial/heterogeneous.html)
- [PyG `InMemoryDataset`](https://pytorch-geometric.readthedocs.io/en/2.8.0/generated/torch_geometric.data.InMemoryDataset.html)
- [PyG neighbor-sampling tutorial](https://pytorch-geometric.readthedocs.io/en/latest/tutorial/neighbor_loader.html)
- [PyG `HGTConv`](https://pytorch-geometric.readthedocs.io/en/2.8.0/generated/torch_geometric.nn.conv.HGTConv.html)
- [PyG DBLP dataset](https://pytorch-geometric.readthedocs.io/en/2.8.0/generated/torch_geometric.datasets.DBLP.html)
- [PyG OGB-MAG dataset](https://pytorch-geometric.readthedocs.io/en/2.8.0/generated/torch_geometric.datasets.OGB_MAG.html)

## Task 1: Freeze the Existing Behavior and Declare PyG Directly

**Files:**

- Modify: `pyproject.toml:20-72`
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

    assert "torch-geometric==2.8.0.post1" in dependencies
```

**Step 3: Run the new test and confirm the failure**

Run:

```bash
uv run pytest test/dependencies/test_torch_geometric_dependency.py -q
```

Expected: FAIL because PyG currently arrives only transitively.

**Step 4: Add the tested direct dependency**

Add `"torch-geometric==2.8.0.post1"` to `dependencies` in `pyproject.toml`.
This matches the existing `uv.lock` resolution and avoids claiming
compatibility with versions that this implementation has not exercised.
Broaden the range only in a later dependency-compatibility change backed by a
CI version matrix.

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

Use one canonical factory plus a thin `InMemoryDataset`. The dataset wrapper
is important because `PreProcessor` already knows how to consume PyG datasets;
tests and debug runs must not maintain different representations.

Implement this structure:

```python
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData, InMemoryDataset


def _stratified_masks(
    labels: torch.Tensor,
    *,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    masks = [
        torch.zeros(labels.numel(), dtype=torch.bool) for _ in range(3)
    ]
    for class_id in labels.unique(sorted=True):
        indices = (labels == class_id).nonzero(as_tuple=False).view(-1)
        indices = indices[
            torch.randperm(indices.numel(), generator=generator)
        ]
        train_end = max(1, int(0.6 * indices.numel()))
        val_end = train_end + max(1, int(0.2 * indices.numel()))
        if val_end >= indices.numel():
            raise ValueError(
                "Each synthetic class needs train, validation, and test nodes"
            )
        masks[0][indices[:train_end]] = True
        masks[1][indices[train_end:val_end]] = True
        masks[2][indices[val_end:]] = True
    return masks[0], masks[1], masks[2]


def make_synthetic_heterogeneous_data(
    *,
    seed: int = 0,
    num_authors: int = 36,
    num_papers: int = 24,
    num_venues: int = 6,
) -> HeteroData:
    if num_authors < 12 or num_authors % 2:
        raise ValueError("num_authors must be even and at least 12")
    if num_papers < num_venues or num_venues < 2 or num_papers % 4:
        raise ValueError(
            "Require num_papers divisible by four and "
            "num_papers >= num_venues >= 2"
        )

    generator = torch.Generator().manual_seed(seed)
    labels = torch.arange(num_authors, dtype=torch.long) % 2
    train_mask, val_mask, test_mask = _stratified_masks(
        labels, generator=generator
    )

    author_x = 0.1 * torch.randn(
        num_authors, 8, generator=generator
    )
    author_x[:, :2] += 2.0 * F.one_hot(labels, num_classes=2).float()
    paper_x = 0.1 * torch.randn(num_papers, 5, generator=generator)
    paper_signal = (torch.arange(num_papers) // 2) % 2
    paper_x[:, :2] += 1.5 * F.one_hot(
        paper_signal, num_classes=2
    ).float()

    author_ids = torch.arange(num_authors).repeat_interleave(2)
    write_slot = torch.arange(2).repeat(num_authors)
    paper_ids = (2 * author_ids + write_slot) % num_papers

    data = HeteroData()
    data["author"].x = author_x
    data["author"].y = labels
    data["author"].train_mask = train_mask
    data["author"].val_mask = val_mask
    data["author"].test_mask = test_mask
    data["paper"].x = paper_x
    data["venue"].num_nodes = num_venues
    data["author", "writes", "paper"].edge_index = torch.stack(
        [author_ids, paper_ids]
    )
    data["paper", "published_in", "venue"].edge_index = torch.stack(
        [
            torch.arange(num_papers),
            torch.arange(num_papers) % num_venues,
        ]
    )
    data.validate(raise_on_error=True)
    return data


class SyntheticHeterogeneousDataset(InMemoryDataset):
    """One deterministic native heterogeneous graph for tests and debugging."""

    def __init__(self, **kwargs: int) -> None:
        super().__init__(root=None)
        data = make_synthetic_heterogeneous_data(**kwargs)
        self.data, self.slices = self.collate([data])
```

Use a local `torch.Generator().manual_seed(seed)`; never mutate global RNG state. Give:

- `author`: 8-dimensional features, two balanced classes, `y`, and official-style masks;
- `paper`: 5-dimensional features;
- `venue`: `num_nodes` only, deliberately featureless;
- forward-only `writes` and `published_in` relations.

Construct edges so every node participates, indices stay in bounds, and
author labels have a simple signal both in `author.x[:, :2]` and in the paper
features reached through their typed neighborhood. This ensures both HGT and
relation-wise SAGE can pass a short overfit diagnostic without depending on a
model-specific self-loop. The fixture is a correctness and overfitting probe,
not a benchmark.

The final implementation should add full NumPy-style docstrings and explicit
parameter types. The abbreviated skeleton above is normative for data shape
and RNG ownership.

The dataset registry auto-discovers the `InMemoryDataset` subclass but not the
factory function. Add an explicit import and `__all__` entry for
`make_synthetic_heterogeneous_data` in
`topobench/data/datasets/__init__.py`, and test both exports from a clean
process.

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
- Modify: `test/data/load/test_datasetloaders.py`

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

Subclass `AbstractLoader`, return the canonical one-element dataset, and expose
only fixture-generation arguments from configuration. Keep the loader free of
model or batching decisions:

```python
from omegaconf import DictConfig
from torch_geometric.data import Dataset

from topobench.data.datasets import SyntheticHeterogeneousDataset
from topobench.data.loaders.base import AbstractLoader


class SyntheticHeterogeneousDatasetLoader(AbstractLoader):
    """Load the deterministic native heterogeneous debug graph."""

    def __init__(self, parameters: DictConfig) -> None:
        super().__init__(parameters)

    def load_dataset(self) -> Dataset:
        return SyntheticHeterogeneousDataset(
            seed=int(self.parameters.seed),
            num_authors=int(self.parameters.get("num_authors", 36)),
            num_papers=int(self.parameters.get("num_papers", 24)),
            num_venues=int(self.parameters.get("num_venues", 6)),
        )
```

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
  loss_type: cross_entropy
  monitor_metric: accuracy

split_params:
  learning_setting: transductive
  source: official_masks

dataloader_params:
  mode: full_batch
  num_workers: 0
  pin_memory: false
  persistent_workers: false
```

Do not add fake top-level `num_features`; heterogeneous input widths belong to the data stores.

**Step 4: Verify loader and config composition**

Before running the repository loader suite, update its generic state check:

```python
data = dataset[0]
if isinstance(data, HeteroData):
    assert data.node_types
    assert data.edge_types
    for node_type in data.node_types:
        assert data[node_type].num_nodes > 0
else:
    assert data.x.size(0) > 0
    assert data.y.size(0) > 0
```

Do not require every raw heterogeneous node type to have `x`; featureless
stores are resolved later by the configured preprocessor transforms.

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
  test/data/load/test_heterogeneous_dataset_loaders.py \
  test/data/load/test_datasetloaders.py
git commit -m "feat: add synthetic heterogeneous dataset loader"
```

## Task 4: Preserve `HeteroData` in the Existing PreProcessor

**Files:**

- Modify: `topobench/data/preprocessor/preprocessor.py:23-251`
- Modify: `topobench/data/preprocessor/__init__.py`
- Test: `test/data/preprocess/test_preprocessor.py`

**Step 1: Add round-trip tests before changing production code**

Add tests that pass a one-graph dataset containing `HeteroData` to `PreProcessor` and assert:

```python
def test_preprocessor_preserves_heterodata_on_process_and_reload(
    synthetic_heterogeneous_dataset,
    tmp_path,
) -> None:
    identity = DictConfig({"transform_name": "Identity"})
    first = PreProcessor(
        synthetic_heterogeneous_dataset,
        tmp_path,
        transforms_config=identity,
    )
    reloaded = PreProcessor(
        synthetic_heterogeneous_dataset,
        tmp_path,
        transforms_config=identity,
    )

    assert isinstance(first[0], HeteroData)
    assert isinstance(reloaded[0], HeteroData)
    assert first[0].metadata() == reloaded[0].metadata()
    assert torch.equal(
        first[0]["author"].train_mask,
        reloaded[0]["author"].train_mask,
    )
```

An identity transform is intentional here: `transforms_config=None` follows
TopoBench's in-memory fast path and would not exercise process/save/reload.

Retain an explicit homogeneous regression test asserting processed `Data` remains `Data`.

**Step 2: Run and confirm the intended assertion failure**

Run:

```bash
uv run pytest test/data/preprocess/test_preprocessor.py \
  -k "heterodata or preserves_data" -q
```

Expected: the heterogeneous case fails at the current homogeneous-only assertion.

**Step 3: Generalize the preprocessor contract**

Introduce a local type alias and small family check:

```python
from torch_geometric.data import Data, HeteroData

SupportedData = Data | HeteroData


def _data_family(data: SupportedData) -> type[Data] | type[HeteroData]:
    return HeteroData if isinstance(data, HeteroData) else Data
```

Update `process()` and annotations to:

- accept either native type;
- preserve the input representation exactly;
- apply the configured transform to each example;
- reject a transform result that is neither `Data` nor `HeteroData`;
- reject a representation change (`Data` to `HeteroData` or the reverse) unless a future transform explicitly declares such a conversion;
- use PyG's supported save/collation mechanism so reload retains stores, metadata, masks, and edge indices.

The central loop should have this shape:

```python
if isinstance(
    self.dataset,
    (torch_geometric.data.Dataset, torch.utils.data.Dataset),
):
    data_list = list(self.dataset)
elif isinstance(self.dataset, (Data, HeteroData)):
    data_list = [self.dataset]
else:
    raise TypeError(
        "PreProcessor expects a PyG/PyTorch dataset, Data, or HeteroData; "
        f"received {type(self.dataset).__name__}"
    )

processed: list[SupportedData] = []
for original in tqdm(data_list, desc="Processing graphs", unit="graph"):
    if not isinstance(original, (Data, HeteroData)):
        raise TypeError(
            "Dataset item must be Data or HeteroData; "
            f"received {type(original).__name__}"
        )
    transformed = (
        self.pre_transform(original)
        if self.pre_transform is not None
        else original
    )
    if not isinstance(transformed, (Data, HeteroData)):
        raise TypeError(
            "A pre-transform returned unsupported type "
            f"{type(transformed).__name__}"
        )
    if _data_family(original) is not _data_family(transformed):
        raise TypeError(
            "Pre-transforms must preserve Data versus HeteroData "
            "representation"
        )
    processed.append(transformed)

self.data_list = processed
self._data, self.slices = self.collate(processed)
self._data_list = None
self.save(processed, self.processed_paths[0])
```

Keep the existing custom `load()` backward-compatibility branches. PyG's
three-element save format already records `data.__class__`, so its
`data_cls.from_dict(data)` path reconstructs `HeteroData` correctly.

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

- Modify: `topobench/transforms/data_transform.py:8-47`
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

`DataTransform.forward()` should perform the check centrally:

```python
from torch_geometric.data import Data, HeteroData


def forward(self, data: Data | HeteroData) -> Data | HeteroData:
    if self.transform is None:
        return data
    if isinstance(data, HeteroData) and not getattr(
        self.transform, "supports_heterodata", False
    ):
        raise TypeError(
            f"{type(self.transform).__name__} does not declare HeteroData "
            f"support for metadata={data.metadata()}"
        )
    transformed = self.transform(data)
    if not isinstance(transformed, (Data, HeteroData)):
        raise TypeError(
            f"{type(self.transform).__name__} returned unsupported type "
            f"{type(transformed).__name__}"
        )
    return transformed
```

Create thin registered wrappers around PyG's transforms. Alias the PyG imports
so the dynamic manipulation discovery cannot mistake them for TopoBench
classes:

```python
from torch_geometric.data import HeteroData
from torch_geometric.transforms import Constant as PyGConstant
from torch_geometric.transforms import ToUndirected as PyGToUndirected
from torch_geometric.transforms import BaseTransform


class HeterogeneousConstantFeatures(BaseTransform):
    supports_heterodata = True

    def __init__(
        self,
        node_types: str | list[str],
        value: float = 1.0,
        cat: bool = False,
        **_: object,
    ) -> None:
        self.transform = PyGConstant(
            value=value,
            cat=cat,
            node_types=node_types,
        )

    def forward(self, data: HeteroData) -> HeteroData:
        return self.transform(data)


class HeterogeneousToUndirected(BaseTransform):
    supports_heterodata = True

    def __init__(
        self,
        reduce: str = "add",
        merge: bool = False,
        **_: object,
    ) -> None:
        self.transform = PyGToUndirected(reduce=reduce, merge=merge)

    def forward(self, data: HeteroData) -> HeteroData:
        return self.transform(data)
```

Mark only these wrappers as heterogeneous-compatible. Do not blanket-mark existing topological liftings.

**Step 4: Configure the synthetic default transform**

The two reusable manipulation configs are:

```yaml
# configs/transforms/data_manipulations/heterogeneous_constant_features.yaml
transform_name: HeterogeneousConstantFeatures
transform_type: data manipulation
node_types: null
value: 1.0
cat: false
```

```yaml
# configs/transforms/data_manipulations/heterogeneous_to_undirected.yaml
transform_name: HeterogeneousToUndirected
transform_type: data manipulation
merge: false
reduce: add
```

Compose:

1. constant features for `venue`;
2. `ToUndirected(merge=False)`.

The final graph must contain all required `x_dict` entries before model construction. Directionality is configuration, not an implicit model side effect.

Use this concrete dataset-default shape:

```yaml
defaults:
  - data_manipulations@venue_features: heterogeneous_constant_features
  - data_manipulations@reverse_relations: heterogeneous_to_undirected

venue_features:
  node_types: venue
  value: 1.0
  cat: false

reverse_relations:
  merge: false
  reduce: add
```

Each included manipulation config supplies its own `transform_name`. Add a
config-composition test asserting the final transform order is
`venue_features`, then `reverse_relations`; Python mapping order is the
preprocessor execution order.

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
from __future__ import annotations

from dataclasses import dataclass

from torch_geometric.typing import EdgeType, Metadata


@dataclass(frozen=True)
class HeterogeneousDataSpec:
    node_types: tuple[str, ...]
    edge_types: tuple[EdgeType, ...]
    target_node_type: str
    num_classes: int
    input_channels: tuple[tuple[str, int], ...]

    @property
    def input_channels_dict(self) -> dict[str, int]:
        return dict(self.input_channels)

    def pyg_metadata(self) -> Metadata:
        return list(self.node_types), list(self.edge_types)
```

Tuples make the stored specification actually immutable and trivially
serializable. Return fresh lists/dictionaries only at framework boundaries.

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

Build it around small single-purpose validators:

```python
_MASK_NAMES = ("train_mask", "val_mask", "test_mask")


def _validate_node_features(data: HeteroData) -> tuple[tuple[str, int], ...]:
    channels: list[tuple[str, int]] = []
    for node_type in data.node_types:
        store = data[node_type]
        if store.num_nodes is None:
            raise ValueError(f"Node type {node_type!r} has no num_nodes")
        if "x" not in store:
            raise ValueError(
                f"Node type {node_type!r} has no x after preprocessing"
            )
        if store.x.ndim != 2 or store.x.size(0) != store.num_nodes:
            raise ValueError(
                f"Node type {node_type!r} has invalid x shape "
                f"{tuple(store.x.shape)} for {store.num_nodes} nodes"
            )
        if store.x.size(-1) < 1:
            raise ValueError(f"Node type {node_type!r} has zero feature width")
        if not store.x.is_floating_point():
            raise TypeError(
                f"Node type {node_type!r} features must be floating point"
            )
        channels.append((node_type, int(store.x.size(-1))))
    return tuple(channels)


def _validate_target_store(
    data: HeteroData,
    *,
    target_node_type: str,
    num_classes: int,
) -> None:
    if target_node_type not in data.node_types:
        raise ValueError(
            f"Unknown target node type {target_node_type!r}; "
            f"available={data.node_types}"
        )
    store = data[target_node_type]
    if "y" not in store:
        raise ValueError(f"Target store {target_node_type!r} has no y")
    if store.y.dtype != torch.long or store.y.ndim != 1:
        raise TypeError("Target labels must be one-dimensional torch.long")
    if store.y.numel() != store.num_nodes:
        raise ValueError("Target label count must equal target node count")

    masks = []
    for name in _MASK_NAMES:
        if name not in store:
            raise ValueError(f"Target store is missing {name}")
        mask = store[name]
        if mask.dtype != torch.bool or mask.shape != store.y.shape:
            raise TypeError(
                f"{target_node_type}.{name} must be bool with shape "
                f"{tuple(store.y.shape)}"
            )
        if not bool(mask.any()):
            raise ValueError(f"{target_node_type}.{name} is empty")
        masks.append(mask)

    overlap = masks[0].to(torch.int8)
    overlap = overlap + masks[1].to(torch.int8) + masks[2].to(torch.int8)
    if bool((overlap > 1).any()):
        raise ValueError("Target train/val/test masks must be disjoint")

    supervised = masks[0] | masks[1] | masks[2]
    labels = store.y[supervised]
    if bool(((labels < 0) | (labels >= num_classes)).any()):
        raise ValueError(
            f"Target labels must be in [0, {num_classes})"
        )


def validate_heterogeneous_node_data(
    data: HeteroData,
    *,
    target_node_type: str,
    num_classes: int,
) -> HeterogeneousDataSpec:
    if not isinstance(data, HeteroData):
        raise TypeError("Expected native torch_geometric.data.HeteroData")
    try:
        data.validate(raise_on_error=True)
    except ValueError as error:
        raise ValueError(f"Invalid heterogeneous graph: {error}") from error
    if num_classes < 2:
        raise ValueError("num_classes must be at least 2")

    input_channels = _validate_node_features(data)
    _validate_target_store(
        data,
        target_node_type=target_node_type,
        num_classes=num_classes,
    )
    return HeterogeneousDataSpec(
        node_types=tuple(data.node_types),
        edge_types=tuple(data.edge_types),
        target_node_type=target_node_type,
        num_classes=num_classes,
        input_channels=input_channels,
    )
```

This function owns all schema and split-mask validation; loaders, datamodules,
and models must consume the result instead of repeating checks.

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
- Create: `configs/data_pipeline/default.yaml`
- Modify: `configs/run.yaml:1-60`
- Modify: `topobench/run.py:84-145`
- Modify: `test/_utils/simplified_pipeline.py:20-91`
- Test: `test/data/pipelines/test_data_pipelines.py`
- Test: `test/pipeline/test_pipeline.py`

**Step 1: Test the result contract and default equivalence**

Specify:

```python
@dataclass(frozen=True)
class DataPipelineOutput:
    datamodule: LightningDataModule
    preprocessing_time: float
    data_spec: HeterogeneousDataSpec | None = None
```

Add tests that:

- the default pipeline calls the same loader, `PreProcessor.load_dataset_splits`, and `TBDataloader` path as today;
- the default output has `data_spec is None`;
- existing graph and cell experiment composition selects `data_pipeline=default`;
- the production `run()` and simplified test runner consume the same
  `DataPipelineOutput` contract.

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

Use these concrete interfaces:

```python
# topobench/data/pipelines/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import hydra
from lightning import LightningDataModule
from omegaconf import DictConfig

from topobench.data.heterogeneous import HeterogeneousDataSpec
from topobench.data.preprocessor import PreProcessor


@dataclass(frozen=True)
class DataPipelineOutput:
    datamodule: LightningDataModule
    preprocessing_time: float
    data_spec: HeterogeneousDataSpec | None = None


class AbstractDataPipeline(ABC):
    @staticmethod
    def preprocess(cfg: DictConfig) -> PreProcessor:
        loader = hydra.utils.instantiate(cfg.dataset.loader)
        dataset, dataset_dir = loader.load()
        transforms = (
            hydra.utils.instantiate(cfg.transforms)
            if cfg.get("transforms") is not None
            else None
        )
        return PreProcessor(dataset, dataset_dir, transforms)

    @abstractmethod
    def build(self, cfg: DictConfig) -> DataPipelineOutput:
        """Build a Lightning data module and its runtime data contract."""


# topobench/data/pipelines/default.py
class DefaultDataPipeline(AbstractDataPipeline):
    def build(self, cfg: DictConfig) -> DataPipelineOutput:
        preprocessor = self.preprocess(cfg)
        train, val, test = preprocessor.load_dataset_splits(
            cfg.dataset.split_params
        )
        datamodule = TBDataloader(
            dataset_train=train,
            dataset_val=val,
            dataset_test=test,
            **cfg.dataset.get("dataloader_params", {}),
        )
        return DataPipelineOutput(
            datamodule=datamodule,
            preprocessing_time=preprocessor.preprocessing_time,
        )
```

`topobench.data.heterogeneous` depends only on PyTorch/PyG and does not import
pipelines, so this direct type is acyclic. Add a test that imports
`topobench.data.pipelines` in a clean Python subprocess to guard package-export
cycles.

**Step 4: Make `run.py` configuration-driven**

Add to `configs/run.yaml` defaults:

```yaml
- data_pipeline: default
```

`configs/data_pipeline/default.yaml` is:

```yaml
_target_: topobench.data.pipelines.DefaultDataPipeline
```

Replace the hard-coded data block with:

```python
pipeline = hydra.utils.instantiate(cfg.data_pipeline)
pipeline_output = pipeline.build(cfg)
datamodule = pipeline_output.datamodule
```

Log `pipeline_output.preprocessing_time`. Put `data_spec` in `object_dict`. Update `simplified_pipeline.py` to use the same pipeline construction so tests do not maintain a second orchestration implementation.

**Step 5: Verify the default pipeline did not regress**

Run:

```bash
uv run pytest \
  test/data/pipelines/test_data_pipelines.py \
  test/pipeline/test_pipeline.py \
  test/data/dataload/test_Dataloaders.py -q
```

Expected: PASS. Existing tests must not require heterogeneous configuration.

**Step 6: Commit**

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
- Create: `topobench/data/pipelines/heterogeneous.py`
- Create: `configs/data_pipeline/heterogeneous_node.yaml`
- Test: `test/data/dataload/test_heterogeneous_dataloader.py`
- Modify: `test/data/pipelines/test_data_pipelines.py`

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
expected_ids = phase_mask.nonzero(as_tuple=False).view(-1)
assert bool(torch.isin(seed_global_ids, expected_ids).all())
```

Also assert:

- validation and test loaders are not shuffled;
- `input_nodes=("author", phase_mask)` is used;
- `num_neighbors` is passed for every relation;
- the same generic loader class is used for HGT and HeteroSAGE;
- the sampled batch contains `n_id` and target-store `batch_size`;
- no `DataloadDataset`, `collate_fn`, or `TBDataloader` constructor is called.

If the installed PyG sampling backend is unavailable, fail with a precise
diagnostic naming the missing `pyg-lib`/`torch-sparse` capability. Do not mark
the test as an accepted skip: neighbor batching is a core requirement and G3
cannot pass without executing it.

**Step 3: Run and confirm failure**

Run:

```bash
uv run pytest test/data/dataload/test_heterogeneous_dataloader.py -q
```

Expected: FAIL.

**Step 4: Implement `HeterogeneousNodeDataModule`**

Constructor signature:

```python
def __init__(
    self,
    data: HeteroData,
    spec: HeterogeneousDataSpec,
    *,
    mode: Literal["full_batch", "neighbor"],
    batch_size: int = 128,
    num_neighbors: Sequence[int]
    | Mapping[EdgeType, Sequence[int]]
    | None = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    train_shuffle: bool = True,
    replace: bool = False,
    subgraph_type: str = "directional",
    filter_per_worker: bool = False,
) -> None:
    ...
```

The implementation skeleton is:

```python
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

from lightning import LightningDataModule
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader, NeighborLoader
from torch_geometric.typing import EdgeType


Phase = Literal["train", "val", "test"]
SamplingMode = Literal["full_batch", "neighbor"]


class HeterogeneousNodeDataModule(LightningDataModule):
    def __init__(
        self,
        data: HeteroData,
        spec: HeterogeneousDataSpec,
        *,
        mode: SamplingMode,
        batch_size: int = 128,
        num_neighbors: Sequence[int]
        | Mapping[EdgeType, Sequence[int]]
        | None = None,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        train_shuffle: bool = True,
        replace: bool = False,
        subgraph_type: str = "directional",
        filter_per_worker: bool = False,
    ) -> None:
        super().__init__()
        if mode not in {"full_batch", "neighbor"}:
            raise ValueError(f"Unsupported heterogeneous loader mode: {mode}")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        if persistent_workers and num_workers == 0:
            raise ValueError(
                "persistent_workers requires num_workers greater than zero"
            )
        if subgraph_type != "directional":
            raise ValueError(
                "Heterogeneous v1 supports directional sampling only"
            )

        fanout = [15, 10] if num_neighbors is None else num_neighbors
        self.data = data
        self.spec = spec
        self.mode = mode
        self.batch_size = batch_size
        self.num_neighbors = _normalize_fanout(fanout, spec.edge_types)
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.train_shuffle = train_shuffle
        self.replace = replace
        self.subgraph_type = subgraph_type
        self.filter_per_worker = filter_per_worker

    def _common_kwargs(self) -> dict[str, object]:
        return {
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "persistent_workers": self.persistent_workers,
        }

    def _full_loader(self) -> DataLoader:
        return DataLoader(
            [self.data],
            batch_size=1,
            shuffle=False,
            **self._common_kwargs(),
        )

    def _neighbor_loader(self, phase: Phase) -> NeighborLoader:
        mask = self.data[self.spec.target_node_type][f"{phase}_mask"]
        return NeighborLoader(
            self.data,
            input_nodes=(self.spec.target_node_type, mask),
            num_neighbors=self.num_neighbors,
            batch_size=self.batch_size,
            shuffle=self.train_shuffle if phase == "train" else False,
            replace=self.replace,
            subgraph_type=self.subgraph_type,
            filter_per_worker=self.filter_per_worker,
            **self._common_kwargs(),
        )

    def _loader(self, phase: Phase) -> DataLoader | NeighborLoader:
        if self.mode == "full_batch":
            return self._full_loader()
        return self._neighbor_loader(phase)

    def train_dataloader(self) -> DataLoader | NeighborLoader:
        return self._loader("train")

    def val_dataloader(self) -> DataLoader | NeighborLoader:
        return self._loader("val")

    def test_dataloader(self) -> DataLoader | NeighborLoader:
        return self._loader("test")
```

Implement `_normalize_fanout()` as a pure function. It must copy all input
sequences into positive `list[int]` values, reject empty lists and zero values,
and, for relation-specific dictionaries, require keys to equal
`spec.edge_types` exactly. `-1` may be supported later; v1 disallows it so
runtime cost stays bounded.

Implementation rules:

- construct one loader per phase;
- neighbor input nodes come only from the target store's phase mask;
- train shuffles; validation and test do not;
- seed nodes are always the first `batch[target_type].batch_size` target nodes, per the PyG `NeighborLoader` contract;
- keep evaluation sampled and fixed in v1; expose the protocol in config and logs;
- reject unknown mode, empty fanout, nonpositive batch size, or target types absent from the spec.

Add `save_hyperparameters(ignore=["data", "spec"])` only after confirming the
remaining fields are YAML/JSON serializable. Never serialize the full graph in
Lightning datamodule hyperparameters.

Do not subclass or modify `TBDataloader`.

**Step 5: Wire the heterogeneous pipeline**

Add:

```python
class HeterogeneousNodeDataPipeline(AbstractDataPipeline):
    def build(self, cfg: DictConfig) -> DataPipelineOutput:
        preprocessor = self.preprocess(cfg)
        if len(preprocessor) != 1:
            raise ValueError(
                "Heterogeneous node classification v1 requires exactly "
                f"one processed graph; received {len(preprocessor)}"
            )
        data = preprocessor[0]
        if not isinstance(data, HeteroData):
            raise TypeError(
                "The heterogeneous pipeline requires native HeteroData; "
                f"received {type(data).__name__}"
            )
        spec = validate_heterogeneous_node_data(
            data,
            target_node_type=cfg.dataset.parameters.target_node_type,
            num_classes=cfg.dataset.parameters.num_classes,
        )
        datamodule = HeterogeneousNodeDataModule(
            data=data,
            spec=spec,
            **cfg.dataset.dataloader_params,
        )
        return DataPipelineOutput(
            datamodule=datamodule,
            preprocessing_time=preprocessor.preprocessing_time,
            data_spec=spec,
        )
```

Build the data module from `cfg.dataset.dataloader_params` and the validated
spec. The pipeline never calls homogeneous split utilities: target-store
masks already define the phases.

`configs/data_pipeline/heterogeneous_node.yaml` is:

```yaml
_target_: topobench.data.pipelines.HeterogeneousNodeDataPipeline
```

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
  configs/data_pipeline/heterogeneous_node.yaml \
  test/data/pipelines/test_data_pipelines.py \
  test/data/dataload/test_heterogeneous_dataloader.py
git commit -m "feat: add batched heterogeneous data module"
```

## Task 9: Isolate Typed Supervision in `TBModel`

**Files:**

- Create: `topobench/model/supervision.py`
- Modify: `topobench/model/model.py:34-251`
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

Add
`HeterogeneousNodeSupervisionAdapter(target_node_type=..., mode=...)`.
Use this concrete protocol:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from torch import Tensor
from torch_geometric.data import Data, HeteroData


SamplingMode = Literal["full_batch", "neighbor"]

_PHASE_MASK = {
    "Training": "train_mask",
    "Validation": "val_mask",
    "Test": "test_mask",
}


@dataclass(frozen=True)
class SupervisedBatch:
    logits: Tensor
    targets: Tensor
    num_examples: int


class SupervisionAdapter(Protocol):
    def select(
        self,
        model_out: dict[str, object],
        batch: Data | HeteroData,
        phase: str,
    ) -> SupervisedBatch:
        """Select the predictions and labels owned by one phase."""


def _model_tensors(
    model_out: dict[str, object],
) -> tuple[Tensor, Tensor]:
    logits = model_out.get("logits")
    labels = model_out.get("labels")
    if not isinstance(logits, Tensor) or not isinstance(labels, Tensor):
        raise TypeError("model_out must contain tensor logits and labels")
    if logits.size(0) != labels.size(0):
        raise ValueError("Logit and label counts must match before selection")
    return logits, labels


class DefaultSupervisionAdapter:
    def __init__(self, task_level: str) -> None:
        self.task_level = task_level

    def select(
        self,
        model_out: dict[str, object],
        batch: Data | HeteroData,
        phase: str,
    ) -> SupervisedBatch:
        logits, labels = _model_tensors(model_out)
        if self.task_level != "node":
            # Preserve the historical Lightning weighting for graph and
            # node-inductive tasks in this scoped refactor.
            return SupervisedBatch(logits, labels, num_examples=1)
        try:
            mask_name = _PHASE_MASK[phase]
        except KeyError as error:
            raise ValueError(f"Invalid state_str: {phase}") from error
        mask = getattr(batch, mask_name)
        selected = int(mask.sum().item())
        if selected == 0:
            raise ValueError(f"No supervised examples for {phase}")
        return SupervisedBatch(
            logits=logits[mask],
            targets=labels[mask],
            num_examples=selected,
        )


class HeterogeneousNodeSupervisionAdapter:
    def __init__(
        self,
        target_node_type: str,
        mode: SamplingMode,
    ) -> None:
        if mode not in {"full_batch", "neighbor"}:
            raise ValueError(f"Unsupported sampling mode: {mode}")
        self.target_node_type = target_node_type
        self.mode = mode

    def select(
        self,
        model_out: dict[str, object],
        batch: Data | HeteroData,
        phase: str,
    ) -> SupervisedBatch:
        if not isinstance(batch, HeteroData):
            raise TypeError("Heterogeneous supervision requires HeteroData")
        if phase not in _PHASE_MASK:
            raise ValueError(f"Invalid state_str: {phase}")
        if self.target_node_type not in batch.node_types:
            raise ValueError(
                f"Batch has no target store {self.target_node_type!r}"
            )
        logits, labels = _model_tensors(model_out)
        if logits.ndim != 2 or labels.ndim != 1:
            raise ValueError(
                "Heterogeneous classification expects [N, C] logits "
                "and [N] labels"
            )
        target_store = batch[self.target_node_type]

        if self.mode == "neighbor":
            raw_seed_count = target_store.get("batch_size")
            if raw_seed_count is None:
                raise ValueError(
                    "Neighbor batch is missing target-store batch_size"
                )
            seed_count = int(raw_seed_count)
            if seed_count < 1 or seed_count > labels.size(0):
                raise ValueError(
                    f"Invalid target seed count: {seed_count}"
                )
            return SupervisedBatch(
                logits=logits[:seed_count],
                targets=labels[:seed_count],
                num_examples=seed_count,
            )

        mask_name = _PHASE_MASK[phase]
        mask = target_store[mask_name]
        selected = int(mask.sum().item())
        if selected == 0:
            raise ValueError(f"No supervised target nodes for {phase}")
        return SupervisedBatch(
            logits=logits[mask],
            targets=labels[mask],
            num_examples=selected,
        )
```

Keep `TBModel.process_outputs()` as the compatibility entry point, but make it
delegate:

```python
def process_outputs(self, model_out: dict, batch: Data | HeteroData) -> dict:
    supervised = self.supervision_adapter.select(
        model_out=model_out,
        batch=batch,
        phase=self.state_str,
    )
    model_out["logits"] = supervised.logits
    model_out["labels"] = supervised.targets
    model_out["num_supervised_examples"] = supervised.num_examples
    return model_out
```

In `TBModel.__init__`, initialize the fallback only after
`self.task_level = self.readout.task_level`, and add `supervision_adapter` to
`save_hyperparameters(..., ignore=[...])`. Have `model_step()` continue to call
`process_outputs()` at the same point. Use
`model_out["num_supervised_examples"]` as Lightning's `batch_size` for the
train/validation/test loss logs instead of hard-coded `1`.

For heterogeneous batches this produces correct weighted epoch aggregation.
For default graph and node-inductive tasks, the adapter deliberately reports
`1` to avoid changing existing metrics in this feature.

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

- Create: `topobench/nn/activation.py`
- Create: `topobench/nn/encoders/heterogeneous_node_encoder.py`
- Modify: `topobench/nn/__init__.py`
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

First centralize the activation names already supported by `CellHGT`:

```python
# topobench/nn/activation.py
import torch


def make_activation(name: str) -> torch.nn.Module:
    activations: dict[str, type[torch.nn.Module]] = {
        "relu": torch.nn.ReLU,
        "elu": torch.nn.ELU,
        "tanh": torch.nn.Tanh,
        "gelu": torch.nn.GELU,
        "id": torch.nn.Identity,
    }
    try:
        return activations[name]()
    except KeyError as error:
        raise ValueError(f"Unsupported activation: {name}") from error
```

Then implement the encoder with the existing TopoBench batch-in/batch-out
contract:

```python
from collections.abc import Mapping

import torch
from torch_geometric.data import HeteroData

from topobench.nn.activation import make_activation
from topobench.nn.encoders.base import AbstractFeatureEncoder


class HeterogeneousNodeFeatureEncoder(AbstractFeatureEncoder):
    def __init__(
        self,
        input_channels: Mapping[str, int],
        hidden_channels: int,
        activation: str = "relu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_channels < 1:
            raise ValueError("hidden_channels must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if not input_channels:
            raise ValueError("input_channels must not be empty")

        self.input_channels = {
            node_type: int(width)
            for node_type, width in input_channels.items()
        }
        if any(width < 1 for width in self.input_channels.values()):
            raise ValueError("Every heterogeneous input width must be positive")
        self.hidden_channels = hidden_channels
        self.projections = torch.nn.ModuleDict(
            {
                node_type: torch.nn.Linear(width, hidden_channels)
                for node_type, width in self.input_channels.items()
            }
        )
        self.activation = make_activation(activation)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, data: HeteroData) -> HeteroData:
        if not isinstance(data, HeteroData):
            raise TypeError("Heterogeneous encoder requires HeteroData")
        actual = set(data.x_dict)
        expected = set(self.input_channels)
        if actual != expected:
            raise ValueError(
                "Heterogeneous feature keys differ from validated metadata: "
                f"missing={sorted(expected - actual)}, "
                f"unexpected={sorted(actual - expected)}"
            )
        for node_type, expected_width in self.input_channels.items():
            features = data[node_type].x
            if features.ndim != 2 or features.size(-1) != expected_width:
                raise ValueError(
                    f"{node_type!r} expected feature width "
                    f"{expected_width}, received {tuple(features.shape)}"
                )
            data[node_type].x = self.dropout(
                self.activation(self.projections[node_type](features))
            )
        return data
```

Test `make_activation()` here as well. Task 11 replaces `CellHGT._activation`
with this helper while preserving its accepted legacy names.

Subclassing `AbstractFeatureEncoder` is required: the current
`topobench.nn.encoders` discovery code exports only subclasses of that base.
Add an assertion that
`topobench.nn.encoders.HeterogeneousNodeFeatureEncoder` resolves after a clean
module import.

**Step 4: Verify**

Run:

```bash
uv run pytest test/nn/encoders/test_heterogeneous_node_encoder.py -q
```

Expected: PASS.

**Step 5: Commit**

```bash
git add \
  topobench/nn/activation.py \
  topobench/nn/__init__.py \
  topobench/nn/encoders/heterogeneous_node_encoder.py \
  topobench/nn/encoders/__init__.py \
  test/nn/encoders/test_heterogeneous_node_encoder.py
git commit -m "feat: add heterogeneous node feature encoder"
```

## Task 11: Extract a Reusable HGT Backbone Without Breaking Cell HGT

**Files:**

- Create: `topobench/nn/backbones/heterogeneous/__init__.py`
- Create: `topobench/nn/backbones/heterogeneous/common.py`
- Create: `topobench/nn/backbones/heterogeneous/hgt.py`
- Modify: `topobench/nn/backbones/combinatorial/hgt.py:1-194`
- Modify: `topobench/nn/activation.py`
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
from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor
from torch_geometric.nn import HGTConv
from torch_geometric.typing import EdgeType, Metadata

from topobench.nn.activation import make_activation
from topobench.nn.backbones.heterogeneous.common import (
    validate_backbone_arguments,
    validate_forward_dictionaries,
)


class HGTBackbone(torch.nn.Module):
    def __init__(
        self,
        metadata: Metadata,
        hidden_channels: int,
        num_layers: int,
        heads: int,
        dropout: float,
        activation: str,
    ) -> None:
        super().__init__()
        node_types, edge_types = metadata
        validate_backbone_arguments(
            node_types=node_types,
            edge_types=edge_types,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            heads=heads,
            dropout=dropout,
        )
        self.node_types = list(node_types)
        self.edge_types = [tuple(edge_type) for edge_type in edge_types]
        self.metadata = (self.node_types, self.edge_types)
        self.hidden_channels = hidden_channels
        self.out_channels = hidden_channels
        self.num_layers = num_layers
        self.heads = heads
        self.dropout_probability = dropout
        self.activation_name = activation
        self.convs = torch.nn.ModuleList(
            [
                HGTConv(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    metadata=self.metadata,
                    heads=heads,
                )
                for _ in range(num_layers)
            ]
        )
        self.norms = torch.nn.ModuleList(
            [
                torch.nn.ModuleDict(
                    {
                        node_type: torch.nn.LayerNorm(hidden_channels)
                        for node_type in self.node_types
                    }
                )
                for _ in range(num_layers)
            ]
        )
        self.activation = make_activation(activation)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(
        self,
        x_dict: Mapping[str, Tensor],
        edge_index_dict: Mapping[EdgeType, Tensor],
    ) -> dict[str, Tensor]:
        validate_forward_dictionaries(
            x_dict=x_dict,
            edge_index_dict=edge_index_dict,
            node_types=self.node_types,
            edge_types=self.edge_types,
            hidden_channels=self.hidden_channels,
        )
        current = dict(x_dict)
        for conv, norms in zip(self.convs, self.norms, strict=True):
            updates = conv(current, edge_index_dict)
            current = {
                node_type: (
                    features
                    if updates.get(node_type) is None
                    else self.dropout(
                        self.activation(
                            norms[node_type](updates[node_type])
                        )
                    )
                )
                for node_type, features in current.items()
            }
        return current
```

Place `validate_backbone_arguments()` and
`validate_forward_dictionaries()` in
`topobench/nn/backbones/heterogeneous/common.py` so HGT and HeteroSAGE enforce
the identical metadata/input contract. `validate_backbone_arguments()` must
reject empty/duplicate node or edge
types, edges whose endpoints are absent, invalid widths/layers/heads/dropout,
and a width not divisible by heads.
`validate_forward_dictionaries()` must reject missing or unknown node types,
unknown edge types, non-`[2, E]` integer edge indices, and feature widths that
are not the common hidden width. It may accept a known relation absent from a
sampled batch; PyG sampling can produce an empty/omitted relation locally.

Each layer applies `HGTConv`, activation, normalization, and dropout
consistently. Preserve node types that receive no update in a layer through an
explicit carry-forward rule, and test that rule. Do not silently add a new
residual connection: that would change existing `CellHGT` behavior.

**Step 4: Adapt `CellHGT`**

Make `CellHGT` subclass the generic implementation. Compute its rank metadata
first, call `super().__init__(metadata=..., ...)`, and override only input/output
adaptation:

```python
class CellHGT(HGTBackbone):
    def __init__(
        self,
        hidden_channels: int,
        num_layers: int,
        heads: int,
        neighborhoods: list[str],
        max_rank: int = 2,
        dropout: float = 0.0,
        activation: str = "relu",
    ) -> None:
        # Keep all existing neighborhood and route validation verbatim.
        self.max_rank = max_rank
        self.neighborhoods = list(neighborhoods)
        self.routes = [...]
        node_types = [
            self.node_type(rank) for rank in range(max_rank + 1)
        ]
        edge_types = [...]
        super().__init__(
            metadata=(node_types, edge_types),
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            heads=heads,
            dropout=dropout,
            activation=activation,
        )

    def forward(self, batch: Data) -> dict[int, Tensor]:
        x_dict, edge_index_dict = self.to_heterogeneous_inputs(batch)
        output = super().forward(x_dict, edge_index_dict)
        return {
            rank: output[self.node_type(rank)]
            for rank in range(self.max_rank + 1)
        }
```

Retain:

- `edge_types`;
- `metadata`;
- `to_heterogeneous_inputs`;
- constructor arguments and config targets;
- state-dict compatibility where practical;
- current output structure by cell rank.

Subclassing keeps `convs.*` and `norms.*` at the state-dict root, preserving
current checkpoint keys; composition under `self.core` would rename them and
is therefore rejected. Add a regression test comparing the old expected key
prefixes (`convs.0`, `norms.0`) with the refactored model. Do not change the
existing cell-complex HGT experiment semantics.

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
  topobench/nn/activation.py \
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

- one eager
  `SAGEConv((hidden_channels, hidden_channels), hidden_channels)` exists per
  edge type per layer;
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

Build every relation module eagerly in `__init__` from metadata:

```python
from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor
from torch_geometric.nn import HeteroConv, SAGEConv
from torch_geometric.typing import EdgeType, Metadata

from topobench.nn.activation import make_activation
from topobench.nn.backbones.heterogeneous.common import (
    validate_backbone_arguments,
    validate_forward_dictionaries,
)


class HeteroSAGEBackbone(torch.nn.Module):
    def __init__(
        self,
        metadata: Metadata,
        hidden_channels: int,
        num_layers: int,
        dropout: float = 0.0,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        node_types, edge_types = metadata
        validate_backbone_arguments(
            node_types=node_types,
            edge_types=edge_types,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            dropout=dropout,
            heads=None,
        )
        self.node_types = list(node_types)
        self.edge_types = [tuple(edge_type) for edge_type in edge_types]
        self.metadata = (self.node_types, self.edge_types)
        self.hidden_channels = hidden_channels
        self.out_channels = hidden_channels
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList(
            [
                HeteroConv(
                    {
                        edge_type: SAGEConv(
                            (hidden_channels, hidden_channels),
                            hidden_channels,
                        )
                        for edge_type in self.edge_types
                    },
                    aggr="sum",
                )
                for _ in range(num_layers)
            ]
        )
        self.norms = torch.nn.ModuleList(
            [
                torch.nn.ModuleDict(
                    {
                        node_type: torch.nn.LayerNorm(hidden_channels)
                        for node_type in self.node_types
                    }
                )
                for _ in range(num_layers)
            ]
        )
        self.activation = make_activation(activation)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(
        self,
        x_dict: Mapping[str, Tensor],
        edge_index_dict: Mapping[EdgeType, Tensor],
    ) -> dict[str, Tensor]:
        validate_forward_dictionaries(
            x_dict=x_dict,
            edge_index_dict=edge_index_dict,
            node_types=self.node_types,
            edge_types=self.edge_types,
            hidden_channels=self.hidden_channels,
        )
        current = dict(x_dict)
        for conv, norms in zip(self.convs, self.norms, strict=True):
            updates = conv(current, edge_index_dict)
            current = {
                node_type: (
                    features
                    if updates.get(node_type) is None
                    else self.dropout(
                        self.activation(
                            norms[node_type](updates[node_type])
                        )
                    )
                )
                for node_type, features in current.items()
            }
        return current
```

Because the feature encoder already projects every type into
`hidden_channels`, use explicit
`SAGEConv((hidden_channels, hidden_channels), hidden_channels)` rather than
lazy `(-1, -1)` layers. This guarantees complete optimizer and checkpoint
state before the first forward pass.

For every layer:

1. run `HeteroConv`;
2. merge untouched node types from the previous dictionary;
3. apply per-type normalization, activation, and dropout;
4. return the complete typed dictionary.

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

Use these skeletons:

```python
# topobench/nn/wrappers/heterogeneous/heterogeneous_wrapper.py
from __future__ import annotations

import torch
from torch import Tensor
from torch_geometric.data import HeteroData


class HeterogeneousWrapper(torch.nn.Module):
    def __init__(
        self,
        backbone: torch.nn.Module,
        target_node_type: str,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.target_node_type = target_node_type

    def forward(self, batch: HeteroData) -> dict[str, object]:
        if not isinstance(batch, HeteroData):
            raise TypeError("HeterogeneousWrapper requires HeteroData")
        if self.target_node_type not in batch.node_types:
            raise ValueError(
                f"Missing target node type {self.target_node_type!r}"
            )
        labels = batch[self.target_node_type].get("y")
        if not isinstance(labels, Tensor):
            raise TypeError("Target node store must contain tensor y")
        x_dict = self.backbone(
            batch.x_dict,
            batch.edge_index_dict,
        )
        return {"x_dict": x_dict, "labels": labels}
```

```python
# topobench/nn/readouts/heterogeneous_node.py
from __future__ import annotations

import torch
from torch import Tensor
from torch_geometric.data import HeteroData


class HeterogeneousNodeReadout(torch.nn.Module):
    task_level = "node"

    def __init__(
        self,
        target_node_type: str,
        hidden_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()
        if hidden_channels < 1 or out_channels < 2:
            raise ValueError("Invalid heterogeneous readout dimensions")
        self.target_node_type = target_node_type
        self.linear = torch.nn.Linear(hidden_channels, out_channels)

    def forward(
        self,
        model_out: dict[str, object],
        batch: HeteroData,
    ) -> dict[str, object]:
        del batch  # Signature intentionally matches TBModel's readout call.
        x_dict = model_out.get("x_dict")
        if not isinstance(x_dict, dict):
            raise TypeError("model_out must contain an x_dict")
        embeddings = x_dict.get(self.target_node_type)
        if not isinstance(embeddings, Tensor):
            raise ValueError(
                f"No embeddings for target type {self.target_node_type!r}"
            )
        model_out["logits"] = self.linear(embeddings)
        return model_out
```

Do not subclass `AbstractWrapper` or `AbstractZeroCellReadOut`: both encode
rank-based `x_0`/`batch_0` assumptions. Reusing `TBModel`'s call contracts is
the correct shared boundary; inheriting the homogeneous abstractions would
introduce false coupling.

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
- Modify: `topobench/run.py:127-145`
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
- every `_target_` in both heterogeneous model configs resolves in a clean
  Python subprocess, exercising TopoBench's dynamic export registries.

**Step 2: Run and confirm failure**

Run:

```bash
uv run pytest test/utils/test_model_instantiation.py -q
```

Expected: FAIL.

**Step 3: Add one centralized construction helper**

Implement a helper such as:

```python
from __future__ import annotations

from copy import deepcopy

import hydra
from lightning import LightningModule
from omegaconf import DictConfig, open_dict


def instantiate_model(
    cfg: DictConfig,
    *,
    data_spec: HeterogeneousDataSpec | None,
) -> LightningModule:
    model_domain = str(cfg.model.get("model_domain", ""))
    if data_spec is None:
        if model_domain == "heterogeneous":
            raise ValueError(
                "A heterogeneous model requires a validated data specification"
            )
        return hydra.utils.instantiate(
            cfg.model,
            evaluator=cfg.evaluator,
            optimizer=cfg.optimizer,
            loss=cfg.loss,
        )
    if model_domain != "heterogeneous":
        raise ValueError(
            "A heterogeneous data specification requires a heterogeneous model"
        )

    runtime_cfg = deepcopy(cfg)
    model_cfg = runtime_cfg.model
    mode = str(runtime_cfg.dataset.dataloader_params.mode)
    _validate_sampling_depth(runtime_cfg)
    _require_runtime_placeholder(
        "model.feature_encoder.input_channels",
        model_cfg.feature_encoder.get("input_channels"),
    )
    _require_runtime_placeholder(
        "model.backbone.metadata",
        model_cfg.backbone.get("metadata"),
    )
    _require_runtime_placeholder(
        "model.backbone_wrapper.target_node_type",
        model_cfg.backbone_wrapper.get("target_node_type"),
    )
    _require_runtime_placeholder(
        "model.readout.target_node_type",
        model_cfg.readout.get("target_node_type"),
    )
    _require_runtime_placeholder(
        "model.readout.out_channels",
        model_cfg.readout.get("out_channels"),
    )
    _require_runtime_placeholder(
        "model.supervision_adapter.target_node_type",
        model_cfg.supervision_adapter.get("target_node_type"),
    )
    _require_runtime_placeholder(
        "model.supervision_adapter.mode",
        model_cfg.supervision_adapter.get("mode"),
    )

    with open_dict(model_cfg.feature_encoder):
        model_cfg.feature_encoder.input_channels = (
            data_spec.input_channels_dict
        )
    with open_dict(model_cfg.backbone):
        model_cfg.backbone.metadata = data_spec.pyg_metadata()
    with open_dict(model_cfg.backbone_wrapper):
        model_cfg.backbone_wrapper.target_node_type = (
            data_spec.target_node_type
        )
    with open_dict(model_cfg.readout):
        model_cfg.readout.target_node_type = data_spec.target_node_type
        model_cfg.readout.out_channels = data_spec.num_classes
    with open_dict(model_cfg.supervision_adapter):
        model_cfg.supervision_adapter.target_node_type = (
            data_spec.target_node_type
        )
        model_cfg.supervision_adapter.mode = mode

    return hydra.utils.instantiate(
        model_cfg,
        evaluator=runtime_cfg.evaluator,
        optimizer=runtime_cfg.optimizer,
        loss=runtime_cfg.loss,
    )
```

`_validate_sampling_depth()` returns immediately for `full_batch`. For
`neighbor` plus `subgraph_type="directional"`, collect the length of the
global fanout list or every relation-specific list and require every length to
equal `int(cfg.model.backbone.num_layers)`. Its error must print model depth,
observed fanout depths, and the remedy. Reject any other subgraph type in v1.

`_require_runtime_placeholder(path, value)` must raise when `value is not
None`; validated graph metadata and feature widths cannot be overridden by a
static config. Apply the same check to target-node and class-count placeholders
before injection.

Copy the entire config tree—not only `cfg.model`—before injecting runtime-only
values. Model interpolations such as `${model.feature_encoder.hidden_channels}`
are absolute and require the copied root context. Do not mutate Hydra's
composed source configuration. Only the helper knows how a data spec maps to
component arguments; `run.py`, models, and loaders must not duplicate that
mapping.

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

Use this complete shape for HGT:

```yaml
_target_: topobench.model.TBModel

model_name: hgt
model_domain: heterogeneous

feature_encoder:
  _target_: topobench.nn.encoders.HeterogeneousNodeFeatureEncoder
  input_channels: null
  hidden_channels: 64
  activation: relu
  dropout: 0.0

backbone:
  _target_: topobench.nn.backbones.heterogeneous.HGTBackbone
  metadata: null
  hidden_channels: ${model.feature_encoder.hidden_channels}
  num_layers: 2
  heads: 4
  dropout: 0.1
  activation: relu

backbone_wrapper:
  _target_: topobench.nn.wrappers.heterogeneous.HeterogeneousWrapper
  _partial_: true
  target_node_type: null

readout:
  _target_: topobench.nn.readouts.HeterogeneousNodeReadout
  target_node_type: null
  hidden_channels: ${model.feature_encoder.hidden_channels}
  out_channels: null

supervision_adapter:
  _target_: topobench.model.HeterogeneousNodeSupervisionAdapter
  target_node_type: null
  mode: null

compile: false
```

The HeteroSAGE config is identical except for:

```yaml
model_name: heterosage
backbone:
  _target_: topobench.nn.backbones.heterogeneous.HeteroSAGEBackbone
  metadata: null
  hidden_channels: ${model.feature_encoder.hidden_channels}
  num_layers: 2
  dropout: 0.1
  activation: relu
```

Do not include HGT's `heads` key in HeteroSAGE.

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

Use this full-batch experiment shape:

```yaml
# @package _global_
defaults:
  - override /data_pipeline: heterogeneous_node
  - override /dataset: heterogeneous/SyntheticHeterogeneous
  - override /model: heterogeneous/hgt  # heterosage in the second file
  - override /logger: heterogeneous_wandb
  - _self_

seed: 0
tags: [heterogeneous, synthetic, full_batch, hgt]
train: true
test: true

dataset:
  dataloader_params:
    mode: full_batch

optimizer:
  parameters:
    lr: 0.01
    weight_decay: 0.0

trainer:
  accelerator: cpu
  devices: 1
  min_epochs: 1
  max_epochs: 50
  check_val_every_n_epoch: 1
```

The HeteroSAGE file changes only its model override and tag. Test overrides
reduce `max_epochs`; production debug defaults remain useful from the CLI.

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

Add a focused test for `rerun_best_model_checkpoint()` using the real trained
model, real datamodule, real checkpoint, and a `MagicMock(spec=WandbLogger)`.
Assert its `log_metrics` calls contain at least one
`val_best_rerun/*` key and one `test_best_rerun/*` key. Mock only the external
logger sink, not training, validation, testing, or checkpoint loading.

**Step 3: Add an overfit-signal test**

For the small synthetic fixture, train 30–50 direct optimizer steps through
the real `TBModel` and fetch a fresh full-graph batch for every step (the
feature encoder intentionally updates the batch object). Compare evaluation
loss on fresh batches before and after and assert only
`final_loss < initial_loss`. Seed all RNGs and use dropout zero in this test.
This catches disconnected features or incorrect target selection without
asserting an unstable accuracy threshold.

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
    persistent_workers: false
    train_shuffle: true
    replace: false
    subgraph_type: directional
    filter_per_worker: false
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
- Modify: `pyproject.toml`
- Modify: `test/data/load/test_heterogeneous_dataset_loaders.py`
- Modify: `test/data/load/test_datasetloaders.py`
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

Use this loader:

```python
from omegaconf import DictConfig
from torch_geometric.data import Dataset
from torch_geometric.datasets import DBLP

from topobench.data.loaders.base import AbstractLoader


class DBLPDatasetLoader(AbstractLoader):
    def __init__(self, parameters: DictConfig) -> None:
        super().__init__(parameters)

    def load_dataset(self) -> Dataset:
        return DBLP(root=str(self.get_data_dir()))
```

Use these essential dataset values:

```yaml
loader:
  _target_: topobench.data.loaders.DBLPDatasetLoader
  parameters:
    data_domain: heterogeneous
    data_type: bibliographic
    data_name: DBLP
    data_dir: ${paths.data_dir}/${dataset.loader.parameters.data_domain}/${dataset.loader.parameters.data_type}

parameters:
  task: classification
  task_level: node
  target_node_type: author
  num_classes: 4
  loss_type: cross_entropy
  monitor_metric: accuracy

split_params:
  learning_setting: transductive
  source: official_masks

dataloader_params:
  mode: full_batch
  num_workers: 0
  pin_memory: false
  persistent_workers: false
```

The transform default is only:

```yaml
defaults:
  - data_manipulations@conference_features: heterogeneous_constant_features

conference_features:
  node_types: conference
  value: 1.0
  cat: false
```

Each DBLP experiment config must override the data pipeline, dataset, model,
and logger explicitly:

```yaml
# @package _global_
defaults:
  - override /data_pipeline: heterogeneous_node
  - override /dataset: heterogeneous/DBLP
  - override /model: heterogeneous/hgt  # heterosage in the second file
  - override /logger: heterogeneous_wandb
  - _self_

seed: 0
tags: [heterogeneous, DBLP, full_batch, hgt]

optimizer:
  parameters:
    lr: 0.005
    weight_decay: 0.001

trainer:
  accelerator: auto
  devices: 1
  min_epochs: 1
  max_epochs: 200
  check_val_every_n_epoch: 1
```

These are controlled DBLP starting values aligned with PyG's heterogeneous
DBLP examples, not claimed tuned values. The smoke test overrides
`max_epochs=1`; do not copy unrelated ZINC regression hyperparameters.

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
def test_real_dblp_smoke() -> None:
    ...
```

Register both markers in `[tool.pytest.ini_options]` and guard the module:

```toml
markers = [
    "integration: exercises external datasets or complete training flows",
    "download: may download data when explicitly enabled",
]
```

```python
if os.environ.get("TOPOBENCH_ALLOW_DOWNLOADS") != "1":
    pytest.skip(
        "Set TOPOBENCH_ALLOW_DOWNLOADS=1 to run real-data integration tests",
        allow_module_level=True,
    )
```

It should load/process DBLP, validate the spec, take one full-batch train step
for each model, and assert finite loss. The default suite must never initiate
a network download.

The existing config-enumerating loader test must also exclude
`DBLP.yaml` and `OGB_MAG.yaml` unless
`TOPOBENCH_ALLOW_DOWNLOADS=1`. Put that policy in one named set such as
`download_gated_datasets`; do not scatter filename conditions through test
methods. OGB-MAG is listed before its config is added so Task 18 cannot
accidentally enable a default multi-gigabyte download.

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
  pyproject.toml \
  test/data/load/test_heterogeneous_dataset_loaders.py \
  test/data/load/test_datasetloaders.py \
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

Use this loader:

```python
from omegaconf import DictConfig
from torch_geometric.data import Dataset
from torch_geometric.datasets import OGB_MAG

from topobench.data.loaders.base import AbstractLoader


class OGBMAGDatasetLoader(AbstractLoader):
    def __init__(self, parameters: DictConfig) -> None:
        super().__init__(parameters)

    def load_dataset(self) -> Dataset:
        return OGB_MAG(
            root=str(self.get_data_dir()),
            preprocess=str(self.parameters.preprocess),
        )
```

Start with this dataset configuration:

```yaml
loader:
  _target_: topobench.data.loaders.OGBMAGDatasetLoader
  parameters:
    data_domain: heterogeneous
    data_type: academic
    data_name: OGB_MAG
    data_dir: ${paths.data_dir}/${dataset.loader.parameters.data_domain}/${dataset.loader.parameters.data_type}
    preprocess: metapath2vec

parameters:
  task: classification
  task_level: node
  target_node_type: paper
  num_classes: 349
  loss_type: cross_entropy
  monitor_metric: accuracy
  evaluation_protocol: sampled_neighbor

split_params:
  learning_setting: transductive
  source: official_masks

dataloader_params:
  mode: neighbor
  batch_size: 128
  num_neighbors: [15, 10]
  num_workers: 4
  pin_memory: true
  persistent_workers: true
  train_shuffle: true
  replace: false
  subgraph_type: directional
  filter_per_worker: false
```

Use `ToUndirected(merge=False)` because raw OGB-MAG relations are directed. Record fanout, batch size, number of layers, preprocessing variant, and `evaluation_protocol: sampled` in W&B.

The transform default is:

```yaml
defaults:
  - data_manipulations@reverse_relations: heterogeneous_to_undirected

reverse_relations:
  merge: false
  reduce: add
```

Both experiment files override
`/data_pipeline: heterogeneous_node`, `/dataset: heterogeneous/OGB_MAG`,
their respective heterogeneous model, and
`/logger: heterogeneous_wandb`. Use tags such as
`[heterogeneous, OGB_MAG, neighbor, hgt, preflight]`.

Their shared runtime baseline is:

```yaml
optimizer:
  parameters:
    lr: 0.001
    weight_decay: 0.0001

trainer:
  accelerator: auto
  devices: 1
  min_epochs: 1
  max_epochs: 50
  check_val_every_n_epoch: 1
```

This is an operational starting point, not a tuned claim. The preflight and
bounded commands override `max_epochs`.

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

Apply the same registered `integration`/`download` markers and
`TOPOBENCH_ALLOW_DOWNLOADS=1` module guard as the DBLP integration test.

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

**Step 5: Prepare bounded Lightning smoke commands**

HGT:

```bash
uv run python -m topobench.run \
  experiment=heterogeneous_ogb_mag_hgt \
  seed=0 \
  trainer.min_epochs=1 \
  trainer.max_epochs=1 \
  trainer.limit_train_batches=10 \
  trainer.limit_val_batches=5 \
  trainer.limit_test_batches=5 \
  logger=heterogeneous_wandb \
  logger.wandb.name=ogb-mag-hgt-neighbor-bounded-seed0
```

HeteroSAGE:

```bash
uv run python -m topobench.run \
  experiment=heterogeneous_ogb_mag_heterosage \
  seed=0 \
  trainer.min_epochs=1 \
  trainer.max_epochs=1 \
  trainer.limit_train_batches=10 \
  trainer.limit_val_batches=5 \
  trainer.limit_test_batches=5 \
  logger=heterogeneous_wandb \
  logger.wandb.name=ogb-mag-heterosage-neighbor-bounded-seed0
```

Expected: bounded training plus bounded best-checkpoint validation/test rerun
under the sampled evaluation protocol. This checks Lightning, checkpoint, and
W&B integration without pretending to be a benchmark.

Only after the corresponding bounded command passes, launch a complete
one-epoch traversal by removing the three `limit_*_batches` overrides and
changing the W&B name from `bounded` to `epoch1`. A complete OGB-MAG
validation/test pass can be long even with batching; do not use the bounded
result as a performance result.

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
- uses `train=false test=false logger=null` for a fast, network-free
  configuration check.

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
