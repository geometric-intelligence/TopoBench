# Graph, Heterogeneous Graph, and Hypergraph Core Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reduce TopoBench to a high-quality native PyG core for homogeneous graphs, heterogeneous node classification, and lightweight hypergraph node classification, with no TopoModelX/TopoNetX dependency or rank-indexed runtime contract.

**Architecture:** Homogeneous graphs use PyG `Data`/`Batch` and `DataLoader`; heterogeneous graphs retain the existing `HeteroData` plus full-batch/`NeighborLoader` path; hypergraphs use a small `HypergraphData` subclass whose two incidence rows batch with independent node and hyperedge offsets. Each domain has an explicit data pipeline, model adapter, and validation boundary, while shared training, evaluation, callbacks, checkpoint reruns, and logging remain in `TBModel` and `topobench.run`.

**Tech Stack:** Python 3.11, PyTorch 2.3, PyTorch Geometric, Lightning 2.4, Hydra/OmegaConf, pytest, Ruff, uv.

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

Expected: all selected tests pass; tests that require optional sampler extensions may skip with their existing reason.

## Target runtime contracts

Homogeneous `Data`/`Batch`:

```text
x, edge_index, edge_attr?, edge_weight?, y,
batch (inductive batches), train_mask/val_mask/test_mask (transductive)
```

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

Assert that registries are ordinary explicit dictionaries, are deterministically ordered, and expose only classes from the three target domains. Include the current surviving classes first; add `HypergraphConvBackbone` only in Task 9.

```python
from topobench.nn import backbones
from topobench.nn import wrappers


def test_backbone_registry_has_only_surviving_local_models() -> None:
    assert tuple(backbones.MODEL_CLASSES) == tuple(sorted(backbones.MODEL_CLASSES))
    assert set(backbones.MODEL_CLASSES) == {
        "EDGNN", "GPS", "GraphMLP", "HGTBackbone",
        "HeteroSAGEBackbone", "IdentityGNN", "NSD",
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
from .graph import GPS, GraphMLP, IdentityGNN, NSD
from .heterogeneous import HGTBackbone, HeteroSAGEBackbone
from .hypergraph import EDGNN

MODEL_CLASSES = dict(sorted({
    cls.__name__: cls
    for cls in (GPS, GraphMLP, IdentityGNN, NSD, HGTBackbone, HeteroSAGEBackbone, EDGNN)
}.items()))

__all__ = [*MODEL_CLASSES, "MODEL_CLASSES"]
globals().update(MODEL_CLASSES)
```

For loaders and datasets, import every class referenced by a surviving YAML `_target_`; do not construct targets by scanning filenames. Preserve public selector class names used by surviving configs.

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

### Task 3: Introduce native homogeneous graph splitting and batching

**Files:**
- Create: `topobench/dataloader/graph.py`
- Create: `topobench/data/datasets/synthetic_graph_dataset.py`
- Create: `topobench/data/loaders/graph/synthetic.py`
- Create: `configs/dataset/graph/SyntheticGraph.yaml`
- Modify: `topobench/dataloader/__init__.py`
- Modify: `topobench/data/utils/split_utils.py`
- Modify: `topobench/data/preprocessor/preprocessor.py`
- Modify: `topobench/data/pipelines/default.py`
- Replace tests in: `test/data/dataload/test_Dataloaders.py`
- Replace tests in: `test/data/dataload/test_dataload_dataset.py`
- Modify: `test/data/utils/test_split_utils.py`
- Modify: `test/data/pipelines/test_data_pipelines.py`
- Create: `test/data/datasets/test_synthetic_graph_dataset.py`

**Step 1: Write failing inductive batching tests**

Construct two small `Data` graphs, pass lists into `GraphDataModule`, and assert the loader returns a PyG `Batch` with native fields:

```python
def test_graph_datamodule_uses_native_pyg_batching() -> None:
    module = GraphDataModule(
        dataset_train=[graph_a, graph_b],
        dataset_val=[graph_a],
        dataset_test=[graph_b],
        batch_size=2,
        num_workers=0,
    )
    batch = next(iter(module.train_dataloader()))
    assert isinstance(batch, Batch)
    assert batch.num_graphs == 2
    assert batch.batch.tolist() == [0] * graph_a.num_nodes + [1] * graph_b.num_nodes
    assert "x_0" not in batch and "batch_0" not in batch
```

Add a transductive test proving one graph is reused for all phases and `batch_size != 1` is rejected. Add split tests proving inductive helpers return `list[Data]`, not `DataloadDataset`.

**Step 2: Verify failure**

Run:

```bash
uv run pytest test/data/dataload/test_Dataloaders.py \
  test/data/dataload/test_dataload_dataset.py \
  test/data/utils/test_split_utils.py -q
```

Expected: FAIL because `GraphDataModule` is absent and splits return `DataloadDataset`.

**Step 3: Implement `GraphDataModule` with PyG loaders**

Core skeleton:

```python
from collections.abc import Sequence
from lightning import LightningDataModule
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


class GraphDataModule(LightningDataModule):
    def __init__(self, dataset_train: Sequence[Data], dataset_val=None,
                 dataset_test=None, batch_size: int = 1, num_workers: int = 0,
                 pin_memory: bool = False, persistent_workers: bool = False) -> None:
        super().__init__()
        if not dataset_train:
            raise ValueError("dataset_train must not be empty")
        if (dataset_val is None) != (dataset_test is None):
            raise ValueError("dataset_val and dataset_test must either both be set or both be None")
        if dataset_val is None and batch_size != 1:
            raise ValueError("transductive graph loading requires batch_size=1")
        self.dataset_train = list(dataset_train)
        self.dataset_val = self.dataset_train if dataset_val is None else list(dataset_val)
        self.dataset_test = self.dataset_train if dataset_test is None else list(dataset_test)
        self.batch_size = batch_size
        self.loader_kwargs = dict(num_workers=num_workers, pin_memory=pin_memory,
                                  persistent_workers=persistent_workers and num_workers > 0)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset_train, batch_size=self.batch_size,
                          shuffle=True, **self.loader_kwargs)
```

Implement validation for positive integral `batch_size` and non-negative integral `num_workers`. Validation and test loaders use `shuffle=False`.

**Step 4: Return native sequences from split helpers**

- `assign_train_val_test_mask_to_graphs()` returns three lists.
- `load_transductive_splits()` returns `([data], None, None)`.
- `load_inductive_splits()` returns three lists.
- Preserve fixed/random/stratified/k-fold behavior and existing masks.
- Change `PreProcessor.load_dataset_splits()` annotations to sequences of `Data`.
- Change `DefaultDataPipeline` to instantiate `GraphDataModule`.

Add a deterministic, packaged `SyntheticGraphDataset` as a tiny native PyG
`InMemoryDataset`. It must contain several graph-classification examples so
the default CLI exercises real inductive batching without a download. Its
loader must return the dataset directly, and `SyntheticGraph.yaml` must use a
fixed train/validation/test split plus `batch_size: 4`. Do not construct this
dataset through `DataloadDataset` or any transform.

Do not delete the legacy dataloader files until Task 12; this keeps the diff reviewable while all callers migrate.

**Step 5: Run focused and pipeline-boundary tests**

Run:

```bash
uv run pytest test/data/dataload/test_Dataloaders.py \
  test/data/dataload/test_dataload_dataset.py \
  test/data/utils/test_split_utils.py \
  test/data/pipelines/test_data_pipelines.py \
  test/data/datasets/test_synthetic_graph_dataset.py -q
```

Expected: PASS.

**Step 6: Commit**

```bash
git add topobench/dataloader topobench/data/utils/split_utils.py \
  topobench/data/preprocessor/preprocessor.py topobench/data/pipelines/default.py \
  test/data/dataload test/data/utils/test_split_utils.py \
  test/data/pipelines/test_data_pipelines.py
git commit -m "refactor: use native PyG graph batching"
```

### Task 4: Migrate homogeneous feature encoding to `data.x`

**Files:**
- Create: `topobench/nn/encoders/graph_node_encoder.py`
- Modify: `topobench/nn/encoders/__init__.py`
- Create: `test/nn/encoders/test_graph_node_encoder.py`
- Modify: `test/nn/encoders/test_dgm.py`
- Modify: `topobench/nn/encoders/dgm_encoder.py`

**Step 1: Write failing encoder contract tests**

Test eager parameter creation, graph-aware normalization, shape validation, missing `batch` fallback for a single graph, and in-place `data.x` replacement. Explicitly assert no `x_0` is created.

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

**Step 4: Run encoder tests**

Run:

```bash
uv run pytest test/nn/encoders/test_graph_node_encoder.py \
  test/nn/encoders/test_dgm.py \
  test/nn/encoders/test_heterogeneous_node_encoder.py -q
```

Expected: PASS.

**Step 5: Commit**

```bash
git add topobench/nn/encoders test/nn/encoders
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
- Modify: `test/nn/readouts/test_identical.py`
- Modify: `test/nn/readouts/test_mlp_readout.py`

**Step 1: Write failing node- and graph-level adapter tests**

Require `GNNWrapper` to return `{"x", "labels", "batch"}` and graph readouts to consume that contract. Cover a batched graph-classification example and a single-graph node-classification example.

```python
def test_gnn_wrapper_uses_native_graph_fields() -> None:
    wrapped = GNNWrapper(RecordingBackbone())
    out = wrapped(batch)
    assert set(out) == {"x", "labels", "batch"}
    assert out["labels"] is batch.y
    assert torch.equal(out["batch"], batch.batch)
```

Add negative tests for missing/invalid `x`, `edge_index`, `y`, and batched graph-level data without `batch`.

**Step 2: Verify failure**

Run:

```bash
uv run pytest test/nn/wrappers/graph/test_graph_wrappers.py \
  test/nn/readouts/test_identical.py test/nn/readouts/test_mlp_readout.py -q
```

Expected: FAIL because current components use rank-indexed keys.

**Step 3: Make graph wrappers independent of `AbstractWrapper`**

Use a plain `torch.nn.Module`; the wrapper owns only backbone argument translation:

```python
class GNNWrapper(torch.nn.Module):
    def __init__(self, backbone: torch.nn.Module) -> None:
        super().__init__()
        self.backbone = backbone

    def forward(self, batch: Data) -> dict[str, Tensor | None]:
        x = self.backbone(
            batch.x,
            batch.edge_index,
            batch=batch.get("batch"),
            edge_weight=batch.get("edge_weight"),
        )
        return {"x": x, "labels": batch.y, "batch": batch.get("batch")}
```

Adapt `GraphMLPWrapper` to the same output contract. Keep residual connections inside a graph-specific wrapper only if a focused test demonstrates an existing surviving model needs them; do not retain rank iteration.

**Step 4: Refactor the graph readout base**

Rename internal concepts from zero-cells to nodes. `NoReadOut` remains the configured public class name, but computes logits from `model_out["x"]`. Graph pooling uses `model_out["batch"]` and `torch_geometric.utils.scatter`. Node and node-inductive tasks apply the linear head without pooling. `MLPReadout` follows the same keys.

The heterogeneous readout remains untouched because it already consumes `x_dict`.
Refactor `MLPReadout` to own its `torch.nn.Sequential` layers instead of
importing `topobench.nn.backbones.non_relational.MLP`; this makes deletion of
the non-relational domain in Task 12 safe.

**Step 5: Run focused, model, and supervision tests**

Run:

```bash
uv run pytest test/nn/wrappers/graph test/nn/readouts \
  test/model/test_model.py test/model/test_supervision.py -q
```

Expected: PASS after updating graph-only fixtures; heterogeneous readout tests remain green.

**Step 6: Commit**

```bash
git add topobench/nn/wrappers/graph topobench/nn/readouts \
  test/nn/wrappers/graph test/nn/readouts test/model
git commit -m "refactor: use native graph model outputs"
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
- Modify: `topobench/utils/config_resolvers.py`
- Modify: `topobench/utils/model_instantiation.py`
- Create: `test/config/test_surviving_graph_configs.py`
- Modify: `test/pipeline/test_pipeline.py`

**Step 1: Write failing composition and instantiation tests**

Parametrize over the seven surviving graph configs. Compose each with `graph/MUTAG`, resolve the config, and assert:

- `model.model_domain == "graph"`;
- feature encoder is `GraphNodeFeatureEncoder`;
- no resolved key or string contains `num_cell_dimensions`, `AllCellFeatureEncoder`, `x_0`, or a lifting target;
- model instantiation succeeds without a runtime `data_spec`.

Use `OmegaConf.to_container(cfg, resolve=True)` and a recursive key/string walker; do not rely on YAML text grep alone.

**Step 2: Verify failure**

Run: `uv run pytest test/config/test_surviving_graph_configs.py -q`

Expected: FAIL because graph configs still select `AllCellFeatureEncoder` and rank fields.

**Step 3: Rewrite surviving graph model configs**

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

readout:
  _target_: topobench.nn.readouts.NoReadOut
  hidden_dim: ${model.feature_encoder.out_channels}
  out_channels: ${dataset.parameters.num_classes}
  task_level: ${define_task_level:${dataset.parameters.task_level},${dataset.split_params.learning_setting}}
  pooling_type: sum
```

Keep each backbone's actual model-specific fields. Remove `wrapper_name`, `readout_name`, `num_cell_dimensions`, and interpolation through those names unless Hydra requires the public selector for an existing override.

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

**Step 5: Make the small native graph pipeline the broad graph sentinel**

Change `test/pipeline/test_pipeline.py` to cover `graph/gcn` on
`graph/SyntheticGraph` only, two epochs, CPU, a real batch size greater than
one, and final test. Keep a separate optional/download-marked MUTAG smoke test
for real-loader coverage. Add separate model-config unit coverage instead of
looping large/downloaded models in the lifecycle test.

**Step 6: Run configuration and graph model tests**

Run:

```bash
uv run pytest test/config/test_surviving_graph_configs.py \
  test/nn/backbones/graph test/pipeline/test_pipeline.py -q
```

Expected: PASS; the lifecycle test reports an observed training batch size greater than one.

**Step 7: Commit**

```bash
git add configs/model/graph topobench/utils test/config \
  test/pipeline/test_pipeline.py test/nn/backbones/graph
git commit -m "refactor: migrate graph configs to native PyG"
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

Add validation failures for wrong shape/dtype, negative indices, out-of-bounds nodes, missing/invalid `num_hyperedges`, non-contiguous hyperedge IDs, label count mismatch, overlapping or incomplete masks, and empty supervised splits.

**Step 2: Verify failure**

Run: `uv run pytest test/data/test_hypergraph_data.py -q`

Expected: FAIL because `HypergraphData` does not exist.

**Step 3: Implement `HypergraphData.__inc__`**

```python
class HypergraphData(Data):
    def __inc__(self, key: str, value: Tensor, *args, **kwargs):
        if key == "hyperedge_index":
            if self.num_nodes is None or self.num_hyperedges is None:
                raise ValueError("hypergraph batching requires node and hyperedge counts")
            return torch.tensor([[self.num_nodes], [self.num_hyperedges]], device=value.device)
        return super().__inc__(key, value, *args, **kwargs)
```

**Step 4: Implement transactional validation**

`validate_hypergraph_node_data(data)` returns the same object only after all checks pass. It must not mutate or renumber malformed input. Require a rank-2 floating `x`, long `hyperedge_index` of shape `[2, M]`, contiguous hyperedge IDs `0..num_hyperedges-1`, rank-1 node labels, and rank-1 boolean masks whose length equals `num_nodes` and which partition labeled nodes.

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
- Create: `topobench/data/utils/graph_io.py`
- Create: `topobench/data/utils/hypergraph_io.py`
- Modify: `topobench/data/utils/__init__.py`
- Modify: `topobench/data/datasets/citation_hypergraph_dataset.py`
- Modify: `topobench/data/datasets/hypergraph_datasets.py`
- Modify: `topobench/data/loaders/hypergraph/citation_hypergraph_dataset_loader.py`
- Modify: `topobench/data/loaders/hypergraph/hypergraph_dataset_loader.py`
- Modify: `test/data/utils/test_io_utils.py`
- Create: `test/data/load/test_hypergraph_dataset_loaders.py`

**Step 1: Write failing parser tests using local temporary raw fixtures**

Cover both content-style and pickle-style hypergraph inputs without downloading. Require each parser to return `HypergraphData` with canonical, sorted, duplicate-free `hyperedge_index`, contiguous hyperedge IDs, and explicit `num_hyperedges`.

Include a regression test where raw hyperedge identifiers are sparse strings or integers; the parser must remap them deterministically while leaving node indices aligned with features and labels.

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
`downloads.py`, and move the native `read_us_county_demos` parser into
`graph_io.py`. Update surviving dataset modules to import the narrow module
that owns the symbol. This permits complete deletion of the mixed
`io_utils.py`, whose module-level TopoModelX/TopoNetX imports currently poison
otherwise native graph and hypergraph loaders.

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

Make dataset `process()` methods save the canonical class through PyG's normal `InMemoryDataset.save()`/`load()` path. Remove backward handling for rank-based processed artifacts; users must regenerate processed hypergraph caches on this branch.

**Step 5: Run parser and loader tests**

Run:

```bash
uv run pytest test/data/utils/test_io_utils.py \
  test/data/load/test_hypergraph_dataset_loaders.py \
  test/data/test_hypergraph_data.py -q
```

Expected: PASS, with no network access.

**Step 6: Commit**

```bash
git add topobench/data/utils topobench/data/datasets \
  topobench/data/loaders/hypergraph test/data
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

Mock the loader with one `HypergraphData`, verify the pipeline validates it, and require train/validation/test loaders to return native `Batch` objects. Verify phase masks remain on the batch and `batch_size=1` is enforced for this transductive v1.

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
3. call `validate_hypergraph_node_data`;
4. build the hypergraph data module;
5. return `DataPipelineOutput` with no heterogeneous `data_spec`.

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

Require both backbones to accept `(x, hyperedge_index)` and return node embeddings of shape `[num_nodes, hidden_channels]`. Require `HypergraphWrapper` to emit the same `{"x", "labels", "batch"}` contract as graph wrappers. Add invalid incidence and output-shape tests.

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

Pass `num_edges=num_hyperedges` only if required by the pinned PyG signature; confirm against the installed 2.8.0 API during implementation.

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
AllEncodings, CombinedFEs, CombinedPSEs,
ElectrostaticPE, HeterogeneousConstantFeatures, HeterogeneousToUndirected,
HKFE, HKdiagSE, Identity, InfereKNNConnectivity, InfereRadiusConnectivity,
KeepOnlyConnectedComponent, KeepSelectedDataFields,
KeepSelectedTargetIndices, KHopFE, LapPE, NodeDegrees,
NodeFeaturesToFloat, OneHotDegreeFeatures, PPRFE, RWSE,
RenameFields, SheafConnLapPE
```

Use the exact current registered class names discovered from the modules; correct the list in the test to those spellings before implementation. Explicitly reject every lifting and: barycentric subdivision, simplicial curvature, HOPSE preprocessing, SANN feature generators, rank feature duplication/concatenation, and hypergraph incidence homophily transforms that still expect `incidence_hyperedges`.

**Step 2: Verify failure**

Run: `uv run pytest test/architecture/test_transform_registry.py -q`

Expected: FAIL because discovery exposes topological transforms and liftings.

**Step 3: Build an explicit transform map**

Replace discovery with direct imports and a sorted `TRANSFORMS` dictionary. Remove `LIFTINGS` and `FEATURE_LIFTINGS` exports. Make `DataTransform` accept `Data` (including `HypergraphData`) and `HeteroData`; preserve the explicit `supports_heterodata` opt-in.

`AddGPSEInformation` is removed: its neighborhood-route input is the
rank-based HOPSE contract, not a native graph transform.

**Step 4: Delete transforms and configs outside the allowlist**

Use `git rm` for the two entire topological transform trees and their tests. Remove corresponding Hydra configs plus `combined_fe.yaml`, `combined_pe.yaml`, `sheaf_pe.yaml`, or `custom_example.yaml` only when their contained transform has been rejected. Preserve dataset/model defaults required by GPS/NSD and all three heterogeneous defaults.

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

### Task 12: Delete unsupported source, models, configs, and tests

**Files:**
- Delete: `topobench/data/loaders/pointcloud/`
- Delete: `topobench/data/loaders/simplicial/`
- Delete: `topobench/data/loaders/graph/mantra_dataset.py`
- Delete: `topobench/data/loaders/graph/manual_graph_dataset_loader.py`
- Delete: `topobench/data/datasets/mantra_dataset.py`
- Delete: `topobench/data/utils/utils.py`
- Delete: `topobench/data/utils/io_utils.py`
- Delete: `topobench/nn/backbones/cell/`
- Delete: `topobench/nn/backbones/combinatorial/`
- Delete: `topobench/nn/backbones/non_relational/`
- Delete: `topobench/nn/backbones/simplicial/`
- Delete: `topobench/nn/wrappers/cell/`
- Delete: `topobench/nn/wrappers/combinatorial/`
- Delete: `topobench/nn/wrappers/pointcloud/`
- Delete: `topobench/nn/wrappers/simplicial/`
- Delete obsolete files under: `topobench/nn/encoders/`
- Delete obsolete files under: `topobench/nn/readouts/`
- Delete: `topobench/dataloader/dataload_dataset.py`
- Delete: `topobench/dataloader/dataloader.py`
- Delete: `topobench/dataloader/utils.py`
- Delete: `configs/dataset/pointcloud/`
- Delete: `configs/dataset/simplicial/`
- Delete: `configs/dataset/graph/manual_dataset.yaml`
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
- Delete corresponding removed-domain tests under: `test/`
- Modify: `test/architecture/test_domain_contract.py`

**Step 1: Extend the architecture test before deletion**

Assert exact allowed first-level domain directories for loaders, backbones, wrappers, dataset configs, and model configs. Walk production Python and surviving YAML text and reject:

```python
FORBIDDEN_TOKENS = (
    "x_0", "batch_0", "incidence_hyperedges", "num_cell_dimensions",
    "topomodelx", "toponetx", "gudhi", "hypernetx",
)
```

Allow those words only in historical `docs/plans/`, not production, configs, current docs, tests, or scripts.

**Step 2: Verify failure**

Run: `uv run pytest test/architecture/test_domain_contract.py -q`

Expected: FAIL and list the still-present unsupported paths/tokens.

**Step 3: Remove unsupported code in one auditable deletion**

Use explicit `git rm` paths from the file list. In encoders remove `all_cell_encoder.py`, `flat_encoder.py`, `hopse_encoder.py`, and `kdgm.py` if no surviving graph config imports them. In readouts remove `hopse.py` and `propagate_signal_down.py`; remove any non-relational backbone import after confirming `GraphMLP` has its own required implementation.

Delete tests only when the corresponding production behavior was intentionally removed. Never delete a failing test for a surviving graph, heterogeneous, hypergraph, shared trainer, callback, loss, evaluator, optimizer, or logger behavior.

**Step 4: Clean all `__init__.py` and imports**

Run `rg` for each removed module name and update remaining exports. Ensure `topobench.data.loaders`, `topobench.nn.backbones`, and `topobench.nn.wrappers` import from the three domains only.

**Step 5: Run architecture and surviving unit suites**

Run:

```bash
uv run pytest test/architecture -q
uv run pytest test/data test/nn test/model test/loss test/evaluator test/optimizer -q
```

Expected: PASS, excluding only explicitly marked download/integration tests.

**Step 6: Commit**

```bash
git add -A topobench configs test
git commit -m "refactor: remove unsupported topological domains"
```

### Task 13: Remove topological dependencies and regenerate the lock

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Create: `test/dependencies/test_reduced_dependencies.py`
- Modify: `test/dependencies/test_torch_geometric_dependency.py`
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

**Step 2: Verify failure**

Run: `uv run pytest test/dependencies -q`

Expected: FAIL because forbidden direct dependencies remain.

**Step 3: Audit every remaining dependency before editing**

For each candidate run `rg -n` over `topobench`, scripts, and current tests. Remove the six forbidden dependencies unconditionally after source pruning. Also remove `networkx`, `matplotlib`, `decorator`, `yacs`, `einops`, `tabulate`, `pandas`, `torch-cluster`, or `torch-sparse` only if no surviving production path uses them. Keep `torch-scatter`/`torch-sparse` if NSD or EDGNN still imports them.

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
uv run pytest test/dependencies test/architecture -q
```

Expected: PASS.

**Step 6: Update CI caches/commands only as required**

Keep CI using the regenerated lock and the supported Python 3.11 environment. Remove install steps for deleted topology packages; do not weaken the existing test invocation.

**Step 7: Commit**

```bash
git add pyproject.toml uv.lock test/dependencies .github/workflows
git commit -m "build: remove topological dependencies"
```

### Task 14: Make configs and the CLI a coherent three-domain product

**Files:**
- Modify: `configs/run.yaml`
- Modify: `configs/experiment/example.yaml`
- Preserve and verify: `configs/experiment/heterogeneous_*.yaml`
- Create: `configs/experiment/hypergraph_synthetic_edgnn.yaml`
- Create: `configs/experiment/hypergraph_synthetic_hypergraph_conv.yaml`
- Modify: `topobench/run.py` only if domain validation is not already centralized
- Create: `test/config/test_all_surviving_configs.py`
- Modify: `test/pipeline/test_heterogeneous_pipeline.py`
- Modify: `test/pipeline/test_hypergraph_pipeline.py`
- Modify: `test/callbacks/test_best_epoch_metrics.py`

**Step 1: Write a failing full config-tree composition test**

Discover YAML selectors from only these directories:

```text
configs/dataset/{graph,heterogeneous,hypergraph}
configs/model/{graph,heterogeneous,hypergraph}
configs/experiment
```

Compose valid same-domain dataset/model pairs, resolve interpolation, and instantiate components that do not require downloads. Assert every experiment refers only to an existing selector. Assert cross-domain pairs fail with the clear resolver error from Task 6.

**Step 2: Verify failure**

Run: `uv run pytest test/config/test_all_surviving_configs.py -q`

Expected: FAIL because defaults and experiments still mention removed domains.

**Step 3: Set a network-free graph default**

Use the packaged `graph/SyntheticGraph` plus `graph/gcn`. Do not make default
`python -m topobench.run` download MUTAG and do not retain the legacy manual
loader merely to serve the default.

Set `data_pipeline: default`, `transforms: no_transform`, and retain `train: true`, `test: true`. The default must still execute the best-checkpoint validation/test rerun in `topobench.run`.

**Step 4: Rewrite experiments**

- Make `configs/experiment/example.yaml` a native graph example.
- Preserve all eight heterogeneous experiments and their meaningful names.
- Add small hypergraph EDGNN and HypergraphConv experiments using
  `hypergraph/SyntheticHypergraph`; both explicitly override
  `/data_pipeline: hypergraph_node`.
- Remove cell-HGT, HOPSE, SANN, TopoTune, simplicial, and combinatorial experiments.

**Step 5: Verify final evaluation behavior in every domain**

Add a shared callback/run test that one tiny graph, heterogeneous, and hypergraph run each execute best-checkpoint reruns and emit `val_best_rerun/` and `test_best_rerun/` metrics. Mock W&B; do not make the test contact the service.

**Step 6: Run configuration, pipeline, and callback suites**

Run:

```bash
uv run pytest test/config test/pipeline \
  test/callbacks/test_best_epoch_metrics.py -q
```

Expected: PASS.

**Step 7: Commit**

```bash
git add -A configs topobench/run.py test/config test/pipeline \
  test/callbacks/test_best_epoch_metrics.py
git commit -m "refactor: focus CLI on native graph domains"
```

### Task 15: Rewrite current documentation and remove obsolete examples

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
4. heterogeneous full and neighbor-sampled commands;
5. hypergraph EDGNN and HypergraphConv commands;
6. batching semantics by domain;
7. best-checkpoint rerun and W&B metric names;
8. adding a dataset/model using the explicit registries.

Keep `docs/heterogeneous_graphs.md` detailed and update only links/scope. `docs/hypergraphs.md` must document the exact incidence convention and cache incompatibility. `docs/graph_data.md` must document native `x`/`batch` fields.

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

### Task 16: Final clean-environment and lifecycle verification

**Files:**
- Modify only if a genuine defect is found: relevant surviving source/test file
- Create: `test/architecture/verify_forbidden_imports.py`

**Step 1: Run static residue checks**

Run:

```bash
rg -n "x_0|batch_0|incidence_hyperedges|num_cell_dimensions" \
  topobench configs test scripts docs \
  -g '!docs/plans/**'
rg -n "topomodelx|toponetx|gudhi|hypernetx" \
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

**Step 4: Run three end-to-end smoke tests**

Run the small graph, synthetic heterogeneous neighbor-batched, and synthetic hypergraph experiments on CPU with W&B disabled:

```bash
WANDB_MODE=disabled uv run python -m topobench.run \
  experiment=example trainer.accelerator=cpu trainer.devices=1 trainer.max_epochs=1

WANDB_MODE=disabled uv run python -m topobench.run \
  experiment=heterogeneous_synthetic_hgt_neighbor \
  trainer.accelerator=cpu trainer.devices=1 trainer.max_epochs=1

WANDB_MODE=disabled uv run python -m topobench.run \
  experiment=hypergraph_synthetic_hypergraph_conv \
  trainer.accelerator=cpu trainer.devices=1 trainer.max_epochs=1
```

Expected for each: training completes, a best checkpoint is selected, and both `val_best_rerun/` and `test_best_rerun/` metrics are produced. The heterogeneous neighbor run must report a target seed batch size greater than one.

**Step 5: Verify clean forbidden imports in a subprocess**

`test/architecture/verify_forbidden_imports.py` installs a meta-path finder that raises for the four removed module roots, imports every public TopoBench package, composes the default plus one config per domain, and exits zero. Run:

```bash
uv run python test/architecture/verify_forbidden_imports.py
```

Expected: exit 0 with a concise `clean import verified` message.

**Step 6: Inspect repository state and commit final fixes**

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

- homogeneous inductive training uses real PyG mini-batches and native `x`/`batch`;
- heterogeneous neighbor sampling and full-batch modes retain their current tests and lifecycle behavior;
- hypergraphs batch correctly with independent node/hyperedge offsets and run both EDGNN and HypergraphConv;
- only graph, heterogeneous, and hypergraph source/config groups remain;
- no surviving runtime uses a rank-indexed field or a lifting;
- TopoModelX, TopoNetX, GUDHI, and HyperNetX are absent from source, direct dependencies, and lockfile;
- the default CLI is network-free and executes final best-checkpoint evaluation;
- the complete network-free suite, Ruff, clean-import probe, and all three end-to-end smokes pass.
