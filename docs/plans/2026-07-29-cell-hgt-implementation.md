# Batched Cell-Complex HGT Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a batched PyG HGT backbone for rank-0, rank-1, and rank-2 cells, prove its correctness on synthetic and collated complexes, debug it on MUTAG and PROTEINS, and prepare a reproducible ZINC benchmark.

**Architecture:** Keep TopoBench's existing graph-to-cell lifting, `DomainData` batching, feature encoder, wrapper, and readout. A new `CellHGT` backbone converts batched `x_0/x_1/x_2` tensors and four unsigned incidence neighborhoods to PyG `x_dict` and `edge_index_dict`, applies stacked `HGTConv` layers, and returns rank-keyed tensors without iterating over individual graphs.

**Tech Stack:** Python 3.11, PyTorch 2.3, PyTorch Geometric `HGTConv`, Hydra/OmegaConf, Lightning, pytest, Ruff, TopoBench `CellCycleLifting`, and `uv`.

---

## Execution rules

- Work on `main`, as explicitly requested by the user.
- Use @test-driven-development for Tasks 2–7.
- Use @systematic-debugging if a test fails for a reason other than the
  expected missing behavior.
- Use @verification-before-completion before the final commit.
- Do not copy or port `acbull/pyHGT`; use PyG's maintained `HGTConv`.
- Do not convert examples to `HeteroData` and do not loop over graphs in
  `CellHGT.forward`.
- Version 1 accepts only unsigned incidence neighborhoods.
- Do not start the full ZINC run until every promotion gate in Task 7 passes.
- The repository currently loads the standard 12k-example ZINC subset because
  `MoleculeDatasetLoader` passes `subset=True`.

The approved design is in
`docs/plans/2026-07-29-cell-hgt-design.md`.

## Fixed relation schema

| TopoBench field | HGT edge type | Direction after conversion |
|---|---|---|
| `up_incidence-0` | `("rank_0", "up_incidence-0", "rank_1")` | vertex → edge |
| `down_incidence-1` | `("rank_1", "down_incidence-1", "rank_0")` | edge → vertex |
| `up_incidence-1` | `("rank_1", "up_incidence-1", "rank_2")` | edge → face |
| `down_incidence-2` | `("rank_2", "down_incidence-2", "rank_1")` | face → edge |

TopoBench stores sparse neighborhood rows as destinations and columns as
sources. PyG stores sources in `edge_index[0]` and destinations in
`edge_index[1]`. Every conversion must therefore use:

```python
edge_index = neighborhood.coalesce().indices().flip(0).contiguous().long()
```

### Task 1: Establish the baseline

**Files:**

- Read: `pyproject.toml`
- Read: `uv.lock`
- Read: `topobench/nn/backbones/combinatorial/gccn.py`
- Read: `topobench/dataloader/utils.py`
- Read: `configs/transforms/liftings/graph2cell/cycle.yaml`
- Test: `test/nn/backbones/combinatorial/test_gccn.py`
- Test: `test/nn/backbones/combinatorial/test_gccn_onehasse.py`

**Step 1: Confirm the tree contains only intentional documentation changes**

Run:

```bash
git status --short
```

Expected: no source-code changes. The implementation-plan document may be
listed until it is committed.

**Step 2: Synchronize the Python 3.11 environment**

Run:

```bash
uv sync --all-extras
```

Expected: exit code 0.

**Step 3: Verify the exact APIs required by the implementation**

Run:

```bash
uv run python -c "import torch, torch_geometric; from torch_geometric.nn import HGTConv; print(torch.__version__, torch_geometric.__version__, HGTConv.__name__)"
```

Expected: Python imports succeed and the final token is `HGTConv`.

**Step 4: Run the existing combinatorial-backbone tests**

Run:

```bash
uv run pytest test/nn/backbones/combinatorial/test_gccn.py test/nn/backbones/combinatorial/test_gccn_onehasse.py -q
```

Expected: PASS. Record any pre-existing failure before changing code. Do not
attribute a baseline failure to HGT.

**Step 5: Inspect the composed default cell lifting**

Run:

```bash
uv run python -m topobench --cfg job model=cell/topotune dataset=graph/MUTAG
```

Expected: the printed configuration includes `CellCycleLifting`,
`max_cell_length: 10`, and rank-0/1/2 feature encoding.

### Task 2: Test and implement heterogeneous input conversion

**Files:**

- Create: `test/nn/backbones/combinatorial/test_hgt.py`
- Create: `topobench/nn/backbones/combinatorial/hgt.py`

**Step 1: Write the failing conversion tests**

Create `test/nn/backbones/combinatorial/test_hgt.py` with:

```python
"""Unit tests for the batched cell-complex HGT backbone."""

from __future__ import annotations

import pytest
import torch
from torch_geometric.data import Data

from topobench.nn.backbones.combinatorial.hgt import CellHGT


NEIGHBORHOODS = [
    "up_incidence-0",
    "down_incidence-1",
    "up_incidence-1",
    "down_incidence-2",
]


def make_complex(
    *,
    num_faces: int = 1,
    feature_dims: tuple[int, int, int] = (8, 8, 8),
    feature_shift: float = 0.0,
) -> Data:
    """Build a three-vertex, two-edge synthetic cell complex."""
    x_0 = (
        torch.arange(3 * feature_dims[0], dtype=torch.float32)
        .reshape(3, feature_dims[0])
        + feature_shift
    )
    x_1 = (
        torch.arange(2 * feature_dims[1], dtype=torch.float32)
        .reshape(2, feature_dims[1])
        + feature_shift
    )
    x_2 = (
        torch.arange(num_faces * feature_dims[2], dtype=torch.float32)
        .reshape(num_faces, feature_dims[2])
        + feature_shift
    )

    incidence_1 = torch.sparse_coo_tensor(
        torch.tensor([[0, 1, 1, 2], [0, 0, 1, 1]]),
        torch.ones(4),
        size=(3, 2),
    ).coalesce()

    if num_faces:
        incidence_2_indices = torch.tensor([[0, 1], [0, 0]])
        incidence_2_values = torch.ones(2)
    else:
        incidence_2_indices = torch.empty((2, 0), dtype=torch.long)
        incidence_2_values = torch.empty(0)
    incidence_2 = torch.sparse_coo_tensor(
        incidence_2_indices,
        incidence_2_values,
        size=(2, num_faces),
    ).coalesce()

    data = Data(x_0=x_0, x_1=x_1, x_2=x_2, y=torch.tensor([0]))
    data["incidence_1"] = incidence_1
    data["incidence_2"] = incidence_2
    data["up_incidence-0"] = incidence_1.t().coalesce()
    data["down_incidence-1"] = incidence_1
    data["up_incidence-1"] = incidence_2.t().coalesce()
    data["down_incidence-2"] = incidence_2
    data["shape"] = torch.tensor([[3, 2, num_faces]])
    return data


def make_model(
    *,
    neighborhoods: list[str] | None = None,
    dropout: float = 0.0,
) -> CellHGT:
    """Build the smallest useful test model."""
    return CellHGT(
        hidden_channels=8,
        num_layers=2,
        heads=2,
        neighborhoods=neighborhoods or NEIGHBORHOODS,
        max_rank=2,
        dropout=dropout,
        activation="relu",
    )


def test_to_heterogeneous_inputs_preserves_types_and_direction():
    batch = make_complex()
    model = make_model()

    x_dict, edge_index_dict = model.to_heterogeneous_inputs(batch)

    assert list(x_dict) == ["rank_0", "rank_1", "rank_2"]
    assert x_dict["rank_0"] is batch.x_0
    assert x_dict["rank_1"] is batch.x_1
    assert x_dict["rank_2"] is batch.x_2

    expected_up_01 = torch.tensor([[0, 1, 1, 2], [0, 0, 1, 1]])
    expected_down_10 = torch.tensor([[0, 0, 1, 1], [0, 1, 1, 2]])
    expected_up_12 = torch.tensor([[0, 1], [0, 0]])
    expected_down_21 = torch.tensor([[0, 0], [0, 1]])

    torch.testing.assert_close(
        edge_index_dict[("rank_0", "up_incidence-0", "rank_1")],
        expected_up_01,
    )
    torch.testing.assert_close(
        edge_index_dict[("rank_1", "down_incidence-1", "rank_0")],
        expected_down_10,
    )
    torch.testing.assert_close(
        edge_index_dict[("rank_1", "up_incidence-1", "rank_2")],
        expected_up_12,
    )
    torch.testing.assert_close(
        edge_index_dict[("rank_2", "down_incidence-2", "rank_1")],
        expected_down_21,
    )


def test_constructor_rejects_invalid_hyperparameters():
    with pytest.raises(ValueError, match="divisible"):
        CellHGT(
            hidden_channels=10,
            num_layers=1,
            heads=4,
            neighborhoods=NEIGHBORHOODS,
        )

    with pytest.raises(ValueError, match="incidence"):
        CellHGT(
            hidden_channels=8,
            num_layers=1,
            heads=2,
            neighborhoods=["up_adjacency-0"],
        )


def test_conversion_requires_every_configured_field():
    batch = make_complex()
    del batch["up_incidence-1"]

    with pytest.raises(KeyError, match="up_incidence-1"):
        make_model().to_heterogeneous_inputs(batch)
```

**Step 2: Run the tests to verify the expected failure**

Run:

```bash
uv run pytest test/nn/backbones/combinatorial/test_hgt.py -q
```

Expected: FAIL during collection with
`ModuleNotFoundError: ...combinatorial.hgt`.

**Step 3: Implement only validation, metadata, and conversion**

Create `topobench/nn/backbones/combinatorial/hgt.py` with:

```python
"""Heterogeneous Graph Transformer for batched cell complexes."""

from __future__ import annotations

import torch
from torch_geometric.data import Data

from topobench.data.utils import get_routes_from_neighborhoods


class CellHGT(torch.nn.Module):
    """Map cell ranks and incidence relations to a heterogeneous graph."""

    def __init__(
        self,
        hidden_channels: int,
        num_layers: int,
        heads: int,
        neighborhoods,
        max_rank: int = 2,
        dropout: float = 0.0,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        if hidden_channels % heads != 0:
            raise ValueError(
                "hidden_channels must be divisible by the number of heads"
            )
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")

        self.hidden_channels = hidden_channels
        self.out_channels = hidden_channels
        self.num_layers = num_layers
        self.heads = heads
        self.max_rank = max_rank
        self.dropout_probability = dropout
        self.activation_name = activation
        self.neighborhoods = list(neighborhoods)

        if not self.neighborhoods:
            raise ValueError("At least one incidence neighborhood is required")
        if any("incidence" not in name for name in self.neighborhoods):
            raise ValueError(
                "CellHGT version 1 supports incidence neighborhoods only"
            )
        if len(set(self.neighborhoods)) != len(self.neighborhoods):
            raise ValueError("Neighborhood names must be unique")

        self.routes = [
            tuple(route)
            for route in get_routes_from_neighborhoods(self.neighborhoods)
        ]
        if any(max(route) > max_rank for route in self.routes):
            raise ValueError("A neighborhood route exceeds max_rank")

        self.node_types = [
            self.node_type(rank) for rank in range(self.max_rank + 1)
        ]
        self.edge_types = [
            (
                self.node_type(src_rank),
                neighborhood,
                self.node_type(dst_rank),
            )
            for neighborhood, (src_rank, dst_rank) in zip(
                self.neighborhoods, self.routes, strict=True
            )
        ]
        self.metadata = (self.node_types, self.edge_types)

    @staticmethod
    def node_type(rank: int) -> str:
        """Return the PyG node-type name for a cell rank."""
        return f"rank_{rank}"

    def to_heterogeneous_inputs(self, batch: Data):
        """Convert a batched TopoBench complex to PyG HGT dictionaries."""
        x_dict = {}
        for rank in range(self.max_rank + 1):
            field = f"x_{rank}"
            if batch.get(field) is None:
                raise KeyError(f"Missing cell feature field: {field}")
            x_dict[self.node_type(rank)] = batch[field]

        edge_index_dict = {}
        for neighborhood, edge_type in zip(
            self.neighborhoods, self.edge_types, strict=True
        ):
            matrix = batch.get(neighborhood)
            if matrix is None:
                raise KeyError(
                    f"Missing configured neighborhood: {neighborhood}"
                )
            if not matrix.is_sparse:
                raise TypeError(
                    f"Neighborhood {neighborhood} must be a sparse COO tensor"
                )
            edge_index_dict[edge_type] = (
                matrix.coalesce()
                .indices()
                .flip(0)
                .contiguous()
                .long()
            )
        return x_dict, edge_index_dict

    def forward(self, batch: Data):
        """Apply HGT layers; implemented in the next TDD task."""
        raise NotImplementedError
```

**Step 4: Run the conversion tests**

Run:

```bash
uv run pytest test/nn/backbones/combinatorial/test_hgt.py -q
```

Expected: 3 tests PASS.

**Step 5: Commit the conversion boundary**

```bash
git add topobench/nn/backbones/combinatorial/hgt.py test/nn/backbones/combinatorial/test_hgt.py
git commit -m "feat: map cell complexes to HGT inputs"
```

### Task 3: Test and implement HGT message passing

**Files:**

- Modify: `test/nn/backbones/combinatorial/test_hgt.py`
- Modify: `topobench/nn/backbones/combinatorial/hgt.py`

**Step 1: Append failing forward, gradient, and empty-rank tests**

Append:

```python
def test_forward_preserves_every_rank_shape_without_mutating_batch():
    batch = make_complex()
    original = {
        rank: batch[f"x_{rank}"].clone() for rank in range(3)
    }

    output = make_model()(batch)

    assert set(output) == {0, 1, 2}
    for rank in range(3):
        assert output[rank].shape == original[rank].shape
        torch.testing.assert_close(batch[f"x_{rank}"], original[rank])


def test_backward_produces_finite_hgt_gradients():
    batch = make_complex()
    model = make_model()

    output = model(batch)
    loss = sum(value.square().mean() for value in output.values())
    loss.backward()

    hgt_gradients = [
        parameter.grad
        for name, parameter in model.named_parameters()
        if "convs" in name and parameter.requires_grad
    ]
    assert hgt_gradients
    assert any(gradient is not None for gradient in hgt_gradients)
    assert all(
        torch.isfinite(gradient).all()
        for gradient in hgt_gradients
        if gradient is not None
    )


def test_eval_mode_is_deterministic():
    batch = make_complex()
    model = make_model(dropout=0.25).eval()

    first = model(batch)
    second = model(batch)

    for rank in range(3):
        torch.testing.assert_close(first[rank], second[rank])


def test_forward_handles_an_empty_rank_two():
    batch = make_complex(num_faces=0)

    output = make_model()(batch)

    assert output[0].shape == (3, 8)
    assert output[1].shape == (2, 8)
    assert output[2].shape == (0, 8)
    assert all(torch.isfinite(value).all() for value in output.values())


def test_rank_without_destination_relation_is_carried_forward():
    batch = make_complex()
    model = make_model(neighborhoods=["down_incidence-1"])

    output = model(batch)

    torch.testing.assert_close(output[1], batch.x_1)
    torch.testing.assert_close(output[2], batch.x_2)
```

**Step 2: Run only the new tests and verify failure**

Run:

```bash
uv run pytest test/nn/backbones/combinatorial/test_hgt.py -q
```

Expected: the original 3 tests PASS and the new tests FAIL because `forward`
raises `NotImplementedError`.

**Step 3: Replace the production module with the complete minimal backbone**

Replace `topobench/nn/backbones/combinatorial/hgt.py` with:

```python
"""Heterogeneous Graph Transformer for batched cell complexes."""

from __future__ import annotations

import torch
from torch_geometric.data import Data
from torch_geometric.nn import HGTConv

from topobench.data.utils import get_routes_from_neighborhoods


def _activation(name: str) -> torch.nn.Module:
    """Build an activation supported by the HGT configuration."""
    activations = {
        "relu": torch.nn.ReLU,
        "elu": torch.nn.ELU,
        "tanh": torch.nn.Tanh,
        "id": torch.nn.Identity,
    }
    if name not in activations:
        raise ValueError(f"Unsupported activation: {name}")
    return activations[name]()


class CellHGT(torch.nn.Module):
    """Apply heterogeneous attention across cell ranks."""

    def __init__(
        self,
        hidden_channels: int,
        num_layers: int,
        heads: int,
        neighborhoods,
        max_rank: int = 2,
        dropout: float = 0.0,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        if hidden_channels % heads != 0:
            raise ValueError(
                "hidden_channels must be divisible by the number of heads"
            )
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")

        self.hidden_channels = hidden_channels
        self.out_channels = hidden_channels
        self.num_layers = num_layers
        self.heads = heads
        self.max_rank = max_rank
        self.neighborhoods = list(neighborhoods)

        if not self.neighborhoods:
            raise ValueError("At least one incidence neighborhood is required")
        if any("incidence" not in name for name in self.neighborhoods):
            raise ValueError(
                "CellHGT version 1 supports incidence neighborhoods only"
            )
        if len(set(self.neighborhoods)) != len(self.neighborhoods):
            raise ValueError("Neighborhood names must be unique")

        self.routes = [
            tuple(route)
            for route in get_routes_from_neighborhoods(self.neighborhoods)
        ]
        if any(max(route) > max_rank for route in self.routes):
            raise ValueError("A neighborhood route exceeds max_rank")

        self.node_types = [
            self.node_type(rank) for rank in range(self.max_rank + 1)
        ]
        self.edge_types = [
            (
                self.node_type(src_rank),
                neighborhood,
                self.node_type(dst_rank),
            )
            for neighborhood, (src_rank, dst_rank) in zip(
                self.neighborhoods, self.routes, strict=True
            )
        ]
        self.metadata = (self.node_types, self.edge_types)

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
        self.activation = _activation(activation)
        self.dropout = torch.nn.Dropout(dropout)

    @staticmethod
    def node_type(rank: int) -> str:
        """Return the PyG node-type name for a cell rank."""
        return f"rank_{rank}"

    def to_heterogeneous_inputs(self, batch: Data):
        """Convert a batched TopoBench complex to PyG HGT dictionaries."""
        x_dict = {}
        for rank in range(self.max_rank + 1):
            field = f"x_{rank}"
            if batch.get(field) is None:
                raise KeyError(f"Missing cell feature field: {field}")
            x_dict[self.node_type(rank)] = batch[field]

        edge_index_dict = {}
        for neighborhood, edge_type in zip(
            self.neighborhoods, self.edge_types, strict=True
        ):
            matrix = batch.get(neighborhood)
            if matrix is None:
                raise KeyError(
                    f"Missing configured neighborhood: {neighborhood}"
                )
            if not matrix.is_sparse:
                raise TypeError(
                    f"Neighborhood {neighborhood} must be a sparse COO tensor"
                )
            edge_index_dict[edge_type] = (
                matrix.coalesce()
                .indices()
                .flip(0)
                .contiguous()
                .long()
            )
        return x_dict, edge_index_dict

    def forward(self, batch: Data) -> dict[int, torch.Tensor]:
        """Apply all HGT layers to one disjoint-union mini-batch."""
        x_dict, edge_index_dict = self.to_heterogeneous_inputs(batch)

        for conv, norms in zip(self.convs, self.norms, strict=True):
            previous = x_dict
            messages = conv(previous, edge_index_dict)
            x_dict = {}
            for node_type, old_features in previous.items():
                updated = messages.get(node_type)
                if updated is None:
                    x_dict[node_type] = old_features
                    continue
                x_dict[node_type] = self.dropout(
                    self.activation(norms[node_type](updated))
                )

        return {
            rank: x_dict[self.node_type(rank)]
            for rank in range(self.max_rank + 1)
        }
```

PyG's `HGTConv` already includes its learned skip connection when input and
output widths agree. Do not add a second unconditional residual connection.
The explicit fallback above is only for a type absent from the layer output.

**Step 4: Run the unit tests**

Run:

```bash
uv run pytest test/nn/backbones/combinatorial/test_hgt.py -q
```

Expected: 8 tests PASS.

**Step 5: Commit the HGT computation**

```bash
git add topobench/nn/backbones/combinatorial/hgt.py test/nn/backbones/combinatorial/test_hgt.py
git commit -m "feat: add cell-complex HGT message passing"
```

### Task 4: Prove that real TopoBench batching is isolated and efficient

**Files:**

- Modify: `test/nn/backbones/combinatorial/test_hgt.py`
- Read: `topobench/dataloader/utils.py`

**Step 1: Add the real collator import and helper**

Add this import:

```python
from topobench.dataloader.utils import collate_fn
```

Add:

```python
def loader_item(data: Data):
    """Adapt a Data object to the tuple returned by DataloadDataset."""
    keys = list(data.keys())
    return ([data[key] for key in keys], keys)
```

**Step 2: Append the failing batch-isolation tests**

Append:

```python
def test_collated_relations_never_cross_graph_boundaries():
    graph_a = make_complex(feature_shift=0.0)
    graph_b = make_complex(num_faces=0, feature_shift=100.0)
    batch = collate_fn([loader_item(graph_a), loader_item(graph_b)])
    model = make_model()

    _, edge_index_dict = model.to_heterogeneous_inputs(batch)

    memberships = {
        "rank_0": batch.batch_0,
        "rank_1": batch.batch_1,
        "rank_2": batch.batch_2,
    }
    for (src_type, _, dst_type), edge_index in edge_index_dict.items():
        if edge_index.numel() == 0:
            continue
        src_graph = memberships[src_type][edge_index[0]]
        dst_graph = memberships[dst_type][edge_index[1]]
        torch.testing.assert_close(src_graph, dst_graph)


def test_eval_output_is_equal_alone_and_in_a_batch():
    graph_a = make_complex(feature_shift=0.0)
    graph_b = make_complex(num_faces=0, feature_shift=100.0)
    single_a = collate_fn([loader_item(graph_a)])
    single_b = collate_fn([loader_item(graph_b)])
    combined = collate_fn([loader_item(graph_a), loader_item(graph_b)])
    model = make_model().eval()

    output_a = model(single_a)
    output_b = model(single_b)
    output_combined = model(combined)

    counts_a = (3, 2, 1)
    counts_b = (3, 2, 0)
    for rank, (count_a, count_b) in enumerate(
        zip(counts_a, counts_b, strict=True)
    ):
        torch.testing.assert_close(
            output_combined[rank][:count_a],
            output_a[rank],
            rtol=1e-5,
            atol=1e-6,
        )
        torch.testing.assert_close(
            output_combined[rank][count_a : count_a + count_b],
            output_b[rank],
            rtol=1e-5,
            atol=1e-6,
        )


def test_batch_with_no_faces_in_any_graph_is_supported():
    batch = collate_fn(
        [
            loader_item(make_complex(num_faces=0)),
            loader_item(make_complex(num_faces=0, feature_shift=100.0)),
        ]
    )

    output = make_model()(batch)

    assert output[0].shape == (6, 8)
    assert output[1].shape == (4, 8)
    assert output[2].shape == (0, 8)
```

**Step 3: Run the new tests**

Run:

```bash
uv run pytest test/nn/backbones/combinatorial/test_hgt.py -q
```

Expected: 11 tests PASS. If the alone-versus-batched test fails, first inspect
the converted indices and graph-membership vectors. Do not weaken its
tolerance enough to hide cross-graph attention.

**Step 4: Confirm the production forward contains no example loop**

Run:

```bash
rg -n "to_data_list|for .*graph|for .*sample" topobench/nn/backbones/combinatorial/hgt.py
```

Expected: no matches.

**Step 5: Commit the batching guarantees**

```bash
git add test/nn/backbones/combinatorial/test_hgt.py
git commit -m "test: prove HGT batch isolation"
```

### Task 5: Wire HGT into Hydra and the normal TopoBench model

**Files:**

- Create: `configs/model/cell/hgt.yaml`
- Create: `test/pipeline/test_hgt_pipeline.py`

**Step 1: Write the failing Hydra composition test**

Create `test/pipeline/test_hgt_pipeline.py`:

```python
"""Configuration and end-to-end tests for cell HGT."""

from __future__ import annotations

import hydra

from topobench.nn.backbones.combinatorial.hgt import CellHGT


def test_hgt_model_config_composes_and_instantiates():
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    with hydra.initialize(version_base="1.3", config_path="../../configs"):
        cfg = hydra.compose(
            config_name="run.yaml",
            overrides=[
                "model=cell/hgt",
                "dataset=graph/MUTAG",
                "trainer.accelerator=cpu",
                "trainer.devices=1",
            ],
        )
        model = hydra.utils.instantiate(
            cfg.model,
            evaluator=cfg.evaluator,
            optimizer=cfg.optimizer,
            loss=cfg.loss,
        )

    assert isinstance(model.backbone.backbone, CellHGT)
    assert model.backbone.backbone.metadata[0] == [
        "rank_0",
        "rank_1",
        "rank_2",
    ]
    assert model.hparams.model_name == "hgt"
```

**Step 2: Run it and verify the missing-config failure**

Run:

```bash
uv run pytest test/pipeline/test_hgt_pipeline.py::test_hgt_model_config_composes_and_instantiates -q
```

Expected: FAIL because `configs/model/cell/hgt.yaml` does not exist.

**Step 3: Add the model configuration**

Create `configs/model/cell/hgt.yaml`:

```yaml
_target_: topobench.model.TBModel

model_name: hgt
model_domain: cell

feature_encoder:
  _target_: topobench.nn.encoders.${model.feature_encoder.encoder_name}
  encoder_name: AllCellFeatureEncoder
  in_channels: ${infer_in_channels:${dataset},${oc.select:transforms,null}}
  out_channels: 64
  proj_dropout: 0.0
  selected_dimensions:
    - 0
    - 1
    - 2

backbone:
  _target_: topobench.nn.backbones.combinatorial.hgt.CellHGT
  hidden_channels: ${model.feature_encoder.out_channels}
  num_layers: 2
  heads: 4
  max_rank: 2
  dropout: 0.1
  activation: relu
  neighborhoods:
    - up_incidence-0
    - down_incidence-1
    - up_incidence-1
    - down_incidence-2

backbone_wrapper:
  _target_: topobench.nn.wrappers.combinatorial.TuneWrapper
  _partial_: true
  wrapper_name: TuneWrapper
  out_channels: ${model.feature_encoder.out_channels}
  num_cell_dimensions: 3

readout:
  _target_: topobench.nn.readouts.${model.readout.readout_name}
  readout_name: PropagateSignalDown
  num_cell_dimensions: 3
  hidden_dim: ${model.feature_encoder.out_channels}
  out_channels: ${dataset.parameters.num_classes}
  task_level: ${define_task_level:${dataset.parameters.task_level},${dataset.split_params.learning_setting}}
  pooling_type: sum

compile: false
```

**Step 4: Run the configuration test**

Run:

```bash
uv run pytest test/pipeline/test_hgt_pipeline.py::test_hgt_model_config_composes_and_instantiates -q
```

Expected: PASS.

**Step 5: Print and inspect the resolved configurations**

Run:

```bash
uv run python -m topobench --cfg job model=cell/hgt dataset=graph/MUTAG
uv run python -m topobench --cfg job model=cell/hgt dataset=graph/PROTEINS
uv run python -m topobench --cfg job model=cell/hgt dataset=graph/ZINC
```

Expected for all three:

- the transform resolves to `CellCycleLifting`;
- `feature_encoder.out_channels` equals `backbone.hidden_channels`;
- all four incidence neighborhoods are requested;
- the dataset batch size remains greater than one;
- ZINC includes its one-hot input transform.

**Step 6: Commit the configuration**

```bash
git add configs/model/cell/hgt.yaml test/pipeline/test_hgt_pipeline.py
git commit -m "feat: configure cell HGT model"
```

### Task 6: Add batched MUTAG and PROTEINS integration tests

**Files:**

- Modify: `test/pipeline/test_hgt_pipeline.py`

**Step 1: Add imports**

Add:

```python
import pytest

from test._utils.simplified_pipeline import run
```

**Step 2: Append the two-dataset smoke test**

Append:

```python
@pytest.mark.parametrize(
    ("dataset", "batch_size"),
    [
        ("graph/MUTAG", 16),
        ("graph/PROTEINS", 32),
    ],
)
def test_hgt_two_epoch_batched_pipeline(dataset: str, batch_size: int):
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    with hydra.initialize(version_base="1.3", config_path="../../configs"):
        cfg = hydra.compose(
            config_name="run.yaml",
            overrides=[
                "model=cell/hgt",
                f"dataset={dataset}",
                f"dataset.dataloader_params.batch_size={batch_size}",
                "model.feature_encoder.out_channels=32",
                "model.backbone.heads=4",
                "model.backbone.num_layers=2",
                "model.backbone.dropout=0.0",
                "trainer.max_epochs=2",
                "trainer.min_epochs=1",
                "trainer.check_val_every_n_epoch=1",
                "trainer.accelerator=cpu",
                "trainer.devices=1",
                "paths=test",
                "callbacks=model_checkpoint",
            ],
            return_hydra_config=True,
        )
        run(cfg)
```

These are deliberately real integration tests. They may download datasets on
the first run and should be run explicitly rather than silently skipped.

**Step 3: Run MUTAG alone**

Run:

```bash
uv run pytest 'test/pipeline/test_hgt_pipeline.py::test_hgt_two_epoch_batched_pipeline[graph/MUTAG-16]' -q
```

Expected: PASS after two training epochs and a test pass. The logged loss must
remain finite.

**Step 4: Run PROTEINS alone**

Run:

```bash
uv run pytest 'test/pipeline/test_hgt_pipeline.py::test_hgt_two_epoch_batched_pipeline[graph/PROTEINS-32]' -q
```

Expected: PASS after two training epochs and a test pass. This confirms that
the implementation does not require `batch_size=1`.

**Step 5: Re-run all HGT tests together**

Run:

```bash
uv run pytest test/nn/backbones/combinatorial/test_hgt.py test/pipeline/test_hgt_pipeline.py -q
```

Expected: all tests PASS.

**Step 6: Commit the dataset smoke tests**

```bash
git add test/pipeline/test_hgt_pipeline.py
git commit -m "test: exercise batched HGT on small datasets"
```

### Task 7: Add reproducible debug and ZINC experiment configurations

**Files:**

- Create: `configs/experiment/cell_hgt_mutag_debug.yaml`
- Create: `configs/experiment/cell_hgt_proteins_debug.yaml`
- Create: `configs/experiment/cell_hgt_zinc.yaml`
- Modify: `test/pipeline/test_hgt_pipeline.py`

**Step 1: Write the experiment configurations**

Create `configs/experiment/cell_hgt_mutag_debug.yaml`:

```yaml
# @package _global_
defaults:
  - override /dataset: graph/MUTAG
  - override /model: cell/hgt
  - _self_

tags: ["cell", "hgt", "MUTAG", "debug"]
seed: 0

model:
  feature_encoder:
    out_channels: 32
    proj_dropout: 0.0
  backbone:
    num_layers: 2
    heads: 4
    dropout: 0.0

dataset:
  dataloader_params:
    batch_size: 16

trainer:
  accelerator: cpu
  devices: 1
  min_epochs: 1
  max_epochs: 2
  check_val_every_n_epoch: 1
```

Create `configs/experiment/cell_hgt_proteins_debug.yaml`:

```yaml
# @package _global_
defaults:
  - override /dataset: graph/PROTEINS
  - override /model: cell/hgt
  - _self_

tags: ["cell", "hgt", "PROTEINS", "debug"]
seed: 0

model:
  feature_encoder:
    out_channels: 32
    proj_dropout: 0.0
  backbone:
    num_layers: 2
    heads: 4
    dropout: 0.0

dataset:
  dataloader_params:
    batch_size: 32

trainer:
  accelerator: cpu
  devices: 1
  min_epochs: 1
  max_epochs: 2
  check_val_every_n_epoch: 1
```

Create `configs/experiment/cell_hgt_zinc.yaml`:

```yaml
# @package _global_
defaults:
  - override /dataset: graph/ZINC
  - override /model: cell/hgt
  - _self_

tags: ["cell", "hgt", "ZINC"]

model:
  feature_encoder:
    out_channels: 64
    proj_dropout: 0.1
  backbone:
    num_layers: 2
    heads: 4
    dropout: 0.1

dataset:
  dataloader_params:
    batch_size: 128

optimizer:
  parameters:
    lr: 0.001
    weight_decay: 0.0001

trainer:
  accelerator: auto
  devices: 1
  min_epochs: 50
  max_epochs: 500
  check_val_every_n_epoch: 5

callbacks:
  early_stopping:
    patience: 10
    min_delta: 0.005
```

**Step 2: Add a failing composition test for every experiment**

Append to `test/pipeline/test_hgt_pipeline.py`:

```python
@pytest.mark.parametrize(
    "experiment",
    [
        "cell_hgt_mutag_debug",
        "cell_hgt_proteins_debug",
        "cell_hgt_zinc",
    ],
)
def test_hgt_experiment_composes(experiment: str):
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    with hydra.initialize(version_base="1.3", config_path="../../configs"):
        cfg = hydra.compose(
            config_name="run.yaml",
            overrides=[f"experiment={experiment}"],
        )

    assert cfg.model.model_name == "hgt"
    assert cfg.model.backbone.neighborhoods == [
        "up_incidence-0",
        "down_incidence-1",
        "up_incidence-1",
        "down_incidence-2",
    ]
    assert cfg.dataset.dataloader_params.batch_size > 1
```

If the configurations were not yet created when following strict TDD order,
run this test before creating them and expect Hydra's missing-config error.

**Step 3: Run composition tests**

Run:

```bash
uv run pytest test/pipeline/test_hgt_pipeline.py -k 'config or compose' -q
```

Expected: 4 tests PASS without downloading any dataset.

**Step 4: Run the repeatable MUTAG smoke command**

Run:

```bash
uv run python -m topobench experiment=cell_hgt_mutag_debug logger=csv callbacks=model_checkpoint hydra.run.dir=outputs/hgt/mutag-smoke +logger.csv.version=mutag-smoke
```

Expected:

- mini-batches contain up to 16 graphs;
- two epochs finish;
- validation and test execute;
- no `nan`, `inf`, shape, or missing-rank error appears;
- metrics are written to
  `outputs/hgt/mutag-smoke/csv/mutag-smoke/metrics.csv`.

**Step 5: Check MUTAG metrics programmatically**

Run:

```bash
uv run python -c "import numpy as np, pandas as pd; p='outputs/hgt/mutag-smoke/csv/mutag-smoke/metrics.csv'; d=pd.read_csv(p); required={'train/loss','val/loss'}; assert required <= set(d); values=pd.concat([d[c].dropna() for c in required]); assert len(values) and np.isfinite(values).all(); print(d[['epoch','train/loss','val/loss']].dropna(how='all').tail())"
```

Expected: exit code 0 and finite losses.

**Step 6: Run the repeatable PROTEINS smoke command**

Run:

```bash
uv run python -m topobench experiment=cell_hgt_proteins_debug logger=csv callbacks=model_checkpoint hydra.run.dir=outputs/hgt/proteins-smoke +logger.csv.version=proteins-smoke
```

Expected: two epochs finish with batches of up to 32 graphs and finite losses.

**Step 7: Run a small overfit diagnostic**

Run:

```bash
uv run python -m topobench experiment=cell_hgt_mutag_debug logger=csv callbacks=model_checkpoint trainer.overfit_batches=1 trainer.max_epochs=20 test=false hydra.run.dir=outputs/hgt/mutag-overfit +logger.csv.version=mutag-overfit
```

Then run:

```bash
uv run python -c "import numpy as np, pandas as pd; p='outputs/hgt/mutag-overfit/csv/mutag-overfit/metrics.csv'; d=pd.read_csv(p); loss=d['train/loss'].dropna(); assert len(loss) >= 2 and np.isfinite(loss).all(); assert loss.iloc[-1] < loss.iloc[0], (loss.iloc[0], loss.iloc[-1]); print({'first': loss.iloc[0], 'last': loss.iloc[-1]})"
```

Expected: the final training loss is lower than the initial training loss.
This is a debugging diagnostic, not a claim about generalization.

**Step 8: Commit the experiment configurations**

```bash
git add configs/experiment/cell_hgt_mutag_debug.yaml configs/experiment/cell_hgt_proteins_debug.yaml configs/experiment/cell_hgt_zinc.yaml test/pipeline/test_hgt_pipeline.py
git commit -m "config: add HGT debug and ZINC experiments"
```

## ZINC promotion gate

Do not continue until all are true:

- HGT unit tests pass.
- Existing GCCN and OneHasse tests still pass.
- Hydra composes all three HGT experiments.
- MUTAG and PROTEINS complete with batch sizes greater than one.
- The synthetic backward test finds finite HGT gradients.
- MUTAG's one-batch overfit loss decreases.
- No per-graph loop exists in the production backbone.

### Task 8: Run the ZINC preflight and prepare the production benchmark

**Files:**

- Modify: `README.md`
- Verify: `configs/experiment/cell_hgt_zinc.yaml`

**Step 1: Perform a one-epoch CPU ZINC preflight**

Run:

```bash
uv run python -m topobench experiment=cell_hgt_zinc logger=csv callbacks=model_checkpoint seed=0 trainer.accelerator=cpu trainer.devices=1 trainer.min_epochs=1 trainer.max_epochs=1 dataset.dataloader_params.batch_size=16 test=false hydra.run.dir=outputs/hgt/zinc-preflight +logger.csv.version=zinc-preflight
```

Expected:

- ZINC downloads or loads from cache;
- the default cycle lifting and degree-feature transform complete;
- a batched training epoch completes;
- the loss is finite;
- no face-empty or relation-direction error occurs.

**Step 2: Check preflight metrics**

Run:

```bash
uv run python -c "import numpy as np, pandas as pd; p='outputs/hgt/zinc-preflight/csv/zinc-preflight/metrics.csv'; d=pd.read_csv(p); loss=d['train/loss'].dropna(); assert len(loss) and np.isfinite(loss).all(); print(loss.tail())"
```

Expected: exit code 0.

**Step 3: Record parameter count before comparing models**

Run:

```bash
uv run python -c "import hydra; from topobench.utils.config_resolvers import register_all_resolvers; register_all_resolvers(); hydra.initialize(version_base='1.3', config_path='configs'); cfg=hydra.compose(config_name='run.yaml', overrides=['experiment=cell_hgt_zinc']); model=hydra.utils.instantiate(cfg.model,evaluator=cfg.evaluator,optimizer=cfg.optimizer,loss=cfg.loss); print(sum(p.numel() for p in model.parameters() if p.requires_grad))"
```

Expected: one integer. Save it with the experimental results.

**Step 4: Execute the full five-seed ZINC HGT run**

On a CUDA machine, execute:

```bash
uv run python -m topobench experiment=cell_hgt_zinc logger=csv seed=0,3,5,7,9 trainer.accelerator=gpu 'trainer.devices=[0]' --multirun
```

This is the requested production command. It keeps `batch_size=128`, uses the
fixed ZINC train/validation/test split, and varies the model/training seed. Do
not sweep `dataset.split_params.data_seed`: ZINC's split is fixed.

Expected: five Hydra jobs. Each trains for at least 50 epochs and at most 500
epochs, with early stopping checked every five epochs. Report mean and standard
deviation of test MAE across the five seeds.

**Step 5: Run a matched-protocol TopoTune reference**

Run:

```bash
uv run python -m topobench model=cell/topotune dataset=graph/ZINC logger=csv seed=0,3,5,7,9 model.feature_encoder.out_channels=64 model.feature_encoder.proj_dropout=0.1 model.backbone.layers=2 model.backbone.GNN.num_layers=1 'model.backbone.neighborhoods=[up_incidence-0,down_incidence-1,up_incidence-1,down_incidence-2]' dataset.dataloader_params.batch_size=128 optimizer.parameters.lr=0.001 optimizer.parameters.weight_decay=0.0001 trainer.accelerator=gpu 'trainer.devices=[0]' trainer.min_epochs=50 trainer.max_epochs=500 trainer.check_val_every_n_epoch=5 callbacks.early_stopping.patience=10 callbacks.early_stopping.min_delta=0.005 --multirun
```

Expected: five reference jobs using the same lifting, relation set, width,
batch size, optimizer settings, epoch budget, split, and seeds. Report its
parameter count as well; equal width does not imply equal parameter count.

**Step 6: Add user-facing run instructions**

Add this concise section to `README.md`:

```markdown
### Batched cell-complex HGT

Run the two-epoch CPU smoke checks:

```bash
uv run python -m topobench experiment=cell_hgt_mutag_debug logger=csv
uv run python -m topobench experiment=cell_hgt_proteins_debug logger=csv
```

After both smoke checks pass, run the five-seed ZINC benchmark on GPU:

```bash
uv run python -m topobench experiment=cell_hgt_zinc logger=csv seed=0,3,5,7,9 trainer.accelerator=gpu 'trainer.devices=[0]' --multirun
```

All HGT experiments retain inductive mini-batching. The ZINC loader uses its
standard fixed split, so the sweep varies `seed`, not the split seed.
```

**Step 7: Run focused formatting and regression checks**

Run:

```bash
uv run ruff check topobench/nn/backbones/combinatorial/hgt.py test/nn/backbones/combinatorial/test_hgt.py test/pipeline/test_hgt_pipeline.py
uv run pytest test/nn/backbones/combinatorial/test_hgt.py test/nn/backbones/combinatorial/test_gccn.py test/nn/backbones/combinatorial/test_gccn_onehasse.py test/pipeline/test_hgt_pipeline.py -q
git diff --check
```

Expected: Ruff reports no errors, all targeted tests pass, and
`git diff --check` prints nothing.

**Step 8: Run the broader suite**

Run:

```bash
uv run pytest -q
```

Expected: no new failure relative to Task 1. Existing documented skips or
xfails are acceptable; new HGT failures are not.

**Step 9: Review the implementation against the exclusions**

Run:

```bash
rg -n "pyHGT|HeteroData|to_data_list|for .*graph|for .*sample" topobench/nn/backbones/combinatorial/hgt.py
```

Expected: no matches.

**Step 10: Commit the documentation and final verification state**

```bash
git add README.md
git commit -m "docs: document batched HGT experiments"
```

## Results to retain for grading and scientific evaluation

For every MUTAG, PROTEINS, and ZINC run, retain:

- the resolved Hydra configuration;
- git commit hash;
- package versions;
- seed;
- batch size;
- trainable parameter count;
- best validation metric and final test metric;
- wall-clock training time;
- peak accelerator memory when available;
- whether any graph in the observed batches lacked 2-cells.

The initial implementation is successful if it is correct, batched, and
reproducible. A favorable ZINC MAE would support further research, but merely
adding HGT to TopoBench is engineering work rather than a sufficient PhD-level
methodological contribution.

## Primary references

- PyG HGTConv API:
  <https://pytorch-geometric.readthedocs.io/en/stable/generated/torch_geometric.nn.conv.HGTConv.html>
- PyG HGTConv implementation:
  <https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/conv/hgt_conv.html>
- Heterogeneous Graph Transformer paper:
  <https://arxiv.org/abs/2003.01332>
