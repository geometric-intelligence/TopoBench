# HIGT Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add the standalone `HIGT` backbone, combining HGT attention with a relation-separated heterogeneous injective-sum branch over the same four cell-incidence relations, and deliver an unexecuted staged ZINC search launcher.

**Architecture:** Each HIGT layer sends the same rank-typed features and primitive incidence edges through a PyG `HGTConv` and a custom heterogeneous injective-sum convolution in parallel. Rank-specific normalization and concatenation-based fusion combine the branch outputs without defining meta-paths or materializing composed adjacencies.

**Tech Stack:** Python 3.11, PyTorch, PyTorch Geometric `HGTConv`, Hydra/OmegaConf, Lightning, pytest, Ruff, Bash, and TopoBench's cell-complex batching pipeline.

---

## Execution rules

- Read the approved design at
  `docs/plans/2026-07-30-higt-design.md` before implementation.
- Begin execution in an isolated worktree using @using-git-worktrees.
- Use @test-driven-development for every behavior change.
- Use @systematic-debugging for unexpected failures.
- Use @verification-before-completion before claiming completion.
- The public class must be named exactly `HIGT`.
- Keep the existing `CellHGT` public API and behavior unchanged.
- Do not add CIN upper-adjacency or other same-rank relations.
- Do not add a meta-path configuration or construct a composed adjacency.
- Do not loop over graphs in `HIGT.forward`.
- Do not convert individual graphs to `HeteroData`.
- Do not load, instantiate, train, validate, test, or run a forward pass on
  ZINC during development.
- Runtime validation is allowed only on MUTAG, PROTEINS, and synthetic
  in-memory fixtures.
- ZINC launcher tests must use `DRY_RUN=1` and inspect command strings only.
- Do not run the generated ZINC commands.

## Fixed heterogeneous schema

| TopoBench field | Edge type | Direction |
|---|---|---|
| `up_incidence-0` | `("rank_0", "up_incidence-0", "rank_1")` | vertex → edge |
| `down_incidence-1` | `("rank_1", "down_incidence-1", "rank_0")` | edge → vertex |
| `up_incidence-1` | `("rank_1", "up_incidence-1", "rank_2")` | edge → face |
| `down_incidence-2` | `("rank_2", "down_incidence-2", "rank_1")` | face → edge |

TopoBench sparse matrix rows are destinations and columns are sources. PyG
uses `edge_index[0]` for sources and `edge_index[1]` for destinations.

---

### Task 1: Establish the non-ZINC baseline

**Files:**

- Read: `docs/plans/2026-07-30-higt-design.md`
- Read: `topobench/nn/backbones/combinatorial/hgt.py`
- Read: `configs/model/cell/hgt.yaml`
- Test: `test/nn/backbones/combinatorial/test_hgt.py`
- Test: `test/pipeline/test_hgt_pipeline.py`

**Step 1: Confirm the worktree is isolated and clean**

Run:

```bash
git status --short
git branch --show-current
```

Expected: no uncommitted source changes and a dedicated implementation
branch/worktree.

**Step 2: Verify the existing HGT unit baseline**

Run:

```bash
uv run pytest test/nn/backbones/combinatorial/test_hgt.py -q
```

Expected: PASS.

**Step 3: Verify configuration without using ZINC**

Run:

```bash
uv run pytest \
  test/pipeline/test_hgt_pipeline.py::test_hgt_model_config_composes_and_instantiates \
  -q -k "MUTAG or PROTEINS"
```

Expected: PASS. If pytest's generated IDs do not match those expressions,
select the two exact MUTAG and PROTEINS node IDs reported by
`pytest --collect-only`; do not run the ZINC parameter.

**Step 4: Record the baseline**

Add a short execution note to the implementation log or commit message if
any baseline failure exists. Do not attribute a pre-existing failure to HIGT.

---

### Task 2: Extract the shared heterogeneous input adapter

**Files:**

- Create:
  `topobench/nn/backbones/combinatorial/heterogeneous.py`
- Create:
  `test/nn/backbones/combinatorial/test_heterogeneous.py`
- Modify:
  `topobench/nn/backbones/combinatorial/hgt.py`
- Verify:
  `test/nn/backbones/combinatorial/test_hgt.py`

**Step 1: Write failing metadata and conversion tests**

Create `test/nn/backbones/combinatorial/test_heterogeneous.py` with focused
tests for these functions:

```python
from topobench.nn.backbones.combinatorial.heterogeneous import (
    build_cell_heterogeneous_metadata,
    to_cell_heterogeneous_inputs,
)


def test_metadata_contains_only_configured_primitive_relations():
    routes, node_types, edge_types = build_cell_heterogeneous_metadata(
        neighborhoods=[
            "up_incidence-0",
            "down_incidence-1",
            "up_incidence-1",
            "down_incidence-2",
        ],
        max_rank=2,
    )

    assert routes == [(0, 1), (1, 0), (1, 2), (2, 1)]
    assert node_types == ["rank_0", "rank_1", "rank_2"]
    assert edge_types == [
        ("rank_0", "up_incidence-0", "rank_1"),
        ("rank_1", "down_incidence-1", "rank_0"),
        ("rank_1", "up_incidence-1", "rank_2"),
        ("rank_2", "down_incidence-2", "rank_1"),
    ]
```

Reuse or move the synthetic fixture from `test_hgt.py` and assert that
`to_cell_heterogeneous_inputs` preserves all feature objects and flips every
sparse relation to source-first PyG indices.

Also cover:

- an empty neighborhood list;
- a duplicate neighborhood;
- a non-incidence neighborhood;
- a route outside `0..max_rank`;
- a missing `x_<rank>` field;
- a missing configured relation;
- a dense relation tensor.

**Step 2: Run the tests to verify the expected failure**

Run:

```bash
uv run pytest test/nn/backbones/combinatorial/test_heterogeneous.py -q
```

Expected: FAIL during collection because the module does not exist.

**Step 3: Implement the shared adapter**

Create `heterogeneous.py` with:

```python
"""Shared conversion utilities for heterogeneous cell backbones."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch_geometric.data import Data

from topobench.data.utils import get_routes_from_neighborhoods

EdgeType = tuple[str, str, str]


def cell_node_type(rank: int) -> str:
    return f"rank_{rank}"


def build_cell_heterogeneous_metadata(
    neighborhoods: Sequence[str],
    max_rank: int,
) -> tuple[list[tuple[int, int]], list[str], list[EdgeType]]:
    neighborhood_names = list(neighborhoods)
    if not neighborhood_names:
        raise ValueError("At least one incidence neighborhood is required")
    if any("incidence" not in name for name in neighborhood_names):
        raise ValueError(
            "Heterogeneous cell backbones support incidence neighborhoods only"
        )
    if len(set(neighborhood_names)) != len(neighborhood_names):
        raise ValueError("Neighborhood names must be unique")

    routes = [
        tuple(route)
        for route in get_routes_from_neighborhoods(neighborhood_names)
    ]
    if any(
        rank < 0 or rank > max_rank
        for route in routes
        for rank in route
    ):
        raise ValueError(
            "Neighborhood route ranks must be between 0 and max_rank"
        )

    node_types = [cell_node_type(rank) for rank in range(max_rank + 1)]
    edge_types = [
        (
            cell_node_type(source_rank),
            neighborhood,
            cell_node_type(target_rank),
        )
        for neighborhood, (source_rank, target_rank) in zip(
            neighborhood_names, routes, strict=True
        )
    ]
    return routes, node_types, edge_types


def to_cell_heterogeneous_inputs(
    batch: Data,
    *,
    node_types: Sequence[str],
    edge_types: Sequence[EdgeType],
    neighborhoods: Sequence[str],
) -> tuple[dict[str, torch.Tensor], dict[EdgeType, torch.Tensor]]:
    x_dict: dict[str, torch.Tensor] = {}
    for rank, node_type in enumerate(node_types):
        field = f"x_{rank}"
        features = batch.get(field)
        if features is None:
            raise KeyError(f"Missing cell feature field: {field}")
        x_dict[node_type] = features

    edge_index_dict: dict[EdgeType, torch.Tensor] = {}
    for neighborhood, edge_type in zip(
        neighborhoods, edge_types, strict=True
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
            matrix.coalesce().indices().flip(0).contiguous().long()
        )
    return x_dict, edge_index_dict
```

Adjust docstrings to repository standards.

**Step 4: Refactor `CellHGT` to use the adapter**

Replace its duplicated route, metadata, and conversion construction with the
new functions. Preserve:

- `CellHGT.node_type`;
- `routes`, `node_types`, `edge_types`, and `metadata` attributes;
- exact exception behavior covered by existing tests;
- `to_heterogeneous_inputs` as a public instance method.

**Step 5: Run the adapter and HGT tests**

Run:

```bash
uv run pytest \
  test/nn/backbones/combinatorial/test_heterogeneous.py \
  test/nn/backbones/combinatorial/test_hgt.py \
  -q
```

Expected: PASS.

**Step 6: Commit**

```bash
git add \
  topobench/nn/backbones/combinatorial/heterogeneous.py \
  topobench/nn/backbones/combinatorial/hgt.py \
  test/nn/backbones/combinatorial/test_heterogeneous.py
git commit -m "refactor: share heterogeneous cell conversion"
```

---

### Task 3: Implement relation-separated injective aggregation

**Files:**

- Create: `topobench/nn/backbones/combinatorial/higt.py`
- Create: `test/nn/backbones/combinatorial/test_higt.py`

**Step 1: Write the failing aggregate tests**

Define the fixed test schema and instantiate the private layer:

```python
from topobench.nn.backbones.combinatorial.higt import (
    _HeterogeneousInjectiveConv,
)

NODE_TYPES = ["rank_0", "rank_1", "rank_2"]
EDGE_TYPES = [
    ("rank_0", "up_incidence-0", "rank_1"),
    ("rank_1", "down_incidence-1", "rank_0"),
    ("rank_1", "up_incidence-1", "rank_2"),
    ("rank_2", "down_incidence-2", "rank_1"),
]
```

Add tests that:

1. Replace relation message networks with `torch.nn.Identity`.
2. Give one rank-1 target one incoming rank-0 source with a positive vector.
3. Duplicate that source edge.
4. Call `aggregate_relations`.
5. Assert the duplicate aggregate is exactly twice the single-edge aggregate.

Add a second test with both rank-0-to-rank-1 and rank-2-to-rank-1 messages.
Assert the method returns two distinct tensors associated with two distinct
relation keys, not one combined rank-1 tensor.

Add empty-edge tests asserting correctly shaped all-zero aggregates.

**Step 2: Run the tests to verify the expected failure**

Run:

```bash
uv run pytest test/nn/backbones/combinatorial/test_higt.py -q
```

Expected: FAIL because `higt.py` does not exist.

**Step 3: Implement `_HeterogeneousInjectiveConv`**

Implement a private module with this interface:

```python
class _HeterogeneousInjectiveConv(torch.nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        node_types: list[str],
        edge_types: list[EdgeType],
    ) -> None:
        ...

    def aggregate_relations(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[EdgeType, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        ...

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[EdgeType, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        ...
```

Use deterministic module keys derived from the complete edge type, for
example:

```python
def _edge_type_key(edge_type: EdgeType) -> str:
    return "__".join(edge_type)
```

For every relation, create an independent message MLP:

```python
torch.nn.Sequential(
    torch.nn.Linear(hidden_channels, hidden_channels),
    torch.nn.ReLU(),
    torch.nn.Linear(hidden_channels, hidden_channels),
)
```

Aggregate with `index_add_` into a zero tensor created from the target
features:

```python
aggregate = target_features.new_zeros(
    (target_features.size(0), self.hidden_channels)
)
aggregate.index_add_(0, target_indices, messages)
```

For each target node type, concatenate:

1. `(1 + eps[target_type]) * x_dict[target_type]`;
2. every incoming relation aggregate in metadata order.

Pass the concatenation through an independent target update MLP mapping
`hidden_channels * (1 + incoming_relation_count)` to `hidden_channels`.

Store one learnable scalar epsilon per target type in a
`torch.nn.ParameterDict`.

**Step 4: Run the focused tests**

Run:

```bash
uv run pytest test/nn/backbones/combinatorial/test_higt.py -q
```

Expected: all injective-aggregation tests PASS.

**Step 5: Commit**

```bash
git add \
  topobench/nn/backbones/combinatorial/higt.py \
  test/nn/backbones/combinatorial/test_higt.py
git commit -m "feat: add heterogeneous injective aggregation"
```

---

### Task 4: Implement the public HIGT backbone

**Files:**

- Modify: `topobench/nn/backbones/combinatorial/higt.py`
- Modify: `test/nn/backbones/combinatorial/test_higt.py`

**Step 1: Write failing constructor and structure tests**

Add tests that construct:

```python
model = HIGT(
    hidden_channels=8,
    num_layers=2,
    heads=2,
    neighborhoods=NEIGHBORHOODS,
    max_rank=2,
    dropout=0.0,
    activation="relu",
)
```

Assert:

- the public class name is `HIGT`;
- `out_channels == hidden_channels`;
- metadata exactly equals the four primitive relations;
- every relation name contains `incidence`;
- the number of HGT, injective, normalization, and fusion layers equals
  `num_layers`;
- each fusion projection maps `2 * hidden_channels` to `hidden_channels`;
- there is no `metapath`, `meta_path`, `adjacency`, or composed-relation
  constructor argument.

Port HGT's validation tests for:

- nonpositive heads;
- nonpositive width;
- width not divisible by heads;
- fewer than one layer;
- invalid activation;
- duplicate, non-incidence, and out-of-range neighborhoods.

**Step 2: Run the tests to verify failure**

Run:

```bash
uv run pytest test/nn/backbones/combinatorial/test_higt.py -q
```

Expected: FAIL because public `HIGT` is not implemented.

**Step 3: Implement `HIGT.__init__`**

The constructor signature is:

```python
class HIGT(torch.nn.Module):
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
        ...
```

Create one of each per layer:

- `HGTConv`;
- `_HeterogeneousInjectiveConv`;
- rank-specific injective `LayerNorm`;
- rank-specific attention `LayerNorm`;
- rank-specific `Linear(2 * hidden_channels, hidden_channels)` fusion.

Use `torch.nn.ModuleList` across layers and `torch.nn.ModuleDict` across
node types. Reuse the shared metadata/input adapter from Task 2.

**Step 4: Write failing forward tests**

Using the synthetic complex fixture, assert that forward:

- returns integer keys `{0, 1, 2}`;
- preserves every rank shape;
- does not mutate input features or sparse relations;
- supports zero faces;
- supports a relation with zero edges;
- carries the previous representation into the attention branch when
  `HGTConv` returns `None`;
- is deterministic in evaluation mode.

**Step 5: Implement `HIGT.forward`**

Implement the approved layer update:

```python
for (
    attention_conv,
    injective_conv,
    injective_norms,
    attention_norms,
    fusion_layers,
) in zip(..., strict=True):
    previous = x_dict
    injective = injective_conv(previous, edge_index_dict)
    attention = attention_conv(previous, edge_index_dict)
    x_dict = {}

    for node_type, old_features in previous.items():
        attention_features = attention.get(node_type)
        if attention_features is None:
            attention_features = old_features

        combined = torch.cat(
            [
                injective_norms[node_type](injective[node_type]),
                attention_norms[node_type](attention_features),
            ],
            dim=-1,
        )
        delta = fusion_layers[node_type](combined)
        x_dict[node_type] = self.activation(
            old_features + self.dropout(delta)
        )
```

Return:

```python
{
    rank: x_dict[cell_node_type(rank)]
    for rank in range(self.max_rank + 1)
}
```

**Step 6: Run the focused tests**

Run:

```bash
uv run pytest test/nn/backbones/combinatorial/test_higt.py -q
```

Expected: PASS.

**Step 7: Commit**

```bash
git add \
  topobench/nn/backbones/combinatorial/higt.py \
  test/nn/backbones/combinatorial/test_higt.py
git commit -m "feat: add HIGT backbone"
```

---

### Task 5: Prove HIGT invariants and batch safety

**Files:**

- Modify: `test/nn/backbones/combinatorial/test_higt.py`

**Step 1: Add branch-gradient tests**

Run a synthetic forward and squared-output loss, then assert every trainable
parameter in these groups has a finite non-`None` gradient:

- `attention_convs`;
- `injective_convs`;
- `fusions`.

Also assert each learned epsilon receives a finite gradient.

**Step 2: Add type-preserving permutation equivariance tests**

Construct independent permutations for rank 0, rank 1, and rank 2. Apply the
permutations to:

- `x_0`, `x_1`, and `x_2`;
- row and column indices of all four sparse incidence relations.

Run the same evaluation-mode model on original and permuted complexes.
Inverse-permute each result and compare with `torch.testing.assert_close`.

**Step 3: Add batch-isolation tests**

Reuse `collate_fn` and two complexes with distinct feature shifts. Assert:

- relation endpoints always have matching batch memberships;
- combined-batch output equals concatenated alone outputs;
- the assertion holds when the second graph has no faces;
- a batch where neither graph has faces remains valid.

**Step 4: Add primitive-relation-only tests**

Assert:

```python
assert model.edge_types == EDGE_TYPES
assert all("incidence" in relation for _, relation, _ in model.edge_types)
```

Inspect neither features nor metadata for any generated same-rank edge type.

**Step 5: Run all combinatorial HGT/HIGT tests**

Run:

```bash
uv run pytest \
  test/nn/backbones/combinatorial/test_heterogeneous.py \
  test/nn/backbones/combinatorial/test_hgt.py \
  test/nn/backbones/combinatorial/test_higt.py \
  -q
```

Expected: PASS.

**Step 6: Commit**

```bash
git add test/nn/backbones/combinatorial/test_higt.py
git commit -m "test: prove HIGT invariants"
```

---

### Task 6: Add HIGT model and non-ZINC pipeline configurations

**Files:**

- Create: `configs/model/cell/higt.yaml`
- Create: `configs/experiment/higt_mutag_debug.yaml`
- Create: `configs/experiment/higt_proteins_debug.yaml`
- Create: `configs/experiment/higt_zinc.yaml`
- Create: `test/pipeline/test_higt_pipeline.py`

**Step 1: Write failing MUTAG and PROTEINS composition tests**

Create `test/pipeline/test_higt_pipeline.py`. Parameterize only:

- `higt_mutag_debug`;
- `higt_proteins_debug`.

Assert:

- target is `topobench.nn.backbones.combinatorial.higt.HIGT`;
- model name is `higt`;
- public class instantiates as `HIGT`;
- relation schema is exactly `EDGE_TYPES`;
- width is 32;
- depth is 2;
- heads are 4;
- dropout is 0;
- batch sizes are 16 and 32, respectively;
- trainer uses CPU for two epochs.

Do not add a ZINC parameter to any executable pytest parameterization.

**Step 2: Run the tests to verify failure**

Run:

```bash
uv run pytest test/pipeline/test_higt_pipeline.py -q
```

Expected: FAIL because HIGT configs do not exist.

**Step 3: Add the model config**

Create `configs/model/cell/higt.yaml` by matching the HGT pipeline contract
and changing only the public model identity and backbone target:

```yaml
model_name: higt
model_domain: cell

backbone:
  _target_: topobench.nn.backbones.combinatorial.higt.HIGT
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
```

Copy the HGT feature encoder, `TuneWrapper`, `PropagateSignalDown`, and
`compile` settings exactly unless a HIGT test demonstrates a required
difference.

**Step 4: Add debug experiment configs**

Create `higt_mutag_debug.yaml` and `higt_proteins_debug.yaml` by mirroring the
corresponding HGT debug configs and changing:

- model override to `cell/higt`;
- tags from `hgt` to `higt`.

**Step 5: Add the unexecuted ZINC config**

Create `configs/experiment/higt_zinc.yaml` as a static counterpart to
`cell_hgt_zinc.yaml`, with:

- model override `cell/higt`;
- HIGT tags;
- width 64;
- depth 2;
- heads 4;
- dropout 0.1;
- batch size 128;
- learning rate 0.001;
- weight decay 0.0001;
- the same trainer and early-stopping values as HGT.

Do not compose, instantiate, or execute this experiment during development.

**Step 6: Run only MUTAG and PROTEINS config tests**

Run:

```bash
uv run pytest test/pipeline/test_higt_pipeline.py -q
```

Expected: PASS and no ZINC test collected.

**Step 7: Commit**

```bash
git add \
  configs/model/cell/higt.yaml \
  configs/experiment/higt_mutag_debug.yaml \
  configs/experiment/higt_proteins_debug.yaml \
  configs/experiment/higt_zinc.yaml \
  test/pipeline/test_higt_pipeline.py
git commit -m "feat: configure HIGT experiments"
```

---

### Task 7: Validate training on MUTAG and PROTEINS only

**Files:**

- Modify: `test/pipeline/test_higt_pipeline.py`

**Step 1: Add the two-epoch smoke test**

Parameterize only:

```python
[
    ("graph/MUTAG", 16),
    ("graph/PROTEINS", 32),
]
```

Use `test._utils.simplified_pipeline.run` with:

- `model=cell/higt`;
- width 32;
- two layers;
- four heads;
- dropout 0;
- CPU;
- one minimum and two maximum epochs;
- checkpoint output under `tmp_path`.

Assert:

- exactly two epochs complete;
- observed batch size is greater than one;
- train and validation losses are finite;
- test results exist;
- test loss is finite.

**Step 2: Run the MUTAG smoke case**

Run its exact pytest node ID:

```bash
uv run pytest \
  test/pipeline/test_higt_pipeline.py::test_higt_two_epoch_batched_pipeline \
  -q -k MUTAG
```

Expected: PASS.

**Step 3: Run the PROTEINS smoke case**

Run:

```bash
uv run pytest \
  test/pipeline/test_higt_pipeline.py::test_higt_two_epoch_batched_pipeline \
  -q -k PROTEINS
```

Expected: PASS.

**Step 4: Run the complete HIGT pipeline file**

Run:

```bash
uv run pytest test/pipeline/test_higt_pipeline.py -q
```

Expected: PASS; collected runtime datasets are MUTAG and PROTEINS only.

**Step 5: Commit**

```bash
git add test/pipeline/test_higt_pipeline.py
git commit -m "test: validate HIGT debug pipelines"
```

---

### Task 8: Specify the dry-run ZINC launcher contract

**Files:**

- Create: `test/scripts/test_zinc_higt_search.py`
- Reference: `scripts/hgt/zinc_hgt_search.sh`
- Reference: `test/scripts/test_zinc_hgt_search.py`

**Step 1: Write the failing dry-run launcher tests**

Adapt the HGT launcher tests with:

```python
SCRIPT = (
    PROJECT_ROOT / "scripts" / "higt" / "zinc_higt_search.sh"
)
```

The subprocess environment must include:

```python
{
    "DRY_RUN": "1",
    "WANDB_ENTITY": "",
    "WANDB_PROJECT": "shared-higt",
}
```

Assert the `depth` phase emits three commands for depths 2, 4, and 8 and that
every command contains:

```text
experiment=higt_zinc
logger=wandb
logger.wandb.project=shared-higt
logger.wandb.group=zinc-higt-depth-s0
logger.wandb.job_type=depth-screen
+logger.wandb.name=zinc-higt-depth-...
model.backbone.num_layers=...
model.backbone.heads=4
model.feature_encoder.out_channels=64
test=false
```

Port the HGT tests for:

- heads candidates 2 and 8;
- width candidate 128;
- learning rates 0.0005 and 0.002;
- width/head divisibility failure;
- final phase running exactly one selected configuration with `test=true`;
- unknown phase usage failure.

These tests must never omit `DRY_RUN=1`.

**Step 2: Run the tests to verify failure**

Run:

```bash
uv run pytest test/scripts/test_zinc_higt_search.py -q
```

Expected: FAIL because the launcher does not exist. No TopoBench command is
executed.

---

### Task 9: Implement the unexecuted ZINC launcher

**Files:**

- Create: `scripts/higt/zinc_higt_search.sh`
- Verify: `test/scripts/test_zinc_higt_search.py`

**Step 1: Copy the launcher structure**

Create the HIGT launcher using `scripts/hgt/zinc_hgt_search.sh` as the
structural reference. Preserve:

- `set -euo pipefail`;
- project-root and Python discovery;
- usage and validation functions;
- learning-rate tags;
- dry-run command printing;
- `caffeinate` wrapping;
- sequential staged search;
- final-only test evaluation.

**Step 2: Change all model identities**

Use:

```bash
WANDB_PROJECT_NAME="${WANDB_PROJECT:-higt-zinc}"
```

and:

```text
experiment=higt_zinc
group: zinc-higt-<phase>-s<seed>
name: zinc-higt-<phase>-...
tags: [higt,zinc,hpo,<phase>]
```

Do not use `CellHGT`, `cell-hgt`, or `experiment=cell_hgt_zinc` in the HIGT
launcher.

**Step 3: Preserve the staged candidate sets**

Use:

- depth: 2, 4, 8 at heads 4 and width 64;
- heads: 2 and 8 after the selected depth;
- width: 128 after selected depth and heads;
- learning rate: 0.0005 and 0.002;
- final: exactly the selected configuration.

Keep the TopoTune-aligned optimizer, scheduler, batch size, epoch,
validation, and early-stopping overrides from the HGT launcher.

**Step 4: Run shell syntax validation**

Run:

```bash
bash -n scripts/higt/zinc_higt_search.sh
```

Expected: exit code 0.

**Step 5: Run only dry-run contract tests**

Run:

```bash
uv run pytest test/scripts/test_zinc_higt_search.py -q
```

Expected: PASS. The test invokes the script only with `DRY_RUN=1`.

**Step 6: Manually dry-run every phase**

Run:

```bash
DRY_RUN=1 bash scripts/higt/zinc_higt_search.sh depth 0
DRY_RUN=1 bash scripts/higt/zinc_higt_search.sh heads 4 0
DRY_RUN=1 bash scripts/higt/zinc_higt_search.sh width 4 8 0
DRY_RUN=1 bash scripts/higt/zinc_higt_search.sh lr 4 8 128 0
DRY_RUN=1 bash scripts/higt/zinc_higt_search.sh final 4 8 128 0.0005 0
```

Expected: commands are printed and none is executed.

**Step 7: Commit**

```bash
git add \
  scripts/higt/zinc_higt_search.sh \
  test/scripts/test_zinc_higt_search.py
git commit -m "feat: add HIGT ZINC search launcher"
```

---

### Task 10: Final non-ZINC verification

**Files:**

- Verify all files changed by Tasks 2–9
- Verify: `docs/plans/2026-07-30-higt-design.md`

**Step 1: Run focused unit tests**

Run:

```bash
uv run pytest \
  test/nn/backbones/combinatorial/test_heterogeneous.py \
  test/nn/backbones/combinatorial/test_hgt.py \
  test/nn/backbones/combinatorial/test_higt.py \
  test/scripts/test_zinc_higt_search.py \
  -q
```

Expected: PASS.

**Step 2: Run HIGT pipeline tests**

Run:

```bash
uv run pytest test/pipeline/test_higt_pipeline.py -q
```

Expected: PASS using MUTAG and PROTEINS only.

**Step 3: Run HGT regression tests without the ZINC parameter**

Run:

```bash
uv run pytest test/nn/backbones/combinatorial/test_hgt.py -q
uv run pytest \
  test/pipeline/test_hgt_pipeline.py::test_hgt_model_config_composes_and_instantiates \
  -q -k "MUTAG or PROTEINS"
```

Expected: PASS.

**Step 4: Run static quality checks**

Run:

```bash
uv run ruff check \
  topobench/nn/backbones/combinatorial/heterogeneous.py \
  topobench/nn/backbones/combinatorial/hgt.py \
  topobench/nn/backbones/combinatorial/higt.py \
  test/nn/backbones/combinatorial/test_heterogeneous.py \
  test/nn/backbones/combinatorial/test_higt.py \
  test/nn/backbones/combinatorial/test_higt.py \
  test/pipeline/test_higt_pipeline.py \
  test/scripts/test_zinc_higt_search.py
bash -n scripts/higt/zinc_higt_search.sh
git diff --check
```

Expected: all commands succeed.

**Step 5: Audit the no-meta-path and no-ZINC-runtime constraints**

Run:

```bash
rg -n "meta.?path|adjacency" \
  topobench/nn/backbones/combinatorial/higt.py \
  configs/model/cell/higt.yaml
rg -n "ZINC|zinc" \
  test/nn/backbones/combinatorial/test_higt.py \
  test/pipeline/test_higt_pipeline.py
```

Expected:

- no HIGT input or construction of meta-paths/composed adjacency;
- no ZINC runtime parameter in HIGT unit or pipeline tests.

Occurrences in documentation, the static ZINC experiment config, launcher,
and dry-run launcher test are expected.

**Step 6: Report parameter counts using a permitted dataset**

Instantiate HGT and HIGT through MUTAG or PROTEINS configuration only and
report trainable parameter counts. Do not interpret ZINC performance.

**Step 7: Inspect final repository state**

Run:

```bash
git status --short
git log --oneline -10
```

Expected: only intentional changes and commits.

**Step 8: Commit any final documentation-only correction**

If verification required a documentation or static correction:

```bash
git add <exact-corrected-files>
git commit -m "docs: finalize HIGT handoff"
```

Do not create an empty commit.

## Handoff

The final handoff must state:

- the public backbone is `HIGT`;
- the existing `CellHGT` baseline remains available;
- MUTAG and PROTEINS verification results;
- the exact launcher path
  `scripts/higt/zinc_higt_search.sh`;
- that launcher tests were dry-run only;
- that no ZINC workload was executed;
- the command the user can run first:

```bash
bash scripts/higt/zinc_higt_search.sh depth 0
```

Do not run that command during development; it is for the user.
