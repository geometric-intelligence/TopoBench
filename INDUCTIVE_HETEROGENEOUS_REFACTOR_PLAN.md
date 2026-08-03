# Inductive Heterogeneous Node Classification Implementation Plan

> **For future implementation:** Use `skill://executing-plans` to execute this plan task-by-task in a new branch/worktree. This document is a plan only; the current HGT and HeteroSAGE heterogeneous pipelines remain transductive.

**Goal:** Add strict unseen-node inductive classification for large heterogeneous graphs without topology, feature, label, fitted-state, or checkpoint leakage between train, validation, and test.

**Architecture:** Keep one immutable typed feature store, but publish a fingerprinted topology view for each phase. Every node type receives a strict phase assignment; each view exposes only that phase's nodes and edges whose endpoints both belong to that phase. Samplers, transforms, provenance, and checkpoints must bind to the selected view.

**Tech stack:** Python 3.11, PyTorch, PyTorch Geometric `HeteroData`, NumPy memory maps, Parquet/DuckDB typed ingestion, Hydra, Lightning, pytest.

## Invariants

- Node IDs are pairwise disjoint across train, validation, and test for every node type.
- Cross-phase edges are excluded from every view and counted in qualification evidence.
- Missing phase assignments are rejected; they are never inferred.
- Relation metadata remains stable when a phase has zero edges for a declared relation.
- Training loaders, fitted transforms, and checkpoints cannot read validation/test topology or features.
- HGT and HeteroSAGE weights may transfer between views because schema and relation vocabularies are identical.

## Implementation Phases

### 1. Freeze the inductive contract

**Modify:**
- `topobench/data/capabilities.py`
- `topobench/data/qualification.py`
- `topobench/nn/capabilities.py`
- `configs/dataset/heterogeneous/*.yaml`

Add a qualified `classification + node_inductive + inductive` contract. Define required phase assignments for every node type and reject inductive configs without strict topology-view metadata.

**Tests:** Extend `test/config/` and `test/architecture/` capability tests. First prove the current configuration is rejected; then prove only the complete strict contract composes.

### 2. Build immutable phase topology views

**Modify:**
- `topobench/data/stores/typed_graph_ingestion.py`
- `topobench/data/stores/typed_graph_store.py`
- `topobench/data/stores/typed_graph_csc.py`
- `topobench/data/stores/typed_graph_arrays.py`

Add a manifest mapping `split_tag -> phase -> topology_view`. Each view points to memory-mapped node-ID and per-relation CSC arrays, records excluded cross-phase edges, and has its own content fingerprint. Do not duplicate feature arrays or materialize Python lists of graph objects.

**Tests:** Add store tests for pairwise node disjointness, internal-edge-only CSC, zero-edge relations, deterministic fingerprints, malformed assignments, and bounded memory.

### 3. Enforce views in loaders and samplers

**Modify:**
- `topobench/dataloader/disk_graph.py`
- `topobench/dataloader/heterogeneous.py`
- `topobench/dataloader/sequence_state.py`
- `topobench/dataloader/device_prefetch.py`

Require every sampling descriptor to carry a topology-view fingerprint. Cluster and neighbor sampling must read only the selected phase view. Resume must reject checkpoints created from a different view.

**Tests:** Assert that no training batch contains a validation/test node or cross-phase edge. Repeat for full-batch, cluster, neighbor, prefetch, interruption, and resume paths.

### 4. Enable the inductive heterogeneous pipeline

**Modify:**
- `topobench/data/pipelines/heterogeneous.py`
- `topobench/model/supervision.py`
- `topobench/nn/backbones/heterogeneous/`
- `topobench/nn/wrappers/heterogeneous/`
- `configs/model/heterogeneous/`

Create phase-specific `HeteroData`/store views while preserving identical metadata. Reuse HGT/HeteroSAGE parameters across views and supervise only the target node type in the active phase. Fit PCA and other learned transforms exclusively on the training view.

**Tests:** Add `test_hgt_can_memorize_tiny_inductive_training_view`, then evaluate on disjoint validation/test views. Verify finite gradients, decreasing training loss, unchanged held-out inputs, and correct prediction identities.

### 5. Qualify and document

**Modify:**
- `topobench/preflight.py`
- `topobench/profiling/`
- `test/integration/qualify_typed_graph_cuda.py`
- `docs/heterogeneous_graphs.md`
- `README.md`

Add preflight leakage checks, per-view resource evidence, real-data qualification, and CPU/CUDA parity. Document transductive versus strict inductive semantics and ensure experiment names state the learning setting.

**Acceptance:**

1. A deliberately leaked fixture fails before training.
2. Training batches contain only training nodes and internal training edges.
3. HGT memorizes a tiny training view while validation/test remain node-disjoint.
4. Cluster and exhaustive neighbor results match the corresponding phase full-batch view.
5. Fitted transforms and resumed checkpoints are bound to the training topology fingerprint.
6. CPU/RSS, CUDA memory/parity, pre-commit, and the complete test suite pass.

## Future Start Note

Start only after the current `graph-hetero-core-impl` branch is reviewed. Create a dedicated branch such as `inductive-heterogeneous-node-classification`; do not retrofit inductive behavior behind the existing transductive selectors. Preserve transductive configs and add explicit inductive experiment/config names so results cannot be mislabeled.
