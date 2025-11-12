## Quick instructions for AI coding agents working on TopoBench

This file captures the essential, discoverable knowledge an agent needs to be productive in this repository.

- Repo entrypoint / how humans run code:
  - Primary CLI: `python -m topobench` (invokes `topobench.run.main`). See `topobench/__main__.py` and `topobench/run.py`.
  - Use `env_setup.sh` to install deps and editable package: `source env_setup.sh` (installs `pip install -e '.[all]'` and recommended torch wheel lines).
  - Reproducible full experiments are in `scripts/reproduce.sh` (many example `python -m topobench model=... dataset=... --multirun` commands).

- Big-picture architecture (minimal description):
  - `topobench/` contains runtime code: models, dataloaders, transforms, training loop and utilities.
  - `configs/` is a Hydra config tree. Config groups follow the pattern `<group>/<subgroup>/<item>` (e.g. `model=cell/cwn`, `dataset=graph/MUTAG`, `transforms=liftings/graph2cell/...`). The main composed config is `configs/run.yaml`.
  - Data flow: dataset loader -> PreProcessor (lifting / transforms) -> TBDataloader (Lightning datamodule) -> model (LightningModule) -> Trainer (Lightning). See `topobench/run.py` for the end-to-end flow and resolvers.

- Project-specific conventions and gotchas:
  - Hydra config overrides use `group=item` and dotted keys for nested values. Example: `python -m topobench model=cell/cwn dataset=graph/MUTAG transforms=[liftings/graph2cell/discrete_configuration_complex]`.
  - Multi-run experiments use `--multirun` and the project frequently provides comma-separated seeds in `dataset.split_params.data_seed=0,3,5` (see `scripts/reproduce.sh`).
  - `rootutils.setup_root(...)` in `run.py` adds the project root to PYTHONPATH so tests can import without installing; nevertheless prefer `env_setup.sh` for developer environments.
  - Linting/formatting: `format_and_lint.sh` runs `ruff --fix`; `env_setup.sh` also runs `pre-commit install`.

- Tests & CI:
  - Tests run with `pytest` (see `[tool.pytest.ini_options]` in `pyproject.toml`, `--capture=no`).
  - Quick checks: `pytest -q` from repo root (PYTHONPATH is set by rootutils at runtime, but running tests from repo root with installed env is simplest).

- Integration & external deps to be aware of:
  - Important packages pinned in `pyproject.toml`: `hydra-core==1.3.2`, `lightning==2.4.0`, `topomodelx` and `toponetx` are installed from git URLs.
  - GPU/Torch: `env_setup.sh` selects `TORCH` and `CUDA` variables; the script documents which torch+cuda wheels to pick and installs `torch-scatter`, `torch-sparse`, `torch-cluster` from `pyg` wheels.

- Where to look for common edits:
  - Add/modify models: `topobench/model/` (subfolders per domain: `graph/`, `simplicial/`, `hypergraph/`, `cell/`).
  - Add datasets: `topobench/data/` and `configs/dataset/<domain>/` (add a loader class, register it through configs).
  - Liftings and transforms: `configs/transforms/` and `topobench/transforms/` (see README examples and `configs/transforms/liftings/*`).
  - Hydra resolvers and config helpers: `topobench/utils/config_resolvers.py` (used heavily by `run.py`).

- Suggested actionable examples for patches/PRs an AI might generate:
  - Small bugfix in a model: run a single-config quick train locally: `python -m topobench model=graph/gcn dataset=graph/MUTAG trainer.max_epochs=1 dataset.dataloader_params.batch_size=4` to smoke-test.
  - Add a dataset config: create `configs/dataset/<domain>/<name>.yaml` and a loader in `topobench/data/` that implements `.load()` returning (dataset, dataset_dir), consistent with existing loaders.
  - Update linting: modify `format_and_lint.sh` or `pyproject.toml` ruff settings, then run `./format_and_lint.sh`.

- When confused, inspect these canonical files first:
  - `pyproject.toml` (deps, pytest, ruff rules)
  - `README.md` (high-level workflow and hydra examples)
  - `env_setup.sh`, `scripts/reproduce.sh` (how to install and reproduce experiments)
  - `topobench/run.py` (end-to-end entrypoint and data/model wiring)
  - `configs/run.yaml` and `configs/transforms/*` (how configuration groups are organized)

If anything here is unclear or you want the instructions to target a narrower task (for example: only writing datasets, or only model changes), tell me which area to expand and I'll iterate.
