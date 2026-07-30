# Cell-HGT ZINC Search Launcher Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a tested shell launcher for a staged, single-seed CellHGT ZINC
capacity search with consistent W&B organization.

**Architecture:** A phase-based Bash script constructs explicit Hydra
overrides for each candidate and executes candidates sequentially. A dry-run
mode exposes the generated commands so pytest can verify the scientific
protocol without launching training or contacting W&B.

**Tech Stack:** Bash, Hydra, TopoBench, Lightning, Weights & Biases, pytest.

---

### Task 1: Specify launcher behavior with failing tests

**Files:**

- Create: `test/scripts/test_zinc_hgt_search.py`

**Step 1:** Add subprocess tests that invoke the missing launcher with
`DRY_RUN=1`.

**Step 2:** Assert that the depth phase produces exactly three meaningful run
names and fixes dropout, batch size, scheduler-compatible epoch settings,
shared W&B project, and seed.

**Step 3:** Assert that the heads, width, and learning-rate phases produce
only the non-duplicate follow-up candidates.

**Step 4:** Assert that invalid phases and invalid divisibility fail before
training.

**Step 5:** Run:

```bash
.venv/bin/pytest test/scripts/test_zinc_hgt_search.py -q
```

Expected: FAIL because `scripts/hgt/zinc_hgt_search.sh` does not exist.

### Task 2: Implement the minimal launcher

**Files:**

- Create: `scripts/hgt/zinc_hgt_search.sh`

**Step 1:** Add strict Bash mode, repository-root discovery, usage text, and
argument validation.

**Step 2:** Add one `run_candidate` function that builds a descriptive W&B run
name and the complete Hydra command.

**Step 3:** Keep the TopoTune-aligned protocol explicit: minimum 50, maximum
500, validation interval 5, patience 10, StepLR inherited from
`configs/optimizer/default.yaml`, and fixed dropout 0.1.

**Step 4:** Dispatch the four staged phases and use `DRY_RUN=1` to print
instead of execute.

**Step 5:** Run the focused pytest file and make it pass.

### Task 3: Verify configuration and shell quality

**Files:**

- Verify: `scripts/hgt/zinc_hgt_search.sh`
- Verify: `test/scripts/test_zinc_hgt_search.py`

**Step 1:** Run `bash -n scripts/hgt/zinc_hgt_search.sh`.

**Step 2:** Dry-run every phase and inspect the generated names and overrides.

**Step 3:** Compose one representative Hydra command with `--cfg job` and
verify W&B, model, and trainer settings.

**Step 4:** Run Ruff on the pytest file and `git diff --check`.

**Step 5:** Commit the launcher, tests, and documentation.

