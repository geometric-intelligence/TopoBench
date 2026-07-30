"""Contract tests for the staged CellHGT ZINC search launcher."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts" / "hgt" / "zinc_hgt_search.sh"


def run_launcher(*arguments: str) -> subprocess.CompletedProcess[str]:
    """Run the launcher without starting training or contacting W&B."""
    environment = os.environ.copy()
    environment.update(
        {
            "DRY_RUN": "1",
            "WANDB_ENTITY": "",
            "WANDB_PROJECT": "shared-cell-hgt",
        }
    )
    return subprocess.run(
        ["bash", str(SCRIPT), *arguments],
        cwd=PROJECT_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )


def dry_run_commands(
    result: subprocess.CompletedProcess[str],
) -> list[str]:
    """Extract generated commands from launcher output."""
    return [
        line.removeprefix("DRY RUN: ")
        for line in result.stdout.splitlines()
        if line.startswith("DRY RUN: ")
    ]


def test_depth_phase_builds_three_topotune_aligned_wandb_runs() -> None:
    result = run_launcher("depth", "0")

    assert result.returncode == 0, result.stderr
    commands = dry_run_commands(result)
    assert len(commands) == 3

    for depth, command in zip((2, 4, 8), commands, strict=True):
        assert "experiment=cell_hgt_zinc" in command
        assert "logger=wandb" in command
        assert "logger.wandb.project=shared-cell-hgt" in command
        assert "logger.wandb.group=zinc-hgt-depth-s0" in command
        assert "logger.wandb.job_type=depth-screen" in command
        assert (
            "logger.wandb.tags=\\[cell\\,hgt\\,zinc\\,hpo\\,depth\\]"
            in command
        )
        assert (
            f"+logger.wandb.name="
            f"zinc-hgt-depth-d{depth:02d}-h04-w064-lr1e-3-s0" in command
        )
        assert f"model.backbone.num_layers={depth}" in command
        assert "model.backbone.heads=4" in command
        assert "model.feature_encoder.out_channels=64" in command
        assert "model.feature_encoder.proj_dropout=0.1" in command
        assert "model.backbone.dropout=0.1" in command
        assert "dataset.dataloader_params.batch_size=128" in command
        assert "optimizer.parameters.lr=0.001" in command
        assert "optimizer.parameters.weight_decay=0.0001" in command
        assert "optimizer.scheduler.scheduler_id=StepLR" in command
        assert "optimizer.scheduler.scheduler_params.step_size=50" in command
        assert "optimizer.scheduler.scheduler_params.gamma=0.5" in command
        assert "trainer.accelerator=cpu" in command
        assert "trainer.devices=1" in command
        assert "trainer.min_epochs=50" in command
        assert "trainer.max_epochs=500" in command
        assert "trainer.check_val_every_n_epoch=5" in command
        assert "callbacks.early_stopping.patience=10" in command
        assert "callbacks.early_stopping.min_delta=0.005" in command
        assert "seed=0" in command
        assert "test=true" in command


def test_follow_up_phases_skip_existing_baseline_candidates() -> None:
    heads = run_launcher("heads", "4", "3")
    width = run_launcher("width", "4", "8", "3")
    learning_rate = run_launcher("lr", "4", "8", "128", "3")

    assert heads.returncode == 0, heads.stderr
    assert width.returncode == 0, width.stderr
    assert learning_rate.returncode == 0, learning_rate.stderr

    head_commands = dry_run_commands(heads)
    assert len(head_commands) == 2
    assert "d04-h02-w064-lr1e-3-s3" in head_commands[0]
    assert "d04-h08-w064-lr1e-3-s3" in head_commands[1]
    assert all("-h04-" not in command for command in head_commands)

    width_commands = dry_run_commands(width)
    assert len(width_commands) == 1
    assert "d04-h08-w128-lr1e-3-s3" in width_commands[0]

    lr_commands = dry_run_commands(learning_rate)
    assert len(lr_commands) == 2
    assert "d04-h08-w128-lr5e-4-s3" in lr_commands[0]
    assert "optimizer.parameters.lr=0.0005" in lr_commands[0]
    assert "d04-h08-w128-lr2e-3-s3" in lr_commands[1]
    assert "optimizer.parameters.lr=0.002" in lr_commands[1]


def test_width_must_be_divisible_by_heads() -> None:
    result = run_launcher("width", "4", "3", "0")

    assert result.returncode != 0
    assert "divisible" in result.stderr


def test_unknown_phase_prints_usage_and_fails() -> None:
    result = run_launcher("unknown")

    assert result.returncode != 0
    assert "Usage:" in result.stderr
