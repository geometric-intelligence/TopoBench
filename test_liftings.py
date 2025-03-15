"""Test liftings with python to be able to use the debugger more easily."""

import subprocess

# Configurations
project_name = "LIFTING_TEST"
datasets = ["geometric_shapes"]  # Add more datasets if needed
transforms_experiments = ["exp_combinatorial/h2c_universal_strict"]

# Run the commands in Python
for transform in transforms_experiments:
    for _ in datasets:
        command = [
            "python",
            "topobench/run.py",
            "model=cell/topotune",
            "dataset=hypergraph/cocitation_cora",
            "trainer.max_epochs=2",
            "trainer.min_epochs=1",
            "trainer.check_val_every_n_epoch=1",
            f"transforms={transform}",
            f"logger.wandb.project={project_name}",
        ]

        print(f"Running: {' '.join(command)}")
        subprocess.run(command)
