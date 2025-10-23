"""Script to run hypergraph experiments with timing and skipping OOT cases."""

import subprocess
import time

# Output file for timings
timing_file = "timings.txt"

# Ensure timing file is fresh
with open(timing_file, "w") as f:
    f.write("model,dataset,lifting,time(s)\n")

models = [
    "hypergraph/unignn2",
]

datasets = [
    "graph/cocitation_cora",
    "graph/cocitation_citeseer",
    "graph/cocitation_pubmed",
    "graph/amazon_ratings",
    "graph/roman_empire",
    "graph/MUTAG",
    "graph/PROTEINS",
    "graph/ZINC",
]

batch_sizes = [1]  # currently only one value
liftings = [
    "liftings/graph2hypergraph/forman_ricci_curvature",
    "liftings/graph2hypergraph/kernel",
    "liftings/graph2hypergraph/khop",
    "liftings/graph2hypergraph/knn",
    "liftings/graph2hypergraph/mapper",
    "liftings/graph2hypergraph/modularity_maximization",
    "liftings/graph2hypergraph/exclusive_hop",
]

lrs = [0.001]
hidden_channels = [32]

# Datasets where kernel lifting should be skipped
kernel_oot_datasets = {
    "graph/cocitation_pubmed",
    "graph/amazon_ratings",
    "graph/roman_empire",
}

for model in models:
    for dataset, batch_size in zip(
        datasets, batch_sizes * len(datasets), strict=True
    ):
        for lifting in liftings:
            lifting_name = lifting.split("/")[-1]

            # Skip specific kernel lifting combinations
            if lifting_name == "kernel" and dataset in kernel_oot_datasets:
                with open(timing_file, "a") as f:
                    f.write(f"{model},{dataset},{lifting_name},OOT\n")
                continue

            for lr in lrs:
                for h in hidden_channels:
                    cmd = [
                        "python",
                        "-m",
                        "topobench",
                        f"model={model}",
                        f"dataset={dataset}",
                        f"optimizer.parameters.lr={lr}",
                        f"model.feature_encoder.out_channels={h}",
                        "model.readout.readout_name=PropagateSignalDown",
                        "model.feature_encoder.proj_dropout=0.5",
                        f"dataset.dataloader_params.batch_size={batch_size}",
                        f"transforms=[{lifting}]",
                        "dataset.split_params.data_seed=0",
                        "trainer.max_epochs=500",
                        "trainer.min_epochs=50",
                        "trainer.check_val_every_n_epoch=5",
                        "callbacks.early_stopping.patience=10",
                        "logger.wandb.project=temp",
                    ]

                    print(f"Running: {model} | {dataset} | {lifting_name}")
                    time_start = time.time()
                    subprocess.run(cmd)
                    time_end = time.time()
                    duration = round(time_end - time_start, 2)

                    with open(timing_file, "a") as f:
                        f.write(f"{dataset},{lifting_name},{duration}\n")
