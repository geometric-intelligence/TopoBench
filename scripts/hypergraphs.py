"""Script to run hypergraph experiments."""

import subprocess

# def run_command(cmd, log_file="log.txt", error_log_file="error_log.txt", failed_log_file="failed_log.txt"):
#     """Runs a command, logs output, and checks for failure."""
#     with open(log_file, "a") as log, open(error_log_file, "a") as err_log:
#         process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
#         stdout, stderr = process.communicate()

#         log.write(stdout)
#         err_log.write(stderr)

#         if process.returncode != 0:
#             with open(failed_log_file, "a") as fail_log:
#                 fail_log.write(f"Command failed: {cmd}\n")
#                 fail_log.write(f"Check {error_log_file} for details.\n")

#     return process.returncode

models = (
    "hypergraph/allsettransformer",
    "hypergraph/edgnn",
    "hypergraph/unignn2",
)
datasets = (
    "graph/cocitation_cora",
    "graph/cocitation_citeseer",
    # "graph/cocitation_pubmed",
    # "graph/amazon_ratings",
    # "graph/roman_empire",
    "graph/MUTAG",
    "graph/PROTEINS",
    "graph/ZINC",
)
batch_sizes = (1, 1, 256, 256, 256)  # (1, 1, 1, 1, 1, 256, 256, 256)
liftings = (
    # "liftings/graph2hypergraph/forman_ricci_curvature",
    # "liftings/graph2hypergraph/kernel",
    # "liftings/graph2hypergraph/khop",
    # "liftings/graph2hypergraph/knn",
    # "liftings/graph2hypergraph/mapper",
    # "liftings/graph2hypergraph/modularity_maximization",
    "liftings/graph2hypergraph/exclusive_hop",
)
lrs = (0.001, 0.01, 0.1)
hidden_channels = (32, 64, 128)

for model in models:
    for dataset, batch_size in zip(datasets, batch_sizes, strict=True):
        for lifting in liftings:
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
                        "model.backbone.n_layers=2",
                        "model.readout.readout_name=PropagateSignalDown",
                        "model.feature_encoder.proj_dropout=0.5",
                        f"dataset.dataloader_params.batch_size={batch_size}",
                        f"transforms=[{lifting}]",
                        "dataset.split_params.data_seed=0,3,5,7,9",
                        "trainer.max_epochs=500",
                        "trainer.min_epochs=50",
                        "trainer.check_val_every_n_epoch=5",
                        "callbacks.early_stopping.patience=10",
                        "logger.wandb.project=hypergraph_liftings",
                        "--multirun",
                    ]
                    # cmd = f'python -m topobench model={model} dataset={dataset} optimizer.parameters.lr={lr} model.feature_encoder.out_channels={h} model.backbone.n_layers=2 model.readout.readout_name=PropagateSignalDown model.feature_encoder.proj_dropout=0.5 dataset.dataloader_params.batch_size=256 transforms={lifting} dataset.split_params.data_seed=0,3,5,7,9 trainer.max_epochs=500 trainer.min_epochs=50 trainer.check_val_every_n_epoch=5 callbacks.early_stopping.patience=10 logger.wandb.project=hypergraph_liftings --multirun'
                    # run_command(cmd)
                    subprocess.run(cmd)
