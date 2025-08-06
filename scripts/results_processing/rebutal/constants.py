MODEL_ORDER = {
    "graph":["GCN","GAT","GIN"],
    "simplicial": ["SCN", "SCCNN", "SaNN", "GCCN", "HOPSE-M", "HOPSE-G"],
    "cell": ["CCCN", "CWN", "GCCN", "HOPSE-M", "HOPSE-G"]
}
DATASET_ORDER = ["MUTAG", "PROTEINS", "NCI1", "NCI109", "ZINC", "MANTRA-N", "MANTRA-O", "MANTRA-BN", "MANTRA-BN-0", "MANTRA-BN-1", "MANTRA-BN-2"]
performance_classification = [
    "val/accuracy",
    "test/accuracy",
    "val/auroc",
    "test/auroc",
    "val/recall",
    "test/recall",
    "val/precision",
    "test/precision",
    "val/loss",
    # "val/f1",
    # "test/f1"
]
performance_classification_additional = [
    "val/accuracy",
    "test/accuracy",
    "val/auroc",
    "test/auroc",
    "val/recall",
    "test/recall",
    "val/precision",
    "test/precision",
    # "val/f1",
    # "test/f1"
]
performance_regression = [
    # "val/mae",
    # "test/mae",
    # "val/mse",
    # "test/mse",
]

optimization_metrics = {
    "NCI109": {
        "optim_metric": "val/accuracy",
        "eval_metric": "test/accuracy",
        "direction": "max",
        "performance_columns": performance_classification,
    },
    "NCI1": {
        "optim_metric": "val/accuracy",
        "eval_metric": "test/accuracy",
        "direction": "max",
        "performance_columns": performance_classification,
    },
    "PROTEINS": {
        "optim_metric": "val/accuracy",
        "eval_metric": "test/accuracy",
        "direction": "max",
        "performance_columns": performance_classification,
    },
    "MUTAG": {
        "optim_metric": "val/accuracy",
        "eval_metric": "test/accuracy",
        "direction": "max",
        "performance_columns": performance_classification,
    },
    "ZINC": {
        "optim_metric": "val/mae",
        "eval_metric": "test/mae",
        "direction": "min",
        "performance_columns": performance_regression,
    },
    "IMDB-BINARY": {
        "optim_metric": "val/accuracy",
        "eval_metric": "test/accuracy",
        "direction": "max",
        "performance_columns": performance_classification,
    },
    "IMDB-MULTI": {
        "optim_metric": "val/accuracy",
        "eval_metric": "test/accuracy",
        "direction": "max",
        "performance_columns": performance_classification,
    },
    "Cora": {
        "optim_metric": "val/accuracy",
        "eval_metric": "test/accuracy",
        "direction": "max",
        "performance_columns": performance_classification,
    },
    "Citeseer": {
        "optim_metric": "val/accuracy",
        "eval_metric": "test/accuracy",
        "direction": "max",
        "performance_columns": performance_classification,
    },
    "PubMed": {
        "optim_metric": "val/accuracy",
        "eval_metric": "test/accuracy",
        "direction": "max",
        "performance_columns": performance_classification,
    },
    "MANTRA_name": {
        "optim_metric": "val/f1",
        "eval_metric": "test/f1",
        "direction": "max",
        "performance_columns": performance_classification_additional,
    },
    "MANTRA_orientation": {
        "optim_metric": "val/f1",
        "eval_metric": "test/f1",
        "direction": "max",
        "performance_columns": performance_classification_additional,
    },
    "MANTRA_betti_numbers": {
        "optim_metric": "val/loss",
        "eval_metric": "val/loss",
        "direction": "min",
        "performance_columns": performance_classification,
    },
    "MANTRA_betti_numbers_0": {
        "optim_metric": "val/loss",
        "eval_metric": "test/f1",
        "direction": "min",
        "performance_columns": performance_classification,
    },
    "MANTRA_betti_numbers_1": {
        "optim_metric": "val/loss",
        "eval_metric": "test/f1",
        "direction": "min",
        "performance_columns": performance_classification,
    },
    "MANTRA_betti_numbers_2": {
        "optim_metric": "val/loss",
        "eval_metric": "test/f1",
        "direction": "min",
        "performance_columns": performance_classification,
    },
    "CSL": {
        "optim_metric": "val/accuracy",
        "eval_metric": "test/accuracy",
        "direction": "max",
        "performance_columns": performance_classification,
    }
}
sweeped_columns = [
    "transforms.sann_encoding.max_hop",
    "transforms.sann_encoding.max_rank",
    "transforms.sann_encoding.neighborhoods",
    "model.feature_encoder.proj_dropout",
    "model.backbone.num_layers",
    # "model.backbone.n_layers",
    "model.backbone.hidden_channels",
    "model.readout.hidden_dim",
    "model.feature_encoder.out_channels",
    # Others
    "optimizer.parameters.weight_decay",
    "optimizer.parameters.lr",
    "dataset.dataloader_params.batch_size",
    # Additional
    "transforms.sann_encoding.copy_initial",
    "transforms.sann_encoding.pe_types",
    "transforms.graph2cell_lifting.max_cell_length",
    "transforms.sann_encoding.is_undirected",
]
run_columns = [
    "dataset.split_params.data_seed",
    "seed",
]

# Dataset and model columns
dataset_model_columns = [
    "model.model_name",
    "model.model_domain",
    "dataset.loader.parameters.data_name",
]

# Performance columns
performance_columns = [
    "val/loss",
    "test/loss",
    # "val/mae",
    # "test/mae",
    # "val/mse",
    # "test/mse",
    "val/accuracy",
    "test/accuracy",
    "val/auroc",
    "test/auroc",
    "val/recall",
    "test/recall",
    "val/precision",
    "test/precision",
    # "val/f1",
    # "test/f1"
]
time_columns = [
    "AvgTime/train_epoch_mean",
    "AvgTime/train_epoch_std",
]

keep_columns = (
    dataset_model_columns
    + sweeped_columns
    + performance_columns
    + run_columns
    + time_columns
)

