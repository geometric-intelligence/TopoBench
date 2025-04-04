performance_classification = [
    "val/accuracy",
    "test/accuracy",
    "val/auroc",
    "test/auroc",
    "val/recall",
    "test/recall",
    "val/precision",
    "test/precision",
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
    "val/f1",
    "test/f1",
]
performance_regression = [
    "val/mae",
    "test/mae",
    "val/mse",
    "test/mse",
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
}
sweeped_columns = [
    "transforms.sann_encoding.max_hop",
    "transforms.sann_encoding.max_rank",
    "transforms.sann_encoding.neighborhoods",
    "transforms.sann_encoding.pretrain_model",
    "transforms.sann_encoding.complex_dim",
    "model.feature_encoder.proj_dropout",
    "model.backbone.n_layers",
    "model.backbone.hidden_channels",
    "model.readout.hidden_dim",
    "model.feature_encoder.out_channels",
    # SCCNN
    "model.backbone.sc_order",
    "model.backbone.conv_order",
    "model.readout.readout_name",
    # Others
    "optimizer.parameters.weight_decay",
    "optimizer.parameters.lr",
    "dataset.dataloader_params.batch_size",
    "dataset.loader.parameters.manifold_dim",
    # Additional
    "transforms.sann_encoding.copy_initial",
    "transforms.graph2cell_lifting.max_cell_length",
    "transforms.sann_encoding.use_initial_features",
    "transforms.sann_encoding.pe_types",
    "transforms.sann_encoding.is_undirected",
    "transforms.sann_encoding.target_pe_dim",
    "transforms.sann_encoding.laplacian_norm_type",
    "transforms.sann_encoding.posenc_LapPE_eigen_max_freqs",
    "transforms.sann_encoding.posenc_LapPE_eigen_eigvec_abs",
    "transforms.sann_encoding.posenc_LapPE_eigen_eigvec_norm",
    "transforms.sann_encoding.posenc_LapPE_eigen_skip_zero_freq",
    "transforms.redefine_simplicial_neighbourhoods.signed",
    "transforms.redefine_simplicial_neighbourhoods.complex_dim",
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
    "val/mae",
    "test/mae",
    "val/mse",
    "test/mse",
    "val/accuracy",
    "test/accuracy",
    "val/auroc",
    "test/auroc",
    "val/recall",
    "test/recall",
    "val/precision",
    "test/precision",
    "val/f1",
    "test/f1",
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
