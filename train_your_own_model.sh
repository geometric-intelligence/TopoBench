# Play with the parameters to train your own model!

python -m topobench \
    dataset=graph/MUTAG \
    model=cell/topotune \
    model.feature_encoder.out_channels=32 \
    model.tune_gnn=GCN \
    model.backbone.neighborhoods=\[1-up_laplacian-0,1-down_laplacian-1,1-up_laplacian-1,1-down_laplacian-2\] \
    model.backbone.layers=2 \
    model.feature_encoder.proj_dropout=0.3 \
    dataset.split_params.data_seed=1 \
    model.readout.readout_name=PropagateSignalDown \
    trainer.max_epochs=1000 \
    trainer.min_epochs=50 \
    trainer.check_val_every_n_epoch=1 \
    callbacks.early_stopping.patience=50