python topobenchmarkx/run.py \
    dataset=graph/PROTEINS \
    model=cell/sann \
    model.backbone.n_layers=4 \
    model.feature_encoder.out_channels=128\
    model.feature_encoder.proj_dropout=0.25\
    dataset.split_params.data_seed=0 \
    dataset.dataloader_params.batch_size=256 \
    trainer.max_epochs=500 \
    trainer.min_epochs=50 \
    trainer.devices=1 \
    trainer.accelerator=cpu \
    trainer.check_val_every_n_epoch=5 \
    optimizer.scheduler=null \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0001\
    callbacks.early_stopping.patience=10 \
    transforms/data_manipulations@transforms.sann_encoding=add_gpse_information \
    transforms.sann_encoding.pretrain_model=ZINC \
    transforms.sann_encoding.neighborhoods='[incidence_0,0_incidence_0,0_incidence_1]' \
    --multirun