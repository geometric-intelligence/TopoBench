neighborhood="['up_adjacency-0','up_adjacency-1','down_adjacency-1','down_adjacency-2','up_incidence-0','up_incidence-1','down_incidence-1','down_incidence-2']"  #"['up_adjacency-0','up_adjacency-1','down_adjacency-2','down_adjacency-1','up_incidence-0','up_incidence-1']" #'['0-virtualnode_incidence-0','up_incidence-0','up_adjacency-0','up_incidence-1']'
python topobenchmarkx/run.py \
    dataset=graph/ZINC \
    model=cell/sann \
    model.backbone.n_layers=4 \
    model.feature_encoder.out_channels=128\
    model.feature_encoder.proj_dropout=0.25\
    dataset.split_params.data_seed=0 \
    dataset.dataloader_params.batch_size=2048 \
    trainer.max_epochs=500 \
    trainer.min_epochs=50 \
    trainer.devices=\[0\]\
    trainer.check_val_every_n_epoch=5 \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0.0001\
    callbacks.early_stopping.patience=10 \
    transforms/data_manipulations@transforms.sann_encoding=add_gpse_information \
    transforms.sann_encoding.pretrain_model=ZINC \
    transforms.sann_encoding.copy_initial=True \
    transforms.graph2cell_lifting.neighborhoods=$neighborhood \
    transforms.sann_encoding.neighborhoods=$neighborhood

# graph2simplicial_lifting