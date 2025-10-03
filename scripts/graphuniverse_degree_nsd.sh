#!/bin/bash


python -m topobench \
    dataset=graph/GraphUniverse_CD \
    dataset.loader.parameters.generation_parameters.universe_parameters.seed=1,3,5 \
    dataset.loader.parameters.generation_parameters.family_parameters.homophily_range=\[0.4,0.6\] \
    dataset.loader.parameters.generation_parameters.family_parameters.avg_degree_range=\[5.0,10.0\],\[10.0,20.0\] \
    dataset.loader.parameters.generation_parameters.family_parameters.n_graphs=1000 \
    model=graph/nsd \
    model.feature_encoder.out_channels=32 \
    model.feature_encoder.proj_dropout=0.3 \
    model.backbone.num_layers=4,6,8 \
    model.backbone.dropout=0.0 \
    model.backbone.sheaf_type=bundle \
    model.readout.hidden_layers=\[16\],\[\] \
    model.readout.dropout=0.3 \
    dataset.split_params.data_seed=1,3,5 \
    dataset.dataloader_params.batch_size=32 \
    transforms.CombinedPSEs.encodings=\[RWSE\] \
    logger.wandb.project=final_degree_experiments \
    trainer.max_epochs=1000 \
    trainer.min_epochs=50 \
    trainer.devices=\[0\] \
    trainer.check_val_every_n_epoch=1 \
    callbacks.early_stopping.patience=50 \
    tags="[degree]" \
    --multirun &

python -m topobench \
    dataset=graph/GraphUniverse_CD \
    dataset.loader.parameters.generation_parameters.universe_parameters.seed=1,3,5 \
    dataset.loader.parameters.generation_parameters.family_parameters.homophily_range=\[0.4,0.6\] \
    dataset.loader.parameters.generation_parameters.family_parameters.avg_degree_range=\[5.0,10.0\],\[10.0,20.0\] \
    dataset.loader.parameters.generation_parameters.family_parameters.n_graphs=1000 \
    model=graph/nsd \
    model.feature_encoder.out_channels=32 \
    model.feature_encoder.proj_dropout=0.3 \
    model.backbone.num_layers=4,6,8 \
    model.backbone.dropout=0.2 \
    model.backbone.sheaf_type=bundle \
    model.readout.hidden_layers=\[16\],\[\] \
    model.readout.dropout=0.3 \
    dataset.split_params.data_seed=1,3,5 \
    dataset.dataloader_params.batch_size=32 \
    transforms.CombinedPSEs.encodings=\[RWSE\] \
    logger.wandb.project=final_degree_experiments \
    trainer.max_epochs=1000 \
    trainer.min_epochs=50 \
    trainer.devices=\[1\] \
    trainer.check_val_every_n_epoch=1 \
    callbacks.early_stopping.patience=50 \
    tags="[degree]" \
    --multirun &

python -m topobench \
    dataset=graph/GraphUniverse_CD \
    dataset.loader.parameters.generation_parameters.universe_parameters.seed=1,3,5 \
    dataset.loader.parameters.generation_parameters.family_parameters.homophily_range=\[0.4,0.6\] \
    dataset.loader.parameters.generation_parameters.family_parameters.avg_degree_range=\[5.0,10.0\],\[10.0,20.0\] \
    dataset.loader.parameters.generation_parameters.family_parameters.n_graphs=1000 \
    model=graph/nsd \
    model.feature_encoder.out_channels=64 \
    model.feature_encoder.proj_dropout=0.3 \
    model.backbone.num_layers=4,6,8 \
    model.backbone.dropout=0.0 \
    model.backbone.sheaf_type=bundle \
    model.readout.hidden_layers=\[16\],\[\] \
    model.readout.dropout=0.3 \
    dataset.split_params.data_seed=1,3,5 \
    dataset.dataloader_params.batch_size=32 \
    transforms.CombinedPSEs.encodings=\[RWSE\] \
    logger.wandb.project=final_degree_experiments \
    trainer.max_epochs=1000 \
    trainer.min_epochs=50 \
    trainer.devices=\[2\] \
    trainer.check_val_every_n_epoch=1 \
    callbacks.early_stopping.patience=50 \
    tags="[degree]" \
    --multirun &

python -m topobench \
    dataset=graph/GraphUniverse_CD \
    dataset.loader.parameters.generation_parameters.universe_parameters.seed=1,3,5 \
    dataset.loader.parameters.generation_parameters.family_parameters.homophily_range=\[0.4,0.6\] \
    dataset.loader.parameters.generation_parameters.family_parameters.avg_degree_range=\[5.0,10.0\],\[10.0,20.0\] \
    dataset.loader.parameters.generation_parameters.family_parameters.n_graphs=1000 \
    model=graph/nsd \
    model.feature_encoder.out_channels=64 \
    model.feature_encoder.proj_dropout=0.3 \
    model.backbone.num_layers=4,6,8 \
    model.backbone.dropout=0.2 \
    model.backbone.sheaf_type=bundle \
    model.readout.hidden_layers=\[16\],\[\] \
    model.readout.dropout=0.3 \
    dataset.split_params.data_seed=1,3,5 \
    dataset.dataloader_params.batch_size=32 \
    transforms.CombinedPSEs.encodings=\[RWSE\] \
    logger.wandb.project=final_degree_experiments \
    trainer.max_epochs=1000 \
    trainer.min_epochs=50 \
    trainer.devices=\[3\] \
    trainer.check_val_every_n_epoch=1 \
    callbacks.early_stopping.patience=50 \
    tags="[degree]" \
    --multirun &