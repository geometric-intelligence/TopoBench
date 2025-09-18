#!/bin/bash

data_seeds=(1 3 5 7 9)
for i in ${data_seeds[@]}; do

    python -m topobench \
        dataset=graph/GraphUniverse_CD_transductive \
        dataset.loader.parameters.generation_parameters.universe_parameters.seed=$i \
        dataset.loader.parameters.generation_parameters.family_parameters.homophily_range=\[0.05,0.05\],\[0.45,0.45\],\[0.95,0.95\] \
        dataset.loader.parameters.generation_parameters.family_parameters.n_graphs=1 \
        model=graph/gcn \
        model.feature_encoder.out_channels=32,64,128 \
        model.feature_encoder.proj_dropout=0.3 \
        model.backbone.num_layers=2,4 \
        model.backbone.dropout=0.2,0.4 \
        model.readout.hidden_layers=\[16\],\[\] \
        model.readout.dropout=0.3 \
        dataset.split_params.data_seed=$i \
        logger.wandb.project=final_homophily_experiments_transductive \
        trainer.max_epochs=1000 \
        trainer.min_epochs=50 \
        trainer.devices=\[0\] \
        trainer.check_val_every_n_epoch=1 \
        callbacks.early_stopping.patience=50 \
        tags="[homophily_transductive]" \
        --multirun &
    
    python -m topobench \
        dataset=graph/GraphUniverse_CD_transductive \
        dataset.loader.parameters.generation_parameters.universe_parameters.seed=$i \
        dataset.loader.parameters.generation_parameters.family_parameters.homophily_range=\[0.05,0.05\],\[0.45,0.45\],\[0.95,0.95\] \
        dataset.loader.parameters.generation_parameters.family_parameters.n_graphs=1 \
        model=graph/gps \
        model.feature_encoder.out_channels=32,64,128 \
        model.feature_encoder.proj_dropout=0.3 \
        model.backbone.num_layers=2,4 \
        model.backbone.heads=4 \
        model.backbone.dropout=0.2,0.4 \
        model.backbone.attn_type=multihead \
        model.readout.hidden_layers=\[16\],\[\] \
        model.readout.dropout=0.3 \
        dataset.split_params.data_seed=$i \
        dataset.dataloader_params.batch_size=1 \
        transforms.CombinedPSEs.encodings=\[RWSE\],\[\] \
        logger.wandb.project=final_homophily_experiments_transductive \
        trainer.max_epochs=1000 \
        trainer.min_epochs=50 \
        trainer.devices=\[1\] \
        trainer.check_val_every_n_epoch=1 \
        callbacks.early_stopping.patience=50 \
        tags="[homophily_transductive]" \
        --multirun &

    python -m topobench \
        dataset=graph/GraphUniverse_CD_transductive \
        dataset.loader.parameters.generation_parameters.universe_parameters.seed=$i \
        dataset.loader.parameters.generation_parameters.family_parameters.homophily_range=\[0.05,0.05\],\[0.45,0.45\],\[0.95,0.95\] \
        dataset.loader.parameters.generation_parameters.family_parameters.n_graphs=1 \
        model=graph/nsd \
        model.feature_encoder.out_channels=32,64,128 \
        model.feature_encoder.proj_dropout=0.3 \
        model.backbone.num_layers=4,6 \
        model.backbone.dropout=0.0,0.2 \
        model.backbone.sheaf_type=bundle \
        model.readout.hidden_layers=\[16\],\[\] \
        model.readout.dropout=0.3 \
        dataset.split_params.data_seed=$i \
        dataset.dataloader_params.batch_size=1 \
        transforms.CombinedPSEs.encodings=\[RWSE\],\[\] \
        logger.wandb.project=final_homophily_experiments_transductive \
        trainer.max_epochs=1000 \
        trainer.min_epochs=50 \
        trainer.devices=\[2\] \
        trainer.check_val_every_n_epoch=1 \
        callbacks.early_stopping.patience=50 \
        tags="[homophily_transductive]" \
        --multirun &

    python -m topobench \
        dataset=graph/GraphUniverse_CD_transductive \
        dataset.loader.parameters.generation_parameters.universe_parameters.seed=$i \
        dataset.loader.parameters.generation_parameters.family_parameters.homophily_range=\[0.05,0.05\],\[0.45,0.45\],\[0.95,0.95\] \
        dataset.loader.parameters.generation_parameters.family_parameters.n_graphs=1 \
        model=graph/gat \
        model.feature_encoder.out_channels=32,64,128 \
        model.feature_encoder.proj_dropout=0.3 \
        model.backbone.heads=2,4,8 \
        model.backbone.num_layers=2,4 \
        model.backbone.dropout=0.0,0.2 \
        model.readout.dropout=0.3 \
        model.readout.hidden_layers=\[16\],\[\] \
        dataset.split_params.data_seed=$i \
        dataset.dataloader_params.batch_size=1 \
        logger.wandb.project=final_homophily_experiments_transductive \
        trainer.max_epochs=1000 \
        trainer.min_epochs=50 \
        trainer.devices=\[3\] \
        trainer.check_val_every_n_epoch=1 \
        callbacks.early_stopping.patience=50 \
        tags="[homophily_transductive]" \
        --multirun &

    python -m topobench \
        dataset=graph/GraphUniverse_CD_transductive \
        dataset.loader.parameters.generation_parameters.universe_parameters.seed=$i \
        dataset.loader.parameters.generation_parameters.family_parameters.homophily_range=\[0.05,0.05\],\[0.45,0.45\],\[0.95,0.95\] \
        dataset.loader.parameters.generation_parameters.family_parameters.n_graphs=1 \
        model=graph/sage \
        model.feature_encoder.out_channels=32,64,128 \
        model.feature_encoder.proj_dropout=0.3 \
        model.backbone.num_layers=2,4 \
        model.backbone.dropout=0.2,0.4 \
        model.readout.dropout=0.3 \
        model.readout.hidden_layers=\[16\],\[\] \
        dataset.split_params.data_seed=$i \
        dataset.dataloader_params.batch_size=1 \
        logger.wandb.project=final_homophily_experiments_transductive \
        trainer.max_epochs=1000 \
        trainer.min_epochs=50 \
        trainer.devices=\[0\] \
        trainer.check_val_every_n_epoch=1 \
        callbacks.early_stopping.patience=50 \
        tags="[homophily_transductive]" \
        --multirun &

    python -m topobench \
        dataset=graph/GraphUniverse_CD_transductive \
        dataset.loader.parameters.generation_parameters.universe_parameters.seed=$i \
        dataset.loader.parameters.generation_parameters.family_parameters.homophily_range=\[0.05,0.05\],\[0.45,0.45\],\[0.95,0.95\] \
        dataset.loader.parameters.generation_parameters.family_parameters.n_graphs=1 \
        model=graph/gin \
        model.feature_encoder.out_channels=32,64,128 \
        model.feature_encoder.proj_dropout=0.3 \
        model.backbone.num_layers=2,4 \
        model.backbone.dropout=0.2,0.4 \
        model.readout.dropout=0.3 \
        model.readout.hidden_layers=\[16\],\[\] \
        dataset.split_params.data_seed=$i \
        dataset.dataloader_params.batch_size=1 \
        logger.wandb.project=final_homophily_experiments_transductive \
        trainer.max_epochs=1000 \
        trainer.min_epochs=50 \
        trainer.devices=\[1\] \
        trainer.check_val_every_n_epoch=1 \
        callbacks.early_stopping.patience=50 \
        tags="[homophily_transductive]" \
        --multirun &

    python -m topobench \
        dataset=graph/GraphUniverse_CD_transductive \
        dataset.loader.parameters.generation_parameters.universe_parameters.seed=$i \
        dataset.loader.parameters.generation_parameters.family_parameters.homophily_range=\[0.05,0.05\],\[0.45,0.45\],\[0.95,0.95\] \
        dataset.loader.parameters.generation_parameters.family_parameters.n_graphs=1 \
        model=pointcloud/deepset \
        model.feature_encoder.out_channels=32,64,128 \
        model.feature_encoder.proj_dropout=0.3 \
        model.readout.hidden_layers=\[64,32\],\[32,16\],\[16\] \
        model.readout.dropout=0.2,0.4 \
        model.readout.dropout=0.3 \
        dataset.split_params.data_seed=$i \
        dataset.dataloader_params.batch_size=1 \
        logger.wandb.project=final_homophily_experiments_transductive \
        trainer.max_epochs=1000 \
        trainer.min_epochs=50 \
        trainer.devices=\[2\] \
        trainer.check_val_every_n_epoch=1 \
        callbacks.early_stopping.patience=50 \
        tags="[homophily_transductive]" \
        --multirun &

    python -m topobench \
        dataset=graph/GraphUniverse_CD_transductive \
        dataset.loader.parameters.generation_parameters.universe_parameters.seed=$i \
        dataset.loader.parameters.generation_parameters.family_parameters.homophily_range=\[0.05,0.05\],\[0.45,0.45\],\[0.95,0.95\] \
        dataset.loader.parameters.generation_parameters.family_parameters.n_graphs=1 \
        model=graph/graph_mlp \
        model.feature_encoder.out_channels=32,64,128 \
        model.feature_encoder.proj_dropout=0.3 \
        model.backbone.order=2,4 \
        model.backbone.dropout=0.2,0.4 \
        model.readout.dropout=0.3 \
        model.readout.hidden_layers=\[16\],\[\] \
        dataset.split_params.data_seed=$i \
        dataset.dataloader_params.batch_size=1 \
        logger.wandb.project=final_homophily_experiments_transductive \
        trainer.max_epochs=1000 \
        trainer.min_epochs=50 \
        trainer.devices=\[3\] \
        trainer.check_val_every_n_epoch=1 \
        callbacks.early_stopping.patience=50 \
        tags="[homophily_transductive]" \
        --multirun &
done