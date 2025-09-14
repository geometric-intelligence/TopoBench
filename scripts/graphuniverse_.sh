#!/bin/bash

data_seeds=(1 3 5)
for i in ${data_seeds[@]}; do
    python -m topobench \
        dataset=graph/GraphUniverse_CD \
        dataset.loader.parameters.generation_parameters.family_parameters.homophily_range=\[0.4,0.6\] \
        dataset.loader.parameters.generation_parameters.family_parameters.n_graphs=1000 \
        dataset.loader.parameters.generation_parameters.family_parameters.avg_degree_range=\[1.0,5.0\],\[5.0,10.0\],\[10.0,20.0\] \
        model=graph/gps \
        model.feature_encoder.out_channels=32,64 \
        model.backbone.num_layers=2,4 \
        model.backbone.heads=4 \
        model.backbone.dropout=0.2,0.4 \
        model.backbone.attn_type=multihead \
        model.readout.hidden_layers=\[16\],\[\] \
        dataset.split_params.data_seed=$i \
        dataset.dataloader_params.batch_size=32 \
        transforms.CombinedPSEs.encodings=\[LapPE\],\[RWSE\] \
        logger.wandb.project=degree_experiments \
        trainer.max_epochs=1000 \
        trainer.min_epochs=50 \
        trainer.devices=\[0\] \
        trainer.check_val_every_n_epoch=1 \
        callbacks.early_stopping.patience=50 \
        tags="[degree]" \
        --multirun &

    python -m topobench \
        dataset=graph/GraphUniverse_CD \
        dataset.loader.parameters.generation_parameters.family_parameters.homophily_range=\[0.4,0.6\] \
        dataset.loader.parameters.generation_parameters.family_parameters.n_graphs=1000 \
        dataset.loader.parameters.generation_parameters.family_parameters.avg_degree_range=\[1.0,5.0\],\[5.0,10.0\],\[10.0,20.0\] \
        model=graph/nsd \
        model.feature_encoder.out_channels=32,64 \
        model.backbone.num_layers=2,4 \
        model.backbone.dropout=0.2,0.4 \
        model.backbone.sheaf_type=diag,bundle \
        model.readout.hidden_layers=\[16\],\[\] \
        dataset.split_params.data_seed=$i \
        dataset.dataloader_params.batch_size=32 \
        transforms.CombinedPSEs.encodings=\[LapPE\],\[RWSE\] \
        logger.wandb.project=degree_experiments \
        trainer.max_epochs=1000 \
        trainer.min_epochs=50 \
        trainer.devices=\[1\] \
        trainer.check_val_every_n_epoch=1 \
        callbacks.early_stopping.patience=50 \
        tags="[degree]" \
        --multirun &

    python -m topobench \
        dataset=graph/GraphUniverse_CD \
        dataset.loader.parameters.generation_parameters.family_parameters.homophily_range=\[0.4,0.6\] \
        dataset.loader.parameters.generation_parameters.family_parameters.n_graphs=1000 \
        dataset.loader.parameters.generation_parameters.family_parameters.avg_degree_range=\[1.0,5.0\],\[5.0,10.0\],\[10.0,20.0\] \
        model=graph/gcn \
        model.feature_encoder.out_channels=32,64,128 \
        model.backbone.num_layers=2,4 \
        model.backbone.dropout=0.2,0.4 \
        model.readout.hidden_layers=\[16\],\[\] \
        dataset.split_params.data_seed=$i \
        dataset.dataloader_params.batch_size=32 \
        logger.wandb.project=degree_experiments \
        trainer.max_epochs=1000 \
        trainer.min_epochs=50 \
        trainer.devices=\[2\] \
        trainer.check_val_every_n_epoch=1 \
        callbacks.early_stopping.patience=50 \
        tags="[degree]" \
        --multirun &

    python -m topobench \
        dataset=graph/GraphUniverse_CD \
        dataset.loader.parameters.generation_parameters.family_parameters.homophily_range=\[0.4,0.6\] \
        dataset.loader.parameters.generation_parameters.family_parameters.n_graphs=1000 \
        dataset.loader.parameters.generation_parameters.family_parameters.avg_degree_range=\[1.0,5.0\],\[5.0,10.0\],\[10.0,20.0\] \
        model=graph/gat \
        model.feature_encoder.out_channels=32,64,128 \
        model.backbone.heads=1,4,8 \
        model.backbone.num_layers=2,4 \
        model.backbone.dropout=0.2,0.4 \
        model.readout.hidden_layers=\[16\],\[\] \
        dataset.split_params.data_seed=$i \
        dataset.dataloader_params.batch_size=32 \
        logger.wandb.project=degree_experiments \
        trainer.max_epochs=1000 \
        trainer.min_epochs=50 \
        trainer.devices=\[3\] \
        trainer.check_val_every_n_epoch=1 \
        callbacks.early_stopping.patience=50 \
        tags="[degree]" \
        --multirun &

    python -m topobench \
        dataset=graph/GraphUniverse_CD \
        dataset.loader.parameters.generation_parameters.family_parameters.homophily_range=\[0.4,0.6\] \
        dataset.loader.parameters.generation_parameters.family_parameters.n_graphs=1000 \
        dataset.loader.parameters.generation_parameters.family_parameters.avg_degree_range=\[1.0,5.0\],\[5.0,10.0\],\[10.0,20.0\] \
        model=graph/sage \
        model.feature_encoder.out_channels=32,64,128 \
        model.backbone.num_layers=2,4 \
        model.backbone.dropout=0.2,0.4 \
        model.readout.hidden_layers=\[16\],\[\] \
        dataset.split_params.data_seed=$i \
        dataset.dataloader_params.batch_size=32 \
        logger.wandb.project=degree_experiments \
        trainer.max_epochs=1000 \
        trainer.min_epochs=50 \
        trainer.devices=\[0\] \
        trainer.check_val_every_n_epoch=1 \
        callbacks.early_stopping.patience=50 \
        tags="[degree]" \
        --multirun &

    python -m topobench \
        dataset=graph/GraphUniverse_CD \
        dataset.loader.parameters.generation_parameters.family_parameters.homophily_range=\[0.4,0.6\] \
        dataset.loader.parameters.generation_parameters.family_parameters.n_graphs=1000 \
        dataset.loader.parameters.generation_parameters.family_parameters.avg_degree_range=\[1.0,5.0\],\[5.0,10.0\],\[10.0,20.0\] \
        model=graph/gin \
        model.feature_encoder.out_channels=32,64,128 \
        model.backbone.num_layers=2,4 \
        model.backbone.dropout=0.2,0.4 \
        model.readout.hidden_layers=\[16\],\[\] \
        dataset.split_params.data_seed=$i \
        dataset.dataloader_params.batch_size=32 \
        logger.wandb.project=degree_experiments \
        trainer.max_epochs=1000 \
        trainer.min_epochs=50 \
        trainer.devices=\[1\] \
        trainer.check_val_every_n_epoch=1 \
        callbacks.early_stopping.patience=50 \
        tags="[degree]" \
        --multirun &

    python -m topobench \
        dataset=graph/GraphUniverse_CD \
        dataset.loader.parameters.generation_parameters.family_parameters.homophily_range=\[0.4,0.6\] \
        dataset.loader.parameters.generation_parameters.family_parameters.n_graphs=1000 \
        dataset.loader.parameters.generation_parameters.family_parameters.avg_degree_range=\[1.0,5.0\],\[5.0,10.0\],\[10.0,20.0\] \
        model=pointcloud/deepset \
        model.feature_encoder.out_channels=32,64,128 \
        model.readout.hidden_layers=\[64,32\],\[32,16\],\[16\] \
        model.readout.dropout=0.2,0.4 \
        dataset.split_params.data_seed=$i \
        dataset.dataloader_params.batch_size=32 \
        logger.wandb.project=degree_experiments \
        trainer.max_epochs=1000 \
        trainer.min_epochs=50 \
        trainer.devices=\[2\] \
        trainer.check_val_every_n_epoch=1 \
        callbacks.early_stopping.patience=50 \
        tags="[degree]" \
        --multirun &

    python -m topobench \
        dataset=graph/GraphUniverse_CD \
        dataset.loader.parameters.generation_parameters.family_parameters.homophily_range=\[0.4,0.6\] \
        dataset.loader.parameters.generation_parameters.family_parameters.n_graphs=1000 \
        dataset.loader.parameters.generation_parameters.family_parameters.avg_degree_range=\[1.0,5.0\],\[5.0,10.0\],\[10.0,20.0\] \
        model=graph/graph_mlp \
        model.feature_encoder.out_channels=32,64,128 \
        model.backbone.order=2,4 \
        model.backbone.dropout=0.2,0.4 \
        model.readout.hidden_layers=\[16\],\[\] \
        dataset.split_params.data_seed=$i \
        dataset.dataloader_params.batch_size=32 \
        logger.wandb.project=degree_experiments \
        trainer.max_epochs=1000 \
        trainer.min_epochs=50 \
        trainer.devices=\[3\] \
        trainer.check_val_every_n_epoch=1 \
        callbacks.early_stopping.patience=50 \
        tags="[degree]" \
        --multirun &
done