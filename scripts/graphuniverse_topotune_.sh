#!/bin/bash


data_seeds=(1 3 5)

for i in ${data_seeds[@]}; do
    python -m topobench \
        dataset=graph/GraphUniverse_CD \
        dataset.loader.parameters.generation_parameters.universe_parameters.seed=$i \
        dataset.loader.parameters.generation_parameters.family_parameters.homophily_range=\[0.4,0.6\] \
        dataset.loader.parameters.generation_parameters.family_parameters.avg_degree_range=\[5.0,10.0\],\[10.0,20.0\] \
        dataset.loader.parameters.generation_parameters.family_parameters.n_graphs=1000 \
        model=cell/topotune \
        model.feature_encoder.out_channels=32 \
        model.feature_encoder.proj_dropout=0.3 \
        model.tune_gnn=GCN,GIN,GAT,GraphSAGE \
        model.backbone.GNN.num_layers=1 \
        model.backbone.neighborhoods=\[1-up_laplacian-0\],\[1-up_laplacian-0,1-up_laplacian-1\],\[1-up_laplacian-0,1-down_laplacian-2\],\[1-up_laplacian-0,1-down_incidence-2\],\[1-up_laplacian-0,2-down_laplacian-2\],\[1-up_laplacian-0,1-up_incidence-0,1-up_laplacian-1,1-up_incidence-1\] \
        model.backbone.layers=2,4,8 \
        model.readout.readout_name=PropagateSignalDown \
        model.readout.pooling_type=mean,sum \
        dataset.split_params.data_seed=$i \
        dataset.dataloader_params.batch_size=32 \
        logger.wandb.project=final_degree_experiments \
        trainer.max_epochs=1000 \
        trainer.min_epochs=50 \
        trainer.devices=\[0\] \
        trainer.check_val_every_n_epoch=1 \
        callbacks.early_stopping.patience=50 \
        tags="[degree]" \
        --multirun &

    sleep 100
    
    python -m topobench \
        dataset=graph/GraphUniverse_CD \
        dataset.loader.parameters.generation_parameters.universe_parameters.seed=$i \
        dataset.loader.parameters.generation_parameters.family_parameters.homophily_range=\[0.4,0.6\] \
        dataset.loader.parameters.generation_parameters.family_parameters.avg_degree_range=\[5.0,10.0\],\[10.0,20.0\] \
        dataset.loader.parameters.generation_parameters.family_parameters.n_graphs=1000 \
        model=cell/topotune \
        model.feature_encoder.out_channels=32 \
        model.feature_encoder.proj_dropout=0.3 \
        model.tune_gnn=GCN,GIN,GAT,GraphSAGE \
        model.backbone.GNN.num_layers=1 \
        model.backbone.neighborhoods=\[1-up_laplacian-0\],\[1-up_laplacian-0,1-up_laplacian-1\],\[1-up_laplacian-0,1-down_laplacian-2\],\[1-up_laplacian-0,1-down_incidence-2\],\[1-up_laplacian-0,2-down_laplacian-2\],\[1-up_laplacian-0,1-up_incidence-0,1-up_laplacian-1,1-up_incidence-1\] \
        model.backbone.layers=2,4,8 \
        model.readout.readout_name=MLPReadout \
        model.readout.hidden_layers=\[16\],\[\] \
        model.readout.dropout=0.3 \
        dataset.split_params.data_seed=$i \
        dataset.dataloader_params.batch_size=32 \
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
        dataset.loader.parameters.generation_parameters.universe_parameters.seed=$i \
        dataset.loader.parameters.generation_parameters.family_parameters.homophily_range=\[0.4,0.6\] \
        dataset.loader.parameters.generation_parameters.family_parameters.avg_degree_range=\[5.0,10.0\],\[10.0,20.0\] \
        dataset.loader.parameters.generation_parameters.family_parameters.n_graphs=1000 \
        model=simplicial/topotune \
        model.feature_encoder.out_channels=32 \
        model.feature_encoder.proj_dropout=0.3 \
        model.tune_gnn=GCN,GIN,GAT,GraphSAGE \
        model.backbone.GNN.num_layers=1 \
        model.backbone.neighborhoods=\[1-up_laplacian-0\],\[1-up_laplacian-0,1-up_laplacian-1\],\[1-up_laplacian-0,1-down_laplacian-2\],\[1-up_laplacian-0,1-down_incidence-2\],\[1-up_laplacian-0,2-down_laplacian-2\],\[1-up_laplacian-0,1-up_incidence-0,1-up_laplacian-1,1-up_incidence-1\] \
        model.backbone.layers=2,4,8 \
        model.readout.readout_name=PropagateSignalDown \
        model.readout.pooling_type=mean,sum \
        dataset.split_params.data_seed=$i \
        dataset.dataloader_params.batch_size=32 \
        logger.wandb.project=final_degree_experiments \
        trainer.max_epochs=1000 \
        trainer.min_epochs=50 \
        trainer.devices=\[2\] \
        trainer.check_val_every_n_epoch=1 \
        callbacks.early_stopping.patience=50 \
        tags="[degree]" \
        --multirun &

    sleep 100
    
    python -m topobench \
        dataset=graph/GraphUniverse_CD \
        dataset.loader.parameters.generation_parameters.universe_parameters.seed=$i \
        dataset.loader.parameters.generation_parameters.family_parameters.homophily_range=\[0.4,0.6\] \
        dataset.loader.parameters.generation_parameters.family_parameters.avg_degree_range=\[5.0,10.0\],\[10.0,20.0\] \
        dataset.loader.parameters.generation_parameters.family_parameters.n_graphs=1000 \
        model=simplicial/topotune \
        model.feature_encoder.out_channels=32 \
        model.feature_encoder.proj_dropout=0.3 \
        model.tune_gnn=GCN,GIN,GAT,GraphSAGE \
        model.backbone.GNN.num_layers=1 \
        model.backbone.neighborhoods=\[1-up_laplacian-0\],\[1-up_laplacian-0,1-up_laplacian-1\],\[1-up_laplacian-0,1-down_laplacian-2\],\[1-up_laplacian-0,1-down_incidence-2\],\[1-up_laplacian-0,2-down_laplacian-2\],\[1-up_laplacian-0,1-up_incidence-0,1-up_laplacian-1,1-up_incidence-1\] \
        model.backbone.layers=2,4,8 \
        model.readout.readout_name=MLPReadout \
        model.readout.hidden_layers=\[16\],\[\] \
        model.readout.dropout=0.3 \
        dataset.split_params.data_seed=$i \
        dataset.dataloader_params.batch_size=32 \
        logger.wandb.project=final_degree_experiments \
        trainer.max_epochs=1000 \
        trainer.min_epochs=50 \
        trainer.devices=\[3\] \
        trainer.check_val_every_n_epoch=1 \
        callbacks.early_stopping.patience=50 \
        tags="[degree]" \
        --multirun &
done