#!/bin/bash
# Best runs generated for project: TopoBench_Reproduction
# Contains ONLY model, dataset, varying HPs, and seeds.

python -m topobench model=GCN dataset=hm-categories model.readout.hidden_dim=64 model.backbone.dropout=0 model.backbone.num_layers=2 model.backbone.in_channels=64 model.backbone.hidden_channels=64 model.feature_encoder.out_channels=64 model.feature_encoder.proj_dropout=0 model.backbone_wrapper.out_channels=64 optimizer.parameters.lr=0.003 dataset.split_params.data_seed=0,3,5,7,9 logger.wandb.project=TopoBench_Reproduction --multirun

python -m topobench model=GIN dataset=hm-categories model.readout.hidden_dim=64 model.backbone.dropout=0 model.backbone.num_layers=1 model.backbone.in_channels=64 model.backbone.hidden_channels=64 model.feature_encoder.out_channels=64 model.feature_encoder.proj_dropout=0 model.backbone_wrapper.out_channels=64 optimizer.parameters.lr=0.003 dataset.split_params.data_seed=0,3,5,7,9 logger.wandb.project=TopoBench_Reproduction --multirun

python -m topobench model=GCN dataset=pokec-regions model.readout.hidden_dim=32 model.backbone.dropout=0 model.backbone.num_layers=8 model.backbone.in_channels=32 model.backbone.hidden_channels=32 model.feature_encoder.out_channels=32 model.feature_encoder.proj_dropout=0 model.backbone_wrapper.out_channels=32 optimizer.parameters.lr=0.003 dataset.split_params.data_seed=0,3,5,7,9 logger.wandb.project=TopoBench_Reproduction --multirun

python -m topobench model=GCN dataset=tolokers-2 model.readout.hidden_dim=64 model.backbone.dropout=0 model.backbone.num_layers=3 model.backbone.in_channels=64 model.backbone.hidden_channels=64 model.feature_encoder.out_channels=64 model.feature_encoder.proj_dropout=0 model.backbone_wrapper.out_channels=64 optimizer.parameters.lr=0.003 dataset.split_params.data_seed=0,3,5,7,9 logger.wandb.project=TopoBench_Reproduction --multirun

python -m topobench model=GCN dataset=city-reviews model.readout.hidden_dim=64 model.backbone.dropout=0.1 model.backbone.num_layers=8 model.backbone.in_channels=64 model.backbone.hidden_channels=64 model.feature_encoder.out_channels=64 model.feature_encoder.proj_dropout=0 model.backbone_wrapper.out_channels=64 optimizer.parameters.lr=0.003 dataset.split_params.data_seed=0,3,5,7,9 logger.wandb.project=TopoBench_Reproduction --multirun

python -m topobench model=GCN dataset=artnet-exp model.readout.hidden_dim=64 model.backbone.dropout=0 model.backbone.num_layers=4 model.backbone.in_channels=64 model.backbone.hidden_channels=64 model.feature_encoder.out_channels=64 model.feature_encoder.proj_dropout=0.1 model.backbone_wrapper.out_channels=64 optimizer.parameters.lr=0.003 dataset.split_params.data_seed=0,3,5,7,9 logger.wandb.project=TopoBench_Reproduction --multirun

