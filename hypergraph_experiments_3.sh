# python -m topobench \
#     dataset=hypergraph/20newsgroup \
#     model=combinatorial/topotune \
#     transforms=exp_combinatorial/h2c_universal_strict \
#     model.feature_encoder.out_channels=32,64 \
#     model.tune_gnn=GCN,GIN,GAT,GraphSAGE \
#     model.backbone.GNN.num_layers=1,2 \
#     model.backbone.neighborhoods=\[up_adjacency-1,up_incidence-0,down_incidence-2,2-up_adjacency-0\] \
#     model.backbone.layers=2,4 \
#     model.feature_encoder.proj_dropout=0.33 \
#     optimizer.parameters.lr=0.01,0.001 \
#     dataset.split_params.data_seed=0,3,5,7,9 \
#     trainer.max_epochs=500 \
#     trainer.min_epochs=50 \
#     trainer.check_val_every_n_epoch=1 \
#     trainer.devices=\[0\] \
#     logger.wandb.project=TopoBench_Hypergraph_TopoTune_test \
#     callbacks.early_stopping.patience=25 \
#     tags="[HypergraphExperiment]" \
#     --multirun &
    
python -m topobench \
    dataset=hypergraph/20newsgroup \
    model=combinatorial/topotune \
    transforms=exp_combinatorial/h2c_universal_strict \
    model.feature_encoder.out_channels=32,64 \
    model.tune_gnn=GCN,GIN,GAT,GraphSAGE \
    model.backbone.GNN.num_layers=1,2 \
    model.backbone.neighborhoods=\[up_adjacency-1,up_incidence-0,down_incidence-2,2-up_adjacency-0\] \
    model.backbone.layers=2,4 \
    model.feature_encoder.proj_dropout=0.33 \
    optimizer.parameters.lr=0.01,0.001 \
    dataset.split_params.data_seed=0,3,5,7,9 \
    trainer.max_epochs=500 \
    trainer.min_epochs=50 \
    trainer.check_val_every_n_epoch=1 \
    trainer.devices=\[0\] \
    logger.wandb.project=TopoBench_Hypergraph_TopoTune \
    callbacks.early_stopping.patience=25 \
    tags="[HypergraphExperiment]" \
    --multirun &

python -m topobench \
    dataset=hypergraph/zoo \
    model=combinatorial/topotune \
    transforms=exp_combinatorial/h2c_universal_strict \
    model.feature_encoder.out_channels=32,64 \
    model.tune_gnn=GCN,GIN,GAT,GraphSAGE \
    model.backbone.GNN.num_layers=1,2 \
    model.backbone.neighborhoods=\[up_adjacency-1,up_incidence-0,down_incidence-2,2-up_adjacency-0\] \
    model.backbone.layers=2,4 \
    model.feature_encoder.proj_dropout=0.33 \
    optimizer.parameters.lr=0.01,0.001 \
    dataset.split_params.data_seed=0,3,5,7,9 \
    trainer.max_epochs=500 \
    trainer.min_epochs=50 \
    trainer.check_val_every_n_epoch=1 \
    trainer.devices=\[0\] \
    logger.wandb.project=TopoBench_Hypergraph_TopoTune \
    callbacks.early_stopping.patience=25 \
    tags="[HypergraphExperiment]" \
    --multirun &  

python -m topobench \
    dataset=hypergraph/NTU2012 \
    model=combinatorial/topotune \
    transforms=exp_combinatorial/h2c_universal_strict \
    model.feature_encoder.out_channels=32,64 \
    model.tune_gnn=GCN,GIN,GAT,GraphSAGE \
    model.backbone.GNN.num_layers=1,2 \
    model.backbone.neighborhoods=\[up_adjacency-1,up_incidence-0,down_incidence-2,2-up_adjacency-0\] \
    model.backbone.layers=2,4 \
    model.feature_encoder.proj_dropout=0.33 \
    optimizer.parameters.lr=0.01,0.001 \
    dataset.split_params.data_seed=0,3,5,7,9 \
    trainer.max_epochs=500 \
    trainer.min_epochs=50 \
    trainer.check_val_every_n_epoch=1 \
    trainer.devices=\[1\] \
    logger.wandb.project=TopoBench_Hypergraph_TopoTune \
    callbacks.early_stopping.patience=25 \
    tags="[HypergraphExperiment]" \
    --multirun &  

python -m topobench \
    dataset=hypergraph/Mushroom \
    model=combinatorial/topotune \
    transforms=exp_combinatorial/h2c_universal_strict \
    model.feature_encoder.out_channels=32,64 \
    model.tune_gnn=GCN,GIN,GAT,GraphSAGE \
    model.backbone.GNN.num_layers=1,2 \
    model.backbone.neighborhoods=\[up_adjacency-1,up_incidence-0,down_incidence-2,2-up_adjacency-0\] \
    model.backbone.layers=2,4 \
    model.feature_encoder.proj_dropout=0.33 \
    optimizer.parameters.lr=0.01,0.001 \
    dataset.split_params.data_seed=0,3,5,7,9 \
    trainer.max_epochs=500 \
    trainer.min_epochs=50 \
    trainer.check_val_every_n_epoch=1 \
    trainer.devices=\[1\] \
    logger.wandb.project=TopoBench_Hypergraph_TopoTune \
    callbacks.early_stopping.patience=25 \
    tags="[HypergraphExperiment]" \
    --multirun &  

python -m topobench \
    dataset=hypergraph/ModelNet40 \
    model=combinatorial/topotune \
    transforms=exp_combinatorial/h2c_universal_strict \
    model.feature_encoder.out_channels=32,64 \
    model.tune_gnn=GCN,GIN,GAT,GraphSAGE \
    model.backbone.GNN.num_layers=1,2 \
    model.backbone.neighborhoods=\[up_adjacency-1,up_incidence-0,down_incidence-2,2-up_adjacency-0\] \
    model.backbone.layers=2,4 \
    model.feature_encoder.proj_dropout=0.33 \
    optimizer.parameters.lr=0.01,0.001 \
    dataset.split_params.data_seed=0,3,5,7,9 \
    trainer.max_epochs=500 \
    trainer.min_epochs=50 \
    trainer.check_val_every_n_epoch=1 \
    trainer.devices=\[2\] \
    logger.wandb.project=TopoBench_Hypergraph_TopoTune \
    callbacks.early_stopping.patience=25 \
    tags="[HypergraphExperiment]" \
    --multirun &  

python -m topobench \
    dataset=hypergraph/cocitation_pubmed \
    model=combinatorial/topotune \
    transforms=exp_combinatorial/h2c_universal_strict \
    model.feature_encoder.out_channels=32,64 \
    model.tune_gnn=GCN,GIN,GAT,GraphSAGE \
    model.backbone.GNN.num_layers=1,2 \
    model.backbone.neighborhoods=\[up_adjacency-1,up_incidence-0,down_incidence-2,2-up_adjacency-0\] \
    model.backbone.layers=2,4 \
    model.feature_encoder.proj_dropout=0.33 \
    optimizer.parameters.lr=0.01,0.001 \
    dataset.split_params.data_seed=0,3,5,7,9 \
    trainer.max_epochs=500 \
    trainer.min_epochs=50 \
    trainer.check_val_every_n_epoch=1 \
    trainer.devices=\[5\] \
    logger.wandb.project=TopoBench_Hypergraph_TopoTune \
    callbacks.early_stopping.patience=25 \
    tags="[HypergraphExperiment]" \
    --multirun &  

python -m topobench \
    dataset=hypergraph/cocitation_cora \
    model=combinatorial/topotune \
    transforms=exp_combinatorial/h2c_universal_strict \
    model.feature_encoder.out_channels=32,64 \
    model.tune_gnn=GCN,GIN,GAT,GraphSAGE \
    model.backbone.GNN.num_layers=1,2 \
    model.backbone.neighborhoods=\[up_adjacency-1,up_incidence-0,down_incidence-2,2-up_adjacency-0\] \
    model.backbone.layers=2,4 \
    model.feature_encoder.proj_dropout=0.33 \
    optimizer.parameters.lr=0.01,0.001 \
    dataset.split_params.data_seed=0,3,5,7,9 \
    trainer.max_epochs=500 \
    trainer.min_epochs=50 \
    trainer.check_val_every_n_epoch=1 \
    trainer.devices=\[3\] \
    logger.wandb.project=TopoBench_Hypergraph_TopoTune \
    callbacks.early_stopping.patience=25 \
    tags="[HypergraphExperiment]" \
    --multirun &  

python -m topobench \
    dataset=hypergraph/cocitation_citeseer \
    model=combinatorial/topotune \
    transforms=exp_combinatorial/h2c_universal_strict \
    model.feature_encoder.out_channels=32,64 \
    model.tune_gnn=GCN,GIN,GAT,GraphSAGE \
    model.backbone.GNN.num_layers=1,2 \
    model.backbone.neighborhoods=\[up_adjacency-1,up_incidence-0,down_incidence-2,2-up_adjacency-0\] \
    model.backbone.layers=2,4 \
    model.feature_encoder.proj_dropout=0.33 \
    optimizer.parameters.lr=0.01,0.001 \
    dataset.split_params.data_seed=0,3,5,7,9 \
    trainer.max_epochs=500 \
    trainer.min_epochs=50 \
    trainer.check_val_every_n_epoch=1 \
    trainer.devices=\[4\] \
    logger.wandb.project=TopoBench_Hypergraph_TopoTune \
    callbacks.early_stopping.patience=25 \
    tags="[HypergraphExperiment]" \
    --multirun &  

python -m topobench \
    dataset=hypergraph/coauthorship_cora \
    model=combinatorial/topotune \
    transforms=exp_combinatorial/h2c_universal_strict \
    model.feature_encoder.out_channels=32,64 \
    model.tune_gnn=GCN,GIN,GAT,GraphSAGE \
    model.backbone.GNN.num_layers=1,2 \
    model.backbone.neighborhoods=\[up_adjacency-1,up_incidence-0,down_incidence-2,2-up_adjacency-0\] \
    model.backbone.layers=2,4 \
    model.feature_encoder.proj_dropout=0.33 \
    optimizer.parameters.lr=0.01,0.001 \
    dataset.split_params.data_seed=0,3,5,7,9 \
    trainer.max_epochs=500 \
    trainer.min_epochs=50 \
    trainer.check_val_every_n_epoch=1 \
    trainer.devices=\[6\] \
    logger.wandb.project=TopoBench_Hypergraph_TopoTune \
    callbacks.early_stopping.patience=25 \
    tags="[HypergraphExperiment]" \
    --multirun &

python -m topobench \
    dataset=hypergraph/coauthorship_dblp \
    model=combinatorial/topotune \
    transforms=exp_combinatorial/h2c_universal_strict \
    model.feature_encoder.out_channels=32,64 \
    model.tune_gnn=GCN,GIN,GAT,GraphSAGE \
    model.backbone.GNN.num_layers=1,2 \
    model.backbone.neighborhoods=\[up_adjacency-1,up_incidence-0,down_incidence-2,2-up_adjacency-0\] \
    model.backbone.layers=2,4 \
    model.feature_encoder.proj_dropout=0.33 \
    optimizer.parameters.lr=0.01,0.001 \
    dataset.split_params.data_seed=0,3,5,7,9 \
    trainer.max_epochs=500 \
    trainer.min_epochs=50 \
    trainer.check_val_every_n_epoch=1 \
    trainer.devices=\[7\] \
    logger.wandb.project=TopoBench_Hypergraph_TopoTune \
    callbacks.early_stopping.patience=25 \
    tags="[HypergraphExperiment]" \
    --multirun &  

