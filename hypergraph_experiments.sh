python -m topobench \
    dataset=hypergraph/20newsgroup \
    model=hypergraph/edgnn,hypergraph/allsettransformer,hypergraph/unignn2 \
    model.feature_encoder.out_channels=32,64,128 \
    ++model.backbone.n_layers=1,2,3,4 \
    model.feature_encoder.proj_dropout=0.25,0.5 \
    optimizer.parameters.lr=0.01,0.001 \
    dataset.split_params.data_seed=0,3,5,7,9 \
    trainer.max_epochs=500 \
    trainer.min_epochs=50 \
    trainer.check_val_every_n_epoch=1 \
    trainer.devices=\[0\] \
    logger.wandb.project=TopoBench_Hypergraph \
    callbacks.early_stopping.patience=25 \
    tags="[HypergraphExperiment]" \
    --multirun &

python -m topobench \
    dataset=hypergraph/zoo \
    model=hypergraph/edgnn,hypergraph/allsettransformer,hypergraph/unignn2 \
    model.feature_encoder.out_channels=32,64,128 \
    ++model.backbone.n_layers=1,2,3,4 \
    model.feature_encoder.proj_dropout=0.25,0.5 \
    optimizer.parameters.lr=0.01,0.001 \
    dataset.split_params.data_seed=0,3,5,7,9 \
    trainer.max_epochs=500 \
    trainer.min_epochs=50 \
    trainer.check_val_every_n_epoch=1 \
    trainer.devices=\[1\] \
    logger.wandb.project=TopoBench_Hypergraph \
    callbacks.early_stopping.patience=25 \
    tags="[HypergraphExperiment]" \
    --multirun &  

python -m topobench \
    dataset=hypergraph/NTU2012 \
    model=hypergraph/edgnn,hypergraph/allsettransformer,hypergraph/unignn2 \
    model.feature_encoder.out_channels=32,64,128 \
    ++model.backbone.n_layers=1,2,3,4 \
    model.feature_encoder.proj_dropout=0.25,0.5 \
    optimizer.parameters.lr=0.01,0.001 \
    dataset.split_params.data_seed=0,3,5,7,9 \
    trainer.max_epochs=500 \
    trainer.min_epochs=50 \
    trainer.check_val_every_n_epoch=1 \
    trainer.devices=\[2\] \
    logger.wandb.project=TopoBench_Hypergraph \
    callbacks.early_stopping.patience=25 \
    tags="[HypergraphExperiment]" \
    --multirun &  

python -m topobench \
    dataset=hypergraph/Mushroom \
    model=hypergraph/edgnn,hypergraph/allsettransformer,hypergraph/unignn2 \
    model.feature_encoder.out_channels=32,64,128 \
    ++model.backbone.n_layers=1,2,3,4 \
    model.feature_encoder.proj_dropout=0.25,0.5 \
    optimizer.parameters.lr=0.01,0.001 \
    dataset.split_params.data_seed=0,3,5,7,9 \
    trainer.max_epochs=500 \
    trainer.min_epochs=50 \
    trainer.check_val_every_n_epoch=1 \
    trainer.devices=\[3\] \
    logger.wandb.project=TopoBench_Hypergraph \
    callbacks.early_stopping.patience=25 \
    tags="[HypergraphExperiment]" \
    --multirun &  

python -m topobench \
    dataset=hypergraph/ModelNet40 \
    model=hypergraph/edgnn,hypergraph/allsettransformer,hypergraph/unignn2 \
    model.feature_encoder.out_channels=32,64,128 \
    ++model.backbone.n_layers=1,2,3,4 \
    model.feature_encoder.proj_dropout=0.25,0.5 \
    optimizer.parameters.lr=0.01,0.001 \
    dataset.split_params.data_seed=0,3,5,7,9 \
    trainer.max_epochs=500 \
    trainer.min_epochs=50 \
    trainer.check_val_every_n_epoch=1 \
    trainer.devices=\[4\] \
    logger.wandb.project=TopoBench_Hypergraph \
    callbacks.early_stopping.patience=25 \
    tags="[HypergraphExperiment]" \
    --multirun &  

python -m topobench \
    dataset=hypergraph/cocitation_pubmed \
    model=hypergraph/edgnn,hypergraph/allsettransformer,hypergraph/unignn2 \
    model.feature_encoder.out_channels=32,64,128 \
    ++model.backbone.n_layers=1,2,3,4 \
    model.feature_encoder.proj_dropout=0.25,0.5 \
    optimizer.parameters.lr=0.01,0.001 \
    dataset.split_params.data_seed=0,3,5,7,9 \
    trainer.max_epochs=500 \
    trainer.min_epochs=50 \
    trainer.check_val_every_n_epoch=1 \
    trainer.devices=\[5\] \
    logger.wandb.project=TopoBench_Hypergraph \
    callbacks.early_stopping.patience=25 \
    tags="[HypergraphExperiment]" \
    --multirun &  

python -m topobench \
    dataset=hypergraph/cocitation_cora \
    model=hypergraph/edgnn,hypergraph/allsettransformer,hypergraph/unignn2 \
    model.feature_encoder.out_channels=32,64,128 \
    ++model.backbone.n_layers=1,2,3,4 \
    model.feature_encoder.proj_dropout=0.25,0.5 \
    optimizer.parameters.lr=0.01,0.001 \
    dataset.split_params.data_seed=0,3,5,7,9 \
    trainer.max_epochs=500 \
    trainer.min_epochs=50 \
    trainer.check_val_every_n_epoch=1 \
    trainer.devices=\[6\] \
    logger.wandb.project=TopoBench_Hypergraph \
    callbacks.early_stopping.patience=25 \
    tags="[HypergraphExperiment]" \
    --multirun &  

python -m topobench \
    dataset=hypergraph/cocitation_citeseer \
    model=hypergraph/edgnn,hypergraph/allsettransformer,hypergraph/unignn2 \
    model.feature_encoder.out_channels=32,64,128 \
    ++model.backbone.n_layers=1,2,3,4 \
    model.feature_encoder.proj_dropout=0.25,0.5 \
    optimizer.parameters.lr=0.01,0.001 \
    dataset.split_params.data_seed=0,3,5,7,9 \
    trainer.max_epochs=500 \
    trainer.min_epochs=50 \
    trainer.check_val_every_n_epoch=1 \
    trainer.devices=\[7\] \
    logger.wandb.project=TopoBench_Hypergraph \
    callbacks.early_stopping.patience=25 \
    tags="[HypergraphExperiment]" \
    --multirun &  

python -m topobench \
    dataset=hypergraph/coauthorship_cora \
    model=hypergraph/edgnn \
    model.feature_encoder.out_channels=32,64,128 \
    ++model.backbone.n_layers=1,2,3,4 \
    model.feature_encoder.proj_dropout=0.25,0.5 \
    optimizer.parameters.lr=0.01,0.001 \
    dataset.split_params.data_seed=0,3,5,7,9 \
    trainer.max_epochs=500 \
    trainer.min_epochs=50 \
    trainer.check_val_every_n_epoch=1 \
    trainer.devices=\[0\] \
    logger.wandb.project=TopoBench_Hypergraph \
    callbacks.early_stopping.patience=25 \
    tags="[HypergraphExperiment]" \
    --multirun &

python -m topobench \
    dataset=hypergraph/coauthorship_dblp \
    model=hypergraph/edgnn \
    model.feature_encoder.out_channels=32,64,128 \
    ++model.backbone.n_layers=1,2,3,4 \
    model.feature_encoder.proj_dropout=0.25,0.5 \
    optimizer.parameters.lr=0.01,0.001 \
    dataset.split_params.data_seed=0,3,5,7,9 \
    trainer.max_epochs=500 \
    trainer.min_epochs=50 \
    trainer.check_val_every_n_epoch=1 \
    trainer.devices=\[1\] \
    logger.wandb.project=TopoBench_Hypergraph \
    callbacks.early_stopping.patience=25 \
    tags="[HypergraphExperiment]" \
    --multirun &  

python -m topobench \
    dataset=hypergraph/coauthorship_cora \
    model=hypergraph/allsettransformer \
    model.feature_encoder.out_channels=32,64,128 \
    ++model.backbone.n_layers=1,2,3,4 \
    model.feature_encoder.proj_dropout=0.25,0.5 \
    optimizer.parameters.lr=0.01,0.001 \
    dataset.split_params.data_seed=0,3,5,7,9 \
    trainer.max_epochs=500 \
    trainer.min_epochs=50 \
    trainer.check_val_every_n_epoch=1 \
    trainer.devices=\[2\] \
    logger.wandb.project=TopoBench_Hypergraph \
    callbacks.early_stopping.patience=25 \
    tags="[HypergraphExperiment]" \
    --multirun &  

python -m topobench \
    dataset=hypergraph/coauthorship_dblp \
    model=hypergraph/allsettransformer \
    model.feature_encoder.out_channels=32,64,128 \
    ++model.backbone.n_layers=1,2,3,4 \
    model.feature_encoder.proj_dropout=0.25,0.5 \
    optimizer.parameters.lr=0.01,0.001 \
    dataset.split_params.data_seed=0,3,5,7,9 \
    trainer.max_epochs=500 \
    trainer.min_epochs=50 \
    trainer.check_val_every_n_epoch=1 \
    trainer.devices=\[3\] \
    logger.wandb.project=TopoBench_Hypergraph \
    callbacks.early_stopping.patience=25 \
    tags="[HypergraphExperiment]" \
    --multirun &  

python -m topobench \
    dataset=hypergraph/coauthorship_cora \
    model=hypergraph/unignn2 \
    model.feature_encoder.out_channels=32,64,128 \
    ++model.backbone.n_layers=1,2,3,4 \
    model.feature_encoder.proj_dropout=0.25,0.5 \
    optimizer.parameters.lr=0.01,0.001 \
    dataset.split_params.data_seed=0,3,5,7,9 \
    trainer.max_epochs=500 \
    trainer.min_epochs=50 \
    trainer.check_val_every_n_epoch=1 \
    trainer.devices=\[4\] \
    logger.wandb.project=TopoBench_Hypergraph \
    callbacks.early_stopping.patience=25 \
    tags="[HypergraphExperiment]" \
    --multirun &  

python -m topobench \
    dataset=hypergraph/coauthorship_dblp \
    model=hypergraph/unignn2 \
    model.feature_encoder.out_channels=32,64,128 \
    ++model.backbone.n_layers=1,2,3,4 \
    model.feature_encoder.proj_dropout=0.25,0.5 \
    optimizer.parameters.lr=0.01,0.001 \
    dataset.split_params.data_seed=0,3,5,7,9 \
    trainer.max_epochs=500 \
    trainer.min_epochs=50 \
    trainer.check_val_every_n_epoch=1 \
    trainer.devices=\[5\] \
    logger.wandb.project=TopoBench_Hypergraph \
    callbacks.early_stopping.patience=25 \
    tags="[HypergraphExperiment]" \
    --multirun &  