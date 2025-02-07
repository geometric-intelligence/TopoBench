dataset='MUTAG'
project_name="TBX_GPSE_CELL_$dataset"
CUDA=5

pretrain_models=('ZINC' 'GEOM' 'MOLPCBA' 'PCQM4MV2')
for pretrain_model in ${pretrain_models[*]}
do
    python topobenchmarkx/run.py\
        dataset=graph/$dataset\
        model=cell/sann\
        model.backbone.n_layers=2,4\
        model.feature_encoder.out_channels=128\
        model.feature_encoder.proj_dropout=0.25\
        dataset.split_params.data_seed=0,1,2,4\
        dataset.dataloader_params.batch_size=$batch_size\
        trainer.max_epochs=500\
        trainer.min_epochs=50\
        trainer.devices=\[$CUDA\]\
        optimizer.scheduler=null\
        trainer.check_val_every_n_epoch=5\
        logger.wandb.project=$project_name\
        optimizer.parameters.lr=0.01,0.001\
        optimizer.parameters.weight_decay=0,0.0001\
        callbacks.early_stopping.patience=10\
        +transforms/data_manipulations@transforms.node_deg=node_degrees \
        +transforms/data_manipulations@transforms.node_feat=one_hot_node_degree_features \
        transforms/data_manipulations@transforms.sann_encoding=add_gpse_information\
        transforms.sann_encoding.pretrain_model=$pretrain_model\
        transforms.sann_encoding.copy_initial=True \
        transforms.sann_encoding.neighborhoods='[incidence_1,incidence_0,0_incidence_0]','[0_incidence_0,incidence_0]','[0_incidence_0,incidence_1,0_incidence_1,incidence_0]'\
        --multirun &
done

#python -m topobenchmarkx dataset=graph/MUTAG model.feature_encoder.out_channels=32 model=simplicial/sann model.backbone.n_layers=2 model.feature_encoder.proj_dropout=0 dataset.split_params.data_seed=0 dataset.dataloader_params.batch_size=32 trainer.max_epochs=500 trainer.min_epochs=50 trainer.devices=0 trainer.check_val_every_n_epoch=5 optimizer.scheduler=null optimizer.parameters.lr=0.001 optimizer.parameters.weight_decay=0.0001 callbacks.early_stopping.patience=10 transforms/data_manipulations@transforms.sann_encoding=add_gpse_information transforms.sann_encoding.pretrain_model=ZINC logger.wandb.project=TBX_GPSE_MUTAG transforms.sann_encoding.neighborhoods='[incidence_1,0_incidence_1,incidence_0]'
            