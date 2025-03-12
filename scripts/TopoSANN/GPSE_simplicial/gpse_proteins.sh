dataset='PROTEINS'
project_name="TBX_GPSE_$dataset"

CUDA=3
gpus=(0 1 2 3)
pretrain_models=('ZINC' 'GEOM')
for pretrain_model in ${pretrain_models[*]}
do
    for seed in ${seeds[*]}
    do
    python topobenchmarkx/run.py\
        dataset=graph/$dataset\
        model=simplicial/sann\
        model.backbone.n_layers=2,4\
        model.feature_encoder.proj_dropout=0.25\
        dataset.split_params.data_seed=$seed\
        dataset.dataloader_params.batch_size=128,256\
        model.feature_encoder.out_channels=64,128\
        transforms.sann_encoding.pretrain_model=$pretrain_model\
        optimizer.parameters.weight_decay=0,0.0001\
        optimizer.parameters.lr=0.01,0.001\
        trainer.max_epochs=500\
        trainer.min_epochs=50\
        trainer.devices=\[$CUDA\]\
        optimizer.scheduler=null\
        trainer.check_val_every_n_epoch=5\
        callbacks.early_stopping.patience=10\
        transforms/data_manipulations@transforms.sann_encoding=add_gpse_information\
        logger.wandb.project=$project_name\
        transforms.sann_encoding.neighborhoods='[incidence_1,incidence_0,0_incidence_0]','[0_incidence_0,incidence_0]','[0_incidence_0,incidence_1,0_incidence_1,incidence_0]'\
        --multirun &
    done
done



CUDA=4
pretrain_models_2=('MOLPCBA' 'PCQM4MV2')
for pretrain_model in ${pretrain_models_2[*]}
do
    for seed in ${seeds[*]}
    do
    python topobenchmarkx/run.py\
        dataset=graph/$dataset\
        model=simplicial/sann\
        model.backbone.n_layers=2,4\
        model.feature_encoder.proj_dropout=0.25\
        dataset.split_params.data_seed=$seed\
        dataset.dataloader_params.batch_size=128,256\
        model.feature_encoder.out_channels=64,128\
        transforms.sann_encoding.pretrain_model=$pretrain_model\
        optimizer.parameters.weight_decay=0,0.0001\
        optimizer.parameters.lr=0.01,0.001\
        trainer.max_epochs=500\
        trainer.min_epochs=50\
        trainer.devices=\[$CUDA\]\
        optimizer.scheduler=null\
        trainer.check_val_every_n_epoch=5\
        callbacks.early_stopping.patience=10\
        transforms/data_manipulations@transforms.sann_encoding=add_gpse_information\
        logger.wandb.project=$project_name\
        transforms.sann_encoding.neighborhoods='[incidence_1,incidence_0]','[incidence_0]','[incidence_1,0_incidence_1,incidence_0]'\
        --multirun &
    done
done

wait
# transforms.sann_encoding.neighborhoods='[incidence_1,incidence_0]','[incidence_0]','[incidence_1,0_incidence_1,incidence_0]'\