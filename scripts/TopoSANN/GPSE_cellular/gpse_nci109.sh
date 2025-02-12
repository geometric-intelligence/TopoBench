# dataset='NCI109'
# project_name="TBX_GPSE_CELL_$dataset"
# #CUDA=2

# # Define available GPUs
# gpus=(4 5 6 7)

# # Define all pretrain models
# #seeds=()

# # Define all pretrain models
# pretrain_models=('ZINC' 'GEOM' 'MOLPCBA' 'PCQM4MV2')
# batch_sizes=(128 256)

# # Distribute across 4 GPUs
# # Each GPU will handle one pretrain model
# for i in {0..3}; do
#     CUDA=${gpus[$i]}  # Use the GPU number from our gpus array
#     pretrain_model=${pretrain_models[$i]}
    
#     for batch_size in ${batch_sizes[*]}
#     do
#     CUDA_LAUNCH_BLOCKING=1 python topobenchmarkx/run.py\
#         dataset=graph/$dataset\
#         model=cell/sann\
#         model.backbone.n_layers=2,4\
#         model.feature_encoder.proj_dropout=0.25\
#         dataset.split_params.data_seed=0,1,2,4\
#         dataset.dataloader_params.batch_size=$batch_size\
#         model.feature_encoder.out_channels=64,128\
#         transforms.sann_encoding.pretrain_model=$pretrain_model\
#         optimizer.parameters.weight_decay=0,0.0001\
#         optimizer.parameters.lr=0.01,0.001\
#         trainer.max_epochs=500\
#         trainer.min_epochs=50\
#         optimizer.scheduler=null\
#         trainer.check_val_every_n_epoch=5\
#         trainer.devices=\[$CUDA\]\
#         callbacks.early_stopping.patience=10\
#         transforms/data_manipulations@transforms.sann_encoding=add_gpse_information\
#         logger.wandb.project=$project_name\
#         transforms.sann_encoding.neighborhoods='[incidence_1,incidence_0,0_incidence_0]','[0_incidence_0,incidence_0]','[0_incidence_0,incidence_1,0_incidence_1,incidence_0]'\
#         --multirun
#     done
# done


dataset='NCI109'
project_name="TBX_GPSE_SIMP_TEST_$dataset"
#CUDA=2

# Define available GPUs
gpus=(4)

# Define all pretrain models
#seeds=()

# Define all pretrain models
pretrain_models=('ZINC')
batch_sizes=(512)

# Distribute across 4 GPUs
# Each GPU will handle one pretrain model

CUDA=${gpus[0]}  # Use the GPU number from our gpus array
pretrain_model=${pretrain_models[$i]}

for batch_size in ${batch_sizes[*]}
do
HYDRA_FULL_ERROR=1 CUDA_LAUNCH_BLOCKING=1 python topobenchmarkx/run.py\
    dataset=graph/$dataset\
    model=cell/sann\
    model.backbone.n_layers=2,4\
    model.feature_encoder.proj_dropout=0.25\
    dataset.split_params.data_seed=0,1,2,4\
    dataset.dataloader_params.batch_size=$batch_size\
    model.feature_encoder.out_channels=64,128\
    transforms.sann_encoding.pretrain_model=$pretrain_model\
    optimizer.parameters.weight_decay=0,0.0001\
    optimizer.parameters.lr=0.01,0.001\
    trainer.max_epochs=500\
    trainer.min_epochs=50\
    optimizer.scheduler=null\
    trainer.check_val_every_n_epoch=5\
    trainer.devices=\[$CUDA\]\
    callbacks.early_stopping.patience=10\
    transforms/data_manipulations@transforms.sann_encoding=add_gpse_information\
    logger.wandb.project=$project_name\
    transforms.sann_encoding.neighborhoods='[incidence_1,incidence_0,0_incidence_0]'\
    --multirun
done



# transforms.sann_encoding.neighborhoods='[incidence_1,incidence_0]','[incidence_0]','[incidence_1,0_incidence_1,incidence_0]'\