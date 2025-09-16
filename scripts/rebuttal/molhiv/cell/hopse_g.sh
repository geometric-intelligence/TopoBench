#!/bin/bash

dataset='ogbg-molhiv'
project_name="rebutal_HOPSE_G_cell_$dataset"

# =====================
# DATA
# =====================
DATA_SEEDS=(0 3 5 7 9) 

# =====================
# MODEL PARAMETERS
# =====================
N_LAYERS=(1 2 3 4)
OUT_CHANNELS=(64 128 256)

# =====================
# OPTIMIZATION PARAMETERS
# =====================
LEARNING_RATES=(0.01 0.001)
PROJECTION_DROPOUTS=(0.25 0.5)
WEIGHT_DECAYS=(0.0 0.0001)
BATCH_SIZES=(128 256)

# =====================
# PRETRAINED MODELS
# =====================
PRETRAIN_MODELS=('ZINC') #'GEOM' 'MOLPCBA' 'PCQM4MV2'

# =====================
# CONVERT TO STRINGS
# =====================
DATA_SEEDS_STR=$(IFS=,; echo "${DATA_SEEDS[*]}")  # Convert to comma-separated string
N_LAYERS_STR=$(IFS=,; echo "${N_LAYERS[*]}")  # Convert to comma-separated string
OUT_CHANNELS_STR=$(IFS=,; echo "${OUT_CHANNELS[*]}")  # Convert to comma-separated string
LEARNING_RATES_STR=$(IFS=,; echo "${LEARNING_RATES[*]}")  # Convert to comma-separated string
PROJECTION_DROPOUTS_STR=$(IFS=,; echo "${PROJECTION_DROPOUTS[*]}")  # Convert to comma-separated string
WEIGHT_DECAYS_STR=$(IFS=,; echo "${WEIGHT_DECAYS[*]}")  # Convert to comma-separated string
PRETRAIN_MODELS_STR=$(IFS=,; echo "${PRETRAIN_MODELS[*]}")  # Convert to comma-separated string
BATCH_SIZES_STR=$(IFS=,; echo "${BATCH_SIZES[*]}")

# =====================
# PARAMETERS OVER WHICH WE PERFORM PARALLEL RUNS
# =====================
neighborhoods=(
    # adjacency 
    "['up_adjacency-0']"
    "['up_adjacency-0','up_adjacency-1']"
    "['up_adjacency-0','up_adjacency-1','down_adjacency-2']"

    # incidence
    "['up_adjacency-0','up_incidence-0','up_incidence-1']"
    "['up_adjacency-0','down_incidence-1','down_incidence-2']"
    "['up_adjacency-0','up_incidence-0','up_incidence-1','down_incidence-1','down_incidence-2']"
    
    # all together
    "['up_adjacency-0','up_adjacency-1','down_adjacency-1','down_adjacency-2','up_incidence-0','up_incidence-1','down_incidence-1','down_incidence-2']"
    
    # We have 8th gpu hence we can add one more neighbourhood
    "['up_adjacency-0','up_adjacency-1','2-up_adjacency-0','down_adjacency-1','down_adjacency-2','2-down_adjacency-2']"
)

gpus=(0 1 2 3 4 5 6 7)
# for i in {0..7}; do 
#     CUDA=${gpus[$i]}  # Use the GPU number from our gpus array
#     neighborhood=${neighborhoods[$i]} # Use the neighbourhood from our neighbourhoods array

#     for pretrain_model in ${PRETRAIN_MODELS[*]}
#     do 
#         python topobench/run.py\
#             dataset=graph/$dataset\
#             model=cell/hopse_g\
#             model.readout.readout_name=SANNReadout\
#             model.backbone.n_layers=1\
#             model.feature_encoder.out_channels=128\
#             model.feature_encoder.proj_dropout=0.5\
#             model.feature_encoder.use_atom_encoder=True\
#             model.feature_encoder.use_bond_encoder=True\
#             dataset.split_params.data_seed=0\
#             dataset.dataloader_params.batch_size=128\
#             trainer.max_epochs=5\
#             trainer.min_epochs=1\
#             trainer.devices=\[$CUDA\]\
#             trainer.check_val_every_n_epoch=1\
#             callbacks.early_stopping.patience=10\
#             logger.wandb.project='prerun'\
#             optimizer.parameters.lr=0.01\
#             optimizer.parameters.weight_decay=0.25\
#             transforms.sann_encoding.neighborhoods=$neighborhood\
#             transforms.sann_encoding.pretrain_model=$pretrain_model\
#             transforms.graph2cell_lifting.neighborhoods=$neighborhood\
#             transforms.graph2cell_lifting.max_cell_length=10\
#             --multirun
#             sleep 5
#     done
# done
# wait

gpus=(0 1 2 3 4 5 6 7)
# for i in {0..7}; do[] 
#     CUDA=${gpus[$i]}  # Use the GPU number from our gpus array
#     neighborhood=${neighborhoods[$i]} # Use the neighbourhood from our neighbourhoods array

#     for pretrain_model in ${PRETRAIN_MODELS[*]}
#     do
#         for batch_size in ${BATCH_SIZES[*]}
#         do
        
#             python topobench/run.py\
#                 dataset=graph/$dataset\
#                 model=cell/hopse_g\
#                 model.readout.readout_name=SANNReadout\
#                 model.backbone.n_layers=$N_LAYERS_STR\
#                 model.feature_encoder.out_channels=$OUT_CHANNELS_STR\
#                 model.feature_encoder.proj_dropout=$PROJECTION_DROPOUTS_STR\
#                 model.feature_encoder.use_atom_encoder=True\
#                 model.feature_encoder.use_bond_encoder=True\
#                 dataset.split_params.data_seed=$DATA_SEEDS_STR\
#                 dataset.dataloader_params.batch_size=$BATCH_SIZES_STR\
#                 trainer.max_epochs=100\
#                 trainer.min_epochs=10\
#                 trainer.devices=\[$CUDA\]\
#                 trainer.check_val_every_n_epoch=5\
#                 callbacks.early_stopping.patience=10\
#                 logger.wandb.project=$project_name\
#                 optimizer.parameters.lr=$LEARNING_RATES_STR\
#                 optimizer.parameters.weight_decay=$WEIGHT_DECAYS_STR\
#                 transforms.sann_encoding.neighborhoods=$neighborhood\
#                 transforms.sann_encoding.pretrain_model=$pretrain_model\
#                 transforms.graph2cell_lifting.neighborhoods=$neighborhood\
#                 transforms.graph2cell_lifting.max_cell_length=10\
#                 --multirun
#         done
#     done
# done

gpus=(0 1 2 3 4 5 6 7)
num_gpus=${#gpus[@]}
job_idx=0
echo "Number of awailable gpus: $num_gpus"

# Loop through all combinations of hyperparameters
for out_channels in "${OUT_CHANNELS[@]}"; do
for n_layers in "${N_LAYERS[@]}"; do
for drop_out in "${PROJECTION_DROPOUTS[@]}"; do
for data_seed in "${DATA_SEEDS[@]}"; do
for batch_size in "${BATCH_SIZES[@]}"; do
for lr in "${LEARNING_RATES[@]}"; do
for wd in "${WEIGHT_DECAYS[@]}"; do
for pretrain_model in "${PRETRAIN_MODELS[@]}"; do


    # 1. Assign a GPU to the current job in a round-robin fashion
    gpu_idx=$((job_idx % num_gpus))
    CUDA=${gpus[$gpu_idx]}
    
    # 2. Use the 'neighborhood' corresponding to the assigned GPU
    neighborhood=${neighborhoods[$gpu_idx]}

    echo "INFO: Starting job $job_idx on GPU $CUDA -> {model: $pretrain_model, bs: $batch_size, lr: $lr, wd: $wd, neighborhood: $neighborhood}"

    # 3. Run the Python command in the background with '&'
    #    - Note: --multirun is removed since the shell script now handles the runs.
    #    - Parameters like batch_size and lr now use the shell loop variables.
    python topobench/run.py \
        dataset=graph/$dataset \
        model=cell/hopse_g \
        model.readout.readout_name=SANNReadout \
        model.backbone.n_layers=$n_layers \
        model.feature_encoder.out_channels=$out_channels \
        model.feature_encoder.proj_dropout=$drop_out \
        model.feature_encoder.use_atom_encoder=True \
        model.feature_encoder.use_bond_encoder=True \
        dataset.split_params.data_seed=$data_seed \
        dataset.dataloader_params.batch_size=$batch_size \
        trainer.max_epochs=100 \
        trainer.min_epochs=10 \
        trainer.devices=\[$CUDA\] \
        trainer.check_val_every_n_epoch=5 \
        callbacks.early_stopping.patience=10 \
        logger.wandb.project=$project_name \
        optimizer.parameters.lr=$lr \
        optimizer.parameters.weight_decay=$wd \
        transforms.sann_encoding.neighborhoods=$neighborhood \
        transforms.sann_encoding.pretrain_model=$pretrain_model \
        transforms.graph2cell_lifting.neighborhoods=$neighborhood \
        transforms.graph2cell_lifting.max_cell_length=10 \
        --multirun &
    
    # Increment the job index for the next run
    ((job_idx++))
    sleep 5
    
    # If we've launched a job on every GPU, wait for them to finish before starting more.
    # This prevents scheduling hundreds of jobs at once if you have many hyperparameters.
    if [[ $((job_idx % num_gpus)) -eq 0 ]]; then
        echo "INFO: Waiting for a batch of $num_gpus jobs to complete..."
        wait
    fi
done
done
done
done
done
done
done
done

# Wait for any remaining background jobs to complete
echo "INFO: Waiting for the final jobs to complete..."
echo "✅ All experiments are finished."