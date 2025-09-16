#!/bin/bash
dataset='ogbg-molhiv'
project_name="rebutal_HOPSE_M_cell_$dataset"

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
LEARNING_RATES=(0.0001)
PROJECTION_DROPOUTS=(0.0 0.1 0.25)
WEIGHT_DECAYS=(0 0.000001)
BATCH_SIZES=(32 64)

# =====================
# CONVERT TO STRINGS
# =====================
DATA_SEEDS_STR=$(IFS=,; echo "${DATA_SEEDS[*]}")  # Convert to comma-separated string
N_LAYERS_STR=$(IFS=,; echo "${N_LAYERS[*]}")  # Convert to comma-separated string
OUT_CHANNELS_STR=$(IFS=,; echo "${OUT_CHANNELS[*]}")  # Convert to comma-separated string
LEARNING_RATES_STR=$(IFS=,; echo "${LEARNING_RATES[*]}")  # Convert to comma-separated string
PROJECTION_DROPOUTS_STR=$(IFS=,; echo "${PROJECTION_DROPOUTS[*]}")  # Convert to comma-separated string
WEIGHT_DECAYS_STR=$(IFS=,; echo "${WEIGHT_DECAYS[*]}")  # Convert to comma-separated string
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
for i in {0..7}; do 
    CUDA=${gpus[$i]}  # Use the GPU number from our gpus array
    neighborhood=${neighborhoods[$i]} # Use the neighbourhood from our neighbourhoods array

        
    python topobench/run.py\
        dataset=graph/$dataset\
        model=simplicial/hopse_m\
        model.readout.readout_name=SANNReadout\
        model.backbone.n_layers=1\
        model.feature_encoder.out_channels=128\
        model.feature_encoder.proj_dropout=0.5\
        model.feature_encoder.use_atom_encoder=True\
        model.feature_encoder.use_bond_encoder=True\
        dataset.split_params.data_seed=0\
        dataset.dataloader_params.batch_size=128\
        trainer.max_epochs=5\
        trainer.min_epochs=1\
        trainer.devices=\[$CUDA\]\
        trainer.check_val_every_n_epoch=1\
        callbacks.early_stopping.patience=10\
        logger.wandb.project='prerun'\
        optimizer.parameters.lr=0.01\
        optimizer.parameters.weight_decay=0.25\
        transforms.sann_encoding.neighborhoods=$neighborhood\
        transforms.graph2simplicial_lifting.neighborhoods=$neighborhood\
        --multirun
        sleep 5
done
wait

# gpus=(0 1 2 3 4 5 6 7)
# for i in {0..7}; do 
#     CUDA=${gpus[$i]}  # Use the GPU number from our gpus array
#     neighborhood=${neighborhoods[$i]} # Use the neighbourhood from our neighbourhoods array

#     for proj_dropout in ${PROJECTION_DROPOUTS[*]}
#     do
#         for batch_size in ${BATCH_SIZES[*]}
#         do
        
#             python topobench/run.py\
#                 dataset=graph/$dataset\
#                 model=simplicial/hopse_m\
#                 model.readout.readout_name=SANNReadout\
#                 model.backbone.n_layers=$N_LAYERS_STR\
#                 model.feature_encoder.out_channels=$OUT_CHANNELS_STR\
#                 model.feature_encoder.proj_dropout=$proj_dropout\
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
#                 transforms.graph2simplicial_lifting.neighborhoods=$neighborhood\
#                 --multirun
#         done
#     done
# done
# wait