#!/bin/bash
dataset='ogbg-molhiv'
project_name="rebutal_CWN_$dataset"

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
WEIGHT_DECAYS=(0 0.0001)
BATCH_SIZES=(128 256)
READOUT_NAMES=("NoReadOut" "PropagateSignalDown")

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
READOUT_NAMES_STR=$(IFS=,; echo "${READOUT_NAMES[*]}")  # Convert to comma-separated string
INDICENCE_SIGNED_STR=$(IFS=,; echo "${INDICENCE_SIGNED[*]}")  # Convert to comma-separated string

gpus=(0 1 2 3 4 5 6 7)
python topobench/run.py\
    dataset=graph/$dataset \
    model=cell/cwn \
    model.backbone.n_layers=1\
    model.feature_encoder.out_channels=128\
    model.feature_encoder.proj_dropout=0.25\
    dataset.split_params.data_seed=0\
    dataset.dataloader_params.batch_size=128\
    model.readout.readout_name=NoReadOut\
    trainer.max_epochs=5\
    trainer.min_epochs=1\
    trainer.devices=\[0\]\
    trainer.check_val_every_n_epoch=1\
    logger.wandb.project=prerun\
    optimizer.parameters.lr=0.001\
    optimizer.parameters.weight_decay=0.01\
    callbacks.early_stopping.patience=1\
    transforms.graph2cell_lifting.max_cell_length=10\
    --multirun &
wait 
sleep 5

# =====================
for i in {0..4}; do 
    CUDA=${gpus[$i]}  # Use the GPU number from our gpus array
    data_seed=${DATA_SEEDS[$i]} # Use the neighbourhood from our neighbourhoods array
    
    for lr in ${LEARNING_RATES[*]}
    do
        for batch_size in ${BATCH_SIZES[*]}
        do
            python topobench/run.py\
                dataset=graph/$dataset\
                model=cell/cwn\
                model.backbone.n_layers=$N_LAYERS_STR\
                model.feature_encoder.out_channels=$OUT_CHANNELS_STR\
                model.feature_encoder.proj_dropout=$PROJECTION_DROPOUTS_STR\
                model.readout.readout_name=$READOUT_NAMES_STR\
                dataset.split_params.data_seed=$data_seed\
                dataset.dataloader_params.batch_size=$batch_size\
                trainer.devices=\[$CUDA\]\
                trainer.max_epochs=100\
                trainer.min_epochs=10\
                trainer.check_val_every_n_epoch=5\
                callbacks.early_stopping.patience=10\
                optimizer.parameters.lr=$lr\
                optimizer.parameters.weight_decay=$WEIGHT_DECAYS_STR\
                logger.wandb.project=$project_name\
                transforms.graph2cell_lifting.max_cell_length=10\
                --multirun &
            sleep 10
        done
    done
done
