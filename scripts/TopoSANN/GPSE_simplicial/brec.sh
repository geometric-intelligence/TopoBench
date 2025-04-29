#!/bin/bash

#project_name="main_exp_GPSE_$dataset"

# =====================
# DATA
# =====================
DATA_SEEDS=(0 3 5 7 9) 

# =====================
# MODEL PARAMETERS
# =====================
N_LAYERS=(1 2 4)
OUT_CHANNELS=(128 256)

# =====================
# OPTIMIZATION PARAMETERS
# =====================
LEARNING_RATES=(0.0001 0.00001)
PROJECTION_DROPOUTS=(0.25 0.5)
WEIGHT_DECAYS=(0 0.0001 0.00001)
BATCH_SIZES=(16 32 64)
# =====================
# PRETRAINED MODELS
# =====================
PRETRAIN_MODELS=('ZINC') #  'GEOM' 'MOLPCBA' 'PCQM4MV2'


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
# batch_sizes=(128 256)
# learning_rates=(0.01 0.001)
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

datasets=(BREC_basic BREC_regular BREC_str BREC_extension BREC_cfi BREC_4vtx BREC_dr)
gpus=(0 1 2 3 4 5 6 7)
for i in {0..7}; do 
    CUDA=${gpus[$i]}  # Use the GPU number from our gpus array
    neighborhood=${neighborhoods[$i]} # Use the neighbourhood from our neighbourhoods array

    for pretrain_model in ${PRETRAIN_MODELS[*]}
    do
        
        python topobench/run_brec.py\
            dataset=graph/$dataset\
            model=simplicial/hopse_g\
            model.backbone.n_layers=$N_LAYERS_STR\
            model.feature_encoder.out_channels=256\
            model.feature_encoder.proj_dropout=$PROJECTION_DROPOUTS_STR\
            transforms/data_manipulations@transforms.sann_encoding=add_gpse_information\
            transforms.sann_encoding.pretrain_model=$pretrain_model\
            transforms.sann_encoding.copy_initial=True \
            transforms.sann_encoding.neighborhoods=$neighborhood\
            transforms=GPSE_BREC\
            transforms.graph2simplicial_lifting.neighborhoods=$neighborhood\
            optimizer.parameters.lr=$LEARNING_RATES_STR\
            optimizer.parameters.weight_decay=$WEIGHT_DECAYS_STR\
            trainer.devices=\[$CUDA\]\
            dataset.dataloader_params.batch_size=$BATCH_SIZES_STR\
            trainer.max_epochs=100\
            logger.wandb.project=BREC\
            model.readout.readout_name=SANNReadout\
            --multirun
            #trainer.check_val_every_n_epoch=5\
            #dataset.split_params.data_seed=$DATA_SEEDS_STR\
            #dataset.dataloader_params.batch_size=$BATCH_SIZES_STR\
            #trainer.max_epochs=500\
            #trainer.min_epochs=50\
            #logger.wandb.project=$project_name\
            #callbacks.early_stopping.patience=10\
        break
    done
done