#!/bin/bash
dataset='ogbg-molhiv'
project_name="rebutal_GCN_$dataset"

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
PE_TYPES=('RWSE' 'ElstaticPE' 'HKdiagSE' 'LapPE' 'RWSE,ElstaticPE,HKdiagSE,LapPE')
neighborhoods=(
    # adjacency 
    "['up_adjacency-0']"
)

gpus=(0 1 2 3 4 5 6 7)
for i in {0..7}; do 
    CUDA=${gpus[$i]}  # Use the GPU number from our gpus array

    for pe_type in ${PE_TYPES[*]}
    do 
        python topobench/run.py\
            dataset=graph/$dataset\
            model=graph/hopse_gcn\
            experiment=hopse_m_gnn_simplicial\
            model.backbone.num_layers=1\
            model.feature_encoder.out_channels=128\
            model.feature_encoder.proj_dropout=0.1\
            dataset.split_params.data_seed=0\
            dataset.dataloader_params.batch_size=32\
            trainer.max_epochs=5\
            trainer.min_epochs=1\
            trainer.devices=\[$CUDA\]\
            trainer.check_val_every_n_epoch=1\
            logger.wandb.project=$project_name\
            optimizer.parameters.lr=0.001\
            optimizer.parameters.weight_decay=0.0\
            transforms.sann_encoding.pe_types=[$pe_type]\
            transforms.sann_encoding.neighborhoods=${neighborhoods[0]}\
            transforms.graph2simplicial_lifting.neighborhoods=${neighborhood[0]}\
            model.feature_encoder.use_atom_encoder=True\
            model.feature_encoder.use_bond_encoder=True\
            --multirun &
            sleep 5
    done
done
wait

gpus=(0 1 2 3 4 5 6 7)
for i in {0..7}; do 
    CUDA=${gpus[$i]}  # Use the GPU number from our gpus array

    for pe_type in ${PE_TYPES[*]}
    do
        for batch_size in ${batch_sizes}
        do
        
            python topobench/run.py\
                dataset=graph/$dataset\
                model=graph/hopse_gcn\
                experiment=hopse_m_gnn_simplicial\
                model.backbone.num_layers=$N_LAYERS_STR\
                model.feature_encoder.out_channels=$OUT_CHANNELS_STR\
                model.feature_encoder.proj_dropout=$PROJECTION_DROPOUTS_STR\
                dataset.split_params.data_seed=$DATA_SEEDS_STR\
                dataset.dataloader_params.batch_size=$batch_size\
                trainer.max_epochs=100\
                trainer.min_epochs=10\
                trainer.devices=\[$CUDA\]\
                trainer.check_val_every_n_epoch=5\
                callbacks.early_stopping.patience=10\
                logger.wandb.project=$project_name\
                optimizer.parameters.lr=$LEARNING_RATES_STR\
                optimizer.parameters.weight_decay=$WEIGHT_DECAYS_STR\
                transforms.sann_encoding.pe_types=[$pe_type]\
                transforms.sann_encoding.neighborhoods=${neighborhoods[0]}\
                transforms.graph2simplicial_lifting.neighborhoods=${neighborhoods[0]}\
                model.feature_encoder.use_atom_encoder=True\
                model.feature_encoder.use_bond_encoder=True\
                --multirun &
        done
    done
done
wait