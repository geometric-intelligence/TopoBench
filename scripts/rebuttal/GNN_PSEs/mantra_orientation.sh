dataset='mantra_orientation'
project_name="rebuttal_GNN_$dataset"

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
LEARNING_RATES=(0.01 0.001)
PROJECTION_DROPOUTS=(0.25 0.5)
WEIGHT_DECAYS=(0 0.0001)
BATCH_SIZES=(128 256)

# =====================
# PRETRAINED MODELS
# =====================
# PRETRAIN_MODELS=('ZINC' 'GEOM' 'MOLPCBA' 'PCQM4MV2')

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
    "['up_adjacency-0']"
)

PE_TYPES=('RWSE' 'ElstaticPE' 'HKdiagSE' 'LapPE' 'RWSE,ElstaticPE,HKdiagSE,LapPE' 'RWSE,ElstaticPE,HKdiagSEE')

# TODO: fix bug with transforms.one_hot_node_degree_features.degrees_fields=x\
for i in {0..5}; do
    neighborhood=${neighborhoods[0]} # Use the neighbourhood from our neighbourhoods array
    pe_type=${PE_TYPES[$i]} # Use the neighbourhood from our neighbourhoods array

    
    python topobench/run.py\
        dataset=simplicial/$dataset\
        model=graph/hopse_gin\
        experiment=hopse_m_gnn_mantra\
        model.backbone.num_layers=1\
        model.feature_encoder.out_channels=128\
        model.feature_encoder.proj_dropout=0.25\
        dataset.split_params.data_seed=0\
        dataset.dataloader_params.batch_size=128\
        trainer.max_epochs=5\
        trainer.min_epochs=1\
        trainer.devices=\[0\]\
        trainer.check_val_every_n_epoch=1\
        logger.wandb.project='prerun'\
        optimizer.parameters.lr=0.01\
        optimizer.parameters.weight_decay=0.25\
        callbacks.early_stopping.patience=10\
        transforms.sann_encoding.neighborhoods=$neighborhood\
        transforms.redefine_simplicial_neighborhoods.neighborhoods=$neighborhood\
        transforms.sann_encoding.pe_types=[$pe_type]\
        --multirun &
        sleep 300
done
wait

for i in {0..5}; do 
    neighborhood=${neighborhoods[0]} # Use the neighbourhood from our neighbourhoods array
    pe_type=${PE_TYPES[$i]} # Use the neighbourhood from our neighbourhoods array

    for lr in ${LEARNING_RATES[*]}
    do
        for wd in ${WEIGHT_DECAYS[*]}
        do
            python topobench/run.py\
                dataset=simplicial/$dataset\
                model=graph/hopse_gin\
                experiment=hopse_m_gnn_mantra\
                model.backbone.num_layers=$N_LAYERS_STR\
                model.feature_encoder.out_channels=$OUT_CHANNELS_STR\
                model.feature_encoder.proj_dropout=$PROJECTION_DROPOUTS_STR\
                dataset.split_params.data_seed=$DATA_SEEDS_STR\
                dataset.dataloader_params.batch_size=$BATCH_SIZES_STR\
                trainer.max_epochs=500\
                trainer.min_epochs=50\
                trainer.devices=\[0\]\
                trainer.check_val_every_n_epoch=5\
                logger.wandb.project=$project_name\
                optimizer.parameters.lr=$lr\
                optimizer.parameters.weight_decay=$wd\
                callbacks.early_stopping.patience=10\
                transforms.sann_encoding.neighborhoods=$neighborhood\
                transforms.redefine_simplicial_neighborhoods.neighborhoods=$neighborhood\
                transforms.sann_encoding.pe_types=[$pe_type]\
                --multirun &
        done
    done
    wait
done
wait