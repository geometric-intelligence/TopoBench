# =====================
# DATA
# =====================
DATA_SEEDS=(0 3 5 7 9) 

# =====================
# MODEL PARAMETERS
# =====================
N_LAYERS=(1 2 4)
OUT_CHANNELS=(32 64 128)

# =====================
# OPTIMIZATION PARAMETERS
# =====================
LEARNING_RATES=(0.01 0.001)
PROJECTION_DROPOUTS=(0.25 0.5)
WEIGHT_DECAYS=(0 0.0001)

# =====================
# PRETRAINED MODELS
# =====================
# PRETRAIN_MODELS=('ZINC' 'GEOM' 'MOLPCBA' 'PCQM4MV2')

# seed=42,3,5,23,150
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

# =====================
# PARAMETERS OVER WHICH WE PERFORM PARALLEL RUNS
# =====================
batch_sizes=(128 256)
learning_rates=(0.001)
neighborhoods=(
    "['up_adjacency-0']"
)

datasets=('NCI1' 'NCI109' 'MUTAG' 'PROTEINS')
PE_TYPES=('RWSE,ElstaticPE,HKdiagSE')
model_types=('hopse_gcn' 'hopse_gat')
for model_type in ${model_types[*]}
do
    for dataset in ${datasets[*]}
    do
        for pe_type in ${PE_TYPES[*]}
        do
        project_name="fix_gnn_rebuttal_cell_${dataset}"
            # =====================
            for i in {0..1}; do 
                neighborhood=${neighborhoods[$i]} # Use the neighbourhood from our neighbourhoods array

                python topobench/run.py\
                    dataset=graph/$dataset\
                    model=graph/$model_type\
                    experiment=hopse_m_gnn_cell\
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
                    transforms.sann_encoding.pe_types=[$pe_type]\
                    transforms.graph2cell_lifting.max_cell_length=10\
                    transforms.sann_encoding.neighborhoods=$neighborhood\
                    transforms.graph2cell_lifting.neighborhoods=$neighborhood\
                    --multirun &
                    sleep 10
            done
            wait

            for i in {0..1}; do 
                neighborhood=${neighborhoods[$i]} # Use the neighbourhood from our neighbourhoods array

                for batch_size in ${batch_sizes[*]}
                do
                    for pd in ${PROJECTION_DROPOUTS[*]}
                    do
                        python topobench/run.py\
                            dataset=graph/$dataset\
                            model=graph/$model_type\
                            experiment=hopse_m_gnn_cell\
                            model.backbone.num_layers=$N_LAYERS_STR\
                            model.feature_encoder.out_channels=$OUT_CHANNELS_STR\
                            model.feature_encoder.proj_dropout=$pd\
                            dataset.split_params.data_seed=$DATA_SEEDS_STR\
                            dataset.dataloader_params.batch_size=$batch_size\
                            trainer.max_epochs=500\
                            trainer.min_epochs=50\
                            trainer.devices=\[0\]\
                            trainer.check_val_every_n_epoch=5\
                            logger.wandb.project=$project_name\
                            optimizer.parameters.lr=$LEARNING_RATES_STR\
                            optimizer.parameters.weight_decay=$WEIGHT_DECAYS_STR\
                            callbacks.early_stopping.patience=10\
                            transforms.sann_encoding.pe_types=[$pe_type]\
                            transforms.graph2cell_lifting.max_cell_length=10\
                            transforms.sann_encoding.neighborhoods=$neighborhood\
                            transforms.graph2cell_lifting.neighborhoods=$neighborhood\
                            --multirun &
                    done
                done
            done
            wait
        done
    done
done