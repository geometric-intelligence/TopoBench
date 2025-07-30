dataset='ZINC_OGB'
project_name="ZINC_large_$dataset"

# =====================
# DATA
# =====================
DATA_SEEDS=(42 3 5 23 150) 
# 42,3,5,23,150
# =====================
# MODEL PARAMETERS
# =====================
N_LAYERS=(16 18)
OUT_CHANNELS=(128)

# =====================
# OPTIMIZATION PARAMETERS
# =====================
LEARNING_RATES=(0.001 0.0001)
PROJECTION_DROPOUTS=(0.25 0.5)
WEIGHT_DECAYS=(0 0.0001 0.001)

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
    # adjacency 
    "['up_adjacency-0']"
    # incidence
    "['up_adjacency-0','down_incidence-1']"
)

# TODO: fix bug with transforms.one_hot_node_degree_features.degrees_fields=x\
# gpus=(5 6 7)
# for i in {0..1}; do 
#     CUDA=${gpus[$i]}  # Use the GPU number from our gpus array
#     neighborhood=${neighborhoods[$i]} # Use the neighbourhood from our neighbourhoods array

    
#     python topobench/run.py\
#         dataset=graph/$dataset\
#         model=graph/hopse_gin\
#         experiment=hopse_m_gnn_cell_zinc\
#         model.backbone.num_layers=1\
#         model.feature_encoder.out_channels=128\
#         model.feature_encoder.proj_dropout=0.25\
#         dataset.split_params.data_seed=0\
#         dataset.dataloader_params.batch_size=128\
#         trainer.max_epochs=5\
#         trainer.min_epochs=1\
#         trainer.devices=\[$CUDA\]\
#         trainer.check_val_every_n_epoch=1\
#         logger.wandb.project='prerun'\
#         optimizer.parameters.lr=0.01\
#         optimizer.parameters.weight_decay=0.25\
#         callbacks.early_stopping.patience=10\
#         transforms.graph2cell_lifting.max_cell_length=10\
#         transforms.sann_encoding.neighborhoods=$neighborhood\
#         transforms.graph2cell_lifting.neighborhoods=$neighborhood\
#         --multirun &
#         sleep 5
# done
# wait

gpus=(5 6 7)
for i in {0..1}; do 
    CUDA=${gpus[$i]}  # Use the GPU number from our gpus array
    neighborhood=${neighborhoods[$i]} # Use the neighbourhood from our neighbourhoods array

    for batch_size in ${batch_sizes[*]}
    do
        for pd in ${PROJECTION_DROPOUTS[*]}
        do
            python topobench/run.py\
                dataset=graph/$dataset\
                model=graph/hopse_gin\
                experiment=hopse_m_gnn_cell_zinc\
                model.backbone.num_layers=$N_LAYERS_STR\
                model.feature_encoder.out_channels=$OUT_CHANNELS_STR\
                model.feature_encoder.proj_dropout=$pd\
                model.feature_encoder.use_bond_encoder=True\
                model.feature_encoder.use_atom_encoder=True\
                seed=$DATA_SEEDS_STR\
                dataset.dataloader_params.batch_size=$batch_size\
                trainer.max_epochs=1000\
                trainer.min_epochs=500\
                trainer.devices=\[$CUDA\]\
                trainer.check_val_every_n_epoch=5\
                logger.wandb.project=$project_name\
                optimizer.parameters.lr=$LEARNING_RATES_STR\
                optimizer.parameters.weight_decay=$WEIGHT_DECAYS_STR\
                callbacks.early_stopping.patience=10\
                transforms.graph2cell_lifting.max_cell_length=10\
                transforms.sann_encoding.neighborhoods=$neighborhood\
                transforms.graph2cell_lifting.neighborhoods=$neighborhood\
                
                --multirun &
        done
    done
done
wait