dataset='IMDB-MULTI'
project_name="main_exp_SANN_$dataset"

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
BATCH_SIZES=(128)
# =====================
# MAX_HOPS
# =====================
MAX_HOPS=(1 2 3)
# =====================
# CONVERT TO STRINGS
# =====================
N_LAYERS_STR=$(IFS=,; echo "${N_LAYERS[*]}")  # Convert to comma-separated string
OUT_CHANNELS_STR=$(IFS=,; echo "${OUT_CHANNELS[*]}")  # Convert to comma-separated string
LEARNING_RATES_STR=$(IFS=,; echo "${LEARNING_RATES[*]}")  # Convert to comma-separated string
PROJECTION_DROPOUTS_STR=$(IFS=,; echo "${PROJECTION_DROPOUTS[*]}")  # Convert to comma-separated string
WEIGHT_DECAYS_STR=$(IFS=,; echo "${WEIGHT_DECAYS[*]}")  # Convert to comma-separated string
BATCH_SIZES_STR=$(IFS=,; echo "${BATCH_SIZES[*]}")

# =====================
# PARAMETERS OVER WHICH WE PERFORM PARALLEL RUNS
# =====================
# batch_sizes=(128 256)
# learning_rates=(0.01 0.001)


# PRERUN to create datasets

for max_hop in ${MAX_HOPS[*]}
do
        
    python topobench/run.py\
        dataset=graph/$dataset\
        model=simplicial/sann\
        model.backbone.n_layers=1\
        model.feature_encoder.out_channels=128\
        model.feature_encoder.proj_dropout=0.25\
        dataset.split_params.data_seed=0\
        dataset.dataloader_params.batch_size=128\
        trainer.max_epochs=5\
        trainer.min_epochs=1\
        trainer.devices=\[0\]\
        trainer.check_val_every_n_epoch=1\
        logger.wandb.project=prerun\
        optimizer.parameters.lr=0.001\
        optimizer.parameters.weight_decay=0.01\
        callbacks.early_stopping.patience=1\
        transforms=IMDB_sann_experiment_simplicial\
        transforms.sann_encoding.max_hop=$max_hop\
        transforms.sann_encoding.complex_dim=3\
        --multirun &
        
done

wait 

gpus=(0 1 2 3 4 5 6 7)
for i in {0..4}; do 
    CUDA=${gpus[$i]}  # Use the GPU number from our gpus array
    data_seed=${DATA_SEEDS[$i]} # Use the neighbourhood from our neighbourhoods array
    
    for max_hop in ${MAX_HOPS[*]}
    do
        for batch_size in ${BATCH_SIZES[*]}
        do
            python topobench/run.py\
                dataset=graph/$dataset\
                model=simplicial/sann\
                model.backbone.n_layers=$N_LAYERS_STR\
                model.feature_encoder.out_channels=$OUT_CHANNELS_STR\
                model.feature_encoder.proj_dropout=$PROJECTION_DROPOUTS_STR\
                dataset.split_params.data_seed=$data_seed\
                dataset.dataloader_params.batch_size=$batch_size\
                trainer.max_epochs=500\
                trainer.min_epochs=50\
                trainer.devices=\[$CUDA\]\
                trainer.check_val_every_n_epoch=5\
                logger.wandb.project=$project_name\
                optimizer.parameters.lr=$LEARNING_RATES_STR\
                optimizer.parameters.weight_decay=$WEIGHT_DECAYS_STR\
                callbacks.early_stopping.patience=10\
                transforms=IMDB_sann_experiment_simplicial\
                transforms.sann_encoding.max_hop=$max_hop\
                transforms.sann_encoding.complex_dim=3\
                --multirun &
            sleep 3
        done
    done
done

wait

BATCH_SIZES=(256)

gpus=(0 1 2 3 4 5 6 7)
for i in {0..4}; do 
    CUDA=${gpus[$i]}  # Use the GPU number from our gpus array
    data_seed=${DATA_SEEDS[$i]} # Use the neighbourhood from our neighbourhoods array
    
    for max_hop in ${MAX_HOPS[*]}
    do
        for batch_size in ${BATCH_SIZES[*]}
        do
            python topobench/run.py\
                dataset=graph/$dataset\
                model=simplicial/sann\
                model.backbone.n_layers=$N_LAYERS_STR\
                model.feature_encoder.out_channels=$OUT_CHANNELS_STR\
                model.feature_encoder.proj_dropout=$PROJECTION_DROPOUTS_STR\
                dataset.split_params.data_seed=$data_seed\
                dataset.dataloader_params.batch_size=$batch_size\
                trainer.max_epochs=500\
                trainer.min_epochs=50\
                trainer.devices=\[$CUDA\]\
                trainer.check_val_every_n_epoch=5\
                logger.wandb.project=$project_name\
                optimizer.parameters.lr=$LEARNING_RATES_STR\
                optimizer.parameters.weight_decay=$WEIGHT_DECAYS_STR\
                callbacks.early_stopping.patience=10\
                transforms=IMDB_sann_experiment_simplicial\
                transforms.sann_encoding.max_hop=$max_hop\
                transforms.sann_encoding.complex_dim=3\
                --multirun &
            sleep 3
        done
    done
done