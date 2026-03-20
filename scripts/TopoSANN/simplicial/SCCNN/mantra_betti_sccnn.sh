# python -m topobench model=simplicial/sccnn_custom dataset=simplicial/mantra_betti_numbers evaluator=betti_numbers model.readout.readout_name=PropagateSignalDown transforms=MANTRA_name_dataset_default transforms.redefine_simplicial_neighborhoods.signed=True optimizer.parameters.lr=0.001 model.backbone.n_layers=2 trainer.devices=\[4\]
# python -m topobench model=graph/gcn dataset=simplicial/mantra_betti_numbers evaluator=betti_numbers transforms=no_transform optimizer.parameters.lr=0.001 trainer.devices=\[4\]

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
PROJECTION_DROPOUTS=(0.25)
WEIGHT_DECAYS=(0 0.0001)
BATCH_SIZES=(128 256)
READOUT_NAMES=("PropagateSignalDown")
INDICENCE_SIGNED=(True)

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

gpus=(4 5 6 7)
datasets=('mantra_betti_numbers') #'mantra_orientation'

for dataset in ${datasets[*]} 
do
    project_name="SCCNN_$dataset"
            
    # python topobench/run.py\
    #     dataset=simplicial/$dataset \
    #     model=simplicial/sccnn_custom \
    #     model.backbone.n_layers=1\
    #     model.feature_encoder.out_channels=128\
    #     model.feature_encoder.proj_dropout=0.25\
    #     dataset.split_params.data_seed=0\
    #     dataset.dataloader_params.batch_size=128\
    #     model.readout.readout_name=NoReadOut\
    #     trainer.max_epochs=5\
    #     trainer.min_epochs=1\
    #     trainer.devices=\[0\]\
    #     trainer.check_val_every_n_epoch=1\
    #     logger.wandb.project=prerun\
    #     transforms=MANTRA_name_dataset_default\
    #     transforms.redefine_simplicial_neighborhoods.signed=True\
    #     optimizer.parameters.lr=0.001\
    #     optimizer.parameters.weight_decay=0.01\
    #     callbacks.early_stopping.patience=1\
    #     --multirun &
    # wait 
    # sleep 5

    # =====================
    for i in {0..4}; do 
        CUDA=${gpus[$i]}  # Use the GPU number from our gpus array
        data_seed=${DATA_SEEDS[$i]} # Use the neighbourhood from our neighbourhoods array
        
        for lr in ${LEARNING_RATES[*]}
        do
            for batch_size in ${BATCH_SIZES[*]}
            do
                python topobench/run.py\
                    dataset=simplicial/$dataset\
                    model=simplicial/sccnn_custom\
                    model.backbone.n_layers=$N_LAYERS_STR\
                    model.feature_encoder.out_channels=$OUT_CHANNELS_STR\
                    model.feature_encoder.proj_dropout=$PROJECTION_DROPOUTS_STR\
                    model.readout.readout_name=$READOUT_NAMES_STR\
                    dataset.split_params.data_seed=$data_seed\
                    dataset.dataloader_params.batch_size=$batch_size\
                    trainer.devices=\[$CUDA\]\
                    trainer.max_epochs=500\
                    trainer.min_epochs=50\
                    trainer.check_val_every_n_epoch=5\
                    callbacks.early_stopping.patience=10\
                    optimizer.parameters.lr=$lr\
                    optimizer.parameters.weight_decay=$WEIGHT_DECAYS_STR\
                    transforms=MANTRA_name_dataset_default\
                    transforms.redefine_simplicial_neighborhoods.signed=$INDICENCE_SIGNED_STR\
                    evaluator=betti_numbers\
                    logger.wandb.project=$project_name\
                    --multirun &
                sleep 10
            done
        done
    done
done
