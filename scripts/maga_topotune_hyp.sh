

# seeds=(1 3 5)

# for seed in ${seeds[@]}; do
#     python -m topobench \
#         dataset=hypergraph/maga_arlequin \
#         model=pointcloud/deepset \
#         model.feature_encoder.out_channels=64,32,128 \
#         model.feature_encoder.proj_dropout=0.0,0.2 \
#         optimizer.parameters.lr=0.001,0.0001 \
#         optimizer.parameters.weight_decay=0.0,0.0001 \
#         dataset.split_params.data_seed=${seed} \
#         model.readout.pooling_type=mean,sum \
#         trainer.max_epochs=1000 \
#         trainer.min_epochs=250 \
#         callbacks.early_stopping.patience=100 \
#         trainer.devices=[2] \
#         logger.wandb.project=MAGA_TopoTune_Hypergraph \
#         tags=[FirstExperiments] \
#         --multirun &
# done

for seed in ${seeds[@]}; do
    python -m topobench \
        dataset=hypergraph/maga_arlequin \
        model=hypergraph/topotune \
        model.feature_encoder.out_channels=32,64,128 \
        model.feature_encoder.proj_dropout=0.0,0.25 \
        model.backbone.neighborhoods=\[3-up_adjacency-0\] \
        model.backbone.layers=1 \
        model.backbone.activation=relu \
        dataset.split_params.data_seed=${seed} \
        model.readout.readout_name=PropagateSignalDown \
        model.readout.pooling_type=mean \
        model.readout.hierarchical_propagation=False \
        model.readout.ranks_to_propagate=\[\],\[3\] \
        optimizer.parameters.lr=0.001 \
        optimizer.parameters.weight_decay=0.0 \
        logger.wandb.project=MAGA_TopoTune_Hypergraph \
        trainer.max_epochs=1000 \
        trainer.min_epochs=250 \
        trainer.devices=\[0\] \
        trainer.check_val_every_n_epoch=1 \
        callbacks.early_stopping.patience=100 \
        tags="[FirstExperiments]" \
        --multirun &

    python -m topobench \
        dataset=hypergraph/maga_arlequin \
        model=hypergraph/topotune \
        model.feature_encoder.out_channels=32,64,128 \
        model.feature_encoder.proj_dropout=0.0,0.25 \
        model.backbone.neighborhoods=\[2-up_adjacency-0\] \
        model.backbone.layers=1 \
        model.backbone.activation=relu \
        dataset.split_params.data_seed=${seed} \
        model.readout.readout_name=PropagateSignalDown \
        model.readout.pooling_type=mean \
        model.readout.hierarchical_propagation=False \
        model.readout.ranks_to_propagate=\[\],\[2\] \
        optimizer.parameters.lr=0.001 \
        optimizer.parameters.weight_decay=0.0 \
        logger.wandb.project=MAGA_TopoTune_Hypergraph \
        trainer.max_epochs=1000 \
        trainer.min_epochs=250 \
        trainer.devices=\[1\] \
        trainer.check_val_every_n_epoch=1 \
        callbacks.early_stopping.patience=100 \
        tags="[FirstExperiments]" \
        --multirun &

    python -m topobench \
        dataset=hypergraph/maga_arlequin \
        model=hypergraph/topotune \
        model.feature_encoder.out_channels=32,64,128 \
        model.feature_encoder.proj_dropout=0.0,0.25 \
        model.backbone.neighborhoods=\[1-up_adjacency-0\] \
        model.backbone.layers=1 \
        model.backbone.activation=relu \
        dataset.split_params.data_seed=${seed} \
        model.readout.readout_name=PropagateSignalDown \
        model.readout.pooling_type=mean \
        model.readout.hierarchical_propagation=False \
        model.readout.ranks_to_propagate=\[\],\[1\] \
        optimizer.parameters.lr=0.001 \
        optimizer.parameters.weight_decay=0.0 \
        logger.wandb.project=MAGA_TopoTune_Hypergraph \
        trainer.max_epochs=1000 \
        trainer.min_epochs=250 \
        trainer.devices=\[2\] \
        trainer.check_val_every_n_epoch=1 \
        callbacks.early_stopping.patience=100 \
        tags="[FirstExperiments]" \
        --multirun &

    python -m topobench \
        dataset=hypergraph/maga_arlequin \
        model=hypergraph/topotune \
        model.feature_encoder.out_channels=32,64,128 \
        model.feature_encoder.proj_dropout=0.0,0.25 \
        model.backbone.neighborhoods=\[3-up_adjacency-0,2-up_adjacency-0,1-up_adjacency-0\] \
        model.backbone.layers=1 \
        model.backbone.activation=relu \
        dataset.split_params.data_seed=${seed} \
        model.readout.readout_name=PropagateSignalDown \
        model.readout.pooling_type=mean \
        model.readout.hierarchical_propagation=False \
        model.readout.ranks_to_propagate=\[\],\[3,2,1\] \
        optimizer.parameters.lr=0.001 \
        optimizer.parameters.weight_decay=0.0 \
        logger.wandb.project=MAGA_TopoTune_Hypergraph \
        trainer.max_epochs=1000 \
        trainer.min_epochs=250 \
        trainer.devices=\[3\] \
        trainer.check_val_every_n_epoch=1 \
        callbacks.early_stopping.patience=100 \
        tags="[FirstExperiments]" \
        --multirun &

done



# # seeds=(1 3 5)

# # for seed in ${seeds[@]}; do
# #     python -m topobench \
# #         dataset=hypergraph/maga_arlequin \
# #         model=hypergraph/topotune_hyp \
# #         model.feature_encoder.out_channels=32,64,128 \
# #         model.feature_encoder.proj_dropout=0.0,0.25,0.5 \
# #         model.backbone.neighborhoods=\[3-down_incidence-3\] \
# #         model.backbone.layers=1 \
# #         model.backbone.activation=relu,tanh \
# #         dataset.split_params.data_seed=${seed} \
# #         model.readout.readout_name=PropagateSignalDown \
# #         model.readout.pooling_type=mean,sum \
# #         model.readout.hierarchical_propagation=False \
# #         model.readout.ranks_to_propagate=\[\],\[3\] \
# #         optimizer.parameters.lr=0.001 \
# #         optimizer.parameters.weight_decay=0.0,0.0005 \
# #         logger.wandb.project=MAGA_TopoTune_Hypergraph \
# #         trainer.max_epochs=1000 \
# #         trainer.min_epochs=250 \
# #         trainer.devices=\[0\] \
# #         trainer.check_val_every_n_epoch=1 \
# #         callbacks.early_stopping.patience=100 \
# #         tags="[FirstExperiments]" \
# #         --multirun &

# #     python -m topobench \
# #         dataset=hypergraph/maga_arlequin \
# #         model=hypergraph/topotune_hyp \
# #         model.feature_encoder.out_channels=32,64,128 \
# #         model.feature_encoder.proj_dropout=0.0,0.25,0.5 \
# #         model.backbone.neighborhoods=\[2-down_incidence-2\] \
# #         model.backbone.layers=1 \
# #         model.backbone.activation=relu,tanh \
# #         dataset.split_params.data_seed=${seed} \
# #         model.readout.readout_name=PropagateSignalDown \
# #         model.readout.pooling_type=mean,sum \
# #         model.readout.hierarchical_propagation=False \
# #         model.readout.ranks_to_propagate=\[\],\[2\] \
# #         optimizer.parameters.lr=0.001 \
# #         optimizer.parameters.weight_decay=0.0,0.0005 \
# #         logger.wandb.project=MAGA_TopoTune_Hypergraph \
# #         trainer.max_epochs=1000 \
# #         trainer.min_epochs=250 \
# #         trainer.devices=\[1\] \
# #         trainer.check_val_every_n_epoch=1 \
# #         callbacks.early_stopping.patience=100 \
# #         tags="[FirstExperiments]" \
# #         --multirun &

# #     python -m topobench \
# #         dataset=hypergraph/maga_arlequin \
# #         model=hypergraph/topotune_hyp \
# #         model.feature_encoder.out_channels=32,64,128 \
# #         model.feature_encoder.proj_dropout=0.0,0.25,0.5 \
# #         model.backbone.neighborhoods=\[1-down_incidence-1\] \
# #         model.backbone.layers=1 \
# #         model.backbone.activation=relu,tanh \
# #         dataset.split_params.data_seed=${seed} \
# #         model.readout.readout_name=PropagateSignalDown \
# #         model.readout.pooling_type=mean,sum \
# #         model.readout.hierarchical_propagation=False \
# #         model.readout.ranks_to_propagate=\[\],\[1\] \
# #         optimizer.parameters.lr=0.001 \
# #         optimizer.parameters.weight_decay=0.0,0.0005 \
# #         logger.wandb.project=MAGA_TopoTune_Hypergraph \
# #         trainer.max_epochs=1000 \
# #         trainer.min_epochs=250 \
# #         trainer.devices=\[2\] \
# #         trainer.check_val_every_n_epoch=1 \
# #         callbacks.early_stopping.patience=100 \
# #         tags="[FirstExperiments]" \
# #         --multirun &

# #     python -m topobench \
# #         dataset=hypergraph/maga_arlequin \
# #         model=hypergraph/topotune_hyp \
# #         model.feature_encoder.out_channels=32,64,128 \
# #         model.feature_encoder.proj_dropout=0.0,0.25,0.5 \
# #         model.backbone.neighborhoods=\[3-down_incidence-3,2-down_incidence-2,1-down_incidence-1\] \
# #         model.backbone.layers=1 \
# #         model.backbone.activation=relu,tanh \
# #         dataset.split_params.data_seed=${seed} \
# #         model.readout.readout_name=PropagateSignalDown \
# #         model.readout.pooling_type=mean,sum \
# #         model.readout.hierarchical_propagation=False \
# #         model.readout.ranks_to_propagate=\[\],\[3,2,1\] \
# #         optimizer.parameters.lr=0.001 \
# #         optimizer.parameters.weight_decay=0.0,0.0005 \
# #         logger.wandb.project=MAGA_TopoTune_Hypergraph \
# #         trainer.max_epochs=1000 \
# #         trainer.min_epochs=250 \
# #         trainer.devices=\[3\] \
# #         trainer.check_val_every_n_epoch=1 \
# #         callbacks.early_stopping.patience=100 \
# #         tags="[FirstExperiments]" \
# #         --multirun &

# # done
