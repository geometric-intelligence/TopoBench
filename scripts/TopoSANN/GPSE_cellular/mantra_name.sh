


# python train.py \
#   dataset=simplicial/mantra_name \
#   seed=42,3,5,23,150 \
#   model=simplicial/sccnn_custom \
#   model.optimizer.lr=0.01,0.001 \
#   model.optimizer.weight_decay=0 \
#   model.feature_encoder.out_channels=32,64,128 \
#   model.backbone.n_layers=2,4 \
#   model.feature_encoder.proj_dropout=0.25,0.5 \
#   dataset.parameters.batch_size=128,256 \
#   model.readout.readout_name="NoReadOut,PropagateSignalDown" \
#   dataset.parameters.data_seed=0 \
#   logger.wandb.project= \
#   trainer.max_epochs=500 \
#   trainer.min_epochs=50 \
#   callbacks.early_stopping.min_delta=0.005 \
#   trainer.check_val_every_n_epoch=5 \
#   callbacks.early_stopping.patience=10 \
#   tags="[MainExperiment]" \
#   --multirun