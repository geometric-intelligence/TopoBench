python -m topobench model=GCN dataset=hm-categories model.readout.hidden_dim=    64.0
Name: 468, dtype: object model.backbone.dropout=    0.0
Name: 468, dtype: object model.backbone.num_layers=    2.0
Name: 468, dtype: object model.backbone.in_channels=    64.0
Name: 468, dtype: object model.backbone.hidden_channels=    64.0
Name: 468, dtype: object model.feature_encoder.out_channels=    64.0
Name: 468, dtype: object model.feature_encoder.proj_dropout=    0.0
Name: 468, dtype: object model.backbone_wrapper.out_channels=    64.0
Name: 468, dtype: object optimizer.parameters.lr=    0.003
Name: 468, dtype: object dataset.loader.parameters.data_name=hm-categories dataset.loader.parameters.data_type=graphland dataset.loader.parameters.data_domain=graph dataset.loader.parameters.drop_missing_y=False dataset.loader.parameters.impute_missing_x.copy=True dataset.loader.parameters.impute_missing_x.strategy=most_frequent dataset.loader.parameters.impute_missing_x.add_indicator=False dataset.parameters.task=classification dataset.parameters.loss_type=cross_entropy dataset.parameters.task_level=node dataset.parameters.num_classes=21.0 dataset.parameters.num_features=35.0 dataset.split_params.k=10.0 dataset.split_params.split_type=stratified dataset.split_params.train_prop=0.5 dataset.split_params.learning_setting=transductive dataset.dataloader_params.batch_size=1.0 dataset.dataloader_params.pin_memory=False dataset.dataloader_params.num_workers=0.0 callbacks.early_stopping.strict=True callbacks.early_stopping.verbose=True callbacks.early_stopping.patience=10.0 callbacks.early_stopping.min_delta=0.0 callbacks.early_stopping.check_finite=True optimizer.optimizer_id=Adam optimizer.scheduler.scheduler_id=StepLR optimizer.scheduler.scheduler_params.gamma=0.5 optimizer.scheduler.scheduler_params.step_size=50.0 optimizer.parameters.weight_decay=0.0 dataset.split_params.data_seed=0,3,5,7,9 logger.wandb.project=TopoBench_Reproduction --multirun
python -m topobench model=GIN dataset=hm-categories model.readout.hidden_dim=    64.0
Name: 292, dtype: object model.backbone.dropout=    0.0
Name: 292, dtype: object model.backbone.num_layers=    1.0
Name: 292, dtype: object model.backbone.in_channels=    64.0
Name: 292, dtype: object model.backbone.hidden_channels=    64.0
Name: 292, dtype: object model.feature_encoder.out_channels=    64.0
Name: 292, dtype: object model.feature_encoder.proj_dropout=    0.0
Name: 292, dtype: object model.backbone_wrapper.out_channels=    64.0
Name: 292, dtype: object optimizer.parameters.lr=    0.003
Name: 292, dtype: object dataset.loader.parameters.data_name=hm-categories dataset.loader.parameters.data_type=graphland dataset.loader.parameters.data_domain=graph dataset.loader.parameters.drop_missing_y=False dataset.loader.parameters.impute_missing_x.copy=True dataset.loader.parameters.impute_missing_x.strategy=most_frequent dataset.loader.parameters.impute_missing_x.add_indicator=False dataset.parameters.task=classification dataset.parameters.loss_type=cross_entropy dataset.parameters.task_level=node dataset.parameters.num_classes=21.0 dataset.parameters.num_features=35.0 dataset.split_params.k=10.0 dataset.split_params.split_type=stratified dataset.split_params.train_prop=0.5 dataset.split_params.learning_setting=transductive dataset.dataloader_params.batch_size=1.0 dataset.dataloader_params.pin_memory=False dataset.dataloader_params.num_workers=0.0 callbacks.early_stopping.strict=True callbacks.early_stopping.verbose=True callbacks.early_stopping.patience=10.0 callbacks.early_stopping.min_delta=0.0 callbacks.early_stopping.check_finite=True optimizer.optimizer_id=Adam optimizer.scheduler.scheduler_id=StepLR optimizer.scheduler.scheduler_params.gamma=0.5 optimizer.scheduler.scheduler_params.step_size=50.0 optimizer.parameters.weight_decay=0.0 dataset.split_params.data_seed=0,3,5,7,9 logger.wandb.project=TopoBench_Reproduction --multirun
python -m topobench model=GCN dataset=pokec-regions model.readout.hidden_dim=    32.0
Name: 288, dtype: object model.backbone.dropout=    0.0
Name: 288, dtype: object model.backbone.num_layers=    8.0
Name: 288, dtype: object model.backbone.in_channels=    32.0
Name: 288, dtype: object model.backbone.hidden_channels=    32.0
Name: 288, dtype: object model.feature_encoder.out_channels=    32.0
Name: 288, dtype: object model.feature_encoder.proj_dropout=    0.0
Name: 288, dtype: object model.backbone_wrapper.out_channels=    32.0
Name: 288, dtype: object optimizer.parameters.lr=    0.003
Name: 288, dtype: object dataset.loader.parameters.data_name=pokec-regions dataset.loader.parameters.data_type=graphland dataset.loader.parameters.data_domain=graph dataset.loader.parameters.drop_missing_y=False dataset.loader.parameters.impute_missing_x.copy=True dataset.loader.parameters.impute_missing_x.strategy=most_frequent dataset.loader.parameters.impute_missing_x.add_indicator=False dataset.parameters.task=classification dataset.parameters.loss_type=cross_entropy dataset.parameters.task_level=node dataset.parameters.num_classes=183.0 dataset.parameters.num_features=56.0 dataset.split_params.k=10.0 dataset.split_params.split_type=stratified dataset.split_params.train_prop=0.5 dataset.split_params.learning_setting=transductive dataset.dataloader_params.batch_size=1.0 dataset.dataloader_params.pin_memory=False dataset.dataloader_params.num_workers=0.0 callbacks.early_stopping.strict=True callbacks.early_stopping.verbose=True callbacks.early_stopping.patience=10.0 callbacks.early_stopping.min_delta=0.0 callbacks.early_stopping.check_finite=True optimizer.optimizer_id=Adam optimizer.scheduler.scheduler_id=StepLR optimizer.scheduler.scheduler_params.gamma=0.5 optimizer.scheduler.scheduler_params.step_size=50.0 optimizer.parameters.weight_decay=0.0 dataset.split_params.data_seed=0,3,5,7,9 logger.wandb.project=TopoBench_Reproduction --multirun
python -m topobench model=GCN dataset=tolokers-2 model.readout.hidden_dim=    64.0
Name: 480, dtype: object model.backbone.dropout=    0.0
Name: 480, dtype: object model.backbone.num_layers=    3.0
Name: 480, dtype: object model.backbone.in_channels=    64.0
Name: 480, dtype: object model.backbone.hidden_channels=    64.0
Name: 480, dtype: object model.feature_encoder.out_channels=    64.0
Name: 480, dtype: object model.feature_encoder.proj_dropout=    0.0
Name: 480, dtype: object model.backbone_wrapper.out_channels=    64.0
Name: 480, dtype: object optimizer.parameters.lr=    0.003
Name: 480, dtype: object dataset.loader.parameters.data_name=tolokers-2 dataset.loader.parameters.data_type=graphland dataset.loader.parameters.data_domain=graph dataset.loader.parameters.drop_missing_y=False dataset.loader.parameters.impute_missing_x.copy=True dataset.loader.parameters.impute_missing_x.strategy=most_frequent dataset.loader.parameters.impute_missing_x.add_indicator=False dataset.parameters.task=classification dataset.parameters.loss_type=cross_entropy dataset.parameters.task_level=node dataset.parameters.num_classes=2.0 dataset.parameters.num_features=16.0 dataset.split_params.k=10.0 dataset.split_params.split_type=stratified dataset.split_params.train_prop=0.5 dataset.split_params.learning_setting=transductive dataset.dataloader_params.batch_size=1.0 dataset.dataloader_params.pin_memory=False dataset.dataloader_params.num_workers=0.0 callbacks.early_stopping.strict=True callbacks.early_stopping.verbose=True callbacks.early_stopping.patience=10.0 callbacks.early_stopping.min_delta=0.0 callbacks.early_stopping.check_finite=True optimizer.optimizer_id=Adam optimizer.scheduler.scheduler_id=StepLR optimizer.scheduler.scheduler_params.gamma=0.5 optimizer.scheduler.scheduler_params.step_size=50.0 optimizer.parameters.weight_decay=0.0 dataset.split_params.data_seed=0,3,5,7,9 logger.wandb.project=TopoBench_Reproduction --multirun
python -m topobench model=GCN dataset=city-reviews model.readout.hidden_dim=    64.0
Name: 587, dtype: object model.backbone.dropout=    0.1
Name: 587, dtype: object model.backbone.num_layers=    8.0
Name: 587, dtype: object model.backbone.in_channels=    64.0
Name: 587, dtype: object model.backbone.hidden_channels=    64.0
Name: 587, dtype: object model.feature_encoder.out_channels=    64.0
Name: 587, dtype: object model.feature_encoder.proj_dropout=    0.0
Name: 587, dtype: object model.backbone_wrapper.out_channels=    64.0
Name: 587, dtype: object optimizer.parameters.lr=    0.003
Name: 587, dtype: object dataset.loader.parameters.data_name=city-reviews dataset.loader.parameters.data_type=graphland dataset.loader.parameters.data_domain=graph dataset.loader.parameters.drop_missing_y=False dataset.loader.parameters.impute_missing_x.copy=True dataset.loader.parameters.impute_missing_x.strategy=most_frequent dataset.loader.parameters.impute_missing_x.add_indicator=False dataset.parameters.task=classification dataset.parameters.loss_type=cross_entropy dataset.parameters.task_level=node dataset.parameters.num_classes=2.0 dataset.parameters.num_features=37.0 dataset.split_params.k=10.0 dataset.split_params.split_type=stratified dataset.split_params.train_prop=0.5 dataset.split_params.learning_setting=transductive dataset.dataloader_params.batch_size=1.0 dataset.dataloader_params.pin_memory=False dataset.dataloader_params.num_workers=0.0 callbacks.early_stopping.strict=True callbacks.early_stopping.verbose=True callbacks.early_stopping.patience=10.0 callbacks.early_stopping.min_delta=0.0 callbacks.early_stopping.check_finite=True optimizer.optimizer_id=Adam optimizer.scheduler.scheduler_id=StepLR optimizer.scheduler.scheduler_params.gamma=0.5 optimizer.scheduler.scheduler_params.step_size=50.0 optimizer.parameters.weight_decay=0.0 dataset.split_params.data_seed=0,3,5,7,9 logger.wandb.project=TopoBench_Reproduction --multirun
python -m topobench model=GCN dataset=artnet-exp model.readout.hidden_dim=    64.0
Name: 500, dtype: object model.backbone.dropout=    0.0
Name: 500, dtype: object model.backbone.num_layers=    4.0
Name: 500, dtype: object model.backbone.in_channels=    64.0
Name: 500, dtype: object model.backbone.hidden_channels=    64.0
Name: 500, dtype: object model.feature_encoder.out_channels=    64.0
Name: 500, dtype: object model.feature_encoder.proj_dropout=    0.1
Name: 500, dtype: object model.backbone_wrapper.out_channels=    64.0
Name: 500, dtype: object optimizer.parameters.lr=    0.003
Name: 500, dtype: object dataset.loader.parameters.data_name=artnet-exp dataset.loader.parameters.data_type=graphland dataset.loader.parameters.data_domain=graph dataset.loader.parameters.drop_missing_y=False dataset.loader.parameters.impute_missing_x.copy=True dataset.loader.parameters.impute_missing_x.strategy=most_frequent dataset.loader.parameters.impute_missing_x.add_indicator=False dataset.parameters.task=classification dataset.parameters.loss_type=cross_entropy dataset.parameters.task_level=node dataset.parameters.num_classes=2.0 dataset.parameters.num_features=75.0 dataset.split_params.k=10.0 dataset.split_params.split_type=stratified dataset.split_params.train_prop=0.5 dataset.split_params.learning_setting=transductive dataset.dataloader_params.batch_size=1.0 dataset.dataloader_params.pin_memory=False dataset.dataloader_params.num_workers=0.0 callbacks.early_stopping.strict=True callbacks.early_stopping.verbose=True callbacks.early_stopping.patience=10.0 callbacks.early_stopping.min_delta=0.0 callbacks.early_stopping.check_finite=True optimizer.optimizer_id=Adam optimizer.scheduler.scheduler_id=StepLR optimizer.scheduler.scheduler_params.gamma=0.5 optimizer.scheduler.scheduler_params.step_size=50.0 optimizer.parameters.weight_decay=0.0 dataset.split_params.data_seed=0,3,5,7,9 logger.wandb.project=TopoBench_Reproduction --multirun
