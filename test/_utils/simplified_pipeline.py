"""Test pipeline for a particular dataset and model."""

import hydra
from omegaconf import DictConfig, OmegaConf

from topobench.data.preprocessor import OnDiskPreProcessor, PreProcessor
from topobench.data.utils import build_cluster_transform
from topobench.dataloader import ClusterGCNDataModule, TBDataloader
from topobench.utils import instantiate_callbacks
from topobench.utils.config_resolvers import (
    get_default_metrics,
    get_default_trainer,
    get_default_transform,
    get_monitor_metric,
    get_monitor_mode,
    get_required_lifting,
    infer_in_channels,
    infer_num_cell_dimensions,
)

OmegaConf.register_new_resolver(
    "get_default_metrics", get_default_metrics, replace=True
)
OmegaConf.register_new_resolver(
    "get_default_trainer", get_default_trainer, replace=True
)
OmegaConf.register_new_resolver(
    "get_default_transform", get_default_transform, replace=True
)
OmegaConf.register_new_resolver(
    "get_required_lifting", get_required_lifting, replace=True
)
OmegaConf.register_new_resolver(
    "get_monitor_metric", get_monitor_metric, replace=True
)
OmegaConf.register_new_resolver(
    "get_monitor_mode", get_monitor_mode, replace=True
)
OmegaConf.register_new_resolver(
    "infer_in_channels", infer_in_channels, replace=True
)
OmegaConf.register_new_resolver(
    "infer_num_cell_dimensions", infer_num_cell_dimensions, replace=True
)
OmegaConf.register_new_resolver(
    "parameter_multiplication", lambda x, y: int(int(x) * int(y)), replace=True
)

def run(cfg: DictConfig) -> DictConfig:
    """Run pipeline with given configuration.
    
    Parameters
    ----------
    cfg : DictConfig
        Configuration.
    """
    # Instantiate and load dataset
    dataset_loader = hydra.utils.instantiate(cfg.dataset.loader)
    dataset, dataset_dir = dataset_loader.load()
    
    # Preprocess dataset and load the splits
    transform_config = cfg.get("transforms", None)
    print(transform_config)
    
    # memory_type and learning_setting
    memory_type = cfg.dataset.loader.parameters.get("memory_type", "in_memory")
    learning_setting = cfg.dataset.get("split_params", {}).get("learning_setting", "inductive")
    
    # Branches for three cases:
    if memory_type == "on_disk_cluster":
        # Loads a graph in memory, performs partitioning
        preprocessor = PreProcessor(dataset, dataset_dir, None)
        post_batch_transform = build_cluster_transform(transform_config)

        handle = preprocessor.pack_global_partition(
            split_params=cfg.dataset.get("split_params", {}),
            cluster_params=cfg.dataset.loader.parameters.get("cluster", {}),
            stream_params=cfg.dataset.loader.parameters.get("stream", {}),
            dtype_policy=cfg.dataset.loader.parameters.get("dtype_policy", "preserve"),
            pack_db=True, # for future use
            pack_memmaps=True
        )

        # Build streaming loaders that return batches with:
        # edge_index, x, (edge_attr), y, supervised_mask, global_nid, num_nodes
        datamodule = ClusterGCNDataModule(
            data_handle=handle,
            q=cfg.dataset.loader.parameters.get("stream", {}).get("q", 1),
            num_workers=cfg.dataset.loader.parameters.get("stream", {}).get("num_workers", 0),
            pin_memory=cfg.dataset.loader.parameters.get("stream", {}).get("pin_memory", False),
            with_edge_attr=cfg.dataset.loader.parameters.get("stream", {}).get("with_edge_attr", False),
            eval_cover_strategy=cfg.get("eval", {}).get("cover_strategy", "all_parts"),
            seed=cfg.get("seed", 42),
            post_batch_transform=post_batch_transform,
        )
    elif memory_type == "on_disk" and learning_setting == "transductive":
        # Transductive on-disk pipeline (pre-partitioned transductive dataset)
        handle = dataset.handle
        post_batch_transform = build_cluster_transform(transform_config)

        # Build streaming loaders that return batches with:
        # edge_index, x, (edge_attr), y, supervised_mask, global_nid, num_nodes
        datamodule = ClusterGCNDataModule(
            data_handle=handle,
            q=cfg.dataset.loader.parameters.get("stream", {}).get("q", 10),
            num_workers=cfg.dataset.loader.parameters.get("stream", {}).get("num_workers", 0),
            pin_memory=cfg.dataset.loader.parameters.get("stream", {}).get("pin_memory", False),
            with_edge_attr=cfg.dataset.loader.parameters.get("stream", {}).get("with_edge_attr", False),
            eval_cover_strategy=cfg.get("eval", {}).get("cover_strategy", "all_parts"),
            seed=cfg.get("seed", 42),
            post_batch_transform=post_batch_transform,
        )
    else:
        # TB standard in-memory pipeline and on-disk inductive pipeline
        preprocessor_cls = OnDiskPreProcessor if memory_type == "on_disk" else PreProcessor
        preprocessor = preprocessor_cls(dataset, dataset_dir, transform_config)

        dataset_train, dataset_val, dataset_test = (
            preprocessor.load_dataset_splits(cfg.dataset.split_params)
        )
        # Prepare datamodule
        if cfg.dataset.parameters.task_level in ["node", "graph"]:
            datamodule = TBDataloader(
                dataset_train=dataset_train,
                dataset_val=dataset_val,
                dataset_test=dataset_test,
                **cfg.dataset.get("dataloader_params", {}),
            )
        else:
            raise ValueError("Invalid task_level")

    # Model for us is Network + logic: inputs backbone, readout, losses
    model = hydra.utils.instantiate(
        cfg.model,
        evaluator=cfg.evaluator,
        optimizer=cfg.optimizer,
        loss=cfg.loss,
    )
    callbacks = instantiate_callbacks(cfg.get("callbacks"))
    trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=False,
        num_sanity_val_steps=0,
    )
    trainer.fit(
        model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path")
    )
    ckpt_path = trainer.checkpoint_callback.best_model_path
    trainer.test(
        model=model, datamodule=datamodule, ckpt_path=ckpt_path
    )

            
        
