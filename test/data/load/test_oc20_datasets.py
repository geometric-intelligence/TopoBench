"""Unit tests for OC20 and OC22 dataset loaders."""

import os
import pytest
import torch
import hydra
from pathlib import Path
from omegaconf import DictConfig, OmegaConf

# from topobench.data.loaders.graph.oc20_is2re_dataset_loader import IS2REDatasetLoader
# from topobench.data.loaders.graph.oc22_is2re_dataset_loader import OC22IS2REDatasetLoader
from topobench.data.loaders.graph.oc20_dataset_loader import OC20DatasetLoader
from topobench.data.loaders.graph.oc20_asedbs2ef_loader import OC20ASEDBDataset
# from topobench.data.datasets.oc20_is2re_dataset import IS2REDataset
# from topobench.data.datasets.oc22_is2re_dataset import OC22IS2REDataset
from topobench.data.datasets.oc20_dataset import OC20Dataset
from topobench.utils.config_resolvers import (
    get_default_metrics,
    get_default_trainer,
    get_default_transform,
    get_flattened_channels,
    get_monitor_metric,
    get_monitor_mode,
    get_non_relational_out_channels,
    get_required_lifting,
    infer_in_channels,
    infer_num_cell_dimensions,
    infer_topotune_num_cell_dimensions,
)


def register_resolvers():
    """Register OmegaConf resolvers for tests."""
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
        "get_flattened_channels",
        get_flattened_channels,
        replace=True,
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
        "get_non_relational_out_channels",
        get_non_relational_out_channels,
        replace=True,
    )
    OmegaConf.register_new_resolver(
        "infer_in_channels", infer_in_channels, replace=True
    )
    OmegaConf.register_new_resolver(
        "infer_num_cell_dimensions", infer_num_cell_dimensions, replace=True
    )
    OmegaConf.register_new_resolver(
        "infer_topotune_num_cell_dimensions",
        infer_topotune_num_cell_dimensions,
        replace=True,
    )
    OmegaConf.register_new_resolver(
        "parameter_multiplication", lambda x, y: int(int(x) * int(y)), replace=True
    )


def setup_project_root():
    """Set up PROJECT_ROOT environment variable for tests."""
    # Get the path to the test file's directory, then go up 3 levels to project root
    test_file_dir = Path(__file__).resolve().parent
    project_root = test_file_dir.parent.parent.parent
    os.environ["PROJECT_ROOT"] = str(project_root)


# class TestOC20IS2REDatasetLoader:
#     """Test suite for OC20 IS2RE dataset loader."""

#     @pytest.fixture(autouse=True)
#     def setup(self):
#         """Setup test environment."""
#         hydra.core.global_hydra.GlobalHydra.instance().clear()
#         register_resolvers()
#         setup_project_root()
#         self.relative_config_dir = "../../../configs"

#     def test_loader_initialization(self):
#         """Test that the IS2RE loader can be initialized."""
#         with hydra.initialize(
#             version_base="1.3",
#             config_path=self.relative_config_dir,
#             job_name="test_oc20_is2re"
#         ):
#             cfg = hydra.compose(
#                 config_name="run.yaml",
#                 overrides=["dataset=graph/OC20_IS2RE"],
#             )
#             print('Test that the OC20 IS2RE loader can be initialized')
#             loader = hydra.utils.instantiate(cfg.dataset.loader)
#             assert isinstance(loader, IS2REDatasetLoader)
#             assert loader.parameters.data_name == "OC20_IS2RE"
#             assert loader.parameters.task == "is2re"

#     def test_dataset_loading(self):
#         """Test that the IS2RE dataset loads correctly."""
#         with hydra.initialize(
#             version_base="1.3",
#             config_path=self.relative_config_dir,
#             job_name="test_oc20_is2re_load"
#         ):
#             cfg = hydra.compose(
#                 config_name="run.yaml",
#                 overrides=["dataset=graph/OC20_IS2RE"],
#             )
#             print('Test that the OC20 IS2RE dataset loads correctly')
#             loader = hydra.utils.instantiate(cfg.dataset.loader)
#             dataset, data_dir = loader.load()
            
#             # Check dataset type
#             assert isinstance(dataset, IS2REDataset)
            
#             # Check dataset has required attributes
#             assert hasattr(dataset, 'split_idx')
#             assert 'train' in dataset.split_idx
#             assert 'valid' in dataset.split_idx
#             assert 'test' in dataset.split_idx
            
#             # Check splits are not empty (when max_samples is set)
#             assert len(dataset.split_idx['train']) > 0
#             assert len(dataset.split_idx['valid']) > 0
#             assert len(dataset.split_idx['test']) > 0
            
#             # Check dataset length
#             assert len(dataset) > 0

#     def test_dataset_item_access(self):
#         """Test accessing individual items from the dataset."""
#         with hydra.initialize(
#             version_base="1.3",
#             config_path=self.relative_config_dir,
#             job_name="test_oc20_is2re_item"
#         ):
#             cfg = hydra.compose(
#                 config_name="run.yaml",
#                 overrides=["dataset=graph/OC20_IS2RE"],
#             )
#             print('Test that the OC20 IS2RE dataset loads correctly')
#             loader = hydra.utils.instantiate(cfg.dataset.loader)
#             dataset, _ = loader.load()
            
#             # Get first item
#             data = dataset[0]
            
#             # Check data has required PyG attributes
#             assert hasattr(data, 'x')
#             assert hasattr(data, 'edge_index')
#             assert hasattr(data, 'y')
            
#             # Check data types
#             assert isinstance(data.x, torch.Tensor)
#             assert isinstance(data.edge_index, torch.Tensor)
#             assert isinstance(data.y, torch.Tensor)
            
#             # Check shapes
#             assert data.x.dim() >= 1
#             assert data.edge_index.dim() == 2
#             assert data.edge_index.size(0) == 2  # [2, num_edges]

#     def test_split_indices_validity(self):
#         """Test that split indices are valid and non-overlapping."""
#         with hydra.initialize(
#             version_base="1.3",
#             config_path=self.relative_config_dir,
#             job_name="test_oc20_is2re_splits"
#         ):
#             cfg = hydra.compose(
#                 config_name="run.yaml",
#                 overrides=["dataset=graph/OC20_IS2RE"],
#             )
#             print('Test that the OC20 IS2RE dataset loads correctly')
#             loader = hydra.utils.instantiate(cfg.dataset.loader)
#             dataset, _ = loader.load()
            
#             train_idx = dataset.split_idx['train'].numpy()
#             val_idx = dataset.split_idx['valid'].numpy()
#             test_idx = dataset.split_idx['test'].numpy()
            
#             # Check no overlap between splits
#             assert len(set(train_idx) & set(val_idx)) == 0
#             assert len(set(train_idx) & set(test_idx)) == 0
#             # val and test might overlap if test reuses val when test is not available
            
#             # Check all indices are within dataset bounds
#             all_indices = list(train_idx) + list(val_idx)
#             assert all(0 <= idx < len(dataset) for idx in all_indices)


# class TestOC22IS2REDatasetLoader:
#     """Test suite for OC22 IS2RE dataset loader."""

#     @pytest.fixture(autouse=True)
#     def setup(self):
#         """Setup test environment."""
#         hydra.core.global_hydra.GlobalHydra.instance().clear()
#         register_resolvers()
#         setup_project_root()
#         self.relative_config_dir = "../../../configs"

#     def test_loader_initialization(self):
#         """Test that the OC22 IS2RE loader can be initialized."""
#         with hydra.initialize(
#             version_base="1.3",
#             config_path=self.relative_config_dir,
#             job_name="test_oc22_is2re"
#         ):
#             cfg = hydra.compose(
#                 config_name="run.yaml",
#                 overrides=["dataset=graph/OC22_IS2RE"],
#             )
#             print('Test that the OC20 IS2RE loader can be initialized')
#             loader = hydra.utils.instantiate(cfg.dataset.loader)
#             assert isinstance(loader, OC22IS2REDatasetLoader)
#             assert loader.parameters.data_name == "OC22_IS2RE"
#             assert loader.parameters.task == "oc22_is2re"

#     def test_dataset_loading(self):
#         """Test that the OC22 IS2RE dataset loads correctly."""
#         with hydra.initialize(
#             version_base="1.3",
#             config_path=self.relative_config_dir,
#             job_name="test_oc22_is2re_load"
#         ):
#             cfg = hydra.compose(
#                 config_name="run.yaml",
#                 overrides=["dataset=graph/OC22_IS2RE"],
#             )
#             print('Test that the OC20 IS2RE dataset loads correctly')
#             loader = hydra.utils.instantiate(cfg.dataset.loader)
#             dataset, data_dir = loader.load()
            
#             # Check dataset type
#             assert isinstance(dataset, OC22IS2REDataset)
            
#             # Check dataset has required attributes
#             assert hasattr(dataset, 'split_idx')
#             assert 'train' in dataset.split_idx
#             assert 'valid' in dataset.split_idx
#             assert 'test' in dataset.split_idx
            
#             # Check splits are not empty (when max_samples is set)
#             assert len(dataset.split_idx['train']) > 0
#             assert len(dataset.split_idx['valid']) > 0
#             assert len(dataset.split_idx['test']) > 0
            
#             # Check dataset length
#             assert len(dataset) > 0

#     def test_dataset_item_access(self):
#         """Test accessing individual items from the OC22 dataset."""
#         with hydra.initialize(
#             version_base="1.3",
#             config_path=self.relative_config_dir,
#             job_name="test_oc22_is2re_item"
#         ):
#             cfg = hydra.compose(
#                 config_name="run.yaml",
#                 overrides=["dataset=graph/OC22_IS2RE"],
#             )
#             print('Test that the OC20 IS2RE dataset loads correctly')
#             loader = hydra.utils.instantiate(cfg.dataset.loader)
#             dataset, _ = loader.load()
            
#             # Get first item
#             data = dataset[0]
            
#             # Check data has required PyG attributes
#             assert hasattr(data, 'x')
#             assert hasattr(data, 'edge_index')
#             assert hasattr(data, 'y')
            
#             # Check data types
#             assert isinstance(data.x, torch.Tensor)
#             assert isinstance(data.edge_index, torch.Tensor)
#             assert isinstance(data.y, torch.Tensor)
            
#             # Check shapes
#             assert data.x.dim() >= 1
#             assert data.edge_index.dim() == 2
#             assert data.edge_index.size(0) == 2

#     def test_split_indices_validity(self):
#         """Test that split indices are valid and non-overlapping."""
#         with hydra.initialize(
#             version_base="1.3",
#             config_path=self.relative_config_dir,
#             job_name="test_oc22_is2re_splits"
#         ):
#             cfg = hydra.compose(
#                 config_name="run.yaml",
#                 overrides=["dataset=graph/OC22_IS2RE"],
#             )
#             print('Test that the OC20 IS2RE dataset loads correctly')
#             loader = hydra.utils.instantiate(cfg.dataset.loader)
#             dataset, _ = loader.load()
            
#             train_idx = dataset.split_idx['train'].numpy()
#             val_idx = dataset.split_idx['valid'].numpy()
#             test_idx = dataset.split_idx['test'].numpy()
            
#             # Check no overlap between train and val
#             assert len(set(train_idx) & set(val_idx)) == 0
            
#             # Check all indices are within dataset bounds
#             all_indices = list(train_idx) + list(val_idx)
#             assert all(0 <= idx < len(dataset) for idx in all_indices)


class TestOC20S2EFDatasetLoader:
    """Test suite for OC20 S2EF dataset loader."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment."""
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        register_resolvers()
        setup_project_root()
        self.relative_config_dir = "../../../configs"

    def test_loader_initialization_200k(self):
        """Test that the S2EF 200K loader can be initialized."""
        with hydra.initialize(
            version_base="1.3",
            config_path=self.relative_config_dir,
            job_name="test_oc20_s2ef_200k"
        ):
            cfg = hydra.compose(
                config_name="run.yaml",
                overrides=["dataset=graph/OC20_S2EF_200K_mock"],
            )
            print('Test that the OC20 S2EF 200K loader can be initialized')
            loader = hydra.utils.instantiate(cfg.dataset.loader)
            assert isinstance(loader, OC20DatasetLoader)
            assert loader.parameters.task == "s2ef"
            assert loader.parameters.train_split == "200K"

    def test_dataset_loading_200k(self):
        """Test that the S2EF 200K dataset loads correctly."""
        with hydra.initialize(
            version_base="1.3",
            config_path=self.relative_config_dir,
            job_name="test_oc20_s2ef_200k_load"
        ):
            cfg = hydra.compose(
                config_name="run.yaml",
                overrides=["dataset=graph/OC20_S2EF_200K_mock"],
            )
            print('Test that the OC20 S2EF 200K dataset loads correctly')
            loader = hydra.utils.instantiate(cfg.dataset.loader)
            dataset, data_dir = loader.load()
            
            # Check dataset type (S2EF uses ASE DB backend)
            assert isinstance(dataset, OC20ASEDBDataset)
            
            # Check dataset has required attributes
            assert hasattr(dataset, 'split_idx')
            assert 'train' in dataset.split_idx
            assert 'valid' in dataset.split_idx
            assert 'test' in dataset.split_idx
            
            # Check splits are not empty
            assert len(dataset.split_idx['train']) > 0
            # With val_splits=[], validation data will come from random split (if split_type=random)
            # Otherwise it will be empty
            # S2EF test data is LMDB format (incompatible with .extxyz/ASE DB), so test split is empty
            assert len(dataset.split_idx['test']) == 0
            
            # Check dataset length
            assert len(dataset) > 0

    def test_dataset_item_access_s2ef(self):
        """Test accessing individual items from the S2EF dataset."""
        with hydra.initialize(
            version_base="1.3",
            config_path=self.relative_config_dir,
            job_name="test_oc20_s2ef_item"
        ):
            cfg = hydra.compose(
                config_name="run.yaml",
                overrides=["dataset=graph/OC20_S2EF_200K_mock"],
            )
            print('Test that the OC20 S2EF dataset loads correctly')
            loader = hydra.utils.instantiate(cfg.dataset.loader)
            dataset, _ = loader.load()
            
            # Get first item
            data = dataset[0]
            
            # Check data has required PyG attributes
            assert hasattr(data, 'x')
            assert hasattr(data, 'edge_index')
            
            # Check data types
            assert isinstance(data.x, torch.Tensor)
            assert isinstance(data.edge_index, torch.Tensor)
            
            # Check shapes
            assert data.x.dim() >= 1
            assert data.edge_index.dim() == 2
            assert data.edge_index.size(0) == 2

    def test_validation_splits_configuration(self):
        """Test that validation splits can be configured."""
        with hydra.initialize(
            version_base="1.3",
            config_path=self.relative_config_dir,
            job_name="test_oc20_s2ef_val_splits"
        ):
            # Test with val_id only
            cfg = hydra.compose(
                config_name="run.yaml",
                overrides=["dataset=graph/OC20_S2EF_200K_mock"],
            )
            print('Test that the OC20 S2EF 200K loader can be initialized')
            loader = hydra.utils.instantiate(cfg.dataset.loader)
            # val_splits=[] (empty list) for mock config means no separate validation files
            # (validation will come from random split of train data)
            assert loader.parameters.val_splits == []

    def test_split_indices_validity_s2ef(self):
        """Test that S2EF split indices are valid."""
        with hydra.initialize(
            version_base="1.3",
            config_path=self.relative_config_dir,
            job_name="test_oc20_s2ef_splits"
        ):
            cfg = hydra.compose(
                config_name="run.yaml",
                overrides=["dataset=graph/OC20_S2EF_200K_mock"],
            )
            print('Test that the OC20 S2EF 200K dataset loads correctly')
            loader = hydra.utils.instantiate(cfg.dataset.loader)
            dataset, _ = loader.load()
            
            # ASE DB dataset uses lists for split indices, not tensors
            train_idx = dataset.split_idx['train']
            valid_idx = dataset.split_idx['valid']
            test_idx = dataset.split_idx['test']
            
            # Check indices are valid (note: max_samples truncates dataset but indices reflect original positions)
            assert len(train_idx) > 0
            # With val_splits=[], valid_idx may be empty at dataset level
            # (preprocessor will create splits later with random splitting)
            # Only check non-overlap if both have data
            if len(valid_idx) > 0:
                assert len(set(train_idx) & set(valid_idx)) == 0


class TestOC20DatasetIntegration:
    """Integration tests for OC20 datasets with PreProcessor."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment."""
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        register_resolvers()
        setup_project_root()
        self.relative_config_dir = "../../../configs"

#     def test_is2re_with_preprocessor(self):
#         """Test IS2RE dataset with PreProcessor."""
#         with hydra.initialize(
#             version_base="1.3",
#             config_path=self.relative_config_dir,
#             job_name="test_is2re_preprocessor"
#         ):
#             cfg = hydra.compose(
#                 config_name="run.yaml",
#                 overrides=[
#                     "dataset=graph/OC20_IS2RE",
#                     "model=graph/gcn",
#                 ],
#                 return_hydra_config=True,
#             )
            
#             # Load dataset
#             loader = hydra.utils.instantiate(cfg.dataset.loader)
#             dataset, data_dir = loader.load()
            
#             # Use preprocessor
#             from topobench.data.preprocessor import PreProcessor
#             transform_config = cfg.get("transforms", None)
#             preprocessor = PreProcessor(dataset, data_dir, transform_config)
            
#             # Load splits
#             dataset_train, dataset_val, dataset_test = preprocessor.load_dataset_splits(
#                 cfg.dataset.split_params
#             )
            
#             # Verify splits exist and are not empty
#             assert dataset_train is not None
#             assert dataset_val is not None
#             assert dataset_test is not None
#             assert len(dataset_train) > 0
#             assert len(dataset_val) > 0
#             assert len(dataset_test) > 0

#     def test_oc22_is2re_with_preprocessor(self):
#         """Test OC22 IS2RE dataset with PreProcessor."""
#         with hydra.initialize(
#             version_base="1.3",
#             config_path=self.relative_config_dir,
#             job_name="test_oc22_preprocessor"
#         ):
#             cfg = hydra.compose(
#                 config_name="run.yaml",
#                 overrides=[
#                     "dataset=graph/OC22_IS2RE",
#                     "model=graph/gcn",
#                 ],
#                 return_hydra_config=True,
#             )
#             print('Config used for OC22 IS2RE preprocessor test:\n', OmegaConf.to_yaml(cfg))
            
#             # Load dataset
#             loader = hydra.utils.instantiate(cfg.dataset.loader)
#             dataset, data_dir = loader.load()
            
#             # Use preprocessor
#             from topobench.data.preprocessor import PreProcessor
#             transform_config = cfg.get("transforms", None)
#             preprocessor = PreProcessor(dataset, data_dir, transform_config)
            
#             # Load splits
#             dataset_train, dataset_val, dataset_test = preprocessor.load_dataset_splits(
#                 cfg.dataset.split_params
#             )
            
#             # Verify splits exist and are not empty
#             assert dataset_train is not None
#             assert dataset_val is not None
#             assert dataset_test is not None
#             assert len(dataset_train) > 0
#             assert len(dataset_val) > 0
#             assert len(dataset_test) > 0

    def test_s2ef_with_preprocessor(self):
        """Test mock S2EF dataset with PreProcessor."""
        with hydra.initialize(
            version_base="1.3",
            config_path=self.relative_config_dir,
            job_name="test_s2ef_preprocessor"
        ):
            cfg = hydra.compose(
                config_name="run.yaml",
                overrides=["dataset=graph/OC20_S2EF_200K_mock", "model=graph/gcn"]
            )
            print('Test that the OC20 S2EF 200K loader can be initialized')
            
            # Load dataset
            loader = hydra.utils.instantiate(cfg.dataset.loader)
            dataset, data_dir = loader.load()
            
            # Use preprocessor
            from topobench.data.preprocessor import PreProcessor
            transform_config = cfg.get("transforms", None)
            preprocessor = PreProcessor(dataset, data_dir, transform_config)
            
            # Load splits
            dataset_train, dataset_val, dataset_test = preprocessor.load_dataset_splits(
                cfg.dataset.split_params
            )
            
            # Verify splits exist and train/val are not empty
            assert dataset_train is not None
            assert dataset_val is not None
            assert dataset_test is not None
            assert len(dataset_train) > 0
            assert len(dataset_val) > 0
            # S2EF test may be empty if using the official splits, no test for this