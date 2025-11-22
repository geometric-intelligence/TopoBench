"""Unit tests for OC20 and OC22 dataset loaders."""

import pytest
import torch
import hydra
from pathlib import Path
from omegaconf import DictConfig

from topobench.data.loaders.graph.oc20_is2re_dataset_loader import IS2REDatasetLoader
from topobench.data.loaders.graph.oc22_is2re_dataset_loader import OC22IS2REDatasetLoader
from topobench.data.loaders.graph.oc20_dataset_loader import OC20DatasetLoader
from topobench.data.datasets.oc20_is2re_dataset import IS2REDataset
from topobench.data.datasets.oc22_is2re_dataset import OC22IS2REDataset
from topobench.data.datasets.oc20_dataset import OC20Dataset


class TestOC20IS2REDatasetLoader:
    """Test suite for OC20 IS2RE dataset loader."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment."""
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        self.relative_config_dir = "../../../configs"

    def test_loader_initialization(self):
        """Test that the IS2RE loader can be initialized."""
        with hydra.initialize(
            version_base="1.3",
            config_path=self.relative_config_dir,
            job_name="test_oc20_is2re"
        ):
            cfg = hydra.compose(
                config_name="run.yaml",
                overrides=["dataset=graph/OC20_IS2RE"],
                return_hydra_config=True,
            )
            loader = hydra.utils.instantiate(cfg.dataset.loader)
            assert isinstance(loader, IS2REDatasetLoader)
            assert loader.parameters.data_name == "OC20_IS2RE"
            assert loader.parameters.task == "is2re"

    def test_dataset_loading(self):
        """Test that the IS2RE dataset loads correctly."""
        with hydra.initialize(
            version_base="1.3",
            config_path=self.relative_config_dir,
            job_name="test_oc20_is2re_load"
        ):
            cfg = hydra.compose(
                config_name="run.yaml",
                overrides=["dataset=graph/OC20_IS2RE"],
                return_hydra_config=True,
            )
            loader = hydra.utils.instantiate(cfg.dataset.loader)
            dataset, data_dir = loader.load()
            
            # Check dataset type
            assert isinstance(dataset, IS2REDataset)
            
            # Check dataset has required attributes
            assert hasattr(dataset, 'split_idx')
            assert 'train' in dataset.split_idx
            assert 'valid' in dataset.split_idx
            assert 'test' in dataset.split_idx
            
            # Check splits are not empty (when max_samples is set)
            assert len(dataset.split_idx['train']) > 0
            assert len(dataset.split_idx['valid']) > 0
            assert len(dataset.split_idx['test']) > 0
            
            # Check dataset length
            assert len(dataset) > 0

    def test_dataset_item_access(self):
        """Test accessing individual items from the dataset."""
        with hydra.initialize(
            version_base="1.3",
            config_path=self.relative_config_dir,
            job_name="test_oc20_is2re_item"
        ):
            cfg = hydra.compose(
                config_name="run.yaml",
                overrides=["dataset=graph/OC20_IS2RE"],
                return_hydra_config=True,
            )
            loader = hydra.utils.instantiate(cfg.dataset.loader)
            dataset, _ = loader.load()
            
            # Get first item
            data = dataset[0]
            
            # Check data has required PyG attributes
            assert hasattr(data, 'x')
            assert hasattr(data, 'edge_index')
            assert hasattr(data, 'y')
            
            # Check data types
            assert isinstance(data.x, torch.Tensor)
            assert isinstance(data.edge_index, torch.Tensor)
            assert isinstance(data.y, torch.Tensor)
            
            # Check shapes
            assert data.x.dim() >= 1
            assert data.edge_index.dim() == 2
            assert data.edge_index.size(0) == 2  # [2, num_edges]

    def test_split_indices_validity(self):
        """Test that split indices are valid and non-overlapping."""
        with hydra.initialize(
            version_base="1.3",
            config_path=self.relative_config_dir,
            job_name="test_oc20_is2re_splits"
        ):
            cfg = hydra.compose(
                config_name="run.yaml",
                overrides=["dataset=graph/OC20_IS2RE"],
                return_hydra_config=True,
            )
            loader = hydra.utils.instantiate(cfg.dataset.loader)
            dataset, _ = loader.load()
            
            train_idx = dataset.split_idx['train'].numpy()
            val_idx = dataset.split_idx['valid'].numpy()
            test_idx = dataset.split_idx['test'].numpy()
            
            # Check no overlap between splits
            assert len(set(train_idx) & set(val_idx)) == 0
            assert len(set(train_idx) & set(test_idx)) == 0
            # val and test might overlap if test reuses val when test is not available
            
            # Check all indices are within dataset bounds
            all_indices = list(train_idx) + list(val_idx)
            assert all(0 <= idx < len(dataset) for idx in all_indices)


class TestOC22IS2REDatasetLoader:
    """Test suite for OC22 IS2RE dataset loader."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment."""
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        self.relative_config_dir = "../../../configs"

    def test_loader_initialization(self):
        """Test that the OC22 IS2RE loader can be initialized."""
        with hydra.initialize(
            version_base="1.3",
            config_path=self.relative_config_dir,
            job_name="test_oc22_is2re"
        ):
            cfg = hydra.compose(
                config_name="run.yaml",
                overrides=["dataset=graph/OC22_IS2RE"],
                return_hydra_config=True,
            )
            loader = hydra.utils.instantiate(cfg.dataset.loader)
            assert isinstance(loader, OC22IS2REDatasetLoader)
            assert loader.parameters.data_name == "OC22_IS2RE"
            assert loader.parameters.task == "oc22_is2re"

    def test_dataset_loading(self):
        """Test that the OC22 IS2RE dataset loads correctly."""
        with hydra.initialize(
            version_base="1.3",
            config_path=self.relative_config_dir,
            job_name="test_oc22_is2re_load"
        ):
            cfg = hydra.compose(
                config_name="run.yaml",
                overrides=["dataset=graph/OC22_IS2RE"],
                return_hydra_config=True,
            )
            loader = hydra.utils.instantiate(cfg.dataset.loader)
            dataset, data_dir = loader.load()
            
            # Check dataset type
            assert isinstance(dataset, OC22IS2REDataset)
            
            # Check dataset has required attributes
            assert hasattr(dataset, 'split_idx')
            assert 'train' in dataset.split_idx
            assert 'valid' in dataset.split_idx
            assert 'test' in dataset.split_idx
            
            # Check splits are not empty (when max_samples is set)
            assert len(dataset.split_idx['train']) > 0
            assert len(dataset.split_idx['valid']) > 0
            assert len(dataset.split_idx['test']) > 0
            
            # Check dataset length
            assert len(dataset) > 0

    def test_dataset_item_access(self):
        """Test accessing individual items from the OC22 dataset."""
        with hydra.initialize(
            version_base="1.3",
            config_path=self.relative_config_dir,
            job_name="test_oc22_is2re_item"
        ):
            cfg = hydra.compose(
                config_name="run.yaml",
                overrides=["dataset=graph/OC22_IS2RE"],
                return_hydra_config=True,
            )
            loader = hydra.utils.instantiate(cfg.dataset.loader)
            dataset, _ = loader.load()
            
            # Get first item
            data = dataset[0]
            
            # Check data has required PyG attributes
            assert hasattr(data, 'x')
            assert hasattr(data, 'edge_index')
            assert hasattr(data, 'y')
            
            # Check data types
            assert isinstance(data.x, torch.Tensor)
            assert isinstance(data.edge_index, torch.Tensor)
            assert isinstance(data.y, torch.Tensor)
            
            # Check shapes
            assert data.x.dim() >= 1
            assert data.edge_index.dim() == 2
            assert data.edge_index.size(0) == 2

    def test_split_indices_validity(self):
        """Test that split indices are valid and non-overlapping."""
        with hydra.initialize(
            version_base="1.3",
            config_path=self.relative_config_dir,
            job_name="test_oc22_is2re_splits"
        ):
            cfg = hydra.compose(
                config_name="run.yaml",
                overrides=["dataset=graph/OC22_IS2RE"],
                return_hydra_config=True,
            )
            loader = hydra.utils.instantiate(cfg.dataset.loader)
            dataset, _ = loader.load()
            
            train_idx = dataset.split_idx['train'].numpy()
            val_idx = dataset.split_idx['valid'].numpy()
            test_idx = dataset.split_idx['test'].numpy()
            
            # Check no overlap between train and val
            assert len(set(train_idx) & set(val_idx)) == 0
            
            # Check all indices are within dataset bounds
            all_indices = list(train_idx) + list(val_idx)
            assert all(0 <= idx < len(dataset) for idx in all_indices)


class TestOC20S2EFDatasetLoader:
    """Test suite for OC20 S2EF dataset loader."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment."""
        hydra.core.global_hydra.GlobalHydra.instance().clear()
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
                overrides=["dataset=graph/OC20_S2EF_train_200K"],
                return_hydra_config=True,
            )
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
                overrides=["dataset=graph/OC20_S2EF_train_200K"],
                return_hydra_config=True,
            )
            loader = hydra.utils.instantiate(cfg.dataset.loader)
            dataset, data_dir = loader.load()
            
            # Check dataset type
            assert isinstance(dataset, OC20Dataset)
            
            # Check dataset has required attributes
            assert hasattr(dataset, 'split_idx')
            assert 'train' in dataset.split_idx
            assert 'valid' in dataset.split_idx
            assert 'test' in dataset.split_idx
            
            # Check splits are not empty
            assert len(dataset.split_idx['train']) > 0
            assert len(dataset.split_idx['valid']) > 0
            assert len(dataset.split_idx['test']) > 0
            
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
                overrides=["dataset=graph/OC20_S2EF_train_200K"],
                return_hydra_config=True,
            )
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
                overrides=[
                    "dataset=graph/OC20_S2EF_val_id",
                ],
                return_hydra_config=True,
            )
            loader = hydra.utils.instantiate(cfg.dataset.loader)
            assert loader.parameters.val_splits == ["val_id"]

    def test_split_indices_validity_s2ef(self):
        """Test that S2EF split indices are valid."""
        with hydra.initialize(
            version_base="1.3",
            config_path=self.relative_config_dir,
            job_name="test_oc20_s2ef_splits"
        ):
            cfg = hydra.compose(
                config_name="run.yaml",
                overrides=["dataset=graph/OC20_S2EF_train_200K"],
                return_hydra_config=True,
            )
            loader = hydra.utils.instantiate(cfg.dataset.loader)
            dataset, _ = loader.load()
            
            train_idx = dataset.split_idx['train'].numpy()
            val_idx = dataset.split_idx['valid'].numpy()
            test_idx = dataset.split_idx['test'].numpy()
            
            # Check no overlap between train and val
            assert len(set(train_idx) & set(val_idx)) == 0
            
            # Check all indices are within dataset bounds
            all_indices = list(train_idx) + list(val_idx)
            assert all(0 <= idx < len(dataset) for idx in all_indices)

    def test_different_train_splits(self):
        """Test that different training split sizes can be loaded."""
        train_splits = ["200K", "2M", "20M", "all"]
        
        for split in train_splits[:2]:  # Test only 200K and 2M to keep tests fast
            with hydra.initialize(
                version_base="1.3",
                config_path=self.relative_config_dir,
                job_name=f"test_oc20_s2ef_{split}"
            ):
                cfg = hydra.compose(
                    config_name="run.yaml",
                    overrides=[f"dataset=graph/OC20_S2EF_train_{split}"],
                    return_hydra_config=True,
                )
                loader = hydra.utils.instantiate(cfg.dataset.loader)
                assert loader.parameters.train_split == split
                
                # Load and verify dataset
                dataset, _ = loader.load()
                assert len(dataset) > 0
                assert len(dataset.split_idx['train']) > 0


class TestOC20DatasetIntegration:
    """Integration tests for OC20 datasets with preprocessing pipeline."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment."""
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        self.relative_config_dir = "../../../configs"

    def test_is2re_with_preprocessor(self):
        """Test IS2RE dataset with PreProcessor."""
        with hydra.initialize(
            version_base="1.3",
            config_path=self.relative_config_dir,
            job_name="test_is2re_preprocessor"
        ):
            cfg = hydra.compose(
                config_name="run.yaml",
                overrides=[
                    "dataset=graph/OC20_IS2RE",
                    "model=graph/gcn",
                ],
                return_hydra_config=True,
            )
            
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
            
            # Verify splits exist and are not empty
            assert dataset_train is not None
            assert dataset_val is not None
            assert dataset_test is not None
            assert len(dataset_train) > 0
            assert len(dataset_val) > 0
            assert len(dataset_test) > 0

    def test_oc22_is2re_with_preprocessor(self):
        """Test OC22 IS2RE dataset with PreProcessor."""
        with hydra.initialize(
            version_base="1.3",
            config_path=self.relative_config_dir,
            job_name="test_oc22_preprocessor"
        ):
            cfg = hydra.compose(
                config_name="run.yaml",
                overrides=[
                    "dataset=graph/OC22_IS2RE",
                    "model=graph/gcn",
                ],
                return_hydra_config=True,
            )
            
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
            
            # Verify splits exist and are not empty
            assert dataset_train is not None
            assert dataset_val is not None
            assert dataset_test is not None
            assert len(dataset_train) > 0
            assert len(dataset_val) > 0
            assert len(dataset_test) > 0

    def test_s2ef_with_preprocessor(self):
        """Test S2EF dataset with PreProcessor."""
        with hydra.initialize(
            version_base="1.3",
            config_path=self.relative_config_dir,
            job_name="test_s2ef_preprocessor"
        ):
            cfg = hydra.compose(
                config_name="run.yaml",
                overrides=[
                    "dataset=graph/OC20_S2EF_train_200K",
                    "model=graph/gcn",
                ],
                return_hydra_config=True,
            )
            
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
            
            # Verify splits exist and are not empty
            assert dataset_train is not None
            assert dataset_val is not None
            assert dataset_test is not None
            assert len(dataset_train) > 0
            assert len(dataset_val) > 0
            assert len(dataset_test) > 0
