"""Tests for DAG-based incremental caching functionality."""

import json
import tempfile
import time
from pathlib import Path

import torch
from torch_geometric.data import Data

from topobench.data.preprocessor.ondisk_inductive import (
    OnDiskInductivePreprocessor,
)


class SyntheticCustomDataset(torch.utils.data.Dataset):
    """Synthetic custom Dataset used in tests."""

    def __init__(self, num_samples: int):
        self.num_samples = num_samples
    
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Use deterministic random generation based on index
        torch.manual_seed(42 + idx)
        return Data(
            x=torch.randn(5 + idx % 3, 8),
            edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long),
            y=torch.tensor([idx % 3]),
        )


class TestDAGCaching:
    """Test suite for DAG-based incremental caching functionality."""
    
    def test_dag_transform_chain_resolution(self):
        """Test _resolve_transform_chain() builds correct chain metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            source = SyntheticCustomDataset(num_samples=20)
            
            # Create config with lifting transform
            from omegaconf import OmegaConf
            config = OmegaConf.create({
                "clique_lifting": {
                    "transform_type": "lifting",
                    "transform_name": "SimplicialCliqueLifting",
                    "complex_dim": 2,
                },
            })
            
            # Initialize preprocessor (will call _resolve_transform_chain)
            dataset = OnDiskInductivePreprocessor(
                dataset=source,
                data_dir=data_dir,
                transforms_config=config,
                num_workers=1,
                storage_backend="files",
            )
            
            # Verify transform chain was created
            assert hasattr(dataset, "transform_chain")
            assert dataset.transform_chain is not None
            assert len(dataset.transform_chain) > 0
            
            # Verify chain structure
            for entry in dataset.transform_chain:
                assert "transform_id" in entry
                assert "transform_class" in entry
                assert "hash" in entry
                assert "output_dir" in entry
                assert "cached" in entry
            
            # Verify processed_dir points to last transform
            expected_dir = Path(dataset.transform_chain[-1]["output_dir"])
            assert dataset.processed_dir == expected_dir
            
            # Verify transform output exists
            assert expected_dir.exists()
            assert (expected_dir / "dataset_metadata.json").exists()
    
    def test_dag_cache_reuse_when_adding_transforms(self):
        """Test incremental caching when adding transforms to existing pipeline."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            source = SyntheticCustomDataset(num_samples=50)
            
            from omegaconf import OmegaConf
            
            # Scenario 1: Process with lifting only
            config1 = OmegaConf.create({
                "clique_lifting": {
                    "transform_type": "lifting",
                    "transform_name": "SimplicialCliqueLifting",
                    "complex_dim": 2,
                },
            })
            
            start1 = time.time()
            dataset1 = OnDiskInductivePreprocessor(
                dataset=source,
                data_dir=data_dir,
                transforms_config=config1,
                num_workers=1,
                storage_backend="files",
            )
            time1 = time.time() - start1
            
            # Get first transform output directory
            first_transform_dir = Path(dataset1.transform_chain[0]["output_dir"])
            first_transform_mtime = first_transform_dir.stat().st_mtime
            
            # Scenario 2: Add normalization (should reuse lifting!)
            config2 = OmegaConf.create({
                "clique_lifting": {
                    "transform_type": "lifting",
                    "transform_name": "SimplicialCliqueLifting",
                    "complex_dim": 2,
                },
                "degree_normalization": {
                    "transform_type": "feature",
                    "transform_name": "ProjectionSum",
                },
            })
            
            start2 = time.time()
            dataset2 = OnDiskInductivePreprocessor(
                dataset=source,
                data_dir=data_dir,
                transforms_config=config2,
                num_workers=1,
                storage_backend="files",
            )
            time2 = time.time() - start2
            
            # Verify lifting was reused (not reprocessed)
            # The first transform directory should not have been modified
            first_transform_dir2 = Path(dataset2.transform_chain[0]["output_dir"])
            assert first_transform_dir == first_transform_dir2
            first_transform_mtime2 = first_transform_dir2.stat().st_mtime
            assert first_transform_mtime == first_transform_mtime2, \
                "First transform should be reused (not reprocessed)"
            
            # Verify second transform was processed
            assert len(dataset2.transform_chain) == 2
            
            # Verify both transforms exist
            assert first_transform_dir2.exists()
            second_transform_dir = Path(dataset2.transform_chain[1]["output_dir"])
            assert second_transform_dir.exists()
            
            # Time should be less than full reprocessing
            # (though with small dataset overhead might dominate)
            print(f"\nDAG cache reuse: First={time1:.2f}s, Add transform={time2:.2f}s")
    
    def test_dag_full_recompute_on_parameter_change(self):
        """Test that changing transform parameters triggers recomputation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            source = SyntheticCustomDataset(num_samples=30)
            
            from omegaconf import OmegaConf
            
            # Config 1: complex_dim=2
            config1 = OmegaConf.create({
                "clique_lifting": {
                    "transform_type": "lifting",
                    "transform_name": "SimplicialCliqueLifting",
                    "complex_dim": 2,
                },
            })
            
            dataset1 = OnDiskInductivePreprocessor(
                dataset=source,
                data_dir=data_dir,
                transforms_config=config1,
                num_workers=1,
                storage_backend="files",
            )
            
            hash1 = dataset1.transform_chain[0]["hash"]
            
            # Config 2: complex_dim=3 (CHANGED!)
            config2 = OmegaConf.create({
                "clique_lifting": {
                    "transform_type": "lifting",
                    "transform_name": "SimplicialCliqueLifting",
                    "complex_dim": 3,
                },
            })
            
            dataset2 = OnDiskInductivePreprocessor(
                dataset=source,
                data_dir=data_dir,
                transforms_config=config2,
                num_workers=1,
                storage_backend="files",
            )
            
            hash2 = dataset2.transform_chain[0]["hash"]
            
            # Verify different hash (different parameters)
            assert hash1 != hash2, "Different parameters should produce different hash"
            
            # Verify different output directories
            dir1 = Path(dataset1.transform_chain[0]["output_dir"])
            dir2 = Path(dataset2.transform_chain[0]["output_dir"])
            assert dir1 != dir2, "Different hash should use different directory"
    
    def test_dag_check_transform_cached(self):
        """Test _check_transform_cached() validation logic."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            source = SyntheticCustomDataset(num_samples=10)
            
            from omegaconf import OmegaConf
            config = OmegaConf.create({
                "clique_lifting": {
                    "transform_type": "lifting",
                    "transform_name": "SimplicialCliqueLifting",
                    "complex_dim": 2,
                },
            })
            
            dataset = OnDiskInductivePreprocessor(
                dataset=source,
                data_dir=data_dir,
                transforms_config=config,
                num_workers=1,
                storage_backend="files",
            )
            
            # Test with valid cached directory
            valid_dir = Path(dataset.transform_chain[0]["output_dir"])
            assert dataset._check_transform_cached(valid_dir) is True
            
            # Test with non-existent directory
            fake_dir = data_dir / "nonexistent"
            assert dataset._check_transform_cached(fake_dir) is False
            
            # Test with directory but missing metadata
            missing_meta_dir = data_dir / "missing_meta"
            missing_meta_dir.mkdir(parents=True, exist_ok=True)
            assert dataset._check_transform_cached(missing_meta_dir) is False
    
    def test_dag_create_cached_dataset(self):
        """Test _create_cached_dataset() loads from cache correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            source = SyntheticCustomDataset(num_samples=20)
            
            from omegaconf import OmegaConf
            config = OmegaConf.create({
                "clique_lifting": {
                    "transform_type": "lifting",
                    "transform_name": "SimplicialCliqueLifting",
                    "complex_dim": 2,
                },
            })
            
            # Create original dataset
            dataset = OnDiskInductivePreprocessor(
                dataset=source,
                data_dir=data_dir,
                transforms_config=config,
                num_workers=1,
                storage_backend="files",
            )
            
            # Create cached dataset from first transform output
            cached_dir = Path(dataset.transform_chain[0]["output_dir"])
            cached_dataset = dataset._create_cached_dataset(cached_dir)
            
            # Verify it works like a dataset
            assert len(cached_dataset) == 20
            sample = cached_dataset[0]
            assert isinstance(sample, Data)
            assert hasattr(sample, "x")
    
    def test_dag_create_partial_transform(self):
        """Test _create_partial_transform() creates correct transform composition."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            source = SyntheticCustomDataset(num_samples=10)
            
            from omegaconf import OmegaConf
            config = OmegaConf.create({
                "clique_lifting": {
                    "transform_type": "lifting",
                    "transform_name": "SimplicialCliqueLifting",
                    "complex_dim": 2,
                },
                "degree_normalization": {
                    "transform_type": "feature",
                    "transform_name": "ProjectionSum",
                },
            })
            
            dataset = OnDiskInductivePreprocessor(
                dataset=source,
                data_dir=data_dir,
                transforms_config=config,
                num_workers=1,
                storage_backend="files",
            )
            
            # Get partial transform starting from index 1
            if len(dataset.transform_chain) >= 2:
                partial_transform = dataset._create_partial_transform(1)
                assert partial_transform is not None
                
                # Should be a Compose with remaining transforms
                if hasattr(partial_transform, "transforms"):
                    assert len(partial_transform.transforms) >= 1
    
    def test_dag_should_process_with_chain(self):
        """Test _should_process() DAG-aware checking."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            source = SyntheticCustomDataset(num_samples=15)
            
            from omegaconf import OmegaConf
            config = OmegaConf.create({
                "clique_lifting": {
                    "transform_type": "lifting",
                    "transform_name": "SimplicialCliqueLifting",
                    "complex_dim": 2,
                },
            })
            
            # First init - should process
            dataset1 = OnDiskInductivePreprocessor(
                dataset=source,
                data_dir=data_dir,
                transforms_config=config,
                num_workers=1,
                storage_backend="files",
            )
            
            # Manually test _should_process() on second init
            # Create new preprocessor instance but don't process yet
            dataset2 = OnDiskInductivePreprocessor.__new__(OnDiskInductivePreprocessor)
            dataset2.dataset = source
            dataset2.data_dir = data_dir
            dataset2.transforms_config = config
            dataset2.force_reload = False
            dataset2.num_workers = 1
            dataset2.storage_backend = "files"
            dataset2.compression = "lz4"
            dataset2.transform_tier = "all_heavy"
            dataset2.tier_override = None
            dataset2.cache_size = 100
            dataset2._cache = {}
            dataset2._cache_hits = 0
            dataset2._cache_misses = 0
            dataset2._storage = None
            
            # Initialize transform pipeline
            dataset2.pre_transform = dataset2._instantiate_pre_transform(config)
            dataset2._create_transform_pipeline()
            dataset2._set_processed_data_dir(config)
            dataset2.metadata_path = dataset2.processed_dir / "dataset_metadata.json"
            
            # Now _should_process should return False (cache exists)
            should_process = dataset2._should_process()
            assert should_process is False, "Should use cache when all transforms cached"
    
    def test_dag_metadata_includes_chain_info(self):
        """Test that saved metadata includes transform chain information."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            source = SyntheticCustomDataset(num_samples=10)
            
            from omegaconf import OmegaConf
            config = OmegaConf.create({
                "clique_lifting": {
                    "transform_type": "lifting",
                    "transform_name": "SimplicialCliqueLifting",
                    "complex_dim": 2,
                },
            })
            
            dataset = OnDiskInductivePreprocessor(
                dataset=source,
                data_dir=data_dir,
                transforms_config=config,
                num_workers=1,
                storage_backend="files",
            )
            
            # Load metadata and verify chain info
            with open(dataset.metadata_path) as f:
                metadata = json.load(f)
            
            assert "transform_chain" in metadata
            assert isinstance(metadata["transform_chain"], list)
            assert len(metadata["transform_chain"]) > 0
            
            # Verify chain entry structure
            chain_entry = metadata["transform_chain"][0]
            assert "transform_id" in chain_entry
            assert "transform_class" in chain_entry
            assert "hash" in chain_entry
            assert "output_dir" in chain_entry
    
    def test_dag_incremental_processing_with_mmap(self):
        """Test DAG caching works with mmap storage backend."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            source = SyntheticCustomDataset(num_samples=40)
            
            from omegaconf import OmegaConf
            
            # First: lifting only
            config1 = OmegaConf.create({
                "clique_lifting": {
                    "transform_type": "lifting",
                    "transform_name": "SimplicialCliqueLifting",
                    "complex_dim": 2,
                },
            })
            
            dataset1 = OnDiskInductivePreprocessor(
                dataset=source,
                data_dir=data_dir,
                transforms_config=config1,
                num_workers=1,
                storage_backend="mmap",
                compression="lz4",
            )
            
            # Verify mmap files created
            mmap_path1 = Path(dataset1.transform_chain[0]["output_dir"]) / "samples.mmap"
            assert mmap_path1.exists()
            mtime1 = mmap_path1.stat().st_mtime
            
            # Second: add normalization
            config2 = OmegaConf.create({
                "clique_lifting": {
                    "transform_type": "lifting",
                    "transform_name": "SimplicialCliqueLifting",
                    "complex_dim": 2,
                },
                "degree_normalization": {
                    "transform_type": "feature",
                    "transform_name": "ProjectionSum",
                },
            })
            
            dataset2 = OnDiskInductivePreprocessor(
                dataset=source,
                data_dir=data_dir,
                transforms_config=config2,
                num_workers=1,
                storage_backend="mmap",
                compression="lz4",
            )
            
            # Verify lifting mmap not reprocessed
            mmap_path1_check = Path(dataset2.transform_chain[0]["output_dir"]) / "samples.mmap"
            assert mmap_path1 == mmap_path1_check
            mtime1_check = mmap_path1_check.stat().st_mtime
            assert mtime1 == mtime1_check, "Lifting should be reused from cache"
            
            # Verify second transform mmap created
            assert len(dataset2.transform_chain) == 2
            mmap_path2 = Path(dataset2.transform_chain[1]["output_dir"]) / "samples.mmap"
            assert mmap_path2.exists()
            
            # Verify data accessible
            sample = dataset2[0]
            assert isinstance(sample, Data)
    
    def test_dag_handles_duplicate_transforms_correctly(self):
        """Test that duplicate transforms (same class/params) are cached separately."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            source = SyntheticCustomDataset(num_samples=30)
            
            from omegaconf import OmegaConf
            
            # Scenario 1: Build base transform
            config1 = OmegaConf.create({
                "clique_lifting": {
                    "transform_type": "lifting",
                    "transform_name": "SimplicialCliqueLifting",
                    "complex_dim": 2,
                },
            })
            
            dataset1 = OnDiskInductivePreprocessor(
                dataset=source,
                data_dir=data_dir,
                transforms_config=config1,
                num_workers=1,
                storage_backend="files",
            )
            
            # Scenario 2: Add first ProjectionSum
            config2 = OmegaConf.create({
                "clique_lifting": {
                    "transform_type": "lifting",
                    "transform_name": "SimplicialCliqueLifting",
                    "complex_dim": 2,
                },
                "proj1": {
                    "transform_type": "feature",
                    "transform_name": "ProjectionSum",
                },
            })
            
            dataset2 = OnDiskInductivePreprocessor(
                dataset=source,
                data_dir=data_dir,
                transforms_config=config2,
                num_workers=1,
                storage_backend="files",
            )
            
            # Verify we have 2 transforms in chain
            assert len(dataset2.transform_chain) == 2
            
            # Get directories
            dir1 = Path(dataset2.transform_chain[0]["output_dir"])
            dir2 = Path(dataset2.transform_chain[1]["output_dir"])
            
            # Verify both exist
            assert dir1.exists(), "First transform should be cached"
            assert dir2.exists(), "Second transform should be cached"
            
            # Verify they're different directories
            assert dir1 != dir2, "Different transforms should have different cache directories"
            
            # Scenario 3: Add SECOND ProjectionSum (identical to first!)
            config3 = OmegaConf.create({
                "clique_lifting": {
                    "transform_type": "lifting",
                    "transform_name": "SimplicialCliqueLifting",
                    "complex_dim": 2,
                },
                "proj1": {
                    "transform_type": "feature",
                    "transform_name": "ProjectionSum",
                },
                "proj2": {  # Same transform as proj1!
                    "transform_type": "feature",
                    "transform_name": "ProjectionSum",
                },
            })
            
            start_time = time.time()
            dataset3 = OnDiskInductivePreprocessor(
                dataset=source,
                data_dir=data_dir,
                transforms_config=config3,
                num_workers=1,
                storage_backend="files",
            )
            time_taken = time.time() - start_time
            
            # Verify we have 3 transforms in chain
            assert len(dataset3.transform_chain) == 3, \
                "Should have 3 transforms: clique_lifting, proj1, proj2"
            
            # Get all directories
            dir1_check = Path(dataset3.transform_chain[0]["output_dir"])
            dir2_check = Path(dataset3.transform_chain[1]["output_dir"])
            dir3 = Path(dataset3.transform_chain[2]["output_dir"])
            
            # Verify first two transforms were reused (same directories)
            assert dir1_check == dir1, "First transform should be reused from cache"
            assert dir2_check == dir2, "Second transform should be reused from cache"
            
            # Verify third transform exists and is DIFFERENT from second
            assert dir3.exists(), "Third transform should have been processed"
            assert dir3 != dir2, \
                "CRITICAL: Duplicate ProjectionSum transforms must have different cache directories!"
            assert dir3 != dir1, "Third transform should be different from first"
            
            # Verify time is reasonable (should only process third transform, not all)
            # If bug exists, it would think proj2 is already cached and time would be ~0
            assert time_taken > 0.05, \
                f"Processing third transform should take measurable time (got {time_taken:.2f}s)"
            
            # Verify all samples accessible and correct
            for i in range(min(5, len(source))):
                sample = dataset3[i]
                assert isinstance(sample, Data)
                assert hasattr(sample, 'x'), "Sample should have features"
            
            print(f"\n✅ Duplicate transform test passed!")
            print(f"   - Transform 1: {dir1.name}")
            print(f"   - Transform 2: {dir2.name}")  
            print(f"   - Transform 3: {dir3.name} (duplicate of 2, but correctly separate)")
            print(f"   - Processing time: {time_taken:.2f}s")
