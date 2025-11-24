"""Tests for ATLAS Top Tagging Dataset - CORRECTED VERSION"""

import pytest
from pathlib import Path


class TestATLASDatasetImport:
    """Test dataset can be imported."""

    def test_can_import_dataset_class(self):
        """Test ATLASTopTaggingDataset can be imported."""
        from topobench.data.datasets.atlas_top_tagging_dataset import ATLASTopTaggingDataset
        assert ATLASTopTaggingDataset is not None

    def test_class_inherits_from_in_memory_dataset(self):
        """Test class inherits from InMemoryDataset."""
        from topobench.data.datasets.atlas_top_tagging_dataset import ATLASTopTaggingDataset
        from torch_geometric.data import InMemoryDataset
        assert issubclass(ATLASTopTaggingDataset, InMemoryDataset)


class TestATLASDatasetClassAttributes:
    """Test dataset class attributes."""

    def test_has_urls_attribute(self):
        """Test class has URLS attribute."""
        from topobench.data.datasets.atlas_top_tagging_dataset import ATLASTopTaggingDataset
        assert hasattr(ATLASTopTaggingDataset, 'URLS')
        assert isinstance(ATLASTopTaggingDataset.URLS, dict)

    def test_has_constituent_branches(self):
        """Test class has CONSTITUENT_BRANCHES."""
        from topobench.data.datasets.atlas_top_tagging_dataset import ATLASTopTaggingDataset
        assert hasattr(ATLASTopTaggingDataset, 'CONSTITUENT_BRANCHES')
        assert len(ATLASTopTaggingDataset.CONSTITUENT_BRANCHES) == 4

    def test_has_high_level_branches(self):
        """Test class has HIGH_LEVEL_BRANCHES."""
        from topobench.data.datasets.atlas_top_tagging_dataset import ATLASTopTaggingDataset
        assert hasattr(ATLASTopTaggingDataset, 'HIGH_LEVEL_BRANCHES')
        assert len(ATLASTopTaggingDataset.HIGH_LEVEL_BRANCHES) == 15

    def test_has_jet_branches(self):
        """Test class has JET_BRANCHES."""
        from topobench.data.datasets.atlas_top_tagging_dataset import ATLASTopTaggingDataset
        assert hasattr(ATLASTopTaggingDataset, 'JET_BRANCHES')
        assert len(ATLASTopTaggingDataset.JET_BRANCHES) == 4

    def test_constituent_branches_content(self):
        """Test CONSTITUENT_BRANCHES has expected content."""
        from topobench.data.datasets.atlas_top_tagging_dataset import ATLASTopTaggingDataset
        branches = ATLASTopTaggingDataset.CONSTITUENT_BRANCHES
        # Check for actual ATLAS branch names
        assert 'fjet_clus_pt' in branches
        assert 'fjet_clus_eta' in branches
        assert 'fjet_clus_phi' in branches
        assert 'fjet_clus_E' in branches

    def test_urls_are_strings(self):
        """Test URLS values are strings."""
        from topobench.data.datasets.atlas_top_tagging_dataset import ATLASTopTaggingDataset
        for url in ATLASTopTaggingDataset.URLS.values():
            assert isinstance(url, str)


class TestATLASDatasetMethods:
    """Test dataset methods exist."""

    def test_has_download_method(self):
        """Test download method exists."""
        from topobench.data.datasets.atlas_top_tagging_dataset import ATLASTopTaggingDataset
        assert hasattr(ATLASTopTaggingDataset, 'download')

    def test_has_process_method(self):
        """Test process method exists."""
        from topobench.data.datasets.atlas_top_tagging_dataset import ATLASTopTaggingDataset
        assert hasattr(ATLASTopTaggingDataset, 'process')

    def test_has_preprocess_method(self):
        """Test _preprocess method exists."""
        from topobench.data.datasets.atlas_top_tagging_dataset import ATLASTopTaggingDataset
        assert hasattr(ATLASTopTaggingDataset, '_preprocess')

    def test_has_load_h5_flexible_method(self):
        """Test _load_h5_file_flexible method exists."""
        from topobench.data.datasets.atlas_top_tagging_dataset import ATLASTopTaggingDataset
        assert hasattr(ATLASTopTaggingDataset, '_load_h5_file_flexible')

    def test_has_expected_filenames_method(self):
        """Test _expected_filenames method exists."""
        from topobench.data.datasets.atlas_top_tagging_dataset import ATLASTopTaggingDataset
        assert hasattr(ATLASTopTaggingDataset, '_expected_filenames')

    def test_has_total_files_for_split_method(self):
        """Test _total_files_for_split method exists."""
        from topobench.data.datasets.atlas_top_tagging_dataset import ATLASTopTaggingDataset
        assert hasattr(ATLASTopTaggingDataset, '_total_files_for_split')


class TestATLASDatasetProperties:
    """Test dataset properties."""

    def test_has_num_classes_property(self):
        """Test num_classes property exists."""
        from topobench.data.datasets.atlas_top_tagging_dataset import ATLASTopTaggingDataset
        assert hasattr(ATLASTopTaggingDataset, 'num_classes')

    def test_has_num_features_property(self):
        """Test num_features property exists."""
        from topobench.data.datasets.atlas_top_tagging_dataset import ATLASTopTaggingDataset
        assert hasattr(ATLASTopTaggingDataset, 'num_features')

    def test_has_num_high_level_features_property(self):
        """Test num_high_level_features property exists."""
        from topobench.data.datasets.atlas_top_tagging_dataset import ATLASTopTaggingDataset
        assert hasattr(ATLASTopTaggingDataset, 'num_high_level_features')

    def test_has_raw_file_names_property(self):
        """Test raw_file_names property exists."""
        from topobench.data.datasets.atlas_top_tagging_dataset import ATLASTopTaggingDataset
        assert hasattr(ATLASTopTaggingDataset, 'raw_file_names')

    def test_has_processed_file_names_property(self):
        """Test processed_file_names property exists."""
        from topobench.data.datasets.atlas_top_tagging_dataset import ATLASTopTaggingDataset
        assert hasattr(ATLASTopTaggingDataset, 'processed_file_names')

    def test_has_pre_processed_path_property(self):
        """Test pre_processed_path property exists."""
        from topobench.data.datasets.atlas_top_tagging_dataset import ATLASTopTaggingDataset
        assert hasattr(ATLASTopTaggingDataset, 'pre_processed_path')

    def test_has_stats_method(self):
        """Test stats method exists."""
        from topobench.data.datasets.atlas_top_tagging_dataset import ATLASTopTaggingDataset
        assert hasattr(ATLASTopTaggingDataset, 'stats')


class TestATLASDatasetInitialization:
    """Test dataset initialization and parameters."""

    def test_subset_parameter_validation_too_high(self):
        """Test subset validation rejects values > 1.0."""
        from topobench.data.datasets.atlas_top_tagging_dataset import ATLASTopTaggingDataset
        with pytest.raises(AssertionError):
            ATLASTopTaggingDataset(root='/tmp/nonexistent', split='train', subset=1.5)

    def test_subset_parameter_validation_too_low(self):
        """Test subset validation rejects values <= 0."""
        from topobench.data.datasets.atlas_top_tagging_dataset import ATLASTopTaggingDataset
        with pytest.raises(AssertionError):
            ATLASTopTaggingDataset(root='/tmp/nonexistent', split='train', subset=0.0)

    def test_split_parameter_validation(self):
        """Test split parameter validation."""
        from topobench.data.datasets.atlas_top_tagging_dataset import ATLASTopTaggingDataset
        with pytest.raises(AssertionError):
            ATLASTopTaggingDataset(root='/tmp/nonexistent', split='invalid', subset=0.001)

    def test_max_constituents_parameter_stored(self):
        """Test max_constituents parameter is stored."""
        from topobench.data.datasets.atlas_top_tagging_dataset import ATLASTopTaggingDataset
        dataset = ATLASTopTaggingDataset(root='/tmp/nonexistent', split='train', subset=0.001, max_constituents=50)
        assert dataset.max_constituents == 50

    def test_use_high_level_parameter_stored(self):
        """Test use_high_level parameter is stored."""
        from topobench.data.datasets.atlas_top_tagging_dataset import ATLASTopTaggingDataset
        dataset = ATLASTopTaggingDataset(root='/tmp/nonexistent', split='train', subset=0.001, use_high_level=False)
        assert dataset.use_high_level == False

    def test_verbose_parameter_stored(self):
        """Test verbose parameter is stored."""
        from topobench.data.datasets.atlas_top_tagging_dataset import ATLASTopTaggingDataset
        dataset = ATLASTopTaggingDataset(root='/tmp/nonexistent', split='train', subset=0.001, verbose=True)
        assert dataset.verbose == True


class TestATLASDatasetHelperMethods:
    """Test helper methods."""

    def test_total_files_for_split_train(self):
        """Test _total_files_for_split returns 930 for train."""
        from topobench.data.datasets.atlas_top_tagging_dataset import ATLASTopTaggingDataset
        # Create instance with train split to test the method
        dataset = ATLASTopTaggingDataset(root='/tmp/nonexistent', split='train', subset=0.001)
        assert dataset._total_files_for_split() == 930

    def test_total_files_for_split_test(self):
        """Test _total_files_for_split returns 100 for test."""
        from topobench.data.datasets.atlas_top_tagging_dataset import ATLASTopTaggingDataset
        # Create instance with test split to test the method
        dataset = ATLASTopTaggingDataset(root='/tmp/nonexistent', split='test', subset=0.001)
        assert dataset._total_files_for_split() == 100

    def test_expected_filenames_format(self):
        """Test _expected_filenames returns correct format."""
        from topobench.data.datasets.atlas_top_tagging_dataset import ATLASTopTaggingDataset
        dataset = ATLASTopTaggingDataset(root='/tmp/nonexistent', split='train', subset=0.01)
        filenames = dataset._expected_filenames()
        assert len(filenames) > 0
        assert all('.h5.gz' in f for f in filenames)

    def test_expected_filenames_respects_subset(self):
        """Test _expected_filenames respects subset parameter."""
        from topobench.data.datasets.atlas_top_tagging_dataset import ATLASTopTaggingDataset
        small_dataset = ATLASTopTaggingDataset(root='/tmp/nonexistent', split='train', subset=0.001)
        large_dataset = ATLASTopTaggingDataset(root='/tmp/nonexistent', split='train', subset=0.01)
        small = small_dataset._expected_filenames()
        large = large_dataset._expected_filenames()
        assert len(small) < len(large)

    def test_processed_file_names_format(self):
        """Test processed_file_names has correct format."""
        from topobench.data.datasets.atlas_top_tagging_dataset import ATLASTopTaggingDataset
        dataset = ATLASTopTaggingDataset(root='/tmp/nonexistent', split='train', subset=0.001)
        assert 'atlas_top_tagging' in dataset.processed_file_names[0]
        assert '.pt' in dataset.processed_file_names[0]


class TestATLASDatasetPropertiesValues:
    """Test property values."""

    def test_num_classes_equals_two(self):
        """Test num_classes returns 2."""
        from topobench.data.datasets.atlas_top_tagging_dataset import ATLASTopTaggingDataset
        dataset = ATLASTopTaggingDataset(root='/tmp/nonexistent', split='train', subset=0.001)
        assert dataset.num_classes == 2

    def test_num_features_equals_four(self):
        """Test num_features returns 4."""
        from topobench.data.datasets.atlas_top_tagging_dataset import ATLASTopTaggingDataset
        dataset = ATLASTopTaggingDataset(root='/tmp/nonexistent', split='train', subset=0.001)
        assert dataset.num_features == 4

    def test_num_high_level_features_with_flag_true(self):
        """Test num_high_level_features returns 15 when use_high_level=True."""
        from topobench.data.datasets.atlas_top_tagging_dataset import ATLASTopTaggingDataset
        dataset = ATLASTopTaggingDataset(root='/tmp/nonexistent', split='train', subset=0.001, use_high_level=True)
        assert dataset.num_high_level_features == 15

    def test_num_high_level_features_with_flag_false(self):
        """Test num_high_level_features returns 0 when use_high_level=False."""
        from topobench.data.datasets.atlas_top_tagging_dataset import ATLASTopTaggingDataset
        dataset = ATLASTopTaggingDataset(root='/tmp/nonexistent', split='train', subset=0.001, use_high_level=False)
        assert dataset.num_high_level_features == 0

    def test_raw_file_names_includes_split_directory(self):
        """Test raw_file_names includes split directory."""
        from topobench.data.datasets.atlas_top_tagging_dataset import ATLASTopTaggingDataset
        dataset = ATLASTopTaggingDataset(root='/tmp/nonexistent', split='train', subset=0.001)
        assert 'train' in str(dataset.raw_file_names)