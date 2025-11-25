"""Tests for ATLAS Top Tagging Dataset."""

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
            ATLASTopTaggingDataset(root='/tmp/nonexistent', split='invalid', subset=0.01)

    def test_max_constituents_parameter_stored(self):
        """Test max_constituents parameter is stored."""
        from topobench.data.datasets.atlas_top_tagging_dataset import ATLASTopTaggingDataset
        dataset = ATLASTopTaggingDataset(root='/tmp/nonexistent', split='train', subset=0.01, max_constituents=50)
        assert dataset.max_constituents == 50

    def test_use_high_level_parameter_stored(self):
        """Test use_high_level parameter is stored."""
        from topobench.data.datasets.atlas_top_tagging_dataset import ATLASTopTaggingDataset
        dataset = ATLASTopTaggingDataset(root='/tmp/nonexistent', split='train', subset=0.01, use_high_level=False)
        assert dataset.use_high_level == False

    def test_verbose_parameter_stored(self):
        """Test verbose parameter is stored."""
        from topobench.data.datasets.atlas_top_tagging_dataset import ATLASTopTaggingDataset
        dataset = ATLASTopTaggingDataset(root='/tmp/nonexistent', split='train', subset=0.01, verbose=True)
        assert dataset.verbose == True


class TestATLASDatasetHelperMethods:
    """Test helper methods."""

    def test_total_files_for_split_train(self):
        """Test _total_files_for_split returns 930 for train."""
        from topobench.data.datasets.atlas_top_tagging_dataset import ATLASTopTaggingDataset
        # Create instance with train split to test the method
        dataset = ATLASTopTaggingDataset(root='/tmp/nonexistent', split='train', subset=0.01)
        assert dataset._total_files_for_split() == 930

    def test_total_files_for_split_test(self):
        """Test _total_files_for_split returns 100 for test."""
        from topobench.data.datasets.atlas_top_tagging_dataset import ATLASTopTaggingDataset
        # Create instance with test split to test the method
        dataset = ATLASTopTaggingDataset(root='/tmp/nonexistent', split='test', subset=0.01)
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
        small_dataset = ATLASTopTaggingDataset(root='/tmp/nonexistent', split='train', subset=0.01)
        large_dataset = ATLASTopTaggingDataset(root='/tmp/nonexistent', split='train', subset=0.02)
        small = small_dataset._expected_filenames()
        large = large_dataset._expected_filenames()
        assert len(small) < len(large)

    def test_processed_file_names_format(self):
        """Test processed_file_names has correct format."""
        from topobench.data.datasets.atlas_top_tagging_dataset import ATLASTopTaggingDataset
        dataset = ATLASTopTaggingDataset(root='/tmp/nonexistent', split='train', subset=0.01)
        assert 'atlas_top_tagging' in dataset.processed_file_names[0]
        assert '.pt' in dataset.processed_file_names[0]


class TestATLASDatasetPropertiesValues:
    """Test property values."""

    def test_num_classes_equals_two(self):
        """Test num_classes returns 2."""
        from topobench.data.datasets.atlas_top_tagging_dataset import ATLASTopTaggingDataset
        dataset = ATLASTopTaggingDataset(root='/tmp/nonexistent', split='train', subset=0.01)
        assert dataset.num_classes == 2

    def test_num_features_equals_four(self):
        """Test num_features returns 4."""
        from topobench.data.datasets.atlas_top_tagging_dataset import ATLASTopTaggingDataset
        dataset = ATLASTopTaggingDataset(root='/tmp/nonexistent', split='train', subset=0.01)
        assert dataset.num_features == 4

    def test_num_high_level_features_with_flag_true(self):
        """Test num_high_level_features returns 15 when use_high_level=True."""
        from topobench.data.datasets.atlas_top_tagging_dataset import ATLASTopTaggingDataset
        dataset = ATLASTopTaggingDataset(root='/tmp/nonexistent', split='train', subset=0.01, use_high_level=True)
        assert dataset.num_high_level_features == 15

    def test_num_high_level_features_with_flag_false(self):
        """Test num_high_level_features returns 0 when use_high_level=False."""
        from topobench.data.datasets.atlas_top_tagging_dataset import ATLASTopTaggingDataset
        dataset = ATLASTopTaggingDataset(root='/tmp/nonexistent', split='train', subset=0.01, use_high_level=False)
        assert dataset.num_high_level_features == 0

    def test_raw_file_names_includes_split_directory(self):
        """Test raw_file_names includes split directory."""
        from topobench.data.datasets.atlas_top_tagging_dataset import ATLASTopTaggingDataset
        dataset = ATLASTopTaggingDataset(root='/tmp/nonexistent', split='train', subset=0.01)
        assert 'train' in str(dataset.raw_file_names)

class TestATLASDatasetExpectedFilenames:
    """Test _expected_filenames method comprehensively for full coverage."""

    def test_expected_filenames_zero_padding(self):
        """Test filenames have correct zero-padding (000, 001, etc.)."""
        from topobench.data.datasets.atlas_top_tagging_dataset import ATLASTopTaggingDataset
        dataset = ATLASTopTaggingDataset(root='/tmp/nonexistent', split='train', subset=0.01)
        filenames = dataset._expected_filenames()
        # Check that first file is 000, properly zero-padded
        assert 'train_nominal_000.h5.gz' in filenames[0]
        # If we have more than 1 file, check second one too
        if len(filenames) > 1:
            assert 'train_nominal_001.h5.gz' in filenames[1]

    def test_expected_filenames_starts_from_zero(self):
        """Test filename numbering starts from 0."""
        from topobench.data.datasets.atlas_top_tagging_dataset import ATLASTopTaggingDataset
        dataset = ATLASTopTaggingDataset(root='/tmp/nonexistent', split='train', subset=0.01)
        filenames = dataset._expected_filenames()
        # First filename should contain _000
        assert '_000.h5.gz' in filenames[0]

    def test_expected_filenames_sequential_numbering(self):
        """Test filenames are numbered sequentially."""
        from topobench.data.datasets.atlas_top_tagging_dataset import ATLASTopTaggingDataset
        dataset = ATLASTopTaggingDataset(root='/tmp/nonexistent', split='train', subset=0.02)
        filenames = dataset._expected_filenames()
        # Extract numbers from filenames and verify they're sequential
        numbers = []
        for filename in filenames:
            # Extract number from train_nominal_XXX.h5.gz
            num_str = filename.split('_')[-1].replace('.h5.gz', '')
            numbers.append(int(num_str))
        
        # Check sequential: 0, 1, 2, ...
        expected = list(range(len(filenames)))
        assert numbers == expected

    def test_expected_filenames_full_dataset(self):
        """Test _expected_filenames with subset=1.0 (full dataset)."""
        from topobench.data.datasets.atlas_top_tagging_dataset import ATLASTopTaggingDataset
        dataset_train = ATLASTopTaggingDataset(root='/tmp/nonexistent', split='train', subset=1.0)
        filenames_train = dataset_train._expected_filenames()
        # Should return 930 files for train
        assert len(filenames_train) == 930
        
        dataset_test = ATLASTopTaggingDataset(root='/tmp/nonexistent', split='test', subset=1.0)
        filenames_test = dataset_test._expected_filenames()
        # Should return 100 files for test
        assert len(filenames_test) == 100

    def test_expected_filenames_minimum_one_file(self):
        """Test _expected_filenames returns at least 1 file even with tiny subset."""
        from topobench.data.datasets.atlas_top_tagging_dataset import ATLASTopTaggingDataset
        # Even with very small subset, should get at least 1 file
        dataset = ATLASTopTaggingDataset(root='/tmp/nonexistent', split='train', subset=0.001)
        filenames = dataset._expected_filenames()
        assert len(filenames) >= 1

    def test_expected_filenames_different_splits_have_different_prefixes(self):
        """Test train and test splits have different filename prefixes."""
        from topobench.data.datasets.atlas_top_tagging_dataset import ATLASTopTaggingDataset
        dataset_train = ATLASTopTaggingDataset(root='/tmp/nonexistent', split='train', subset=0.01)
        dataset_test = ATLASTopTaggingDataset(root='/tmp/nonexistent', split='test', subset=0.01)
        
        filenames_train = dataset_train._expected_filenames()
        filenames_test = dataset_test._expected_filenames()
        
        # Train filenames should have 'train_nominal' prefix
        assert all('train_nominal' in fn for fn in filenames_train)
        # Test filenames should have 'test_nominal' prefix
        assert all('test_nominal' in fn for fn in filenames_test)

    def test_expected_filenames_calculation_logic(self):
        """Test the calculation logic: max(1, min(total, int(round(total * subset))))."""
        from topobench.data.datasets.atlas_top_tagging_dataset import ATLASTopTaggingDataset
        
        # Test with 10% of train data (930 * 0.1 = 93)
        dataset = ATLASTopTaggingDataset(root='/tmp/nonexistent', split='train', subset=0.1)
        filenames = dataset._expected_filenames()
        assert len(filenames) == 93
        
        # Test with 50% of test data (100 * 0.5 = 50)
        dataset_test = ATLASTopTaggingDataset(root='/tmp/nonexistent', split='test', subset=0.5)
        filenames_test = dataset_test._expected_filenames()
        assert len(filenames_test) == 50

    def test_expected_filenames_exact_format(self):
        """Test exact filename format matches specification."""
        from topobench.data.datasets.atlas_top_tagging_dataset import ATLASTopTaggingDataset
        dataset = ATLASTopTaggingDataset(root='/tmp/nonexistent', split='train', subset=0.01)
        filenames = dataset._expected_filenames()
        
        # Each filename should match: {split}_nominal_{number:03d}.h5.gz
        import re
        pattern = r'^train_nominal_\d{3}\.h5\.gz$'
        for filename in filenames:
            assert re.match(pattern, filename), f"Filename {filename} doesn't match expected pattern"

    def test_expected_filenames_list_comprehension_coverage(self):
        """Test that exercises the list comprehension on line 278."""
        from topobench.data.datasets.atlas_top_tagging_dataset import ATLASTopTaggingDataset
        
        # Create dataset that will generate multiple files
        dataset = ATLASTopTaggingDataset(root='/tmp/nonexistent', split='train', subset=0.05)
        filenames = dataset._expected_filenames()
        
        # Verify the list comprehension generates correct range
        n_files = len(filenames)
        assert n_files > 1  # Make sure we have multiple files
        
        # Verify each index from 0 to n-1 is represented
        for i in range(n_files):
            expected_filename = f"train_nominal_{i:03d}.h5.gz"
            assert expected_filename in filenames[i]


class TestATLASDatasetTotalFilesForSplit:
    """Test _total_files_for_split method for complete coverage."""

    def test_total_files_for_split_train_exact_value(self):
        """Test train split returns exactly 930 files."""
        from topobench.data.datasets.atlas_top_tagging_dataset import ATLASTopTaggingDataset
        dataset = ATLASTopTaggingDataset(root='/tmp/nonexistent', split='train', subset=0.01)
        total = dataset._total_files_for_split()
        assert total == 930
        assert isinstance(total, int)

    def test_total_files_for_split_test_exact_value(self):
        """Test test split returns exactly 100 files."""
        from topobench.data.datasets.atlas_top_tagging_dataset import ATLASTopTaggingDataset
        dataset = ATLASTopTaggingDataset(root='/tmp/nonexistent', split='test', subset=0.01)
        total = dataset._total_files_for_split()
        assert total == 100
        assert isinstance(total, int)

    def test_total_files_consistent_across_calls(self):
        """Test _total_files_for_split returns same value on multiple calls."""
        from topobench.data.datasets.atlas_top_tagging_dataset import ATLASTopTaggingDataset
        dataset = ATLASTopTaggingDataset(root='/tmp/nonexistent', split='train', subset=0.01)
        first_call = dataset._total_files_for_split()
        second_call = dataset._total_files_for_split()
        assert first_call == second_call

    def test_total_files_independent_of_subset(self):
        """Test _total_files_for_split is independent of subset parameter."""
        from topobench.data.datasets.atlas_top_tagging_dataset import ATLASTopTaggingDataset
        small_subset = ATLASTopTaggingDataset(root='/tmp/nonexistent', split='train', subset=0.01)
        large_subset = ATLASTopTaggingDataset(root='/tmp/nonexistent', split='train', subset=0.5)
        
        # Both should return same total (subset doesn't affect total count)
        assert small_subset._total_files_for_split() == large_subset._total_files_for_split()