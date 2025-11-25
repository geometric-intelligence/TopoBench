"""Test QM9SDataset class."""

import os
import torch
from torch_geometric.data import Data

from qm9s_dataset import QM9SDataset


class TestQM9SDataset:
    """Test QM9SDataset class."""
    
    def setup_method(self):
        """Set up test environment."""
        self.test_root = './data/QM9S_test_basic'
        
        # Clean up any existing test directory
        if os.path.exists(self.test_root):
            import shutil
            shutil.rmtree(self.test_root)

    def teardown_method(self):
        """Clean up test directory."""
        if os.path.exists(self.test_root):
            import shutil
            shutil.rmtree(self.test_root)

    def test_basic_functionality(self):
        """Test basic dataset creation and sample access."""
        print("TEST: Basic Dataset Functionality")
        print("=" * 50)
        
        # Create dataset
        dataset = QM9SDataset(
            root=self.test_root,
            num_molecules=10,
            force_process=True
        )
        
        # Test dataset length
        assert len(dataset) == 10
        print("Dataset length correct")
        
        # Test sample access
        for i in range(3):  # Test first 3 samples
            mol = dataset[i]
            assert isinstance(mol, Data)
            assert hasattr(mol, 'x')
            assert hasattr(mol, 'edge_index') 
            assert hasattr(mol, 'y')
            assert hasattr(mol, 'idx')
            assert mol.idx == i
            print(f"Sample {i} accessible: {mol.num_nodes} nodes, {mol.num_edges} edges")
        
        # Test all samples are accessible
        for i in range(len(dataset)):
            mol = dataset[i]
            assert mol is not None
        print("All samples accessible")
        
        print("Basic functionality test passed!")


def run_basic_test():
    """Run the basic functionality test."""
    print("Running QM9SDataset Basic Test")
    print("=" * 60)
    
    # Create test directory
    os.makedirs('./data', exist_ok=True)
    
    test_class = TestQM9SDataset()
    
    try:
        test_class.setup_method()
        test_class.test_basic_functionality()
        test_class.teardown_method()
        print("All tests passed! QM9SDataset basic functionality is working correctly.")
        return True
    except Exception as e:
        print(f"Test failed: {e}")
        test_class.teardown_method()
        return False


if __name__ == "__main__":
    # Run the basic test
    success = run_basic_test()
    
    # Exit with appropriate code
    exit(0 if success else 1)