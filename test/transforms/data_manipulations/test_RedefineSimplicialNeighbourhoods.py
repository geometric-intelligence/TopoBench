"""Test IdentityTransform class."""

import pytest
import hydra
import torch
from torch_geometric.data import Data
from topobench.transforms.data_manipulations import RedefineSimplicialNeighbourhoods

class RedefineSimplicialNeighbourhoods:
    """Test IdentityTransform class."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.transform = RedefineSimplicialNeighbourhoods()


    def test_forward_simple_graph(self):
        """Test transform on mantra dataset."""

        with hydra.initialize(
            version_base="1.3",
            config_path=self.relative_config_dir,
            job_name="run"
        ):
            parameters = hydra.compose(
                config_name="run.yaml",
                overrides=["dataset=simplicial/mantra", f"model=graph/gat"], 
                return_hydra_config=True, 
            )
            dataset_loader = hydra.utils.instantiate(parameters.dataset.loader)
            

          
            dataset, data_dir = dataset_loader.load(slice=100)
          
        self.transform(dataset[0])
        
        
    # def test_repr(self):
    #     """Test string representation of the transform."""
    #     repr_str = repr(self.transform)
    #     assert "IdentityTransform" in repr_str
    #     assert "domain2domain" in repr_str
    #     assert "parameters={}" in repr_str

    #     # Test repr with parameters
    #     transform = RedefineSimplicialNeighbourhoods(param="test")
    #     repr_str = repr(transform)
    #     assert "param" in repr_str
    #     assert "test" in repr_str

    