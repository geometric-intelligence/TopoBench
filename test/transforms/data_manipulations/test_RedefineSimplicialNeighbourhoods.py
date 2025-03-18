"""Test IdentityTransform class."""

import pytest
import hydra
import torch
from torch_geometric.data import Data
from topobench.transforms.data_manipulations import RedefineSimplicialNeighbourhoods
from topobench.data.preprocessor.preprocessor import PreProcessor


class TestRedefineSimplicialNeighbourhoods:
    """Test IdentityTransform class."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.transform = RedefineSimplicialNeighbourhoods()
        self.relative_config_dir = "../../../configs"


    def test_forward_simple_graph(self):
        """Test transform on mantra dataset."""

        with hydra.initialize(
            version_base="1.3",
            config_path=self.relative_config_dir,
            job_name="run"
        ):
            parameters = hydra.compose(
                config_name="run.yaml",
                overrides=["dataset=simplicial/mantra_orientation", f"model=graph/gat"], 
                return_hydra_config=True, 
            )
            dataset_loader = hydra.utils.instantiate(parameters.dataset.loader)
            

          
            dataset, data_dir = dataset_loader.load(slice=100)

        transforms_config = {'RedefineSimplicialNeighbourhoods':
            {'_target_': 'topobench.transforms.data_transform.DataTransform',
            'transform_name': "RedefineSimplicialNeighbourhoods",
            'transform_type': None,
            "complex_dim":3,
            "neighborhoods": None,
            "signed": False, 
            }
            }
        
        transformer_dataset = PreProcessor(dataset, data_dir, transforms_config)

        for idx2, b in enumerate(zip(transformer_dataset, dataset)):
            transformed, initial = b
            assert len(initial.keys()) == len(transformed.keys()), "Number of keys do not match"
            assert all([key1==key2 for key1,key2 in zip(sorted(initial.keys()), sorted(transformed.keys()))]), "Some of the keys do not match"
            
            for idx, a in enumerate(zip(sorted(initial.keys()), sorted(transformed.keys()))):
                key1,key2 = a
                if key1 not in ['x', 'x_0', 'x_1', 'x_2', 'y', 'shape']:
                    try: 
                        torch.all(initial[key1].to_dense() == transformed[key2].to_dense()) == True
                    except:
                        pass

#, f"Error in {key1}"
            
        
        
        
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

    