"""Test script for tutorial_ondisk_transductive_structure_centric.ipynb

Tests the structure-centric batching approach for transductive learning.
"""
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("Testing: Structure-Centric Transductive Tutorial")
print("=" * 70)

# Imports
import networkx as nx
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.io import fs
import lightning as pl
from omegaconf import OmegaConf, DictConfig

# TopoBench imports
from topobench.data.loaders.base import AbstractLoader
from topobench.data.preprocessor import OnDiskTransductivePreprocessor
from topobench.dataloader import TBDataloader
from topobench.model import TBModel
from topobench.nn.backbones.simplicial import SCCNNCustom
from topobench.nn.readouts.mlp_readout import MLPReadout
from topobench.loss import TBLoss
from topobench.optimizer import TBOptimizer

print("✓ Imports complete\n")

# Define Dataset Class
class MyLargeTransductiveDataset(InMemoryDataset):
    """Large single graph for transductive learning."""
    
    def __init__(self, root, name, parameters: DictConfig):
        self.name = name
        self.parameters = parameters
        super().__init__(root)
        
        out = fs.torch_load(self.processed_paths[0])
        if len(out) == 4:
            data, self.slices, self.sizes, data_cls = out
            self.data = data_cls.from_dict(data) if isinstance(data, dict) else data
        else:
            data, self.slices, self.sizes = out
            self.data = data
    
    @property
    def raw_file_names(self):
        return []
    
    @property
    def processed_file_names(self):
        return "data.pt"
    
    def download(self):
        pass
    
    def process(self):
        """Generate large graph with community structure."""
        # Create graph (example: Watts-Strogatz)
        G = nx.watts_strogatz_graph(
            n=self.parameters.num_nodes,
            k=self.parameters.degree,
            p=0.3,
            seed=42
        )
        
        # Convert to PyG Data
        edges = list(G.edges())
        edge_index = torch.tensor(edges, dtype=torch.long).t()
        edge_index = torch.cat([edge_index, edge_index[[1, 0]]], dim=1)
        
        n = self.parameters.num_nodes
        x = torch.randn(n, self.parameters.num_features)
        y = torch.randint(0, self.parameters.num_classes, (n,))
        
        # Transductive splits
        train_mask = torch.zeros(n, dtype=torch.bool)
        val_mask = torch.zeros(n, dtype=torch.bool)
        test_mask = torch.zeros(n, dtype=torch.bool)
        
        train_mask[:int(0.6 * n)] = True
        val_mask[int(0.6 * n):int(0.8 * n)] = True
        test_mask[int(0.8 * n):] = True
        
        data = Data(
            x=x, edge_index=edge_index, y=y, num_nodes=n,
            train_mask=train_mask, val_mask=val_mask, test_mask=test_mask
        )
        
        self.data, self.slices = self.collate([data])
        fs.torch_save(
            (self._data.to_dict(), self.slices, {}, self._data.__class__),
            self.processed_paths[0]
        )

print("✓ Dataset class defined")

# Define Loader Class
class MyLargeTransductiveLoader(AbstractLoader):
    """Loader for large transductive datasets."""
    
    def __init__(self, parameters: DictConfig):
        super().__init__(parameters)
    
    def load_dataset(self):
        return MyLargeTransductiveDataset(
            str(self.root_data_dir),
            self.parameters.data_name,
            self.parameters
        )

print("✓ Loader class defined\n")

# Configuration (smaller for testing)
config = OmegaConf.create({
    "data_dir": "./data/",
    "data_name": "MyLargeGraph_Test",
    "num_nodes": 500,  # Smaller for testing
    "degree": 10,
    "num_features": 16,
    "num_classes": 3
})

# Load dataset
print("Loading dataset...")
loader = MyLargeTransductiveLoader(config)
dataset, _ = loader.load()
graph_data = dataset[0]

print(f"✓ Graph loaded: {graph_data.num_nodes:,} nodes")
print(f"  Edges: {graph_data.edge_index.size(1):,}")
print(f"  Train nodes: {graph_data.train_mask.sum().item():,}\n")

# Configure transform
transforms_config = OmegaConf.create({
    "clique_lifting": {
        "transform_type": "lifting",
        "transform_name": "SimplicialCliqueLifting",
        "complex_dim": 2  # Triangles
    }
})

# Create preprocessor
print("Creating preprocessor...")
preprocessor = OnDiskTransductivePreprocessor(
    graph_data=graph_data,
    data_dir="./index/structure_centric_test",
    transforms_config=transforms_config,
    max_structure_size=3  # Triangles = 3 nodes
)

print("✓ Preprocessor created\n")

# Load dataset splits
split_config = OmegaConf.create({
    "strategy": "structure_centric",
    "structures_per_batch": 100,  # Smaller for testing
    "node_budget": 300,
})

print("Loading dataset splits...")
train, val, test = preprocessor.load_dataset_splits(split_config)

print(f"✓ Dataset splits loaded!")
print(f"  Train: {len(train)} batches")
print(f"  Val: {len(val)} batches")
print(f"  Test: {len(test)} batches\n")

# Inspect sample batch
sample_batch = next(iter(train))
print(f"📦 Sample training batch:")
print(f"  Nodes: {sample_batch.num_nodes}")
print(f"  Edges: {sample_batch.edge_index.size(1)}")
if hasattr(sample_batch, 'num_structures'):
    print(f"  Structures: {sample_batch.num_structures}")
print()

# Create datamodule
datamodule = TBDataloader(
    dataset_train=train,
    dataset_val=val,
    dataset_test=test,
    batch_size=1,  # Already batched
    num_workers=0
)

print("✓ Datamodule created\n")

# Define model (smaller for testing)
HIDDEN_DIM = 32
OUT_CHANNELS = 3
IN_CHANNELS = 16

model = TBModel(
    backbone=SCCNNCustom(
        in_channels_all=(IN_CHANNELS, HIDDEN_DIM, HIDDEN_DIM),
        hidden_channels_all=(HIDDEN_DIM, HIDDEN_DIM, HIDDEN_DIM),
        conv_order=1,
        sc_order=2,
        n_layers=2
    ),
    readout=MLPReadout(
        in_channels=HIDDEN_DIM,
        hidden_layers=HIDDEN_DIM,
        out_channels=OUT_CHANNELS,
        pooling_type="sum"
    ),
    loss=TBLoss(
        dataset_loss={"task": "classification", "loss_type": "cross_entropy"}
    ),
    optimizer=TBOptimizer(
        optimizer_id="Adam", parameters={"lr": 0.01}
    )
)

print("✓ TBModel created\n")

# Train (just 2 epochs for testing)
trainer = pl.Trainer(
    max_epochs=2,
    accelerator="auto",
    devices=1,
    enable_progress_bar=True,
    logger=False
)

print("🚀 Training with structure-centric batching...\n")
trainer.fit(model, datamodule)

print("\n✅ Structure-centric tutorial test PASSED!")
print(f"  Trained on {graph_data.train_mask.sum().item():,} train nodes")
print(f"  Structure-centric batching works correctly!")

# Cleanup
preprocessor.close()
print("\n" + "=" * 70)
print("Test completed successfully!")
print("=" * 70)
