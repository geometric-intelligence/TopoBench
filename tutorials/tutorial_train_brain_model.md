# Training TBModel on Auditory Cortex Data

This notebook demonstrates three different tasks using the A123 mouse auditory cortex dataset:

1. **Graph-level Classification**: Predict frequency bin (0-8) from graph structure
2. **Triangle Classification**: Classify topological role of triangles in the correlation graph
3. **Triangle Common-Neighbors**: Predict the number of common neighbors for triangles

We'll show how to load the dataset, apply lifting transformations, define a backbone, and train a `TBModel` using `TBLoss` and `TBOptimizer`.

Requirements: the project installed in PYTHONPATH and optional dependencies (torch_geometric, networkx, ripser/persim) if you want advanced features.


```python
import os
os.chdir('..')
```


```python
# 1) Imports
import torch
import numpy as np
import lightning as pl
from omegaconf import OmegaConf

# Data loading / preprocessing utilities from the repo
from topobench.data.loaders.graph.a123_loader import A123DatasetLoader
from topobench.dataloader.dataloader import TBDataloader
from topobench.data.preprocessor import PreProcessor

# Model / training building blocks
from topobench.model.model import TBModel
# example backbone building block (SCN2 is optional; we provide a tiny custom backbone below)
# from topomodelx.nn.simplicial.scn2 import SCN2
from topobench.nn.wrappers.simplicial import SCNWrapper
from topobench.nn.encoders import AllCellFeatureEncoder
from topobench.nn.readouts import PropagateSignalDown

# Optimization / evaluation
from topobench.loss.loss import TBLoss
from topobench.optimizer import TBOptimizer
from topobench.evaluator.evaluator import TBEvaluator

print('Imports OK')
```

    Imports OK


    /Users/mariayuffa/anaconda3/envs/tb3/lib/python3.11/site-packages/outdated/__init__.py:36: UserWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html. The pkg_resources package is slated for removal as early as 2025-11-30. Refrain from using this package or pin to Setuptools<81.
      from pkg_resources import parse_version



```python
# 2) Configurations for different tasks
# Note: We'll demonstrate each task separately by changing the specific_task parameter

# Common loader config
loader_config_base = {
    'data_domain': 'graph',
    'data_type': 'A123',
    'data_name': 'a123_cortex_m',
    'data_dir': './data/a123/',
    'corr_threshold': 0.3,  # Higher threshold ensures graphs have meaningful edges
}

# Transform config: using CellCycleLifting (more robust for graphs with few edges)
# CellCycleLifting finds cycles and lifts them to 2-cells, handles empty graphs gracefully
transform_config = {
    'transform_type': 'lifting',
    'transform_name': 'CellCycleLifting',
    'max_cell_length': None,  # No limit on cycle length
}

split_config = {
    'learning_setting': 'inductive',
    'split_type': 'random',
    'data_seed': 0,
    'data_split_dir': './data/a123/splits/',
    'train_prop': 0.5,
}

# Task configurations
tasks = {
    'graph_classification': {
        'description': 'Graph-level classification (predict frequency bin 0-8)',
        'specific_task': 'classification',
        'in_channels': 3,
        'out_channels': 9,
        'task_level': 'graph',
    },
    'triangle_classification': {
        'description': 'Triangle classification (predict topological role, 9 classes)',
        'specific_task': 'triangle_classification',
        'in_channels': 3,
        'out_channels': 9,
        'task_level': 'graph',
    },
    'triangle_common_neighbors': {
        'description': 'Triangle common-neighbors (predict # common neighbors, 9 classes)',
        'specific_task': 'triangle_common_neighbors',
        'in_channels': 3,
        'out_channels': 9,
        'task_level': 'graph',
    }
}

# Select task to run (change to 'classification', 'triangle_classification' or 'triangle_common_neighbors' to run different tasks)
TASK_NAME = 'triangle_common_neighbors'
TASK_CONFIG = tasks[TASK_NAME]

print(f"Selected task: {TASK_NAME}")
print(f"Description: {TASK_CONFIG['description']}")

# Create loader config with specific task
loader_config = OmegaConf.create({**loader_config_base, 'specific_task': TASK_CONFIG['specific_task']})

dim_hidden = 16
in_channels = TASK_CONFIG['in_channels']
out_channels = TASK_CONFIG['out_channels']

readout_config = {
    'readout_name': 'PropagateSignalDown',
    'num_cell_dimensions': 1,
    'hidden_dim': dim_hidden,
    'out_channels': out_channels,
    'task_level': TASK_CONFIG['task_level'],
    'pooling_type': 'sum',
}

loss_config = {
    'dataset_loss': {
        'task': 'classification',
        'loss_type': 'cross_entropy',
    }
}

evaluator_config = {
    'task': 'classification',
    'num_classes': out_channels,
    'metrics': ['f1', 'precision', 'recall', 'accuracy'],
}

optimizer_config = {
    'optimizer_id': 'Adam',
    'parameters': {'lr': 0.001, 'weight_decay': 0.0005},
}

# Convert to OmegaConf
transform_config = OmegaConf.create(transform_config)
split_config = OmegaConf.create(split_config)
readout_config = OmegaConf.create(readout_config)
loss_config = OmegaConf.create(loss_config)
evaluator_config = OmegaConf.create(evaluator_config)
optimizer_config = OmegaConf.create(optimizer_config)

print('Configs created')
print(f"Loader config: {loader_config}")
print(f"Input channels: {in_channels}, Output channels: {out_channels}")
```

    Selected task: triangle_common_neighbors
    Description: Triangle common-neighbors (predict # common neighbors, 9 classes)
    Configs created
    Loader config: {'data_domain': 'graph', 'data_type': 'A123', 'data_name': 'a123_cortex_m', 'data_dir': './data/a123/', 'corr_threshold': 0.3, 'specific_task': 'triangle_common_neighbors'}
    Input channels: 3, Output channels: 9



```python
# 3) Loading the data

# Use the A123-specific loader (A123DatasetLoader) to construct the dataset
graph_loader = A123DatasetLoader(loader_config)

dataset, dataset_dir = graph_loader.load()
print(f'Dataset loaded: {len(dataset)} samples')

# For triangle-level tasks, skip lifting transformations (triangles have no edge_index)
# Only apply lifting for graph-level classification
task_type = TASK_CONFIG['specific_task']
if task_type in ['triangle_classification', 'triangle_common_neighbors']:
    # Skip lifting for triangle tasks - they don't have graph structure
    print(f"Task '{task_type}' uses triangle-level features (no edge_index)")
    print("Skipping lifting transformation for triangle data")
    preprocessor = PreProcessor(dataset, dataset_dir, transforms_config=None)
else:
    # Apply lifting for graph-level tasks
    preprocessor = PreProcessor(dataset, dataset_dir, transform_config)

dataset_train, dataset_val, dataset_test = preprocessor.load_dataset_splits(split_config)
print(f'Dataset splits created:')
print(f'  Train: {len(dataset_train)} samples')
print(f'  Val: {len(dataset_val)} samples')
print(f'  Test: {len(dataset_test)} samples')

# create the TopoBench datamodule / dataloader wrappers
datamodule = TBDataloader(dataset_train, dataset_val, dataset_test, batch_size=32)

print('Datasets and datamodule ready')
```

    Processing...


    [A123] Processing dataset from: data/a123_cortex_m/raw
    [A123] Files in raw_dir: ['Auditory cortex data', '__MACOSX']
    [A123] Starting extract_samples()...
    Processing session 0: allPlanesVariables27-Feb-2021.mat
    Processing session 1: allPlanesVariables27-Feb-2021.mat
    Processing session 1: allPlanesVariables27-Feb-2021.mat
    Processing session 2: allPlanesVariables27-Feb-2021.mat
    Processing session 2: allPlanesVariables27-Feb-2021.mat
    Processing session 3: allPlanesVariables27-Feb-2021.mat
    Processing session 3: allPlanesVariables27-Feb-2021.mat
    Processing session 4: allPlanesVariables27-Feb-2021.mat
    Processing session 4: allPlanesVariables27-Feb-2021.mat
    Processing session 5: allPlanesVariables27-Feb-2021.mat
    Processing session 5: allPlanesVariables27-Feb-2021.mat
    Processing session 6: allPlanesVariables27-Feb-2021.mat
    Processing session 6: allPlanesVariables27-Feb-2021.mat
    Processing session 7: allPlanesVariables27-Feb-2021.mat
    Processing session 7: allPlanesVariables27-Feb-2021.mat
    [A123] Extracted 250 samples
    [A123] Converting sample 0/250 to PyG Data...
    [A123] Converting sample 100/250 to PyG Data...
    [A123] Converting sample 200/250 to PyG Data...
    [A123] Collating 250 samples (removed 0 empty graphs)...
    [A123] Saving processed data to data/a123_cortex_m/processed/data.pt...
    [A123] Triangle common-neighbours task enabled. Creating dataset...
    [A123] Starting triangle extraction from graphs...
    [A123] Processing graph 0: extracting triangles...
    [A123]   Found 2268 triangles in graph 0
    [A123] Processing graph 1: extracting triangles...
    [A123]   Found 1277 triangles in graph 1
    [A123] Processing graph 2: extracting triangles...
    [A123]   Found 946 triangles in graph 2
    [A123] Processing graph 3: extracting triangles...
    [A123]   Found 462 triangles in graph 3
    [A123] Processing graph 4: extracting triangles...
    [A123]   Found 334 triangles in graph 4
    [A123] Processing graph 5: extracting triangles...
    [A123]   Found 304 triangles in graph 5
    [A123] Processing graph 6: extracting triangles...
    [A123]   Found 5021 triangles in graph 6
    [A123] Processing graph 7: extracting triangles...
    [A123]   Found 1343 triangles in graph 7
    [A123] Processing graph 8: extracting triangles...
    [A123]   Found 246 triangles in graph 8
    [A123] Processing graph 9: extracting triangles...
    [A123]   Found 80 triangles in graph 9
    [A123] Processed 10/250 graphs (0s elapsed, ~2s remaining)...
    [A123] Processing graph 10: extracting triangles...
    [A123]   Found 69 triangles in graph 10
    [A123] Processing graph 11: extracting triangles...
    [A123]   Found 303 triangles in graph 11
    [A123] Processing graph 12: extracting triangles...
    [A123]   Found 4 triangles in graph 12
    [A123] Processing graph 13: extracting triangles...
    [A123]   Found 2 triangles in graph 13
    [A123] Processing graph 14: extracting triangles...
    [A123]   Found 404 triangles in graph 14
    [A123] Processing graph 15: extracting triangles...
    [A123]   Found 399 triangles in graph 15
    [A123] Processing graph 16: extracting triangles...
    [A123]   Found 291 triangles in graph 16
    [A123] Processing graph 17: extracting triangles...
    [A123]   Found 766 triangles in graph 17
    [A123] Processing graph 18: extracting triangles...
    [A123]   Found 179 triangles in graph 18
    [A123] Processing graph 19: extracting triangles...
    [A123]   Found 95 triangles in graph 19
    [A123] Processed 20/250 graphs (0s elapsed, ~1s remaining)...
    [A123] Processing graph 20: extracting triangles...
    [A123]   Found 41 triangles in graph 20
    [A123] Processing graph 21: extracting triangles...
    [A123]   Found 6 triangles in graph 21
    [A123] Processing graph 22: extracting triangles...
    [A123]   Found 114 triangles in graph 22
    [A123] Processing graph 23: extracting triangles...
    [A123]   Found 381 triangles in graph 23
    [A123] Processing graph 24: extracting triangles...
    [A123]   Found 72 triangles in graph 24
    [A123] Processing graph 25: extracting triangles...
    [A123]   Found 477 triangles in graph 25
    [A123] Extracted 250 samples
    [A123] Converting sample 0/250 to PyG Data...
    [A123] Converting sample 100/250 to PyG Data...
    [A123] Converting sample 200/250 to PyG Data...
    [A123] Collating 250 samples (removed 0 empty graphs)...
    [A123] Saving processed data to data/a123_cortex_m/processed/data.pt...
    [A123] Triangle common-neighbours task enabled. Creating dataset...
    [A123] Starting triangle extraction from graphs...
    [A123] Processing graph 0: extracting triangles...
    [A123]   Found 2268 triangles in graph 0
    [A123] Processing graph 1: extracting triangles...
    [A123]   Found 1277 triangles in graph 1
    [A123] Processing graph 2: extracting triangles...
    [A123]   Found 946 triangles in graph 2
    [A123] Processing graph 3: extracting triangles...
    [A123]   Found 462 triangles in graph 3
    [A123] Processing graph 4: extracting triangles...
    [A123]   Found 334 triangles in graph 4
    [A123] Processing graph 5: extracting triangles...
    [A123]   Found 304 triangles in graph 5
    [A123] Processing graph 6: extracting triangles...
    [A123]   Found 5021 triangles in graph 6
    [A123] Processing graph 7: extracting triangles...
    [A123]   Found 1343 triangles in graph 7
    [A123] Processing graph 8: extracting triangles...
    [A123]   Found 246 triangles in graph 8
    [A123] Processing graph 9: extracting triangles...
    [A123]   Found 80 triangles in graph 9
    [A123] Processed 10/250 graphs (0s elapsed, ~2s remaining)...
    [A123] Processing graph 10: extracting triangles...
    [A123]   Found 69 triangles in graph 10
    [A123] Processing graph 11: extracting triangles...
    [A123]   Found 303 triangles in graph 11
    [A123] Processing graph 12: extracting triangles...
    [A123]   Found 4 triangles in graph 12
    [A123] Processing graph 13: extracting triangles...
    [A123]   Found 2 triangles in graph 13
    [A123] Processing graph 14: extracting triangles...
    [A123]   Found 404 triangles in graph 14
    [A123] Processing graph 15: extracting triangles...
    [A123]   Found 399 triangles in graph 15
    [A123] Processing graph 16: extracting triangles...
    [A123]   Found 291 triangles in graph 16
    [A123] Processing graph 17: extracting triangles...
    [A123]   Found 766 triangles in graph 17
    [A123] Processing graph 18: extracting triangles...
    [A123]   Found 179 triangles in graph 18
    [A123] Processing graph 19: extracting triangles...
    [A123]   Found 95 triangles in graph 19
    [A123] Processed 20/250 graphs (0s elapsed, ~1s remaining)...
    [A123] Processing graph 20: extracting triangles...
    [A123]   Found 41 triangles in graph 20
    [A123] Processing graph 21: extracting triangles...
    [A123]   Found 6 triangles in graph 21
    [A123] Processing graph 22: extracting triangles...
    [A123]   Found 114 triangles in graph 22
    [A123] Processing graph 23: extracting triangles...
    [A123]   Found 381 triangles in graph 23
    [A123] Processing graph 24: extracting triangles...
    [A123]   Found 72 triangles in graph 24
    [A123] Processing graph 25: extracting triangles...
    [A123]   Found 477 triangles in graph 25
    [A123] Processing graph 26: extracting triangles...
    [A123]   Found 137 triangles in graph 26
    [A123] Processing graph 27: extracting triangles...
    [A123]   Found 48 triangles in graph 27
    [A123] Processing graph 28: extracting triangles...
    [A123]   Found 2 triangles in graph 28
    [A123] Processing graph 29: extracting triangles...
    [A123]   Found 3 triangles in graph 29
    [A123] Processed 30/250 graphs (0s elapsed, ~1s remaining)...
    [A123] Processing graph 30: extracting triangles...
    [A123]   Found 309 triangles in graph 30
    [A123] Processing graph 31: extracting triangles...
    [A123]   Found 418 triangles in graph 31
    [A123] Processing graph 32: extracting triangles...
    [A123]   Found 230 triangles in graph 32
    [A123] Processing graph 33: extracting triangles...
    [A123]   Found 188 triangles in graph 33
    [A123] Processing graph 34: extracting triangles...
    [A123]   Found 299 triangles in graph 34
    [A123] Processing graph 35: extracting triangles...
    [A123]   Found 79 triangles in graph 35
    [A123] Processing graph 36: extracting triangles...
    [A123]   Found 22 triangles in graph 36
    [A123] Processing graph 37: extracting triangles...
    [A123]   Found 3 triangles in graph 37
    [A123] Processing graph 38: extracting triangles...
    [A123]   Found 10 triangles in graph 38
    [A123] Processing graph 39: extracting triangles...
    [A123]   Found 86 triangles in graph 39
    [A123] Processed 40/250 graphs (0s elapsed, ~0s remaining)...
    [A123] Processing graph 40: extracting triangles...
    [A123]   Found 10005 triangles in graph 40
    [A123] Processing graph 41: extracting triangles...
    [A123]   Found 476 triangles in graph 41
    [A123] Processing graph 42: extracting triangles...
    [A123] Processing graph 26: extracting triangles...
    [A123]   Found 137 triangles in graph 26
    [A123] Processing graph 27: extracting triangles...
    [A123]   Found 48 triangles in graph 27
    [A123] Processing graph 28: extracting triangles...
    [A123]   Found 2 triangles in graph 28
    [A123] Processing graph 29: extracting triangles...
    [A123]   Found 3 triangles in graph 29
    [A123] Processed 30/250 graphs (0s elapsed, ~1s remaining)...
    [A123] Processing graph 30: extracting triangles...
    [A123]   Found 309 triangles in graph 30
    [A123] Processing graph 31: extracting triangles...
    [A123]   Found 418 triangles in graph 31
    [A123] Processing graph 32: extracting triangles...
    [A123]   Found 230 triangles in graph 32
    [A123] Processing graph 33: extracting triangles...
    [A123]   Found 188 triangles in graph 33
    [A123] Processing graph 34: extracting triangles...
    [A123]   Found 299 triangles in graph 34
    [A123] Processing graph 35: extracting triangles...
    [A123]   Found 79 triangles in graph 35
    [A123] Processing graph 36: extracting triangles...
    [A123]   Found 22 triangles in graph 36
    [A123] Processing graph 37: extracting triangles...
    [A123]   Found 3 triangles in graph 37
    [A123] Processing graph 38: extracting triangles...
    [A123]   Found 10 triangles in graph 38
    [A123] Processing graph 39: extracting triangles...
    [A123]   Found 86 triangles in graph 39
    [A123] Processed 40/250 graphs (0s elapsed, ~0s remaining)...
    [A123] Processing graph 40: extracting triangles...
    [A123]   Found 10005 triangles in graph 40
    [A123] Processing graph 41: extracting triangles...
    [A123]   Found 476 triangles in graph 41
    [A123] Processing graph 42: extracting triangles...
    [A123]   Found 43198 triangles in graph 42
    [A123] Processing graph 43: extracting triangles...
    [A123]   Found 20598 triangles in graph 43
    [A123] Processing graph 44: extracting triangles...
    [A123]   Found 135 triangles in graph 44
    [A123] Processing graph 45: extracting triangles...
    [A123]   Found 2451 triangles in graph 45
    [A123] Processing graph 46: extracting triangles...
    [A123]   Found 8356 triangles in graph 46
    [A123]   Found 43198 triangles in graph 42
    [A123] Processing graph 43: extracting triangles...
    [A123]   Found 20598 triangles in graph 43
    [A123] Processing graph 44: extracting triangles...
    [A123]   Found 135 triangles in graph 44
    [A123] Processing graph 45: extracting triangles...
    [A123]   Found 2451 triangles in graph 45
    [A123] Processing graph 46: extracting triangles...
    [A123]   Found 8356 triangles in graph 46
    [A123] Processing graph 47: extracting triangles...
    [A123]   Found 231 triangles in graph 47
    [A123] Processing graph 48: extracting triangles...
    [A123]   Found 8334 triangles in graph 48
    [A123] Processing graph 49: extracting triangles...
    [A123]   Found 784 triangles in graph 49
    [A123] Processed 50/250 graphs (0s elapsed, ~2s remaining)...
    [A123] Processing graph 50: extracting triangles...
    [A123]   Found 5156 triangles in graph 50
    [A123] Processing graph 51: extracting triangles...
    [A123]   Found 4752 triangles in graph 51
    [A123] Processing graph 52: extracting triangles...
    [A123]   Found 31 triangles in graph 52
    [A123] Processing graph 53: extracting triangles...
    [A123]   Found 1268 triangles in graph 53
    [A123] Processing graph 54: extracting triangles...
    [A123]   Found 831 triangles in graph 54
    [A123] Processing graph 55: extracting triangles...
    [A123]   Found 110 triangles in graph 55
    [A123] Processing graph 56: extracting triangles...
    [A123]   Found 4363 triangles in graph 56
    [A123] Processing graph 47: extracting triangles...
    [A123]   Found 231 triangles in graph 47
    [A123] Processing graph 48: extracting triangles...
    [A123]   Found 8334 triangles in graph 48
    [A123] Processing graph 49: extracting triangles...
    [A123]   Found 784 triangles in graph 49
    [A123] Processed 50/250 graphs (0s elapsed, ~2s remaining)...
    [A123] Processing graph 50: extracting triangles...
    [A123]   Found 5156 triangles in graph 50
    [A123] Processing graph 51: extracting triangles...
    [A123]   Found 4752 triangles in graph 51
    [A123] Processing graph 52: extracting triangles...
    [A123]   Found 31 triangles in graph 52
    [A123] Processing graph 53: extracting triangles...
    [A123]   Found 1268 triangles in graph 53
    [A123] Processing graph 54: extracting triangles...
    [A123]   Found 831 triangles in graph 54
    [A123] Processing graph 55: extracting triangles...
    [A123]   Found 110 triangles in graph 55
    [A123] Processing graph 56: extracting triangles...
    [A123]   Found 4363 triangles in graph 56
    [A123] Processing graph 57: extracting triangles...
    [A123]   Found 729 triangles in graph 57
    [A123] Processing graph 58: extracting triangles...
    [A123]   Found 13722 triangles in graph 58
    [A123] Processing graph 59: extracting triangles...
    [A123]   Found 3316 triangles in graph 59
    [A123] Processed 60/250 graphs (1s elapsed, ~3s remaining)...
    [A123] Processing graph 60: extracting triangles...
    [A123]   Found 123 triangles in graph 60
    [A123] Processing graph 61: extracting triangles...
    [A123]   Found 432 triangles in graph 61
    [A123] Processing graph 62: extracting triangles...
    [A123]   Found 1163 triangles in graph 62
    [A123] Processing graph 63: extracting triangles...
    [A123]   Found 2 triangles in graph 63
    [A123] Processing graph 64: extracting triangles...
    [A123]   Found 2972 triangles in graph 64
    [A123] Processing graph 65: extracting triangles...
    [A123]   Found 589 triangles in graph 65
    [A123] Processing graph 66: extracting triangles...
    [A123]   Found 4603 triangles in graph 66
    [A123] Processing graph 57: extracting triangles...
    [A123]   Found 729 triangles in graph 57
    [A123] Processing graph 58: extracting triangles...
    [A123]   Found 13722 triangles in graph 58
    [A123] Processing graph 59: extracting triangles...
    [A123]   Found 3316 triangles in graph 59
    [A123] Processed 60/250 graphs (1s elapsed, ~3s remaining)...
    [A123] Processing graph 60: extracting triangles...
    [A123]   Found 123 triangles in graph 60
    [A123] Processing graph 61: extracting triangles...
    [A123]   Found 432 triangles in graph 61
    [A123] Processing graph 62: extracting triangles...
    [A123]   Found 1163 triangles in graph 62
    [A123] Processing graph 63: extracting triangles...
    [A123]   Found 2 triangles in graph 63
    [A123] Processing graph 64: extracting triangles...
    [A123]   Found 2972 triangles in graph 64
    [A123] Processing graph 65: extracting triangles...
    [A123]   Found 589 triangles in graph 65
    [A123] Processing graph 66: extracting triangles...
    [A123]   Found 4603 triangles in graph 66
    [A123] Processing graph 67: extracting triangles...
    [A123]   Found 4089 triangles in graph 67
    [A123] Processing graph 68: extracting triangles...
    [A123]   Found 76 triangles in graph 68
    [A123] Processing graph 69: extracting triangles...
    [A123]   Found 1092 triangles in graph 69
    [A123] Processed 70/250 graphs (1s elapsed, ~2s remaining)...
    [A123] Processing graph 70: extracting triangles...
    [A123]   Found 976 triangles in graph 70
    [A123] Processing graph 71: extracting triangles...
    [A123]   Found 79 triangles in graph 71
    [A123] Processing graph 72: extracting triangles...
    [A123]   Found 3432 triangles in graph 72
    [A123] Processing graph 73: extracting triangles...
    [A123]   Found 193 triangles in graph 73
    [A123] Processing graph 74: extracting triangles...
    [A123]   Found 8775 triangles in graph 74
    [A123] Processing graph 75: extracting triangles...
    [A123]   Found 1758 triangles in graph 75
    [A123] Processing graph 76: extracting triangles...
    [A123]   Found 62 triangles in graph 76
    [A123] Processing graph 77: extracting triangles...
    [A123]   Found 746 triangles in graph 77
    [A123] Processing graph 78: extracting triangles...
    [A123]   Found 628 triangles in graph 78
    [A123] Processing graph 79: extracting triangles...
    [A123]   Found 117 triangles in graph 79
    [A123] Processed 80/250 graphs (1s elapsed, ~2s remaining)...
    [A123] Processing graph 80: extracting triangles...
    [A123]   Found 169 triangles in graph 80
    [A123] Processing graph 81: extracting triangles...
    [A123]   Found 36 triangles in graph 81
    [A123] Processing graph 82: extracting triangles...
    [A123] Processing graph 67: extracting triangles...
    [A123]   Found 4089 triangles in graph 67
    [A123] Processing graph 68: extracting triangles...
    [A123]   Found 76 triangles in graph 68
    [A123] Processing graph 69: extracting triangles...
    [A123]   Found 1092 triangles in graph 69
    [A123] Processed 70/250 graphs (1s elapsed, ~2s remaining)...
    [A123] Processing graph 70: extracting triangles...
    [A123]   Found 976 triangles in graph 70
    [A123] Processing graph 71: extracting triangles...
    [A123]   Found 79 triangles in graph 71
    [A123] Processing graph 72: extracting triangles...
    [A123]   Found 3432 triangles in graph 72
    [A123] Processing graph 73: extracting triangles...
    [A123]   Found 193 triangles in graph 73
    [A123] Processing graph 74: extracting triangles...
    [A123]   Found 8775 triangles in graph 74
    [A123] Processing graph 75: extracting triangles...
    [A123]   Found 1758 triangles in graph 75
    [A123] Processing graph 76: extracting triangles...
    [A123]   Found 62 triangles in graph 76
    [A123] Processing graph 77: extracting triangles...
    [A123]   Found 746 triangles in graph 77
    [A123] Processing graph 78: extracting triangles...
    [A123]   Found 628 triangles in graph 78
    [A123] Processing graph 79: extracting triangles...
    [A123]   Found 117 triangles in graph 79
    [A123] Processed 80/250 graphs (1s elapsed, ~2s remaining)...
    [A123] Processing graph 80: extracting triangles...
    [A123]   Found 169 triangles in graph 80
    [A123] Processing graph 81: extracting triangles...
    [A123]   Found 36 triangles in graph 81
    [A123] Processing graph 82: extracting triangles...
    [A123]   Found 25415 triangles in graph 82
    [A123] Processing graph 83: extracting triangles...
    [A123]   Found 1636 triangles in graph 83
    [A123] Processing graph 84: extracting triangles...
    [A123]   Found 1006 triangles in graph 84
    [A123] Processing graph 85: extracting triangles...
    [A123]   Found 141 triangles in graph 85
    [A123] Processing graph 86: extracting triangles...
    [A123]   Found 21 triangles in graph 86
    [A123] Processing graph 87: extracting triangles...
    [A123]   Found 41 triangles in graph 87
    [A123] Processing graph 88: extracting triangles...
    [A123]   Found 37 triangles in graph 88
    [A123] Processing graph 89: extracting triangles...
    [A123]   Found 7975 triangles in graph 89
    [A123] Processed 90/250 graphs (1s elapsed, ~2s remaining)...
    [A123] Processing graph 90: extracting triangles...
    [A123]   Found 3172 triangles in graph 90
    [A123] Processing graph 91: extracting triangles...
    [A123]   Found 242 triangles in graph 91
    [A123] Processing graph 92: extracting triangles...
    [A123]   Found 145 triangles in graph 92
    [A123] Processing graph 93: extracting triangles...
    [A123]   Found 16 triangles in graph 93
    [A123] Processing graph 94: extracting triangles...
    [A123]   Found 77 triangles in graph 94
    [A123] Processing graph 95: extracting triangles...
    [A123]   Found 2 triangles in graph 95
    [A123] Processing graph 96: extracting triangles...
    [A123]   Found 25415 triangles in graph 82
    [A123] Processing graph 83: extracting triangles...
    [A123]   Found 1636 triangles in graph 83
    [A123] Processing graph 84: extracting triangles...
    [A123]   Found 1006 triangles in graph 84
    [A123] Processing graph 85: extracting triangles...
    [A123]   Found 141 triangles in graph 85
    [A123] Processing graph 86: extracting triangles...
    [A123]   Found 21 triangles in graph 86
    [A123] Processing graph 87: extracting triangles...
    [A123]   Found 41 triangles in graph 87
    [A123] Processing graph 88: extracting triangles...
    [A123]   Found 37 triangles in graph 88
    [A123] Processing graph 89: extracting triangles...
    [A123]   Found 7975 triangles in graph 89
    [A123] Processed 90/250 graphs (1s elapsed, ~2s remaining)...
    [A123] Processing graph 90: extracting triangles...
    [A123]   Found 3172 triangles in graph 90
    [A123] Processing graph 91: extracting triangles...
    [A123]   Found 242 triangles in graph 91
    [A123] Processing graph 92: extracting triangles...
    [A123]   Found 145 triangles in graph 92
    [A123] Processing graph 93: extracting triangles...
    [A123]   Found 16 triangles in graph 93
    [A123] Processing graph 94: extracting triangles...
    [A123]   Found 77 triangles in graph 94
    [A123] Processing graph 95: extracting triangles...
    [A123]   Found 2 triangles in graph 95
    [A123] Processing graph 96: extracting triangles...
    [A123]   Found 14369 triangles in graph 96
    [A123] Processing graph 97: extracting triangles...
    [A123]   Found 3609 triangles in graph 97
    [A123] Processing graph 98: extracting triangles...
    [A123]   Found 676 triangles in graph 98
    [A123] Processing graph 99: extracting triangles...
    [A123]   Found 88 triangles in graph 99
    [A123] Processed 100/250 graphs (1s elapsed, ~2s remaining)...
    [A123] Processing graph 100: extracting triangles...
    [A123]   Found 32 triangles in graph 100
    [A123] Processing graph 101: extracting triangles...
    [A123]   Found 61 triangles in graph 101
    [A123] Processing graph 102: extracting triangles...
    [A123]   Found 31 triangles in graph 102
    [A123] Processing graph 103: extracting triangles...
    [A123]   Found 19245 triangles in graph 103
    [A123] Processing graph 104: extracting triangles...
    [A123]   Found 14369 triangles in graph 96
    [A123] Processing graph 97: extracting triangles...
    [A123]   Found 3609 triangles in graph 97
    [A123] Processing graph 98: extracting triangles...
    [A123]   Found 676 triangles in graph 98
    [A123] Processing graph 99: extracting triangles...
    [A123]   Found 88 triangles in graph 99
    [A123] Processed 100/250 graphs (1s elapsed, ~2s remaining)...
    [A123] Processing graph 100: extracting triangles...
    [A123]   Found 32 triangles in graph 100
    [A123] Processing graph 101: extracting triangles...
    [A123]   Found 61 triangles in graph 101
    [A123] Processing graph 102: extracting triangles...
    [A123]   Found 31 triangles in graph 102
    [A123] Processing graph 103: extracting triangles...
    [A123]   Found 19245 triangles in graph 103
    [A123] Processing graph 104: extracting triangles...
    [A123]   Found 6080 triangles in graph 104
    [A123] Processing graph 105: extracting triangles...
    [A123]   Found 574 triangles in graph 105
    [A123] Processing graph 106: extracting triangles...
    [A123]   Found 236 triangles in graph 106
    [A123] Processing graph 107: extracting triangles...
    [A123]   Found 23 triangles in graph 107
    [A123] Processing graph 108: extracting triangles...
    [A123]   Found 36 triangles in graph 108
    [A123] Processing graph 109: extracting triangles...
    [A123]   Found 6080 triangles in graph 104
    [A123] Processing graph 105: extracting triangles...
    [A123]   Found 574 triangles in graph 105
    [A123] Processing graph 106: extracting triangles...
    [A123]   Found 236 triangles in graph 106
    [A123] Processing graph 107: extracting triangles...
    [A123]   Found 23 triangles in graph 107
    [A123] Processing graph 108: extracting triangles...
    [A123]   Found 36 triangles in graph 108
    [A123] Processing graph 109: extracting triangles...
    [A123]   Found 36818 triangles in graph 109
    [A123] Processed 110/250 graphs (2s elapsed, ~2s remaining)...
    [A123] Processing graph 110: extracting triangles...
    [A123]   Found 11193 triangles in graph 110
    [A123] Processing graph 111: extracting triangles...
    [A123]   Found 474 triangles in graph 111
    [A123] Processing graph 112: extracting triangles...
    [A123]   Found 44 triangles in graph 112
    [A123] Processing graph 113: extracting triangles...
    [A123]   Found 12 triangles in graph 113
    [A123] Processing graph 114: extracting triangles...
    [A123]   Found 297 triangles in graph 114
    [A123] Processing graph 115: extracting triangles...
    [A123]   Found 232 triangles in graph 115
    [A123] Processing graph 116: extracting triangles...
    [A123]   Found 303 triangles in graph 116
    [A123] Processing graph 117: extracting triangles...
    [A123]   Found 44 triangles in graph 117
    [A123] Processing graph 118: extracting triangles...
    [A123]   Found 111 triangles in graph 118
    [A123] Processing graph 119: extracting triangles...
    [A123]   Found 463 triangles in graph 119
    [A123] Processed 120/250 graphs (2s elapsed, ~2s remaining)...
    [A123] Processing graph 120: extracting triangles...
    [A123]   Found 42 triangles in graph 120
    [A123] Processing graph 121: extracting triangles...
    [A123]   Found 159 triangles in graph 121
    [A123] Processing graph 122: extracting triangles...
    [A123]   Found 123 triangles in graph 122
    [A123] Processing graph 123: extracting triangles...
    [A123]   Found 12 triangles in graph 123
    [A123] Processing graph 124: extracting triangles...
    [A123]   Found 92 triangles in graph 124
    [A123] Processing graph 125: extracting triangles...
    [A123]   Found 177 triangles in graph 125
    [A123] Processing graph 126: extracting triangles...
    [A123]   Found 2 triangles in graph 126
    [A123] Processing graph 127: extracting triangles...
    [A123]   Found 4 triangles in graph 127
    [A123] Processing graph 128: extracting triangles...
    [A123]   Found 67 triangles in graph 128
    [A123] Processing graph 129: extracting triangles...
    [A123]   Found 17 triangles in graph 129
    [A123] Processed 130/250 graphs (2s elapsed, ~2s remaining)...
    [A123] Processing graph 130: extracting triangles...
    [A123]   Found 27 triangles in graph 130
    [A123] Processing graph 131: extracting triangles...
    [A123]   Found 36 triangles in graph 131
    [A123] Processing graph 132: extracting triangles...
    [A123]   Found 55 triangles in graph 132
    [A123] Processing graph 133: extracting triangles...
    [A123]   Found 25 triangles in graph 133
    [A123] Processing graph 134: extracting triangles...
    [A123]   Found 1 triangles in graph 134
    [A123] Processing graph 135: extracting triangles...
    [A123]   Found 1 triangles in graph 135
    [A123] Processing graph 136: extracting triangles...
    [A123]   Found 11 triangles in graph 136
    [A123] Processing graph 137: extracting triangles...
    [A123]   Found 3 triangles in graph 137
    [A123] Processing graph 138: extracting triangles...
    [A123]   Found 14 triangles in graph 138
    [A123] Processing graph 139: extracting triangles...
    [A123]   Found 53 triangles in graph 139
    [A123] Processed 140/250 graphs (2s elapsed, ~1s remaining)...
    [A123] Processing graph 140: extracting triangles...
    [A123]   Found 6 triangles in graph 140
    [A123] Processing graph 141: extracting triangles...
    [A123]   Found 11 triangles in graph 141
    [A123] Processing graph 142: extracting triangles...
    [A123]   Found 52 triangles in graph 142
    [A123] Processing graph 143: extracting triangles...
    [A123]   Found 8 triangles in graph 143
    [A123] Processing graph 144: extracting triangles...
    [A123]   Found 6 triangles in graph 144
    [A123] Processing graph 145: extracting triangles...
    [A123]   Found 130 triangles in graph 145
    [A123] Processing graph 146: extracting triangles...
    [A123]   Found 317 triangles in graph 146
    [A123] Processing graph 147: extracting triangles...
    [A123]   Found 60 triangles in graph 147
    [A123] Processing graph 148: extracting triangles...
    [A123]   Found 10 triangles in graph 148
    [A123] Processing graph 149: extracting triangles...
    [A123]   Found 123 triangles in graph 149
    [A123] Processed 150/250 graphs (2s elapsed, ~1s remaining)...
    [A123] Processing graph 150: extracting triangles...
    [A123]   Found 154 triangles in graph 150
    [A123] Processing graph 151: extracting triangles...
    [A123]   Found 13 triangles in graph 151
    [A123] Processing graph 152: extracting triangles...
    [A123]   Found 7 triangles in graph 152
    [A123] Processing graph 153: extracting triangles...
    [A123]   Found 5 triangles in graph 153
    [A123] Processing graph 154: extracting triangles...
    [A123]   Found 119 triangles in graph 154
    [A123] Processing graph 155: extracting triangles...
    [A123]   Found 160 triangles in graph 155
    [A123] Processing graph 156: extracting triangles...
    [A123]   Found 15 triangles in graph 156
    [A123] Processing graph 157: extracting triangles...
    [A123]   Found 10 triangles in graph 157
    [A123] Processing graph 158: extracting triangles...
    [A123]   Found 8 triangles in graph 158
    [A123] Processing graph 159: extracting triangles...
    [A123]   Found 30 triangles in graph 159
    [A123] Processed 160/250 graphs (2s elapsed, ~1s remaining)...
    [A123] Processing graph 160: extracting triangles...
    [A123]   Found 13 triangles in graph 160
    [A123] Processing graph 161: extracting triangles...
    [A123]   Found 183 triangles in graph 161
    [A123] Processing graph 162: extracting triangles...
    [A123]   Found 177 triangles in graph 162
    [A123] Processing graph 163: extracting triangles...
    [A123]   Found 0 triangles in graph 163
    [A123] Processing graph 164: extracting triangles...
    [A123]   Found 53 triangles in graph 164
    [A123] Processing graph 165: extracting triangles...
    [A123]   Found 21 triangles in graph 165
    [A123] Processing graph 166: extracting triangles...
    [A123]   Found 21 triangles in graph 166
    [A123] Processing graph 167: extracting triangles...
    [A123]   Found 59 triangles in graph 167
    [A123] Processing graph 168: extracting triangles...
    [A123]   Found 52 triangles in graph 168
    [A123] Processing graph 169: extracting triangles...
    [A123]   Found 7 triangles in graph 169
    [A123] Processed 170/250 graphs (2s elapsed, ~1s remaining)...
    [A123] Processing graph 170: extracting triangles...
    [A123]   Found 35 triangles in graph 170
    [A123] Processing graph 171: extracting triangles...
    [A123]   Found 916 triangles in graph 171
    [A123] Processing graph 172: extracting triangles...
    [A123]   Found 142 triangles in graph 172
    [A123] Processing graph 173: extracting triangles...
    [A123]   Found 23 triangles in graph 173
    [A123] Processing graph 174: extracting triangles...
    [A123]   Found 467 triangles in graph 174
    [A123]   Found 36818 triangles in graph 109
    [A123] Processed 110/250 graphs (2s elapsed, ~2s remaining)...
    [A123] Processing graph 110: extracting triangles...
    [A123]   Found 11193 triangles in graph 110
    [A123] Processing graph 111: extracting triangles...
    [A123]   Found 474 triangles in graph 111
    [A123] Processing graph 112: extracting triangles...
    [A123]   Found 44 triangles in graph 112
    [A123] Processing graph 113: extracting triangles...
    [A123]   Found 12 triangles in graph 113
    [A123] Processing graph 114: extracting triangles...
    [A123]   Found 297 triangles in graph 114
    [A123] Processing graph 115: extracting triangles...
    [A123]   Found 232 triangles in graph 115
    [A123] Processing graph 116: extracting triangles...
    [A123]   Found 303 triangles in graph 116
    [A123] Processing graph 117: extracting triangles...
    [A123]   Found 44 triangles in graph 117
    [A123] Processing graph 118: extracting triangles...
    [A123]   Found 111 triangles in graph 118
    [A123] Processing graph 119: extracting triangles...
    [A123]   Found 463 triangles in graph 119
    [A123] Processed 120/250 graphs (2s elapsed, ~2s remaining)...
    [A123] Processing graph 120: extracting triangles...
    [A123]   Found 42 triangles in graph 120
    [A123] Processing graph 121: extracting triangles...
    [A123]   Found 159 triangles in graph 121
    [A123] Processing graph 122: extracting triangles...
    [A123]   Found 123 triangles in graph 122
    [A123] Processing graph 123: extracting triangles...
    [A123]   Found 12 triangles in graph 123
    [A123] Processing graph 124: extracting triangles...
    [A123]   Found 92 triangles in graph 124
    [A123] Processing graph 125: extracting triangles...
    [A123]   Found 177 triangles in graph 125
    [A123] Processing graph 126: extracting triangles...
    [A123]   Found 2 triangles in graph 126
    [A123] Processing graph 127: extracting triangles...
    [A123]   Found 4 triangles in graph 127
    [A123] Processing graph 128: extracting triangles...
    [A123]   Found 67 triangles in graph 128
    [A123] Processing graph 129: extracting triangles...
    [A123]   Found 17 triangles in graph 129
    [A123] Processed 130/250 graphs (2s elapsed, ~2s remaining)...
    [A123] Processing graph 130: extracting triangles...
    [A123]   Found 27 triangles in graph 130
    [A123] Processing graph 131: extracting triangles...
    [A123]   Found 36 triangles in graph 131
    [A123] Processing graph 132: extracting triangles...
    [A123]   Found 55 triangles in graph 132
    [A123] Processing graph 133: extracting triangles...
    [A123]   Found 25 triangles in graph 133
    [A123] Processing graph 134: extracting triangles...
    [A123]   Found 1 triangles in graph 134
    [A123] Processing graph 135: extracting triangles...
    [A123]   Found 1 triangles in graph 135
    [A123] Processing graph 136: extracting triangles...
    [A123]   Found 11 triangles in graph 136
    [A123] Processing graph 137: extracting triangles...
    [A123]   Found 3 triangles in graph 137
    [A123] Processing graph 138: extracting triangles...
    [A123]   Found 14 triangles in graph 138
    [A123] Processing graph 139: extracting triangles...
    [A123]   Found 53 triangles in graph 139
    [A123] Processed 140/250 graphs (2s elapsed, ~1s remaining)...
    [A123] Processing graph 140: extracting triangles...
    [A123]   Found 6 triangles in graph 140
    [A123] Processing graph 141: extracting triangles...
    [A123]   Found 11 triangles in graph 141
    [A123] Processing graph 142: extracting triangles...
    [A123]   Found 52 triangles in graph 142
    [A123] Processing graph 143: extracting triangles...
    [A123]   Found 8 triangles in graph 143
    [A123] Processing graph 144: extracting triangles...
    [A123]   Found 6 triangles in graph 144
    [A123] Processing graph 145: extracting triangles...
    [A123]   Found 130 triangles in graph 145
    [A123] Processing graph 146: extracting triangles...
    [A123]   Found 317 triangles in graph 146
    [A123] Processing graph 147: extracting triangles...
    [A123]   Found 60 triangles in graph 147
    [A123] Processing graph 148: extracting triangles...
    [A123]   Found 10 triangles in graph 148
    [A123] Processing graph 149: extracting triangles...
    [A123]   Found 123 triangles in graph 149
    [A123] Processed 150/250 graphs (2s elapsed, ~1s remaining)...
    [A123] Processing graph 150: extracting triangles...
    [A123]   Found 154 triangles in graph 150
    [A123] Processing graph 151: extracting triangles...
    [A123]   Found 13 triangles in graph 151
    [A123] Processing graph 152: extracting triangles...
    [A123]   Found 7 triangles in graph 152
    [A123] Processing graph 153: extracting triangles...
    [A123]   Found 5 triangles in graph 153
    [A123] Processing graph 154: extracting triangles...
    [A123]   Found 119 triangles in graph 154
    [A123] Processing graph 155: extracting triangles...
    [A123]   Found 160 triangles in graph 155
    [A123] Processing graph 156: extracting triangles...
    [A123]   Found 15 triangles in graph 156
    [A123] Processing graph 157: extracting triangles...
    [A123]   Found 10 triangles in graph 157
    [A123] Processing graph 158: extracting triangles...
    [A123]   Found 8 triangles in graph 158
    [A123] Processing graph 159: extracting triangles...
    [A123]   Found 30 triangles in graph 159
    [A123] Processed 160/250 graphs (2s elapsed, ~1s remaining)...
    [A123] Processing graph 160: extracting triangles...
    [A123]   Found 13 triangles in graph 160
    [A123] Processing graph 161: extracting triangles...
    [A123]   Found 183 triangles in graph 161
    [A123] Processing graph 162: extracting triangles...
    [A123]   Found 177 triangles in graph 162
    [A123] Processing graph 163: extracting triangles...
    [A123]   Found 0 triangles in graph 163
    [A123] Processing graph 164: extracting triangles...
    [A123]   Found 53 triangles in graph 164
    [A123] Processing graph 165: extracting triangles...
    [A123]   Found 21 triangles in graph 165
    [A123] Processing graph 166: extracting triangles...
    [A123]   Found 21 triangles in graph 166
    [A123] Processing graph 167: extracting triangles...
    [A123]   Found 59 triangles in graph 167
    [A123] Processing graph 168: extracting triangles...
    [A123]   Found 52 triangles in graph 168
    [A123] Processing graph 169: extracting triangles...
    [A123]   Found 7 triangles in graph 169
    [A123] Processed 170/250 graphs (2s elapsed, ~1s remaining)...
    [A123] Processing graph 170: extracting triangles...
    [A123]   Found 35 triangles in graph 170
    [A123] Processing graph 171: extracting triangles...
    [A123]   Found 916 triangles in graph 171
    [A123] Processing graph 172: extracting triangles...
    [A123]   Found 142 triangles in graph 172
    [A123] Processing graph 173: extracting triangles...
    [A123]   Found 23 triangles in graph 173
    [A123] Processing graph 174: extracting triangles...
    [A123]   Found 467 triangles in graph 174
    [A123] Processing graph 175: extracting triangles...
    [A123]   Found 27 triangles in graph 175
    [A123] Processing graph 176: extracting triangles...
    [A123]   Found 148 triangles in graph 176
    [A123] Processing graph 177: extracting triangles...
    [A123]   Found 65 triangles in graph 177
    [A123] Processing graph 178: extracting triangles...
    [A123]   Found 21 triangles in graph 178
    [A123] Processing graph 179: extracting triangles...
    [A123]   Found 80 triangles in graph 179
    [A123] Processed 180/250 graphs (2s elapsed, ~0s remaining)...
    [A123] Processing graph 180: extracting triangles...
    [A123]   Found 20 triangles in graph 180
    [A123] Processing graph 181: extracting triangles...
    [A123]   Found 10 triangles in graph 181
    [A123] Processing graph 182: extracting triangles...
    [A123]   Found 35 triangles in graph 182
    [A123] Processing graph 183: extracting triangles...
    [A123]   Found 32 triangles in graph 183
    [A123] Processing graph 184: extracting triangles...
    [A123]   Found 197 triangles in graph 184
    [A123] Processing graph 185: extracting triangles...
    [A123]   Found 26 triangles in graph 185
    [A123] Processing graph 186: extracting triangles...
    [A123]   Found 76 triangles in graph 186
    [A123] Processing graph 187: extracting triangles...
    [A123]   Found 54 triangles in graph 187
    [A123] Processing graph 188: extracting triangles...
    [A123]   Found 31 triangles in graph 188
    [A123] Processing graph 189: extracting triangles...
    [A123]   Found 22 triangles in graph 189
    [A123] Processed 190/250 graphs (2s elapsed, ~0s remaining)...
    [A123] Processing graph 190: extracting triangles...
    [A123]   Found 15 triangles in graph 190
    [A123] Processing graph 191: extracting triangles...
    [A123]   Found 107 triangles in graph 191
    [A123] Processing graph 192: extracting triangles...
    [A123]   Found 544 triangles in graph 192
    [A123] Processing graph 193: extracting triangles...
    [A123]   Found 29 triangles in graph 193
    [A123] Processing graph 194: extracting triangles...
    [A123]   Found 226 triangles in graph 194
    [A123] Processing graph 195: extracting triangles...
    [A123]   Found 298 triangles in graph 195
    [A123] Processing graph 196: extracting triangles...
    [A123]   Found 9 triangles in graph 196
    [A123] Processing graph 197: extracting triangles...
    [A123]   Found 5 triangles in graph 197
    [A123] Processing graph 198: extracting triangles...
    [A123]   Found 11 triangles in graph 198
    [A123] Processing graph 199: extracting triangles...
    [A123]   Found 166 triangles in graph 199
    [A123] Processed 200/250 graphs (2s elapsed, ~0s remaining)...
    [A123] Processing graph 200: extracting triangles...
    [A123]   Found 162 triangles in graph 200
    [A123] Processing graph 201: extracting triangles...
    [A123]   Found 88 triangles in graph 201
    [A123] Processing graph 202: extracting triangles...
    [A123]   Found 139 triangles in graph 202
    [A123] Processing graph 203: extracting triangles...
    [A123]   Found 49 triangles in graph 203
    [A123] Processing graph 204: extracting triangles...
    [A123]   Found 15 triangles in graph 204
    [A123] Processing graph 205: extracting triangles...
    [A123]   Found 114 triangles in graph 205
    [A123] Processing graph 206: extracting triangles...
    [A123]   Found 792 triangles in graph 206
    [A123] Processing graph 207: extracting triangles...
    [A123]   Found 65 triangles in graph 207
    [A123] Processing graph 208: extracting triangles...
    [A123]   Found 222 triangles in graph 208
    [A123] Processing graph 209: extracting triangles...
    [A123]   Found 95 triangles in graph 209
    [A123] Processed 210/250 graphs (2s elapsed, ~0s remaining)...
    [A123] Processing graph 210: extracting triangles...
    [A123]   Found 58 triangles in graph 210
    [A123] Processing graph 211: extracting triangles...
    [A123]   Found 51 triangles in graph 211
    [A123] Processing graph 212: extracting triangles...
    [A123]   Found 5 triangles in graph 212
    [A123] Processing graph 213: extracting triangles...
    [A123]   Found 25 triangles in graph 213
    [A123] Processing graph 214: extracting triangles...
    [A123]   Found 26 triangles in graph 214
    [A123] Processing graph 215: extracting triangles...
    [A123]   Found 6 triangles in graph 215
    [A123] Processing graph 216: extracting triangles...
    [A123]   Found 38 triangles in graph 216
    [A123] Processing graph 217: extracting triangles...
    [A123]   Found 27 triangles in graph 217
    [A123] Processing graph 218: extracting triangles...
    [A123]   Found 210 triangles in graph 218
    [A123] Processing graph 219: extracting triangles...
    [A123]   Found 24 triangles in graph 219
    [A123] Processed 220/250 graphs (2s elapsed, ~0s remaining)...
    [A123] Processing graph 220: extracting triangles...
    [A123]   Found 135 triangles in graph 220
    [A123] Processing graph 221: extracting triangles...
    [A123]   Found 136 triangles in graph 221
    [A123] Processing graph 222: extracting triangles...
    [A123]   Found 64 triangles in graph 222
    [A123] Processing graph 223: extracting triangles...
    [A123]   Found 11 triangles in graph 223
    [A123] Processing graph 224: extracting triangles...
    [A123]   Found 43 triangles in graph 224
    [A123] Processing graph 225: extracting triangles...
    [A123]   Found 20 triangles in graph 225
    [A123] Processing graph 226: extracting triangles...
    [A123]   Found 177 triangles in graph 226
    [A123] Processing graph 227: extracting triangles...
    [A123]   Found 19 triangles in graph 227
    [A123] Processing graph 228: extracting triangles...
    [A123]   Found 92 triangles in graph 228
    [A123] Processing graph 229: extracting triangles...
    [A123]   Found 12 triangles in graph 229
    [A123] Processed 230/250 graphs (2s elapsed, ~0s remaining)...
    [A123] Processing graph 230: extracting triangles...
    [A123]   Found 418 triangles in graph 230
    [A123] Processing graph 231: extracting triangles...
    [A123]   Found 86 triangles in graph 231
    [A123] Processing graph 232: extracting triangles...
    [A123]   Found 152 triangles in graph 232
    [A123] Processing graph 233: extracting triangles...
    [A123]   Found 387 triangles in graph 233
    [A123] Processing graph 234: extracting triangles...
    [A123]   Found 4 triangles in graph 234
    [A123] Processing graph 235: extracting triangles...
    [A123]   Found 283 triangles in graph 235
    [A123] Processing graph 236: extracting triangles...
    [A123]   Found 44 triangles in graph 236
    [A123] Processing graph 237: extracting triangles...
    [A123]   Found 189 triangles in graph 237
    [A123] Processing graph 238: extracting triangles...
    [A123]   Found 120 triangles in graph 238
    [A123] Processing graph 239: extracting triangles...
    [A123]   Found 63 triangles in graph 239
    [A123] Processed 240/250 graphs (2s elapsed, ~0s remaining)...
    [A123] Processing graph 240: extracting triangles...
    [A123]   Found 868 triangles in graph 240
    [A123] Processing graph 241: extracting triangles...
    [A123]   Found 920 triangles in graph 241
    [A123] Processing graph 242: extracting triangles...
    [A123]   Found 680 triangles in graph 242
    [A123] Processing graph 243: extracting triangles...
    [A123]   Found 59 triangles in graph 243
    [A123] Processing graph 244: extracting triangles...
    [A123]   Found 266 triangles in graph 244
    [A123] Processing graph 245: extracting triangles...
    [A123]   Found 1680 triangles in graph 245
    [A123] Processing graph 246: extracting triangles...
    [A123]   Found 3037 triangles in graph 246
    [A123] Processing graph 247: extracting triangles...
    [A123]   Found 654 triangles in graph 247
    [A123] Processing graph 248: extracting triangles...
    [A123]   Found 2370 triangles in graph 248
    [A123] Processing graph 249: extracting triangles...
    [A123]   Found 17 triangles in graph 249
    [A123] Triangle extraction completed in 2s, found 335458 triangles
    [A123] Creating triangle common-neighbors task...
    [A123] Processing graph 175: extracting triangles...
    [A123]   Found 27 triangles in graph 175
    [A123] Processing graph 176: extracting triangles...
    [A123]   Found 148 triangles in graph 176
    [A123] Processing graph 177: extracting triangles...
    [A123]   Found 65 triangles in graph 177
    [A123] Processing graph 178: extracting triangles...
    [A123]   Found 21 triangles in graph 178
    [A123] Processing graph 179: extracting triangles...
    [A123]   Found 80 triangles in graph 179
    [A123] Processed 180/250 graphs (2s elapsed, ~0s remaining)...
    [A123] Processing graph 180: extracting triangles...
    [A123]   Found 20 triangles in graph 180
    [A123] Processing graph 181: extracting triangles...
    [A123]   Found 10 triangles in graph 181
    [A123] Processing graph 182: extracting triangles...
    [A123]   Found 35 triangles in graph 182
    [A123] Processing graph 183: extracting triangles...
    [A123]   Found 32 triangles in graph 183
    [A123] Processing graph 184: extracting triangles...
    [A123]   Found 197 triangles in graph 184
    [A123] Processing graph 185: extracting triangles...
    [A123]   Found 26 triangles in graph 185
    [A123] Processing graph 186: extracting triangles...
    [A123]   Found 76 triangles in graph 186
    [A123] Processing graph 187: extracting triangles...
    [A123]   Found 54 triangles in graph 187
    [A123] Processing graph 188: extracting triangles...
    [A123]   Found 31 triangles in graph 188
    [A123] Processing graph 189: extracting triangles...
    [A123]   Found 22 triangles in graph 189
    [A123] Processed 190/250 graphs (2s elapsed, ~0s remaining)...
    [A123] Processing graph 190: extracting triangles...
    [A123]   Found 15 triangles in graph 190
    [A123] Processing graph 191: extracting triangles...
    [A123]   Found 107 triangles in graph 191
    [A123] Processing graph 192: extracting triangles...
    [A123]   Found 544 triangles in graph 192
    [A123] Processing graph 193: extracting triangles...
    [A123]   Found 29 triangles in graph 193
    [A123] Processing graph 194: extracting triangles...
    [A123]   Found 226 triangles in graph 194
    [A123] Processing graph 195: extracting triangles...
    [A123]   Found 298 triangles in graph 195
    [A123] Processing graph 196: extracting triangles...
    [A123]   Found 9 triangles in graph 196
    [A123] Processing graph 197: extracting triangles...
    [A123]   Found 5 triangles in graph 197
    [A123] Processing graph 198: extracting triangles...
    [A123]   Found 11 triangles in graph 198
    [A123] Processing graph 199: extracting triangles...
    [A123]   Found 166 triangles in graph 199
    [A123] Processed 200/250 graphs (2s elapsed, ~0s remaining)...
    [A123] Processing graph 200: extracting triangles...
    [A123]   Found 162 triangles in graph 200
    [A123] Processing graph 201: extracting triangles...
    [A123]   Found 88 triangles in graph 201
    [A123] Processing graph 202: extracting triangles...
    [A123]   Found 139 triangles in graph 202
    [A123] Processing graph 203: extracting triangles...
    [A123]   Found 49 triangles in graph 203
    [A123] Processing graph 204: extracting triangles...
    [A123]   Found 15 triangles in graph 204
    [A123] Processing graph 205: extracting triangles...
    [A123]   Found 114 triangles in graph 205
    [A123] Processing graph 206: extracting triangles...
    [A123]   Found 792 triangles in graph 206
    [A123] Processing graph 207: extracting triangles...
    [A123]   Found 65 triangles in graph 207
    [A123] Processing graph 208: extracting triangles...
    [A123]   Found 222 triangles in graph 208
    [A123] Processing graph 209: extracting triangles...
    [A123]   Found 95 triangles in graph 209
    [A123] Processed 210/250 graphs (2s elapsed, ~0s remaining)...
    [A123] Processing graph 210: extracting triangles...
    [A123]   Found 58 triangles in graph 210
    [A123] Processing graph 211: extracting triangles...
    [A123]   Found 51 triangles in graph 211
    [A123] Processing graph 212: extracting triangles...
    [A123]   Found 5 triangles in graph 212
    [A123] Processing graph 213: extracting triangles...
    [A123]   Found 25 triangles in graph 213
    [A123] Processing graph 214: extracting triangles...
    [A123]   Found 26 triangles in graph 214
    [A123] Processing graph 215: extracting triangles...
    [A123]   Found 6 triangles in graph 215
    [A123] Processing graph 216: extracting triangles...
    [A123]   Found 38 triangles in graph 216
    [A123] Processing graph 217: extracting triangles...
    [A123]   Found 27 triangles in graph 217
    [A123] Processing graph 218: extracting triangles...
    [A123]   Found 210 triangles in graph 218
    [A123] Processing graph 219: extracting triangles...
    [A123]   Found 24 triangles in graph 219
    [A123] Processed 220/250 graphs (2s elapsed, ~0s remaining)...
    [A123] Processing graph 220: extracting triangles...
    [A123]   Found 135 triangles in graph 220
    [A123] Processing graph 221: extracting triangles...
    [A123]   Found 136 triangles in graph 221
    [A123] Processing graph 222: extracting triangles...
    [A123]   Found 64 triangles in graph 222
    [A123] Processing graph 223: extracting triangles...
    [A123]   Found 11 triangles in graph 223
    [A123] Processing graph 224: extracting triangles...
    [A123]   Found 43 triangles in graph 224
    [A123] Processing graph 225: extracting triangles...
    [A123]   Found 20 triangles in graph 225
    [A123] Processing graph 226: extracting triangles...
    [A123]   Found 177 triangles in graph 226
    [A123] Processing graph 227: extracting triangles...
    [A123]   Found 19 triangles in graph 227
    [A123] Processing graph 228: extracting triangles...
    [A123]   Found 92 triangles in graph 228
    [A123] Processing graph 229: extracting triangles...
    [A123]   Found 12 triangles in graph 229
    [A123] Processed 230/250 graphs (2s elapsed, ~0s remaining)...
    [A123] Processing graph 230: extracting triangles...
    [A123]   Found 418 triangles in graph 230
    [A123] Processing graph 231: extracting triangles...
    [A123]   Found 86 triangles in graph 231
    [A123] Processing graph 232: extracting triangles...
    [A123]   Found 152 triangles in graph 232
    [A123] Processing graph 233: extracting triangles...
    [A123]   Found 387 triangles in graph 233
    [A123] Processing graph 234: extracting triangles...
    [A123]   Found 4 triangles in graph 234
    [A123] Processing graph 235: extracting triangles...
    [A123]   Found 283 triangles in graph 235
    [A123] Processing graph 236: extracting triangles...
    [A123]   Found 44 triangles in graph 236
    [A123] Processing graph 237: extracting triangles...
    [A123]   Found 189 triangles in graph 237
    [A123] Processing graph 238: extracting triangles...
    [A123]   Found 120 triangles in graph 238
    [A123] Processing graph 239: extracting triangles...
    [A123]   Found 63 triangles in graph 239
    [A123] Processed 240/250 graphs (2s elapsed, ~0s remaining)...
    [A123] Processing graph 240: extracting triangles...
    [A123]   Found 868 triangles in graph 240
    [A123] Processing graph 241: extracting triangles...
    [A123]   Found 920 triangles in graph 241
    [A123] Processing graph 242: extracting triangles...
    [A123]   Found 680 triangles in graph 242
    [A123] Processing graph 243: extracting triangles...
    [A123]   Found 59 triangles in graph 243
    [A123] Processing graph 244: extracting triangles...
    [A123]   Found 266 triangles in graph 244
    [A123] Processing graph 245: extracting triangles...
    [A123]   Found 1680 triangles in graph 245
    [A123] Processing graph 246: extracting triangles...
    [A123]   Found 3037 triangles in graph 246
    [A123] Processing graph 247: extracting triangles...
    [A123]   Found 654 triangles in graph 247
    [A123] Processing graph 248: extracting triangles...
    [A123]   Found 2370 triangles in graph 248
    [A123] Processing graph 249: extracting triangles...
    [A123]   Found 17 triangles in graph 249
    [A123] Triangle extraction completed in 2s, found 335458 triangles
    [A123] Creating triangle common-neighbors task...
    [A123] Created 335458 triangle CN samples
    [A123] Collating 335458 triangle CN samples...
    [A123] Created 335458 triangle CN samples
    [A123] Collating 335458 triangle CN samples...
    [A123] Saving triangle CN dataset to data/a123_cortex_m/processed/data_triangles_common_neighbors.pt...
    [A123] Triangle CN dataset saved!
    [A123] Processing complete!
    [A123] Saving triangle CN dataset to data/a123_cortex_m/processed/data_triangles_common_neighbors.pt...
    [A123] Triangle CN dataset saved!
    [A123] Processing complete!


    Done!
    Processing...


    [A123 Loader] Loaded triangle common-neighbours task dataset
    Dataset loaded: 335458 samples
    Task 'triangle_common_neighbors' uses triangle-level features (no edge_index)
    Skipping lifting transformation for triangle data


    Done!


    Dataset splits created:
      Train: 167729 samples
      Val: 83864 samples
      Test: 83865 samples
    Datasets and datamodule ready



```python
def undersample_majority_class(dataset, target_samples_per_class=100, random_state=42):
    """
    Undersample all classes to a target number of samples per class.
    
    Parameters
    ----------
    dataset : DataloadDataset
        Dataset to undersample
    target_samples_per_class : int
        Target number of samples per class (default: 100)
    random_state : int
        Random seed for reproducibility
        
    Returns
    -------
    DataloadDataset
        Undersampled dataset
    """
    np.random.seed(random_state)
    
    # Handle DataloadDataset which returns (values, keys) tuples
    labels = []
    for item in dataset:
        # The dataset returns (values_list, keys_list)
        if isinstance(item, (list, tuple)) and len(item) == 2:
            values, keys = item
            # The 'y' label is the last value in the list
            y = values[-1]
        else:
            # Fallback: try to access .y attribute
            if hasattr(item, 'y'):
                y = item.y
            else:
                continue  # Skip if we can't extract label
        
        # Convert tensor to scalar
        if hasattr(y, 'item'):
            labels.append(int(y.item()))
        elif hasattr(y, '__len__') and len(y) == 1:
            # Single-element tensor or array
            labels.append(int(y[0]))
        else:
            labels.append(int(y))
    
    # Check if we extracted any labels
    if len(labels) == 0:
        raise ValueError(f"No labels extracted from dataset of size {len(dataset)}")
    
    labels = np.array(labels)
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    print(f"Original class distribution:")
    for label, count in zip(unique_labels, counts):
        print(f"  Class {label}: {count} samples")
    
    # Get indices for each class
    indices_by_class = {label: np.where(labels == label)[0] for label in unique_labels}
    
    # Undersample each class to target_samples_per_class (or fewer if class has fewer samples)
    undersampled_indices = []
    for label in unique_labels:
        indices = indices_by_class[label]
        # Select up to target_samples_per_class indices from this class
        actual_samples = min(len(indices), target_samples_per_class)
        selected = np.random.choice(indices, size=actual_samples, replace=False)
        undersampled_indices.extend(selected)
    
    # Shuffle the final indices
    undersampled_indices = np.random.permutation(undersampled_indices)
    
    # Create subset of dataset
    from torch.utils.data import Subset
    undersampled_dataset = Subset(dataset, undersampled_indices)
    
    # Get new label distribution
    new_labels = labels[undersampled_indices]
    new_unique, new_counts = np.unique(new_labels, return_counts=True)
    
    print(f"\nAfter undersampling to {target_samples_per_class} per class:")
    
    for label, count in zip(new_unique, new_counts):
        print(f"  Class {label}: {count} samples")
    
    imbalance_ratio_before = counts.max() / counts.min()
    imbalance_ratio_after = new_counts.max() / new_counts.min()
    print(f"\nImbalance ratio: {imbalance_ratio_before:.2f} → {imbalance_ratio_after:.2f}")
    print(f"Dataset size: {len(dataset)} → {len(undersampled_dataset)} samples\n")
    
    return undersampled_dataset

# Apply undersampling to training set
print("Undersampling training set...")
dataset_train = undersample_majority_class(dataset_train, target_samples_per_class=200, random_state=0)

# Optionally also undersample validation set for consistency
print("Undersampling validation set...")
dataset_val = undersample_majority_class(dataset_val, target_samples_per_class=200, random_state=0)

# Optionally also undersample test set for consistency
print("Undersampling test set...")
dataset_test = undersample_majority_class(dataset_test, target_samples_per_class=200, random_state=0)

# Recreate datamodule with undersampled datasets
datamodule = TBDataloader(dataset_train, dataset_val, dataset_test, batch_size=32)

print('Datasets rebalanced and datamodule recreated')
```

    Undersampling training set...
    Original class distribution:
      Class 0: 507 samples
      Class 1: 1119 samples
      Class 2: 1574 samples
      Class 3: 1804 samples
      Class 4: 1946 samples
      Class 5: 2081 samples
      Class 6: 2203 samples
      Class 7: 2157 samples
      Class 8: 154338 samples
    
    After undersampling to 200 per class:
      Class 0: 200 samples
      Class 1: 200 samples
      Class 2: 200 samples
      Class 3: 200 samples
      Class 4: 200 samples
      Class 5: 200 samples
      Class 6: 200 samples
      Class 7: 200 samples
      Class 8: 200 samples
    
    Imbalance ratio: 304.41 → 1.00
    Dataset size: 167729 → 1800 samples
    
    Undersampling validation set...
    Original class distribution:
      Class 0: 507 samples
      Class 1: 1119 samples
      Class 2: 1574 samples
      Class 3: 1804 samples
      Class 4: 1946 samples
      Class 5: 2081 samples
      Class 6: 2203 samples
      Class 7: 2157 samples
      Class 8: 154338 samples
    
    After undersampling to 200 per class:
      Class 0: 200 samples
      Class 1: 200 samples
      Class 2: 200 samples
      Class 3: 200 samples
      Class 4: 200 samples
      Class 5: 200 samples
      Class 6: 200 samples
      Class 7: 200 samples
      Class 8: 200 samples
    
    Imbalance ratio: 304.41 → 1.00
    Dataset size: 167729 → 1800 samples
    
    Undersampling validation set...
    Original class distribution:
      Class 0: 288 samples
      Class 1: 542 samples
      Class 2: 788 samples
      Class 3: 966 samples
      Class 4: 988 samples
      Class 5: 1043 samples
      Class 6: 1078 samples
      Class 7: 1116 samples
      Class 8: 77055 samples
    
    After undersampling to 200 per class:
      Class 0: 200 samples
      Class 1: 200 samples
      Class 2: 200 samples
      Class 3: 200 samples
      Class 4: 200 samples
      Class 5: 200 samples
      Class 6: 200 samples
      Class 7: 200 samples
      Class 8: 200 samples
    
    Imbalance ratio: 267.55 → 1.00
    Dataset size: 83864 → 1800 samples
    
    Undersampling test set...
    Original class distribution:
      Class 0: 288 samples
      Class 1: 542 samples
      Class 2: 788 samples
      Class 3: 966 samples
      Class 4: 988 samples
      Class 5: 1043 samples
      Class 6: 1078 samples
      Class 7: 1116 samples
      Class 8: 77055 samples
    
    After undersampling to 200 per class:
      Class 0: 200 samples
      Class 1: 200 samples
      Class 2: 200 samples
      Class 3: 200 samples
      Class 4: 200 samples
      Class 5: 200 samples
      Class 6: 200 samples
      Class 7: 200 samples
      Class 8: 200 samples
    
    Imbalance ratio: 267.55 → 1.00
    Dataset size: 83864 → 1800 samples
    
    Undersampling test set...
    Original class distribution:
      Class 0: 243 samples
      Class 1: 541 samples
      Class 2: 760 samples
      Class 3: 909 samples
      Class 4: 983 samples
      Class 5: 1023 samples
      Class 6: 1072 samples
      Class 7: 1171 samples
      Class 8: 77163 samples
    
    After undersampling to 200 per class:
      Class 0: 200 samples
      Class 1: 200 samples
      Class 2: 200 samples
      Class 3: 200 samples
      Class 4: 200 samples
      Class 5: 200 samples
      Class 6: 200 samples
      Class 7: 200 samples
      Class 8: 200 samples
    
    Imbalance ratio: 317.54 → 1.00
    Dataset size: 83865 → 1800 samples
    
    Datasets rebalanced and datamodule recreated
    Original class distribution:
      Class 0: 243 samples
      Class 1: 541 samples
      Class 2: 760 samples
      Class 3: 909 samples
      Class 4: 983 samples
      Class 5: 1023 samples
      Class 6: 1072 samples
      Class 7: 1171 samples
      Class 8: 77163 samples
    
    After undersampling to 200 per class:
      Class 0: 200 samples
      Class 1: 200 samples
      Class 2: 200 samples
      Class 3: 200 samples
      Class 4: 200 samples
      Class 5: 200 samples
      Class 6: 200 samples
      Class 7: 200 samples
      Class 8: 200 samples
    
    Imbalance ratio: 317.54 → 1.00
    Dataset size: 83865 → 1800 samples
    
    Datasets rebalanced and datamodule recreated


## 4) Backbone definition

We implement a tiny backbone as a `pl.LightningModule` which computes node and hyperedge features: $X_1 = B_1 dot X_0$ and applies two linear layers with ReLU.


```python
class MyBackbone(pl.LightningModule):
    def __init__(self, dim_hidden):
        super().__init__()
        self.linear_0 = torch.nn.Linear(dim_hidden, dim_hidden)
        self.linear_1 = torch.nn.Linear(dim_hidden, dim_hidden)

    def forward(self, batch):
        # batch.x_0: node features (dense tensor of shape [N, dim_hidden])
        # batch.incidence_hyperedges: sparse incidence matrix with shape [m, n] or [n, m] depending on preprocessor convention
        x_0 = batch.x_0
        incidence_hyperedges = getattr(batch, 'incidence_hyperedges', None)
        if incidence_hyperedges is None:
            # fallback: try incidence as batch.incidence if available
            incidence_hyperedges = getattr(batch, 'incidence', None)

        # compute hyperedge features X_1 = B_1 dot X_0 (we assume B_1 is sparse and transposed appropriately)
        x_1 = None
        if incidence_hyperedges is not None:
            try:
                x_1 = torch.sparse.mm(incidence_hyperedges, x_0)
            except Exception:
                # if orientation differs, try transpose
                x_1 = torch.sparse.mm(incidence_hyperedges.T, x_0)
        else:
            # no incidence available: create a zero hyperedge feature placeholder
            x_1 = torch.zeros_like(x_0)

        x_0 = self.linear_0(x_0)
        x_0 = torch.relu(x_0)

        x_1 = self.linear_1(x_1)
        x_1 = torch.relu(x_1)

        model_out = {'labels': batch.y, 'batch_0': getattr(batch, 'batch_0', None)}
        model_out['x_0'] = x_0
        model_out['hyperedge'] = x_1
        return model_out

print('Backbone defined')
```

    Backbone defined



```python
# 5) Model initialization (components)
backbone = MyBackbone(dim_hidden)
readout = PropagateSignalDown(**readout_config)
loss = TBLoss(**loss_config)
feature_encoder = AllCellFeatureEncoder(in_channels=[in_channels], out_channels=dim_hidden)
evaluator = TBEvaluator(**evaluator_config)
optimizer = TBOptimizer(**optimizer_config)

print('Components instantiated')
```

    Components instantiated



```python
# 6) Instantiate TBModel
model = TBModel(backbone=backbone,
                backbone_wrapper=None,
                readout=readout,
                loss=loss,
                feature_encoder=feature_encoder,
                evaluator=evaluator,
                optimizer=optimizer,
                compile=False)

# Print a short summary (repr) to verify construction
print(model)
```

    TBModel(backbone=MyBackbone(
      (linear_0): Linear(in_features=16, out_features=16, bias=True)
      (linear_1): Linear(in_features=16, out_features=16, bias=True)
    ), readout=PropagateSignalDown(num_cell_dimensions=0, self.hidden_dim=16, readout_name=PropagateSignalDown, loss=TBLoss(losses=[DatasetLoss(task=classification, loss_type=cross_entropy)]), feature_encoder=AllCellFeatureEncoder(in_channels=[3], out_channels=16, dimensions=range(0, 1)))



```python
# 7) Training loop (Lightning trainer)
# Suppress some warnings for cleaner output
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='torchmetrics')

trainer = pl.Trainer(
    max_epochs=50,  # reduced for faster iteration
    accelerator='cpu',
    enable_progress_bar=True,
    log_every_n_steps=1,
    enable_model_summary=False,  # skip the model summary printout
)
trainer.fit(model, datamodule)
train_metrics = trainer.callback_metrics

print('\nTraining finished. Collected metrics:')
for key, val in train_metrics.items():
    try:
        print(f'{key:25s} {float(val):.4f}')
    except Exception:
        print(key, val)
```

    GPU available: True (mps), used: False
    TPU available: False, using: 0 TPU cores
    HPU available: False, using: 0 HPUs
    /Users/mariayuffa/anaconda3/envs/tb3/lib/python3.11/site-packages/lightning/pytorch/trainer/setup.py:177: GPU available but not used. You can set it by doing `Trainer(accelerator='gpu')`.
    TPU available: False, using: 0 TPU cores
    HPU available: False, using: 0 HPUs
    /Users/mariayuffa/anaconda3/envs/tb3/lib/python3.11/site-packages/lightning/pytorch/trainer/setup.py:177: GPU available but not used. You can set it by doing `Trainer(accelerator='gpu')`.



    Sanity Checking: |          | 0/? [00:00<?, ?it/s]


    /Users/mariayuffa/anaconda3/envs/tb3/lib/python3.11/site-packages/lightning/pytorch/trainer/connectors/data_connector.py:424: The 'val_dataloader' does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` to `num_workers=9` in the `DataLoader` to improve performance.
    /Users/mariayuffa/anaconda3/envs/tb3/lib/python3.11/site-packages/lightning/pytorch/trainer/connectors/data_connector.py:424: The 'train_dataloader' does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` to `num_workers=9` in the `DataLoader` to improve performance.



    Training: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]


    `Trainer.fit` stopped: `max_epochs=50` reached.


    
    Training finished. Collected metrics:
    train/accuracy            0.1050
    train/f1                  0.0329
    train/precision           0.0284
    train/recall              0.1050
    val/loss                  2.1973
    val/accuracy              0.1111
    val/f1                    0.0222
    val/precision             0.0123
    val/recall                0.1111
    train/loss                2.1978



```python
# 8) Testing and printing metrics
trainer.test(model, datamodule)
test_metrics = trainer.callback_metrics
print('\nTest metrics:')
for key, val in test_metrics.items():
    try:
        print(f'{key:25s} {float(val):.4f}')
    except Exception:
        print(key, val)
```

    /Users/mariayuffa/anaconda3/envs/tb3/lib/python3.11/site-packages/lightning/pytorch/trainer/connectors/data_connector.py:424: The 'test_dataloader' does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` to `num_workers=9` in the `DataLoader` to improve performance.



    Testing: |          | 0/? [00:00<?, ?it/s]



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold">        Test metric        </span>┃<span style="font-weight: bold">       DataLoader 0        </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│<span style="color: #008080; text-decoration-color: #008080">       test/accuracy       </span>│<span style="color: #800080; text-decoration-color: #800080">    0.1111111119389534     </span>│
│<span style="color: #008080; text-decoration-color: #008080">          test/f1          </span>│<span style="color: #800080; text-decoration-color: #800080">    0.02222222276031971    </span>│
│<span style="color: #008080; text-decoration-color: #008080">         test/loss         </span>│<span style="color: #800080; text-decoration-color: #800080">     2.197321653366089     </span>│
│<span style="color: #008080; text-decoration-color: #008080">      test/precision       </span>│<span style="color: #800080; text-decoration-color: #800080">   0.012345679104328156    </span>│
│<span style="color: #008080; text-decoration-color: #008080">        test/recall        </span>│<span style="color: #800080; text-decoration-color: #800080">    0.1111111119389534     </span>│
└───────────────────────────┴───────────────────────────┘
</pre>



    
    Test metrics:
    test/loss                 2.1973
    test/accuracy             0.1111
    test/f1                   0.0222
    test/precision            0.0123
    test/recall               0.1111


## Running Other Tasks

To run a different task, modify the `TASK_NAME` variable in cell 4 (configurations) to one of:
- `graph_classification` (default): Predict frequency bin from graph structure
- `triangle_classification`: Classify topological role of triangles (9 embedding × weight classes)
- `triangle_common_neighbors`: Predict number of common neighbors for each triangle

Then re-run the configuration cell and subsequent cells. The dataset will automatically load the appropriate task variant, and the model will be configured with the correct number of output classes (9 for all tasks).

### Task Details:

**Task 1: Graph-level Classification**
- Input: Graph structure with node features (mean correlation, std correlation, noise diagonal)
- Output: Frequency bin (0-8) representing the best frequency
- Level: Graph-level prediction

**Task 2: Triangle Classification**
- Input: Topological features of triangles (3 edge weights from correlation matrix)
- Output: Triangle role classification (9 classes based on embedding × weight):
  - Embedding classes: Core (many common neighbors), Bridge (some), Isolated (few)
  - Weight classes: Strong (high correlation), Medium, Weak (low correlation)
- Level: Triangle (motif) level prediction

**Task 3: Triangle Common-Neighbors**
- Input: Triangle node degrees (structural features)
- Output: Number of common neighbors (0-8, mapping neighbors count to class)
- Level: Triangle (motif) level prediction
