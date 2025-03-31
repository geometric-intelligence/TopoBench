""" Test for the NeighborCellsLoader class."""
import os
import shutil
import rootutils
from hydra import compose
import torch

from topobenchmark.data.preprocessor import PreProcessor
from topobenchmark.data.utils.utils import load_manual_graph
from topobenchmark.data.batching import NeighborCellsLoader
from topobenchmark.run import initialize_hydra

initialize_hydra()

path = "./graph2simplicial_lifting/"
if os.path.isdir(path):
    shutil.rmtree(path)
cfg = compose(config_name="run.yaml", 
              overrides=["dataset=graph/manual_dataset", "model=simplicial/san"], 
              return_hydra_config=True)

data = load_manual_graph()
preprocessed_dataset = PreProcessor(data, path, cfg['transforms'])
data = preprocessed_dataset[0]

batch_size=2

rank = 0
n_cells = data[f'x_{rank}'].shape[0]
train_prop = 0.5
n_train = int(train_prop * n_cells)
train_mask = torch.zeros(n_cells, dtype=torch.bool)
train_mask[:n_train] = 1

y = torch.zeros(n_cells, dtype=torch.long)
data.y = y

loader = NeighborCellsLoader(data,
                             rank=rank,
                             num_neighbors=[-1],
                             input_nodes=train_mask,
                             batch_size=batch_size,
                             shuffle=False)
train_nodes = []
for batch in loader:
    train_nodes += [n for n in batch.n_id[:batch_size]]
for i in range(n_train):
    assert i in train_nodes

rank = 1
n_cells = data[f'x_{rank}'].shape[0]
train_prop = 0.5
n_train = int(train_prop * n_cells)
train_mask = torch.zeros(n_cells, dtype=torch.bool)
train_mask[:n_train] = 1

y = torch.zeros(n_cells, dtype=torch.long)
data.y = y

loader = NeighborCellsLoader(data,
                             rank=rank,
                             num_neighbors=[-1,-1],
                             input_nodes=train_mask,
                             batch_size=batch_size,
                             shuffle=False)

train_nodes = []
for batch in loader:
    train_nodes += [n for n in batch.n_id[:batch_size]]
for i in range(n_train):
    assert i in train_nodes
shutil.rmtree(path)


path = "./graph2hypergraph_lifting/"
if os.path.isdir(path):
    shutil.rmtree(path)
cfg = compose(config_name="run.yaml", 
              overrides=["dataset=graph/manual_dataset", "model=hypergraph/allsettransformer"], 
              return_hydra_config=True)

data = load_manual_graph()
preprocessed_dataset = PreProcessor(data, path, cfg['transforms'])
data = preprocessed_dataset[0]

batch_size=2

rank = 0
n_cells = data[f'x_0'].shape[0]
train_prop = 0.5
n_train = int(train_prop * n_cells)
train_mask = torch.zeros(n_cells, dtype=torch.bool)
train_mask[:n_train] = 1

y = torch.zeros(n_cells, dtype=torch.long)
data.y = y

loader = NeighborCellsLoader(data,
                             rank=rank,
                             num_neighbors=[-1],
                             input_nodes=train_mask,
                             batch_size=batch_size,
                             shuffle=False)
train_nodes = []
for batch in loader:
    train_nodes += [n for n in batch.n_id[:batch_size]]
for i in range(n_train):
    assert i in train_nodes

rank = 1
n_cells = data[f'x_hyperedges'].shape[0]
train_prop = 0.5
n_train = int(train_prop * n_cells)
train_mask = torch.zeros(n_cells, dtype=torch.bool)
train_mask[:n_train] = 1

y = torch.zeros(n_cells, dtype=torch.long)
data.y = y

loader = NeighborCellsLoader(data,
                             rank=rank,
                             num_neighbors=[-1,-1],
                             input_nodes=train_mask,
                             batch_size=batch_size,
                             shuffle=False)

train_nodes = []
for batch in loader:
    train_nodes += [n for n in batch.n_id[:batch_size]]
for i in range(n_train):
    assert i in train_nodes
shutil.rmtree(path)