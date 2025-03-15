<h2 align="center">
  <img src="resources/logo.jpg" width="800">
</h2>

<h3 align="center">
    A Comprehensive Benchmark Suite for Topological Deep Learning
</h3>

<p align="center">
Assess how your model compares against state-of-the-art topological neural networks.
</p>

<div align="center">

[![Lint](https://github.com/geometric-intelligence/TopoBench/actions/workflows/lint.yml/badge.svg)](https://github.com/geometric-intelligence/TopoBench/actions/workflows/lint.yml)
[![Test](https://github.com/geometric-intelligence/TopoBench/actions/workflows/test.yml/badge.svg)](https://github.com/geometric-intelligence/TopoBench/actions/workflows/test.yml)
[![Codecov](https://codecov.io/gh/geometric-intelligence/TopoBench/branch/main/graph/badge.svg)](https://app.codecov.io/gh/geometric-intelligence/TopoBench)
[![Docs](https://img.shields.io/badge/docs-website-brightgreen)](https://geometric-intelligence.github.io/topobench/index.html)
[![Python](https://img.shields.io/badge/python-3.10+-blue?logo=python)](https://www.python.org/)
[![license](https://badgen.net/github/license/geometric-intelligence/TopoBench?color=green)](https://github.com/geometric-intelligence/TopoBench/blob/main/LICENSE)
[![slack](https://img.shields.io/badge/chat-on%20slack-purple?logo=slack)](https://join.slack.com/t/geometric-intelligenceworkspace/shared_invite/zt-2k63sv99s-jbFMLtwzUCc8nt3sIRWjEw)


</div>

<p align="center">
  <a href="#pushpin-overview">Overview</a> •
  <a href="#jigsaw-get-started">Get Started</a> •
  <a href="#anchor-tutorials">Tutorials</a> •
  <a href="#gear-neural-networks">Neural Networks</a> •
  <a href="#rocket-liftings">Liftings</a> •
  <a href="#books-datasets">Datasets</a> •
  <a href="#mag-references">References</a> 
</p>



## :pushpin: Overview

`TopoBench` (TB) is a modular Python library designed to standardize benchmarking and accelerate research in Topological Deep Learning (TDL). In particular, TB allows to train and compare the performances of all sorts of Topological Neural Networks (TNNs) across the different topological domains, where by _topological domain_ we refer to a graph, a simplicial complex, a cellular complex, or a hypergraph. For detailed information, please refer to the [`TopoBench: A Framework for Benchmarking Topological Deep Learning`](https://arxiv.org/pdf/2406.06642) paper.

<p align="center">
  <img src="resources/workflow.jpg" width="700">
</p>

The main pipeline trains and evaluates a wide range of state-of-the-art TNNs and Graph Neural Networks (GNNs) (see <a href="#gear-neural-networks">:gear: Neural Networks</a>) on numerous and varied datasets and benchmark tasks (see <a href="#books-datasets">:books: Datasets</a> ). Additionally, the library offers the ability to transform, i.e. _lift_, each dataset from one topological domain to another (see <a href="#rocket-liftings">:rocket: Liftings</a>), enabling for the first time an exhaustive inter-domain comparison of TNNs.

## :jigsaw: Get Started

### Create Environment

If you do not have conda on your machine, please follow [their guide](https://docs.anaconda.com/free/miniconda/miniconda-install/) to install it. 

First, clone the `TopoBench` repository and set up a conda environment `tb` with python 3.11.3. 

```
git clone git@github.com:geometric-intelligence/topobench.git
cd TopoBench
conda create -n tb python=3.11.3
```

Next, check the CUDA version of your machine:
```
/usr/local/cuda/bin/nvcc --version
```
and ensure that it matches the CUDA version specified in the `env_setup.sh` file (`CUDA=cu121` by default). If it does not match, update `env_setup.sh` accordingly by changing both the `CUDA` and `TORCH` environment variables to compatible values as specified on [this website](https://github.com/pyg-team/pyg-lib).

Next, set up the environment with the following command.

```
source env_setup.sh
```
This command installs the `TopoBench` library and its dependencies. 

### Run Training Pipeline

Next, train the neural networks by running the following command:

```
python -m topobench 
```

Thanks to `hydra` implementation, one can easily override the default experiment configuration through the command line. For instance, the model and dataset can be selected as:

```
python -m topobench model=cell/cwn dataset=graph/MUTAG
```

**Remark:** By default, our pipeline identifies the source and destination topological domains, and applies a default lifting between them if required.

The same CLI override mechanism also applies when modifying more finer configurations within a `CONFIG GROUP`. Please, refer to the official [`hydra`documentation](https://hydra.cc/docs/intro/) for further details.



## :bike: Experiments Reproducibility
To reproduce Table 1 from the [`TopoBench: A Framework for Benchmarking Topological Deep Learning`](https://arxiv.org/pdf/2406.06642) paper, please run the following command:

```
bash scripts/reproduce.sh
```
**Remark:** We have additionally provided a public [W&B (Weights & Biases) project](https://wandb.ai/telyatnikov_sap/TopoBenchmark_main?nw=nwusertelyatnikov_sap) with logs for the corresponding runs (updated on June 11, 2024).


## :anchor: Tutorials

Explore our [tutorials](https://github.com/geometric-intelligence/TopoBench/tree/main/tutorials) for further details on how to add new datasets, transforms/liftings, and benchmark tasks. 

## :gear: Neural Networks

We list the neural networks trained and evaluated by `TopoBench`, organized by the topological domain over which they operate: graph, simplicial complex, cellular complex or hypergraph. Many of these neural networks were originally implemented in [`TopoModelX`](https://github.com/pyt-team/TopoModelX).


### Graphs
| Model | Reference |
| --- | --- |
| GAT | [Graph Attention Networks](https://openreview.net/pdf?id=rJXMpikCZ) |
| GIN | [How Powerful are Graph Neural Networks?](https://openreview.net/pdf?id=ryGs6iA5Km) |
| GCN | [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/pdf/1609.02907v4) |
| GraphMLP | [Graph-MLP: Node Classification without Message Passing in Graph](https://arxiv.org/pdf/2106.04051) |

### Simplicial complexes
| Model | Reference |
| --- | --- |
| SAN | [Simplicial Attention Neural Networks](https://arxiv.org/pdf/2203.07485) |
| SCCN | [Efficient Representation Learning for Higher-Order Data with Simplicial Complexes](https://openreview.net/pdf?id=nGqJY4DODN) |
| SCCNN | [Convolutional Learning on Simplicial Complexes](https://arxiv.org/pdf/2301.11163) |
| SCN | [Simplicial Complex Neural Networks](https://ieeexplore.ieee.org/document/10285604) |

### Cellular complexes
| Model | Reference |
| --- | --- |
| CAN | [Cell Attention Network](https://arxiv.org/pdf/2209.08179) |
| CCCN | Inspired by [A learning algorithm for computational connected cellular network](https://ieeexplore.ieee.org/document/1202221), implementation adapted from [Generalized Simplicial Attention Neural Networks](https://arxiv.org/abs/2309.02138)|
| CXN | [Cell Complex Neural Networks](https://openreview.net/pdf?id=6Tq18ySFpGU) |
| CWN | [Weisfeiler and Lehman Go Cellular: CW Networks](https://arxiv.org/pdf/2106.12575) |

### Hypergraphs
| Model | Reference |
| --- | --- |
| AllDeepSet | [You are AllSet: A Multiset Function Framework for Hypergraph Neural Networks](https://openreview.net/pdf?id=hpBTIv2uy_E) |
| AllSetTransformer | [You are AllSet: A Multiset Function Framework for Hypergraph Neural Networks](https://openreview.net/pdf?id=hpBTIv2uy_E) |
| EDGNN | [Equivariant Hypergraph Diffusion Neural Operators](https://arxiv.org/pdf/2207.06680) |
| UniGNN | [UniGNN: a Unified Framework for Graph and Hypergraph Neural Networks](https://arxiv.org/pdf/2105.00956) |
| UniGNN2 | [UniGNN: a Unified Framework for Graph and Hypergraph Neural Networks](https://arxiv.org/pdf/2105.00956) |

### Combinatorial complexes
| Model | Reference |
| --- | --- |
| GCCN | [TopoTune: A Framework for Generalized Combinatorial Complex Neural Networks](https://arxiv.org/pdf/2410.06530) |

**Remark:** TopoBench includes [TopoTune](https://arxiv.org/pdf/2410.06530), a comprehensive framework for easily designing new, general TDL models on any domain using any (graph) neural network as a backbone. Please check out the extended [TopoTune wiki page](https://github.com/geometric-intelligence/TopoBench/wiki/TopoTune) for further details on how to leverage this framework to define and train customized topological neural network architectures.


## :rocket: Liftings & Transforms

We list the liftings used in `TopoBench` to transform datasets. Here, a _lifting_ refers to a function that transforms a dataset defined on a topological domain (_e.g._, on a graph) into the same dataset but supported on a different topological domain (_e.g._, on a simplicial complex). 

### Structural Liftings

The structural lifting is responsible for the transformation of the underlying relationships or elements of the data. For instance, it might determine how nodes and edges in a graph are mapped into triangles and tetrahedra in a simplicial complex. This structural transformation can be further categorized into connectivity-based, where the mapping relies solely on the existing connections within the data, and feature-based, where the data's inherent properties or features guide the new structure.

We enumerate below the structural liftings currently implemented in `TopoBench`; please check out the provided description links for further details. 

**Remark:**: Most of these liftings are adaptations of winner submissions of the ICML TDL Challenge 2024 ([paper](https://proceedings.mlr.press/v251/bernardez24a.html) | [repo](https://github.com/pyt-team/challenge-icml-2024)); see the [Structural Liftings wiki](https://github.com/geometric-intelligence/TopoBench/wiki/Structural-Liftings) for a complete list of compatible liftings.

#### Graph to Simplicial Complex
| Name | Type | Description |
| --- | --- | --- |
|   DnD Lifting  |   Feature-based  |  [Wiki page](https://github.com/geometric-intelligence/TopoBench/wiki/DnD-Lifting-(Graph-to-Simplicial))   |
|  Random Latent Clique Lifting   |   Connectivity-based  |   [Wiki page](https://github.com/geometric-intelligence/TopoBench/wiki/Random-Latent-Clique-Lifting-(Graph-to-Simplicial))  |
|  Line Lifting   |   Connectivity-based  |  [Wiki page](https://github.com/geometric-intelligence/TopoBench/wiki/Line-Lifting-(Graph-to-Simplicial))   |
|  Neighbourhood Complex Lifting   |   Connectivity-based  |   [Wiki page](https://github.com/geometric-intelligence/TopoBench/wiki/Neighbourhood-Complex-Lifting-(Graph-to-Simplicial))  |
|  Graph Induced Lifting   |   Connectivity-based  |  [Wiki page](https://github.com/geometric-intelligence/TopoBench/wiki/Graph-Induced-Lifting-(Graph-to-Simplicial))   |
|  Eccentricity Lifting  |  Connectivity-based  |  [Wiki page](https://github.com/geometric-intelligence/TopoBench/wiki/Eccentricity-Lifting-(Graph-to-Simplicial))  |
| Feature‐Based Rips Complex  | Both connectivity and feature-based  |  [Wiki page](https://github.com/geometric-intelligence/TopoBench/wiki/Feature%E2%80%90Based-Rips-Complex-(Graph-to-Simplicial)) |
| Clique Lifting | Connectivity-based | [Wiki page](https://github.com/geometric-intelligence/TopoBench/wiki/Clique-Lifting-(Graph-to-Simplicial)) |
| K-hop Lifting | Connectivity-based | [Wiki page](https://github.com/geometric-intelligence/TopoBench/wiki/KHop-Lifting-(Graph-to-Simplicial)) |

#### Graph to Cell Complex
| Name | Type | Description |
| --- | --- | --- |
|  Discrete Configuration Complex  | Connectivity-based  |  [Wiki page](https://github.com/geometric-intelligence/TopoBench/wiki/Discrete-Configuration-Complex-(Graph-to-Cell))  |
|  Cycle Lifting  | Connectivity-based  |  [Wiki page](https://github.com/geometric-intelligence/TopoBench/wiki/Cycle-Lifting-(Graph-to-Cell))  |


#### Graph to Hypergraph
| Name | Type | Description |
| --- | --- | --- |
|  Expander Hypergraph Lifting  | Connectivity-based  |  [Wiki page](https://github.com/geometric-intelligence/TopoBench/wiki/Expander-Hypergraph-Lifting-(Graph-to-Hypergraph))  |
|  Kernel Lifting  | Both connectivity and feature-based  |  [Wiki page](https://github.com/geometric-intelligence/TopoBench/wiki/Kernel-Lifting-(Graph-to-Hypergraph))  |
|  Mapper Lifting  | Connectivity-based  |  [Wiki page](https://github.com/geometric-intelligence/TopoBench/wiki/Mapper-Lifting-(Graph-to-Hypergraph))  |
|  Forman‐Ricci Curvature Coarse Geometry Lifting  |  Connectivity-based |  [Wiki page](https://github.com/geometric-intelligence/TopoBench/wiki/Forman%E2%80%90Ricci-Curvature-Coarse-Geometry-Lifting-(Graph-to-Hypergraph))  |
|  KNN Lifting  | Feature-based  |  [Wiki page](https://github.com/geometric-intelligence/TopoBench/wiki/KNN-Lifting-(Graph-to-Hypergraph))  |
|  K-hop Lifting  |  Connectivity-based |  [Wiki page](https://github.com/geometric-intelligence/TopoBench/wiki/KHop-Lifting-(Graph-to-Hypergraph))  |


#### Pointcloud to Simplicial
| Name | Type | Description |
| --- | --- | --- |
|  Delaunay Lifting  | Feature-based  |  [Wiki page](https://github.com/geometric-intelligence/TopoBench/wiki/Delaunay-Lifting-(Pointcloud-to-Simplicial))  |
|  Random Flag Complex  |  Feature-based |  [Wiki page](https://github.com/geometric-intelligence/TopoBench/wiki/Random-Flag-Complex-(Pointcloud-to-Simplicial))  |


#### Pointcloud to Hypergraph
| Name | Type | Description |
| --- | --- | --- |
|  Mixture of Gaussians MST lifting  |  Feature-based |  [Wiki page](https://github.com/geometric-intelligence/TopoBench/wiki/Mixture-of-Gaussians---MST-lifting-(Pointcloud-to-Hypergraph))  |
|  PointNet Lifting  | Feature-based  |  [Wiki page](https://github.com/geometric-intelligence/TopoBench/wiki/PointNet--Lifting-(Pointcloud-to-Hypergraph))  |
|  Voronoi Lifting  | Feature-based  |  [Wiki page](https://github.com/geometric-intelligence/TopoBench/wiki/Voronoi-Lifting-(Pointcloud-to-Hypergraph))  |

#### Simplicial to Combinatorial
| Name | Type | Description |
| --- | --- | --- |
| Coface Lifting | Connectivity-based | [Wiki page](https://github.com/geometric-intelligence/TopoBench/wiki/Coface-Lifting-(Simplicial-to-Combinatorial)) |

#### Hypergraph to Combinatorial
| Name | Type | Description |
| --- | --- | --- |
|  Universal Strict Lifting  | Connectivity-based  |  [Wiki page](https://github.com/geometric-intelligence/TopoBench/wiki/Universal-Strict-Lifting-(Hypergraph-to-Combinatorial))  |

### Feature Liftings

Feature liftings address the transfer of data attributes or features during mapping, ensuring that the properties associated with the data elements are consistently preserved in the new representation.

| Name                | Description                                                                 | Supported Domains |
|---------------------|-----------------------------------------------------------------------------|-------------------|
| ProjectionSum       | Projects r-cell features of a graph to r+1-cell structures utilizing incidence matrices \(B_{r}\). | All  |
| ConcatenationLifting | Concatenate r-cell features to obtain r+1-cell features.                   | Simplicial        |

### Data Transformations 

Specially useful in pre-processing steps, these are the general data manipulations currently implemented in `TopoBench`:

| Transform | Description | 
| --- | --- |
| OneHotDegreeFeatures | Adds the node degree as one hot encodings to the node features. |
|NodeFeaturesToFloat |Converts the node features of the input graph to float. |
| NodeDegrees | Calculates the node degrees of the input graph.|
| NodeDegrees | Keeps only the selected fields of the input data. |
| KeepOnlyConnectedComponent | Keep only the largest connected components of the input graph. |
| InfereRadiusConnectivity | Generates the radius connectivity of the input point cloud. |
| InfereKNNConnectivity | Generates the k-nearest neighbor connectivity of the input point cloud. |
| IdentityTransform | An identity transform that does nothing to the input data. |
|  EqualGausFeatures | Generates equal Gaussian features for all nodes. |
|  CalculateSimplicialCurvature |  Calculates the simplicial curvature of the input graph.  |

</details>

## :books: Datasets

### Graph
| Dataset | Task | Description | Reference |
| --- | --- | --- | --- |
| Cora | Classification | Cocitation dataset. | [Source](https://link.springer.com/article/10.1023/A:1009953814988) |
| Citeseer | Classification | Cocitation dataset. | [Source](https://dl.acm.org/doi/10.1145/276675.276685) |
| Pubmed | Classification | Cocitation dataset. | [Source](https://ojs.aaai.org/aimagazine/index.php/aimagazine/article/view/2157) |
| MUTAG | Classification | Graph-level classification. | [Source](https://pubs.acs.org/doi/abs/10.1021/jm00106a046) |
| PROTEINS | Classification | Graph-level classification. | [Source](https://academic.oup.com/bioinformatics/article/21/suppl_1/i47/202991) |
| NCI1 | Classification | Graph-level classification. | [Source](https://ieeexplore.ieee.org/document/4053093) |
| NCI109 | Classification | Graph-level classification. | [Source](https://arxiv.org/pdf/2007.08663) |
| IMDB-BIN | Classification | Graph-level classification. | [Source](https://dl.acm.org/doi/10.1145/2783258.2783417) |
| IMDB-MUL | Classification | Graph-level classification. | [Source](https://dl.acm.org/doi/10.1145/2783258.2783417) |
| REDDIT | Classification | Graph-level classification. | [Source](https://proceedings.neurips.cc/paper_files/paper/2017/file/5dd9db5e033da9c6fb5ba83c7a7ebea9-Paper.pdf) |
| Amazon | Classification | Heterophilic dataset. | [Source](https://arxiv.org/pdf/1205.6233) |
| Minesweeper | Classification | Heterophilic dataset. | [Source](https://arxiv.org/pdf/2302.11640) |
| Empire | Classification | Heterophilic dataset. | [Source](https://arxiv.org/pdf/2302.11640) |
| Tolokers | Classification | Heterophilic dataset. | [Source](https://arxiv.org/pdf/2302.11640) |
| US-county-demos | Regression | In turn each node attribute is used as the target label. | [Source](https://arxiv.org/pdf/2002.08274) |
| ZINC | Regression | Graph-level regression. | [Source](https://pubs.acs.org/doi/10.1021/ci3001277) |


### Simplicial
| Dataset | Task | Description | Reference |
| --- | --- | --- | --- |
| Mantra |  Classification, Multi-label Classification  |  Predict topological attributes of manifold triangulations |  [Source](https://github.com/aidos-lab/MANTRA)  |

### Hypergraph
| Dataset | Task | Description | Reference |
| --- | --- | --- | --- |
| Cora-Cocitation | Classification | Cocitation dataset. | [Source](https://proceedings.neurips.cc/paper_files/paper/2019/file/1efa39bcaec6f3900149160693694536-Paper.pdf) |
| Citeseer-Cocitation | Classification | Cocitation dataset. | [Source](https://proceedings.neurips.cc/paper_files/paper/2019/file/1efa39bcaec6f3900149160693694536-Paper.pdf) |
| PubMed-Cocitation | Classification | Cocitation dataset. | [Source](https://proceedings.neurips.cc/paper_files/paper/2019/file/1efa39bcaec6f3900149160693694536-Paper.pdf) |
| Cora-Coauthorship | Classification | Cocitation dataset. | [Source](https://proceedings.neurips.cc/paper_files/paper/2019/file/1efa39bcaec6f3900149160693694536-Paper.pdf) |
| DBLP-Coauthorship | Classification | Cocitation dataset. | [Source](https://proceedings.neurips.cc/paper_files/paper/2019/file/1efa39bcaec6f3900149160693694536-Paper.pdf) |



## :mag: References ##

To learn more about `TopoBench`, we invite you to read the paper:

```
@article{telyatnikov2024topobench,
      title={TopoBench: A Framework for Benchmarking Topological Deep Learning}, 
      author={Lev Telyatnikov and Guillermo Bernardez and Marco Montagna and Pavlo Vasylenko and Ghada Zamzmi and Mustafa Hajij and Michael T Schaub and Nina Miolane and Simone Scardapane and Theodore Papamarkou},
      year={2024},
      eprint={2406.06642},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2406.06642}, 
}
```
If you find `TopoBench` useful, we would appreciate if you cite us!



## :mouse: Additional Details
<details>
<summary><b>Hierarchy of configuration files</b></summary>

```
├── configs                   <- Hydra configs
│   ├── callbacks                <- Callbacks configs
│   ├── dataset                  <- Dataset configs
│   │   ├── graph                    <- Graph dataset configs
│   │   ├── hypergraph               <- Hypergraph dataset configs
│   │   └── simplicial               <- Simplicial dataset configs
│   ├── debug                    <- Debugging configs
│   ├── evaluator                <- Evaluator configs
│   ├── experiment               <- Experiment configs
│   ├── extras                   <- Extra utilities configs
│   ├── hparams_search           <- Hyperparameter search configs
│   ├── hydra                    <- Hydra configs
│   ├── local                    <- Local configs
│   ├── logger                   <- Logger configs
│   ├── loss                     <- Loss function configs
│   ├── model                    <- Model configs
│   │   ├── cell                     <- Cell model configs
│   │   ├── graph                    <- Graph model configs
│   │   ├── hypergraph               <- Hypergraph model configs
│   │   └── simplicial               <- Simplicial model configs
│   ├── optimizer                <- Optimizer configs
│   ├── paths                    <- Project paths configs
│   ├── scheduler                <- Scheduler configs
│   ├── trainer                  <- Trainer configs
│   ├── transforms               <- Data transformation configs
│   │   ├── data_manipulations       <- Data manipulation transforms
│   │   ├── dataset_defaults         <- Default dataset transforms
│   │   ├── feature_liftings         <- Feature lifting transforms
│   │   └── liftings                 <- Lifting transforms
│   │       ├── graph2cell               <- Graph to cell lifting transforms
│   │       ├── graph2hypergraph         <- Graph to hypergraph lifting transforms
│   │       ├── graph2simplicial         <- Graph to simplicial lifting transforms
│   │       ├── graph2cell_default.yaml  <- Default graph to cell lifting config
│   │       ├── graph2hypergraph_default.yaml <- Default graph to hypergraph lifting config
│   │       ├── graph2simplicial_default.yaml <- Default graph to simplicial lifting config
│   │       ├── no_lifting.yaml           <- No lifting config
│   │       ├── custom_example.yaml       <- Custom example transform config
│   │       └── no_transform.yaml         <- No transform config
│   ├── wandb_sweep              <- Weights & Biases sweep configs
│   │
│   ├── __init__.py              <- Init file for configs module
│   └── run.yaml               <- Main config for training
```


</details>

<details>
<summary><b> More information regarding Topological Deep Learning </b></summary>
  
  [Topological Graph Signal Compression](https://arxiv.org/pdf/2308.11068)
  
  [Architectures of Topological Deep Learning: A Survey on Topological Neural Networks](https://par.nsf.gov/servlets/purl/10477141)
  
  [TopoX: a suite of Python packages for machine learning on topological domains](https://arxiv.org/pdf/2402.02441)	
</details>
