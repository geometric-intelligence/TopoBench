---
title: Getting Started
permalink: /docs/getting-started/
---

<div class="workflow-image" style="text-align: center; margin: 30px 0;">
    <img src="{{ site.baseurl }}/assets/img/workflow.jpg" alt="TopoBench Workflow" style="max-width: 100%; height: auto;">
</div>

## :pushpin: Overview

`TopoBench` (TB) is a modular Python library designed to standardize benchmarking and accelerate research in Topological Deep Learning (TDL). In particular, TB allows to train and compare the performances of all sorts of Topological Neural Networks (TNNs) across the different topological domains, where by _topological domain_ we refer to a graph, a simplicial complex, a cellular complex, or a hypergraph. For detailed information, please refer to the [`TopoBench: A Framework for Benchmarking Topological Deep Learning`](https://arxiv.org/pdf/2406.06642) paper.

The main pipeline trains and evaluates a wide range of state-of-the-art TNNs and Graph Neural Networks (GNNs) on numerous and varied datasets and benchmark tasks. Additionally, the library offers the ability to transform, i.e. _lift_, each dataset from one topological domain to another, enabling for the first time an exhaustive inter-domain comparison of TNNs.

## :jigsaw: Get Started

### Create Environment

First, ensure `conda` is installed:  
```bash
conda --version
```
If not, we recommend intalling Miniconda [following the official command line instructions](https://www.anaconda.com/docs/getting-started/miniconda/install).

Then, clone and navigate to the `TopoBench` repository  
```bash
git clone git@github.com:geometric-intelligence/topobench.git
cd TopoBench
```

Next, set up and activate a conda environment `tb` with Python 3.11.3:
```bash
conda create -n tb python=3.11.3
conda activate tb
```

If working with GPUs, check the CUDA version of your machine:
```bash
which nvcc && nvcc --version
```
and ensure that it matches the CUDA version specified in the `env_setup.sh` file (`CUDA=cpu` by default for a broader compatibility). If it does not match, update `env_setup.sh` accordingly by changing both the `CUDA` and `TORCH` environment variables to compatible values as specified on [this website](https://github.com/pyg-team/pyg-lib).

Next, set up the environment with the following command.
```bash
source env_setup.sh
```
This command installs the `TopoBench` library and its dependencies. 

### Run Training Pipeline

Once the setup is completed, train and evaluate a neural network by running the following command:

```bash
python -m topobench 
```

---

### Customizing Experiment Configuration
Thanks to `hydra` implementation, one can easily override the default experiment configuration through the command line. For instance, the model and dataset can be selected as:

```
python -m topobench model=cell/cwn dataset=graph/MUTAG
```
**Remark:** By default, our pipeline identifies the source and destination topological domains, and applies a default lifting between them if required.

<details>
<summary><strong>Configuring Individual Transforms</strong></summary>

When configuring a single transform, follow these steps:

1. Choose a desired transform (e.g., a lifting transform).
2. Identify the relative path to the transform configuration.

The folder structure for transforms is as follows:

```
├── configs
│ ├── data_manipulations
│ ├── transforms
│ │ └── liftings
│ │   ├── graph2cell
│ │   ├── graph2hypergraph
│ │   └── graph2simplicial
```

To override the default transform, use the following command structure:

```bash
python -m topobench model=<model_type>/<model_name> dataset=<data_type>/<dataset_name> transforms=[<transform_path>/<transform_name>]
```

For example, to use the `discrete_configuration_complex` lifting with the `cell/cwn` model:

```bash
python -m topobench model=cell/cwn dataset=graph/MUTAG transforms=[liftings/graph2cell/discrete_configuration_complex]
```

</details>

<details>
<summary><strong>Configuring Transform Groups</strong></summary>

For more complex scenarios, such as combining multiple data manipulations, use transform groups:

1. Create a new configuration file in the `configs/transforms` directory (e.g., `custom_example.yaml`).
2. Define the transform group in the YAML file:

```yaml
defaults:
- data_manipulations@data_transform_1: identity
- data_manipulations@data_transform_2: node_degrees
- data_manipulations@data_transform_3: one_hot_node_degree_features
- liftings/graph2cell@graph2cell_lifting: cycle
```

**Important:** When composing multiple data manipulations, use the `@` operator to assign unique names to each transform.

3. Run the experiment with the custom transform group:

```bash
python -m topobench model=cell/cwn dataset=graph/ZINC transforms=custom_example
```

This approach allows you to create complex transform pipelines, including multiple data manipulations and liftings, in a single configuration file.

</details>

By mastering these configuration options, you can easily customize your experiments to suit your specific needs, from simple model and dataset selections to complex data transformation pipelines.

---

### Additional Notes

- **Automatic Lifting:** By default, our pipeline identifies the source and destination topological domains and applies a default lifting between them if required.  
- **Fine-Grained Configuration:** The same CLI override mechanism applies when modifying finer configurations within a `CONFIG GROUP`.  
  Please refer to the official [`hydra` documentation](https://hydra.cc/docs/intro/) for further details.

## :bike: Experiments Reproducibility
To reproduce Table 1 from the [`TopoBench: A Framework for Benchmarking Topological Deep Learning`](https://arxiv.org/pdf/2406.06642) paper, please run the following command:

```bash
bash scripts/reproduce.sh
```
**Remark:** We have additionally provided a public [W&B (Weights & Biases) project](https://wandb.ai/telyatnikov_sap/TopoBenchmark_main?nw=nwusertelyatnikov_sap) with logs for the corresponding runs (updated on June 11, 2024).

## :anchor: Tutorials

Explore our [tutorials](https://github.com/geometric-intelligence/TopoBench/tree/main/tutorials) for further details on how to add new datasets, transforms/liftings, and benchmark tasks. 

## :gear: Neural Networks

We list the neural networks trained and evaluated by `TopoBench`, organized by the topological domain over which they operate: graph, simplicial complex, cellular complex or hypergraph. Many of these neural networks were originally implemented in [`TopoModelX`](https://github.com/pyt-team/TopoModelX). 