================================================================================
API Reference
================================================================================

This section contains the complete API documentation for the TopoBench package.
The documentation is automatically generated from the source code and organized
into logical categories for easy navigation.

Overview
--------------------------------------------------------------------------------

TopoBench provides a comprehensive framework for topological deep learning, including:

* **Core**: Core TopoBench functionality and main entry points
* **Data Loading & Datasets**: Dataset loaders for different topological domains
* **Neural Network Architectures**: Neural network backbones, encoders, readouts, and wrappers
* **Transformations & Liftings**: Data transformations and topological lifting operations
* **Training & Evaluation**: Loss functions, optimizers, metrics, and evaluation tools
* **Utilities**: Helper functions and utility modules

.. contents:: Quick Navigation
   :local:
   :depth: 2


Core
----

Core TopoBench functionality and main entry points

.. toctree::
   :maxdepth: 1

   topobench
   topobench.callbacks
   topobench.data
   topobench.evaluator
   topobench.loss
   topobench.model
   topobench.nn
   topobench.optimizer
   topobench.run


Data Loading & Datasets
-----------------------

Dataset loaders for different topological domains

.. toctree::
   :maxdepth: 1

   topobench.dataloader

Data
^^^^

.. toctree::
   :maxdepth: 1

   topobench.data.datasets
   topobench.data.datasets.citation_hypergaph_dataset
   topobench.data.datasets.hypergraph_datasets
   topobench.data.datasets.mantra_dataset
   topobench.data.datasets.us_county_demos_dataset
   topobench.data.loaders
   topobench.data.loaders.base
   topobench.data.loaders.graph
   topobench.data.loaders.graph.hetero_datasets
   topobench.data.loaders.graph.mantra_dataset
   topobench.data.loaders.graph.manual_graph_dataset_loader
   topobench.data.loaders.graph.modecule_datasets
   topobench.data.loaders.graph.ogbg_datasets
   topobench.data.loaders.graph.planetoid_datasets
   topobench.data.loaders.graph.tu_datasets
   topobench.data.loaders.graph.us_county_demos_dataset_loader
   topobench.data.loaders.hypergraph
   topobench.data.loaders.hypergraph.citation_hypergraph_dataset_loader
   topobench.data.loaders.hypergraph.hypergraph_dataset_loader
   topobench.data.loaders.pointcloud
   topobench.data.loaders.pointcloud.geometric_shapes
   topobench.data.loaders.simplicial
   topobench.data.loaders.simplicial.mantra_dataset_loader
   topobench.data.preprocessor
   topobench.data.preprocessor.preprocessor

Dataloader
^^^^^^^^^^

.. toctree::
   :maxdepth: 1

   topobench.dataloader.dataload_dataset
   topobench.dataloader.dataloader
   topobench.dataloader.utils


Neural Network Architectures
----------------------------

Neural network backbones, encoders, readouts, and wrappers

Loss
^^^^

.. toctree::
   :maxdepth: 1

   topobench.loss.model.DGMLoss
   topobench.loss.model.GraphMLPLoss

Model
^^^^^

.. toctree::
   :maxdepth: 1

   topobench.model.model

Nn
^^

.. toctree::
   :maxdepth: 1

   topobench.nn.backbones
   topobench.nn.backbones.cell
   topobench.nn.backbones.cell.cccn
   topobench.nn.backbones.combinatorial
   topobench.nn.backbones.combinatorial.gccn
   topobench.nn.backbones.combinatorial.gccn_onehasse
   topobench.nn.backbones.graph
   topobench.nn.backbones.graph.gps
   topobench.nn.backbones.graph.graph_mlp
   topobench.nn.backbones.graph.identity_gnn
   topobench.nn.backbones.hypergraph
   topobench.nn.backbones.hypergraph.edgnn
   topobench.nn.backbones.non_relational
   topobench.nn.backbones.non_relational.mlp
   topobench.nn.backbones.simplicial
   topobench.nn.backbones.simplicial.sccnn
   topobench.nn.encoders
   topobench.nn.encoders.all_cell_encoder
   topobench.nn.encoders.base
   topobench.nn.encoders.dgm_encoder
   topobench.nn.encoders.flat_encoder
   topobench.nn.encoders.kdgm
   topobench.nn.readouts
   topobench.nn.readouts.base
   topobench.nn.readouts.identical
   topobench.nn.readouts.mlp_readout
   topobench.nn.readouts.propagate_signal_down
   topobench.nn.wrappers
   topobench.nn.wrappers.base
   topobench.nn.wrappers.cell
   topobench.nn.wrappers.cell.can_wrapper
   topobench.nn.wrappers.cell.cccn_wrapper
   topobench.nn.wrappers.cell.ccxn_wrapper
   topobench.nn.wrappers.cell.cwn_wrapper
   topobench.nn.wrappers.combinatorial
   topobench.nn.wrappers.combinatorial.tune_wrapper
   topobench.nn.wrappers.graph
   topobench.nn.wrappers.graph.gnn_wrapper
   topobench.nn.wrappers.graph.graph_mlp_wrapper
   topobench.nn.wrappers.hypergraph
   topobench.nn.wrappers.hypergraph.hypergraph_wrapper
   topobench.nn.wrappers.pointcloud
   topobench.nn.wrappers.pointcloud.pointcloud_wrapper
   topobench.nn.wrappers.simplicial
   topobench.nn.wrappers.simplicial.san_wrapper
   topobench.nn.wrappers.simplicial.sccn_wrapper
   topobench.nn.wrappers.simplicial.sccnn_wrapper
   topobench.nn.wrappers.simplicial.scn_wrapper


Transformations & Liftings
--------------------------

Data transformations and topological lifting operations

.. toctree::
   :maxdepth: 1

   topobench.transforms

Transforms
^^^^^^^^^^

.. toctree::
   :maxdepth: 1

   topobench.transforms.data_manipulations
   topobench.transforms.data_manipulations.calculate_simplicial_curvature
   topobench.transforms.data_manipulations.equal_gaus_features
   topobench.transforms.data_manipulations.group_homophily
   topobench.transforms.data_manipulations.identity_transform
   topobench.transforms.data_manipulations.infere_knn_connectivity
   topobench.transforms.data_manipulations.infere_radius_connectivity
   topobench.transforms.data_manipulations.keep_only_connected_component
   topobench.transforms.data_manipulations.keep_selected_data_fields
   topobench.transforms.data_manipulations.laplacian_encodings
   topobench.transforms.data_manipulations.mp_homophily
   topobench.transforms.data_manipulations.node_degrees
   topobench.transforms.data_manipulations.node_features_to_float
   topobench.transforms.data_manipulations.one_hot_degree_features
   topobench.transforms.data_manipulations.positional_and_structural_encodings
   topobench.transforms.data_manipulations.random_walk_encodings
   topobench.transforms.data_manipulations.redefine_simplicial_neighbourhoods
   topobench.transforms.data_transform
   topobench.transforms.feature_liftings
   topobench.transforms.feature_liftings.concatenation
   topobench.transforms.feature_liftings.identity
   topobench.transforms.feature_liftings.projection_sum
   topobench.transforms.feature_liftings.set
   topobench.transforms.liftings
   topobench.transforms.liftings.base
   topobench.transforms.liftings.graph2cell
   topobench.transforms.liftings.graph2cell.base
   topobench.transforms.liftings.graph2cell.cycle_lifting
   topobench.transforms.liftings.graph2cell.discrete_configuration_complex_lifting
   topobench.transforms.liftings.graph2combinatorial
   topobench.transforms.liftings.graph2combinatorial.base
   topobench.transforms.liftings.graph2combinatorial.graph_induced_cc
   topobench.transforms.liftings.graph2hypergraph
   topobench.transforms.liftings.graph2hypergraph.base
   topobench.transforms.liftings.graph2hypergraph.expander_graph_lifting
   topobench.transforms.liftings.graph2hypergraph.forman_ricci_curvature_lifting
   topobench.transforms.liftings.graph2hypergraph.kernel_lifting
   topobench.transforms.liftings.graph2hypergraph.khop_lifting
   topobench.transforms.liftings.graph2hypergraph.knn_lifting
   topobench.transforms.liftings.graph2hypergraph.mapper_lifting
   topobench.transforms.liftings.graph2hypergraph.modularity_maximization_lifting
   topobench.transforms.liftings.graph2simplicial
   topobench.transforms.liftings.graph2simplicial.base
   topobench.transforms.liftings.graph2simplicial.clique_lifting
   topobench.transforms.liftings.graph2simplicial.dnd_lifting
   topobench.transforms.liftings.graph2simplicial.eccentricity_lifting
   topobench.transforms.liftings.graph2simplicial.graph_induced_lifting
   topobench.transforms.liftings.graph2simplicial.khop_lifting
   topobench.transforms.liftings.graph2simplicial.latentclique_lifting
   topobench.transforms.liftings.graph2simplicial.line_lifting
   topobench.transforms.liftings.graph2simplicial.neighborhood_complex_lifting
   topobench.transforms.liftings.graph2simplicial.vietoris_rips_lifting
   topobench.transforms.liftings.hypergraph2combinatorial
   topobench.transforms.liftings.hypergraph2combinatorial.base
   topobench.transforms.liftings.hypergraph2combinatorial.universal_strict_lifting
   topobench.transforms.liftings.liftings
   topobench.transforms.liftings.pointcloud2hypergraph
   topobench.transforms.liftings.pointcloud2hypergraph.base
   topobench.transforms.liftings.pointcloud2hypergraph.mogmst_lifting
   topobench.transforms.liftings.pointcloud2hypergraph.pointnet_lifting
   topobench.transforms.liftings.pointcloud2hypergraph.voronoi_lifting
   topobench.transforms.liftings.pointcloud2simplicial
   topobench.transforms.liftings.pointcloud2simplicial.alpha_complex_lifting
   topobench.transforms.liftings.pointcloud2simplicial.base
   topobench.transforms.liftings.pointcloud2simplicial.random_flag_complex
   topobench.transforms.liftings.simplicial2combinatorial
   topobench.transforms.liftings.simplicial2combinatorial.base
   topobench.transforms.liftings.simplicial2combinatorial.coface_cc_lifting


Training & Evaluation
---------------------

Loss functions, optimizers, metrics, and evaluation tools

Callbacks
^^^^^^^^^

.. toctree::
   :maxdepth: 1

   topobench.callbacks.timer_callback

Evaluator
^^^^^^^^^

.. toctree::
   :maxdepth: 1

   topobench.evaluator.base
   topobench.evaluator.evaluator
   topobench.evaluator.metrics
   topobench.evaluator.metrics.example

Loss
^^^^

.. toctree::
   :maxdepth: 1

   topobench.loss.base
   topobench.loss.dataset
   topobench.loss.dataset.DatasetLoss
   topobench.loss.loss
   topobench.loss.model

Optimizer
^^^^^^^^^

.. toctree::
   :maxdepth: 1

   topobench.optimizer.base
   topobench.optimizer.optimizer


Utilities
---------

Helper functions and utility modules

.. toctree::
   :maxdepth: 1

   topobench.utils

Data
^^^^

.. toctree::
   :maxdepth: 1

   topobench.data.utils
   topobench.data.utils.io_utils
   topobench.data.utils.split_utils
   topobench.data.utils.utils

Utils
^^^^^

.. toctree::
   :maxdepth: 1

   topobench.utils.config_resolvers
   topobench.utils.instantiators
   topobench.utils.logging_utils
   topobench.utils.pylogger
   topobench.utils.rich_utils
   topobench.utils.utils

