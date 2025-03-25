🌐 TopoBench (TB) 🍩
==========================

.. figure:: https://github.com/geometric-intelligence/TopoBench/raw/main/resources/logo.jpg
   :alt: topobench
   :class: with-shadow
   :width: 1000px

`TopoBench` (TB) is a modular Python library designed to standardize benchmarking and accelerate research in Topological Deep Learning (TDL). 
In particular, TB allows to train and compare the performances of all sorts of Topological Neural Networks (TNNs) across the different topological domains, 
where by *topological domain* we refer to a graph, a simplicial complex, a cellular complex, or a hypergraph.

.. figure:: https://github.com/geometric-intelligence/TopoBench/raw/main/resources/workflow.jpg
   :alt: workflow
   :class: with-shadow
   :width: 1000px

📌 Overview
-----------

The main pipeline trains and evaluates a wide range of state-of-the-art TNNs and Graph Neural Networks (GNNs) 
(see :ref:`Neural Networks`) on numerous and varied datasets and benchmark
tasks (see :ref:`Datasets`). 

Additionally, the library offers the ability to transform, i.e., *lift*, each dataset from one topological domain to another 
(see :ref:`Liftings`), enabling for the first time an exhaustive inter-domain comparison of TNNs.


⚙ Neural Networks
-----------------

We list the neural networks trained and evaluated by `TopoBench`, organized by the topological domain over which they operate: graph, simplicial complex, cellular complex or hypergraph. Many of these neural networks were originally implemented in `TopoModelX <https://github.com/pyt-team/TopoModelX>`_.


Graphs
******
.. list-table:: 
   :widths: 20 80
   :header-rows: 1

   * - Model
     - Reference
   * - GAT
     - `Graph Attention Networks <https://openreview.net/pdf?id=rJXMpikCZ>`_
   * - GIN
     - `How Powerful are Graph Neural Networks? <https://openreview.net/pdf?id=ryGs6iA5Km>`_
   * - GCN
     - `Semi-Supervised Classification with Graph Convolutional Networks <https://arxiv.org/pdf/1609.02907v4>`_

Simplicial complexes
********************
.. list-table:: 
   :widths: 20 80
   :header-rows: 1

   * - Model
     - Reference
   * - SAN
     - `Simplicial Attention Neural Networks <https://arxiv.org/pdf/2203.07485>`_
   * - SCCN
     - `Efficient Representation Learning for Higher-Order Data with Simplicial Complexes <https://openreview.net/pdf?id=nGqJY4DODN>`_
   * - SCCNN
     - `Convolutional Learning on Simplicial Complexes <https://arxiv.org/pdf/2301.11163>`_
   * - SCN
     - `Simplicial Complex Neural Networks <https://ieeexplore.ieee.org/document/10285604>`_


Cellular complexes
******************
.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Model
     - Reference
   * - CAN
     - `Cell Attention Network <https://arxiv.org/pdf/2209.08179>`_
   * - CCCN
     - `A learning algorithm for computational connected cellular network <https://ieeexplore.ieee.org/document/1202221>`_
   * - CXN
     - `Cell Complex Neural Networks <https://openreview.net/pdf?id=6Tq18ySFpGU>`_
   * - CWN
     - `Weisfeiler and Lehman Go Cellular: CW Networks <https://arxiv.org/pdf/2106.12575>`_


Hypergraphs
***********
.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Model
     - Reference
   * - AllDeepSet
     - `You are AllSet: A Multiset Function Framework for Hypergraph Neural Networks <https://openreview.net/pdf?id=hpBTIv2uy_E>`_
   * - AllSetTransformer
     - `You are AllSet: A Multiset Function Framework for Hypergraph Neural Networks <https://openreview.net/pdf?id=hpBTIv2uy_E>`_
   * - EDGNN
     - `Equivariant Hypergraph Diffusion Neural Operators <https://arxiv.org/pdf/2207.06680>`_
   * - UniGNN
     - `UniGNN: a Unified Framework for Graph and Hypergraph Neural Networks <https://arxiv.org/pdf/2105.00956>`_
   * - UniGNN2
     - `UniGNN: a Unified Framework for Graph and Hypergraph Neural Networks <https://arxiv.org/pdf/2105.00956>`_


🚀 Liftings
-----------

We list the liftings used in `TopoBench` to transform datasets. Here, a *lifting* refers to a function that transforms a dataset defined on a topological domain (*e.g.*, on a graph) into the same dataset but supported on a different topological domain (*e.g.*, on a simplicial complex).

Graph2Simplicial
****************
.. list-table::
   :widths: 20 60 20
   :header-rows: 1

   * - Name
     - Description
     - Reference
   * - CliqueLifting
     - The algorithm finds the cliques in the graph and creates simplices. Given a clique the first simplex added is the one containing all the nodes of the clique, then the simplices composed of all the possible combinations with one node missing, then two nodes missing, and so on, until all the possible pairs are added. Then the method moves to the next clique.
     - `Simplicial Complexes <https://en.wikipedia.org/wiki/Clique_complex>`_
   * - KHopLifting
     - For each node in the graph, take the set of its neighbors, up to k distance, and the node itself. These sets are then treated as simplices. The dimension of each simplex depends on the degree of the nodes. For example, a node with d neighbors forms a d-simplex.
     - `Neighborhood Complexes <https://arxiv.org/pdf/math/0512077>`_

Graph2Cell
**********

.. list-table::
   :widths: 20 60 20
   :header-rows: 1

   * - Name
     - Description
     - Reference
   * - CellCycleLifting
     - To lift a graph to a cell complex (CC) we proceed as follows. First, we identify a finite set of cycles (closed loops) within the graph. Second, each identified cycle in the graph is associated to a 2-cell, such that the boundary of the 2-cell is the cycle. The nodes and edges of the cell complex are inherited from the graph.
     - `Appendix B <https://arxiv.org/abs/2206.00606>`_


Graph2Hypergraph
****************

.. list-table::
   :widths: 20 60 20
   :header-rows: 1

   * - Name
     - Description
     - Reference
   * - KHopLifting
     - For each node in the graph, the algorithm finds the set of nodes that are at most k connections away from the initial node. This set is then used to create a hyperedge. The process is repeated for all nodes in the graph.
     - `Section 3.4 <https://ieeexplore.ieee.org/abstract/document/9264674>`_
   * - KNearestNeighborsLifting
     - For each node in the graph, the method finds the k nearest nodes by using the Euclidean distance between the vectors of features. The set of k nodes found is considered as a hyperedge. The process is repeated for all nodes in the graph.
     - `Section 3.1 <https://ieeexplore.ieee.org/abstract/document/9264674>`_


📚 Datasets
-----------

.. list-table::
   :widths: 15 15 40 30
   :header-rows: 1

   * - Dataset
     - Task
     - Description
     - Reference
   * - Cora
     - Classification
     - Cocitation dataset.
     - `Source <https://link.springer.com/article/10.1023/A:1009953814988>`_
   * - Citeseer
     - Classification
     - Cocitation dataset.
     - `Source <https://dl.acm.org/doi/10.1145/276675.276685>`_
   * - Pubmed
     - Classification
     - Cocitation dataset.
     - `Source <https://ojs.aaai.org/aimagazine/index.php/aimagazine/article/view/2157>`_
   * - MUTAG
     - Classification
     - Graph-level classification.
     - `Source <https://pubs.acs.org/doi/abs/10.1021/jm00106a046>`_
   * - PROTEINS
     - Classification
     - Graph-level classification.
     - `Source <https://academic.oup.com/bioinformatics/article/21/suppl_1/i47/202991>`_
   * - NCI1
     - Classification
     - Graph-level classification.
     - `Source <https://ieeexplore.ieee.org/document/4053093>`_
   * - NCI109
     - Classification
     - Graph-level classification.
     - `Source <https://arxiv.org/pdf/2007.08663>`_
   * - IMDB-BIN
     - Classification
     - Graph-level classification.
     - `Source <https://dl.acm.org/doi/10.1145/2783258.2783417>`_
   * - IMDB-MUL
     - Classification
     - Graph-level classification.
     - `Source <https://dl.acm.org/doi/10.1145/2783258.2783417>`_
   * - REDDIT
     - Classification
     - Graph-level classification.
     - `Source <https://proceedings.neurips.cc/paper_files/paper/2017/file/5dd9db5e033da9c6fb5ba83c7a7ebea9-Paper.pdf>`_
   * - Amazon
     - Classification
     - Heterophilic dataset.
     - `Source <https://arxiv.org/pdf/1205.6233>`_
   * - Minesweeper
     - Classification
     - Heterophilic dataset.
     - `Source <https://arxiv.org/pdf/2302.11640>`_
   * - Empire
     - Classification
     - Heterophilic dataset.
     - `Source <https://arxiv.org/pdf/2302.11640>`_
   * - Tolokers
     - Classification
     - Heterophilic dataset.
     - `Source <https://arxiv.org/pdf/2302.11640>`_
   * - US-county-demos
     - Regression
     - In turn each node attribute is used as the target label.
     - `Source <https://arxiv.org/pdf/2002.08274>`_
   * - ZINC
     - Regression
     - Graph-level regression.
     - `Source <https://pubs.acs.org/doi/10.1021/ci3001277>`_


🔍 References
-------------

To learn more about `TopoBench`, we invite you to read the paper:

.. code-block:: BibTeX

    @misc{topobench2024,
            title={TopoBench},
            author={PyT-Team},
            year={2024},
            eprint={TBD},
            archivePrefix={arXiv},
            primaryClass={cs.LG}
    }

If you find `TopoBench` useful, we would appreciate if you cite us!

🦾 Getting Started
------------------

Check out our `tutorials <https://github.com/geometric-intelligence/TopoBench/tree/main/tutorials>`_ to get started!


.. toctree::
   :maxdepth: 2
   :hidden:

   api/index
   contributing/index
   tutorials/index
