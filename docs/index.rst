TopoBench
=========

.. figure:: https://github.com/geometric-intelligence/TopoBench/raw/main/resources/logo.jpg
   :alt: TopoBench
   :class: with-shadow
   :width: 1000px

TopoBench is a reproducible benchmark core for graph, heterogeneous graph,
and hypergraph learning. Hydra composes a native dataset, a compatible model,
a trainer, evaluation, checkpointing, and logging without converting one data
domain into another.

If you are here to...
----------------------

* **Run a first experiment:** install with ``uv`` and run the graph command
  below.
* **Implement or inspect homogeneous data:** open :doc:`graph_data`.
* **Use typed relations and sampled seeds:** open
  :doc:`heterogeneous_graphs`.
* **Work with node-to-hyperedge incidence:** open :doc:`hypergraphs`.
* **Find a public class or function:** open :doc:`api/index`.
* **Propose a change:** read :doc:`contributing/index`.

Install
-------

TopoBench requires Python 3.11. Clone the repository, then let the setup script
create and synchronize the ``uv`` environment:

.. code-block:: bash

   git clone https://github.com/geometric-intelligence/topobench.git
   cd topobench
   source uv_env_setup.sh

The committed lock selects CPU packages on macOS and CUDA 12.1 packages on
Linux. The setup script does not modify ``pyproject.toml`` or ``uv.lock``. In
an existing compatible environment, perform the same immutable sync directly:

.. code-block:: bash

   uv sync --frozen --all-extras

Run the current product
-----------------------

The smallest graph run states the default selectors explicitly:

.. code-block:: bash

   uv run python -m topobench dataset=graph/SyntheticGraph model=graph/gcn

The retained experiment selectors provide fixed examples for every domain:

.. code-block:: bash

   uv run python -m topobench experiment=graph_synthetic_regression
   uv run python -m topobench experiment=heterogeneous_synthetic_hgt_full
   uv run python -m topobench experiment=heterogeneous_synthetic_hgt_neighbor
   uv run python -m topobench experiment=hypergraph_synthetic_edgnn
   uv run python -m topobench experiment=hypergraph_synthetic_hypergraph_conv

The synthetic inputs are deterministic and require no dataset download. The
heterogeneous guide gives the HeteroSAGE, DBLP, and OGB-MAG commands and
explains the resource implications of neighbor sampling.

One trainer, three native representations
-----------------------------------------

**Graph** uses PyTorch Geometric ``Data``. PyG concatenates node features,
offsets ``edge_index``, and creates ``batch`` so graph-level readouts know
which nodes belong to each example.

**Heterogeneous graph** uses ``HeteroData`` with separate node and relation
stores. Full mode supervises one target-node mask on the complete graph.
Neighbor mode supervises only target seeds, which PyG places first and counts
in ``batch[target_node_type].batch_size``.

**Hypergraph** uses ``HypergraphData.hyperedge_index`` with shape ``[2, M]``.
Row 0 contains node IDs and row 1 contains contiguous hyperedge IDs. PyG
offsets the two rows independently when batching examples.

Checkpoint reruns
-----------------

After training, TopoBench reloads the selected checkpoint and evaluates it on
the validation and test loaders. These final values are logged as
``val_best_rerun/<metric>`` and ``test_best_rerun/<metric>``. They are distinct
from the metrics produced by the final training epoch and are the appropriate
namespaces for reporting the selected model.

Attribution and license
-----------------------

TopoBench is developed by the Topological-Intelligence Team Authors. The
project is distributed under the terms in the repository's
`LICENSE <https://github.com/geometric-intelligence/TopoBench/blob/main/LICENSE>`__;
third-party materials may carry additional notices in ``third_party_licenses.txt``.
For research context and citation metadata, see
`TopoBench: A Framework for Benchmarking Topological Deep Learning
<https://openreview.net/forum?id=07sTzyEVtY>`__.

.. toctree::
   :maxdepth: 2
   :caption: Guides

   graph_data
   heterogeneous_graphs
   hypergraphs

.. toctree::
   :maxdepth: 2
   :caption: Reference and contribution policy

   api/index
   contributing/index
