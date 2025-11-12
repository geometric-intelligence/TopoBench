TAG-DS TDL Challenge 2025
Welcome to the Topological Deep Learning Challenge 2025: Expanding the Data Landscape, sponsored by Arlequin AI and hosted at the first Topology, Algebra, and Geometry in Data Science (TAG-DS) Conference.

Organizers: Guillermo Bernardez, Lev Telyatnikov, Mathilde Papillon, Marco Montagna, Miquel Ferriol, Raffael Theiler, Olga Fink, Nina Miolane

See also

Link to the challenge repository: geometric-intelligence/TopoBench.

Motivation
../_images/challenge_image.png
Topological Deep Learning (TDL) is emerging as a powerful paradigm for analyzing data with topological neural networks, a class of models that extends Graph Neural Networks (GNNs) to capture interactions beyond simple pairs of nodes. While GNNs model pairwise links—like friendships in a social network—many systems involve higher-order relationships, such as six atoms forming a ring in a molecule or multiple researchers co-authoring a paper in a co-authorship network. TDL provides a principled framework to represent and learn from these richer, multi-way relationships, making it especially suited for complex data in science and engineering.

TopoBench is the premier open-source platform for developing and benchmarking TDL models. It provides a unified interface for datasets, loaders, and tasks across topological domains. TopoBench enables reproducible comparisons, accelerates model development, and serves as the central hub for TDL experimentation.

For TDL to realize its full potential, the data landscape must evolve (Papamarkou et al., 2024). The field’s current reliance on a handful of well-worn benchmarks creates a bottleneck for innovation and rigorous evaluation. To propel the field forward, we focus on the following fronts:

Broadening: the foundational benchmarks of TDL by adapting diverse datasets from the GNN community and beyond.

Deepening: TDL capabilities by building infrastructure for large-scale datasets and novel benchmark tasks.

We are calling on the community to build a richer, more robust, and scalable data ecosystem for TDL. By participating, you will be paving the way for the next generation of topological models.

Description of the Challenge
The 2025 Topological Deep Learning Challenge is organized into two primary missions designed to systematically expand the TDL ecosystem: expanding the data landscape and advancing the core data infrastructure. Participants are invited to contribute one dataset (or dataset class, such as TUDataset) per submission, where each submission belongs to one category. For examples of datasets to contribute, please refer to the Reference Datasets Table below.. We also welcome other datasets, either existing or novel (as long as they are hosted on an open-source platform).

We invite participants to review this webpage regularly, as more details might be added to answer relevant questions.

Mission A — Expanding the Data Landscape
This mission focuses on enriching the TDL ecosystem with a diverse and meaningful collection of datasets.

Category A.1: Broadening Benchmarks with Graphs & Point Clouds
(Difficulty: Easy/Medium)

Implement a well-established graph or point cloud dataset. Expand TDL’s applicability to new fields. Alternatively, help bridge the gap between TDL and mainstream GNN research by incorporating the challenging benchmarks used to test modern graph models. TopoBench includes a suite of topological transforms that will be able to “lift” these graph-based or point cloud datasets to higher-order domains, such as hypergraphs or simplicial complexes.

Category A.2: Curating Natively Higher-Order Datasets
(Difficulty: Medium)

Integrate datasets where higher-order structures (e.g., simplicial complexes, hypergraphs) are a native feature of the data, rather than being algorithmically inferred by transforms in TopoBench. These datasets are critical for testing the unique expressive power of topological models.

Mission B — Advancing the Data Infrastructure
Category B.1: Developing Large-Scale Inductive Data Infrastructure
(Difficulty: Hard)

This mission focuses on building the scalable and versatile software tools needed for next-generation TDL research. It challenges participants to engineer a scalable data loading pipeline for large-scale inductive learning settings, accommodating datasets with either many small graphs or a few giant ones.

The primary obstacle is that standard InMemoryDataset loaders often fail when preprocessing large datasets—especially during memory-intensive “lifting” operations that construct higher-order topological domains. This process can easily exhaust system RAM, leading to crashes.

Core task. Implement a robust OnDiskDataset loader that processes and saves each sample to disk individually, effectively bypassing memory bottlenecks and enabling datasets that were previously intractable.

Bonus (Difficulty: Survival)

Tackle large-scale transductive learning with an OnDiskDataset loader. This is significantly more complex and requires a deep understanding of both TDL pipelines and the TopoBench architecture. If interested, please contact the organizers (see Questions section) for details.

Category B.2: Pioneering New TDL Benchmark Tasks
(Difficulty: Varies)

Introduce novel benchmark tasks beyond standard node-level classification. Each submission must pair a dataset with the infrastructure needed to establish it as a new benchmark, including data splits, evaluation metrics, and a baseline model.

Example topics:

Link prediction in hypergraphs

Higher-order cell classification or regression

Hyperedge-dependent node classification

Your own creative proposal

Reward Outcomes
⭐ White paper: Every submission that meets the requirements will be included in a white paper summarizing the challenge’s findings (planned via PMLR through Topology, Algebra, and Geometry in Data Science 2025). Authors of qualifying submissions will be offered co-authorship. [1]

🏆 Cash prizes: Four winning teams (one per category) will be announced at TAG-DS 2025 during the Awards Ceremony.

💰 Mission A winners: $200 USD each (sponsored by Arlequin AI).

💰 Mission B winners: $800 USD each (sponsored by Arlequin AI).

🌴 Research internship — Geometric Intelligence Lab, UCSB (USA): A team, pending evaluation results and interest, will be invited for a visit of up to two months at the Geometric Intelligence Lab, University of California, Santa Barbara. During the visit, winners will work on cutting-edge methods for TDL. Travel costs will be reimbursed and financial assistance for lodging will be provided. Remote participation is also available. [2]

🏔️ Research internship — IMOS Lab, EPFL (Switzerland): A team, pending evaluation results and interest, will be invited for a research internship at the Intelligent Maintenance and Operations Systems (IMOS) Lab at EPFL in Lausanne, Switzerland. Winners will develop TDL methods in a world-class academic environment. MSc enrollment at the time of the internship is required. Financial assistance for lodging will be provided; winners will likely need to secure a visa and work authorization.

Note

Organizers reserve the right to reallocate prize money between categories in the event of a significant disparity in the number or quality of submissions.

[1]
Due to legal restrictions, U.S. researchers may be unable to co-author papers with scholars from certain countries or institutions. Participants are responsible for verifying their eligibility.

[2]
The visit is contingent on obtaining the necessary permission to work in the United States and fulfilling all other legal, institutional, and administrative requirements.

Deadline
The final submission deadline is November 25th, 2025 (AoE). Participants may continue modifying their PRs until this time.

Guidelines
Eligibility: Participation is free and open to all. However, for legal reasons, individuals affiliated with institutions that appear in the sections of the Restricted Foreign Research Institutions list are not eligible for the reward outcomes of the challenge — including co-authorship on the white paper summarizing the challenge findings and submissions.

Registration: To participate in the challenge, participants must (1) open a Pull Request (PR) on TopoBench and (2) fill out the Registration Google Form with PR and team information. Each submission (i.e., each PR) must be accompanied by a Registration Form to be valid.

Picking a dataset: Please refer to the open Pull Requests in TopoBench (see open PRs) to see which datasets are already being implemented in each category.

Submission: For a submission to be valid, teams must:

Submit a valid PR before the deadline.

Fill out the registration form before the deadline.

Ensure the PR passes all tests.

Tag the PR with the appropriate category (one of: category-A1, category-A2, category-B1, category-B2).

Respect all submission requirements.

Dataset loaders:

Each PR may contain at most one dataset loader.

A loader may support multiple datasets, but in such cases, the PR should include separate configuration files for each dataset. (Example: TopoBench uses the “TUDataset” class from PyG and provides multiple configuration files for its different datasets.)

Teams:

Teams are allowed, with a maximum of 2 members. (If you wish to form a larger team, please contact the organizers — see the Questions section — for discussion and approval.)

The same team can submit multiple datasets through different PRs. Make sure to register on the Google Form (link) for each PR.

The same team can participate in multiple challenge categories.

Early submissions:

We encourage participants to submit PRs early. This allows time to resolve potential issues.

In cases where multiple high-quality submissions cover overlapping datasets, earlier submissions will be given priority consideration.

Submission Requirements
A submission consists of a Pull Request (PR) to TopoBench. The PR title must be:

Category: [A1|A2|B1|B2];  Team name: <team name>;  Dataset: <Dataset Name>

Submissions may implement datasets already proposed in the literature or novel datasets. For novel datasets, neither the challenge nor its related publications will prevent you from publishing your own work — you retain full credit for your data. We only ask that the data be hosted on an open-source platform.

Requirements for Mission A (Categories A1 and A2)
{name}_dataset_loader.py or {name}_datasets_loader.py

Store in topobench/data/loaders/{domain} where {domain} is the dataset’s domain (e.g., graph).

Define a class {Name}DatasetLoader implementing load_dataset() that loads the entire dataset (optionally with pre-defined splits).

This class must inherit from data.loaders.base.AbstractLoader.

(Only if necessary) {name}_dataset.py or {name}_datasets.py

Store in topobench/data/datasets.

Define a class {Name} implementing the required interfaces for InMemoryDataset. See existing examples in the directory for the required functions.

Testing

All contributed files must pass pre-existing unit tests in test/data.

Any contributed methods that are not currently tested must be accompanied by new test files stored in the appropriate subdirectory of test/data. Each contributed test file should include test functions that correspond one-to-one with contributed functions (as in existing tests).

TopoBench uses Codecov to measure coverage. Your PR must match or exceed 93% coverage. You can find the Codecov report as a bot comment on your PR after CI runs.

Include a pipeline test showing that a model (any model already in TopoBench) can train on your dataset. Specifically, fill out test/pipeline/test_pipeline.py with a dataset and model configuration of your choice to prove the pipeline runs successfully. We do not evaluate submissions by training performance.

Tip

For a step-by-step example, see the tutorial: tutorial_add_custom_dataset.ipynb.

Requirements for Mission B (Categories B1 and B2)
This mission may require significant contributions and modifications to TopoBench, so explicit, exhaustive requirements are not feasible. Participants are encouraged to follow the general structure and logic of TopoBench as much as possible; however, given the complexity of these tasks, this constraint is flexible.

Minimum expectations

Coverage: Match or exceed the existing 93% Codecov coverage target.

Testing: Include at least one pipeline test demonstrating that a model (either existing or newly contributed) can train on the contributed dataset/task for a limited number of epochs (see point 3.d of the Mission A submission requirements for an example).

Coordination and support

We are happy to discuss implementation and testing ideas. Please reach out at topological.intelligence@gmail.com.

Note

For very large datasets (Mission B1), write tests that operate on small slices of the data to keep CI/runtime reasonable and avoid lengthy tests.

Evaluation
Award Categories
One winning submission will be crowned per mission category:

Category A.1: Best “Expanding Graph-Based Datasets” implementation

Category A.2: Best “Expanding Higher-Order Dataset” implementation

Category B.1: Best “Large-Scale Inductive Dataset” implementation

Category B.2: Best “New TDL Benchmark Task” implementation

Additionally, two teams will be selected for invited visits across all mission categories based on overall quality, level of difficulty, and impact of contribution. Honorable mentions may also be awarded. Exceptional attention will be given to any successful B1 Bonus submission.

Evaluation Procedure
A panel of TopoBench maintainers and collaborators will vote using the Condorcet method to select the best submission in each mission category.

Evaluation criteria include:

Correctness: Does the submission implement the dataset correctly? Is it reasonable and well-defined?

Code quality: How readable and clean is the implementation? How well does the submission respect the requirements?

Documentation & tests: Are docstrings clear? Are unit tests robust?

Important

These criteria do not reward final model performance on the dataset. The goal is to deliver well-written, usable datasets/infrastructure that enable further experimentation and insight.

A panel of TopoBench developers and TDL experts will decide on the two teams to be invited for visits, pending interest as indicated in their Registration Forms. Internship opportunities and cash prizes are not mutually exclusive.

Questions
Feel free to contact the organizers at topological.intelligence@gmail.com.

Important

👉 Now live: Join us in expanding the data landscape of Topological Deep Learning!

Related References
Bibliography
As a support to participants, this section lists related references that propose topological liftings or may help in defining novel ones.

Papillon, M., Sanborn, S., Hajij, M., & Miolane, N. (2023). Architectures of Topological Deep Learning: A Survey on Topological Neural Networks. arXiv:2304.10031.

Hajij, M., Zamzmi, G., Papamarkou, T., Miolane, N., Guzmán-Sáenz, A., Ramamurthy, K. N., et al. (2022). Topological deep learning: Going beyond graph data. arXiv:2206.00606.

Baccini, F., Geraci, F., & Bianconi, G. (2022). Weighted simplicial complexes and their representation power of higher-order network data and topology. Physical Review E, 106(3), 034319.

Barbarossa, S., & Sardellitti, S. (2020). Topological signal processing over simplicial complexes. IEEE Transactions on Signal Processing, 68, 2992–3007.

Battiloro, C., Spinelli, I., Telyatnikov, L., Bronstein, M., Scardapane, S., & Di Lorenzo, P. (2023). From latent graph to latent topology inference: differentiable cell complex module. arXiv:2305.16174.

Benson, A. R., Gleich, D. F., & Higham, D. J. (2021). Higher-order network analysis takes off, fueled by classical ideas and new data. arXiv:2103.05031.

Bodnar, C., Frasca, F., Otter, N., Wang, Y., Lió, P., Montúfar, G. F., & Bronstein, M. (2021). Weisfeiler and Lehman go cellular: CW networks. In Advances in Neural Information Processing Systems (Vol. 34, pp. 2625–2640).

Bodnar, C., Frasca, F., Wang, Y., Otter, N., Montúfar, G. F., Lió, P., & Bronstein, M. (2021, July). Weisfeiler and Lehman go topological: Message passing simplicial networks. In International Conference on Machine Learning (pp. 1026–1037). PMLR.

Elshakhs, Y. S., Deliparaschos, K. M., Charalambous, T., Oliva, G., & Zolotas, A. (2024). A Comprehensive Survey on Delaunay Triangulation: Applications, Algorithms, and Implementations Over CPUs, GPUs, and FPGAs. IEEE Access.

Ferri, M., Bergomi, D. M. G., & Zu, L. (2018). Simplicial complexes from graphs towards graph persistence. arXiv:1805.10716.

Gao, Y., Zhang, Z., Lin, H., Zhao, X., Du, S., & Zou, C. (2020). Hypergraph learning: Methods and practices. IEEE Transactions on Pattern Analysis and Machine Intelligence, 44(5), 2548–2566.

Hajij, M., & Istvan, K. (2021). Topological deep learning: Classification neural networks. arXiv:2102.08354.

Hajij, M., Zamzmi, G., Papamarkou, T., Miolane, N., Guzmán-Sáenz, A., & Ramamurthy, K. N. (2022). Higher-order attention networks. arXiv:2206.00606, 2(3), 4.

Hajij, M., Zamzmi, G., Papamarkou, T., Guzman-Saenz, A., Birdal, T., & Schaub, M. T. (2023). Combinatorial complexes: bridging the gap between cell complexes and hypergraphs. In 2023 57th Asilomar Conference on Signals, Systems, and Computers (pp. 799–803). IEEE.

Hoppe, J., & Schaub, M. T. (2024). Representing Edge Flows on Graphs via Sparse Cell Complexes. In Learning on Graphs Conference (pp. 1–1). PMLR.

Jogl, F., Thiessen, M., & Gärtner, T. (2022). Reducing learning on cell complexes to graphs. In ICLR 2022 Workshop on Geometrical and Topological Representation Learning.

Kahle, M. (2007). The neighborhood complex of a random graph. Journal of Combinatorial Theory, Series A, 114(2), 380–387.

Kahle, M., et al. (2014). Topology of random simplicial complexes: a survey. AMS Contemporary Mathematics, 620, 201–222.

Kahle, M. (2016). Random simplicial complexes. arXiv:1607.07069.

Liu, X., & Zhao, C. (2023). Eigenvector centrality in simplicial complexes of hypergraphs. Chaos: An Interdisciplinary Journal of Nonlinear Science, 33(9).

Lucas, M., Gallo, L., Ghavasieh, A., Battiston, F., & De Domenico, M. (2024). Functional reducibility of higher-order networks. arXiv:2404.08547.

Ruggeri, N., Battiston, F., & De Bacco, C. (2024). Framework to generate hypergraphs with community structure. Physical Review E, 109(3), 03430.

Reference Datasets Table
The following tables list potential datasets that can be implemented for the challenge. However, participants are not limited to these datasets and are welcome to propose others, either existing or novel (as long as they are hosted on an open-source platform). We also encourage participants to explore pointcloud datasets representing 3D objects, which can be lifted to higher-order domains using TopoBench transforms.

Important

By default, if the selected dataset is available in PyTorch Geometric (PyG), the submission should use the PyG adaptation. In practice, your dataloader should build on the appropriate PyG loaders.

Graphs
Dataset

Reference

TwiBot-20

A Comprehensive Twitter Bot Detection Benchmark (Large).

TwiBot-22

Towards Graph-Based Twitter Bot Detection (Large).

MGTAB

A Multi-Relational Graph-Based Twitter Account Detection Benchmark.

GraphLand (14 datasets)

GraphLand: Evaluating Graph Machine Learning Models on Diverse Industrial Data.

Karate Club (7 datasets)

Karate Club: An API Oriented Open-Source Python Framework for Unsupervised…

Coauth-DBLP

Classification of Edge-dependent Labels of Nodes in Hypergraphs.

Coauth-AMiner

Classification of Edge-dependent Labels of Nodes in Hypergraphs.

Email-Enron

Classification of Edge-dependent Labels of Nodes in Hypergraphs.

Email-Eu

Classification of Edge-dependent Labels of Nodes in Hypergraphs.

Stack-Physics

Classification of Edge-dependent Labels of Nodes in Hypergraphs.

Wiki-CS

Wiki-CS: A Wikipedia-Based Benchmark for Graph Neural Networks.

FacebookPagePage

Multi-Scale Attributed Node Embedding (MUSAE).

GitHub (MUSAE)

Multi-Scale Attributed Node Embedding (MUSAE).

LastFMAsia

Multi-Scale Attributed Node Embedding (MUSAE).

Deezer Europe

Multi-Scale Attributed Node Embedding (MUSAE).

Twitch

Multi-Scale Attributed Node Embedding (MUSAE).

GemsecDeezer

GEMSEC: Graph Embedding with Self Clustering.

Torus dataset

Topological Blindspots: Understanding and Extending Topological Deep Learning through the Lens of Expressivity.

QM9

MoleculeNet: A Benchmark for Molecular Machine Learning.

Hypergraphs
Dataset

Reference

CornellTemporalHyperGraphDataset

Simplicial Closure and Higher-Order Link Prediction.

CornellLabelledNodes

Various references.

RHG-3

Hypergraph Isomorphism Computation.

RHG-10

Hypergraph Isomorphism Computation.

IMDB-Dir-Form

Hypergraph Isomorphism Computation.

IMDB-Dir-Genre

Hypergraph Isomorphism Computation.

Steam-Player

Hypergraph Isomorphism Computation.

Twitter-Friend

Hypergraph Isomorphism Computation.


_____________________________




A Comprehensive Benchmark Suite for Topological Deep Learning
Assess how your model compares against state-of-the-art topological neural networks.

Lint Test Codecov Docs Python license slack

Overview • Get Started • Tutorials • Neural Networks • Liftings and Transforms • Datasets • References

🏆 Announcing the Topological Deep Learning Challenge 2025!
We are excited to announce that TopoBench is hosting the Topological Deep Learning Challenge 2025, as part of the Topology, Algebra, and Geometry in Data Science (TAG-DS) 2025 conference. This year's theme is "Expanding the Data Landscape": participants are invited to implement a new or existing dataset within TopoBench.

TDL Challenge 2025 Flyer

Why Participate?
🌟 Co-author a white paper for the Proceedings of Machine Learning Research (PMLR).
🌟 Win cash prizes up to $800 USD, sponsored by Arlequin AI.
🌟 Secure internship opportunities at UC Santa Barbara and EPFL.
Deadline for submission: November 25th, 2025 (AoE)

For more details, rules, and to get started, please visit the link to the challenge website.

📌 Overview
TopoBench (TB) is a modular Python library designed to standardize benchmarking and accelerate research in Topological Deep Learning (TDL). In particular, TB allows to train and compare the performances of all sorts of Topological Neural Networks (TNNs) across the different topological domains, where by topological domain we refer to a graph, a simplicial complex, a cellular complex, or a hypergraph. For detailed information, please refer to the TopoBench: A Framework for Benchmarking Topological Deep Learning paper.



The main pipeline trains and evaluates a wide range of state-of-the-art TNNs and Graph Neural Networks (GNNs) (see ⚙️ Neural Networks) on numerous and varied datasets and benchmark tasks (see 📚 Datasets ). Additionally, the library offers the ability to transform, i.e. lift, each dataset from one topological domain to another (see 🚀 Liftings and Transforms), enabling for the first time an exhaustive inter-domain comparison of TNNs.

🧩 Get Started
Create Environment
First, ensure conda is installed:

conda --version
If not, we recommend intalling Miniconda following the official command line instructions.

Then, clone and navigate to the TopoBench repository

git clone git@github.com:geometric-intelligence/topobench.git
cd TopoBench
Next, set up and activate a conda environment tb with Python 3.11.3:

conda create -n tb python=3.11.3
conda activate tb
If working with GPUs, check the CUDA version of your machine:

which nvcc && nvcc --version
and ensure that it matches the CUDA version specified in the env_setup.sh file (CUDA=cpu by default for a broader compatibility). If it does not match, update env_setup.sh accordingly by changing both the CUDA and TORCH environment variables to compatible values as specified on this website.

Next, set up the environment with the following command.

source env_setup.sh
This command installs the TopoBench library and its dependencies.

Run Training Pipeline
Once the setup is completed, train and evaluate a neural network by running the following command:

python -m topobench 
Customizing Experiment Configuration
Thanks to hydra implementation, one can easily override the default experiment configuration through the command line. For instance, the model and dataset can be selected as:

python -m topobench model=cell/cwn dataset=graph/MUTAG
Remark: By default, our pipeline identifies the source and destination topological domains, and applies a default lifting between them if required.

Transforms allow you to modify your data before processing. There are two main ways to configure transforms: individual transforms and transform groups.

Configuring Individual Transforms
Configuring Transform Groups
By mastering these configuration options, you can easily customize your experiments to suit your specific needs, from simple model and dataset selections to complex data transformation pipelines. ---
Additional Notes
Automatic Lifting: By default, our pipeline identifies the source and destination topological domains and applies a default lifting between them if required.
Fine-Grained Configuration: The same CLI override mechanism applies when modifying finer configurations within a CONFIG GROUP.
Please refer to the official hydra documentation for further details.
🚲 Experiments Reproducibility
To reproduce Table 1 from the TopoBench: A Framework for Benchmarking Topological Deep Learning paper, please run the following command:

bash scripts/reproduce.sh
Remark: We have additionally provided a public W&B (Weights & Biases) project with logs for the corresponding runs (updated on June 11, 2024).

⚓ Tutorials
Explore our tutorials for further details on how to add new datasets, transforms/liftings, and benchmark tasks.

⚙️ Neural Networks
We list the neural networks trained and evaluated by TopoBench, organized by the topological domain over which they operate: graph, simplicial complex, cellular complex or hypergraph. Many of these neural networks were originally implemented in TopoModelX.

Pointclouds
Model	Reference
DeepSets	Deep Sets
Graphs
Model	Reference
GAT	Graph Attention Networks
GIN	How Powerful are Graph Neural Networks?
GCN	Semi-Supervised Classification with Graph Convolutional Networks
GraphMLP	Graph-MLP: Node Classification without Message Passing in Graph
GPS	Recipe for a General, Powerful, Scalable Graph Transformer
Simplicial Complexes
Model	Reference
SAN	Simplicial Attention Neural Networks
SCCN	Efficient Representation Learning for Higher-Order Data with Simplicial Complexes
SCCNN	Convolutional Learning on Simplicial Complexes
SCN	Simplicial Complex Neural Networks
Cellular Complexes
Model	Reference
CAN	Cell Attention Network
CCCN	Inspired by A learning algorithm for computational connected cellular network, implementation adapted from Generalized Simplicial Attention Neural Networks
CXN	Cell Complex Neural Networks
CWN	Weisfeiler and Lehman Go Cellular: CW Networks
Hypergraphs
Model	Reference
AllDeepSet	You are AllSet: A Multiset Function Framework for Hypergraph Neural Networks
AllSetTransformer	You are AllSet: A Multiset Function Framework for Hypergraph Neural Networks
EDGNN	Equivariant Hypergraph Diffusion Neural Operators
UniGNN	UniGNN: a Unified Framework for Graph and Hypergraph Neural Networks
UniGNN2	UniGNN: a Unified Framework for Graph and Hypergraph Neural Networks
Combinatorial Complexes
Model	Reference
GCCN	TopoTune: A Framework for Generalized Combinatorial Complex Neural Networks
Remark: TopoBench includes TopoTune, a comprehensive framework for easily designing new, general TDL models on any domain using any (graph) neural network as a backbone. Please check out the extended TopoTune wiki page for further details on how to leverage this framework to define and train customized topological neural network architectures.

Non-relational Models
Model	Reference
MLP	Standard implementation of a Multi-Layer Perceptron.
Remark: Note that MLP only works in single-graph transductive settings or with datasets where all graphs have the same number of nodes.

🚀 Liftings and Transforms
We list the liftings used in TopoBench to transform datasets. Here, a lifting refers to a function that transforms a dataset defined on a topological domain (e.g., on a graph) into the same dataset but supported on a different topological domain (e.g., on a simplicial complex).

Structural Liftings
The structural lifting is responsible for the transformation of the underlying relationships or elements of the data. For instance, it might determine how nodes and edges in a graph are mapped into triangles and tetrahedra in a simplicial complex. This structural transformation can be further categorized into connectivity-based, where the mapping relies solely on the existing connections within the data, and feature-based, where the data's inherent properties or features guide the new structure.

We enumerate below the structural liftings currently implemented in TopoBench; please check out the provided description links for further details.

Remark:: Most of these liftings are adaptations of winner submissions of the ICML TDL Challenge 2024 (paper | repo); see the Structural Liftings wiki for a complete list of compatible liftings.

Graph to Simplicial Complex
Name	Type	Description
DnD Lifting	Feature-based	Wiki page
Random Latent Clique Lifting	Connectivity-based	Wiki page
Line Lifting	Connectivity-based	Wiki page
Neighbourhood Complex Lifting	Connectivity-based	Wiki page
Graph Induced Lifting	Connectivity-based	Wiki page
Eccentricity Lifting	Connectivity-based	Wiki page
Feature‐Based Rips Complex	Both connectivity and feature-based	Wiki page
Clique Lifting	Connectivity-based	Wiki page
K-hop Lifting	Connectivity-based	Wiki page
Graph to Cell Complex
Name	Type	Description
Discrete Configuration Complex	Connectivity-based	Wiki page
Cycle Lifting	Connectivity-based	Wiki page
Graph to Hypergraph
Name	Type	Description
Expander Hypergraph Lifting	Connectivity-based	Wiki page
Kernel Lifting	Both connectivity and feature-based	Wiki page
Mapper Lifting	Connectivity-based	Wiki page
Forman‐Ricci Curvature Coarse Geometry Lifting	Connectivity-based	Wiki page
KNN Lifting	Feature-based	Wiki page
K-hop Lifting	Connectivity-based	Wiki page
Pointcloud to Simplicial
Name	Type	Description
Delaunay Lifting	Feature-based	Wiki page
Random Flag Complex	Feature-based	Wiki page
Pointcloud to Hypergraph
Name	Type	Description
Mixture of Gaussians MST lifting	Feature-based	Wiki page
PointNet Lifting	Feature-based	Wiki page
Voronoi Lifting	Feature-based	Wiki page
Simplicial to Combinatorial
Name	Type	Description
Coface Lifting	Connectivity-based	Wiki page
Hypergraph to Combinatorial
Name	Type	Description
Universal Strict Lifting	Connectivity-based	Wiki page
Feature Liftings
Feature liftings address the transfer of data attributes or features during mapping, ensuring that the properties associated with the data elements are consistently preserved in the new representation.

Name	Description	Supported Domains
ProjectionSum	Projects r-cell features of a graph to r+1-cell structures utilizing incidence matrices (B_{r}).	All
ConcatenationLifting	Concatenate r-cell features to obtain r+1-cell features.	Simplicial
Data Transformations
Specially useful in pre-processing steps, these are the general data manipulations currently implemented in TopoBench:

Transform	Description
OneHotDegreeFeatures	Adds the node degree as one hot encodings to the node features.
NodeFeaturesToFloat	Converts the node features of the input graph to float.
NodeDegrees	Calculates the node degrees of the input graph.
KeepSelectedDataFields	Keeps only the selected fields of the input data.
KeepOnlyConnectedComponent	Keep only the largest connected components of the input graph.
InfereRadiusConnectivity	Generates the radius connectivity of the input point cloud.
InfereKNNConnectivity	Generates the k-nearest neighbor connectivity of the input point cloud.
IdentityTransform	An identity transform that does nothing to the input data.
EqualGausFeatures	Generates equal Gaussian features for all nodes.
CalculateSimplicialCurvature	Calculates the simplicial curvature of the input graph.
LapPE	Computes Laplacian eigenvectors positional encodings.
RWSE	Computes Random Walk structural encodings.
CombinedPSEs	Computes one or several positional and/or structural encodings.
📚 Datasets
Graph
Dataset	Task	Description	Reference
Cora	Classification	Cocitation dataset.	Source
Citeseer	Classification	Cocitation dataset.	Source
Pubmed	Classification	Cocitation dataset.	Source
MUTAG	Classification	Graph-level classification.	Source
PROTEINS	Classification	Graph-level classification.	Source
NCI1	Classification	Graph-level classification.	Source
NCI109	Classification	Graph-level classification.	Source
IMDB-BIN	Classification	Graph-level classification.	Source
IMDB-MUL	Classification	Graph-level classification.	Source
REDDIT	Classification	Graph-level classification.	Source
Amazon	Classification	Heterophilic dataset.	Source
Minesweeper	Classification	Heterophilic dataset.	Source
Empire	Classification	Heterophilic dataset.	Source
Tolokers	Classification	Heterophilic dataset.	Source
US-county-demos	Regression	In turn each node attribute is used as the target label.	Source
ZINC	Regression	Graph-level regression.	Source
Simplicial
Dataset	Task	Description	Reference
Mantra	Classification, Multi-label Classification	Predict topological attributes of manifold triangulations	Source (This project includes third-party datasets. See third_party_licenses.txt for licensing information.)
Hypergraph
Dataset	Task	Description	Reference
Cora-Cocitation	Classification	Cocitation dataset.	Source
Citeseer-Cocitation	Classification	Cocitation dataset.	Source
PubMed-Cocitation	Classification	Cocitation dataset.	Source
Cora-Coauthorship	Classification	Cocitation dataset.	Source
DBLP-Coauthorship	Classification	Cocitation dataset.	Source
🔍 References
To learn more about TopoBench, we invite you to read the paper:

@article{telyatnikov2024topobench,
      title={TopoBench: A Framework for Benchmarking Topological Deep Learning}, 
      author={Lev Telyatnikov and Guillermo Bernardez and Marco Montagna and Pavlo Vasylenko and Ghada Zamzmi and Mustafa Hajij and Michael T Schaub and Nina Miolane and Simone Scardapane and Theodore Papamarkou},
      year={2024},
      eprint={2406.06642},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2406.06642}, 
}
If you find TopoBench useful, we would appreciate if you cite us!

🐭 Additional Details
Hierarchy of configuration files
More information regarding Topological Deep Learning
📢 Get in Touch!
We are always open to collaborations and discussions on TDL research.
Feel free to reach out via email if you want to collaborate, do your thesis with our team, or open a discussion for various opportunities.

📧 Contact Email: topological.intelligence@gmail.com
▶️ YouTube Channel: Topological Intelligence