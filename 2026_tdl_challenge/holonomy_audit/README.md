# Do sheaf networks use their geometry?

This directory contains supplementary analysis for Team Sheafu's Neural Sheaf Propagation (NSP) submission to the Topological Deep Learning Challenge 2026, Track 1. The analysis opens trained sheaf neural networks and looks at the geometry they learn around triangles.

## Question and findings

The analysis asks two things: does a trained sheaf network actually develop cycle geometry, and do its predictions come to depend on it?

In these experiments the pattern is consistent. Training a model to count triangles makes its SO(2) maps rotate features around each triangle about 40x more than at initialisation; training the same architecture to detect communities leaves those maps nearly flat. The geometry is recruited, and only for the task that should need it. With enough data the model also starts to rely on it: flatten the learned connection afterwards and the error grows sharply.

What the geometry never does is count. A plain ridge regression on a handful of degree statistics still beats every sheaf model, so the networks do not even match that simple shortcut. And once we remove the shortcut, using regular graphs where every node has the same degree so that reading the wiring is the only way to tell graphs apart, no model does better than predicting the average, at any data scale and however large the twist grows. The geometry gets recruited and becomes load-bearing, but it never turns into a working cycle counter.

The companion paper is *Measuring Triangle Holonomy in Sheaf Neural Networks* ([arXiv:2607.19514](https://arxiv.org/abs/2607.19514)).

## Files

```
holonomy_audit/
├── README.md
├── holonomy_audit.ipynb          the analysis notebook
├── build_holonomy_notebook.py    builds the notebook
├── holonomy_experiment.py        main sweep
├── holonomy_lobotomy.py          flattening intervention
├── holonomy_regular.py           regular-graph counting
├── degree_baselines.py           degree-shortcut baseline
├── verify_diagnostic.py          measurement-tool checks
├── scaling_experiment.py         data-scaling sweep
├── scaling_baselines.py          baselines at each size
├── bigtest_eval.py               200-graph re-check
└── *_results.json  (11 files)    saved outputs the notebook reads
```

| File | Purpose |
|---|---|
| `holonomy_audit.ipynb` | The analysis itself: the notebook you read. |
| `build_holonomy_notebook.py` | Builds `holonomy_audit.ipynb`. The notebook's text and code live here as a plain Python script. |
| `holonomy_experiment.py` | The main sweep. Trains the small sheaf models (both NSP and NSD dynamics, across the three map families, on both tasks, over many seeds) and records the twist, gain, and flip each one learns. |
| `holonomy_lobotomy.py` | The intervention. Takes a trained model, deletes its learned geometry by setting every restriction map back to the identity, and re-measures it, to test whether the model actually depends on that geometry. |
| `holonomy_regular.py` | Runs triangle counting on fixed-degree (regular) graphs, where every node has the same degree, so the degree shortcut is gone and only the wiring distinguishes one graph from another. |
| `degree_baselines.py` | Fits the shortcut itself: simple ridge regressions that predict triangle counts from structural summaries like node count, edge count, and degree statistics. This is the bar the sheaf models are measured against. |
| `verify_diagnostic.py` | Checks that the measurement tool is trustworthy, not the models: gauge invariance, numerical stability, and that flattening a connection really does zero out the readouts. |
| `scaling_experiment.py` | Repeats the counting experiment at training-set sizes from 30 up to 3000 graphs, to see how the findings change as the model gets more data. |
| `scaling_baselines.py` | Fits the constant predictor and the degree-statistics baseline at each of those training-set sizes, so the models can be compared to the shortcut at every scale. |
| `bigtest_eval.py` | Re-checks the main scaling results on 200 freshly generated test graphs instead of the original 8, to make sure the conclusions are not an artefact of a small test set. |
| `*_results.json` (11 files) | The saved outputs of the scripts above. The notebook reads these committed results and recomputes its figures and statistics. |

The measurement tool itself lives with the library it audits, outside this directory: `topobench/nn/backbones/graph/nsd_utils/sheaf_holonomy.py` and `holonomy_capture.py`, with 26 unit tests under `test/nn/backbones/graph/`.

## Reproducing the notebook

From this directory, with the TopoBench Python 3.11 environment active, run:

```bash
jupyter nbconvert --to notebook --execute --inplace holonomy_audit.ipynb
```

The notebook does no training and needs no GPU. Drawing the figures and computing the statistics needs `numpy`, `pandas`, `scipy`, and `matplotlib`. A few cells re-run the small checks that confirm the holonomy measurement tool is behaving correctly; those import `torch` and `torch_geometric` and run fine on CPU. Regenerating the JSON files from scratch is optional and is not needed to read or re-render the notebook.
