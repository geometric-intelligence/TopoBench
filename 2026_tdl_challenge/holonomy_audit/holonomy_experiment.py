"""Local CPU driver for the Holonomy Audit experiment.

Trains tiny diagonal / bundle / general sheaf models -- both NSP (wave) and NSD
(heat) dynamics -- on small GraphUniverse families for two tasks (community
detection, triangle counting) across a couple of homophily regimes, then reads
the *learned* per-triangle sheaf holonomy off each trained model and records it
alongside the task metric.

The question the resulting numbers answer: does a model whose restriction maps
can geometrically "feel" a triangle (non-trivial holonomy) actually count
triangles better than one whose connection is flat (diagonal maps)?

Runs entirely on CPU with no disk cache and no Modal. Output is a single JSON
(``holonomy_results.json``) consumed by the analysis notebook.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from graph_universe import GraphFamilyGenerator, GraphUniverse
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool

from topobench.nn.backbones.graph.nsd_utils.holonomy_capture import (
    sheaf_transports,
)
from topobench.nn.backbones.graph.nsd_utils.sheaf_holonomy import (
    enumerate_triangles,
    triangle_holonomies,
)
from topobench.nn.backbones.graph.nsd import NSDEncoder
from topobench.nn.backbones.graph.nsp import NSPEncoder

# ---------------------------------------------------------------------------
# Experiment grid. Degree is held high in every regime so triangles exist;
# homophily is the varied axis (community structure -> triangle density).
# ---------------------------------------------------------------------------
REGIMES = {
    "homophily_hi": dict(homophily_range=(0.9, 1.0)),
    "homophily_lo": dict(homophily_range=(0.0, 0.1)),
}
SHEAF_TYPES = ["diag", "bundle", "general"]
TASKS = ["community_detection", "triangle_counting"]
# Two dynamics on the SAME connection: NSP = wave (transport-preserving),
# NSD = heat (dissipative). The holonomy readout is identical; only the
# training pressure differs, so NSD is a robustness axis.
MODEL_FAMILIES = {"nsp": NSPEncoder, "nsd": NSDEncoder}

UNIVERSE_KWARGS = dict(
    K=20,
    feature_dim=15,
    center_variance=0.2,
    cluster_variance=0.4,
    edge_propensity_variance=1.0,
    seed=42,
)
FAMILY_BASE = dict(
    n_nodes_range=(100, 160),
    n_communities_range=(5, 7),
    avg_degree_range=(5.0, 7.0),
    power_law_exponent_range=(2.0, 2.5),
    degree_separation_range=(0.5, 1.0),
)
NUM_CLASSES = UNIVERSE_KWARGS["K"]  # community IDs live in [0, K)


class SheafTaskModel(torch.nn.Module):
    """A sheaf encoder plus a task head (node classifier or graph regressor).

    Parameters
    ----------
    sheaf_type : str
        Restriction-map family: ``"diag"``, ``"bundle"`` or ``"general"``.
    task : str
        ``"community_detection"`` (node classification) or
        ``"triangle_counting"`` (graph regression).
    model_family : str, optional
        ``"nsp"`` (wave) or ``"nsd"`` (heat). Default ``"nsp"``.
    hidden_dim : int
        Encoder hidden width.
    d : int
        Stalk dimension.
    num_layers : int
        Number of propagation layers.
    """

    def __init__(
        self, sheaf_type, task, model_family="nsp", hidden_dim=32, d=2,
        num_layers=2,
    ):
        super().__init__()
        self.task = task
        self.encoder = MODEL_FAMILIES[model_family](
            input_dim=UNIVERSE_KWARGS["feature_dim"],
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            d=d,
            sheaf_type=sheaf_type,
        )
        out_dim = NUM_CLASSES if task == "community_detection" else 1
        self.head = torch.nn.Linear(hidden_dim, out_dim)

    def propagation_model(self):
        """Return the underlying sheaf model regardless of family.

        Returns
        -------
        torch.nn.Module
            The ``InductiveDiscrete*`` model holding ``sheaf_learners[*].L``.
        """
        encoder = self.encoder
        if hasattr(encoder, "get_sheaf_propagation_model"):
            return encoder.get_sheaf_propagation_model()
        return encoder.get_sheaf_model()

    def forward(self, x, edge_index, batch):
        """Encode the graph and apply the task head.

        Parameters
        ----------
        x : torch.Tensor
            Node features ``[num_nodes, feature_dim]``.
        edge_index : torch.Tensor
            Edge indices ``[2, num_edges]``.
        batch : torch.Tensor
            Graph assignment vector ``[num_nodes]`` for pooling.

        Returns
        -------
        torch.Tensor
            Node logits ``[num_nodes, num_classes]`` for community detection, or
            graph predictions ``[num_graphs]`` for triangle counting.
        """
        emb = self.encoder(x, edge_index)
        if self.task == "community_detection":
            return self.head(emb)
        pooled = global_mean_pool(emb, batch)
        return self.head(pooled).squeeze(-1)


def make_family(task, regime_kwargs, n_graphs, seed):
    """Generate a list of PyG graphs for one task and regime.

    Parameters
    ----------
    task : str
        The GraphUniverse task name.
    regime_kwargs : dict
        Regime-specific family overrides (e.g. ``homophily_range``).
    n_graphs : int
        Number of graphs to sample.
    seed : int
        Family random seed.

    Returns
    -------
    list of torch_geometric.data.Data
        The generated graphs with ``x``, ``edge_index`` and task target ``y``.
    """
    universe = GraphUniverse(**UNIVERSE_KWARGS)
    family = GraphFamilyGenerator(
        universe=universe,
        n_graphs=n_graphs,
        seed=seed,
        **{**FAMILY_BASE, **regime_kwargs},
    )
    family.generate_family(show_progress=False)
    return family.to_pyg_graphs(task)


def train_eval(model, train_graphs, test_graphs, task, epochs, lr=0.01):
    """Train the model and return the held-out task metric.

    Parameters
    ----------
    model : SheafTaskModel
        The model to train (mutated in place).
    train_graphs, test_graphs : list of torch_geometric.data.Data
        Train / test graph splits.
    task : str
        Task name (selects loss and metric).
    epochs : int
        Number of training epochs.
    lr : float, optional
        Adam learning rate. Default ``0.01``.

    Returns
    -------
    float
        Test accuracy (community detection) or test MSE in log1p-standardised
        space (triangle counting; lower is better).
    """
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    train_loader = DataLoader(train_graphs, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=8)

    # For regression, standardise log1p targets using the train split.
    if task == "triangle_counting":
        y_train = torch.log1p(torch.stack([g.y for g in train_graphs]).float())
        y_mean, y_std = y_train.mean(), y_train.std().clamp_min(1e-6)
    else:
        y_mean = y_std = None

    def _target(batch):
        if task == "community_detection":
            return batch.y.long()
        return (torch.log1p(batch.y.float()) - y_mean) / y_std

    model.train()
    for _ in range(epochs):
        for batch in train_loader:
            opt.zero_grad()
            pred = model(batch.x, batch.edge_index, batch.batch)
            target = _target(batch)
            loss = (
                F.cross_entropy(pred, target)
                if task == "community_detection"
                else F.mse_loss(pred, target)
            )
            loss.backward()
            opt.step()

    model.eval()
    correct = total = 0
    sq_err = n_graphs = 0.0
    with torch.no_grad():
        for batch in test_loader:
            pred = model(batch.x, batch.edge_index, batch.batch)
            target = _target(batch)
            if task == "community_detection":
                correct += (pred.argmax(-1) == target).sum().item()
                total += target.numel()
            else:
                sq_err += F.mse_loss(pred, target, reduction="sum").item()
                n_graphs += target.numel()
    return correct / total if task == "community_detection" else sq_err / n_graphs


def measure_holonomy(model, test_graphs):
    """Read the learned per-triangle holonomy across the test graphs.

    Parameters
    ----------
    model : SheafTaskModel
        A trained model.
    test_graphs : list of torch_geometric.data.Data
        Graphs to probe (one forward pass each).

    Returns
    -------
    dict
        Aggregate holonomy statistics over every probed triangle: mean/median
        rotation angle, mean Frobenius magnitude, mean deconfounded polar twist,
        mean/median gain (``|det H|``), the fraction of frustrated (flipped)
        triangles, and the total triangle count.
    """
    model.eval()
    cols = {k: [] for k in ("angle", "magnitude", "twist", "gain", "flip")}
    prop = model.propagation_model()
    with torch.no_grad():
        for g in test_graphs:
            triangles = enumerate_triangles(g.edge_index)
            if not triangles:
                continue
            zeros = torch.zeros(g.num_nodes, dtype=torch.long)
            model(g.x, g.edge_index, zeros)  # populate sheaf_learners[*].L
            transports = sheaf_transports(prop, g.edge_index)
            out = triangle_holonomies(transports, triangles)
            for key in cols:
                cols[key].append(out[key])
    cat = {key: torch.cat(vals) for key, vals in cols.items()}
    return dict(
        angle_mean=cat["angle"].mean().item(),
        angle_median=cat["angle"].median().item(),
        twist_mean=cat["twist"].mean().item(),
        twist_median=cat["twist"].median().item(),
        magnitude_mean=cat["magnitude"].mean().item(),
        gain_mean=cat["gain"].mean().item(),
        gain_median=cat["gain"].median().item(),
        frustrated_frac=(cat["flip"] < 0).float().mean().item(),
        n_triangles=int(cat["angle"].numel()),
    )


def main():
    """Run the full grid and write ``holonomy_results.json``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--n-train", type=int, default=30)
    parser.add_argument("--n-test", type=int, default=12)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument(
        "--out",
        type=str,
        default=str(Path(__file__).parent / "holonomy_results.json"),
    )
    args = parser.parse_args()
    records = run_sweep(
        seeds=list(range(args.seeds)),
        epochs=args.epochs,
        n_train=args.n_train,
        n_test=args.n_test,
    )
    Path(args.out).write_text(json.dumps(records, indent=2))
    print(f"\nWrote {len(records)} records -> {args.out}")
    return records


def run_sweep(seeds, epochs, n_train, n_test):
    """Run the full map x task x regime x seed sweep with init->final holonomy.

    For every combination we measure holonomy at initialisation and again after
    training, so the notebook can show the *recruited* change (final - init) and
    correlate it with the task metric.

    Parameters
    ----------
    seeds : list of int
        Random seeds; one trained model per seed (for error bars).
    epochs : int
        Training epochs per model.
    n_train, n_test : int
        Number of train / test graphs per (task, regime).

    Returns
    -------
    list of dict
        One record per (regime, task, sheaf_type, seed) with the task metric and
        both the ``init`` and ``final`` holonomy statistics.
    """
    records = []
    for family in MODEL_FAMILIES:
        for regime in REGIMES:
            for task in TASKS:
                for seed in seeds:
                    for sheaf_type in SHEAF_TYPES:
                        records.append(
                            run_single(
                                family, regime, task, sheaf_type, seed,
                                epochs, n_train, n_test,
                            )
                        )
    return records


def run_single(
    family, regime, task, sheaf_type, seed, epochs, n_train, n_test
):
    """Train and probe ONE configuration (the Modal fan-out unit).

    Each call is fully self-contained: it generates its own data, trains one
    model, and measures holonomy at init and after training.

    Parameters
    ----------
    family : str
        Model family: ``"nsp"`` (wave) or ``"nsd"`` (heat).
    regime : str
        Key into :data:`REGIMES`.
    task : str
        ``"community_detection"`` or ``"triangle_counting"``.
    sheaf_type : str
        ``"diag"``, ``"bundle"`` or ``"general"``.
    seed : int
        Random seed (model init AND data family).
    epochs : int
        Training epochs.
    n_train, n_test : int
        Number of train / test graphs.

    Returns
    -------
    dict
        One result record with the task metric and init/final holonomy stats.
    """
    regime_kwargs = REGIMES[regime]
    metric_name = (
        "accuracy" if task == "community_detection" else "mse_log1p_std"
    )
    # Vary data per seed too, so error bars reflect data + init.
    train_graphs = make_family(task, regime_kwargs, n_train, seed=100 + seed)
    test_graphs = make_family(task, regime_kwargs, n_test, seed=900 + seed)
    torch.manual_seed(seed)
    model = SheafTaskModel(sheaf_type, task, model_family=family)
    holo_init = measure_holonomy(model, test_graphs)
    metric = train_eval(model, train_graphs, test_graphs, task, epochs)
    holo_final = measure_holonomy(model, test_graphs)
    record = dict(
        family=family,
        regime=regime,
        task=task,
        sheaf_type=sheaf_type,
        seed=seed,
        metric_name=metric_name,
        metric=metric,
        init={f"init_{k}": v for k, v in holo_init.items()},
        final=holo_final,
    )
    print(
        f"[{family}|{regime:12s}|{task:20s}|{sheaf_type:8s}|s{seed}] "
        f"{metric_name}={metric:.4f}  "
        f"twist {holo_init['twist_mean']:.3f}->{holo_final['twist_mean']:.3f}  "
        f"gain {holo_final['gain_mean']:.3f}  "
        f"frust {holo_final['frustrated_frac']:.3f}",
        flush=True,
    )
    return record


if __name__ == "__main__":
    main()
