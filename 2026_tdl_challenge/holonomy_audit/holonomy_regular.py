"""Degree-regular follow-up: triangle counting with the degree shortcut removed.

Chen et al. (2020) prove that message-passing networks can subgraph-count
essentially only star-shaped patterns -- i.e. degree statistics -- and evaluate
substructure counting on random regular graphs, where degrees are constant by
construction. Following that protocol, this script re-runs the holonomy audit's
triangle-counting task on random ``r``-regular graphs with fixed ``(n, r)``:
every graph in the family has an identical degree sequence, so the only signal
distinguishing targets is the actual wiring. Node features are i.i.d. Gaussian
(random identifiers to break symmetry).

With targets standardised on the training split, predicting the training mean
scores MSE ~= 1.0 on the test split; values below 1 indicate genuine structural
signal. For bundle models we additionally apply the flattening intervention of
``holonomy_lobotomy`` after training.

Output: ``regular_results.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

sys.path.insert(0, str(Path(__file__).parent))
from holonomy_experiment import (  # noqa: E402
    UNIVERSE_KWARGS,
    SheafTaskModel,
    measure_holonomy,
    train_eval,
)
from holonomy_lobotomy import lobotomize  # noqa: E402

FAMILIES = ["nsp", "nsd"]
SHEAF_TYPES = ["diag", "bundle", "general"]
FEATURE_DIM = UNIVERSE_KWARGS["feature_dim"]  # match the audited models


def make_regular_family(n_graphs, n, r, seed, features="gaussian"):
    """Generate random r-regular graphs with triangle-count targets.

    Parameters
    ----------
    n_graphs : int
        Number of graphs.
    n : int
        Nodes per graph (fixed across the family).
    r : int
        Degree (fixed across the family).
    seed : int
        Base random seed; graph ``i`` uses ``1000 * seed + i`` for both the
        wiring and the Gaussian node features.
    features : str, optional
        ``"gaussian"`` (i.i.d. random identifiers; default) or ``"constant"``
        (all-ones features -- the anonymous-node setting, where 1-WL-style
        limits apply cleanly).

    Returns
    -------
    list of torch_geometric.data.Data
        Graphs with ``x`` (random features), ``edge_index`` and scalar ``y``
        (total triangle count).
    """
    graphs = []
    for i in range(n_graphs):
        s = 1000 * seed + i
        g = nx.random_regular_graph(r, n, seed=s)
        tri = sum(nx.triangles(g).values()) // 3
        edge_index = to_undirected(
            torch.tensor(list(g.edges), dtype=torch.long).t().contiguous()
        )
        if features == "constant":
            x = torch.ones(n, FEATURE_DIM)
        else:
            gen = torch.Generator().manual_seed(s)
            x = torch.randn(n, FEATURE_DIM, generator=gen)
        graphs.append(
            Data(x=x, edge_index=edge_index, y=torch.tensor(float(tri)))
        )
    return graphs


def run_one(family, sheaf_type, seed, epochs, n_train, n_test, n, r,
            features="gaussian"):
    """Train one model on regular-graph counting; lobotomise if bundle.

    Parameters
    ----------
    family : str
        ``"nsp"`` or ``"nsd"``.
    sheaf_type : str
        ``"diag"``, ``"bundle"`` or ``"general"``.
    seed : int
        Random seed (data and initialisation).
    epochs : int
        Training epochs.
    n_train, n_test : int
        Number of train / test graphs.
    n, r : int
        Nodes per graph and (constant) degree.
    features : str, optional
        Node-feature mode: ``"gaussian"`` or ``"constant"``.

    Returns
    -------
    dict
        Metric, init/final holonomy summaries, and -- for bundle models --
        the post-flattening metric.
    """
    task = "triangle_counting"
    train_graphs = make_regular_family(n_train, n, r, seed=100 + seed,
                                       features=features)
    test_graphs = make_regular_family(n_test, n, r, seed=900 + seed,
                                      features=features)
    torch.manual_seed(seed)
    model = SheafTaskModel(sheaf_type, task, model_family=family)
    holo_init = measure_holonomy(model, test_graphs)
    metric = train_eval(model, train_graphs, test_graphs, task, epochs)
    holo_final = measure_holonomy(model, test_graphs)
    record = dict(
        family=family,
        sheaf_type=sheaf_type,
        seed=seed,
        n=n,
        r=r,
        features=features,
        metric_name="mse_log1p_std",
        metric=metric,
        init_twist=holo_init["twist_mean"],
        final_twist=holo_final["twist_mean"],
        final_gain=holo_final["gain_mean"],
    )
    if sheaf_type == "bundle":
        lobotomize(model)
        record["metric_lobotomy"] = train_eval(
            model, train_graphs, test_graphs, task, 0
        )
        record["twist_lobotomy"] = measure_holonomy(model, test_graphs)[
            "twist_mean"
        ]
    print(
        f"[{family}|{sheaf_type:8s}|s{seed}|{features[:5]}] mse={metric:.4f}"
        + (
            f" -> flattened {record['metric_lobotomy']:.4f}"
            if sheaf_type == "bundle"
            else ""
        )
        + f"   twist {holo_init['twist_mean']:.3f}->{holo_final['twist_mean']:.3f}",
        flush=True,
    )
    return record


def main():
    """Run the regular-graph grid and write ``regular_results.json``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--n-train", type=int, default=30)
    parser.add_argument("--n-test", type=int, default=8)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--r", type=int, default=6)
    parser.add_argument("--features", type=str, default="gaussian",
                        choices=["gaussian", "constant"])
    parser.add_argument(
        "--out",
        type=str,
        default=str(Path(__file__).parent / "regular_results.json"),
    )
    args = parser.parse_args()

    records = [
        run_one(
            family, sheaf_type, seed,
            args.epochs, args.n_train, args.n_test, args.n, args.r,
            features=args.features,
        )
        for family in FAMILIES
        for sheaf_type in SHEAF_TYPES
        for seed in range(args.seeds)
    ]
    Path(args.out).write_text(json.dumps(records, indent=2))
    print(f"\nWrote {len(records)} records -> {args.out}")


if __name__ == "__main__":
    main()
