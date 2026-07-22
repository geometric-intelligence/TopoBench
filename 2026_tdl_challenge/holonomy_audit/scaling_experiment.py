"""Data-scaling experiment: does more data convert geometry into counting?

Trains NSP diagonal and bundle models on triangle counting with
``n_train`` in {30, 100, 300, 1000, 3000}, on both GraphUniverse
(high-homophily) and fixed-(n, r) regular graphs (Gaussian identifiers),
holding the 8-graph test split FIXED across sizes (test seed 900 + seed) so
curves are paired within seed. Bundle runs additionally apply the flattening
intervention after training, so error, recruited twist and the flattening
effect are tracked jointly across dataset size. Ridge/constant baselines on
the identical splits come from ``scaling_baselines.py``.

Output: one record per (dataset, sheaf_type, n_train, seed) in
``scaling_results.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from holonomy_experiment import (  # noqa: E402
    REGIMES,
    SheafTaskModel,
    make_family,
    measure_holonomy,
    train_eval,
)
from holonomy_lobotomy import lobotomize  # noqa: E402
from holonomy_regular import make_regular_family  # noqa: E402

SIZES = [30, 100, 300, 1000, 3000]
SHEAF_TYPES = ["diag", "bundle"]
DATASETS = ["graphuniverse", "regular"]
TASK = "triangle_counting"


def build_split(dataset, n_train, seed, n_test=8):
    """Build one (train, test) split for the given dataset and size.

    Parameters
    ----------
    dataset : str
        ``"graphuniverse"`` (high-homophily regime) or ``"regular"``
        (fixed n=100, r=6, Gaussian identifiers).
    n_train : int
        Number of training graphs.
    seed : int
        Split seed; the test family always uses ``900 + seed`` so the test
        graphs are identical across ``n_train``.
    n_test : int, optional
        Number of test graphs. Default 8.

    Returns
    -------
    tuple of list
        ``(train_graphs, test_graphs)``.
    """
    if dataset == "graphuniverse":
        kw = REGIMES["homophily_hi"]
        return (make_family(TASK, kw, n_train, seed=100 + seed),
                make_family(TASK, kw, n_test, seed=900 + seed))
    return (make_regular_family(n_train, 100, 6, seed=100 + seed),
            make_regular_family(n_test, 100, 6, seed=900 + seed))


def run_one(dataset, sheaf_type, n_train, seed, epochs=60, n_test=8):
    """Train one model at one dataset size; flatten if bundle.

    Parameters
    ----------
    dataset : str
        ``"graphuniverse"`` or ``"regular"``.
    sheaf_type : str
        ``"diag"`` or ``"bundle"``.
    n_train : int
        Training-set size.
    seed : int
        Seed (data and initialisation).
    epochs : int, optional
        Training epochs (fixed across sizes). Default 60.
    n_test : int, optional
        Test-set size. Default 8.

    Returns
    -------
    dict
        Metric, init/final twist, and (bundle only) the flattened metric.
    """
    train_graphs, test_graphs = build_split(dataset, n_train, seed, n_test)
    torch.manual_seed(seed)
    model = SheafTaskModel(sheaf_type, TASK, model_family="nsp")
    holo_init = measure_holonomy(model, test_graphs)
    metric = train_eval(model, train_graphs, test_graphs, TASK, epochs)
    holo_final = measure_holonomy(model, test_graphs)
    record = dict(
        dataset=dataset,
        sheaf_type=sheaf_type,
        n_train=n_train,
        seed=seed,
        metric_name="mse_log1p_std",
        metric=metric,
        init_twist=holo_init["twist_mean"],
        final_twist=holo_final["twist_mean"],
        final_gain=holo_final["gain_mean"],
    )
    if sheaf_type == "bundle":
        lobotomize(model)
        record["metric_lobotomy"] = train_eval(
            model, train_graphs, test_graphs, TASK, 0)
        record["twist_lobotomy"] = measure_holonomy(
            model, test_graphs)["twist_mean"]
    print(
        f"[{dataset[:5]}|{sheaf_type:6s}|N={n_train:4d}|s{seed}] "
        f"mse={metric:.4f}"
        + (f" -> flat {record['metric_lobotomy']:.4f}"
           if sheaf_type == "bundle" else "")
        + f"  twist {holo_init['twist_mean']:.3f}->{holo_final['twist_mean']:.3f}",
        flush=True,
    )
    return record


def main():
    """Run the full scaling grid locally (use the Modal runner instead)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument(
        "--out",
        type=str,
        default=str(Path(__file__).parent / "scaling_results.json"),
    )
    args = parser.parse_args()
    records = [
        run_one(dataset, sheaf_type, n_train, seed, epochs=args.epochs)
        for dataset in DATASETS
        for sheaf_type in SHEAF_TYPES
        for n_train in SIZES
        for seed in range(args.seeds)
    ]
    Path(args.out).write_text(json.dumps(records, indent=2))
    print(f"\nWrote {len(records)} records -> {args.out}")


if __name__ == "__main__":
    main()
