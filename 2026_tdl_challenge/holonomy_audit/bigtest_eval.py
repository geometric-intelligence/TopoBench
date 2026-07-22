"""Big-test verification: re-evaluate the key scaling models on 200 graphs.

The scaling curves of ``scaling_experiment.py`` were evaluated on 8 fixed test
graphs. Checkpoints were not saved, but training is deterministic (fixed
seeds, CPU), so re-running a configuration reproduces the identical model;
each run verifies this by matching its original 8-graph test metric
(reproduction checksum) before evaluating on a freshly generated 200-graph
test family (seed ``5000 + s``). Recomputed on the big test set: normal
error, flattened error (bundle), the twist summary, and the constant +
degree-moment ridge baselines fitted on identical splits.

Scope (pre-registered): GraphUniverse at N in {300, 3000} and regular graphs
at N = 3000, diagonal + bundle, 3 seeds -- 18 runs.

Output: ``bigtest_results.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from degree_baselines import eval_setting  # noqa: E402
from holonomy_experiment import (  # noqa: E402
    REGIMES,
    SheafTaskModel,
    make_family,
    measure_holonomy,
    train_eval,
)
from holonomy_lobotomy import lobotomize  # noqa: E402
from holonomy_regular import make_regular_family  # noqa: E402
from scaling_experiment import TASK, build_split  # noqa: E402

CONFIGS = (
    [("graphuniverse", m, n) for m in ("diag", "bundle") for n in (300, 3000)]
    + [("regular", m, 3000) for m in ("diag", "bundle")]
)


def big_test_family(dataset, seed, n_bigtest):
    """Generate the fresh big test family for one seed.

    Parameters
    ----------
    dataset : str
        ``"graphuniverse"`` or ``"regular"``.
    seed : int
        Run seed; the big test family uses ``5000 + seed``.
    n_bigtest : int
        Number of test graphs.

    Returns
    -------
    list of torch_geometric.data.Data
        The fresh test graphs.
    """
    if dataset == "graphuniverse":
        return make_family(TASK, REGIMES["homophily_hi"], n_bigtest,
                           seed=5000 + seed)
    return make_regular_family(n_bigtest, 100, 6, seed=5000 + seed)


def run_one(dataset, sheaf_type, n_train, seed, n_bigtest=200, epochs=60):
    """Reproduce one scaling model and evaluate it on the big test set.

    Parameters
    ----------
    dataset : str
        ``"graphuniverse"`` or ``"regular"``.
    sheaf_type : str
        ``"diag"`` or ``"bundle"``.
    n_train : int
        Training-set size of the reproduced configuration.
    seed : int
        Seed (identical to the scaling run being reproduced).
    n_bigtest : int, optional
        Fresh test-set size. Default 200.
    epochs : int, optional
        Training epochs (must match the scaling protocol). Default 60.

    Returns
    -------
    dict
        Reproduction checksum (8-graph metric), big-test metrics, twist
        summary, flattened big-test metric (bundle) and baselines.
    """
    train_graphs, test8 = build_split(dataset, n_train, seed)
    torch.manual_seed(seed)
    model = SheafTaskModel(sheaf_type, TASK, model_family="nsp")
    metric8 = train_eval(model, train_graphs, test8, TASK, epochs)

    big = big_test_family(dataset, seed, n_bigtest)
    metric_big = train_eval(model, train_graphs, big, TASK, 0)
    holo_big = measure_holonomy(model, big)
    baselines = eval_setting(train_graphs, big)

    record = dict(
        dataset=dataset,
        sheaf_type=sheaf_type,
        n_train=n_train,
        seed=seed,
        n_bigtest=n_bigtest,
        metric_name="mse_log1p_std",
        metric8_reproduced=metric8,
        metric_big=metric_big,
        twist_big=holo_big["twist_mean"],
        gain_big=holo_big["gain_mean"],
        const_big=baselines["const"],
        ridge_deg_moments_big=baselines["deg_moments"],
    )
    if sheaf_type == "bundle":
        lobotomize(model)
        record["metric_big_lobotomy"] = train_eval(
            model, train_graphs, big, TASK, 0)
    print(
        f"[{dataset[:5]}|{sheaf_type:6s}|N={n_train:4d}|s{seed}] "
        f"repro8={metric8:.4f}  big={metric_big:.4f}"
        + (f" -> flat {record['metric_big_lobotomy']:.4f}"
           if sheaf_type == "bundle" else "")
        + f"  const={baselines['const']:.3f} twist={holo_big['twist_mean']:.3f}",
        flush=True,
    )
    return record


def main():
    """Run all 18 configurations locally (use the Modal runner instead)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--n-bigtest", type=int, default=200)
    parser.add_argument(
        "--out",
        type=str,
        default=str(Path(__file__).parent / "bigtest_results.json"),
    )
    args = parser.parse_args()
    records = [
        run_one(dataset, sheaf_type, n_train, seed,
                n_bigtest=args.n_bigtest)
        for dataset, sheaf_type, n_train in CONFIGS
        for seed in range(args.seeds)
    ]
    Path(args.out).write_text(json.dumps(records, indent=2))
    print(f"\nWrote {len(records)} records -> {args.out}")


if __name__ == "__main__":
    main()
