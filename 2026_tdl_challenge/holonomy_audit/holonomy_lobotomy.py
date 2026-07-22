"""Holonomy lobotomy: is the recruited twist load-bearing?

Intervention experiment. Train a bundle (O(d)) sheaf model, then *delete its
learned geometry* -- zero every sheaf-learner parameter, so the Cayley map of the
zero parameters makes every edge transport exactly the identity (a flat
connection) -- while keeping all other trained weights. Re-evaluate.

If triangle-counting performance craters, the recruited twist was load-bearing;
if it barely moves, the model recruited curvature it never cashed in (and the
task is being solved by degree statistics elsewhere in the network). The
community-detection model, which never recruited twist, is the control: its
lobotomy should be near-harmless.

Only the bundle family admits this clean intervention: zeroing the learner gives
identity transports (Cayley(0) = I). For diagonal maps, zero parameters give
*zero* maps (tanh(0) = 0), which kills propagation outright -- a different, far
blunter intervention -- so we restrict the lobotomy to bundle.
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

FAMILIES = ["nsp", "nsd"]
TASKS = ["triangle_counting", "community_detection"]


def lobotomize(model):
    """Zero every sheaf-learner parameter: transports become identity.

    Parameters
    ----------
    model : SheafTaskModel
        A trained bundle-map model (mutated in place).
    """
    prop = model.propagation_model()
    with torch.no_grad():
        for learner in prop.sheaf_learners:
            for p in learner.parameters():
                p.zero_()


def run_one(family, task, seed, epochs, n_train, n_test, regime="homophily_hi"):
    """Train one bundle model, lobotomize it, and re-evaluate.

    Parameters
    ----------
    family : str
        ``"nsp"`` or ``"nsd"``.
    task : str
        ``"triangle_counting"`` or ``"community_detection"``.
    seed : int
        Random seed (data + init), matching the sweep's convention.
    epochs : int
        Training epochs before the intervention.
    n_train, n_test : int
        Number of train / test graphs.
    regime : str, optional
        Regime key. Default ``"homophily_hi"`` (the triangle-dense regime).

    Returns
    -------
    dict
        Metric and mean twist, before and after the lobotomy.
    """
    regime_kwargs = REGIMES[regime]
    train_graphs = make_family(task, regime_kwargs, n_train, seed=100 + seed)
    test_graphs = make_family(task, regime_kwargs, n_test, seed=900 + seed)
    torch.manual_seed(seed)
    model = SheafTaskModel("bundle", task, model_family=family)

    metric = train_eval(model, train_graphs, test_graphs, task, epochs)
    holo = measure_holonomy(model, test_graphs)

    lobotomize(model)
    # epochs=0 -> no training steps; recomputes the identical train-set target
    # standardisation and evaluates, so the two metrics are directly comparable.
    metric_lob = train_eval(model, train_graphs, test_graphs, task, 0)
    holo_lob = measure_holonomy(model, test_graphs)

    record = dict(
        family=family,
        task=task,
        seed=seed,
        regime=regime,
        metric_name=(
            "accuracy" if task == "community_detection" else "mse_log1p_std"
        ),
        metric=metric,
        metric_lobotomy=metric_lob,
        twist_mean=holo["twist_mean"],
        twist_lobotomy=holo_lob["twist_mean"],
        gain_lobotomy=holo_lob["gain_mean"],
    )
    print(
        f"[{family}|{task[:12]:12s}|s{seed}] "
        f"metric {metric:.4f} -> {metric_lob:.4f}   "
        f"twist {holo['twist_mean']:.3f} -> {holo_lob['twist_mean']:.4f}",
        flush=True,
    )
    return record


def main():
    """Run the lobotomy grid and write ``lobotomy_results.json``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--n-train", type=int, default=30)
    parser.add_argument("--n-test", type=int, default=8)
    parser.add_argument(
        "--out",
        type=str,
        default=str(Path(__file__).parent / "lobotomy_results.json"),
    )
    args = parser.parse_args()

    records = [
        run_one(family, task, seed, args.epochs, args.n_train, args.n_test)
        for family in FAMILIES
        for task in TASKS
        for seed in range(args.seeds)
    ]
    Path(args.out).write_text(json.dumps(records, indent=2))
    print(f"\nWrote {len(records)} records -> {args.out}")


if __name__ == "__main__":
    main()
