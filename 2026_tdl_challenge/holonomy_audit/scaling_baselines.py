"""Degree-regressor and constant baselines across training-set sizes.

Companion to ``scaling_experiment.py``: fits the constant predictor and the
degree-moment ridge regression of ``degree_baselines.py`` on exactly the
scaling experiment's splits (train seed ``100 + s``, FIXED test seed
``900 + s``), for ``n_train`` in {30, 100, 300, 1000, 3000}, on GraphUniverse
(high-homophily) and regular graphs. Output: ``scaling_baseline_results.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from degree_baselines import eval_setting  # noqa: E402
from scaling_experiment import DATASETS, SIZES, build_split  # noqa: E402


def main():
    """Run all baseline fits and write ``scaling_baseline_results.json``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument(
        "--out",
        type=str,
        default=str(Path(__file__).parent / "scaling_baseline_results.json"),
    )
    args = parser.parse_args()
    records = []
    for dataset in DATASETS:
        for n_train in SIZES:
            for seed in range(args.seeds):
                train_graphs, test_graphs = build_split(dataset, n_train, seed)
                res = eval_setting(train_graphs, test_graphs)
                records.append(dict(dataset=dataset, n_train=n_train,
                                    seed=seed, **res))
                print(f"[{dataset[:5]}|N={n_train:4d}|s{seed}] "
                      f"const={res['const']:.3f} "
                      f"deg_moments={res['deg_moments']:.3f}", flush=True)
    Path(args.out).write_text(json.dumps(records, indent=2))
    print(f"\nWrote {len(records)} records -> {args.out}")


if __name__ == "__main__":
    main()
