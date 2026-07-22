"""Direct measurement of the degree shortcut: simple regressors vs sheaf models.

Fits ridge regressions from purely structural summary statistics -- node count,
edge count, degree mean/variance/max, and degree moments/histogram -- to the
standardised log1p triangle count, using exactly the sweep's splits (30 train /
8 test graphs per seed, same seed convention). If such a regressor matches or
beats the neural models, the degree shortcut is measured rather than inferred.
Feature sets are nested so the marginal value of each block is visible.

Runs on GraphUniverse (both regimes) and on the fixed-(n, r) regular graphs
(where all degree features are constant by construction, so every regressor
must collapse to the constant predictor -- a built-in sanity check).

Output: ``degree_baseline_results.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).parent))
from holonomy_experiment import REGIMES, make_family  # noqa: E402
from holonomy_regular import make_regular_family  # noqa: E402

FEATURE_SETS = {
    "n_nodes": ["n"],
    "n+edges": ["n", "m"],
    "deg_stats": ["n", "m", "deg_mean", "deg_var", "deg_max"],
    "deg_moments": ["n", "m", "deg_mean", "deg_var", "deg_max",
                    "deg_m3", "deg_m4"],
    "deg_hist": ["n", "m", "deg_mean", "deg_var", "deg_max",
                 "h0", "h1", "h2", "h3", "h4", "h5"],
}
HIST_EDGES = [0, 2, 4, 6, 8, 12, np.inf]  # degree bins for the histogram block


def graph_features(g):
    """Structural summary features of one graph.

    Parameters
    ----------
    g : torch_geometric.data.Data
        A graph with ``edge_index`` (both orientations) and ``num_nodes``.

    Returns
    -------
    dict
        Named scalar features (node/edge counts, degree statistics, moments,
        and a fixed-bin degree histogram, density-normalised).
    """
    n = g.num_nodes
    deg = torch.bincount(g.edge_index[0], minlength=n).float().numpy()
    m = deg.sum() / 2.0
    hist, _ = np.histogram(deg, bins=HIST_EDGES)
    feats = {
        "n": float(n),
        "m": float(m),
        "deg_mean": float(deg.mean()),
        "deg_var": float(deg.var(ddof=1)) if n > 1 else 0.0,
        "deg_max": float(deg.max()),
        "deg_m3": float(((deg - deg.mean()) ** 3).mean()),
        "deg_m4": float(((deg - deg.mean()) ** 4).mean()),
    }
    feats.update({f"h{i}": float(h) / n for i, h in enumerate(hist)})
    return feats


def eval_setting(train_graphs, test_graphs):
    """Fit every feature set on one split and return test MSEs.

    Targets are standardised exactly as in ``train_eval`` (log1p, train-split
    z-scoring), so MSEs are directly comparable with the neural models'.

    Parameters
    ----------
    train_graphs, test_graphs : list of torch_geometric.data.Data
        The split.

    Returns
    -------
    dict
        ``{"const": mse, "<feature_set>": mse, ...}`` for this split.
    """
    ytr = torch.log1p(torch.stack([g.y for g in train_graphs]).float())
    mean, sd = ytr.mean(), ytr.std().clamp_min(1e-6)
    ztr = ((ytr - mean) / sd).numpy()
    zte = ((torch.log1p(torch.stack([g.y for g in test_graphs]).float())
            - mean) / sd).numpy()
    ftr = [graph_features(g) for g in train_graphs]
    fte = [graph_features(g) for g in test_graphs]
    out = {"const": float((zte ** 2).mean())}
    for name, cols in FEATURE_SETS.items():
        xtr = np.array([[f[c] for c in cols] for f in ftr])
        xte = np.array([[f[c] for c in cols] for f in fte])
        mu, sig = xtr.mean(0), xtr.std(0) + 1e-9
        model = Ridge(alpha=1.0).fit((xtr - mu) / sig, ztr)
        pred = model.predict((xte - mu) / sig)
        out[name] = float(((pred - zte) ** 2).mean())
    return out


def main():
    """Run all settings x seeds and write ``degree_baseline_results.json``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--n-train", type=int, default=30)
    parser.add_argument("--n-test", type=int, default=8)
    parser.add_argument(
        "--out",
        type=str,
        default=str(Path(__file__).parent / "degree_baseline_results.json"),
    )
    args = parser.parse_args()

    records = []
    for regime, kw in REGIMES.items():
        for seed in range(args.seeds):
            res = eval_setting(
                make_family("triangle_counting", kw, args.n_train,
                            seed=100 + seed),
                make_family("triangle_counting", kw, args.n_test,
                            seed=900 + seed),
            )
            records.append(dict(setting=f"graphuniverse_{regime}",
                                seed=seed, **res))
            print(f"[GU {regime} s{seed}] " + "  ".join(
                f"{k}={v:.3f}" for k, v in res.items()), flush=True)
    for seed in range(args.seeds):
        res = eval_setting(
            make_regular_family(args.n_train, 100, 6, seed=100 + seed),
            make_regular_family(args.n_test, 100, 6, seed=900 + seed),
        )
        records.append(dict(setting="regular_n100_r6", seed=seed, **res))
        print(f"[regular s{seed}] " + "  ".join(
            f"{k}={v:.3f}" for k, v in res.items()), flush=True)

    Path(args.out).write_text(json.dumps(records, indent=2))
    print(f"\nWrote {len(records)} records -> {args.out}")


if __name__ == "__main__":
    main()
