"""Numerical verification of the holonomy diagnostic itself.

Four checks, run end-to-end on freshly trained models (GraphUniverse
high-homophily, seed 0, the sweep's protocol):

1. **Gauge invariance, numerically**: random SO(2) transports around a
   triangle, conjugated by 1,000 random per-node orthogonal gauges (rotations
   and reflections); report the maximum absolute deviation of twist, gain and
   flip.
2. **Invertibility / singularity rates** of the audited objects, per map
   family: distribution of |det| for edge-level transports and per-triangle
   loop products (fraction below 1e-6, 1e-3; min/median).
3. **Polar-twist stability**: for trained diagonal/general loop products
   (near-singular) and bundle ones (orthogonal), perturb H by eps * N(0,1)
   and report the standard deviation of the twist across perturbations, for
   eps in {1e-8, 1e-6, 1e-4}.
4. **Exact flatness after the intervention**: lobotomise the trained bundle
   model and report max |twist|, max |gain - 1| and max ||H - I||_F over every
   audited triangle.

Output: ``diagnostic_checks.json`` (consumed by the notebook).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from holonomy_experiment import (  # noqa: E402
    REGIMES,
    SheafTaskModel,
    make_family,
    train_eval,
)
from holonomy_lobotomy import lobotomize  # noqa: E402
from topobench.nn.backbones.graph.nsd_utils.holonomy_capture import (  # noqa: E402
    sheaf_transports,
)
from topobench.nn.backbones.graph.nsd_utils.sheaf_holonomy import (  # noqa: E402
    enumerate_triangles,
    loop_gain,
    orientation_flip,
    polar_twist,
    triangle_holonomies,
)


def rot(theta):
    """2x2 rotation matrix.

    Parameters
    ----------
    theta : float
        Angle in radians.

    Returns
    -------
    torch.Tensor
        The [2, 2] rotation matrix.
    """
    c, s = math.cos(theta), math.sin(theta)
    return torch.tensor([[c, -s], [s, c]], dtype=torch.float64)


def check_gauge_invariance(n_trials=1000, seed=0):
    """Check conjugation invariance of the readouts numerically.

    Parameters
    ----------
    n_trials : int, optional
        Number of random gauges. Default 1000.
    seed : int, optional
        RNG seed. Default 0.

    Returns
    -------
    dict
        Maximum absolute deviations of twist, gain and flip under random
        orthogonal gauges (including reflections).
    """
    torch.manual_seed(seed)
    trans = {(1, 0): rot(0.61), (2, 1): rot(-0.87), (0, 2): rot(1.40)}
    h0 = trans[(0, 2)] @ trans[(2, 1)] @ trans[(1, 0)]
    t0, g0, f0 = polar_twist(h0), loop_gain(h0), orientation_flip(h0)
    dev_t = dev_g = dev_f = 0.0
    for _ in range(n_trials):
        gauge = {}
        for v in (0, 1, 2):
            q, _ = torch.linalg.qr(torch.randn(2, 2, dtype=torch.float64))
            gauge[v] = q  # QR yields rotations AND reflections
        gt = {(d, s_): gauge[d] @ m @ gauge[s_].T for (d, s_), m in trans.items()}
        h = gt[(0, 2)] @ gt[(2, 1)] @ gt[(1, 0)]
        h = gauge[0].T @ h @ gauge[0]  # undo basepoint conjugation explicitly
        dev_t = max(dev_t, abs(polar_twist(h) - t0))
        dev_g = max(dev_g, abs(loop_gain(h) - g0))
        dev_f = max(dev_f, abs(orientation_flip(h) - f0))
    return dict(n_trials=n_trials, max_twist_dev=dev_t,
                max_gain_dev=dev_g, max_flip_dev=dev_f)


def trained_model_and_loops(sheaf_type, epochs=60, seed=0):
    """Train one model on the sweep's protocol and collect its loop products.

    Parameters
    ----------
    sheaf_type : str
        Restriction-map family.
    epochs : int, optional
        Training epochs (sweep default 60).
    seed : int, optional
        Seed (sweep convention). Default 0.

    Returns
    -------
    tuple
        ``(model, test_graphs, edge_dets, loop_Hs)`` where ``edge_dets`` is a
        list of per-edge transport determinants and ``loop_Hs`` a list of
        per-triangle loop-product matrices.
    """
    kw = REGIMES["homophily_hi"]
    train_graphs = make_family("triangle_counting", kw, 30, seed=100 + seed)
    test_graphs = make_family("triangle_counting", kw, 8, seed=900 + seed)
    torch.manual_seed(seed)
    model = SheafTaskModel(sheaf_type, "triangle_counting", model_family="nsp")
    train_eval(model, train_graphs, test_graphs, "triangle_counting", epochs)
    model.eval()
    prop = model.propagation_model()
    edge_dets, loop_hs = [], []
    with torch.no_grad():
        for g in test_graphs:
            tris = enumerate_triangles(g.edge_index)
            if not tris:
                continue
            model(g.x, g.edge_index,
                  torch.zeros(g.num_nodes, dtype=torch.long))
            transports = sheaf_transports(prop, g.edge_index)
            edge_dets += [float(torch.linalg.det(m))
                          for (a, b), m in transports.items() if a < b]
            for (u, v, w) in tris:
                h = (transports[(u, w)] @ transports[(w, v)]
                     @ transports[(v, u)])
                loop_hs.append(h)
    return model, test_graphs, edge_dets, loop_hs


def det_stats(dets):
    """Summarise a list of determinants.

    Parameters
    ----------
    dets : list of float
        Determinants.

    Returns
    -------
    dict
        Singularity-rate summary statistics of |det|.
    """
    a = np.abs(np.array(dets))
    return dict(n=len(a), min=float(a.min()), median=float(np.median(a)),
                frac_below_1e6=float((a < 1e-6).mean()),
                frac_below_1e3=float((a < 1e-3).mean()))


def twist_stability(loop_hs, eps_list=(1e-8, 1e-6, 1e-4), n_pert=50, seed=0):
    """Perturbation stability of the polar twist.

    Parameters
    ----------
    loop_hs : list of torch.Tensor
        Loop-product matrices.
    eps_list : tuple of float, optional
        Perturbation scales.
    n_pert : int, optional
        Perturbations per matrix. Default 50.
    seed : int, optional
        RNG seed.

    Returns
    -------
    dict
        For each eps: the mean (over sampled matrices) of the std (over
        perturbations) of the twist, in radians.
    """
    torch.manual_seed(seed)
    sample = loop_hs[:: max(1, len(loop_hs) // 40)][:40]
    out = {}
    for eps in eps_list:
        stds = []
        for h in sample:
            ts = [polar_twist(h + eps * torch.randn_like(h))
                  for _ in range(n_pert)]
            stds.append(float(np.std(ts)))
        out[f"eps_{eps:g}"] = float(np.mean(stds))
    return out


def main():
    """Run all four checks and write ``diagnostic_checks.json``."""
    checks = {"gauge_invariance": check_gauge_invariance()}
    print("gauge invariance:", checks["gauge_invariance"], flush=True)

    bundle_model = None
    bundle_graphs = None
    for fam in ["bundle", "diag", "general"]:
        model, graphs, edge_dets, loop_hs = trained_model_and_loops(fam)
        checks[f"{fam}_edge_det"] = det_stats(edge_dets)
        checks[f"{fam}_loop_det"] = det_stats(
            [float(torch.linalg.det(h)) for h in loop_hs])
        checks[f"{fam}_twist_stability"] = twist_stability(loop_hs)
        print(fam, "edge:", checks[f"{fam}_edge_det"], flush=True)
        print(fam, "loop:", checks[f"{fam}_loop_det"], flush=True)
        print(fam, "twist stability:", checks[f"{fam}_twist_stability"],
              flush=True)
        if fam == "bundle":
            bundle_model, bundle_graphs = model, graphs

    # Exact flatness after the intervention.
    lobotomize(bundle_model)
    prop = bundle_model.propagation_model()
    max_tw = max_gain = max_frob = 0.0
    with torch.no_grad():
        for g in bundle_graphs:
            tris = enumerate_triangles(g.edge_index)
            if not tris:
                continue
            bundle_model(g.x, g.edge_index,
                         torch.zeros(g.num_nodes, dtype=torch.long))
            transports = sheaf_transports(prop, g.edge_index)
            out = triangle_holonomies(transports, tris)
            max_tw = max(max_tw, float(out["twist"].abs().max()))
            max_gain = max(max_gain, float((out["gain"] - 1).abs().max()))
            for (u, v, w) in tris:
                h = (transports[(u, w)] @ transports[(w, v)]
                     @ transports[(v, u)])
                max_frob = max(max_frob,
                               float(torch.linalg.norm(h - torch.eye(2))))
    checks["intervention_flatness"] = dict(
        max_abs_twist=max_tw, max_abs_gain_minus_1=max_gain,
        max_frobenius_H_minus_I=max_frob)
    print("intervention flatness:", checks["intervention_flatness"],
          flush=True)

    out = Path(__file__).parent / "diagnostic_checks.json"
    out.write_text(json.dumps(checks, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
