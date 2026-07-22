"""Holonomy diagnostics for learned sheaf restriction maps.

Measures the *twist* a stalk vector accumulates when it is transported around a
closed cycle (e.g. a triangle) through a sheaf's per-edge transport maps -- the
graph analogue of parallel transport on a curved surface. A non-trivial holonomy
means the learned sheaf can "feel" the cycle, a capability plain message-passing
GNNs provably lack (they are bounded by 1-WL and cannot count triangles).

**Gauge invariance (why loops, not single edges).**
A single edge map depends on the *arbitrary* choice of stalk basis at each node
(a gauge). Only quantities computed around a **closed** loop are meaningful: the
basis changes cancel when the walk returns to its start node, so the loop
holonomy transforms by conjugation at the base node, ``H -> G H G^T``. Every
readout exposed here is invariant under such orthogonal conjugation:

- :func:`holonomy_magnitude` (Frobenius twist ``||H - I||_F``),
- :func:`rotation_angle` (from the trace / eigenvalues),
- :func:`frustration_sign` (the Z2 sign product for diagonal maps).

The gauge invariance is asserted by unit test (``test_gauge_invariance``).
"""

from collections import defaultdict

import torch


def enumerate_triangles(edge_index):
    """
    List every triangle (3-clique) of an undirected graph.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge indices of shape [2, num_edges]. Direction is ignored; self-loops
        are dropped.

    Returns
    -------
    list of tuple of int
        Sorted ``(u, v, w)`` node triples with ``u < v < w``, one per triangle.
    """
    nbr = defaultdict(set)
    for u, v in edge_index.detach().cpu().t().tolist():
        if u != v:
            nbr[u].add(v)
            nbr[v].add(u)
    triangles = []
    for u in sorted(nbr):
        for v in nbr[u]:
            if v <= u:
                continue
            triangles.extend((u, v, w) for w in nbr[u] & nbr[v] if w > v)
    return triangles


def _transport(transports, dst, src):
    """
    Fetch the transport map carrying a stalk vector from ``src`` to ``dst``.

    Falls back to the (pseudo)inverse of the reverse edge when only one
    orientation is stored.

    Parameters
    ----------
    transports : dict
        Maps a directed edge ``(dst, src)`` to a [d, d] transport matrix that
        carries a stalk vector from ``src`` to ``dst``.
    dst : int
        Destination node.
    src : int
        Source node.

    Returns
    -------
    torch.Tensor
        The [d, d] transport matrix for ``src -> dst``.

    Raises
    ------
    KeyError
        If neither the edge nor its reverse is present.
    """
    if (dst, src) in transports:
        return transports[(dst, src)]
    if (src, dst) in transports:
        return torch.linalg.pinv(transports[(src, dst)])
    raise KeyError(f"no transport map for edge ({src} -> {dst})")


def cycle_holonomy(transports, cycle):
    """
    Compose transport maps around a closed cycle into the holonomy matrix.

    Walks ``cycle[0] -> cycle[1] -> ... -> cycle[-1] -> cycle[0]`` and left-
    multiplies the per-edge transports, so the returned ``H`` carries a stalk
    vector at ``cycle[0]`` back to ``cycle[0]``. ``H ~= I`` means trivial
    holonomy (the sheaf is "flat" around this loop).

    Parameters
    ----------
    transports : dict
        Maps a directed edge ``(dst, src)`` to its [d, d] transport matrix
        (carrying ``src -> dst``).
    cycle : sequence of int
        Ordered node ids describing the closed loop (length >= 2).

    Returns
    -------
    torch.Tensor
        The [d, d] holonomy matrix ``H`` accumulated around the loop.
    """
    k = len(cycle)
    holonomy = None
    for i in range(k):
        cur, nxt = cycle[i], cycle[(i + 1) % k]
        step = _transport(transports, nxt, cur)  # carry cur -> nxt
        holonomy = step if holonomy is None else step @ holonomy
    return holonomy


def holonomy_magnitude(holonomy):
    """
    Gauge-invariant twist size ``||H - I||_F`` (0 == trivial holonomy).

    Parameters
    ----------
    holonomy : torch.Tensor
        A [d, d] holonomy matrix.

    Returns
    -------
    float
        The Frobenius distance of ``H`` from the identity.
    """
    d = holonomy.shape[-1]
    eye = torch.eye(d, dtype=holonomy.dtype, device=holonomy.device)
    return torch.linalg.norm(holonomy - eye).item()


def rotation_angle(holonomy):
    """
    Principal rotation angle (radians) of an (approximately) orthogonal ``H``.

    Uses the conjugation-invariant trace identity ``tr(H) = (d - 2) + 2 cos(theta)``
    (exact for a single rotation plane, and for the ``d = 2`` bundle maps). The
    returned angle is in ``[0, pi]`` and gauge-invariant; it is 0 exactly when
    ``H`` is the identity.

    Parameters
    ----------
    holonomy : torch.Tensor
        A [d, d] (approximately orthogonal) holonomy matrix.

    Returns
    -------
    float
        The principal rotation angle in radians.
    """
    d = holonomy.shape[-1]
    trace = torch.diagonal(holonomy).sum()
    cos_theta = torch.clamp((trace - (d - 2)) / 2.0, -1.0, 1.0)
    return torch.arccos(cos_theta).item()


def frustration_sign(holonomy):
    """
    Per-channel Z2 holonomy of a diagonal ``H`` (the "friend/foe" flip).

    For a diagonal holonomy each channel's diagonal entry is the product of the
    edge signs around the loop; its sign is ``+1`` when the loop is balanced and
    ``-1`` when it is "frustrated" (an odd number of sign-flipping edges). This
    is the gauge-invariant Z2 holonomy.

    Parameters
    ----------
    holonomy : torch.Tensor
        A [d, d] diagonal holonomy matrix.

    Returns
    -------
    torch.Tensor
        A length-``d`` tensor of per-channel signs in ``{-1, +1}``.
    """
    return torch.sign(torch.diagonal(holonomy))


def loop_gain(holonomy):
    """
    Volume stretch around the loop, ``|det H|`` (1 == area-preserving).

    The determinant magnitude measures how much the loop amplifies (>1) or
    shrinks (<1) signal. For an orthogonal (bundle) holonomy it is exactly 1; for
    diagonal / general maps it is the gauge-invariant "gain" that raw
    :func:`holonomy_magnitude` conflates with twist.

    Parameters
    ----------
    holonomy : torch.Tensor
        A [d, d] holonomy matrix.

    Returns
    -------
    float
        ``|det H|``, the absolute determinant of the loop.
    """
    return torch.linalg.det(holonomy).abs().item()


def orientation_flip(holonomy):
    """
    Z2 orientation of the loop, ``sign(det H)`` (-1 == the loop mirrors).

    This is the general-map version of :func:`frustration_sign`: the loop either
    preserves orientation (+1) or reverses it (-1, a reflection). For a diagonal
    ``H`` it equals the product of the per-channel frustration signs.

    Parameters
    ----------
    holonomy : torch.Tensor
        A [d, d] holonomy matrix.

    Returns
    -------
    float
        ``sign(det H)`` in ``{-1, +1}``.
    """
    return torch.sign(torch.linalg.det(holonomy)).item()


def polar_twist(holonomy):
    """
    Deconfounded rotation angle from the polar (SVD) orthogonal part of ``H``.

    Writes ``H = U S Vᵀ`` (SVD) and discards the singular values ``S`` (the
    stretch), keeping the pure rotation ``Q = U Vᵀ``; the angle is then read from
    ``tr(Q)`` exactly as in :func:`rotation_angle`. Unlike ``rotation_angle`` on
    the raw ``H``, this strips any stretch/shear first, so a pure stretch reports
    zero twist. When ``det H < 0`` the ``Q`` is a reflection (see
    :func:`orientation_flip`) and the angle should be read together with the flip.

    Parameters
    ----------
    holonomy : torch.Tensor
        A [d, d] holonomy matrix.

    Returns
    -------
    float
        The polar-rotation angle in radians, in ``[0, pi]``.
    """
    d = holonomy.shape[-1]
    u, _, vh = torch.linalg.svd(holonomy)
    q = u @ vh
    trace = torch.diagonal(q).sum()
    cos_theta = torch.clamp((trace - (d - 2)) / 2.0, -1.0, 1.0)
    return torch.arccos(cos_theta).item()


def triangle_holonomies(transports, triangles):
    """
    Batch the holonomy readouts over a list of triangles.

    Parameters
    ----------
    transports : dict
        Maps a directed edge ``(dst, src)`` to its [d, d] transport matrix.
    triangles : list of tuple of int
        Node triples, e.g. the output of :func:`enumerate_triangles`.

    Returns
    -------
    dict
        One Tensor[T] per readout, all keyed by name: ``'magnitude'``,
        ``'angle'`` (raw trace angle), ``'frustrated'`` (``{-1, +1}`` product of
        :func:`frustration_sign`), ``'gain'`` (``|det H|``, :func:`loop_gain`),
        ``'flip'`` (``sign det H``, :func:`orientation_flip`) and ``'twist'``
        (deconfounded polar angle, :func:`polar_twist`). Empty tensors when
        ``triangles`` is empty.
    """
    keys = ("magnitude", "angle", "frustrated", "gain", "flip", "twist")
    if not triangles:
        empty = torch.empty(0)
        return {key: empty for key in keys}

    # Gather the three directed steps of every triangle, then compose them in a
    # single batched matmul (H = step_ca @ step_bc @ step_ab) instead of one
    # Python loop iteration per triangle -- ~orders faster on many triangles.
    steps = [
        torch.stack(
            [
                _transport(transports, b, a),
                _transport(transports, c, b),
                _transport(transports, a, c),
            ]
        )
        for a, b, c in triangles
    ]
    step = torch.stack(steps)  # [T, 3, d, d]
    holonomy = step[:, 2] @ step[:, 1] @ step[:, 0]  # [T, d, d]

    d = holonomy.shape[-1]
    eye = torch.eye(d, dtype=holonomy.dtype, device=holonomy.device)
    diag = torch.diagonal(holonomy, dim1=-2, dim2=-1)  # [T, d]
    trace = diag.sum(-1)
    cos_theta = torch.clamp((trace - (d - 2)) / 2.0, -1.0, 1.0)

    det = torch.linalg.det(holonomy)  # [T]
    # Polar rotation part Q = U Vᵀ (SVD with the stretch S discarded).
    u, _, vh = torch.linalg.svd(holonomy)
    q_trace = torch.diagonal(u @ vh, dim1=-2, dim2=-1).sum(-1)
    twist = torch.arccos(torch.clamp((q_trace - (d - 2)) / 2.0, -1.0, 1.0))
    return {
        "magnitude": torch.linalg.norm(holonomy - eye, dim=(-2, -1)),
        "angle": torch.arccos(cos_theta),
        "frustrated": torch.sign(diag).prod(-1),
        "gain": det.abs(),
        "flip": torch.sign(det),
        "twist": twist,
    }
