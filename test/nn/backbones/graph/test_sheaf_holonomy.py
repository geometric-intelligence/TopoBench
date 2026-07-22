"""Unit tests for the sheaf holonomy diagnostics."""

import math

import pytest
import torch

from topobench.nn.backbones.graph.nsd_utils.sheaf_holonomy import (
    cycle_holonomy,
    enumerate_triangles,
    frustration_sign,
    holonomy_magnitude,
    loop_gain,
    orientation_flip,
    polar_twist,
    rotation_angle,
    triangle_holonomies,
)


def _rot(theta):
    """Return the 2x2 rotation matrix for angle ``theta`` (radians).

    Parameters
    ----------
    theta : float
        Rotation angle in radians.

    Returns
    -------
    torch.Tensor
        A [2, 2] rotation matrix.
    """
    c, s = math.cos(theta), math.sin(theta)
    return torch.tensor([[c, -s], [s, c]])


def _triangle_transports(a, b, c):
    """Build a directed transport dict for the triangle 0->1->2->0.

    Parameters
    ----------
    a : torch.Tensor
        Transport carrying 0 -> 1.
    b : torch.Tensor
        Transport carrying 1 -> 2.
    c : torch.Tensor
        Transport carrying 2 -> 0.

    Returns
    -------
    dict
        Directed-edge -> transport-matrix mapping for the loop.
    """
    return {(1, 0): a, (2, 1): b, (0, 2): c}


class TestEnumerateTriangles:
    """Test triangle enumeration."""

    def test_single_triangle(self):
        """A 3-cycle yields exactly one triangle."""
        ei = torch.tensor([[0, 1, 2, 1, 2, 0], [1, 2, 0, 0, 1, 2]])
        assert enumerate_triangles(ei) == [(0, 1, 2)]

    def test_triangle_plus_pendant(self):
        """A pendant node adds no triangle."""
        ei = torch.tensor([[0, 1, 2, 2], [1, 2, 0, 3]])
        assert enumerate_triangles(ei) == [(0, 1, 2)]

    def test_no_triangle(self):
        """A path has no triangles; self-loops are ignored."""
        ei = torch.tensor([[0, 1, 2, 2], [1, 2, 3, 2]])
        assert enumerate_triangles(ei) == []

    def test_two_triangles_sharing_edge(self):
        """A diamond (0-1-2, 0-2-3) has two triangles."""
        ei = torch.tensor(
            [[0, 1, 2, 0, 3], [1, 2, 0, 3, 2]]
        )
        # edges: 0-1,1-2,2-0,0-3,3-2  -> triangles (0,1,2) and (0,2,3)
        assert set(enumerate_triangles(ei)) == {(0, 1, 2), (0, 2, 3)}


class TestHolonomyReadouts:
    """Test the holonomy matrix and its scalar readouts."""

    def test_identity_is_trivial(self):
        """All-identity transports give H = I: zero twist (the ISN case)."""
        eye = torch.eye(2)
        H = cycle_holonomy(_triangle_transports(eye, eye, eye), (0, 1, 2))
        assert torch.allclose(H, eye, atol=1e-6)
        assert holonomy_magnitude(H) < 1e-6
        assert rotation_angle(H) < 1e-6
        assert torch.equal(frustration_sign(H), torch.ones(2))

    def test_rotations_compose(self):
        """Three 40 deg rotations compose to a 120 deg holonomy."""
        r = _rot(math.radians(40))
        H = cycle_holonomy(_triangle_transports(r, r, r), (0, 1, 2))
        assert math.isclose(
            rotation_angle(H), math.radians(120), abs_tol=1e-4
        )
        assert holonomy_magnitude(H) > 1.0  # clearly non-trivial

    def test_rotations_cancel_to_identity(self):
        """Three 120 deg rotations sum to 360 deg = identity: flat loop."""
        r = _rot(math.radians(120))
        H = cycle_holonomy(_triangle_transports(r, r, r), (0, 1, 2))
        assert holonomy_magnitude(H) < 1e-4  # non-trivial edges, trivial loop
        assert rotation_angle(H) < 1e-3

    def test_diagonal_frustration(self):
        """Diagonal sign maps: per-channel product gives the Z2 holonomy."""
        a = torch.diag(torch.tensor([1.0, -1.0]))
        b = torch.diag(torch.tensor([-1.0, -1.0]))
        c = torch.diag(torch.tensor([1.0, 1.0]))
        H = cycle_holonomy(_triangle_transports(a, b, c), (0, 1, 2))
        # channel 0: 1*-1*1 = -1 (frustrated); channel 1: -1*-1*1 = +1
        assert torch.equal(
            frustration_sign(H), torch.tensor([-1.0, 1.0])
        )


class TestGaugeInvariance:
    """The holonomy readouts must be invariant to per-node basis choices."""

    def test_gauge_invariance(self):
        """Random per-node orthogonal gauge leaves magnitude + angle unchanged.

        A gauge change assigns each node its own orthogonal basis ``G_i`` and
        maps ``T_{(i,j)} -> G_i T_{(i,j)} G_j^T``. Around a closed loop this
        conjugates the holonomy by the base node's ``G``, so both the Frobenius
        twist and the rotation angle are invariant. This is the property that
        makes the loop holonomy a meaningful (coordinate-free) quantity.
        """
        torch.manual_seed(0)
        # Arbitrary orthogonal transports around the triangle.
        a, b, c = (
            _rot(math.radians(35)),
            _rot(math.radians(-50)),
            _rot(math.radians(80)),
        )
        base = _triangle_transports(a, b, c)
        H = cycle_holonomy(base, (0, 1, 2))
        mag0, ang0 = holonomy_magnitude(H), rotation_angle(H)

        # Per-node random orthogonal gauge (via QR of a random matrix).
        gauge = {}
        for node in (0, 1, 2):
            q, _ = torch.linalg.qr(torch.randn(2, 2))
            gauge[node] = q
        gauged = {
            (dst, src): gauge[dst] @ T @ gauge[src].T
            for (dst, src), T in base.items()
        }
        Hg = cycle_holonomy(gauged, (0, 1, 2))

        assert math.isclose(holonomy_magnitude(Hg), mag0, abs_tol=1e-5)
        assert math.isclose(rotation_angle(Hg), ang0, abs_tol=1e-5)


class TestGainFlipTwist:
    """Test the det-based gain/flip and the polar (deconfounded) twist."""

    def test_gain_is_abs_det(self):
        """loop_gain returns |det H|: 1 for a rotation, product for a stretch."""
        assert math.isclose(loop_gain(_rot(math.radians(50))), 1.0, abs_tol=1e-5)
        stretch = torch.diag(torch.tensor([2.0, 3.0]))
        assert math.isclose(loop_gain(stretch), 6.0, abs_tol=1e-5)

    def test_flip_detects_mirror(self):
        """orientation_flip is +1 for a rotation, -1 for a reflection."""
        assert orientation_flip(_rot(math.radians(30))) == 1.0
        mirror = torch.diag(torch.tensor([-1.0, 1.0]))
        assert orientation_flip(mirror) == -1.0

    def test_polar_twist_strips_stretch(self):
        """Polar twist ignores pure stretch and recovers the hidden rotation."""
        r = _rot(math.radians(40))
        stretch = torch.diag(torch.tensor([2.0, 3.0]))
        # H = R @ D is already its own polar form (Q=R, P=D) -> twist = 40 deg.
        assert math.isclose(
            polar_twist(r @ stretch), math.radians(40), abs_tol=1e-4
        )
        # A pure stretch has no rotation -> zero twist (unlike rotation_angle).
        assert polar_twist(stretch) < 1e-4

    def test_batch_exposes_all_readouts(self):
        """triangle_holonomies returns gain/flip/twist alongside the originals."""
        eye = torch.eye(2)
        transports = {
            (1, 0): eye, (0, 1): eye,
            (2, 1): eye, (1, 2): eye,
            (0, 2): eye, (2, 0): eye,
        }
        out = triangle_holonomies(transports, [(0, 1, 2)])
        for key in ("magnitude", "angle", "frustrated", "gain", "flip", "twist"):
            assert out[key].shape == (1,)
        assert math.isclose(out["gain"][0].item(), 1.0, abs_tol=1e-5)
        assert out["twist"][0].item() < 1e-5

    def test_empty_batch_has_all_keys(self):
        """Empty input still returns every readout key as an empty tensor."""
        out = triangle_holonomies({}, [])
        for key in ("magnitude", "angle", "frustrated", "gain", "flip", "twist"):
            assert out[key].numel() == 0


class TestTransportFallbackAndBatch:
    """Test the inverse fallback and batched readouts."""

    def test_inverse_fallback(self):
        """A missing direction is recovered from the reverse edge's inverse."""
        r = _rot(math.radians(30))
        # Only forward edges present; cycle needs (0,2) which is absent.
        transports = {(1, 0): r, (2, 1): r, (2, 0): r}  # (2,0) is reverse of (0,2)
        H = cycle_holonomy(transports, (0, 1, 2))
        # carry 2->0 uses inv of (2,0)=r  -> r @ r @ inv(r) = r  (60 deg)
        assert math.isclose(
            rotation_angle(H), math.radians(30), abs_tol=1e-4
        )

    def test_triangle_holonomies_batch(self):
        """Batched readouts return one value per triangle."""
        ei = torch.tensor([[0, 1, 2, 1, 2, 0], [1, 2, 0, 0, 1, 2]])
        tris = enumerate_triangles(ei)
        r = _rot(math.radians(40))
        transports = {}
        for (u, v, w) in tris:
            transports[(v, u)] = r
            transports[(w, v)] = r
            transports[(u, w)] = r
        out = triangle_holonomies(transports, tris)
        assert out["magnitude"].shape == (len(tris),)
        assert out["angle"].shape == (len(tris),)
        assert out["frustrated"].shape == (len(tris),)
        assert math.isclose(
            out["angle"][0].item(), math.radians(120), abs_tol=1e-4
        )

    def test_empty_triangles(self):
        """No triangles yields empty result tensors."""
        out = triangle_holonomies({}, [])
        assert out["magnitude"].numel() == 0
        assert out["angle"].numel() == 0

    def test_missing_edge_raises(self):
        """A cycle edge absent in both directions raises KeyError."""
        with pytest.raises(KeyError):
            cycle_holonomy({(1, 0): torch.eye(2)}, (0, 1, 2))
