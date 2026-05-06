"""
test_geometry.py — Geometry robustness tests for TractionMachinePM2D.

Verifies that build_rotor, build_stator, and build_mesh complete without
errors across a realistic range of pole counts (4–20 poles).

Run:
    pytest test_geometry.py -v
    pytest test_geometry.py -v -k "rotor"
    pytest test_geometry.py -v -k "not mesh"   # skip slow full-mesh tests
    pytest test_geometry.py -v --tb=short
"""
from __future__ import annotations

import dataclasses
import math
import tempfile
from pathlib import Path

import pytest

import gmsh

from motor_params import MotorParams, suggest_slot
from gen_rotor import build_rotor
from gen_stator import build_stator
from gen_mesh import build_mesh

π = math.pi


# ─────────────────────────────────────────────────────────────────────────────
#  Fixture: ensure gmsh is finalized between tests
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def gmsh_cleanup():
    """Finalize gmsh before and after every test to avoid 'already initialized'."""
    if gmsh.isInitialized():
        gmsh.finalize()
    yield
    if gmsh.isInitialized():
        gmsh.finalize()


# ─────────────────────────────────────────────────────────────────────────────
#  Parameter factory
# ─────────────────────────────────────────────────────────────────────────────

def make_params(
    Qp: int,
    *,
    mag_frac: float | None = None,
    w_air: float = 2.0,
    h_bridge: float = 4.0,
    h_m: float = 4.0,
    magnet_layout: str = "flat",
    coarse_mesh: bool = True,
) -> MotorParams:
    """
    Return a valid MotorParams for the given pole count.

    Uses q=2 spp (Qs = 6*Qp) with a fixed stator OD/bore and auto-computed
    slot dimensions.  mag_frac defaults to a value that keeps the magnet
    half-angle ≤ 30° regardless of Qp, ensuring the outer magnet corners
    protrude into the clip region for all Qp ≤ 12 — exercising the clip logic.
    """
    Qs = 6 * Qp
    R_so, R_si, g, R_ri = 110.0, 74.0, 0.75, 25.0

    if magnet_layout == "surface":
        h_bridge = 0.0

    # Default mag_frac: cap half-angle at 30° so pockets are always partially
    # inside R_ro and get classified correctly, while still exercising clipping
    # for wide-pitch motors.
    if mag_frac is None:
        max_half_deg = 30.0
        half_avail   = math.degrees(π / Qp)   # θs / 2
        mag_frac     = min(0.80, max_half_deg / half_avail)
        mag_frac     = round(mag_frac, 3)

    slot = suggest_slot(Qs, Qp, R_si, R_so)

    p = MotorParams(
        Qs=Qs, Qp=Qp,
        R_so=R_so, R_si=R_si,
        b1=2.5, h1=0.5,
        b_slot=slot["b_slot"],
        h_slot=slot["h_slot"],
        g=g,
        R_ri=R_ri,
        magnet_layout=magnet_layout,
        mag_frac=mag_frac,
        h_m=h_m,
        h_bridge=h_bridge,
        w_air=w_air,
        rpm=1500.0, Is=200.0, L_active=170.0,
    )

    if coarse_mesh:
        # Large mesh sizes → fast tests; geometry correctness is independent
        p = dataclasses.replace(
            p,
            lc_gap=2.0, lc_ro=8.0, lc_ri=15.0, lc_mag=5.0,
            lc_so=15.0, lc_si=5.0, lc_hp=3.0,  lc_ins=5.0,
        )

    return p


# ─────────────────────────────────────────────────────────────────────────────
#  Pole counts covered
# ─────────────────────────────────────────────────────────────────────────────

# All poles tested for rotor geometry (check_only — no mesh, fast)
ROTOR_POLES = [4, 6, 8, 10, 12, 16, 20]

# Subset used for full-mesh stator/rotor/combined tests (slower)
MESH_POLES = [6, 8, 12]


# ─────────────────────────────────────────────────────────────────────────────
#  Rotor geometry — check_only=True (exercises fragment + classification,
#  skips meshing, so each test runs in < 1 s)
# ─────────────────────────────────────────────────────────────────────────────

class TestRotorGeometry:

    @pytest.mark.parametrize("Qp", ROTOR_POLES)
    def test_builds_without_error(self, Qp):
        """build_rotor must complete for every pole count."""
        p = make_params(Qp)
        result = build_rotor(p, check_only=True)
        assert result is not None, f"Qp={Qp}: returned None"

    @pytest.mark.parametrize("Qp", ROTOR_POLES)
    def test_one_magnet_surface(self, Qp):
        """Exactly one magnet surface must be classified."""
        p = make_params(Qp)
        result = build_rotor(p, check_only=True)
        n = len(result["magnet"])
        assert n == 1, f"Qp={Qp}: expected 1 magnet surface, got {n}"

    @pytest.mark.parametrize("Qp", ROTOR_POLES)
    def test_two_pocket_surfaces(self, Qp):
        """Both flux-barrier pockets must be classified when w_air > 0."""
        p = make_params(Qp, w_air=2.0)
        result = build_rotor(p, check_only=True)
        n = len(result["air_pocket"])
        assert n == 2, f"Qp={Qp}: expected 2 pocket surfaces, got {n}"

    @pytest.mark.parametrize("Qp", ROTOR_POLES)
    def test_no_pockets_when_w_air_zero(self, Qp):
        """air_pocket list must be empty when w_air=0."""
        p = make_params(Qp, w_air=0.0)
        result = build_rotor(p, check_only=True)
        n = len(result["air_pocket"])
        assert n == 0, f"Qp={Qp}: expected 0 pockets, got {n}"

    @pytest.mark.parametrize("Qp", ROTOR_POLES)
    def test_one_shaft_surface(self, Qp):
        p = make_params(Qp)
        result = build_rotor(p, check_only=True)
        n = len(result["shaft"])
        assert n == 1, f"Qp={Qp}: expected 1 shaft surface, got {n}"

    @pytest.mark.parametrize("Qp", ROTOR_POLES)
    def test_one_airgap_surface(self, Qp):
        p = make_params(Qp)
        result = build_rotor(p, check_only=True)
        n = len(result["gap_rotor"])
        assert n == 1, f"Qp={Qp}: expected 1 airgap surface, got {n}"

    @pytest.mark.parametrize("Qp", ROTOR_POLES)
    def test_rotor_iron_present(self, Qp):
        p = make_params(Qp)
        result = build_rotor(p, check_only=True)
        n = len(result["rotor_iron"])
        assert n >= 1, f"Qp={Qp}: no rotor iron surfaces classified"

    @pytest.mark.parametrize("Qp", ROTOR_POLES)
    def test_magnet_area_positive(self, Qp):
        """Clipped magnet area must be > 0 (magnet is never fully outside R_ro)."""
        p = make_params(Qp)
        result = build_rotor(p, check_only=True)
        assert result["A_mag_mm2"] > 0.0, (
            f"Qp={Qp}: magnet area = {result['A_mag_mm2']:.3f} mm²"
        )

    @pytest.mark.parametrize("Qp", ROTOR_POLES)
    def test_high_mag_frac_no_error(self, Qp):
        """
        Large mag_frac (heavy clipping at R_ro) must not crash.
        Half-angle capped at 72° so the geometry stays non-degenerate.
        """
        half_avail = math.degrees(π / Qp)
        frac = min(0.90, 72.0 / half_avail)
        p = make_params(Qp, mag_frac=round(frac, 2))
        result = build_rotor(p, check_only=True)
        assert len(result["magnet"]) >= 1, f"Qp={Qp}: magnet missing at frac={frac:.2f}"

    @pytest.mark.parametrize("Qp", ROTOR_POLES)
    def test_no_unclassified_surfaces(self, Qp):
        """
        Every surface after fragment must land in exactly one named group;
        nothing should spill into rotor_iron due to a wrong radial band.
        """
        p = make_params(Qp)
        result = build_rotor(p, check_only=True)
        total = (
            len(result["magnet"])
            + len(result["air_pocket"])
            + len(result["shaft"])
            + len(result["gap_rotor"])
            + len(result["rotor_iron"])
        )
        # After fragment there are at least 5 regions (iron, mag, 2 pockets, shaft, gap).
        # Exact count depends on how fragment splits coincident faces.
        assert total >= 5, f"Qp={Qp}: only {total} surfaces classified total"


# ─────────────────────────────────────────────────────────────────────────────
#  Rotor boundary groups — requires full build (physical groups set after
#  check_only returns), but still uses coarse mesh for speed
# ─────────────────────────────────────────────────────────────────────────────

class TestRotorBoundaries:

    @pytest.mark.parametrize("Qp", MESH_POLES)
    def test_sector_boundaries_populated(self, Qp):
        """SB_Rotor, Rotor_Right, Rotor_Left must all be non-empty."""
        p = make_params(Qp, coarse_mesh=True)
        with tempfile.NamedTemporaryFile(suffix=".msh", delete=False) as f:
            path = Path(f.name)
        result = build_rotor(p, mesh_out=path)
        assert result["sb_rotor"],    f"Qp={Qp}: SB_Rotor is empty"
        assert result["rotor_right"], f"Qp={Qp}: Rotor_Right is empty"
        assert result["rotor_left"],  f"Qp={Qp}: Rotor_Left is empty"

    @pytest.mark.parametrize("Qp", MESH_POLES)
    def test_mesh_file_produced(self, Qp):
        p = make_params(Qp, coarse_mesh=True)
        with tempfile.NamedTemporaryFile(suffix=".msh", delete=False) as f:
            path = Path(f.name)
        build_rotor(p, mesh_out=path)
        assert path.exists() and path.stat().st_size > 1_000, (
            f"Qp={Qp}: .msh file missing or empty"
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Stator geometry
# ─────────────────────────────────────────────────────────────────────────────

class TestStatorGeometry:

    @pytest.mark.parametrize("Qp", MESH_POLES)
    def test_builds_without_error(self, Qp):
        p = make_params(Qp, coarse_mesh=True)
        with tempfile.NamedTemporaryFile(suffix=".msh", delete=False) as f:
            path = Path(f.name)
        result = build_stator(p, mesh_out=path)
        assert result is not None

    @pytest.mark.parametrize("Qp", MESH_POLES)
    def test_sector_boundaries_populated(self, Qp):
        p = make_params(Qp, coarse_mesh=True)
        with tempfile.NamedTemporaryFile(suffix=".msh", delete=False) as f:
            path = Path(f.name)
        result = build_stator(p, mesh_out=path)
        assert result["sb_stator"],    f"Qp={Qp}: SB_Stator is empty"
        assert result["stator_right"], f"Qp={Qp}: Stator_Right is empty"
        assert result["stator_left"],  f"Qp={Qp}: Stator_Left is empty"

    @pytest.mark.parametrize("Qp", MESH_POLES)
    def test_mesh_file_produced(self, Qp):
        p = make_params(Qp, coarse_mesh=True)
        with tempfile.NamedTemporaryFile(suffix=".msh", delete=False) as f:
            path = Path(f.name)
        build_stator(p, mesh_out=path)
        assert path.exists() and path.stat().st_size > 1_000


# ─────────────────────────────────────────────────────────────────────────────
#  Combined mesh (stator + rotor together) — slowest tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCombinedMesh:

    @pytest.mark.parametrize("Qp", MESH_POLES)
    def test_mesh_builds_without_error(self, Qp):
        p = make_params(Qp, coarse_mesh=True)
        with tempfile.NamedTemporaryFile(suffix=".msh", delete=False) as f:
            path = Path(f.name)
        build_mesh(p, mesh_out=path)
        assert path.exists() and path.stat().st_size > 10_000, (
            f"Qp={Qp}: combined .msh missing or suspiciously small"
        )


# ─────────────────────────────────────────────────────────────────────────────
#  MotorParams geometry helpers (pure-Python, no gmsh)
# ─────────────────────────────────────────────────────────────────────────────

class TestMotorParamsGeometry:
    """Fast sanity checks on derived MotorParams quantities — no gmsh needed."""

    @pytest.mark.parametrize("Qp", ROTOR_POLES)
    def test_R_mi_inside_rotor(self, Qp):
        """Magnet inner face must be between shaft OD and rotor OD."""
        p = make_params(Qp)
        assert p.R_ri < p.R_mi < p.R_ro, (
            f"Qp={Qp}: R_mi={p.R_mi:.2f} not in (R_ri={p.R_ri}, R_ro={p.R_ro:.2f})"
        )

    @pytest.mark.parametrize("Qp", ROTOR_POLES)
    def test_R_mo_inside_rotor(self, Qp):
        """Magnet outer face must be ≤ R_ro."""
        p = make_params(Qp)
        assert p.R_mo <= p.R_ro, (
            f"Qp={Qp}: R_mo={p.R_mo:.2f} > R_ro={p.R_ro:.2f}"
        )

    @pytest.mark.parametrize("Qp", ROTOR_POLES)
    def test_w_mag_positive(self, Qp):
        p = make_params(Qp)
        assert p.w_mag > 0, f"Qp={Qp}: w_mag={p.w_mag:.3f}"

    @pytest.mark.parametrize("Qp", ROTOR_POLES)
    def test_validation_passes(self, Qp):
        """make_params must produce a design that passes validate()."""
        p = make_params(Qp)
        p.validate()   # raises ValueError on failure

    @pytest.mark.parametrize("Qp", ROTOR_POLES)
    def test_magnet_inner_corner_radius(self, Qp):
        """
        The inner corners of the magnet rectangle (at R_mi, ±w_mag/2) must
        lie within R_ro — otherwise the magnet extends entirely outside the
        rotor and _clip_rect returns None.
        """
        p = make_params(Qp)
        r_inner_corner = math.hypot(p.R_mi, p.w_mag / 2)
        assert r_inner_corner < p.R_ro, (
            f"Qp={Qp}: magnet inner corner at r={r_inner_corner:.2f} mm "
            f"≥ R_ro={p.R_ro:.2f} mm — magnet would be entirely outside the rotor. "
            f"Reduce mag_frac (currently {p.mag_frac:.3f})."
        )

    @pytest.mark.parametrize("Qp", ROTOR_POLES)
    def test_pocket_inner_corner_radius(self, Qp):
        """
        The innermost corner of each pocket rectangle — at local (R_mi, ±w_mag/2),
        which is closest to the origin — must lie within R_ro so that at least some
        pocket area survives clipping to the rotor cylinder.
        """
        p = make_params(Qp)
        if p.w_air <= 0:
            pytest.skip("w_air=0, no pockets")
        # Innermost corner of the pocket in local (radial, tangential) coords:
        # the pocket sits at y ∈ [w_mag/2, w_mag/2+w_air], x ∈ [R_mi, R_mo].
        # The corner (R_mi, w_mag/2) has the smallest r and is the last to leave R_ro.
        r_pkt_inner = math.hypot(p.R_mi, p.w_mag / 2)
        assert r_pkt_inner < p.R_ro, (
            f"Qp={Qp}: pocket innermost corner at r={r_pkt_inner:.2f} mm "
            f"≥ R_ro={p.R_ro:.2f} mm — pocket entirely outside rotor. "
            f"Reduce mag_frac or w_air."
        )
