"""
Tests for gen_rotor.py.

check_only=True skips meshing → runs in ~0.3 s.
The @pytest.mark.slow tests do a full mesh and are skipped
unless you run:  pytest -m slow
"""
import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from motor_params import MotorParams
from gen_rotor import build_rotor


@pytest.fixture(scope="module")
def result_check():
    """Classification result only (check_only=True, no mesh)."""
    p = MotorParams()
    return p, build_rotor(p, check_only=True)


@pytest.fixture(scope="module")
def result_mesh(tmp_path_factory):
    """Full mesh result — cached per module."""
    p = MotorParams()
    out = tmp_path_factory.mktemp("rotor") / "rotor.msh"
    return p, build_rotor(p, mesh_out=out)


# ─────────────────────────────────────────────────────────────────────────────
#  Classification (fast — no mesh)
# ─────────────────────────────────────────────────────────────────────────────
class TestClassification:
    def test_rotor_iron_found(self, result_check):
        _, r = result_check
        assert len(r["rotor_iron"]) >= 1

    def test_gap_rotor_found(self, result_check):
        _, r = result_check
        assert len(r["gap_rotor"]) == 1

    def test_shaft_found(self, result_check):
        _, r = result_check
        assert len(r["shaft"]) == 1

    def test_magnet_found(self, result_check):
        _, r = result_check
        assert len(r["magnet"]) == 1

    def test_pockets_found(self, result_check):
        p, r = result_check
        expected = 2 if p.w_air > 0 else 0
        assert len(r["air_pocket"]) == expected, (
            f"Expected {expected} pockets, got {len(r['air_pocket'])}"
        )

    def test_total_surface_count(self, result_check):
        """6 surfaces: 1 iron + 1 gap + 1 magnet + 2 pockets + 1 shaft."""
        p, r = result_check
        n_pockets = 2 if p.w_air > 0 else 0
        total = (
            len(r["rotor_iron"]) + len(r["gap_rotor"])
            + len(r["magnet"]) + len(r["air_pocket"]) + len(r["shaft"])
        )
        expected = 4 + n_pockets
        assert total == expected, f"Expected {expected} surfaces, got {total}"

    def test_mag_area_positive(self, result_check):
        _, r = result_check
        assert r["A_mag_mm2"] > 0

    def test_mag_area_at_least_half_rect(self, result_check):
        """Clipped magnet area should be ≥ 50% of full rectangle area."""
        p, r = result_check
        assert r["A_mag_mm2"] >= p.A_mag * 0.5, (
            f"Clipped area {r['A_mag_mm2']:.1f} mm² < 50% of "
            f"rect area {p.A_mag:.1f} mm²"
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Modified parameters
# ─────────────────────────────────────────────────────────────────────────────
class TestModifiedParams:
    def test_no_pockets_when_w_air_zero(self):
        p = MotorParams(w_air=0.0)
        r = build_rotor(p, check_only=True)
        assert len(r["air_pocket"]) == 0

    def test_invalid_params_raises(self):
        with pytest.raises(ValueError):
            build_rotor(MotorParams(g=0.1), check_only=True)

    def test_smaller_mag_frac(self):
        """Lower mag_frac → narrower magnet, still 1 magnet surface."""
        p = MotorParams(mag_frac=0.60)
        r = build_rotor(p, check_only=True)
        assert len(r["magnet"]) == 1


# ─────────────────────────────────────────────────────────────────────────────
#  Full mesh (slow)
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.slow
class TestMesh:
    def test_mesh_file_created(self, result_mesh):
        _, r = result_mesh
        assert r["n_nodes"] > 100

    def test_mesh_element_count(self, result_mesh):
        _, r = result_mesh
        assert r["n_elems"] > 200

    def test_sb_rotor_single_arc(self, result_mesh):
        _, r = result_mesh
        assert len(r["sb_rotor"]) == 1, (
            f"Expected 1 SB_Rotor arc, got {r['sb_rotor']}"
        )

    def test_sector_boundaries_exist(self, result_mesh):
        _, r = result_mesh
        assert len(r["rotor_right"]) >= 1
        assert len(r["rotor_left"]) >= 1
