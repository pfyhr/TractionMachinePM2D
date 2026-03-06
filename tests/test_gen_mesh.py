"""
Tests for gen_mesh.py.

check_only=True skips meshing → runs in ~0.4 s.
The @pytest.mark.slow tests do a full mesh and are skipped
unless you run:  pytest -m slow
"""
import math
import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from motor_params import MotorParams
from gen_mesh import build_mesh


@pytest.fixture(scope="module")
def result_check():
    """Classification result only (check_only=True, no mesh)."""
    p = MotorParams()
    return p, build_mesh(p, check_only=True)


@pytest.fixture(scope="module")
def result_mesh(tmp_path_factory):
    """Full mesh result — cached per module."""
    p = MotorParams()
    out = tmp_path_factory.mktemp("mesh") / "motor.msh"
    return p, build_mesh(p, mesh_out=out)


# ─────────────────────────────────────────────────────────────────────────────
#  Classification (fast — no mesh)
# ─────────────────────────────────────────────────────────────────────────────
class TestClassification:
    def test_stator_iron_found(self, result_check):
        _, r = result_check
        assert len(r["stator_iron"]) >= 1

    def test_gap_stator_found(self, result_check):
        _, r = result_check
        assert len(r["gap_stator"]) == 1

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
        assert len(r["air_pocket"]) == expected

    def test_all_slots_have_opening(self, result_check):
        p, r = result_check
        for i in range(p.ns):
            assert len(r["slot_opening"][i]) == 1, f"Slot {i} missing opening"

    def test_all_slots_have_insulation(self, result_check):
        p, r = result_check
        for i in range(p.ns):
            assert len(r["slot_insul"][i]) >= 1, f"Slot {i} missing insulation"

    def test_all_hairpins_found(self, result_check):
        p, r = result_check
        for i in range(p.ns):
            for k in range(p.n_hp):
                n = len(r["slot_hp"][i][k])
                assert n == 1, f"Slot {i} layer {k}: expected 1 HP, got {n}"

    def test_total_surface_count(self, result_check):
        """56 surfaces: 50 stator + 6 rotor."""
        p, r = result_check
        n_pockets = len(r["air_pocket"])
        total = (
            len(r["stator_iron"]) + len(r["gap_stator"])
            + sum(len(r["slot_opening"][i]) for i in range(p.ns))
            + sum(len(r["slot_insul"][i])   for i in range(p.ns))
            + sum(len(r["slot_hp"][i][k])
                  for i in range(p.ns) for k in range(p.n_hp))
            + len(r["gap_rotor"]) + len(r["rotor_iron"])
            + len(r["magnet"])    + n_pockets + len(r["shaft"])
        )
        expected = 1 + 1 + p.ns*(1 + 1 + p.n_hp) + 1 + 1 + 1 + n_pockets + 1
        assert total == expected, f"Expected {expected} surfaces, got {total}"

    def test_fill_factor_matches_params(self, result_check):
        p, r = result_check
        assert math.isclose(r["fill_factor"], p.fill_factor)

    def test_carea_positive(self, result_check):
        _, r = result_check
        assert r["Carea_m2"] > 0

    def test_mag_area_positive(self, result_check):
        _, r = result_check
        assert r["A_mag_mm2"] > 0

    def test_mag_area_at_least_half_rect(self, result_check):
        p, r = result_check
        assert r["A_mag_mm2"] >= p.A_mag * 0.5


# ─────────────────────────────────────────────────────────────────────────────
#  Full mesh (slow)
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.slow
class TestMesh:
    def test_mesh_node_count(self, result_mesh):
        _, r = result_mesh
        assert r["n_nodes"] > 1000

    def test_mesh_element_count(self, result_mesh):
        _, r = result_mesh
        assert r["n_elems"] > 2000

    def test_sb_stator_single_arc(self, result_mesh):
        _, r = result_mesh
        assert len(r["sb_stator"]) == 1, f"Expected 1 SB_Stator arc, got {r['sb_stator']}"

    def test_sb_rotor_single_arc(self, result_mesh):
        _, r = result_mesh
        assert len(r["sb_rotor"]) == 1, f"Expected 1 SB_Rotor arc, got {r['sb_rotor']}"

    def test_sb_non_conforming(self, result_mesh):
        """SB_Stator and SB_Rotor must be independent curves (non-conforming mesh)."""
        _, r = result_mesh
        assert not set(r["sb_stator"]) & set(r["sb_rotor"]), (
            "SB_Stator and SB_Rotor share curves — mesh is conforming!"
        )

    def test_domain_single_arc(self, result_mesh):
        _, r = result_mesh
        assert len(r["domain"]) == 1

    def test_stator_sector_boundaries_exist(self, result_mesh):
        _, r = result_mesh
        assert len(r["stator_right"]) >= 1
        assert len(r["stator_left"])  >= 1

    def test_rotor_sector_boundaries_exist(self, result_mesh):
        _, r = result_mesh
        assert len(r["rotor_right"]) >= 1
        assert len(r["rotor_left"])  >= 1
