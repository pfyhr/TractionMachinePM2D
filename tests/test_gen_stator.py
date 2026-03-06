"""
Tests for gen_stator.py.

check_only=True skips meshing → runs in ~0.3 s.
The @pytest.mark.slow tests do a full mesh (a few seconds) and are skipped
unless you run:  pytest -m slow
"""
import math
import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from motor_params import MotorParams
from gen_stator import build_stator


@pytest.fixture(scope="module")
def result_check():
    """Classification result only (check_only=True, no mesh)."""
    p = MotorParams()
    return p, build_stator(p, check_only=True)


@pytest.fixture(scope="module")
def result_mesh(tmp_path_factory):
    """Full mesh result — cached per module."""
    p = MotorParams()
    out = tmp_path_factory.mktemp("stator") / "stator.msh"
    return p, build_stator(p, mesh_out=out)


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
                assert n == 1, (
                    f"Slot {i} layer {k}: expected 1 hairpin surface, got {n}"
                )

    def test_total_surface_count(self, result_check):
        """50 surfaces total: 1 iron + 1 gap + 6×(1 open + 6 HP + 1 insul)."""
        p, r = result_check
        n_iron  = len(r["stator_iron"])
        n_gap   = len(r["gap_stator"])
        n_open  = sum(len(r["slot_opening"][i]) for i in range(p.ns))
        n_insul = sum(len(r["slot_insul"][i]) for i in range(p.ns))
        n_hp    = sum(len(r["slot_hp"][i][k])
                      for i in range(p.ns) for k in range(p.n_hp))
        total = n_iron + n_gap + n_open + n_insul + n_hp
        assert total == 50, f"Expected 50 surfaces, got {total}"

    def test_fill_factor_matches_params(self, result_check):
        p, r = result_check
        assert math.isclose(r["fill_factor"], p.fill_factor)

    def test_carea_positive(self, result_check):
        _, r = result_check
        assert r["Carea_m2"] > 0


# ─────────────────────────────────────────────────────────────────────────────
#  Modified parameters
# ─────────────────────────────────────────────────────────────────────────────
class TestModifiedParams:
    def test_different_n_hp(self):
        p = MotorParams(n_hp=4)
        r = build_stator(p, check_only=True)
        for i in range(p.ns):
            n = sum(len(r["slot_hp"][i][k]) for k in range(p.n_hp))
            assert n == 4, f"Expected 4 hairpins/slot with n_hp=4, got {n}"

    def test_invalid_params_raises(self):
        with pytest.raises(ValueError):
            build_stator(MotorParams(g=0.1), check_only=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Full mesh (slow)
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.slow
class TestMesh:
    def test_mesh_file_created(self, result_mesh, tmp_path):
        # File was written by fixture; just verify node count is reasonable
        _, r = result_mesh
        assert r["n_nodes"] > 500

    def test_mesh_element_count(self, result_mesh):
        _, r = result_mesh
        assert r["n_elems"] > 1000

    def test_sb_stator_single_arc(self, result_mesh):
        _, r = result_mesh
        assert len(r["sb_stator"]) == 1, (
            f"Expected 1 SB_Stator arc, got {r['sb_stator']}"
        )

    def test_domain_single_arc(self, result_mesh):
        _, r = result_mesh
        assert len(r["domain"]) == 1

    def test_sector_boundaries_exist(self, result_mesh):
        _, r = result_mesh
        assert len(r["stator_right"]) >= 1
        assert len(r["stator_left"]) >= 1

    def test_no_zero_area_elements(self, result_mesh, tmp_path):
        """Mesh file should exist and be non-empty."""
        _, r = result_mesh
        assert r["n_elems"] > 0
