"""
Tests for gen_sif.py.

All tests are fast — no gmsh / meshing required.
"""
import math
import re
import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from motor_params import MotorParams
from gen_sif import gen_sif, _PHASE_MAP, _PHASE_DECODE

π  = math.pi
μ0 = 4e-7 * π


@pytest.fixture(scope="module")
def default_sif():
    p = MotorParams()
    return p, gen_sif(p)


def _count_bodies(text: str) -> int:
    """Count Body N blocks, excluding Body Force blocks."""
    return len(re.findall(r"^Body \d", text, re.MULTILINE))


def _count_body_forces(text: str) -> int:
    return len(re.findall(r"^Body Force \d", text, re.MULTILINE))


# ─────────────────────────────────────────────────────────────────────────────
#  Basic structure
# ─────────────────────────────────────────────────────────────────────────────
class TestBasicStructure:
    def test_generates_string(self, default_sif):
        _, s = default_sif
        assert isinstance(s, str) and len(s) > 500

    def test_header_section(self, default_sif):
        _, s = default_sif
        assert "Header" in s
        assert "Mesh DB" in s

    def test_simulation_section(self, default_sif):
        _, s = default_sif
        assert "Coordinate System = Cartesian 2D" in s
        assert "Simulation Type = Transient" in s

    def test_use_mesh_names(self, default_sif):
        _, s = default_sif
        assert "Use Mesh Names = True" in s

    def test_rotor_radius_metres(self, default_sif):
        p, s = default_sif
        assert f"{p.R_sb * 1e-3:.6f}" in s

    def test_coordinate_scaling(self, default_sif):
        _, s = default_sif
        assert "Coordinate Scaling = 0.001" in s

    def test_six_solvers(self, default_sif):
        _, s = default_sif
        # Solver 1 through Solver 6
        for i in range(1, 7):
            assert f"\nSolver {i}\n" in s

    def test_seven_boundary_conditions(self, default_sif):
        _, s = default_sif
        for i in range(1, 8):
            assert f"\nBoundary Condition {i}\n" in s


# ─────────────────────────────────────────────────────────────────────────────
#  Materials
# ─────────────────────────────────────────────────────────────────────────────
class TestMaterials:
    def test_air_material(self, default_sif):
        _, s = default_sif
        assert '"Air"' in s

    def test_copper_material(self, default_sif):
        _, s = default_sif
        assert '"Copper"' in s

    def test_stator_material_pmf(self, default_sif):
        _, s = default_sif
        assert '"StatorMaterial"' in s
        assert "m350-50a_20c_extrapolated.pmf" in s

    def test_rotor_material(self, default_sif):
        _, s = default_sif
        assert '"RotorMaterial"' in s

    def test_pm_mu_r(self, default_sif):
        p, s = default_sif
        assert f"Relative Permeability = {p.mu_r}" in s

    def test_pm_h_field(self, default_sif):
        p, s = default_sif
        H_PM = p.B_r / (p.mu_r * μ0)
        assert f"{H_PM:.1f}" in s

    def test_carea_in_matc_header(self, default_sif):
        p, s = default_sif
        assert f"{p.Carea:.6e}" in s

    def test_magnetization_matc(self, default_sif):
        _, s = default_sif
        assert "Magnetization 1 = Variable time" in s
        assert "H_PM*cos(" in s
        assert "H_PM*sin(" in s


# ─────────────────────────────────────────────────────────────────────────────
#  Body forces
# ─────────────────────────────────────────────────────────────────────────────
class TestBodyForces:
    def test_rotation_bf_exists(self, default_sif):
        _, s = default_sif
        assert '"BodyForce_Rotation"' in s

    def test_phase_bf_count(self, default_sif):
        p, s = default_sif
        n_unique_phases = len(set(_PHASE_MAP))
        # 1 rotation + n_unique phase body forces
        assert _count_body_forces(s) == 1 + n_unique_phases

    def test_current_density_in_bf(self, default_sif):
        _, s = default_sif
        assert "Current Density = Variable time" in s
        assert "Is/Carea" in s

    def test_negative_direction_in_bf(self, default_sif):
        _, s = default_sif
        # C- has sign=-1 → "-Is/Carea"
        assert "-Is/Carea" in s


# ─────────────────────────────────────────────────────────────────────────────
#  Bodies
# ─────────────────────────────────────────────────────────────────────────────
class TestBodies:
    def test_stator_iron_body(self, default_sif):
        _, s = default_sif
        assert "Name = Stator_Iron" in s

    def test_airgap_stator_body(self, default_sif):
        _, s = default_sif
        assert "Name = Airgap_Stator" in s

    def test_airgap_rotor_body(self, default_sif):
        _, s = default_sif
        assert "Name = Airgap_Rotor" in s

    def test_rotor_iron_body(self, default_sif):
        _, s = default_sif
        assert "Name = Rotor_Iron" in s

    def test_magnet_body(self, default_sif):
        _, s = default_sif
        assert "Name = Magnet" in s

    def test_air_pocket_body(self, default_sif):
        p, s = default_sif
        if p.w_air > 0:
            assert "Name = AirPocket" in s

    def test_shaft_body(self, default_sif):
        _, s = default_sif
        assert "Name = Shaft" in s

    def test_all_slot_openings(self, default_sif):
        p, s = default_sif
        for i in range(p.ns):
            assert f"Name = S{i}_Opening" in s

    def test_all_slot_insulations(self, default_sif):
        p, s = default_sif
        for i in range(p.ns):
            assert f"Name = S{i}_Insul" in s

    def test_all_hairpin_bodies(self, default_sif):
        p, s = default_sif
        for i in range(p.ns):
            ph = _PHASE_MAP[i]
            for k in range(p.n_hp):
                assert f"Name = S{i}_HP{k}_{ph}" in s

    def test_body_count(self, default_sif):
        p, s = default_sif
        n_pockets = 1 if p.w_air > 0 else 0
        # Stator_Iron + Airgap_Stator + ns*(Opening + Insul + n_hp HP)
        # + Airgap_Rotor + Rotor_Iron + Magnet + [AirPocket] + Shaft
        expected = 2 + p.ns * (2 + p.n_hp) + 4 + n_pockets
        assert _count_bodies(s) == expected

    def test_torque_group_on_rotor_iron(self, default_sif):
        _, s = default_sif
        idx = s.index("Name = Rotor_Iron")
        block = s[idx: idx + 200]
        assert "Torque Groups = Integer 1" in block

    def test_rotation_bf_on_airgap_rotor(self, default_sif):
        _, s = default_sif
        idx = s.index("Name = Airgap_Rotor")
        block = s[idx: idx + 200]
        assert "Body Force = 1" in block

    def test_airgap_rotor_no_torque_group(self, default_sif):
        _, s = default_sif
        idx = s.index("Name = Airgap_Rotor")
        # The next "End" closes this block; torque group should not be in it
        end_idx = s.index("End", idx)
        block = s[idx: end_idx]
        assert "Torque Groups" not in block


# ─────────────────────────────────────────────────────────────────────────────
#  Solvers
# ─────────────────────────────────────────────────────────────────────────────
class TestSolvers:
    def test_rigid_mesh_mapper(self, default_sif):
        _, s = default_sif
        assert "RigidMeshMapper" in s
        assert "Rotor Mode = Logical True" in s

    def test_rotor_bodies_count_in_solver(self, default_sif):
        p, s = default_sif
        n_pockets = 1 if p.w_air > 0 else 0
        # Airgap_Rotor + Rotor_Iron + Magnet + [AirPocket] + Shaft
        n_rotor = 4 + n_pockets
        assert f"Rotor Bodies({n_rotor})" in s

    def test_mgdyn2d_solver(self, default_sif):
        _, s = default_sif
        assert "MagnetoDynamics2D" in s
        assert "Variable = A" in s
        assert "Mortar BCs Additive = True" in s

    def test_calcfields_solver(self, default_sif):
        _, s = default_sif
        assert "MagnetoDynamicsCalcFields" in s
        assert "Linear System Solver = Direct" in s
        assert "Linear System Direct Method = umfpack" in s

    def test_vtu_solver(self, default_sif):
        _, s = default_sif
        assert "ResultOutputSolver" in s
        assert "Vtu Format = True" in s

    def test_fourier_loss_solver(self, default_sif):
        p, s = default_sif
        assert "FourierLossSolver" in s
        assert f"Fourier Series Components = {p.Qs}" in s

    def test_save_scalars_solver(self, default_sif):
        _, s = default_sif
        assert "SaveScalars" in s
        assert 'Filename = "scalars.dat"' in s

    def test_equations_defined(self, default_sif):
        _, s = default_sif
        assert '"Model_Domain"' in s
        assert '"Laminations"' in s
        assert "Active Solvers(6) = 1 2 3 4 5 6" in s


# ─────────────────────────────────────────────────────────────────────────────
#  Boundary conditions
# ─────────────────────────────────────────────────────────────────────────────
class TestBoundaryConditions:
    def test_domain_bc(self, default_sif):
        _, s = default_sif
        assert "Name = Domain" in s
        assert "A = Real 0" in s

    def test_sb_rotor_mortar(self, default_sif):
        _, s = default_sif
        assert "Name = SB_Rotor" in s
        assert "Mortar BC = 2" in s
        assert "Anti Rotational Projector = True" in s

    def test_sb_stator_bc(self, default_sif):
        _, s = default_sif
        assert "Name = SB_Stator" in s

    def test_stator_right_conforming(self, default_sif):
        _, s = default_sif
        assert "Name = Stator_Right" in s
        assert "Conforming BC = 5" in s
        assert "Anti Radial Projector = True" in s

    def test_stator_left_bc(self, default_sif):
        _, s = default_sif
        assert "Name = Stator_Left" in s

    def test_rotor_right_conforming(self, default_sif):
        _, s = default_sif
        assert "Name = Rotor_Right" in s
        assert "Conforming BC = 7" in s

    def test_rotor_left_bc(self, default_sif):
        _, s = default_sif
        assert "Name = Rotor_Left" in s


# ─────────────────────────────────────────────────────────────────────────────
#  File I/O and customisation
# ─────────────────────────────────────────────────────────────────────────────
class TestCustomisation:
    def test_writes_file(self, tmp_path):
        p = MotorParams()
        out = tmp_path / "test.sif"
        text = gen_sif(p, out_path=out)
        assert out.exists()
        assert out.read_text(encoding="utf-8") == text

    def test_custom_mesh_dir(self):
        p = MotorParams()
        assert 'Mesh DB "my_mesh"' in gen_sif(p, mesh_dir="my_mesh")

    def test_custom_results_dir(self):
        p = MotorParams()
        assert 'Results Directory "my_results"' in gen_sif(p, results_dir="my_results")

    def test_custom_suffix(self):
        p = MotorParams()
        assert '"my_suffix"' in gen_sif(p, suffix="my_suffix")

    def test_custom_phase_map(self):
        p = MotorParams()
        # Reverse phase assignments
        pm = ["B+", "B+", "A-", "A-", "C+", "C+"]
        s = gen_sif(p, phase_map=pm)
        assert "Name = S0_HP0_B+" in s
        assert "Name = S2_HP0_A-" in s


# ─────────────────────────────────────────────────────────────────────────────
#  Validation
# ─────────────────────────────────────────────────────────────────────────────
class TestValidation:
    def test_wrong_phase_map_length(self):
        p = MotorParams()
        with pytest.raises(ValueError, match="phase_map length"):
            gen_sif(p, phase_map=["A+", "B+"])

    def test_unknown_phase_string(self):
        p = MotorParams()
        bad = ["A+", "A+", "X!", "X!", "B+", "B+"]
        with pytest.raises(ValueError, match="Unknown phase"):
            gen_sif(p, phase_map=bad)

    def test_invalid_motor_params_raises(self):
        p = MotorParams(R_si=50.0, R_so=40.0)  # inverted radii
        with pytest.raises(ValueError):
            gen_sif(p)

    def test_no_air_pockets_when_w_air_zero(self):
        p = MotorParams(w_air=0.0)
        s = gen_sif(p)
        assert "Name = AirPocket" not in s
        # Body count should be one less
        n_pockets = 0
        expected = 2 + p.ns * (2 + p.n_hp) + 4 + n_pockets
        assert _count_bodies(s) == expected
