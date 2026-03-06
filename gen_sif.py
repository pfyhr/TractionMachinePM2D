"""
gen_sif.py — Elmer SIF generator for the TractionMachinePM2D parametric model.

Generates a complete .sif file from a MotorParams object, using the physical
group naming convention of gen_mesh.py.

Usage:
    python3 gen_sif.py [--out case_gen.sif] [--mesh mesh] [--results results]
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

from motor_params import MotorParams, DEFAULT_PARAMS

π  = math.pi
μ0 = 4e-7 * π   # permeability of free space [H/m]

# ── Material constants (single source of truth used inside gen_sif()) ─────────
M350_50A_PMF_FILE = "m350-50a_20c_extrapolated.pmf"   # B-H curve (extrapolated to 5 T)
M350_50A_LOSS_C1  = 159.9615          # Harmonic Loss Coefficient 1 [W/(T²·Hz·m³)]
M350_50A_LOSS_C2  = 0.934621628891    # Harmonic Loss Coefficient 2 [W/(T²·Hz²·m³)]
N45SH_SIGMA       = 625_000.0         # electrical conductivity [S/m]  (ρ ≈ 1.6 µΩ·m)
N45SH_DENSITY     = 7_500.0           # mass density [kg/m³]

# Phase assignment for the default 48s/8p/3ph design (6 slots per sector).
# Matches gen_mesh._PHASE_MAP exactly.
_PHASE_MAP = ["A+", "A+", "C-", "C-", "B+", "B+"]

# Decode phase string → (phase_index 0/1/2, sign +1/-1)
_PHASE_DECODE: dict[str, tuple[int, int]] = {
    "A+": (0, +1), "A-": (0, -1),
    "B+": (1, +1), "B-": (1, -1),
    "C+": (2, +1), "C-": (2, -1),
}


# ─────────────────────────────────────────────────────────────────────────────
def material_table(params: MotorParams) -> list[dict]:
    """
    Return a list of material property dicts exactly as they appear in the SIF.

    Each dict has keys: id, name, and material-specific property keys.
    Values are the same literals written into the SIF — no separate copy.
    """
    p = params
    H_PM = p.B_r / (p.mu_r * μ0)
    return [
        {
            "id": 1,
            "name": "Air",
            "mu_r": 1.0,
            "epsilon_r": 1.0,
            "sigma [S/m]": 0.0,
        },
        {
            "id": 2,
            "name": "Copper (hairpin)",
            "mu_r": 1.0,
            "epsilon_r": 1.0,
            "sigma [S/m]": 0.0,
            "note": "J applied as body force; resistive loss not modelled in 2D",
        },
        {
            "id": 3,
            "name": "M350-50A (stator laminations)",
            "mu_r": "nonlinear — see PMF",
            "BH curve": M350_50A_PMF_FILE,
            "loss C1 [W/(T²·Hz·m³)]": M350_50A_LOSS_C1,
            "loss C2 [W/(T²·Hz²·m³)]": M350_50A_LOSS_C2,
        },
        {
            "id": 4,
            "name": "M350-50A (rotor laminations)",
            "mu_r": "nonlinear — see PMF",
            "BH curve": M350_50A_PMF_FILE,
            "loss C1 [W/(T²·Hz·m³)]": M350_50A_LOSS_C1,
            "loss C2 [W/(T²·Hz²·m³)]": M350_50A_LOSS_C2,
        },
        {
            "id": 5,
            "name": "N45SH (NdFeB magnet)",
            "mu_r": p.mu_r,
            "B_r [T]": p.B_r,
            "H_PM [A/m]": round(H_PM),
            "sigma [S/m]": N45SH_SIGMA,
            "density [kg/m³]": N45SH_DENSITY,
            "note": "Hci ≈ 2000 kA/m (SH high-temp grade)",
        },
    ]


def conductor_body_map(
    params: MotorParams,
    phase_map: Optional[list[str]] = None,
) -> dict[str, list[int]]:
    """
    Return a dict mapping each phase string (e.g. 'A+', 'B+', 'C-') to the
    list of Elmer body numbers that carry its current density.

    Body numbering is identical to the one gen_sif() writes into the SIF, so
    this can be used to correlate VTU GeometryIds with phases for post-processing.
    """
    p = params
    if phase_map is None:
        phase_map = _PHASE_MAP[:]
    result: dict[str, list[int]] = {}
    n = 1
    n += 1  # Stator_Iron
    n += 1  # Airgap_Stator
    for i in range(p.ns):
        ph = phase_map[i]
        n += 1  # S{i}_Opening
        n += 1  # S{i}_Insul
        for _ in range(p.n_hp):
            result.setdefault(ph, []).append(n)
            n += 1
    return result


# ─────────────────────────────────────────────────────────────────────────────
def gen_sif(
    params: MotorParams,
    mesh_dir: str = "mesh",
    results_dir: str = "results",
    out_path: Optional[Path] = None,
    suffix: str = "2d",
    nper: int = 120,
    ncycle: int = 2,
    rot_dir: int = 1,
    rotor_init_pos: float = 0.0,
    phase_map: Optional[list[str]] = None,
    gamma_deg: float = 0.0,
) -> str:
    """
    Generate a complete Elmer SIF string for the given MotorParams.

    Parameters
    ----------
    params         : motor geometry and physics parameters
    mesh_dir       : path to Elmer mesh directory (relative to SIF location)
    results_dir    : path to results directory
    out_path       : if given, write SIF to this path
    suffix         : suffix for VTU/loss output file names
    nper           : timesteps per mechanical period
    ncycle         : number of mechanical cycles to simulate
    rot_dir        : rotation direction (+1 CCW, -1 CW)
    rotor_init_pos : initial rotor position [deg]
    phase_map      : phase string per slot in sector (length must equal ns).
                     Defaults to ["A+","A+","C-","C-","B+","B+"].
    gamma_deg      : current advance angle [deg] relative to q-axis.
                     0° → pure q-axis (id=0, iq=Is, max SPM torque).
                     Negative → demagnetising id (IPM MTPA direction).
                     Positive → magnetising id (field boosting).

    Returns the SIF as a string (and writes to out_path if given).
    """
    params.validate()
    p = params

    if phase_map is None:
        phase_map = _PHASE_MAP[:]

    if len(phase_map) != p.ns:
        raise ValueError(
            f"phase_map length {len(phase_map)} != ns={p.ns} (slots per sector)"
        )
    for ph in phase_map:
        if ph not in _PHASE_DECODE:
            raise ValueError(
                f"Unknown phase '{ph}'; must be one of {list(_PHASE_DECODE)}"
            )

    # ── Derived electromagnetic parameters ─────────────────────────────────
    H_PM   = p.B_r / (p.mu_r * μ0)   # magnet H-field [A/m]
    Mangle = math.degrees(p.θs / 2)  # magnet sector-centre angle [deg]

    # ── Body force assignment ───────────────────────────────────────────────
    # BF 1 = rotor rotation (no current density)
    # BF 2, 3, ... = unique (phase_index, sign) combos, in appearance order
    seen: dict[tuple[int, int], int] = {}  # (ph_idx, sign) → BF number
    bf_counter = 2
    hp_to_bf: dict[str, int] = {}          # phase_string → BF number
    for ph in dict.fromkeys(phase_map):    # unique, order-preserving
        key = _PHASE_DECODE[ph]
        if key not in seen:
            seen[key] = bf_counter
            bf_counter += 1
        hp_to_bf[ph] = seen[key]

    # ── Body numbering (matches gen_mesh.py physical group order) ───────────
    body_num: dict[str, int] = {}
    n = 1
    body_num["Stator_Iron"]   = n; n += 1
    body_num["Airgap_Stator"] = n; n += 1
    for i in range(p.ns):
        ph = phase_map[i]
        body_num[f"S{i}_Opening"] = n; n += 1
        body_num[f"S{i}_Insul"]   = n; n += 1
        for k in range(p.n_hp):
            body_num[f"S{i}_HP{k}_{ph}"] = n; n += 1
    body_num["Airgap_Rotor"] = n; n += 1
    body_num["Rotor_Iron"]   = n; n += 1
    body_num["Magnet"]       = n; n += 1
    if p.w_air > 0:
        body_num["AirPocket"] = n; n += 1
    body_num["Shaft"]        = n; n += 1

    # Rotor body indices for RigidMeshMapper
    rotor_names = ["Airgap_Rotor", "Rotor_Iron", "Magnet"]
    if p.w_air > 0:
        rotor_names.append("AirPocket")
    rotor_names.append("Shaft")
    rotor_ids = [body_num[nm] for nm in rotor_names]

    # ── Helper: emit one Body block ─────────────────────────────────────────
    def body_block(
        num: int, name: str, eq: int, mat: int,
        bf: Optional[int] = None,
        tg: Optional[int] = None,
    ) -> list[str]:
        b = [f"Body {num}", f"  Name = {name}",
             f"  Equation = {eq}", f"  Material = {mat}"]
        if bf is not None:
            b.append(f"  Body Force = {bf}")
        if tg is not None:
            b.append(f"  Torque Groups = Integer {tg}")
        b.append("End")
        return b

    # ── Assemble SIF ────────────────────────────────────────────────────────
    L: list[str] = []

    # ── MATC header ─────────────────────────────────────────────────────────
    L += [
        f"! 2D Magnetodynamics — {p.Qs}s/{p.Qp}p/{p.m}ph inset-PM traction motor",
        f"! Stator OD={2*p.R_so:.0f}mm, L_active={p.L_active:.0f}mm, "
        f"{math.degrees(p.θs):.0f}° sector with anti-periodic BCs.",
        "! Mesh in mm → Coordinate Scaling = 0.001 converts to SI metres.",
        "!",
        f"$ WM = 2*pi*{p.rpm}/60       ! Mechanical angular velocity [rad/s]",
        f"$ PP = {p.PP}",
        f"$ H_PM = {H_PM:.1f}           ! Magnet H-field [A/m]",
        f"$ Is = {p.Is}              ! Peak phase current [A]",
        f"$ Carea = {p.Carea:.6e}    ! Conductor area per slot [m²]",
        f"$ Mangle1 = {Mangle:.4f}   ! Magnet centre angle in sector [deg]",
        "$ DegreesPerSec = WM*180.0/pi",
        f"$ RotorInitPos = {rotor_init_pos}",
        f"$ gamma_deg = {gamma_deg:.4f}    ! Current advance angle (0=q-axis, -=IPM MTPA)",
        f"$ rpm = {p.rpm}",
        "$ fme = rpm/60",
        "$ fel = fme*PP",
        f"$ rot_dir = {rot_dir}",
        "!",
        f"$ nper  = {nper}",
        "$ dt    = 1.0/(fme*nper)",
        f"$ ncycle = {ncycle}",
        "!",
        f'$ suffix = "{suffix}"',
        "",
        "echo off",
        "",
    ]

    # ── Header ──────────────────────────────────────────────────────────────
    L += [
        "Header",
        "  CHECK KEYWORDS Warn",
        f'  Mesh DB "{mesh_dir}"',
        '  Include Path "."',
        f'  Results Directory "{results_dir}"',
        "End",
        "",
    ]

    # ── Constants ───────────────────────────────────────────────────────────
    L += [
        "Constants",
        "  Permittivity of Vacuum = 8.8542e-12",
        "End",
        "",
    ]

    # ── Simulation ──────────────────────────────────────────────────────────
    L += [
        "Simulation",
        "  Max Output Level = 5",
        "  Coordinate System = Cartesian 2D",
        "  Coordinate Scaling = 0.001    ! mesh in mm → SI metres",
        "  Simulation Type = Transient",
        "  Timestepping Method = BDF",
        "  BDF Order = 2",
        "  Timestep Sizes = $dt",
        "  Timestep Intervals = $nper*ncycle+1",
        "  Timestep Start Zero = Logical True",
        "  Time Period = $1/fme",
        "",
        "  Output Intervals = 1",
        "",
        "  Use Mesh Names = True",
        "",
        f"  Rotor Radius = Real {p.R_sb * 1e-3:.6f}   ! R_sb [m]",
        "End",
        "",
    ]

    # ── Materials ───────────────────────────────────────────────────────────
    L += [
        "!--- MATERIALS ---",
        "Material 1",
        '  Name = "Air"',
        "  Relative Permeability = 1",
        "  Relative Permittivity = 1",
        "  Electric Conductivity = 0",
        "End",
        "",
        "Material 2",
        '  Name = "Copper"',
        "  Relative Permeability = 1",
        "  Relative Permittivity = 1",
        "  Electric Conductivity = 0",
        "End",
        "",
        "Material 3",
        '  Name = "StatorMaterial"',
        f'  Include "{M350_50A_PMF_FILE}"',
        f"  Harmonic Loss Coefficient 1 = Real {M350_50A_LOSS_C1}",
        f"  Harmonic Loss Coefficient 2 = Real {M350_50A_LOSS_C2}",
        "End",
        "",
        "Material 4",
        '  Name = "RotorMaterial"',
        f'  Include "{M350_50A_PMF_FILE}"',
        f"  Harmonic Loss Coefficient 1 = Real {M350_50A_LOSS_C1}",
        f"  Harmonic Loss Coefficient 2 = Real {M350_50A_LOSS_C2}",
        "End",
        "",
        "Material 5",
        '  Name = "N45SH"',
        f"  ! NdFeB N45SH: Br={p.B_r:.3f} T, Hci≈2000 kA/m, mu_r={p.mu_r}, sigma={N45SH_SIGMA/1e3:.0f} kS/m",
        f"  Relative Permeability = {p.mu_r}",
        "  Relative Permittivity = 1",
        f"  ! H_PM = Br / (mu_r * mu0) = {H_PM:.0f} A/m",
        "",
        "  Magnetization 1 = Variable time",
        '    Real MATC "H_PM*cos(rot_dir*WM*tx(0) + (RotorInitPos + Mangle1)*pi/180)"',
        "  Magnetization 2 = Variable time",
        '    Real MATC "H_PM*sin(rot_dir*WM*tx(0) + (RotorInitPos + Mangle1)*pi/180)"',
        "",
        f"  Electric Conductivity = {N45SH_SIGMA:.1f}  ! rho ≈ 1.6 µΩ·m",
        f"  Density = {N45SH_DENSITY:.1f}              ! kg/m³",
        "End",
        "",
    ]

    # ── Body Forces ─────────────────────────────────────────────────────────
    L += [
        "!--- BODY FORCES ---",
        "Body Force 1",
        '  Name = "BodyForce_Rotation"',
        "End",
        "",
    ]
    for ph_str, bf_num in sorted(hp_to_bf.items(), key=lambda x: x[1]):
        ph_idx, sign = _PHASE_DECODE[ph_str]
        jstr = "Is/Carea" if sign > 0 else "-Is/Carea"
        L += [
            f"Body Force {bf_num}",
            f'  Name = "Phase_{ph_str}_BF"',
            "  Current Density = Variable time",
            f'    Real MATC "{jstr}*sin(tx(0)*2*pi*fel - {ph_idx}*2*pi/3 - PP*(RotorInitPos+Mangle1)*pi/180 - gamma_deg*pi/180)"',
            "End",
            "",
        ]

    # ── Bodies ──────────────────────────────────────────────────────────────
    L.append("!--- BODIES ---")

    L += body_block(body_num["Stator_Iron"], "Stator_Iron", 2, 3)
    L.append("")
    L += body_block(body_num["Airgap_Stator"], "Airgap_Stator", 1, 1)
    L.append("")

    for i in range(p.ns):
        ph = phase_map[i]
        bf = hp_to_bf[ph]
        L += body_block(body_num[f"S{i}_Opening"], f"S{i}_Opening", 1, 1)
        L.append("")
        L += body_block(body_num[f"S{i}_Insul"], f"S{i}_Insul", 1, 1)
        L.append("")
        for k in range(p.n_hp):
            nm = f"S{i}_HP{k}_{ph}"
            L += body_block(body_num[nm], nm, 1, 2, bf=bf)
            L.append("")

    L += body_block(body_num["Airgap_Rotor"], "Airgap_Rotor", 1, 1, bf=1)
    L.append("")
    L += body_block(body_num["Rotor_Iron"], "Rotor_Iron", 2, 4, bf=1, tg=1)
    L.append("")
    L += body_block(body_num["Magnet"], "Magnet", 1, 5, bf=1, tg=1)
    L.append("")
    if p.w_air > 0:
        L += body_block(body_num["AirPocket"], "AirPocket", 1, 1, bf=1, tg=1)
        L.append("")
    L += body_block(body_num["Shaft"], "Shaft", 1, 1, bf=1, tg=1)
    L.append("")

    # ── Equations ───────────────────────────────────────────────────────────
    L += [
        "Equation 1",
        '  Name = "Model_Domain"',
        "  Active Solvers(5) = 1 2 3 4 5",
        "End",
        "",
        "Equation 2",
        '  Name = "Laminations"',
        "  Active Solvers(6) = 1 2 3 4 5 6",
        "End",
        "",
    ]

    # ── Solvers ─────────────────────────────────────────────────────────────
    n_rot = len(rotor_ids)
    rot_ids_str = " ".join(str(x) for x in rotor_ids)

    L += [
        "!--- SOLVERS ---",
        "Solver 1",
        "  Exec Solver = Before Timestep",
        "  Equation = MeshDeform",
        '  Procedure = "RigidMeshMapper" "RigidMeshMapper"',
        "",
        "  Rotor Mode = Logical True",
        f"  Rotor Bodies({n_rot}) = Integer {rot_ids_str}",
        "  Mesh Rotate 3 = Variable time",
        '    Real MATC "RotorInitPos + rot_dir*tx(0)*DegreesPerSec"',
        "End",
        "",
        "Solver 2",
        "  Equation = MgDyn2D",
        '  Procedure = "MagnetoDynamics2D" "MagnetoDynamics2D"',
        "  Exec Solver = Always",
        "  Variable = A",
        "",
        "  Nonlinear System Convergence Tolerance = 1e-05",
        "  Nonlinear System Max Iterations = 100",
        "  Nonlinear System Newton After Iterations = 3",
        "  Nonlinear System Relaxation Factor = 0.9",
        "  Nonlinear System Convergence Without Constraints = True",
        "",
        "  Export Lagrange Multiplier = True",
        "  Linear System Abort Not Converged = False",
        "  Linear System Solver = Iterative",
        "  Linear System Iterative Method = idrs",
        "  Optimize Bandwidth = Logical True",
        "  Linear System Preconditioning = ILU2",
        "  Linear System Max Iterations = 5000",
        "  Linear System Residual Output = 20",
        "  Linear System Convergence Tolerance = 1e-08",
        "",
        "  Mortar BCs Additive = True",
        "  Apply Conforming BCs = True",
        "",
        "  Handle Assembly = Logical True",
        "End",
        "",
        "Solver 3",
        "  Exec Solver = Always",
        "  Equation = CalcFields",
        '  Potential Variable = "A"',
        '  Procedure = "MagnetoDynamics" "MagnetoDynamicsCalcFields"',
        "  Calculate Nodal Forces = True",
        "  Calculate Magnetic Vector Potential = False",
        "  Calculate Winding Voltage = Logical False",
        "  Calculate Current Density = True",
        "  Calculate Maxwell Stress = False",
        "  Calculate JxB = False",
        "  Calculate Magnetic Field Strength = True",
        "  Calculate Magnetic Flux Density = True",
    ]
    L += [
        "  Linear System Solver = Direct",
        "  Linear System Direct Method = umfpack",
        ]
    L += [
        "",
        "  Calculate Nodal Fields = False",
        "  Calculate Elemental Fields = True",
        "End",
        "",
        "Solver 4",
        "  Exec Solver = After Timestep",
        '  Equation = "VtuOutput"',
        '  Procedure = "ResultOutputSolve" "ResultOutputSolver"',
        '  Output File Name = step-$suffix$',
        "  Vtu Format = True",
        "  Binary Output = True",
        "  Single Precision = True",
        "  Save Geometry Ids = True",
        "  Show Variables = True",
        "  Discontinuous Bodies = True",
        "End",
        "",
        "Solver 5",
        "  Exec Solver = After Timestep",
        "  Equation = SaveScalars",
        '  Filename = "scalars.dat"',
        '  Procedure = "SaveData" "SaveScalars"',
        "  Show Norm Index = 1",
        "End",
        "",
        "Solver 6",
        "  Exec Solver = Always",
        "  Equation = FourierLoss",
        '  Procedure = "FourierLoss" "FourierLossSolver"',
        "  Target Variable = A",
        "",
        "  Frequency = Real $ fel",
        "",
        "  Fourier Start Time = Real 0.0",
        "  Fourier Integrate Cycles = Integer 1",
        "  Separate Loss Components = Logical True",
        "",
        f"  Fourier Series Components = {p.Qs}",
        '  Fourier Loss Filename = File "results/loss-$suffix$.dat"',
        "",
        "  Harmonic Loss Frequency Exponent 1 = Real 1.0",
        "  Harmonic Loss Frequency Exponent 2 = Real 2.0",
        "",
        "  Harmonic Loss Field Exponent 1 = Real 2.0",
        "  Harmonic Loss Field Exponent 2 = Real 2.0",
        "",
        "  Calculate Elemental Fields = True",
        "End",
        "",
    ]

    # ── Boundary Conditions ─────────────────────────────────────────────────
    L += [
        "!--- BOUNDARY CONDITIONS ---",
        "Boundary Condition 1",
        "  Name = SB_Rotor",
        "  Mortar BC = 2",
        "  Anti Rotational Projector = True",
        "  Galerkin Projector = True",
        "End",
        "",
        "Boundary Condition 2",
        "  Name = SB_Stator",
        "End",
        "",
        "Boundary Condition 3",
        "  Name = Domain",
        "  A = Real 0",
        "End",
        "",
        "Boundary Condition 4",
        "  Name = Stator_Right",
        "  Conforming BC = 5",
        "  Mortar BC Static = True",
        "  Anti Radial Projector = True",
        "  Galerkin Projector = True",
        "End",
        "",
        "Boundary Condition 5",
        "  Name = Stator_Left",
        "End",
        "",
        "Boundary Condition 6",
        "  Name = Rotor_Right",
        "  Conforming BC = 7",
        "  Mortar BC Static = True",
        "  Anti Radial Projector = True",
        "  Galerkin Projector = True",
        "End",
        "",
        "Boundary Condition 7",
        "  Name = Rotor_Left",
        "End",
        "",
    ]

    text = "\n".join(L)

    if out_path is not None:
        Path(out_path).write_text(text, encoding="utf-8")
        print(f"Wrote {len(text.splitlines())} lines to {out_path}")

    return text


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Generate Elmer SIF for the parametric TractionMachinePM2D model"
    )
    ap.add_argument("--out",     default="case_gen.sif", help="Output SIF path")
    ap.add_argument("--mesh",    default="mesh",          help="Elmer mesh directory")
    ap.add_argument("--results", default="results",       help="Results directory")
    ap.add_argument("--suffix",  default="2d",            help="Output file suffix")
    args = ap.parse_args()

    p = MotorParams().validate()
    text = gen_sif(
        p,
        mesh_dir=args.mesh,
        results_dir=args.results,
        out_path=Path(args.out),
        suffix=args.suffix,
    )
    lines = text.split("\n")
    print(f"\nGenerated {len(lines)} lines, {len(text)} chars")
    print("─" * 60)
    print("\n".join(lines[:30]))
    print("...")
