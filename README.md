# TractionMachinePM2D

Parametric Python toolchain for designing, meshing, and simulating a
48-slot / 8-pole / 3-phase inset-PM traction motor using
[Elmer FEM](https://www.elmerfem.org/).

## Motor Specification

| Parameter | Value |
|---|---|
| Topology | 48 slots / 8 poles / 3 phases |
| Stator OD / bore | 220 mm / 148 mm |
| Active length | 170 mm |
| Air gap | 0.75 mm |
| Rotor OD / shaft OD | 146.5 mm / 50 mm |
| Magnet material | NdFeB N45SH — Br = 1.35 T, μr = 1.05 |
| Lamination | M350-50A silicon steel |
| Winding | Hairpin, 6 layers/slot, fill factor ~55 % |
| Rated speed | 1500 rpm (f_el = 100 Hz) |
| Peak phase current | 200 A |
| Model sector | 45° (1/8 machine) with anti-periodic BCs |

## Toolchain Overview

```
motor_params.py   →  MotorParams dataclass: all geometry, winding, and
                       operating-point parameters with validation
      ↓
gen_stator.py     →  gmsh geometry for the stator sector
gen_rotor.py      →  gmsh geometry for the rotor sector
      ↓
gen_mesh.py       →  Combined sector mesh with sliding air-gap boundary
      ↓
gen_sif.py        →  Elmer SIF generator (MATC time-stepping, mortar BCs,
                       FourierLoss, RigidMeshMapper rotor motion)
      ↓
ElmerSolver       →  Transient 2D magnetodynamics simulation
      ↓
postprocess.py    →  Torque / iron-loss extraction, back-EMF from flux
                       linkage, magnetic flux density plot
```

Everything is a **single source of truth**: material constants, body
numbering, phase maps, and scaling factors live in one place and are
imported by all downstream modules and the notebook.

## Results

Key outputs from the reference design at rated conditions (1500 rpm, 200 A,
γ = 0° / q-axis):

| Quantity | Value |
|---|---|
| Mean torque | computed from scalars.dat |
| Back-EMF peak (1000 rpm, Is = 0) | computed from VTU flux linkage |
| Stator iron loss | computed from FourierLoss |
| Rotor iron loss | computed from FourierLoss |

Run `example.ipynb` end-to-end to populate the table above for your
specific design.

## Directory Structure

```
TractionMachinePM2D/
├── motor_params.py          # MotorParams dataclass + suggest_slot()
├── gen_stator.py            # Stator sector geometry (gmsh)
├── gen_rotor.py             # Rotor sector geometry (gmsh)
├── gen_mesh.py              # Combined sector mesh with air-gap
├── gen_sif.py               # Elmer SIF generator
├── postprocess.py           # Results post-processing
├── example.ipynb            # End-to-end showcase notebook
├── m350-50a_20c_extrapolated.pmf  # B-H curve (extrapolated to 5 T)
├── tests/                   # pytest unit tests (155+ tests)
│   ├── test_motor_params.py
│   ├── test_gen_stator.py
│   ├── test_gen_rotor.py
│   ├── test_gen_mesh.py
│   ├── test_gen_sif.py
│   └── test_postprocess.py
├── mesh/                    # Elmer mesh (auto-generated, git-ignored)
└── results*/                # Simulation output (auto-generated, git-ignored)
```

## Dependencies

### Python

```
pip install numpy matplotlib meshio
```

- Python ≥ 3.10
- [gmsh](https://gmsh.info/) Python API: `pip install gmsh`
- [meshio](https://github.com/nschloe/meshio): `pip install meshio`

### Elmer FEM

Install [Elmer FEM](https://www.elmerfem.org/blog/binaries/) with
`ElmerSolver` and `ElmerGrid` on your `PATH`.

On macOS with Homebrew:
```bash
brew install elmerfem
```

## Workflow

### 1. Run the notebook

Open `example.ipynb` in JupyterLab or VS Code and run all cells in order.
The notebook will:

1. Define the motor geometry via `MotorParams`
2. Display material properties (directly from `gen_sif.py` constants)
3. Build and visualise the stator, rotor, and full-sector mesh
4. Generate the Elmer SIF
5. Run the transient simulation (serial `ElmerSolver`)
6. Run a current-angle (γ) sweep from −90° to 0° to find MTPA
7. Run a zero-current back-EMF simulation at 1000 rpm
8. Plot torque, flux density, and back-EMF waveforms

### 2. Run tests

```bash
pytest tests/ -v
```

Fast tests (~1 s) cover geometry, parameter validation, SIF generation,
and post-processing. Slow mesh-build tests are marked `slow` and skipped
by default.

### 3. Custom design

Modify parameters at the top of the notebook:

```python
p = MotorParams(
    R_si=80.0,      # bore radius [mm]
    h_slot=22.0,    # slot depth [mm]
    rpm=3000.0,     # rated speed [rpm]
    Is=250.0,       # peak phase current [A]
)
```

`p.validate()` checks geometric feasibility and raises a descriptive
`ValueError` if the design is infeasible.

## Key Physics Notes

- **2D sector model**: 45° sector (1/8 machine) with anti-periodic BCs on
  the sector edges. The solution is scaled to full machine via
  `SCALE = Qp × L_active = 8 × 0.170 = 1.36 m`.

- **Current advance angle γ**: `γ = 0°` → pure q-axis (maximum torque per
  amp for SPM); `γ < 0°` → negative d-axis current for IPM flux weakening
  / MTPA.

- **Back-EMF extraction**: At zero current, the induced phase voltage is
  computed as `e = −dΨ/dt` where `Ψ` is the flux linkage obtained by
  integrating the MVP field A over each phase's conductor cross-sections in
  the VTU output.

- **Iron losses**: Computed by Elmer's `FourierLoss` solver using M350-50A
  harmonic loss coefficients (C₁ = 159.96 W/(T²·Hz·m³),
  C₂ = 0.935 W/(T²·Hz²·m³)).

- **Mortar boundary conditions**: The sliding air-gap interface uses Elmer's
  mortar BC projector with anti-rotational coupling, which requires serial
  `ElmerSolver` (MPI domain decomposition conflicts with this BC type for
  small 2D meshes).

## References

- [Elmer FEM Documentation](https://www.elmerfem.org/elmersolver/Documentation/)
- [ElmerCSC elmer-elmag examples](https://github.com/ElmerCSC/elmer-elmag)
- M350-50A B-H data: ThyssenKrupp Steel datasheet, extrapolated to 5 T
  using exponential blending (see `bh_extrapolation.ipynb`)
