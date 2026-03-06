# TractionMachinePM2D — Notes

## Machine Specifications
- 48 slots, 8 poles (PP=4), 3-phase, inset PM (surface-inset)
- Stator OD = 220mm, bore R_si = 74mm, split ratio 0.673
- Active length = 170mm
- Air gap g = 0.75mm, sliding boundary R_sb = 73.625mm
- Magnet: 80% pole pitch arc (36°), depth 4mm, R_mi = 69.25mm
- Shaft R_ri = 25mm
- Speed = 1500 RPM, fel = 100 Hz, dt = 3.333e-4 s

## Mesh Generation
- Script: `generate_mesh.py` (gmsh OCC Python)
- **Two separate fragment() calls** (stator region + rotor region) to create non-conforming
  sliding mesh at R_sb — required for Elmer Mortar BC
- Mesh: ~2022 nodes, 3547 elements (coarse, good for development)
- Coordinate system: mm in gmsh → use `Coordinate Scaling = 0.001` in SIF
- Bodies: 12 (Stator_Lam, 6 slots S0–S5, Airgap_Stator, Airgap_Rotor, Rotor_Lam, Magnet, Shaft)
- Boundaries: 7 (SB_Rotor=1, SB_Stator=2, Domain=3, Stator-Right=4, Stator-Left=5, Rotor-Right=6, Rotor-Left=7)

## Winding Assignment (45° sector, 6 slots)
| Slot | Body | Phase | Body Force |
|------|------|-------|------------|
| S0, S1 | 2, 3 | Phase A+ | BF 2 (dir=+1) |
| S2, S3 | 4, 5 | Phase C- | BF 4 (dir=-1) |
| S4, S5 | 6, 7 | Phase B+ | BF 3 (dir=+1) |

## Scaling (CRITICAL)
- 45° sector model, N_sectors = 8 (= 2×PP)
- FourierLoss 2D outputs W/m (per unit depth)
- **SCALE = N_sectors × L_active = 8 × 0.170 = 1.36**
- Torque (raw per sector) × 8 = full machine torque

## Placeholder Parameters (update for real design)
- Is = 200.0 A (peak ampere-turns per coil side)
- Carea = 6.85e-5 m² (approx 45% fill × slot area 152mm²)
- These give ~640-720 N·m and ~258 W iron losses at 1500 RPM

## Run Command
```bash
cd /Users/pontus/repos/elmer-elmag/TractionMachinePM2D
ElmerSolver case.sif     # ~70s walltime, 241 steps
```

## Results Location
- `results/scalars.dat` — timestep torque data
- `results/loss-2d.dat` — FourierLoss per body
- `results/step-2d_t*.vtu` — VTU field files
