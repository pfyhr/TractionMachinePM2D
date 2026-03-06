# PM Motor FEM Toolchain — TODO

Goal: a clean, parametric Python toolchain that takes motor design parameters,
generates a 2D FEM mesh (gmsh), sets up Elmer transient/magnetostatic simulations,
runs current-angle and rotor-angle sweeps, and post-processes key metrics.

---

## Phase 1 — Motor Parameters  `motor_params.py`

- [ ] `MotorParams` dataclass with all geometric + winding parameters
      (Qs, Qp, R_so, R_si, g, b_slot, h_slot, h_m, mag_frac, n_hp, t_liner,
       t_enam, Is, L_active, rpm, material strings …)
- [ ] `validate()` method: check feasibility and raise informative errors
      - tooth width > 0 at bore and at slot top
      - fill factor in [0.3, 0.85]
      - magnet arc < pole pitch
      - slot depth fits inside stator back iron
      - air gap > 0.3 mm (manufacturing floor)
- [ ] `suggest_slot()` helper: given Qs/Qp/R_si/R_so, propose b_slot & h_slot
      that hit a target fill factor
- [ ] `__str__` / `summary()` for clean printing
- [ ] Tests: `tests/test_motor_params.py`
      - valid design passes
      - tooth-width violation raises ValueError
      - fill-factor ceiling raises ValueError
      - suggest_slot returns feasible result

---

## Phase 2 — Stator Geometry  `gen_stator.py`

- [ ] `build_stator(params, occ)` function:
      - annular sector R_si → R_so, 0 → θs
      - 6 rectangular slot bodies (b_slot × h_slot) + slot openings (b1 × h1)
      - n_hp hairpin rectangles per slot (stacked radially)
      - airgap stator sector R_sb → R_si
      - fragment stator side (non-conforming at SB)
      - return dict of classified surface tag-lists
- [ ] `classify_stator(all_surfs, params)` — area+centroid rules
- [ ] Physical groups: Stator_Iron, Airgap_Stator, S{i}_Opening,
      S{i}_Insul, S{i}_HP{k}_{phase}
- [ ] Boundary groups: Domain, SB_Stator, Stator_Right, Stator_Left
- [ ] Mesh size field (Distance from SB + Threshold)
- [ ] Standalone runner: `python gen_stator.py [--gui]`
- [ ] Tests: `tests/test_gen_stator.py`
      - correct number of surfaces per group
      - hairpin area matches expected
      - no unclassified surfaces
      - SB_Stator is a single arc

---

## Phase 3 — Rotor Geometry  `gen_rotor.py`

- [ ] `build_rotor(params, occ)` function:
      - rotor iron annular sector R_ri → R_ro
      - rectangular magnet (w_mag × h_m) centred at pole midpoint
      - air pockets (w_air × h_m) at each tangential end of magnet
      - shaft sector 0 → R_ri
      - airgap rotor sector R_ro → R_sb
      - fragment rotor side
      - return classified tag dict
- [ ] Physical groups: Rotor_Iron, Magnet, AirPocket, Shaft, Airgap_Rotor
- [ ] Boundary groups: SB_Rotor, Rotor_Right, Rotor_Left
- [ ] Standalone runner: `python gen_rotor.py [--gui]`
- [ ] Tests: `tests/test_gen_rotor.py`
      - magnet area matches expected
      - air pockets detected
      - rotor iron has correct area

---

## Phase 4 — Full Mesh Assembly  `gen_mesh.py`

- [ ] `generate_mesh(params, out_dir)`:
      - calls build_stator + build_rotor in two separate fragment passes
      - assigns all physical groups
      - mesh-size background field (fine at SB arcs)
      - gmsh.write → ElmerGrid conversion
      - return mesh metadata (node count, body/boundary names)
- [ ] Verify SB_Rotor / SB_Stator share zero nodes (non-conforming check)
- [ ] Standalone runner with --gui and --check flags
- [ ] Tests: `tests/test_gen_mesh.py`
      - mesh file created
      - mesh.names contains expected body/boundary names
      - SB_Rotor elem count == SB_Stator elem count (±5%)
      - no zero-area elements

---

## Phase 5 — Simulation Setup  `gen_sif.py`

- [ ] `gen_sif(params, mesh_meta, out_path, sweep)`:
      - sweep = {mode: "rotor_angle"|"current_angle"|"transient",
                 angles: [...], n_steps: int, ...}
      - writes Elmer .sif with correct body forces per phase
      - Mortar BC for sliding mesh
      - anti-periodic BCs
      - FourierLoss + Torque + Energy solvers
- [ ] Current density computed from params.Is / params.Carea (from hairpin dims)
- [ ] Support sinusoidal (transient) and fixed (magnetostatic) excitation modes
- [ ] Tests: `tests/test_gen_sif.py`
      - output file is valid text with required keywords
      - body count matches mesh_meta
      - current density sanity check

---

## Phase 6 — Post-processing  `postprocess.py`

- [ ] `load_scalars(results_dir)` → DataFrame (torque, losses, field energy …)
- [ ] `load_losses(results_dir)` → dict {body_id: {hyst: W, eddy: W}}
- [ ] `compute_airgap_B(results_dir, params)`:
      - read VTU files (pyvista if available, else skip)
      - sample B along arc at R_sb, FFT → B_g1 fundamental
- [ ] `torque_summary(df, params)`:
      - mean torque, ripple %, mechanical power, efficiency estimate
- [ ] `loss_summary(df_losses, params)` → total Fe + Cu losses
- [ ] `plot_torque(df, params)` → matplotlib figure
- [ ] `plot_airgap_B(B_theta, params)` → matplotlib figure
- [ ] Tests: `tests/test_postprocess.py`
      - torque mean/ripple computed correctly from synthetic data
      - SCALE factor applied correctly

---

## Phase 7 — Sweep Runner  `run_sweep.py`

- [ ] CLI: `python run_sweep.py --mode current_angle --angles 0,15,30,45`
- [ ] Calls gen_mesh → gen_sif → ElmerSolver → postprocess for each point
- [ ] Saves results to JSON + CSV
- [ ] Basic progress reporting

---

## Phase 8 — Report  `report.py` / `report.ipynb`

- [ ] Update report.ipynb to use motor_params + postprocess functions
- [ ] Machine drawing with rectangular slots + rectangular magnets
- [ ] Per-slot hairpin visualization
- [ ] MTPA operating point from current-angle sweep

---

## Order of attack

1. `motor_params.py` + tests          ← start here
2. `gen_stator.py` + tests
3. `gen_rotor.py` + tests
4. `gen_mesh.py` + tests
5. `gen_sif.py` + tests
6. `postprocess.py` + tests
7. `run_sweep.py`
8. `report.py` / update notebook
