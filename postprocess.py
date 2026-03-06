"""
postprocess.py — Extract key metrics from Elmer FEM results.

Reads the output files produced by gen_sif.py simulations:
  - scalars.dat         : per-timestep torque and global scalars
  - scalars.dat.names   : column labels for scalars.dat
  - loss-{suffix}.dat   : iron losses per body (FourierLoss solver)

All raw values from Elmer are **per sector, per unit depth (W/m or N·m/m)**
for 2D models.  Multiply by SCALE = Qp * L_active_m to get full-machine SI
values.

Usage:
    python3 postprocess.py [--results results] [--suffix 2d] [--skip 1]
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np

from motor_params import MotorParams, DEFAULT_PARAMS
from gen_sif import _PHASE_MAP   # for default body numbering

π = math.pi


# ─────────────────────────────────────────────────────────────────────────────
#  Body-ID helpers
# ─────────────────────────────────────────────────────────────────────────────

def stator_body_id(params: MotorParams, phase_map: Optional[list[str]] = None) -> int:
    """Body number assigned to Stator_Iron in the SIF (= body_id in loss file)."""
    return 1  # always first in gen_sif.py body numbering


def rotor_body_id(params: MotorParams, phase_map: Optional[list[str]] = None) -> int:
    """Body number assigned to Rotor_Iron in the SIF (= body_id in loss file)."""
    if phase_map is None:
        phase_map = _PHASE_MAP[:]
    p = params
    # Stator_Iron(1) + Airgap_Stator(1) + ns*(Opening + Insul + n_hp HP) +
    # Airgap_Rotor(1) + Rotor_Iron(1)
    return 2 + p.ns * (2 + p.n_hp) + 2


# ─────────────────────────────────────────────────────────────────────────────
#  Parse scalars.dat
# ─────────────────────────────────────────────────────────────────────────────

def _parse_names(names_file: Path) -> dict[str, int]:
    """
    Return {column_label: 0-based column index} from scalars.dat.names.
    Lines look like:  '   3: res: fourier loss total'
    """
    col_map: dict[str, int] = {}
    with open(names_file) as fh:
        for line in fh:
            line = line.strip()
            if line and line[0].isdigit() and ':' in line:
                idx_str, _, label = line.partition(':')
                col_map[label.strip().lower()] = int(idx_str.strip()) - 1
    return col_map


def read_scalars(
    scalars_file: Path,
    params: MotorParams,
    skip_cycles: int = 1,
) -> dict:
    """
    Load scalars.dat and return a dict of per-timestep and summary quantities.

    Parameters
    ----------
    scalars_file : path to scalars.dat
    params       : MotorParams (for SCALE, rpm, nper)
    skip_cycles  : number of mechanical cycles to skip for steady-state stats
                   (default 1: use only cycle 2 onwards for means)

    Returns dict with:
      t_s            : time array [s]
      torque_Nm      : full-machine group-1 torque [N·m], each timestep
      airgap_Nm      : full-machine air-gap torque [N·m], each timestep
      T_mean_Nm      : mean torque over steady-state window [N·m]
      T_max_Nm, T_min_Nm, T_ripple_pct
      P_mech_kW      : mechanical power from mean torque [kW]
      n_steps        : total number of timesteps
      SCALE          : scaling factor used
      col_map        : column label → index mapping
    """
    scalars_file = Path(scalars_file)
    p = params

    # Parse column map from companion .names file (if present)
    names_file = scalars_file.with_suffix(scalars_file.suffix + ".names")
    if names_file.exists():
        col_map = _parse_names(names_file)
    else:
        # Fallback: use standard column order from case.sif SaveScalars
        col_map = {
            "res: eddy current power":       0,
            "res: electromagnetic field energy": 1,
            "res: fourier loss total":       2,
            "res: fourier loss 1":           3,
            "res: fourier loss 2":           4,
            "res: air gap torque":           5,
            "res: inertial volume":          6,
            "res: inertial moment":          7,
            "res: group 1 torque":           8,
        }

    raw = np.loadtxt(scalars_file)
    if raw.ndim == 1:
        raw = raw[np.newaxis, :]
    n_steps = raw.shape[0]

    # Time vector (timestep index × dt, starting from dt)
    fme = p.rpm / 60.0
    dt  = 1.0 / (fme * 120)   # default nper=120
    t_s = np.arange(1, n_steps + 1) * dt

    SCALE = p.SCALE

    # Torque columns
    def _col(name: str, fallback: int) -> np.ndarray:
        idx = col_map.get(name, fallback)
        if idx < raw.shape[1]:
            return raw[:, idx]
        return np.zeros(n_steps)

    group1_raw   = _col("res: group 1 torque",  8)
    airgap_raw   = _col("res: air gap torque",   5)

    # Prefer group-1 (nodal force) torque; fall back to air-gap torque when
    # group-1 is all zeros (CalcFields nodal forces not converged/configured).
    if np.all(group1_raw == 0.0) and not np.all(airgap_raw == 0.0):
        torque_raw = airgap_raw
        torque_source = "air gap"
    else:
        torque_raw = group1_raw
        torque_source = "group 1"

    torque_full  = torque_raw  * SCALE
    airgap_full  = airgap_raw  * SCALE

    # Steady-state mask: skip first `skip_cycles` mechanical cycles
    T_mech = 1.0 / fme
    ss_mask = t_s > skip_cycles * T_mech
    if not np.any(ss_mask):
        ss_mask = np.ones(n_steps, dtype=bool)

    T_ss     = torque_full[ss_mask]
    T_mean   = float(np.mean(T_ss))
    T_max    = float(np.max(T_ss))
    T_min    = float(np.min(T_ss))
    T_ripple = (T_max - T_min) / abs(T_mean) * 100.0 if T_mean != 0 else float("nan")
    P_mech   = T_mean * (2 * π * p.rpm / 60) / 1000.0  # kW

    return {
        "t_s":            t_s,
        "torque_Nm":      torque_full,
        "airgap_Nm":      airgap_full,
        "torque_source":  torque_source,
        "T_mean_Nm":      T_mean,
        "T_max_Nm":       T_max,
        "T_min_Nm":       T_min,
        "T_ripple_pct":   T_ripple,
        "P_mech_kW":      P_mech,
        "n_steps":        n_steps,
        "SCALE":          SCALE,
        "col_map":        col_map,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Parse loss-{suffix}.dat
# ─────────────────────────────────────────────────────────────────────────────

def read_losses(
    loss_file: Path,
    params: MotorParams,
    phase_map: Optional[list[str]] = None,
) -> dict:
    """
    Load a FourierLoss output file and return full-machine iron losses.

    The file has one row per lamination body:
        body_id   loss_hyst[W/m/sector]   loss_eddy[W/m/sector]

    Parameters
    ----------
    loss_file  : path to loss-{suffix}.dat
    params     : MotorParams (for SCALE and body numbering)
    phase_map  : phase map used in gen_sif (to compute rotor body id)

    Returns dict with:
      stator_hyst_W, stator_eddy_W, stator_total_W
      rotor_hyst_W,  rotor_eddy_W,  rotor_total_W
      total_W
      SCALE
      body_ids   : {body_id: (hyst_W/m, eddy_W/m)} raw values
    """
    loss_file = Path(loss_file)
    raw = np.loadtxt(loss_file, comments="!")
    if raw.ndim == 1:
        raw = raw[np.newaxis, :]

    SCALE = params.SCALE
    body_ids: dict[int, tuple[float, float]] = {
        int(r[0]): (float(r[1]), float(r[2])) for r in raw
    }

    sid = stator_body_id(params, phase_map)
    rid = rotor_body_id(params, phase_map)

    def _loss(body_id: int) -> tuple[float, float, float]:
        if body_id in body_ids:
            h, e = body_ids[body_id]
            return h * SCALE, e * SCALE, (h + e) * SCALE
        return 0.0, 0.0, 0.0

    s_h, s_e, s_tot = _loss(sid)
    r_h, r_e, r_tot = _loss(rid)

    return {
        "stator_hyst_W":  s_h,
        "stator_eddy_W":  s_e,
        "stator_total_W": s_tot,
        "rotor_hyst_W":   r_h,
        "rotor_eddy_W":   r_e,
        "rotor_total_W":  r_tot,
        "total_W":        s_tot + r_tot,
        "SCALE":          SCALE,
        "body_ids":       body_ids,
        "stator_body_id": sid,
        "rotor_body_id":  rid,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Combined summary
# ─────────────────────────────────────────────────────────────────────────────

def summarise(
    results_dir: Path,
    params: MotorParams,
    suffix: str = "2d",
    skip_cycles: int = 1,
    phase_map: Optional[list[str]] = None,
) -> dict:
    """
    Load all result files from *results_dir* and return a unified metrics dict.

    Keys: everything from read_scalars() and read_losses() plus:
      - efficiency_pct (indicative, iron loss only)
      - results_dir, suffix
    """
    results_dir = Path(results_dir)
    scalars_file = results_dir / "scalars.dat"
    loss_file    = results_dir / f"loss-{suffix}.dat"

    out: dict = {"results_dir": str(results_dir), "suffix": suffix}

    if scalars_file.exists():
        sc = read_scalars(scalars_file, params, skip_cycles=skip_cycles)
        out.update(sc)
    else:
        print(f"WARNING: {scalars_file} not found")
        out.update({k: None for k in [
            "t_s", "torque_Nm", "airgap_Nm", "T_mean_Nm", "T_max_Nm",
            "T_min_Nm", "T_ripple_pct", "P_mech_kW", "n_steps", "SCALE",
        ]})

    if loss_file.exists():
        lc = read_losses(loss_file, params, phase_map=phase_map)
        out.update(lc)
    else:
        print(f"WARNING: {loss_file} not found")
        out.update({k: None for k in [
            "stator_hyst_W", "stator_eddy_W", "stator_total_W",
            "rotor_hyst_W",  "rotor_eddy_W",  "rotor_total_W",
            "total_W",
        ]})

    # Indicative efficiency
    P_mech = out.get("P_mech_kW")
    P_iron = out.get("total_W")
    if P_mech and P_iron and P_mech > 0 and P_iron is not None:
        out["efficiency_pct"] = P_mech / (P_mech + P_iron / 1000.0) * 100.0
    else:
        out["efficiency_pct"] = None

    return out


def print_summary(metrics: dict) -> None:
    """Print a formatted engineering summary from the metrics dict."""
    SEP = "=" * 68
    sep = "-" * 68

    print(SEP)
    print("  TractionMachinePM2D — FEM Results Summary")
    print(f"  Results: {metrics.get('results_dir', '?')}  "
          f"suffix={metrics.get('suffix', '?')}")
    print(SEP)

    SCALE = metrics.get("SCALE") or 0.0
    print(f"\n  Scaling factor   SCALE = {SCALE:.4f} m  "
          f"(Qp x L_active = {SCALE:.4f} m)")

    # Torque
    T = metrics.get("T_mean_Nm")
    if T is not None:
        src = metrics.get("torque_source", "group 1")
        print(f"\n  TORQUE  (steady-state, full machine, source: {src})")
        print(sep)
        print(f"  {'Mean torque':<30} {metrics['T_mean_Nm']:>10.2f} N·m")
        print(f"  {'Max torque':<30} {metrics['T_max_Nm']:>10.2f} N·m")
        print(f"  {'Min torque':<30} {metrics['T_min_Nm']:>10.2f} N·m")
        print(f"  {'Torque ripple':<30} {metrics['T_ripple_pct']:>10.2f} %")
        print(f"  {'Mechanical power':<30} {metrics['P_mech_kW']:>10.3f} kW")

    # Iron losses
    s_tot = metrics.get("stator_total_W")
    r_tot = metrics.get("rotor_total_W")
    if s_tot is not None:
        s_h = metrics["stator_hyst_W"]
        s_e = metrics["stator_eddy_W"]
        r_h = metrics["rotor_hyst_W"]
        r_e = metrics["rotor_eddy_W"]
        tot = metrics["total_W"]
        print(f"\n  IRON LOSSES  (full machine, FourierLoss)")
        print(sep)
        print(f"  {'Component':<25} {'Hysteresis':>12} {'Eddy':>10} {'Total':>10}")
        print("  " + "-" * 60)
        print(f"  {'Stator iron':<25} {s_h:>12.2f} {s_e:>10.2f} {s_tot:>10.2f}  W")
        print(f"  {'Rotor iron':<25} {r_h:>12.2f} {r_e:>10.2f} {r_tot:>10.2f}  W")
        print("  " + "-" * 60)
        print(f"  {'TOTAL':<25} {s_h+r_h:>12.2f} {s_e+r_e:>10.2f} {tot:>10.2f}  W")

    # Efficiency
    eta = metrics.get("efficiency_pct")
    if eta is not None:
        print(f"\n  Indicative efficiency (iron only):  η ≥ {eta:.2f}%")

    print(SEP)


# ─────────────────────────────────────────────────────────────────────────────
#  Flux density plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_flux_density(
    vtu_file: str | Path,
    ax=None,
    figsize: tuple = (7, 7),
    cmap: str = "plasma",
    clim: tuple | None = None,
    title: str | None = None,
):
    """
    Plot magnetic flux density magnitude from an Elmer VTU result file.

    Reads the 'magnetic flux density e' (or 'magnetic field strength e')
    point data written by Elmer's CalcFields solver.  Requires *meshio*:

        pip install meshio

    Parameters
    ----------
    vtu_file : path to a *.vtu file produced by the VtuOutput solver
    ax       : matplotlib Axes to draw on (created if None)
    figsize  : figure size if ax is None
    cmap     : matplotlib colormap name
    clim     : (vmin, vmax) colour limits in Tesla; auto if None
    title    : axes title; auto-generated if None

    Returns
    -------
    (fig, ax) tuple
    """
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    import numpy as np

    try:
        import meshio
    except ImportError as err:
        raise ImportError("Install meshio:  pip install meshio") from err

    vtu_file = Path(vtu_file)
    if not vtu_file.exists():
        raise FileNotFoundError(f"VTU file not found: {vtu_file}")

    m = meshio.read(str(vtu_file))

    # ── node coordinates (Elmer outputs in metres) ─────────────────────────
    pts_m = m.points[:, :2]           # (N, 2) in metres
    pts   = pts_m * 1e3               # convert to mm for display

    # ── triangle connectivity ──────────────────────────────────────────────
    tris = None
    for block in m.cells:
        if "triangle" in block.type:
            tris = block.data          # (T, 3) node indices
            break
    if tris is None:
        raise ValueError(f"No triangle cells found in {vtu_file.name}")

    # Drop degenerate triangles (Elmer Discontinuous Bodies padding uses -1)
    valid = np.all(tris >= 0, axis=1)
    tris = tris[valid]

    # ── choose field: prefer B, fall back to H ─────────────────────────────
    prefer = ["magnetic flux density e", "magnetic flux density",
              "magnetic field strength e", "magnetic field strength"]
    field_name = None
    for name in prefer:
        if name in m.point_data:
            field_name = name
            break
    if field_name is None:
        # last resort: any field with "magnetic" in the name
        for key in m.point_data:
            if "magnetic" in key.lower():
                field_name = key
                break
    if field_name is None:
        raise KeyError(
            f"No magnetic field found in {vtu_file.name}. "
            "Available point data: " + str(list(m.point_data.keys()))
        )

    vec   = np.asarray(m.point_data[field_name])   # (N, 3)
    B_mag = np.linalg.norm(vec[:, :2], axis=1)      # in-plane magnitude

    is_B  = "flux" in field_name.lower()
    unit  = "T" if is_B else "A/m"
    label = f"|B|  [{unit}]" if is_B else f"|H|  [{unit}]"

    # ── matplotlib triangulation using actual mesh connectivity ────────────
    triang = mtri.Triangulation(pts[:, 0], pts[:, 1], tris)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    vmin = clim[0] if clim else float(B_mag.min())
    vmax = clim[1] if clim else float(B_mag.max())
    levels = np.linspace(vmin, vmax, 128)

    tcf = ax.tricontourf(triang, B_mag, levels=levels, cmap=cmap)
    fig.colorbar(tcf, ax=ax, label=label)

    ax.set_aspect("equal")
    ax.set_xlabel("x  [mm]")
    ax.set_ylabel("y  [mm]")
    ax.set_title(title or f"{field_name} — {vtu_file.name}")

    return fig, ax


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Extract metrics from Elmer FEM results"
    )
    ap.add_argument("--results", default="results",  help="Results directory")
    ap.add_argument("--suffix",  default="2d",       help="Loss file suffix")
    ap.add_argument("--skip",    type=int, default=1,
                    help="Mechanical cycles to skip for steady-state stats")
    args = ap.parse_args()

    p = MotorParams()
    m = summarise(Path(args.results), p, suffix=args.suffix, skip_cycles=args.skip)
    print_summary(m)
