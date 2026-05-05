"""
plot_eddy_demo.py — Magnet-eddy-current effect on torque + |J_eddy| field.

Iron (M350-50A) is already modelled as fully laminated in the
gen_sif Materials 3/4 (no `Electric Conductivity` line → Elmer
defaults to 0, breaking axial eddy current paths). Real M350-50A is
0.50 mm sheets stacked with insulation between, so this is a good
2D approximation. Magnets default to bulk N45SH (σ ≈ 625 kS/m).
High-end traction designs add axial magnet segmentation to reduce
eddy losses; you can mimic this by setting `magnet_sigma=0` (no
axial path) or scaling sigma down by 1/N² for N-piece segmentation.

This script runs the default 48s/8p design twice:
  A) magnet_sigma = MotorParams default (625 kS/m, bulk magnet)
  B) magnet_sigma = 0 (perfectly segmented or laminated magnet)
Re-uses the existing `results/` directory if it was solved with the
same sigma; otherwise re-runs Elmer.

Outputs:
  results/png/eddy_torque_compare.png   — torque waveform A vs B
  results/png/eddy_current_field.png    — |J_eddy| colormap in magnet

Usage:
    python3 plot_eddy_demo.py
"""
from __future__ import annotations

import dataclasses
import shutil
import subprocess
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import meshio
import numpy as np

from gen_sif import gen_sif
from motor_params import MotorParams
from postprocess import summarise

ROOT      = Path(__file__).resolve().parent
MESH_DIR  = ROOT / "mesh"
RES_BULK  = ROOT / "results"                # σ=625 kS/m (existing solve)
RES_LAM   = ROOT / "results_eddy_lam"       # σ=0 (this script's solve)
SIF_BULK  = ROOT / "case_nb.sif"
SIF_LAM   = ROOT / "case_eddy_lam.sif"
PNG_DIR   = ROOT / "results/png"


def _run_elmer_if_needed(p: MotorParams, sif_path: Path, res_dir: Path,
                         label: str) -> None:
    res_dir.mkdir(parents=True, exist_ok=True)
    needed = not (res_dir / "scalars.dat").exists() or \
             not list(res_dir.glob("step-*_t*.vtu"))
    if not needed:
        print(f"  [{label}] re-using cached {res_dir}/")
        return

    sif_text = gen_sif(p, mesh_dir=str(MESH_DIR.resolve()),
                       results_dir=str(res_dir.resolve()),
                       suffix="2d")
    sif_path.write_text(sif_text)
    solver = shutil.which("ElmerSolver")
    if solver is None:
        raise RuntimeError("ElmerSolver not on PATH.")
    print(f"  [{label}] running ElmerSolver ({sif_path.name}) ...")
    t0 = time.perf_counter()
    proc = subprocess.run([solver, str(sif_path.resolve())],
                          capture_output=True, text=True)
    elapsed = time.perf_counter() - t0
    if proc.returncode != 0:
        raise RuntimeError(f"ElmerSolver failed for {label}\n"
                           f"stderr (last 500): {proc.stderr[-500:]}")
    print(f"  [{label}] done in {elapsed:.0f} s")


def _torque_series(res_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.loadtxt(res_dir / "scalars.dat")
    p = MotorParams()
    SCALE = p.SCALE
    nper = 120
    fme  = p.rpm / 60
    dt   = 1 / (fme * nper)
    n    = data.shape[0]
    t_s  = np.arange(1, n + 1) * dt
    return t_s * 1000, data[:, 5] * SCALE, data[:, 8] * SCALE


def _plot_torque_compare(p: MotorParams) -> None:
    t_b, ag_b, g1_b = _torque_series(RES_BULK)
    t_l, ag_l, g1_l = _torque_series(RES_LAM)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    ax = axes[0]
    ax.plot(t_b, ag_b, lw=1.0, color="C3",
            label=f"σ_magnet = {p.magnet_sigma/1e3:.0f} kS/m  "
                  f"(mean={np.mean(ag_b[t_b > 1000/(p.rpm/60)]):+.1f} N·m)")
    ax.plot(t_l, ag_l, lw=1.0, color="C0",
            label=f"σ_magnet = 0 kS/m (laminated/segmented)  "
                  f"(mean={np.mean(ag_l[t_l > 1000/(p.rpm/60)]):+.1f} N·m)")
    ax.set_ylabel("Air-gap torque [N·m]")
    ax.set_title("Torque waveform — bulk vs eddy-free magnet")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="upper right")
    ax.axhline(0, color="k", lw=0.4)

    ax = axes[1]
    ax.plot(t_b, ag_b - ag_l, lw=1.0, color="C2")
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Δ torque  (bulk − eddy-free)  [N·m]")
    ax.set_title("Eddy-current contribution to torque")
    ax.grid(alpha=0.3)
    ax.axhline(0, color="k", lw=0.4)

    plt.tight_layout()
    out = PNG_DIR / "eddy_torque_compare.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"Saved: {out}")


def _plot_eddy_field(p: MotorParams, vtu_path: Path) -> None:
    m = meshio.read(str(vtu_path))
    pts_mm = m.points[:, :2] * 1e3
    tris_raw = next(b.data for b in m.cells if "triangle" in b.type)
    valid = np.all(tris_raw >= 0, axis=1)
    tris = tris_raw[valid]

    if "current density e" not in m.point_data:
        raise KeyError("'current density e' not in VTU — "
                       "needs Calculate Current Density = True in CalcFields")
    # In 2D MgDyn, J is out-of-plane (z-component). Use the signed value
    # but plot the magnitude so the colormap reads positive throughout.
    J = np.asarray(m.point_data["current density e"])
    Jz = J[:, 2]
    Jmag = np.abs(Jz)

    # ── Pick out the magnet region for zoom + masking
    R_mi_mm, R_mo_mm = p.R_mi, p.R_mo
    rs = np.hypot(pts_mm[:, 0], pts_mm[:, 1])
    mag_mask = (rs > R_mi_mm - 0.5) & (rs < R_mo_mm + 0.5)
    mag_J_max = float(np.max(Jmag[mag_mask])) if mag_mask.any() else float(np.max(Jmag))
    print(f"  |J| stats — global max {Jmag.max()/1e6:.2f} MA/m², "
          f"magnet-region max {mag_J_max/1e6:.2f} MA/m², "
          f"global mean {Jmag.mean()/1e6:.4f} MA/m²")
    if mag_J_max < 1e-6:
        # Fallback: cover the whole plotting range so contour levels are valid
        mag_J_max = max(float(Jmag.max()), 1e-6)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
    triang = mtri.Triangulation(pts_mm[:, 0], pts_mm[:, 1], tris)

    # Left: full sector, log-scaled
    ax = axes[0]
    levels = np.linspace(0, mag_J_max, 96)
    tcf = ax.tricontourf(triang, np.clip(Jmag, 0, mag_J_max),
                         levels=levels, cmap="viridis", extend="max")
    fig.colorbar(tcf, ax=ax, label="|J_eddy|  [A/m²]")
    ax.set_aspect("equal")
    ax.set_title(f"Sector overview — bulk magnet σ = {p.magnet_sigma/1e3:.0f} kS/m")
    ax.set_xlabel("x  [mm]"); ax.set_ylabel("y  [mm]")

    # Right: magnet zoom
    ax = axes[1]
    tcf = ax.tricontourf(triang, np.clip(Jmag, 0, mag_J_max),
                         levels=levels, cmap="viridis", extend="max")
    fig.colorbar(tcf, ax=ax, label="|J_eddy|  [A/m²]")
    ax.set_aspect("equal")
    # Bound around magnet
    th_c = float(p.θs / 2)
    cx = (R_mi_mm + R_mo_mm) / 2 * np.cos(th_c)
    cy = (R_mi_mm + R_mo_mm) / 2 * np.sin(th_c)
    half = max(p.w_mag / 2 + 4, p.h_m * 1.5)
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_title(f"Magnet zoom (peak |J| = {mag_J_max/1e6:.2f} MA/m²)")
    ax.set_xlabel("x  [mm]"); ax.set_ylabel("y  [mm]")

    fig.suptitle(f"Eddy current density |J_eddy|  —  {Path(vtu_path).name}",
                 fontsize=12)
    plt.tight_layout()
    out = PNG_DIR / "eddy_current_field.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"Saved: {out}")


def main() -> None:
    PNG_DIR.mkdir(parents=True, exist_ok=True)

    # A) Bulk magnet — assume cached in results/ (default magnet_sigma)
    p_bulk = MotorParams()
    print(f"[bulk] magnet_sigma = {p_bulk.magnet_sigma/1e3:.0f} kS/m")
    _run_elmer_if_needed(p_bulk, SIF_BULK, RES_BULK, "bulk")

    # B) Eddy-free magnet
    p_lam = dataclasses.replace(p_bulk, magnet_sigma=0.0)
    print(f"[lam ] magnet_sigma = 0 (idealised lamination/segmentation)")
    _run_elmer_if_needed(p_lam, SIF_LAM, RES_LAM, "lam")

    # Plots
    print("\n--- Torque comparison ---")
    _plot_torque_compare(p_bulk)

    print("\n--- |J_eddy| field plot (bulk magnet, last steady-state VTU) ---")
    vtus = sorted(RES_BULK.glob("step-2d_t*.vtu"))
    if not vtus:
        raise RuntimeError(f"No VTUs in {RES_BULK}")
    _plot_eddy_field(p_bulk, vtus[-1])


if __name__ == "__main__":
    main()
