"""
plot_flux_multi_pole.py — Multi-Qp magnetostatic flux density panels.

For each pole count in ``ALL_QP``, builds the mesh, runs a short Elmer
solve at no-load (Is=0, magnet_sigma=0) and plots ``|B|`` over the
sector geometry. Mirrors the layout of ``multi_pole_sectors.png``.

Usage:
    python3 plot_flux_multi_pole.py

Outputs:
    results/png/multi_pole_flux.png

Each Qp run takes ~30-60 s (mesh build + 3-step transient solve), so
the full sweep is ~5-10 min.
"""
from __future__ import annotations

import dataclasses
import math
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import meshio
import numpy as np

from motor_params import motor_params_for_poles
from gen_mesh import build_mesh
from gen_sif import gen_sif

ALL_QP    = [4, 6, 8, 10, 12, 14, 16, 18, 20]
# Qp=2 is excluded outright (180° sector breaks the periodic BC).
# Other Qp may also fail Elmer's RadialInterfaceMeshes/ConformingNodePerm
# check if the airgap mesh is non-conforming — those are skipped at
# runtime and reported in the final log.
WORKDIR   = Path("examples") / "flux_multi_pole"
PNG_OUT   = Path("results/png/multi_pole_flux.png")
NPER      = 4    # very few timesteps — we only want the first VTU
NCYCLE    = 1


def _solve_one(Qp: int, base: Path) -> Path:
    """Build mesh + solve magnetostatic-style for a given Qp.

    Returns the path to the first VTU file produced.
    """
    p = motor_params_for_poles(Qp)
    p = dataclasses.replace(p, Is=0.0, magnet_sigma=0.0)

    mesh_dir = base / f"qp{Qp:02d}_mesh"
    res_dir  = base / f"qp{Qp:02d}_results"
    res_dir.mkdir(parents=True, exist_ok=True)

    # 1. mesh
    mesh_dir.parent.mkdir(parents=True, exist_ok=True)
    if not mesh_dir.exists():
        gmsh_msh = mesh_dir.parent / f"qp{Qp:02d}.msh"
        build_mesh(p, mesh_out=gmsh_msh)
        subprocess.run(
            ["ElmerGrid", "14", "2", str(gmsh_msh),
             "-autoclean", "-out", str(mesh_dir)],
            check=True, capture_output=True,
        )

    # 2. SIF: short transient as a magnetostatic snapshot.
    # Use absolute paths so ElmerSolver's cwd doesn't matter.
    suffix = f"qp{Qp:02d}"
    sif_text = gen_sif(
        p,
        mesh_dir=str(mesh_dir.resolve()),
        results_dir=str(res_dir.resolve()),
        suffix=suffix,
        nper=NPER,
        ncycle=NCYCLE,
    )
    sif_path = base / f"qp{Qp:02d}.sif"
    sif_path.write_text(sif_text)

    # 3. solve
    solver = shutil.which("ElmerSolver")
    if solver is None:
        raise RuntimeError("ElmerSolver not on PATH — install Elmer FEM.")
    proc = subprocess.run(
        [solver, str(sif_path.resolve())],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"ElmerSolver failed for Qp={Qp} (rc={proc.returncode})\n"
            f"stdout (last 1.5k):\n{proc.stdout[-1500:]}\n"
            f"stderr (last 500):\n{proc.stderr[-500:]}"
        )

    # 4. find first VTU written
    vtus = sorted(res_dir.glob(f"step-{suffix}_t*.vtu"))
    if not vtus:
        raise RuntimeError(f"No VTU written for Qp={Qp} in {res_dir}")
    return vtus[0]


def _read_b_magnitude(vtu: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (xy_mm, triangles, |B|) for the given VTU."""
    m = meshio.read(str(vtu))
    pts = m.points[:, :2] * 1e3   # m -> mm
    tris = None
    for block in m.cells:
        if "triangle" in block.type:
            tris = block.data
            break
    valid = np.all(tris >= 0, axis=1)
    tris = tris[valid]

    field = None
    for nm in ("magnetic flux density e", "magnetic flux density"):
        if nm in m.point_data:
            field = np.asarray(m.point_data[nm])
            break
    if field is None:
        raise KeyError(f"No magnetic flux density field in {vtu.name}")
    B_mag = np.linalg.norm(field[:, :2], axis=1)
    return pts, tris, B_mag


def _draw_sector(ax, pts, tris, B_mag, vmax: float, p, label: str) -> None:
    triang = mtri.Triangulation(pts[:, 0], pts[:, 1], tris)
    levels = np.linspace(0.0, vmax, 96)
    tcf = ax.tricontourf(triang, B_mag, levels=levels, cmap="plasma",
                         extend="max")
    # Sector outline
    θs = p.θs
    R_so = p.R_so
    R_ri = p.R_ri
    n = 200
    t_arc = np.linspace(0, θs, n)
    ax.plot(R_so * np.cos(t_arc), R_so * np.sin(t_arc),
            "k-", lw=0.5, alpha=0.6)
    ax.plot([0, R_so], [0, 0], "k--", lw=0.4, alpha=0.4)
    ax.plot([0, R_so * math.cos(θs)],
            [0, R_so * math.sin(θs)],
            "k--", lw=0.4, alpha=0.4)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(label, fontsize=9)
    return tcf


def main() -> None:
    base = WORKDIR
    base.mkdir(parents=True, exist_ok=True)

    print(f"Multi-Qp flux density: {len(ALL_QP)} solves into {base}/")
    panels = []
    skipped = []
    for Qp in ALL_QP:
        print(f"  Qp={Qp:2d} ...", end=" ", flush=True)
        try:
            vtu = _solve_one(Qp, base)
        except RuntimeError as err:
            short = str(err).split("\n")[0]
            print(f"SKIP ({short})")
            skipped.append(Qp)
            continue
        pts, tris, B_mag = _read_b_magnitude(vtu)
        p = motor_params_for_poles(Qp)
        panels.append((Qp, p, pts, tris, B_mag))
        print(f"|B| max = {B_mag.max():.2f} T")
    if skipped:
        print(f"Skipped Qp values (mesh / mortar BC failure): {skipped}")

    vmax = max(B.max() for _, _, _, _, B in panels)
    print(f"Global |B| max for shared colormap: {vmax:.2f} T")

    nrows, ncols = 2, 5
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 7.5))
    axes = axes.flatten()
    last_tcf = None
    for i, (Qp, p, pts, tris, B_mag) in enumerate(panels):
        label = (f"Qp={Qp}  Qs={p.Qs}\n"
                 f"θs={math.degrees(p.θs):.0f}°  "
                 f"|B|max={B_mag.max():.2f}T")
        last_tcf = _draw_sector(axes[i], pts, tris, B_mag, vmax, p, label)

    # Hide unused axes
    for j in range(len(panels), nrows * ncols):
        axes[j].axis("off")

    fig.suptitle("FEM flux density |B| — Qp = 2 … 20  (no-load, Is=0)",
                 fontsize=12, y=0.99)
    cbar = fig.colorbar(last_tcf, ax=axes.tolist(),
                        location="bottom", shrink=0.6, pad=0.04,
                        label="|B|  [T]")
    cbar.ax.tick_params(labelsize=8)

    PNG_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PNG_OUT, dpi=140, bbox_inches="tight")
    print(f"Saved: {PNG_OUT}")


if __name__ == "__main__":
    main()
