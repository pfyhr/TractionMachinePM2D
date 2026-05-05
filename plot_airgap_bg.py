"""
plot_airgap_bg.py — Airgap radial flux density: FEM vs analytical.

Reads a no-load Elmer VTU (Is=0), samples A_z on a circle just inside
the stator bore, takes B_r = (1/r)·dA_z/dθ, anti-periodically continues
to 360° using the Qp-fold symmetry, and overlays:

  * the textbook rectangular Bg = B_r/(1+μ_r·g/h_m) wave,
  * the analytical fundamental (4/π)·Bg·sin(α_m·π/2) cos(PP·θ),
  * the FEM waveform with its slot-tooth ripple intact.

The point: for inset PM rotors with an iron bridge above the magnet,
the simple reluctance model overestimates Bg fundamental by ~2-3×
because it ignores bridge leakage.  This explains why the FEM peak
back-EMF is much smaller than the textbook estimate.

Usage:
    python3 plot_airgap_bg.py                 # default Qp=8 from results_bemf
    python3 plot_airgap_bg.py --qp 12         # multi-Qp (uses examples/flux_multi_pole/)
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import meshio
import numpy as np
from scipy.interpolate import griddata

from motor_params import MotorParams, motor_params_for_poles


def airgap_bg_fft(vtu_path: Path, p: MotorParams, n_samp: int = 720
                  ) -> tuple[np.ndarray, np.ndarray, dict]:
    """Sample airgap A_z, derive B_r(θ) over the full 360°, return + harmonics."""
    m = meshio.read(str(vtu_path))
    pts = m.points[:, :2]
    A = np.asarray(m.point_data["a"]).ravel()

    R_air = (p.R_si - 0.1) * 1e-3
    theta = np.linspace(0.001, p.θs - 0.001, n_samp)
    xr = R_air * np.cos(theta)
    yr = R_air * np.sin(theta)
    Az = griddata(pts, A, np.column_stack([xr, yr]),
                  method="linear", fill_value=0.0)

    dAz = np.gradient(Az, theta)
    Br_sec = dAz / R_air

    Br_full = np.concatenate([(((-1) ** k) * Br_sec) for k in range(p.Qp)])
    th_full = np.linspace(0, 2 * math.pi, p.Qp * n_samp, endpoint=False)

    F = np.fft.rfft(Br_full)
    mag = np.abs(F) / len(Br_full) * 2
    summary = {
        "R_sample_mm": R_air * 1e3,
        "Br_peak":     float(np.abs(Br_full).max()),
        "Bg1_FEM":     float(mag[p.PP]),
        "Bg3_FEM":     float(mag[3 * p.PP]),
        "Bg5_FEM":     float(mag[5 * p.PP]),
        "phase1":      float(np.angle(F[p.PP])),
    }
    return th_full, Br_full, summary


def make_panel(ax, p: MotorParams, th_full, Br_full, summary, title=None) -> None:
    Bg_anal      = p.B_r / (1 + p.mu_r * p.g / p.h_m)
    Bg1_anal     = (4 / math.pi) * Bg_anal * math.sin(p.mag_frac * math.pi / 2)

    ax.plot(np.degrees(th_full), Br_full, lw=1.3, color="C0",
            label=f"FEM (|Br|max={summary['Br_peak']:.2f} T,  "
                  f"Bg1={summary['Bg1_FEM']:.2f} T)")

    # Rectangular textbook wave at Bg_anal
    B_rect = np.zeros_like(th_full)
    theta_s_deg = math.degrees(p.θs)
    half_mag    = p.mag_frac * theta_s_deg / 2
    for k in range(p.Qp):
        centre = (k + 0.5) * theta_s_deg
        sign   = +1 if k % 2 == 0 else -1
        mask   = ((np.degrees(th_full) >= centre - half_mag) &
                  (np.degrees(th_full) <  centre + half_mag))
        B_rect[mask] = sign * Bg_anal
    ax.plot(np.degrees(th_full), B_rect, lw=0.9, ls="--", color="C3",
            alpha=0.55, label=f"Analytical rect ±{Bg_anal:.2f} T")

    # Analytical fundamental, phase-aligned to FEM
    B_fund = Bg1_anal * np.cos(p.PP * th_full + summary["phase1"])
    ax.plot(np.degrees(th_full), B_fund, lw=0.9, ls=":", color="C2",
            alpha=0.75, label=f"Analytical fund {Bg1_anal:.2f} T")

    pct = summary["Bg1_FEM"] / Bg1_anal * 100
    ax.set_title(title or f"FEM Bg1 = {pct:.0f}% of analytical")
    ax.set_xlabel("Mechanical angle [°]")
    ax.set_ylabel("B_radial [T]")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 360)
    ax.axhline(0, color="k", lw=0.4)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vtu", default="results_bemf/step-bemf_t0001.vtu",
                    help="VTU file to sample (default: results_bemf BEMF t=1)")
    ap.add_argument("--qp", type=int, default=None,
                    help="If set, use motor_params_for_poles(qp). Otherwise default 48s/8p.")
    ap.add_argument("--out", default="results/png/airgap_Bg_compare.png")
    ap.add_argument("--multi", action="store_true",
                    help="Multi-Qp panel using cached multi-pole VTUs in "
                         "examples/flux_multi_pole/.")
    ap.add_argument("--multi-out", default="results/png/multi_pole_airgap_Bg.png")
    args = ap.parse_args()

    if args.multi:
        return _main_multi(args)

    p = motor_params_for_poles(args.qp) if args.qp else MotorParams()
    vtu = Path(args.vtu)
    if not vtu.exists():
        raise FileNotFoundError(f"VTU not found: {vtu}")

    th_full, Br_full, summary = airgap_bg_fft(vtu, p)
    print(f"Sample radius : {summary['R_sample_mm']:.2f} mm")
    print(f"|Br| max      : {summary['Br_peak']:.3f} T")
    print(f"FEM Bg1       : {summary['Bg1_FEM']:.3f} T")
    print(f"FEM Bg3       : {summary['Bg3_FEM']:.3f} T")
    Bg_anal  = p.B_r / (1 + p.mu_r * p.g / p.h_m)
    Bg1_anal = (4 / math.pi) * Bg_anal * math.sin(p.mag_frac * math.pi / 2)
    print(f"Analytical Bg : {Bg_anal:.3f} T  (Bg1: {Bg1_anal:.3f} T)")
    print(f"Ratio FEM/anal: {summary['Bg1_FEM'] / Bg1_anal * 100:.1f}%")

    fig, ax = plt.subplots(figsize=(12, 4.5))
    title = (f"No-load airgap flux density — {p.Qs}s/{p.Qp}p  "
             f"(R={summary['R_sample_mm']:.1f} mm, FEM/anal = "
             f"{summary['Bg1_FEM']/Bg1_anal*100:.0f}%)")
    make_panel(ax, p, th_full, Br_full, summary, title)
    plt.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"Saved: {args.out}")


_MULTI_QP = [4, 6, 8, 10, 12, 14, 16, 18, 20]


def _main_multi(args) -> None:
    """Build a 2x5-panel multi-Qp airgap Bg comparison from cached VTUs."""
    base = Path("examples/flux_multi_pole")
    if not base.exists():
        raise FileNotFoundError(
            f"Cache directory {base} not found — run plot_flux_multi_pole.py first.")

    print(f"Multi-Qp airgap Bg comparison ({len(_MULTI_QP)} pole counts)")
    panels = []
    for Qp in _MULTI_QP:
        res_dir = base / f"qp{Qp:02d}_results"
        vtus = sorted(res_dir.glob(f"step-qp{Qp:02d}_t*.vtu"))
        if not vtus:
            print(f"  Qp={Qp:2d}  no VTU found in {res_dir}, skip")
            continue
        p = motor_params_for_poles(Qp)
        th_full, Br_full, summary = airgap_bg_fft(vtus[0], p)
        Bg_anal  = p.B_r / (1 + p.mu_r * p.g / p.h_m)
        Bg1_anal = (4 / math.pi) * Bg_anal * math.sin(p.mag_frac * math.pi / 2)
        ratio = summary["Bg1_FEM"] / Bg1_anal * 100
        print(f"  Qp={Qp:2d}  Bg1_FEM={summary['Bg1_FEM']:.3f}  "
              f"Bg1_anal={Bg1_anal:.3f}  ({ratio:.0f}%)")
        panels.append((Qp, p, th_full, Br_full, summary, Bg_anal, Bg1_anal))

    nrows, ncols = 2, 5
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 7),
                             sharex=True, sharey=True)
    axes = axes.flatten()
    for i, (Qp, p, th_full, Br_full, summary, Bg_anal, Bg1_anal) in enumerate(panels):
        title = (f"Qp={Qp}  Qs={p.Qs}\n"
                 f"Bg1_FEM={summary['Bg1_FEM']:.2f} T  "
                 f"({summary['Bg1_FEM']/Bg1_anal*100:.0f}% of analytical)")
        make_panel(axes[i], p, th_full, Br_full, summary, title)
        axes[i].set_xlim(0, 360)

    for j in range(len(panels), nrows * ncols):
        axes[j].axis("off")

    fig.suptitle("No-load airgap flux density — Qp = 4 … 20  "
                 "(FEM solid, analytical rectangular dashed, fundamental dotted)",
                 fontsize=12, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = Path(args.multi_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
