"""
plot_multi_pole.py — Analytical geometry plots for Qp = 2 … 20.

Draws rotor and stator sector outlines for each pole count using pure
matplotlib (no gmsh).  Saves a multi-panel figure to results/png/.

Usage:
    python3 plot_multi_pole.py
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Arc, FancyArrowPatch
import numpy as np

from motor_params import motor_params_for_poles

ALL_QP = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

π = math.pi


def arc_xy(r: float, t1: float, t2: float, n: int = 120) -> tuple[np.ndarray, np.ndarray]:
    t = np.linspace(t1, t2, n)
    return r * np.cos(t), r * np.sin(t)


def sector_patch(r1, r2, t1, t2, **kw):
    """Return a matplotlib Polygon for an annular sector."""
    n = 120
    t = np.linspace(t1, t2, n)
    xs = np.concatenate([r1 * np.cos(t), r2 * np.cos(t[::-1])])
    ys = np.concatenate([r1 * np.sin(t), r2 * np.sin(t[::-1])])
    return plt.Polygon(np.column_stack([xs, ys]), **kw)


def pie_patch(r, t1, t2, **kw):
    """Return a matplotlib Polygon for a pie sector (origin → arc)."""
    n = 120
    t = np.linspace(t1, t2, n)
    xs = np.concatenate([[0], r * np.cos(t), [0]])
    ys = np.concatenate([[0], r * np.sin(t), [0]])
    return plt.Polygon(np.column_stack([xs, ys]), **kw)


def magnet_patch(p, **kw):
    """Rectangular magnet corners clipped to R_ro, returned as a Polygon."""
    θ_c = p.θs / 2
    R_mi, h_m, w_mag = p.R_mi, p.h_m, p.w_mag
    # Rectangle in local frame: x ∈ [R_mi, R_mi+h_m], y ∈ [-w_mag/2, +w_mag/2]
    corners_local = np.array([
        [R_mi,        -w_mag / 2],
        [R_mi + h_m,  -w_mag / 2],
        [R_mi + h_m,  +w_mag / 2],
        [R_mi,        +w_mag / 2],
    ])
    c, s = math.cos(θ_c), math.sin(θ_c)
    rot = np.array([[c, -s], [s, c]])
    corners = corners_local @ rot.T
    # Clip outer corners to R_ro
    for i, (x, y) in enumerate(corners):
        r = math.hypot(x, y)
        if r > p.R_ro:
            corners[i] *= p.R_ro / r
    return plt.Polygon(corners, **kw)


def pocket_patch(p, side: int, **kw):
    """Air pocket (w_air × h_m) on left (side=+1) or right (side=-1) of magnet."""
    θ_c = p.θs / 2
    R_mi, h_m, w_mag, w_air = p.R_mi, p.h_m, p.w_mag, p.w_air
    if w_air <= 0:
        return None
    y0 = -w_mag / 2 - w_air if side == -1 else w_mag / 2
    corners_local = np.array([
        [R_mi,       y0],
        [R_mi + h_m, y0],
        [R_mi + h_m, y0 + w_air],
        [R_mi,       y0 + w_air],
    ])
    c, s = math.cos(θ_c), math.sin(θ_c)
    rot = np.array([[c, -s], [s, c]])
    corners = corners_local @ rot.T
    # Clip to sector [0, θs] and rotor radius R_ro
    for i, (x, y) in enumerate(corners):
        r = math.hypot(x, y)
        if r > p.R_ro:
            corners[i] *= p.R_ro / r
    return plt.Polygon(corners, **kw)


def draw_rotor_sector(ax, p, title: str):
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=8)
    ax.axis("off")

    # Shaft
    shaft = pie_patch(p.R_ri, 0, p.θs, color="#cccccc", zorder=1)
    ax.add_patch(shaft)

    # Rotor iron
    iron = sector_patch(p.R_ri, p.R_ro, 0, p.θs, color="#aec6e8", zorder=2)
    ax.add_patch(iron)

    # Airgap (rotor side)
    gap = sector_patch(p.R_ro, p.R_sb, 0, p.θs, color="#f0f0f0", alpha=0.5, zorder=3)
    ax.add_patch(gap)

    # Magnet
    mag = magnet_patch(p, color="#e87a5d", zorder=4)
    ax.add_patch(mag)

    # Air pockets
    for side in (-1, +1):
        pk = pocket_patch(p, side, color="#ffffff", edgecolor="#555555",
                          linewidth=0.6, zorder=5)
        if pk is not None:
            ax.add_patch(pk)

    # Sector boundary lines
    for angle in (0, p.θs):
        ax.plot([0, p.R_sb * math.cos(angle)],
                [0, p.R_sb * math.sin(angle)],
                "k--", lw=0.5, zorder=6)

    r_max = p.R_sb * 1.05
    ax.set_xlim(-r_max * 0.1, r_max)
    ax.set_ylim(-r_max * 0.1, r_max)


def draw_stator_sector(ax, p, title: str):
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=8)
    ax.axis("off")

    # Stator iron
    iron = sector_patch(p.R_si, p.R_so, 0, p.θs, color="#aec6e8", zorder=1)
    ax.add_patch(iron)

    # Airgap (stator side)
    gap = sector_patch(p.R_sb, p.R_si, 0, p.θs, color="#f0f0f0", alpha=0.5, zorder=2)
    ax.add_patch(gap)

    # Slots
    for i in range(p.ns):
        θ_c = (i + 0.5) * p.sp

        # Slot body
        R_si, h1, b_slot, h_slot = p.R_si, p.h1, p.b_slot, p.h_slot
        corners_local = np.array([
            [R_si + h1,           -b_slot / 2],
            [R_si + h1 + h_slot,  -b_slot / 2],
            [R_si + h1 + h_slot,  +b_slot / 2],
            [R_si + h1,           +b_slot / 2],
        ])
        c, s = math.cos(θ_c), math.sin(θ_c)
        rot = np.array([[c, -s], [s, c]])
        corners = corners_local @ rot.T
        ax.add_patch(plt.Polygon(corners, color="#f5c842", zorder=3))

        # Slot opening
        op_corners_local = np.array([
            [R_si,       -p.b1 / 2],
            [R_si + h1,  -p.b1 / 2],
            [R_si + h1,  +p.b1 / 2],
            [R_si,       +p.b1 / 2],
        ])
        op_corners = op_corners_local @ rot.T
        ax.add_patch(plt.Polygon(op_corners, color="#ffffff", zorder=4))

    # Sector boundary lines
    for angle in (0, p.θs):
        ax.plot([0, p.R_so * math.cos(angle)],
                [0, p.R_so * math.sin(angle)],
                "k--", lw=0.5, zorder=6)

    r_max = p.R_so * 1.05
    ax.set_xlim(-r_max * 0.1, r_max)
    ax.set_ylim(-r_max * 0.1, r_max)


def main():
    out_dir = Path(__file__).parent / "results" / "png"
    out_dir.mkdir(parents=True, exist_ok=True)

    n = len(ALL_QP)
    ncols = 5
    nrows = (n + ncols - 1) // ncols

    # ── Rotor panels ──────────────────────────────────────────────────────────
    fig_r, axes_r = plt.subplots(nrows, ncols, figsize=(16, 7))
    axes_r = axes_r.flatten()
    for ax in axes_r[n:]:
        ax.axis("off")

    for idx, Qp in enumerate(ALL_QP):
        p = motor_params_for_poles(Qp)
        draw_rotor_sector(
            axes_r[idx], p,
            f"Qp={Qp}  (θs={math.degrees(p.θs):.0f}°)\n"
            f"w_mag={p.w_mag:.1f} mm  mag_frac={p.mag_frac:.2f}"
        )

    fig_r.suptitle("Rotor sectors — Qp = 2 … 20", fontsize=11, y=1.01)
    fig_r.tight_layout()
    out_r = out_dir / "multi_pole_rotor.png"
    fig_r.savefig(out_r, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_r}")
    plt.close(fig_r)

    # ── Stator panels ─────────────────────────────────────────────────────────
    fig_s, axes_s = plt.subplots(nrows, ncols, figsize=(16, 7))
    axes_s = axes_s.flatten()
    for ax in axes_s[n:]:
        ax.axis("off")

    for idx, Qp in enumerate(ALL_QP):
        p = motor_params_for_poles(Qp)
        draw_stator_sector(
            axes_s[idx], p,
            f"Qp={Qp}  Qs={p.Qs}\n"
            f"b_slot={p.b_slot:.1f} mm  kf={p.fill_factor*100:.0f}%"
        )

    fig_s.suptitle("Stator sectors — Qp = 2 … 20", fontsize=11, y=1.01)
    fig_s.tight_layout()
    out_s = out_dir / "multi_pole_stator.png"
    fig_s.savefig(out_s, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_s}")
    plt.close(fig_s)

    # ── Parameter table ───────────────────────────────────────────────────────
    print("\nPole-count parameter summary:")
    print(f"{'Qp':>4} {'Qs':>4} {'θs°':>6} {'w_mag':>7} {'mag_f':>6} "
          f"{'b_slot':>7} {'b1':>5} {'kf%':>5} {'tooth_b':>8} {'tooth_t':>8}")
    print("-" * 70)
    for Qp in ALL_QP:
        p = motor_params_for_poles(Qp)
        print(f"{Qp:>4} {p.Qs:>4} {math.degrees(p.θs):>6.1f} "
              f"{p.w_mag:>7.2f} {p.mag_frac:>6.3f} "
              f"{p.b_slot:>7.2f} {p.b1:>5.2f} {p.fill_factor*100:>5.1f} "
              f"{p.tooth_body_width:>8.2f} {p.tooth_tip_width:>8.2f}")


if __name__ == "__main__":
    main()
