"""
plot_gamma_sweep.py — Visualise the γ-sweep with IPM torque fit.

Reads T_mean from results_sweep/g*/scalars.dat (written by the
example.ipynb sweep cell), fits the standard IPM torque equation

    T(γ) = a·cos(γ) + c·sin(2γ)
         = (3/2)·PP · [Ψ_PM·cos(γ) − ½·(L_q−L_d)·I_s·sin(2γ)]·I_s

and overlays the FEM points, the PM-torque component, the reluctance-
torque component, and the total fit. Identifies the MTPA angle.

Outputs:
    results/png/gamma_sweep.png

Usage:
    python3 plot_gamma_sweep.py
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from motor_params import MotorParams

ROOT      = Path(__file__).resolve().parent
SWEEP_DIR = ROOT / "results_sweep"
PNG_OUT   = ROOT / "results/png/gamma_sweep.png"


def _read_t_mean(scalars_file: Path, p: MotorParams,
                 nper: int = 120, skip_cycles: int = 1) -> float:
    """Read group-1 torque from scalars.dat, return steady-state mean (full machine)."""
    data = np.loadtxt(scalars_file)
    g1 = data[:, 8] * p.SCALE
    fme = p.rpm / 60
    dt  = 1 / (fme * nper)
    t_s = np.arange(1, len(g1) + 1) * dt
    return float(np.mean(g1[t_s > skip_cycles / fme]))


def main() -> None:
    p = MotorParams()
    if not SWEEP_DIR.exists():
        raise FileNotFoundError(f"{SWEEP_DIR} not found — "
                                "run example.ipynb cell 25 first.")

    # Recover (γ, T_mean) pairs from the sweep
    runs = []
    n_per_sweep = 10
    gammas = np.linspace(-90, 0, n_per_sweep)
    for idx, gamma in enumerate(gammas):
        sf = SWEEP_DIR / f"g{idx}" / "scalars.dat"
        if not sf.exists():
            print(f"  γ={gamma:+5.1f}°  no scalars.dat, skip")
            continue
        T = _read_t_mean(sf, p)
        runs.append((float(gamma), T))
    runs.sort()
    γ_arr = np.array([r[0] for r in runs])
    T_arr = np.array([r[1] for r in runs])
    print("γ-sweep T(γ):")
    for g, T in runs:
        print(f"  γ={g:+6.1f}°  T_mean={T:+7.2f} N·m")

    # ── Fit T(γ) = a·cos(γ−γ0) + c·sin(2(γ−γ0)) ───────────────────────────
    # The IPM torque should reduce to a·cos(γ−γ0) at high γ, with reluctance
    # contributing the 2γ term. γ0 is a residual phase offset (alignment).
    # Linearise as A·cos γ + B·sin γ + C·cos 2γ + D·sin 2γ:
    γ_rad = np.radians(γ_arr)
    M = np.column_stack([np.cos(γ_rad), np.sin(γ_rad),
                          np.cos(2 * γ_rad), np.sin(2 * γ_rad)])
    coef, *_ = np.linalg.lstsq(M, T_arr, rcond=None)
    A_, B_, C_, D_ = coef
    res = T_arr - M @ coef
    rms_res = float(np.sqrt(np.mean(res ** 2)))
    a   = math.hypot(A_, B_)                # PM amplitude
    γ0  = math.degrees(math.atan2(B_, A_))  # PM phase offset
    rel = math.hypot(C_, D_)                # reluctance amplitude (2× harmonic)
    γ0r = math.degrees(math.atan2(D_, C_)) / 2  # reluctance phase offset
    a_signed = a if math.cos(math.radians(γ0)) >= 0 else -a
    print(f"\nFit:  T(γ) = {A_:+.2f}·cos γ + {B_:+.2f}·sin γ + "
          f"{C_:+.2f}·cos 2γ + {D_:+.2f}·sin 2γ")
    print(f"      ≈ {a:.1f}·cos(γ − {γ0:.1f}°)  "
          f"+ {rel:.1f}·cos(2γ − {2*γ0r:.1f}°)")
    print(f"      RMS residual: {rms_res:.2f} N·m")
    # For the symmetric IPM model (γ0 = 0), c is the sin(2γ) coefficient:
    c = D_

    # IPM physical interpretation:
    # a = (3/2)·PP·Is·Ψ_PM        → Ψ_PM = a / (1.5·PP·Is)   [Wb]
    # c = −(3/4)·PP·Is²·(Lq−Ld)   → Lq−Ld = −c / (0.75·PP·Is²)   [H]
    psi_pm = a / (1.5 * p.PP * p.Is)
    Lq_minus_Ld = -c / (0.75 * p.PP * p.Is ** 2)
    print(f"  Implied Ψ_PM        = {psi_pm:.4f} Wb")
    print(f"  Implied (L_q − L_d) = {Lq_minus_Ld * 1e3:.3f} mH")

    # ── MTPA: dT/dγ = 0 ────────────────────────────────────────────────────
    γ_fine = np.linspace(-180, 0, 1801)
    γ_fine_rad = np.radians(γ_fine)
    T_fit_fine = (A_ * np.cos(γ_fine_rad) + B_ * np.sin(γ_fine_rad) +
                  C_ * np.cos(2 * γ_fine_rad) + D_ * np.sin(2 * γ_fine_rad))
    γ_mtpa_deg = γ_fine[np.argmax(T_fit_fine)]
    T_mtpa     = T_fit_fine.max()

    # ── Plot ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 6))
    γ_plot = np.linspace(-180, 0, 1801)
    γ_plot_rad = np.radians(γ_plot)
    T_pm  = A_ * np.cos(γ_plot_rad) + B_ * np.sin(γ_plot_rad)
    T_rel = C_ * np.cos(2 * γ_plot_rad) + D_ * np.sin(2 * γ_plot_rad)
    T_fit = T_pm + T_rel

    ax.plot(γ_plot, T_pm, ls="--", color="C0", lw=1.2, alpha=0.7,
            label=f"PM torque  ({a:.0f}·cos(γ−{γ0:.0f}°))")
    ax.plot(γ_plot, T_rel, ls=":", color="C2", lw=1.2, alpha=0.7,
            label=f"Reluctance torque  ({rel:.0f}·cos(2γ−{2*γ0r:.0f}°))")
    ax.plot(γ_plot, T_fit, color="C1", lw=2.0, label="Fit (PM + reluctance)")
    ax.scatter(γ_arr, T_arr, color="C3", s=50, zorder=4,
               edgecolor="k", lw=0.6,
               label=f"FEM ({len(γ_arr)} runs, Is={p.Is:.0f}A)")

    ax.axvline(γ_mtpa_deg, color="k", lw=0.6, ls=":")
    ax.text(γ_mtpa_deg, T_mtpa * 0.96,
            f"MTPA: γ = {γ_mtpa_deg:.1f}°,\n"
            f"T = {T_mtpa:.0f} N·m",
            ha="center", va="top", fontsize=9,
            bbox=dict(boxstyle="round", fc="white", alpha=0.85))

    ax.set_xlabel("Current advance angle γ from q-axis [°]")
    ax.set_ylabel("Steady-state mean torque [N·m]")
    ax.set_title(f"IPM torque vs current advance angle  "
                 f"({p.Qs}s/{p.Qp}p, Is={p.Is:.0f}A peak, n={p.rpm:.0f} rpm)")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.3)
    ax.axhline(0, color="k", lw=0.4)
    ax.set_xlim(-180, 0)
    plt.tight_layout()
    PNG_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PNG_OUT, dpi=140, bbox_inches="tight")
    print(f"Saved: {PNG_OUT}")

    # Diagnostic: what fraction of MTPA torque is from reluctance?
    γm = math.radians(γ_mtpa_deg)
    Tpm  = A_ * math.cos(γm) + B_ * math.sin(γm)
    Trel = C_ * math.cos(2 * γm) + D_ * math.sin(2 * γm)
    print(f"\nAt MTPA (γ = {γ_mtpa_deg:.1f}°):")
    print(f"  PM torque        : {Tpm:7.1f} N·m  ({Tpm/T_mtpa*100:.0f}%)")
    print(f"  Reluctance torque: {Trel:7.1f} N·m  ({Trel/T_mtpa*100:.0f}%)")
    print(f"  Total            : {T_mtpa:7.1f} N·m")


if __name__ == "__main__":
    main()
