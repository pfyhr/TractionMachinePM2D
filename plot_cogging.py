"""
plot_cogging.py — Cogging torque from the no-load (Is=0) BEMF run.

Reads `results_bemf/scalars.dat` (the open-circuit transient solve at
1000 rpm with no stator current — any torque present is purely
cogging) and produces:

  * Time-domain torque waveform vs rotor angle.
  * FFT of the steady-state portion with the LCM(Qs,Qp) cogging
    fundamental marked.

For the default 48s/8p design, LCM(48, 8) = 48 cogging cycles per
mech revolution → 48·fme = 800 Hz at 1000 rpm.

Output: results/png/cogging_torque.png
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from motor_params import MotorParams

ROOT     = Path(__file__).resolve().parent
RES_DIR  = ROOT / "results_bemf"
PNG_OUT  = ROOT / "results/png/cogging_torque.png"


def main() -> None:
    p = MotorParams()
    BEMF_RPM = 1000.0   # matches example.ipynb cell 27

    sf = RES_DIR / "scalars.dat"
    if not sf.exists():
        raise FileNotFoundError(
            f"{sf} not found — run the no-load BEMF cell in example.ipynb first."
        )
    data = np.loadtxt(sf)
    ag = data[:, 5] * p.SCALE   # full-machine air-gap torque
    nper = 120
    fme  = BEMF_RPM / 60
    dt   = 1.0 / (fme * nper)
    n    = data.shape[0]
    t_s  = np.arange(1, n + 1) * dt
    rotor_deg = (BEMF_RPM / 60) * t_s * 360
    ss   = t_s > 1.0 / fme

    fig, axes = plt.subplots(2, 1, figsize=(11, 7))

    # ── Top: time domain torque + rotor angle ────────────────────────────
    ax = axes[0]
    ax.plot(rotor_deg, ag, color="C0", lw=1.0)
    ax.axhline(0, color="k", lw=0.4)
    cog_pp = float(ag[ss].max() - ag[ss].min())
    cog_mean = float(ag[ss].mean())
    cog_std = float(ag[ss].std())
    ax.set_xlabel("Rotor angle [°]")
    ax.set_ylabel("Cogging torque [N·m]")
    ax.set_title(f"No-load (cogging) torque at {BEMF_RPM:.0f} rpm  "
                 f"— mean = {cog_mean:+.2f}, p-p = {cog_pp:.1f}, "
                 f"σ = {cog_std:.2f} N·m")
    ax.grid(alpha=0.3)
    # Mark every 1/Qs of a revolution = slot pitch boundaries
    for k in range(p.Qs):
        ax.axvline(k * 360 / p.Qs, color="k", lw=0.2, alpha=0.25)

    # ── Bottom: FFT ───────────────────────────────────────────────────────
    ax = axes[1]
    ag_ss = ag[ss] - ag[ss].mean()
    freqs = np.fft.rfftfreq(len(ag_ss), d=dt)
    mag = np.abs(np.fft.rfft(ag_ss)) / len(ag_ss) * 2
    ax.stem(freqs / fme, mag, basefmt=" ", linefmt="C2-", markerfmt="C2o")
    lcm = abs(p.Qs * p.Qp) // math.gcd(p.Qs, p.Qp)
    ax.axvline(lcm, color="k", ls="--", lw=0.7, alpha=0.6,
               label=f"LCM(Qs,Qp) = {lcm}× fme")
    ax.set_xlabel("Harmonic order  (multiples of fme = "
                  f"{fme:.1f} Hz)")
    ax.set_ylabel("Cogging amplitude [N·m]")
    ax.set_title("Cogging torque FFT — fundamental at "
                 f"LCM(Qs, Qp) = {lcm} (= {lcm * fme:.0f} Hz)")
    ax.set_xlim(0, max(75, lcm * 1.5))
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    PNG_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PNG_OUT, dpi=140, bbox_inches="tight")
    print(f"Saved: {PNG_OUT}")
    print(f"Cogging summary:")
    print(f"  Peak-peak amplitude    : {cog_pp:.2f} N·m")
    print(f"  RMS                    : {ag_ss.std():.2f} N·m")
    print(f"  As fraction of MTPA T  : "
          f"{cog_pp / 315 * 100:.1f}%   (MTPA ≈ 315 N·m for default)")


if __name__ == "__main__":
    main()
