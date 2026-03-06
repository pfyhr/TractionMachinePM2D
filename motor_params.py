"""
motor_params.py — MotorParams dataclass with validation and helpers.

All lengths in mm unless stated.  Angles in degrees unless stated.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

π = math.pi


# ─────────────────────────────────────────────────────────────────────────────
#  Defaults matching the TractionMachinePM2D reference design
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class MotorParams:
    # ── Topology ──────────────────────────────────────────────────────────────
    Qs:  int   = 48       # total slot count
    Qp:  int   = 8        # pole count  (must be even)
    m:   int   = 3        # phase count

    # ── Stator geometry [mm] ─────────────────────────────────────────────────
    R_so: float = 110.0   # stator outer radius
    R_si: float =  74.0   # stator bore radius

    # ── Rectangular slot [mm] ────────────────────────────────────────────────
    b1:     float = 2.5   # slot opening width  (tooth-tip gap)
    h1:     float = 0.5   # slot opening height (tooth-tip depth)
    b_slot: float = 7.0   # slot body width  (constant, tangential)
    h_slot: float = 19.5  # slot body height (radial depth)

    # ── Hairpin winding ───────────────────────────────────────────────────────
    n_hp:    int   = 6    # radial conductor layers per slot
    t_liner: float = 0.5  # slot liner thickness, each side  [mm]
    t_enam:  float = 0.1  # inter-conductor enamel gap, each side [mm]

    # ── Air gap [mm] ─────────────────────────────────────────────────────────
    g: float = 0.75       # mechanical air gap (min recommended: 0.3 mm)

    # ── Rotor geometry [mm] ──────────────────────────────────────────────────
    R_ri: float = 25.0    # rotor inner radius (shaft outer)

    # ── Rectangular magnet [mm, fraction] ────────────────────────────────────
    mag_frac:  float = 0.80  # magnet arc fraction of pole pitch (tangential)
    h_m:       float =  4.0  # magnet radial depth
    h_bridge:  float =  4.0  # iron bridge above magnet (R_mo = R_ro - h_bridge)
    w_air:     float =  2.0  # air-pocket width at each magnet end

    # ── Magnet material (default: N45SH NdFeB at 20 °C) ─────────────────────
    # N45SH: Br=1.35 T, Hci≈2000 kA/m (SH high-temp grade), mu_r≈1.05,
    #        σ≈625 kS/m (ρ≈1.6 µΩ·m), density≈7500 kg/m³
    B_r:   float = 1.35   # remanence [T]
    mu_r:  float = 1.05   # relative permeability of magnet

    # ── Operational ───────────────────────────────────────────────────────────
    rpm:      float = 1500.0  # rated speed
    Is:       float = 200.0   # peak phase current [A]
    L_active: float = 170.0   # active (stack) length [mm]

    # ── Mesh sizes [mm] ──────────────────────────────────────────────────────
    lc_so:  float = 5.0    # stator outer
    lc_si:  float = 1.5    # stator bore / tooth tip
    lc_hp:  float = 0.8    # hairpin conductor
    lc_ins: float = 1.0    # slot insulation
    lc_gap: float = 0.35   # air gap
    lc_ro:  float = 2.5    # rotor outer
    lc_ri:  float = 5.0    # rotor inner (shaft)
    lc_mag: float = 1.2    # magnet

    # ─────────────────────────────────────────────────────────────────────────
    #  Derived quantities (computed, not set by user)
    # ─────────────────────────────────────────────────────────────────────────
    @property
    def PP(self) -> int:
        """Pole pairs."""
        return self.Qp // 2

    @property
    def ns(self) -> int:
        """Slots per sector (pole pitch)."""
        return self.Qs // self.Qp

    @property
    def q(self) -> float:
        """Slots per pole per phase."""
        return self.Qs / (self.Qp * self.m)

    @property
    def θs(self) -> float:
        """Sector angle [rad]."""
        return 2 * π / self.Qp

    @property
    def sp(self) -> float:
        """Slot pitch [rad]."""
        return self.θs / self.ns

    @property
    def R_sb(self) -> float:
        """Sliding boundary radius (mid air gap) [mm]."""
        return self.R_si - self.g / 2

    @property
    def R_ro(self) -> float:
        """Rotor outer radius [mm]."""
        return self.R_si - self.g

    @property
    def R_mo(self) -> float:
        """Magnet outer radius [mm] (= R_ro − h_bridge, inner face of iron bridge)."""
        return self.R_ro - self.h_bridge

    @property
    def R_mi(self) -> float:
        """Magnet inner radius [mm] (= R_mo − h_m)."""
        return self.R_ro - self.h_bridge - self.h_m

    @property
    def slot_pitch_arc(self) -> float:
        """Arc length of slot pitch at bore [mm]."""
        return 2 * π * self.R_si / self.Qs

    @property
    def tooth_tip_width(self) -> float:
        """Tooth-tip arc width at bore between adjacent slot openings [mm]."""
        return self.slot_pitch_arc - self.b1

    @property
    def tooth_body_width(self) -> float:
        """Tooth width at mid-slot radius [mm]."""
        r_mid = self.R_si + self.h1 + self.h_slot / 2
        return 2 * π * r_mid / self.Qs - self.b_slot

    @property
    def back_iron_depth(self) -> float:
        """Stator back iron radial depth [mm]."""
        return self.R_so - (self.R_si + self.h1 + self.h_slot)

    # ── Hairpin / copper ─────────────────────────────────────────────────────
    @property
    def b_cond(self) -> float:
        return self.b_slot - 2 * self.t_liner

    @property
    def h_cond(self) -> float:
        return self.h_slot - 2 * self.t_liner

    @property
    def h_layer(self) -> float:
        """Height of one conductor layer (including enamel gap) [mm]."""
        return self.h_cond / self.n_hp

    @property
    def b_hp(self) -> float:
        """Hairpin copper width [mm]."""
        return self.b_cond - 2 * self.t_enam

    @property
    def h_hp(self) -> float:
        """Hairpin copper height [mm]."""
        return self.h_layer - 2 * self.t_enam

    @property
    def A_hp(self) -> float:
        """Copper area of one hairpin conductor [mm²]."""
        return self.b_hp * self.h_hp

    @property
    def Carea(self) -> float:
        """Total conductor (copper) area per slot [m²]."""
        return self.n_hp * self.A_hp * 1e-6  # convert mm² → m²

    @property
    def fill_factor(self) -> float:
        """Copper fill factor (Cu area / slot body area)."""
        return self.n_hp * self.A_hp / (self.b_slot * self.h_slot)

    # ── Magnet geometry ───────────────────────────────────────────────────────
    @property
    def w_mag(self) -> float:
        """Rectangular magnet width (chord at inner-magnet radius R_mi) [mm].

        Using sin() (chord at R_mi) ensures the magnet inner corners sit within
        the rotor (r ≤ R_ro).  The outer corners still extend beyond R_ro and
        are clipped to the cylindrical boundary — this is standard FEM practice
        for rectangular inset-PM magnets.
        """
        half_angle = self.mag_frac * self.θs / 2
        return 2 * self.R_mi * math.sin(half_angle)

    @property
    def A_mag(self) -> float:
        """Magnet cross-sectional area [mm²]."""
        return self.w_mag * self.h_m

    # ── Operating point ───────────────────────────────────────────────────────
    @property
    def omega_mech(self) -> float:
        """Mechanical angular velocity [rad/s]."""
        return self.rpm * π / 30

    @property
    def f_el(self) -> float:
        """Electrical frequency [Hz]."""
        return self.PP * self.rpm / 60

    @property
    def J_peak(self) -> float:
        """Peak current density [A/m²]."""
        return self.Is / self.Carea

    @property
    def SCALE(self) -> float:
        """2D → full-machine scaling factor (sectors × active length in m)."""
        return self.Qp * (self.L_active * 1e-3)

    # ─────────────────────────────────────────────────────────────────────────
    #  Validation
    # ─────────────────────────────────────────────────────────────────────────
    def validate(self) -> "MotorParams":
        """
        Check feasibility of the design.  Raises ValueError with a descriptive
        message on the first violated constraint.  Returns self on success so
        you can chain: params = MotorParams(...).validate()
        """
        errors: list[str] = []

        # ── Topology ──────────────────────────────────────────────────────────
        if self.Qp % 2 != 0:
            errors.append(f"Pole count Qp={self.Qp} must be even.")
        if self.Qs % self.Qp != 0:
            errors.append(
                f"Qs={self.Qs} is not divisible by Qp={self.Qp} "
                f"(ns = {self.Qs/self.Qp:.2f} slots/sector must be integer)."
            )
        if self.Qs % (self.Qp * self.m) != 0:
            errors.append(
                f"Qs={self.Qs} not divisible by Qp*m = {self.Qp*self.m}; "
                f"q = {self.q:.4f} must be integer."
            )

        # ── Radii ordering ────────────────────────────────────────────────────
        if not (self.R_ri < self.R_ro < self.R_si < self.R_so):
            errors.append(
                f"Radii must satisfy R_ri < R_ro < R_si < R_so: "
                f"{self.R_ri} < {self.R_ro} < {self.R_si} < {self.R_so}"
            )

        # ── Air gap ───────────────────────────────────────────────────────────
        if self.g < 0.3:
            errors.append(
                f"Air gap g={self.g:.2f} mm < 0.3 mm manufacturing floor."
            )

        # ── Slot opening ──────────────────────────────────────────────────────
        if self.b1 >= self.slot_pitch_arc:
            errors.append(
                f"Slot opening b1={self.b1} mm ≥ slot pitch at bore "
                f"{self.slot_pitch_arc:.3f} mm — no room for teeth."
            )

        # ── Tooth widths ──────────────────────────────────────────────────────
        if self.tooth_tip_width <= 0:
            errors.append(
                f"Tooth tip width at bore = {self.tooth_tip_width:.3f} mm ≤ 0 "
                f"(slot pitch={self.slot_pitch_arc:.3f}, b1={self.b1})."
            )
        if self.tooth_body_width <= 0:
            errors.append(
                f"Tooth body width at mid-slot = {self.tooth_body_width:.3f} mm ≤ 0 "
                f"(b_slot={self.b_slot})."
            )
        _MIN_TOOTH = 1.0  # mm — absolute minimum for structural integrity
        if 0 < self.tooth_tip_width < _MIN_TOOTH:
            errors.append(
                f"Tooth tip too thin: {self.tooth_tip_width:.3f} mm < {_MIN_TOOTH} mm."
            )
        if 0 < self.tooth_body_width < _MIN_TOOTH:
            errors.append(
                f"Tooth body too thin: {self.tooth_body_width:.3f} mm < {_MIN_TOOTH} mm."
            )

        # ── Slot fits inside stator ───────────────────────────────────────────
        if self.back_iron_depth < 2.0:
            errors.append(
                f"Back iron depth = {self.back_iron_depth:.2f} mm < 2 mm "
                f"(R_so={self.R_so}, slot top at {self.R_si + self.h1 + self.h_slot:.1f})."
            )

        # ── Slot body wider than opening ──────────────────────────────────────
        if self.b_slot < self.b1:
            errors.append(
                f"Slot body b_slot={self.b_slot} < slot opening b1={self.b1}."
            )

        # ── Hairpin conductor positive dimensions ────────────────────────────
        if self.b_hp <= 0:
            errors.append(
                f"Hairpin width b_hp={self.b_hp:.3f} mm ≤ 0 "
                f"(b_slot={self.b_slot}, t_liner={self.t_liner}, t_enam={self.t_enam})."
            )
        if self.h_hp <= 0:
            errors.append(
                f"Hairpin height h_hp={self.h_hp:.3f} mm ≤ 0 "
                f"(h_slot={self.h_slot}, n_hp={self.n_hp}, "
                f"t_liner={self.t_liner}, t_enam={self.t_enam})."
            )

        # ── Fill factor ───────────────────────────────────────────────────────
        _MIN_FILL, _MAX_FILL = 0.25, 0.85
        kf = self.fill_factor
        if kf < _MIN_FILL:
            errors.append(
                f"Fill factor {kf*100:.1f}% < {_MIN_FILL*100:.0f}% — "
                f"consider increasing n_hp or reducing t_liner/t_enam."
            )
        if kf > _MAX_FILL:
            errors.append(
                f"Fill factor {kf*100:.1f}% > {_MAX_FILL*100:.0f}% — "
                f"insulation is unphysically thin."
            )

        # ── Magnet ────────────────────────────────────────────────────────────
        if self.mag_frac <= 0 or self.mag_frac >= 1:
            errors.append(
                f"mag_frac={self.mag_frac} must be in (0, 1)."
            )
        if self.h_m <= 0:
            errors.append(f"Magnet depth h_m={self.h_m} mm must be positive.")
        if self.h_bridge < 0:
            errors.append(f"Iron bridge h_bridge={self.h_bridge:.2f} mm must be ≥ 0.")
        if self.R_mi <= self.R_ri:
            errors.append(
                f"Magnet inner radius R_mi={self.R_mi:.2f} ≤ shaft outer R_ri={self.R_ri}."
            )
        if self.w_mag + 2 * self.w_air > self.R_ro * self.θs * self.mag_frac * 1.5:
            # rough sanity only — exact fit check is geometric
            pass
        if self.w_air < 0:
            errors.append(f"Air pocket width w_air={self.w_air} must be ≥ 0.")

        if errors:
            raise ValueError("MotorParams validation failed:\n" +
                             "\n".join(f"  • {e}" for e in errors))
        return self

    # ─────────────────────────────────────────────────────────────────────────
    #  Summary
    # ─────────────────────────────────────────────────────────────────────────
    def summary(self) -> str:
        lines = [
            "=" * 60,
            f"  Motor: {self.Qs}s / {self.Qp}p / {self.m}ph  "
            f"(q={self.q:.0f} spp,  PP={self.PP})",
            "-" * 60,
            f"  Stator OD / bore      : {2*self.R_so:.0f} / {2*self.R_si:.0f} mm",
            f"  Air gap               : {self.g:.2f} mm",
            f"  Rotor OD / shaft OD   : {2*self.R_ro:.0f} / {2*self.R_ri:.0f} mm",
            "-" * 60,
            f"  Slot pitch at bore    : {self.slot_pitch_arc:.3f} mm",
            f"  Tooth tip / body      : {self.tooth_tip_width:.2f} / "
            f"{self.tooth_body_width:.2f} mm",
            f"  Slot (w × h)          : {self.b_slot:.1f} × {self.h_slot:.1f} mm",
            f"  Back iron depth       : {self.back_iron_depth:.1f} mm",
            "-" * 60,
            f"  Hairpin (w × h)       : {self.b_hp:.2f} × {self.h_hp:.3f} mm",
            f"  Conductors / slot     : {self.n_hp}",
            f"  Cu area / slot        : {self.n_hp * self.A_hp:.1f} mm²  "
            f"({self.Carea*1e6:.3f} mm²)",
            f"  Fill factor           : {self.fill_factor*100:.1f} %",
            "-" * 60,
            f"  Magnet (w × h)        : {self.w_mag:.2f} × {self.h_m:.1f} mm",
            f"  Magnet material       : Br={self.B_r:.2f} T  mu_r={self.mu_r}  (N45SH)",
            f"  Iron bridge           : {self.h_bridge:.1f} mm  "
            f"(R_mo={self.R_mo:.2f}, R_mi={self.R_mi:.2f} mm)",
            f"  Magnet fraction       : {self.mag_frac*100:.0f}% of pole pitch",
            f"  Air pocket width      : {self.w_air:.1f} mm",
            "-" * 60,
            f"  Speed / frequency     : {self.rpm:.0f} rpm / {self.f_el:.0f} Hz",
            f"  Peak current          : {self.Is:.0f} A",
            f"  Peak J                : {self.J_peak/1e6:.2f} MA/m²",
            f"  SCALE (2D → full)     : {self.SCALE:.4f} m",
            "=" * 60,
        ]
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()


# ─────────────────────────────────────────────────────────────────────────────
#  Helper: suggest slot dimensions to hit a target fill factor
# ─────────────────────────────────────────────────────────────────────────────
def suggest_slot(
    Qs: int,
    Qp: int,
    R_si: float,
    R_so: float,
    g: float = 0.75,
    *,
    target_fill: float = 0.55,
    n_hp: int = 6,
    t_liner: float = 0.5,
    t_enam: float = 0.1,
    tooth_fraction: float = 0.45,   # tooth width as fraction of slot pitch
    back_iron_min: float = 8.0,     # minimum back iron [mm]
) -> dict:
    """
    Given topology and bore/OD, suggest b_slot and h_slot that achieve
    approximately target_fill copper fill factor.

    tooth_fraction: tooth_body_width / slot_pitch_mid — 0.45 ≈ equal tooth+slot.
    Returns a dict with suggested values and actual fill factor achieved.
    """
    slot_pitch_bore = 2 * π * R_si / Qs
    # Maximum slot body width so tooth at mid-slot ≥ 1 mm
    R_ro = R_si - g
    h_slot_max = R_so - R_si - back_iron_min - 0.5  # 0.5 = h1 opening
    h_slot_max = max(h_slot_max, 1.0)

    # Iteratively find b_slot for tooth_fraction at mid-radius
    h_slot = h_slot_max
    r_mid = R_si + 0.5 + h_slot / 2    # h1=0.5 opening
    slot_pitch_mid = 2 * π * r_mid / Qs
    b_slot = slot_pitch_mid * (1.0 - tooth_fraction)

    # Clamp b_slot to sensible range
    b1 = 2.5
    b_slot = max(b_slot, b1 + 1.0)     # must be wider than opening
    b_slot = min(b_slot, slot_pitch_mid - 1.0)  # leave at least 1 mm tooth

    # Compute fill factor with this b_slot and h_slot
    def fill(bs, hs):
        bc = bs - 2 * t_liner
        hc = hs - 2 * t_liner
        if bc <= 0 or hc <= 0 or n_hp <= 0:
            return 0.0
        h_lay = hc / n_hp
        bh = bc - 2 * t_enam
        hh = h_lay - 2 * t_enam
        if bh <= 0 or hh <= 0:
            return 0.0
        return n_hp * bh * hh / (bs * hs)

    kf = fill(b_slot, h_slot)

    # If fill is too high, reduce b_slot; if too low, widen it
    for _ in range(50):
        kf = fill(b_slot, h_slot)
        if abs(kf - target_fill) < 0.005:
            break
        # Adjust b_slot proportionally
        if kf > 0:
            b_slot *= (target_fill / kf) ** 0.5
        b_slot = max(b_slot, b1 + 0.5)
        b_slot = min(b_slot, slot_pitch_mid - 1.0)

    b_slot = round(b_slot, 1)
    h_slot = round(h_slot, 1)
    kf = fill(b_slot, h_slot)

    tooth_tip = slot_pitch_bore - b1
    r_mid_final = R_si + 0.5 + h_slot / 2
    slot_pitch_mid_final = 2 * π * r_mid_final / Qs
    tooth_body = slot_pitch_mid_final - b_slot
    back_iron = R_so - (R_si + 0.5 + h_slot)

    return {
        "b_slot":      b_slot,
        "h_slot":      h_slot,
        "fill_factor": kf,
        "tooth_tip":   tooth_tip,
        "tooth_body":  tooth_body,
        "back_iron":   back_iron,
        "Carea_m2":    n_hp * (b_slot - 2*t_liner - 2*t_enam) *
                       ((h_slot - 2*t_liner)/n_hp - 2*t_enam) * 1e-6,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Default reference design (used by other scripts)
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_PARAMS = MotorParams()


if __name__ == "__main__":
    p = MotorParams().validate()
    print(p)
    print()
    sug = suggest_slot(48, 8, 74.0, 110.0, target_fill=0.55)
    print("suggest_slot(target_fill=0.55):")
    for k, v in sug.items():
        print(f"  {k:15s} = {v:.4g}")
