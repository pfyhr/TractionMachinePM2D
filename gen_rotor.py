"""
gen_rotor.py — Rotor sector geometry with rectangular magnets and air pockets.

Generates a 45° (one pole pitch) rotor sector with:
  - Rotor iron annular sector (R_ri → R_ro)
  - Rectangular inset magnet (w_mag × h_m) centred on pole midpoint
  - Air pockets (w_air × h_m) at each tangential end of the magnet
    to reduce leakage flux around the magnet corners
  - Shaft sector (0 → R_ri)
  - Rotor airgap region (rotor side, R_ro → R_sb)

Usage:
    python3 gen_rotor.py [--gui] [--check]
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Optional

import gmsh

from motor_params import MotorParams, DEFAULT_PARAMS

π = math.pi
cos, sin, tan = math.cos, math.sin, math.tan


def _annular_sector(occ, r1: float, r2: float, t1: float, t2: float,
                    lc: float) -> int:
    O  = occ.addPoint(0, 0, 0, lc)
    p1 = occ.addPoint(r1*cos(t1), r1*sin(t1), 0, lc)
    p2 = occ.addPoint(r2*cos(t1), r2*sin(t1), 0, lc)
    p3 = occ.addPoint(r2*cos(t2), r2*sin(t2), 0, lc)
    p4 = occ.addPoint(r1*cos(t2), r1*sin(t2), 0, lc)
    l1 = occ.addLine(p1, p2)
    a2 = occ.addCircleArc(p2, O, p3)
    l3 = occ.addLine(p3, p4)
    a4 = occ.addCircleArc(p4, O, p1)
    return occ.addPlaneSurface([occ.addCurveLoop([l1, a2, l3, a4])])


def _pie_sector(occ, r: float, t1: float, t2: float, lc: float) -> int:
    O  = occ.addPoint(0, 0, 0, lc)
    p1 = occ.addPoint(r*cos(t1), r*sin(t1), 0, lc)
    p2 = occ.addPoint(r*cos(t2), r*sin(t2), 0, lc)
    l1 = occ.addLine(O, p1)
    a  = occ.addCircleArc(p1, O, p2)
    l2 = occ.addLine(p2, O)
    return occ.addPlaneSurface([occ.addCurveLoop([l1, a, l2])])


def _get_rc(occ, tag: int) -> tuple[float, float, float]:
    x, y, _ = occ.getCenterOfMass(2, tag)
    return math.hypot(x, y), math.atan2(y, x), occ.getMass(2, tag)


def _get_bound_curves(model, surf_tags: list[int]) -> set[int]:
    curves: set[int] = set()
    for st in surf_tags:
        _, down = model.getAdjacencies(2, st)
        curves.update(abs(int(c)) for c in down)
    return curves


def _curve_rc(occ, ctag: int) -> float:
    x, y, _ = occ.getCenterOfMass(1, ctag)
    return math.hypot(x, y)


def _curve_length(occ, ctag: int) -> float:
    return occ.getMass(1, ctag)


def _classify_sector_bounds(
    model, curve_set: set[int], excl: set[int]
) -> tuple[list[int], list[int]]:
    right, left = [], []
    for c in sorted(curve_set - excl):
        x1, y1, _, x2, y2, _ = model.getBoundingBox(1, c)
        xm, ym = (x1 + x2) / 2, (y1 + y2) / 2
        if math.hypot(xm, ym) < 0.5:
            continue
        if abs(y1) < 0.5 and abs(y2) < 0.5 and xm > 1.0:
            right.append(c)
        elif (abs(x1-y1) < 0.5 and abs(x2-y2) < 0.5
              and xm > 1.0 and ym > 1.0):
            left.append(c)
    return right, left


# ─────────────────────────────────────────────────────────────────────────────
#  Rectangular magnet + air pocket builder (local coords then rotated)
# ─────────────────────────────────────────────────────────────────────────────
def _make_magnet_and_pockets(occ, p: MotorParams
                             ) -> tuple[int, list[int]]:
    """
    Create rectangular magnet and end air pockets in the sector.

    Magnet centred on pole midpoint (θ = θs/2), at radial depth R_mi → R_ro.
    Sides are straight tangential lines (the "rectangular" characteristic).
    Top/bottom faces are clipped to the rotor cylinder (r ≤ R_ro), giving
    slightly curved outer/inner faces — standard practice in cylindrical FEM.

    Air pockets (w_air × h_m) are placed at each tangential end of the magnet.

    Returns (magnet_tag, [left_pocket_tag, right_pocket_tag]).
    """
    θ_c = p.θs / 2          # pole centre angle

    # In local frame (x = radial outward, y = tangential, then rotate by θ_c):
    #   Magnet rect: x ∈ [R_mi, R_mo], y ∈ [-w_mag/2, +w_mag/2]
    # The iron bridge (h_bridge thick) sits between R_mo and R_ro.
    # With h_bridge ≥ 0 the magnet outer corners may still protrude beyond R_ro
    # for unusual parameter combinations; the clip disk guards against this.

    x0 = p.R_mi

    # Clipping disk: keeps only the region r ≤ R_ro
    clip = occ.addDisk(0, 0, 0, p.R_ro, p.R_ro)
    occ.synchronize()

    # ── Magnet ────────────────────────────────────────────────────────────────
    t_mag_raw = occ.addRectangle(x0, -p.w_mag/2, 0, p.h_m, p.w_mag)
    occ.rotate([(2, t_mag_raw)], 0, 0, 0, 0, 0, 1, θ_c)
    occ.synchronize()
    res, _ = occ.intersect([(2, t_mag_raw)], [(2, clip)],
                           removeObject=True, removeTool=False)
    occ.synchronize()
    t_mag = res[0][1] if res else t_mag_raw

    # ── Air pockets ───────────────────────────────────────────────────────────
    pockets: list[int] = []
    if p.w_air > 0:
        for y0 in (-p.w_mag/2 - p.w_air, +p.w_mag/2):
            t_p = occ.addRectangle(x0, y0, 0, p.h_m, p.w_air)
            occ.rotate([(2, t_p)], 0, 0, 0, 0, 0, 1, θ_c)
            occ.synchronize()
            res_p, _ = occ.intersect([(2, t_p)], [(2, clip)],
                                     removeObject=True, removeTool=False)
            occ.synchronize()
            if res_p:
                pockets.append(res_p[0][1])

    # Remove clipping disk (no longer needed)
    occ.remove([(2, clip)], recursive=True)
    occ.synchronize()

    return t_mag, pockets


# ─────────────────────────────────────────────────────────────────────────────
#  Main geometry builder
# ─────────────────────────────────────────────────────────────────────────────
def build_rotor(
    params: MotorParams,
    mesh_out: Optional[Path] = None,
    *,
    gui: bool = False,
    check_only: bool = False,
) -> dict:
    """
    Build the rotor sector geometry, mesh it, and (optionally) export.

    Returns a dict with:
        rotor_iron   : list[int]   — surface tags
        magnet       : list[int]
        air_pocket   : list[int]
        shaft        : list[int]
        gap_rotor    : list[int]
        sb_rotor     : list[int]   — boundary curve tags (SB outer arc)
        rotor_right  : list[int]
        rotor_left   : list[int]
        A_mag_mm2    : float
        n_nodes      : int
        n_elems      : int
    """
    params.validate()
    p = params

    gmsh.initialize()
    gmsh.model.add("Rotor")
    occ   = gmsh.model.occ
    model = gmsh.model

    # ── 1. Rotor airgap sector (rotor side, R_ro → R_sb) ─────────────────────
    s_gap_r = _annular_sector(occ, p.R_ro, p.R_sb, 0, p.θs, p.lc_gap)

    # ── 2. Rotor iron sector (R_ri → R_ro, full sector) ──────────────────────
    s_rotor = _annular_sector(occ, p.R_ri, p.R_ro, 0, p.θs, p.lc_ro)

    # ── 3. Rectangular magnet + air pockets ───────────────────────────────────
    s_mag, s_pockets = _make_magnet_and_pockets(occ, p)

    # ── 4. Shaft sector (0 → R_ri) ────────────────────────────────────────────
    s_shaft = _pie_sector(occ, p.R_ri, 0, p.θs, p.lc_ri)

    # (synchronize already called inside _make_magnet_and_pockets)

    # ── 5. Fragment (rotor side only — SB stays non-conforming) ───────────────
    frags = (
        [(2, s_gap_r), (2, s_rotor), (2, s_shaft), (2, s_mag)]
        + [(2, t) for t in s_pockets]
    )
    occ.fragment(frags, [])
    occ.synchronize()

    # ── 6. Classify surfaces ─────────────────────────────────────────────────
    all_surfs = [t for _, t in model.getEntities(2)]

    rotor_iron_tags: list[int] = []
    magnet_tags:     list[int] = []
    air_pocket_tags: list[int] = []
    shaft_tags:      list[int] = []
    gap_r_tags:      list[int] = []

    # Airgap: thin annular sector — use area, NOT centroid radius.
    # The centroid of a 45° thin ring sits well inside R_sb (sinc factor
    # in the centroid integral), so r > R_sb - ε does not work here.
    A_gap_exp = 0.5 * (p.R_sb**2 - p.R_ro**2) * p.θs

    # θ and r reference values for position-based magnet/pocket detection.
    # Note: the rectangular magnet's outer corners extend beyond R_ro and are
    # clipped by occ.intersect, so area-based detection fails — use position.
    θ_c      = p.θs / 2
    r_mag_c  = p.R_mi + p.h_m / 2
    half_mag = math.atan2(p.w_mag / 2, r_mag_c)
    half_pkt = half_mag + math.atan2(max(p.w_air, 0.01), r_mag_c)

    for tag in all_surfs:
        r, θ, A = _get_rc(occ, tag)
        if θ < 0:
            θ += 2 * π

        # ── Airgap: matches expected thin-annulus area ────────────────────
        if abs(A - A_gap_exp) / A_gap_exp < 0.20:
            gap_r_tags.append(tag)
            continue

        # ── Shaft: small radius ───────────────────────────────────────────
        if r < p.R_ri + 0.5:
            shaft_tags.append(tag)
            continue

        # ── Magnet: radial band + pole-centre angle + area threshold ─────
        in_mag_r = (p.R_mi - 1.0) < r < (p.R_ro + 0.5)
        in_mag_θ = abs(θ - θ_c) < half_mag * 1.2
        if in_mag_r and in_mag_θ and A > 20.0:
            magnet_tags.append(tag)
            continue

        # ── Air pocket: same radial band, just outside magnet θ ───────────
        in_pkt_r = (p.R_mi - 1.0) < r < (p.R_ro + 0.5)
        in_pkt_θ = half_mag * 0.8 < abs(θ - θ_c) < half_pkt * 1.5
        if in_pkt_r and in_pkt_θ and p.w_air > 0 and A > 0.01:
            air_pocket_tags.append(tag)
            continue

        # ── Everything else → rotor iron ──────────────────────────────────
        rotor_iron_tags.append(tag)

    # Actual magnet area (may differ from p.A_mag — corners are clipped to R_ro)
    A_mag_actual = (
        sum(_get_rc(occ, t)[2] for t in magnet_tags) if magnet_tags else 0.0
    )

    # ── 7. Summary ────────────────────────────────────────────────────────────
    print(f"\nRotor classification ({len(all_surfs)} total surfaces):")
    print(f"  Rotor iron   : {len(rotor_iron_tags)} surface(s)")
    print(f"  Magnet       : {len(magnet_tags)} surface(s)  "
          f"(A = {A_mag_actual:.1f} mm² actual, {p.A_mag:.1f} mm² rect)")
    print(f"  Air pockets  : {len(air_pocket_tags)} surface(s)  "
          f"(expected {len(s_pockets)})")
    print(f"  Shaft        : {len(shaft_tags)} surface(s)")
    print(f"  Airgap rotor : {len(gap_r_tags)} surface(s)")

    for label, lst, exp in [
        ("Magnet",      magnet_tags,     1),
        ("Air pockets", air_pocket_tags, len(s_pockets)),
        ("Shaft",       shaft_tags,      1),
        ("Airgap",      gap_r_tags,      1),
    ]:
        if len(lst) != exp:
            print(f"  WARNING: {label} has {len(lst)} surfaces (expected {exp})")

    result = {
        "rotor_iron":  rotor_iron_tags,
        "magnet":      magnet_tags,
        "air_pocket":  air_pocket_tags,
        "shaft":       shaft_tags,
        "gap_rotor":   gap_r_tags,
        "A_mag_mm2":   A_mag_actual,
        "sb_rotor":    [],
        "rotor_right": [],
        "rotor_left":  [],
        "n_nodes":     0,
        "n_elems":     0,
    }

    if check_only:
        gmsh.finalize()
        return result

    # ── 8. Physical groups — surfaces ─────────────────────────────────────────
    def pg(dim, tags, name):
        if tags:
            model.addPhysicalGroup(dim, tags, name=name)
        else:
            print(f"  SKIP empty group: {name}")

    pg(2, rotor_iron_tags,  "Rotor_Iron")
    pg(2, magnet_tags,      "Magnet")
    pg(2, air_pocket_tags,  "AirPocket")
    pg(2, shaft_tags,       "Shaft")
    pg(2, gap_r_tags,       "Airgap_Rotor")

    # ── 9. Physical groups — boundary curves ──────────────────────────────────
    iron_curves  = _get_bound_curves(model, rotor_iron_tags)
    mag_curves   = _get_bound_curves(model, magnet_tags)
    pkt_curves   = _get_bound_curves(model, air_pocket_tags)
    shaft_curves = _get_bound_curves(model, shaft_tags)
    gap_curves   = _get_bound_curves(model, gap_r_tags)

    all_rotor_curves = iron_curves | mag_curves | pkt_curves | shaft_curves | gap_curves

    # SB_Rotor: outer arc of gap_r at R_sb — longest arc (full 45°)
    gap_arcs = {c: _curve_rc(occ, c) for c in gap_curves
                if _curve_rc(occ, c) < p.R_si}
    gap_arc_len = {c: _curve_length(occ, c) for c in gap_arcs}
    sb_rotor = [max(gap_arc_len, key=gap_arc_len.get)] if gap_arc_len else []

    excl = set(sb_rotor)
    ro_right, ro_left = _classify_sector_bounds(model, all_rotor_curves, excl)

    pg(1, sb_rotor,  "SB_Rotor")
    pg(1, ro_right,  "Rotor_Right")
    pg(1, ro_left,   "Rotor_Left")

    print(f"\nBoundaries:")
    print(f"  SB_Rotor     : {sb_rotor}")
    print(f"  Rotor_Right  : {len(ro_right)} curves")
    print(f"  Rotor_Left   : {len(ro_left)} curves")

    # ── 10. Mesh size (point-based) ───────────────────────────────────────────
    for _, pt in model.getEntities(0):
        x, y, _, _, _, _ = model.getBoundingBox(0, pt)
        r = math.hypot(x, y)
        if r > p.R_sb - 0.5:
            sz = p.lc_gap
        elif r > p.R_ro - 0.5:
            sz = p.lc_ro
        elif r > p.R_mi - 0.5:
            sz = p.lc_mag
        elif r > p.R_ri + 0.5:
            sz = p.lc_ro
        else:
            sz = p.lc_ri
        model.mesh.setSize([(0, pt)], sz)

    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 1)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints",         1)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature",      0)
    gmsh.option.setNumber("Mesh.Algorithm", 6)

    # ── 11. Generate mesh ─────────────────────────────────────────────────────
    print("\nGenerating rotor mesh ...")
    model.mesh.generate(2)
    model.mesh.optimize("Netgen")

    n_nodes = len(model.mesh.getNodes()[0])
    n_elems = sum(len(model.mesh.getElements(2, t)[1][0])
                  for _, t in model.getEntities(2))
    print(f"Mesh: {n_nodes} nodes, {n_elems} 2D elements")

    if mesh_out is not None:
        mesh_out = Path(mesh_out)
        mesh_out.parent.mkdir(parents=True, exist_ok=True)
        gmsh.write(str(mesh_out))
        print(f"Written: {mesh_out}")

    if gui:
        gmsh.fltk.run()

    gmsh.finalize()

    result.update({
        "sb_rotor":    sb_rotor,
        "rotor_right": ro_right,
        "rotor_left":  ro_left,
        "n_nodes":     n_nodes,
        "n_elems":     n_elems,
    })
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  Standalone runner
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    gui        = "--gui"   in sys.argv
    check_only = "--check" in sys.argv

    p = DEFAULT_PARAMS
    print(p.summary())

    out = Path(__file__).parent / "mesh_rotor" / "rotor.msh"
    result = build_rotor(p, mesh_out=out, gui=gui, check_only=check_only)

    if result:
        print(f"\nMagnet area : {result['A_mag_mm2']:.1f} mm²")
