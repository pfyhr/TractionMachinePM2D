"""
gen_mesh.py — Full 45° sector mesh (stator + rotor, non-conforming SB).

Combines the stator and rotor sectors into one gmsh mesh with:
  - Non-conforming sliding boundary at R_sb (SB_Stator / SB_Rotor)
    required for Elmer's Mortar boundary condition.
  - All physical groups: surfaces (bodies) + curves (boundaries).
  - Optional ElmerGrid conversion to Elmer native format.

Usage:
    python3 gen_mesh.py [--gui] [--out DIR] [--elmergrid] [--check]
"""
from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path
from typing import Optional

import gmsh

from motor_params import MotorParams, DEFAULT_PARAMS
from gen_rotor import _make_magnet_and_pockets

π = math.pi
cos, sin = math.cos, math.sin

# Phase assignment for 48s/8p/3ph, 6 slots per sector
_PHASE_MAP = ["A+", "A+", "C-", "C-", "B+", "B+"]


# ─────────────────────────────────────────────────────────────────────────────
#  Geometry primitives (local copies — same as gen_stator/gen_rotor)
# ─────────────────────────────────────────────────────────────────────────────
def _annular_sector(occ, r1, r2, t1, t2, lc):
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


def _pie_sector(occ, r, t1, t2, lc):
    O  = occ.addPoint(0, 0, 0, lc)
    p1 = occ.addPoint(r*cos(t1), r*sin(t1), 0, lc)
    p2 = occ.addPoint(r*cos(t2), r*sin(t2), 0, lc)
    l1 = occ.addLine(O, p1)
    a  = occ.addCircleArc(p1, O, p2)
    l2 = occ.addLine(p2, O)
    return occ.addPlaneSurface([occ.addCurveLoop([l1, a, l2])])


def _get_rc(occ, tag):
    x, y, _ = occ.getCenterOfMass(2, tag)
    return math.hypot(x, y), math.atan2(y, x), occ.getMass(2, tag)


def _get_bound_curves(model, surf_tags):
    curves: set[int] = set()
    for st in surf_tags:
        _, down = model.getAdjacencies(2, st)
        curves.update(abs(int(c)) for c in down)
    return curves


def _curve_rc(occ, ctag):
    x, y, _ = occ.getCenterOfMass(1, ctag)
    return math.hypot(x, y)


def _curve_length(occ, ctag):
    return occ.getMass(1, ctag)


def _classify_sector_bounds(model, curve_set, excl):
    right, left = [], []
    for c in sorted(curve_set - excl):
        x1, y1, _, x2, y2, _ = model.getBoundingBox(1, c)
        xm, ym = (x1 + x2) / 2, (y1 + y2) / 2
        if math.hypot(xm, ym) < 0.5:
            continue
        if abs(y1) < 0.5 and abs(y2) < 0.5 and xm > 1.0:
            right.append(c)
        elif abs(x1-y1) < 0.5 and abs(x2-y2) < 0.5 and xm > 1.0 and ym > 1.0:
            left.append(c)
    return right, left


def _r_centroid_annular(R_in, R_out, half_angle):
    """Centroid radius of an annular sector (R_in→R_out, ±half_angle)."""
    fc  = math.sin(half_angle) / half_angle if half_angle > 1e-9 else 1.0
    r_m = 2/3 * (R_out**3 - R_in**3) / (R_out**2 - R_in**2)
    return fc * r_m


# ─────────────────────────────────────────────────────────────────────────────
#  Main mesh builder
# ─────────────────────────────────────────────────────────────────────────────
def build_mesh(
    params: MotorParams,
    mesh_out: Optional[Path] = None,
    elmer_dir: Optional[Path] = None,
    *,
    gui: bool = False,
    check_only: bool = False,
) -> dict:
    """
    Build the full 45° sector mesh (stator + rotor, non-conforming SB).

    Parameters
    ----------
    mesh_out  : path to write the .msh file (optional)
    elmer_dir : directory for ElmerGrid output (requires ElmerGrid in PATH)
    gui       : open gmsh GUI after meshing
    check_only: classify surfaces without meshing (fast)

    Returns a dict with all surface/curve tags and mesh statistics.
    """
    params.validate()
    p = params

    gmsh.initialize()
    gmsh.model.add("Motor_Sector")
    occ   = gmsh.model.occ
    model = gmsh.model

    # ═══════════════════════════════════════════════════════════════════════
    #  STATOR GEOMETRY
    # ═══════════════════════════════════════════════════════════════════════

    # ── 1. Stator iron annulus (R_si → R_so) ─────────────────────────────
    s_stator = _annular_sector(occ, p.R_si, p.R_so, 0, p.θs, p.lc_si)

    # ── 2. Stator airgap sector (R_sb → R_si, fixed side) ────────────────
    s_gap_s  = _annular_sector(occ, p.R_sb, p.R_si, 0, p.θs, p.lc_gap)

    # ── 3. Slots: opening + body + hairpins ───────────────────────────────
    all_openings: list[int] = []
    all_bodies:   list[int] = []
    all_hps:      list[int] = []

    for i in range(p.ns):
        θ_c = (i + 0.5) * p.sp

        t_op = occ.addRectangle(p.R_si, -p.b1/2, 0, p.h1, p.b1)
        occ.rotate([(2, t_op)], 0, 0, 0, 0, 0, 1, θ_c)
        all_openings.append(t_op)

        t_bd = occ.addRectangle(p.R_si + p.h1, -p.b_slot/2, 0, p.h_slot, p.b_slot)
        occ.rotate([(2, t_bd)], 0, 0, 0, 0, 0, 1, θ_c)
        all_bodies.append(t_bd)

        for k in range(p.n_hp):
            r_bot = p.R_si + p.h1 + p.t_liner + k * p.h_layer + p.t_enam
            t_hp = occ.addRectangle(r_bot, -p.b_hp/2, 0, p.h_hp, p.b_hp)
            occ.rotate([(2, t_hp)], 0, 0, 0, 0, 0, 1, θ_c)
            all_hps.append(t_hp)

    occ.synchronize()

    # ── 4. Fragment stator side (SB arc at R_sb stays free) ───────────────
    stator_frags = (
        [(2, s_stator), (2, s_gap_s)]
        + [(2, t) for t in all_openings]
        + [(2, t) for t in all_bodies]
        + [(2, t) for t in all_hps]
    )
    occ.fragment(stator_frags, [])
    occ.synchronize()

    # ═══════════════════════════════════════════════════════════════════════
    #  ROTOR GEOMETRY
    # ═══════════════════════════════════════════════════════════════════════

    # ── 5. Rotor airgap (R_ro → R_sb, rotor side) ────────────────────────
    s_gap_r = _annular_sector(occ, p.R_ro, p.R_sb, 0, p.θs, p.lc_gap)

    # ── 6. Rotor iron (R_ri → R_ro, full sector) ─────────────────────────
    s_rotor = _annular_sector(occ, p.R_ri, p.R_ro, 0, p.θs, p.lc_ro)

    # ── 7. Rectangular magnet + air pockets (clipped to R_ro) ────────────
    s_mag, s_pockets = _make_magnet_and_pockets(occ, p)

    # ── 8. Shaft (0 → R_ri) ──────────────────────────────────────────────
    s_shaft = _pie_sector(occ, p.R_ri, 0, p.θs, p.lc_ri)

    # ── 9. Fragment rotor side (SB arc at R_sb stays free) ────────────────
    rotor_frags = (
        [(2, s_gap_r), (2, s_rotor), (2, s_shaft), (2, s_mag)]
        + [(2, t) for t in s_pockets]
    )
    occ.fragment(rotor_frags, [])
    occ.synchronize()

    # ═══════════════════════════════════════════════════════════════════════
    #  CLASSIFY SURFACES
    # ═══════════════════════════════════════════════════════════════════════
    all_surfs = [t for _, t in model.getEntities(2)]

    stator_iron_tags: list[int]             = []
    gap_s_tags:       list[int]             = []
    slot_open_tags:   list[list[int]]       = [[] for _ in range(p.ns)]
    slot_insul_tags:  list[list[int]]       = [[] for _ in range(p.ns)]
    slot_hp_tags:     list[list[list[int]]] = [[[] for _ in range(p.n_hp)]
                                                for _ in range(p.ns)]
    gap_r_tags:       list[int]             = []
    rotor_iron_tags:  list[int]             = []
    magnet_tags:      list[int]             = []
    air_pocket_tags:  list[int]             = []
    shaft_tags:       list[int]             = []

    # Expected thin-annulus areas (mm²) for the two airgap layers
    A_gap_s_exp = 0.5 * (p.R_si**2 - p.R_sb**2) * p.θs
    A_gap_r_exp = 0.5 * (p.R_sb**2 - p.R_ro**2) * p.θs
    # Both areas are similar; use centroid radius to disambiguate
    r_c_gap_s = _r_centroid_annular(p.R_sb, p.R_si, p.θs / 2)
    r_c_gap_r = _r_centroid_annular(p.R_ro, p.R_sb, p.θs / 2)
    r_divider  = (r_c_gap_s + r_c_gap_r) / 2   # ≈ 71.75 mm

    # Stator slot region bounds
    R_slot_top = p.R_si + p.h1 + p.h_slot + 0.3

    # Rotor magnet/pocket reference values
    θ_c_mag  = p.θs / 2
    r_mag_c  = p.R_mi + p.h_m / 2
    half_mag = math.atan2(p.w_mag / 2, r_mag_c)
    half_pkt = half_mag + math.atan2(max(p.w_air, 0.01), r_mag_c)

    for tag in all_surfs:
        r, θ, A = _get_rc(occ, tag)
        if θ < 0:
            θ += 2 * π

        # ── Airgap (stator or rotor): area-based, then discriminate by r ─
        if abs(A - A_gap_s_exp) / A_gap_s_exp < 0.20:
            if r > r_divider:
                gap_s_tags.append(tag)
            else:
                gap_r_tags.append(tag)
            continue

        # ── Shaft ─────────────────────────────────────────────────────────
        if r < p.R_ri + 0.5:
            shaft_tags.append(tag)
            continue

        # ── Stator iron: above slot top, or large area in stator zone ─────
        if r > R_slot_top or (A > 500 and r > p.R_si):
            stator_iron_tags.append(tag)
            continue

        # ── Stator slot zone (bore side) ──────────────────────────────────
        if r > p.R_si - 0.5:
            slot_idx = -1
            for i in range(p.ns):
                θ_ci = (i + 0.5) * p.sp
                if abs(θ - θ_ci) < p.sp * 0.55:
                    slot_idx = i
                    break
            if slot_idx < 0:
                stator_iron_tags.append(tag)   # half-tooth at sector edge
                continue

            # Slot opening: tiny area near bore
            if r < p.R_si + p.h1 + 0.3 and A < 3.0:
                slot_open_tags[slot_idx].append(tag)
                continue

            # Hairpin: area within 10% of A_hp, at correct radial layer
            if p.A_hp > 0 and abs(A - p.A_hp) / p.A_hp < 0.10:
                for k in range(p.n_hp):
                    r_exp = p.R_si + p.h1 + p.t_liner + (k + 0.5) * p.h_layer
                    if abs(r - r_exp) < p.h_layer * 0.45:
                        slot_hp_tags[slot_idx][k].append(tag)
                        break
                else:
                    slot_insul_tags[slot_idx].append(tag)
                continue

            # Everything else in slot angular range → insulation / liner
            slot_insul_tags[slot_idx].append(tag)
            continue

        # ── Rotor magnet ──────────────────────────────────────────────────
        in_mag_r = (p.R_mi - 1.0) < r < (p.R_ro + 0.5)
        in_mag_θ = abs(θ - θ_c_mag) < half_mag * 1.2
        if in_mag_r and in_mag_θ and A > 20.0:
            magnet_tags.append(tag)
            continue

        # ── Air pocket ────────────────────────────────────────────────────
        in_pkt_r = (p.R_mi - 1.0) < r < (p.R_ro + 0.5)
        in_pkt_θ = half_mag * 0.8 < abs(θ - θ_c_mag) < half_pkt * 1.5
        if in_pkt_r and in_pkt_θ and p.w_air > 0 and A > 0.01:
            air_pocket_tags.append(tag)
            continue

        # ── Rotor iron (catch-all for rotor region) ────────────────────────
        rotor_iron_tags.append(tag)

    A_mag_actual = (
        sum(_get_rc(occ, t)[2] for t in magnet_tags) if magnet_tags else 0.0
    )

    # ── Classification summary ─────────────────────────────────────────────
    print(f"\nCombined mesh classification ({len(all_surfs)} total surfaces):")
    print(f"  Stator iron    : {len(stator_iron_tags)}")
    print(f"  Airgap stator  : {len(gap_s_tags)}")
    for i in range(p.ns):
        n_hp_found = sum(len(slot_hp_tags[i][k]) for k in range(p.n_hp))
        print(f"  Slot {i} ({_PHASE_MAP[i]:2s}):  "
              f"open={len(slot_open_tags[i])}, "
              f"HP={n_hp_found}/{p.n_hp}, "
              f"insul={len(slot_insul_tags[i])}")
    print(f"  Airgap rotor   : {len(gap_r_tags)}")
    print(f"  Rotor iron     : {len(rotor_iron_tags)}")
    print(f"  Magnet         : {len(magnet_tags)}  "
          f"(A = {A_mag_actual:.1f} mm² actual, {p.A_mag:.1f} mm² rect)")
    print(f"  Air pockets    : {len(air_pocket_tags)}  "
          f"(expected {len(s_pockets)})")
    print(f"  Shaft          : {len(shaft_tags)}")

    for label, lst, exp in [
        ("Airgap stator", gap_s_tags,   1),
        ("Airgap rotor",  gap_r_tags,   1),
        ("Magnet",        magnet_tags,  1),
        ("Shaft",         shaft_tags,   1),
    ]:
        if len(lst) != exp:
            print(f"  WARNING: {label} has {len(lst)} (expected {exp})")
    for i in range(p.ns):
        n_hp_found = sum(len(slot_hp_tags[i][k]) for k in range(p.n_hp))
        if n_hp_found != p.n_hp:
            print(f"  WARNING: slot {i} HP count {n_hp_found} (expected {p.n_hp})")

    result: dict = {
        "stator_iron":  stator_iron_tags,
        "gap_stator":   gap_s_tags,
        "slot_opening": slot_open_tags,
        "slot_insul":   slot_insul_tags,
        "slot_hp":      slot_hp_tags,
        "gap_rotor":    gap_r_tags,
        "rotor_iron":   rotor_iron_tags,
        "magnet":       magnet_tags,
        "air_pocket":   air_pocket_tags,
        "shaft":        shaft_tags,
        "A_mag_mm2":    A_mag_actual,
        "Carea_m2":     p.Carea,
        "fill_factor":  p.fill_factor,
        "sb_stator":    [],
        "sb_rotor":     [],
        "domain":       [],
        "stator_right": [],
        "stator_left":  [],
        "rotor_right":  [],
        "rotor_left":   [],
        "n_nodes":      0,
        "n_elems":      0,
    }

    if check_only:
        gmsh.finalize()
        return result

    # ═══════════════════════════════════════════════════════════════════════
    #  PHYSICAL GROUPS
    # ═══════════════════════════════════════════════════════════════════════
    def pg(dim, tags, name):
        if tags:
            model.addPhysicalGroup(dim, tags, name=name)
        else:
            print(f"  SKIP empty group: {name}")

    # Surface bodies
    pg(2, stator_iron_tags, "Stator_Iron")
    pg(2, gap_s_tags,       "Airgap_Stator")
    for i in range(p.ns):
        ph = _PHASE_MAP[i]
        pg(2, slot_open_tags[i],  f"S{i}_Opening")
        pg(2, slot_insul_tags[i], f"S{i}_Insul")
        for k in range(p.n_hp):
            pg(2, slot_hp_tags[i][k], f"S{i}_HP{k}_{ph}")
    pg(2, gap_r_tags,       "Airgap_Rotor")
    pg(2, rotor_iron_tags,  "Rotor_Iron")
    pg(2, magnet_tags,      "Magnet")
    pg(2, air_pocket_tags,  "AirPocket")
    pg(2, shaft_tags,       "Shaft")

    # Boundary curves
    iron_curves  = _get_bound_curves(model, stator_iron_tags)
    slot_all_surf = (
        [t for sl in slot_open_tags  for t in sl]
        + [t for sl in slot_insul_tags for t in sl]
        + [t for sl in slot_hp_tags  for lay in sl for t in lay]
    )
    slot_curves  = _get_bound_curves(model, slot_all_surf)
    gap_s_curves = _get_bound_curves(model, gap_s_tags)
    gap_r_curves = _get_bound_curves(model, gap_r_tags)
    iron_r_curves = _get_bound_curves(model, rotor_iron_tags)
    mag_curves   = _get_bound_curves(model, magnet_tags)
    pkt_curves   = _get_bound_curves(model, air_pocket_tags)
    shaft_curves = _get_bound_curves(model, shaft_tags)

    # Domain: outer stator arcs at R_so
    domain_tags: list[int] = []
    for c in iron_curves:
        x1, y1, _, x2, y2, _ = model.getBoundingBox(1, c)
        if math.hypot((x1+x2)/2, (y1+y2)/2) > 100:
            domain_tags.append(c)

    # SB_Stator: inner arc of stator airgap (minimum r_c, at R_sb)
    gap_s_arcs = {c: _curve_rc(occ, c) for c in gap_s_curves
                  if _curve_rc(occ, c) < p.R_si - 0.1}
    sb_stator = [min(gap_s_arcs, key=gap_s_arcs.get)] if gap_s_arcs else []

    # SB_Rotor: outer arc of rotor airgap (maximum arc length = full 45° arc)
    gap_r_arcs    = {c: _curve_rc(occ, c) for c in gap_r_curves
                     if _curve_rc(occ, c) < p.R_si}
    gap_r_arc_len = {c: _curve_length(occ, c) for c in gap_r_arcs}
    sb_rotor = [max(gap_r_arc_len, key=gap_r_arc_len.get)] if gap_r_arc_len else []

    # Sector boundaries: radial lines at θ=0 and θ=45°
    excl = set(domain_tags) | set(sb_stator) | set(sb_rotor)
    all_stator_curves = iron_curves | slot_curves | gap_s_curves
    all_rotor_curves  = (iron_r_curves | gap_r_curves | mag_curves
                         | pkt_curves | shaft_curves)
    st_right, st_left = _classify_sector_bounds(model, all_stator_curves, excl)
    ro_right, ro_left = _classify_sector_bounds(model, all_rotor_curves, excl)

    pg(1, domain_tags, "Domain")
    pg(1, sb_stator,   "SB_Stator")
    pg(1, sb_rotor,    "SB_Rotor")
    pg(1, st_right,    "Stator_Right")
    pg(1, st_left,     "Stator_Left")
    pg(1, ro_right,    "Rotor_Right")
    pg(1, ro_left,     "Rotor_Left")

    print(f"\nBoundaries:")
    print(f"  Domain        : {len(domain_tags)} curve(s)")
    print(f"  SB_Stator     : {sb_stator}")
    print(f"  SB_Rotor      : {sb_rotor}")
    print(f"  Stator right/left : {len(st_right)}/{len(st_left)}")
    print(f"  Rotor  right/left : {len(ro_right)}/{len(ro_left)}")
    if set(sb_stator) & set(sb_rotor):
        print("  WARNING: SB_Stator and SB_Rotor share curves — mesh is conforming!")

    # ═══════════════════════════════════════════════════════════════════════
    #  MESH SIZES (point-based)
    # ═══════════════════════════════════════════════════════════════════════
    for _, pt in model.getEntities(0):
        x, y, _, _, _, _ = model.getBoundingBox(0, pt)
        r = math.hypot(x, y)
        if r > p.R_so - 2:
            sz = p.lc_so
        elif r > p.R_si + p.h1 + p.h_slot:
            sz = p.lc_si                              # back iron
        elif r > p.R_si + p.h1 + p.t_liner:
            sz = p.lc_hp                              # hairpin zone
        elif r > p.R_si + p.h1:
            sz = p.lc_ins                             # liner / tooth-tip
        elif r > p.R_si:
            sz = p.lc_si                              # bore / tooth face
        elif r > p.R_sb:
            sz = p.lc_gap                             # stator airgap
        elif r > p.R_ro - 0.5:
            sz = p.lc_gap                             # near SB (rotor side)
        elif r > p.R_mi - 0.5:
            sz = p.lc_mag                             # magnet region
        elif r > p.R_ri + 0.5:
            sz = p.lc_ro                              # rotor iron
        else:
            sz = p.lc_ri                              # shaft
        model.mesh.setSize([(0, pt)], sz)

    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 1)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints",         1)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature",      0)
    gmsh.option.setNumber("Mesh.Algorithm", 6)   # Frontal-Delaunay

    # ═══════════════════════════════════════════════════════════════════════
    #  GENERATE MESH
    # ═══════════════════════════════════════════════════════════════════════
    print("\nGenerating combined mesh ...")
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
        "sb_stator":    sb_stator,
        "sb_rotor":     sb_rotor,
        "domain":       domain_tags,
        "stator_right": st_right,
        "stator_left":  st_left,
        "rotor_right":  ro_right,
        "rotor_left":   ro_left,
        "n_nodes":      n_nodes,
        "n_elems":      n_elems,
    })

    # ── Optional ElmerGrid conversion ──────────────────────────────────────
    if elmer_dir is not None and mesh_out is not None:
        elmer_dir = Path(elmer_dir)
        elmer_dir.mkdir(parents=True, exist_ok=True)
        cmd = ["ElmerGrid", "14", "2", str(mesh_out),
               "-autoclean", "-out", str(elmer_dir)]
        print("\nRunning ElmerGrid ...")
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            print("ElmerGrid FAILED:")
            print(proc.stderr[-2000:])
        else:
            for line in proc.stdout.splitlines():
                if any(k in line for k in ["nodes", "elements", "bodies",
                                            "boundary", "Error"]):
                    print(" ", line)
            print("ElmerGrid done.")
        result["elmer_dir"] = str(elmer_dir)

    return result


# ─────────────────────────────────────────────────────────────────────────────
#  Standalone runner
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    gui        = "--gui"       in sys.argv
    check_only = "--check"     in sys.argv
    elmergrid  = "--elmergrid" in sys.argv

    # Parse optional --out DIR argument
    out_dir = Path(__file__).parent / "mesh"
    for i, arg in enumerate(sys.argv):
        if arg == "--out" and i + 1 < len(sys.argv):
            out_dir = Path(sys.argv[i + 1])

    p = DEFAULT_PARAMS
    print(p.summary())

    mesh_out  = out_dir / "motor.msh"
    elmer_dir = out_dir if elmergrid else None

    result = build_mesh(p, mesh_out=mesh_out, elmer_dir=elmer_dir,
                        gui=gui, check_only=check_only)

    print(f"\nFill factor : {result['fill_factor']*100:.1f} %")
    print(f"Cu/slot     : {result['Carea_m2']*1e6:.1f} mm²")
    print(f"Magnet area : {result['A_mag_mm2']:.1f} mm²")
