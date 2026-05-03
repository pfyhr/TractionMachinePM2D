"""
gen_stator.py — Stator sector geometry with rectangular slots and hairpin conductors.

Generates a 45° (one pole pitch) stator sector with:
  - Rectangular slot bodies (constant tangential width, not sector-shaped)
  - Individual hairpin conductor rectangles stacked radially in each slot
  - Slot liner insulation and enamel gaps modelled as air
  - Stator airgap region (fixed side, R_sb → R_si)

Usage:
    python3 gen_stator.py [--gui] [--check]
    --gui    open gmsh GUI after meshing
    --check  print classification summary only, no mesh (faster)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Optional

import gmsh

from motor_params import MotorParams, DEFAULT_PARAMS
from winding_config import winding_phase_map

π = math.pi
cos, sin = math.cos, math.sin


# ─────────────────────────────────────────────────────────────────────────────
#  Phase assignment for a 48s/8p/3ph machine (one 45° sector, 6 slots)
#  Pattern repeats every pole pitch.  Winding: A+ A+ C- C- B+ B+
# ─────────────────────────────────────────────────────────────────────────────
_PHASE_MAP = ["A+", "A+", "C-", "C-", "B+", "B+"]


def _annular_sector(occ, r1: float, r2: float, t1: float, t2: float,
                    lc: float) -> int:
    """Return tag of an annular sector surface r1→r2, t1→t2 (CCW)."""
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


def _get_rc(occ, tag: int) -> tuple[float, float, float]:
    """Return (r, theta, area) centroid of surface tag."""
    x, y, _ = occ.getCenterOfMass(2, tag)
    return math.hypot(x, y), math.atan2(y, x), occ.getMass(2, tag)


def _get_bound_curves(model, surf_tags: list[int]) -> set[int]:
    """Return set of curve tags bounding a list of surfaces."""
    curves: set[int] = set()
    for st in surf_tags:
        _, down = model.getAdjacencies(2, st)
        curves.update(abs(int(c)) for c in down)
    return curves


def _curve_rc(occ, ctag: int) -> float:
    x, y, _ = occ.getCenterOfMass(1, ctag)
    return math.hypot(x, y)


def _classify_sector_bounds(
    model, curve_set: set[int], excl: set[int], theta_s: float
) -> tuple[list[int], list[int]]:
    """Split curves into right (θ=0) and left (θ=θs) sector boundaries.

    Uses the cross-product formula |x·sin(θs) − y·cos(θs)| < tol for the
    left boundary, generalising the hardcoded x≈y check (θs=45°) to any
    pole count.
    """
    sin_s = math.sin(theta_s)
    cos_s = math.cos(theta_s)
    right, left = [], []
    for c in sorted(curve_set - excl):
        x1, y1, _, x2, y2, _ = model.getBoundingBox(1, c)
        xm, ym = (x1 + x2) / 2, (y1 + y2) / 2
        if math.hypot(xm, ym) < 0.5:
            continue
        if abs(y1) < 0.5 and abs(y2) < 0.5 and xm > 1.0:
            right.append(c)
            continue
        cross1 = abs(x1 * sin_s - y1 * cos_s)
        cross2 = abs(x2 * sin_s - y2 * cos_s)
        dot    = xm * cos_s + ym * sin_s
        if cross1 < 0.5 and cross2 < 0.5 and dot > 1.0:
            left.append(c)
    return right, left


# ─────────────────────────────────────────────────────────────────────────────
#  Main geometry builder
# ─────────────────────────────────────────────────────────────────────────────
def build_stator(
    params: MotorParams,
    mesh_out: Optional[Path] = None,
    *,
    gui: bool = False,
    check_only: bool = False,
) -> dict:
    """
    Build the stator sector geometry, mesh it, and (optionally) export.

    Returns a dict with:
        stator_iron    : list[int]   — surface tags
        gap_stator     : list[int]
        slot_opening   : list[list[int]]  — [slot_i]
        slot_insul     : list[list[int]]  — [slot_i]
        slot_hp        : list[list[list[int]]]  — [slot_i][layer_k]
        sb_stator      : list[int]   — boundary curve tags (SB inner arc)
        domain         : list[int]   — boundary curve tags (outer arc)
        stator_right   : list[int]
        stator_left    : list[int]
        Carea_m2       : float       — copper area per slot [m²]
        fill_factor    : float
    """
    params.validate()
    p = params

    gmsh.initialize()
    gmsh.model.add("Stator")
    occ   = gmsh.model.occ
    model = gmsh.model

    # ── 1. Stator annulus sector ──────────────────────────────────────────────
    s_stator = _annular_sector(occ, p.R_si, p.R_so, 0, p.θs, p.lc_si)

    # ── 2. Stator airgap sector (fixed side, R_sb → R_si) ────────────────────
    s_gap_s = _annular_sector(occ, p.R_sb, p.R_si, 0, p.θs, p.lc_gap)

    # ── 3. Slots and hairpins ─────────────────────────────────────────────────
    all_openings: list[int] = []
    all_bodies:   list[int] = []
    all_hps:      list[int] = []   # flat — for fragment call

    for i in range(p.ns):
        θ_c = (i + 0.5) * p.sp

        # Slot opening: narrow rectangle at bore, rotated to slot centre
        t_op = occ.addRectangle(p.R_si, -p.b1/2, 0, p.h1, p.b1)
        occ.rotate([(2, t_op)], 0, 0, 0, 0, 0, 1, θ_c)
        all_openings.append(t_op)

        # Slot body: full-width rectangle above opening
        t_bd = occ.addRectangle(p.R_si + p.h1, -p.b_slot/2, 0, p.h_slot, p.b_slot)
        occ.rotate([(2, t_bd)], 0, 0, 0, 0, 0, 1, θ_c)
        all_bodies.append(t_bd)

        # Hairpin conductors — stacked radially inside slot body
        for k in range(p.n_hp):
            r_bot = p.R_si + p.h1 + p.t_liner + k * p.h_layer + p.t_enam
            t_hp = occ.addRectangle(r_bot, -p.b_hp/2, 0, p.h_hp, p.b_hp)
            occ.rotate([(2, t_hp)], 0, 0, 0, 0, 0, 1, θ_c)
            all_hps.append(t_hp)

    occ.synchronize()

    # ── 4. Fragment (stator side only — SB stays non-conforming) ─────────────
    frags = (
        [(2, s_stator), (2, s_gap_s)]
        + [(2, t) for t in all_openings]
        + [(2, t) for t in all_bodies]
        + [(2, t) for t in all_hps]
    )
    _, mapping = occ.fragment(frags, [])
    occ.synchronize()

    # ── 5. Classify surfaces via fragment parent-map ─────────────────────────
    # frags order: [0]=s_stator, [1]=s_gap_s, [2..ns+1]=openings,
    #              [ns+2..2*ns+1]=bodies, [2*ns+2..end]=hps (i*n_hp+k order)
    all_surfs_set = {t for _, t in model.getEntities(2)}
    all_surfs     = list(all_surfs_set)

    def _unique_from(tags_iter) -> list[int]:
        seen: set[int] = set()
        out:  list[int] = []
        for t in tags_iter:
            if t in all_surfs_set and t not in seen:
                seen.add(t)
                out.append(t)
        return out

    gap_s_tags:       list[int]             = _unique_from(
        t for _, t in mapping[1]
    )
    slot_open_tags:   list[list[int]]       = [[] for _ in range(p.ns)]
    slot_hp_tags:     list[list[list[int]]] = [[[] for _ in range(p.n_hp)]
                                                for _ in range(p.ns)]
    slot_insul_tags:  list[list[int]]       = [[] for _ in range(p.ns)]

    for i in range(p.ns):
        slot_open_tags[i] = _unique_from(t for _, t in mapping[2 + i])
        for k in range(p.n_hp):
            frag_idx = 2 + p.ns + p.ns + i * p.n_hp + k
            slot_hp_tags[i][k] = _unique_from(t for _, t in mapping[frag_idx])

    # Insulation for slot i = slot-body surfaces minus its openings and HPs
    for i in range(p.ns):
        body_surfs = _unique_from(t for _, t in mapping[2 + p.ns + i])
        slot_surfs = (set(slot_open_tags[i])
                      | {t for k in range(p.n_hp) for t in slot_hp_tags[i][k]})
        slot_insul_tags[i] = [t for t in body_surfs if t not in slot_surfs]

    # Stator iron = everything not classified above
    classified: set[int] = set(gap_s_tags)
    for i in range(p.ns):
        classified.update(slot_open_tags[i])
        classified.update(slot_insul_tags[i])
        for k in range(p.n_hp):
            classified.update(slot_hp_tags[i][k])
    stator_iron_tags: list[int] = [t for t in all_surfs if t not in classified]

    # ── 6. Sanity check and summary ──────────────────────────────────────────
    _phase_map = winding_phase_map(p.ns)
    print(f"\nStator classification ({len(all_surfs)} total surfaces):")
    print(f"  Stator iron  : {len(stator_iron_tags)} surface(s)")
    print(f"  Airgap       : {len(gap_s_tags)} surface(s)")
    for i in range(p.ns):
        n_hp_found = sum(len(slot_hp_tags[i][k]) for k in range(p.n_hp))
        print(f"  Slot {i} ({_phase_map[i]:2s}): "
              f"opening={len(slot_open_tags[i])}, "
              f"HP={n_hp_found}/{p.n_hp}, "
              f"insul={len(slot_insul_tags[i])}")

    # Warn if any slot has wrong HP count
    for i in range(p.ns):
        n_hp_found = sum(len(slot_hp_tags[i][k]) for k in range(p.n_hp))
        if n_hp_found != p.n_hp:
            print(f"  WARNING: slot {i} has {n_hp_found} HP surfaces "
                  f"(expected {p.n_hp})")

    # Always build the return dict (check_only skips meshing but keeps data)
    result = {
        "stator_iron":  stator_iron_tags,
        "gap_stator":   gap_s_tags,
        "slot_opening": slot_open_tags,
        "slot_insul":   slot_insul_tags,
        "slot_hp":      slot_hp_tags,
        "Carea_m2":     p.Carea,
        "fill_factor":  p.fill_factor,
        # boundaries filled in below unless check_only
        "sb_stator":    [],
        "domain":       [],
        "stator_right": [],
        "stator_left":  [],
        "n_nodes":      0,
        "n_elems":      0,
    }

    if check_only:
        gmsh.finalize()
        return result

    # ── 7. Physical groups — surfaces ────────────────────────────────────────
    def pg(dim, tags, name):
        if tags:
            model.addPhysicalGroup(dim, tags, name=name)
        else:
            print(f"  SKIP empty group: {name}")

    pg(2, stator_iron_tags, "Stator_Iron")
    pg(2, gap_s_tags,       "Airgap_Stator")
    _phase_map = winding_phase_map(p.ns)
    for i in range(p.ns):
        ph = _phase_map[i]
        pg(2, slot_open_tags[i],  f"S{i}_Opening")
        pg(2, slot_insul_tags[i], f"S{i}_Insul")
        for k in range(p.n_hp):
            pg(2, slot_hp_tags[i][k], f"S{i}_HP{k}_{ph}")

    # ── 8. Physical groups — boundary curves ─────────────────────────────────
    iron_curves  = _get_bound_curves(model, stator_iron_tags)
    slot_all_surf = ([t for sl in slot_open_tags for t in sl]
                     + [t for sl in slot_insul_tags for t in sl]
                     + [t for sl in slot_hp_tags for lay in sl for t in lay])
    slot_curves  = _get_bound_curves(model, slot_all_surf)
    gap_curves   = _get_bound_curves(model, gap_s_tags)

    # Outer arc (Domain): the arc at R_so.
    # Identified by: x_max ≈ R_so (the arc passes through θ=0) AND the curve
    # spans a non-trivial y range (distinguishes it from radial boundary lines).
    domain_tags = []
    for c in iron_curves:
        x1, y1, _, x2, y2, _ = model.getBoundingBox(1, c)
        if x2 > p.R_so - 0.5 and (y2 - y1) > 1.0:
            domain_tags.append(c)

    # SB_Stator: inner arc of airgap sector — minimum r_c among gap arcs < 73 mm
    gap_arcs = {c: _curve_rc(occ, c) for c in gap_curves
                if _curve_rc(occ, c) < p.R_si - 0.1}
    sb_stator = [min(gap_arcs, key=gap_arcs.get)] if gap_arcs else []

    # Sector boundaries
    excl = set(domain_tags) | set(sb_stator)
    all_stator_curves = iron_curves | slot_curves | gap_curves
    st_right, st_left = _classify_sector_bounds(model, all_stator_curves, excl, p.θs)

    pg(1, domain_tags, "Domain")
    pg(1, sb_stator,   "SB_Stator")
    pg(1, st_right,    "Stator_Right")
    pg(1, st_left,     "Stator_Left")

    print(f"\nBoundaries:")
    print(f"  Domain       : {len(domain_tags)} curves")
    print(f"  SB_Stator    : {sb_stator}")
    print(f"  Stator_Right : {len(st_right)} curves")
    print(f"  Stator_Left  : {len(st_left)} curves")

    # ── 9. Mesh size (point-based) ────────────────────────────────────────────
    for _, pt in model.getEntities(0):
        x, y, _, _, _, _ = model.getBoundingBox(0, pt)
        r = math.hypot(x, y)
        if r > p.R_so - 2:
            sz = p.lc_so
        elif r < p.R_sb + 0.5:
            sz = p.lc_gap
        elif r < p.R_si + p.h1 + 0.5:
            sz = p.lc_si
        elif r < p.R_si + p.h1 + p.t_liner + 0.5:
            sz = p.lc_ins
        else:
            sz = p.lc_hp
        model.mesh.setSize([(0, pt)], sz)

    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 1)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints",         1)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature",      0)
    gmsh.option.setNumber("Mesh.Algorithm", 6)   # Frontal-Delaunay

    # ── 10. Generate mesh ─────────────────────────────────────────────────────
    print("\nGenerating stator mesh ...")
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
        "domain":       domain_tags,
        "stator_right": st_right,
        "stator_left":  st_left,
        "n_nodes":      n_nodes,
        "n_elems":      n_elems,
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

    out = Path(__file__).parent / "mesh_stator" / "stator.msh"
    result = build_stator(p, mesh_out=out, gui=gui, check_only=check_only)

    if result:
        print(f"\nFill factor : {result['fill_factor']*100:.1f} %")
        print(f"Cu/slot     : {result['Carea_m2']*1e6:.1f} mm²")
