#!/usr/bin/env python3
"""
Mesh generator for a 48-slot, 8-pole, 3-phase inset-PM traction motor.
  Stator OD = 220 mm, active length = 170 mm.
  Minimum sector = 45° (one pole pitch) with anti-periodic BCs.
"""

import gmsh
import math
import sys
import subprocess
from pathlib import Path

π = math.pi
cos, sin, asin, sqrt = math.cos, math.sin, math.asin, math.sqrt

# ═══════════════════════════════════════════════════════════════════════════════
#  Machine parameters
# ═══════════════════════════════════════════════════════════════════════════════
Qs, Qp = 48, 8;  PP = 4;  Qph = 3
ns  = Qs // Qp             # slots per sector = 6
θs  = 2*π / Qp             # sector angle = π/4 = 45°
sp  = θs / ns              # slot pitch = 7.5°

R_so = 110.0               # stator outer radius [mm]
R_si = 74.0                # stator bore [mm]

# Slot [mm]
b1, h1, h2 = 2.5, 0.5, 0.5   # opening width/height, wedge height
b4, b5, h5 = 5.0, 8.5, 20.0  # winding bottom/top width, height
s_off  = R_si - sqrt(R_si**2 - (b1/2)**2)
R_wb   = R_si - s_off + h1 + h2   # winding bottom radius ≈ 75.0
R_st   = R_wb + h5                 # slot top radius ≈ 95.0

# Air gap [mm]
g    = 0.75
R_sb = R_si - g / 2        # sliding boundary = 73.625
R_ro = R_si - g            # rotor outer = 73.25

# Rotor [mm]
R_ri  = 25.0               # rotor inner (shaft outer)

# Inset magnet [mm, rad]
mag_frac = 0.80
α_m   = mag_frac * θs      # magnet arc = 36°
h_m   = 4.0
R_mi  = R_ro - h_m         # = 69.25
θ_ml  = (θs - α_m) / 2    # = 4.5° from sector start
θ_mr  = θs - θ_ml          # = 40.5°

# Mesh sizes [mm]
lc_so, lc_si  = 6.0, 3.0  # stator outer / bore
lc_sl         = 2.0        # slot conductor
lc_gap        = 0.30       # air gap (fine)
lc_ro, lc_ri  = 2.5, 5.0  # rotor outer / inner (shaft)
lc_mag        = 1.5        # magnet

print("=== TractionMachinePM2D: 48s-8p inset-PM, OD=220mm ===")
print(f"  R_wb={R_wb:.3f}  R_st={R_st:.3f}  R_sb={R_sb:.3f}  R_ro={R_ro:.3f}  R_mi={R_mi:.3f}")
print(f"  θ_ml={math.degrees(θ_ml):.2f}°  θ_mr={math.degrees(θ_mr):.2f}°")

# ═══════════════════════════════════════════════════════════════════════════════
#  gmsh setup
# ═══════════════════════════════════════════════════════════════════════════════
gmsh.initialize()
gmsh.model.add("TractionPM2D")
occ = gmsh.model.occ

# ─── Helper: create an annular sector (r1..r2, t1..t2 CCW) ──────────────────
def annular_sector(r1, r2, t1, t2, lc):
    O  = occ.addPoint(0, 0, 0, lc)
    p1 = occ.addPoint(r1*cos(t1), r1*sin(t1), 0, lc)
    p2 = occ.addPoint(r2*cos(t1), r2*sin(t1), 0, lc)
    p3 = occ.addPoint(r2*cos(t2), r2*sin(t2), 0, lc)
    p4 = occ.addPoint(r1*cos(t2), r1*sin(t2), 0, lc)
    l12 = occ.addLine(p1, p2)
    a23 = occ.addCircleArc(p2, O, p3)
    l34 = occ.addLine(p3, p4)
    a41 = occ.addCircleArc(p4, O, p1)
    return occ.addPlaneSurface([occ.addCurveLoop([l12, a23, l34, a41])])

# ─── Helper: create a pie sector (0..r, t1..t2) ──────────────────────────────
def pie_sector(r, t1, t2, lc):
    O  = occ.addPoint(0, 0, 0, lc)
    p1 = occ.addPoint(r*cos(t1), r*sin(t1), 0, lc)
    p2 = occ.addPoint(r*cos(t2), r*sin(t2), 0, lc)
    l1 = occ.addLine(O, p1)
    a  = occ.addCircleArc(p1, O, p2)
    l2 = occ.addLine(p2, O)
    return occ.addPlaneSurface([occ.addCurveLoop([l1, a, l2])])

# ─── 1. Stator sector (bore → outer, full sector) ────────────────────────────
s_stator = annular_sector(R_si, R_so, 0, θs, lc_si)

# ─── 2. Six slot conductor surfaces ──────────────────────────────────────────
slot_surfs = []
for i in range(ns):
    θ_c = (i + 0.5) * sp
    θh_wb = asin(b4 / 2 / R_wb)
    θh_st = asin(b5 / 2 / R_st)
    θWL = θ_c - θh_wb;  θWR = θ_c + θh_wb
    θSL = θ_c - θh_st;  θSR = θ_c + θh_st
    O2  = occ.addPoint(0, 0, 0, lc_sl)
    pWL = occ.addPoint(R_wb*cos(θWL), R_wb*sin(θWL), 0, lc_sl)
    pWR = occ.addPoint(R_wb*cos(θWR), R_wb*sin(θWR), 0, lc_sl)
    pSL = occ.addPoint(R_st*cos(θSL), R_st*sin(θSL), 0, lc_sl)
    pSR = occ.addPoint(R_st*cos(θSR), R_st*sin(θSR), 0, lc_sl)
    a_bot = occ.addCircleArc(pWL, O2, pWR)
    l_R   = occ.addLine(pWR, pSR)
    a_top = occ.addCircleArc(pSL, O2, pSR)   # L→R at slot top
    l_L   = occ.addLine(pSL, pWL)
    # CCW: bot(L→R), right wall up, -top(R→L), left wall down
    slot_surfs.append(occ.addPlaneSurface([occ.addCurveLoop([a_bot, l_R, -a_top, l_L])]))

# ─── 3. Stator airgap: R_sb → R_si (stator side, fixed) ─────────────────────
s_gap_s = annular_sector(R_sb, R_si, 0, θs, lc_gap)

# ─── 4. Rotor airgap: R_ro → R_sb (rotor side, rotates) ─────────────────────
s_gap_r = annular_sector(R_ro, R_sb, 0, θs, lc_gap)

# ─── 5. Rotor iron sector (R_ri → R_ro, full sector) ────────────────────────
s_rotor = annular_sector(R_ri, R_ro, 0, θs, lc_ro)

# ─── 6. Inset magnet (R_mi → R_ro, θ_ml → θ_mr) ─────────────────────────────
s_mag = annular_sector(R_mi, R_ro, θ_ml, θ_mr, lc_mag)

# ─── 7. Shaft (0 → R_ri, full sector) ───────────────────────────────────────
s_shaft = pie_sector(R_ri, 0, θs, lc_ri)

# ─── 8. Fragment in two independent regions (keeps SB non-conforming) ────────
# Stator side: iron + slots + stator airgap (gap_s inner arc at R_sb stays free)
stator_in = [(2, s_stator), (2, s_gap_s)] + [(2, s) for s in slot_surfs]
occ.fragment(stator_in, [])
occ.synchronize()
# Rotor side: rotor airgap + rotor iron + magnet + shaft
# (gap_r outer arc at R_sb stays free, independent of gap_s inner arc)
rotor_in = [(2, s_gap_r), (2, s_rotor), (2, s_mag), (2, s_shaft)]
occ.fragment(rotor_in, [])
occ.synchronize()

# ═══════════════════════════════════════════════════════════════════════════════
#  Classify surfaces by centroid + area
# ═══════════════════════════════════════════════════════════════════════════════
def surf_info(tag):
    x, y, _ = gmsh.model.occ.getCenterOfMass(2, tag)
    r = math.hypot(x, y)
    θ = math.atan2(y, x)
    A = gmsh.model.occ.getMass(2, tag)
    return r, θ, A

all_surfs = [t for d, t in gmsh.model.getEntities(2)]

# Expected areas (approximate) for classification:
#   Stator iron  ≈ π*(R_so²-R_si²)/8 - 6*slot_area  ≈ 1780-1900 W
#   Slot (each)  ≈ 0.5*(R_st²-R_wb²)*2*θh_wb        ≈ 125-145 mm²
#   Stator gap   ≈ 0.5*(R_si²-R_sb²)*θs              ≈ 21-22 mm²
#   Rotor gap    ≈ 0.5*(R_sb²-R_ro²)*θs              ≈ 21-22 mm²  (same thickness, same angle)
#   Magnet       ≈ 0.5*(R_ro²-R_mi²)*α_m             ≈ 178-180 mm²
#   Rotor iron   ≈ 0.5*(R_ro²-R_ri²)*θs - mag_area   ≈ 1660-1690 mm²
#   Shaft        ≈ 0.5*R_ri²*θs                      ≈ 245 mm²

A_slot_exp  = 0.5 * (R_st**2 - R_wb**2) * 2*asin(b5/2/R_st)
A_gap_exp   = 0.5 * (R_si**2 - R_sb**2) * θs
A_mag_exp   = 0.5 * (R_ro**2 - R_mi**2) * α_m
A_shaft_exp = 0.5 * R_ri**2 * θs
print(f"\nExpected areas: slot={A_slot_exp:.1f}  gap={A_gap_exp:.1f}  "
      f"mag={A_mag_exp:.1f}  shaft={A_shaft_exp:.1f}")

# Classify using area brackets
stator_iron_tags = []
slot_tags        = [[] for _ in range(ns)]
gap_stator_tags  = []
gap_rotor_tags   = []
rotor_iron_tags  = []
magnet_tags      = []
shaft_tags       = []
unclassified     = []

# Compute expected centroid radius for each body type (using analytic formula)
# r_c = sin(half_angle)/half_angle * (2/3)*(R_out^3-R_in^3)/(R_out^2-R_in^2)
def r_centroid_sector(R_in, R_out, t1, t2):
    """Centroid distance from origin for an annular arc sector."""
    half = (t2 - t1) / 2
    fc   = math.sin(half) / half if half > 1e-9 else 1.0
    r_m  = (2/3) * (R_out**3 - R_in**3) / (R_out**2 - R_in**2)
    return fc * r_m

r_c_gap_s  = r_centroid_sector(R_sb, R_si, 0, θs)   # ≈ 71.94
r_c_gap_r  = r_centroid_sector(R_ro, R_sb, 0, θs)   # ≈ 71.57
r_c_mag    = r_centroid_sector(R_mi, R_ro, θ_ml, θ_mr)  # ≈ 70.1
print(f"Expected centroids: gap_s={r_c_gap_s:.3f}  gap_r={r_c_gap_r:.3f}  mag={r_c_mag:.3f}")

for tag in all_surfs:
    r, θ, A = surf_info(tag)

    if A > 1400:
        # Large surface: stator iron (r>80) or rotor iron (r<60)
        if r > 80:
            stator_iron_tags.append(tag)
        elif r > 40:
            rotor_iron_tags.append(tag)
        else:
            shaft_tags.append(tag)   # unlikely but fallback

    elif A < 30:
        # Thin airgap layer (area ≈ 21 mm²)
        if r > r_c_gap_r + 0.1:
            gap_stator_tags.append(tag)
        else:
            gap_rotor_tags.append(tag)

    elif 100 < A < 300 and r > 75:
        # Slot conductor (area ≈ 130 mm², r_centroid ≈ 85)
        # Find which slot by angle
        found = False
        for i in range(ns):
            θ_c = (i + 0.5) * sp
            if abs(θ - θ_c) < sp * 0.6:
                slot_tags[i].append(tag)
                found = True
                break
        if not found:
            stator_iron_tags.append(tag)  # partial iron at sector edge

    elif 100 < A < 215 and r > 60:
        # Magnet (area ≈ 179 mm², r_centroid ≈ 70)
        magnet_tags.append(tag)

    elif A < 350 and r < 35:
        # Shaft (area ≈ 245 mm², r_centroid ≈ 16)
        shaft_tags.append(tag)

    else:
        # Small extra iron pieces (rotor corners or tooth tip iron)
        if r > R_wb:
            stator_iron_tags.append(tag)
        elif r > 40:
            rotor_iron_tags.append(tag)
        else:
            shaft_tags.append(tag)

print("\nSurface classification:")
print(f"  Stator iron:    {stator_iron_tags}")
for i in range(ns):
    print(f"  Slot {i}:         {slot_tags[i]}")
print(f"  Airgap stator:  {gap_stator_tags}")
print(f"  Airgap rotor:   {gap_rotor_tags}")
print(f"  Rotor iron:     {rotor_iron_tags}")
print(f"  Magnet:         {magnet_tags}")
print(f"  Shaft:          {shaft_tags}")

# Sanity check
for i in range(ns):
    if not slot_tags[i]:
        print(f"  WARNING: slot {i} has no surface!")
required = [stator_iron_tags, gap_stator_tags, gap_rotor_tags,
            rotor_iron_tags, magnet_tags]
names_r  = ['Stator iron', 'Gap stator', 'Gap rotor', 'Rotor iron', 'Magnet']
for nm, lst in zip(names_r, required):
    if not lst:
        print(f"  WARNING: {nm} is EMPTY — check classification!")

# ═══════════════════════════════════════════════════════════════════════════════
#  Physical groups — bodies
# ═══════════════════════════════════════════════════════════════════════════════
def pg(dim, tags, name):
    if tags:
        gmsh.model.addPhysicalGroup(dim, tags, name=name)
    else:
        print(f"  SKIP empty group: {name}")

pg(2, stator_iron_tags,  "Stator-0_Lamination")
for i in range(ns):
    pg(2, slot_tags[i],  f"Stator-0_Winding_R0-T0-S{i}")
pg(2, gap_stator_tags,   "Airgap_Stator")
pg(2, gap_rotor_tags,    "Airgap_Rotor")
pg(2, rotor_iron_tags,   "Rotor-0_Lamination")
pg(2, magnet_tags,       "Rotor-0_HoleMag_R0-T0-S0")
pg(2, shaft_tags,        "None_Shaft")

# ═══════════════════════════════════════════════════════════════════════════════
#  Physical groups — boundaries (adjacency-based, robust)
# ═══════════════════════════════════════════════════════════════════════════════

def get_bound_curves(surf_tags):
    """Return set of curve tags bounding a list of surfaces."""
    curves = set()
    for st in surf_tags:
        _, down = gmsh.model.getAdjacencies(2, st)
        curves.update(abs(int(c)) for c in down)
    return curves

# Collect curves per body group
stator_curves = get_bound_curves(stator_iron_tags)
slot_curves   = get_bound_curves([t for sl in slot_tags for t in sl])
gap_s_curves  = get_bound_curves(gap_stator_tags)
gap_r_curves  = get_bound_curves(gap_rotor_tags)
rotor_curves  = get_bound_curves(rotor_iron_tags)
magnet_curves = get_bound_curves(magnet_tags)
shaft_curves  = get_bound_curves(shaft_tags)

# SB: with two separate fragment() calls, gap_s and gap_r have INDEPENDENT curves
# at R_sb (non-conforming mesh — required for Elmer mortar BC).
# SB_Stator = inner arc of gap_s (r_c ≈ R_sb * sinc(θs/2) ≈ 71.73mm)
# SB_Rotor  = outer arc of gap_r (r_c ≈ R_sb * sinc(θs/2) ≈ 71.73mm)
# Distinguish them: gap_s also has outer arc at R_si (r_c ≈ 72.09mm) and
# lines (r_c ≈ 73.8mm); gap_r also has inner arc at R_ro (r_c ≈ 71.36mm)
# and lines (r_c ≈ 73.45mm).  The SB arc has the minimum arc-r_c in gap_s
# and the maximum arc-r_c in gap_r.  Arcs have r_c < 73mm; lines r_c > 73mm.

def curve_r_center(ctag):
    x, y, _ = gmsh.model.occ.getCenterOfMass(1, ctag)
    return math.hypot(x, y)

def curve_length(ctag):
    return gmsh.model.occ.getMass(1, ctag)

# SB_Stator = inner arc of gap_s (at R_sb) = arc with minimum r_c in gap_s.
# Arcs have r_c < 73mm; lines have r_c > 73mm.
gap_s_arcs = {c: curve_r_center(c) for c in gap_s_curves if curve_r_center(c) < 73.0}
sb_stator_tags = [min(gap_s_arcs, key=gap_s_arcs.get)] if gap_s_arcs else []

# SB_Rotor = outer arc of gap_r (at R_sb, full 45° sector).
# After fragment, gap_r inner boundary at R_ro may be split at magnet edges,
# creating shorter arcs with larger r_c.  Use max arc LENGTH to find the full
# outer arc (L_sb = R_sb*θs ≈ 57.8mm vs L_ro_36deg ≈ 46.0mm).
gap_r_arcs = {c: curve_r_center(c) for c in gap_r_curves if curve_r_center(c) < 73.0}
gap_r_arc_len = {c: curve_length(c) for c in gap_r_arcs}
sb_rotor_tags  = [max(gap_r_arc_len, key=gap_r_arc_len.get)] if gap_r_arc_len else []

print(f"  SB_Stator arc r_c / len: "
      f"{[(f'{gap_s_arcs[c]:.3f}', f'{curve_length(c):.2f}') for c in sb_stator_tags]}")
print(f"  SB_Rotor  arc r_c / len: "
      f"{[(f'{gap_r_arcs[c]:.3f}', f'{curve_length(c):.2f}') for c in sb_rotor_tags]}")

# Domain: outer stator arcs (on stator iron, with large bounding-box midpoint r)
domain_tags = []
for c in stator_curves | slot_curves:
    x1, y1, _, x2, y2, _ = gmsh.model.getBoundingBox(1, c)
    xm, ym = (x1+x2)/2, (y1+y2)/2
    if math.hypot(xm, ym) > 100:
        domain_tags.append(c)

# Sector boundary helper: classify a curve set into right (θ=0) and left (θ=θs=45°)
def classify_sector_bounds(curve_set, excl):
    right, left = [], []
    for c in sorted(curve_set - excl):
        x1, y1, _, x2, y2, _ = gmsh.model.getBoundingBox(1, c)
        xm, ym = (x1+x2)/2, (y1+y2)/2
        if math.hypot(xm, ym) < 1.0:
            continue  # near-origin, skip
        # Radial line at θ=0: both bbox corners have y≈0
        if abs(y1) < 0.5 and abs(y2) < 0.5 and xm > 2.0:
            right.append(c)
        # Radial line at θ=45°: both bbox corners have x≈y
        elif abs(x1 - y1) < 0.5 and abs(x2 - y2) < 0.5 and xm > 2.0 and ym > 2.0:
            left.append(c)
    return right, left

excl_set = set(sb_stator_tags) | set(sb_rotor_tags) | set(domain_tags)
s_right_tags, s_left_tags = classify_sector_bounds(
    stator_curves | slot_curves | gap_s_curves, excl_set)
r_right_tags, r_left_tags = classify_sector_bounds(
    rotor_curves | gap_r_curves | magnet_curves | shaft_curves, excl_set)

pg(1, sb_rotor_tags,  "SB_Rotor")
pg(1, sb_stator_tags, "SB_Stator")
pg(1, domain_tags,    "Domain")
pg(1, s_right_tags,   "Stator-Right")
pg(1, s_left_tags,    "Stator-Left")
pg(1, r_right_tags,   "Rotor-Right")
pg(1, r_left_tags,    "Rotor-Left")

print(f"\nBoundaries:")
print(f"  SB_Rotor:       {sb_rotor_tags}")
print(f"  SB_Stator:      {sb_stator_tags}")
print(f"  Domain:         {domain_tags}")
print(f"  Stator-Right:   {s_right_tags}")
print(f"  Stator-Left:    {s_left_tags}")
print(f"  Rotor-Right:    {r_right_tags}")
print(f"  Rotor-Left:     {r_left_tags}")
# Warn if SB still shared (would mean no non-conforming nodes)
if set(sb_rotor_tags) & set(sb_stator_tags):
    print("  WARNING: SB_Rotor and SB_Stator share curves — mesh may be conforming!")

# ═══════════════════════════════════════════════════════════════════════════════
#  Mesh refinement — fine in airgap
# ═══════════════════════════════════════════════════════════════════════════════
all_sb_tags = sb_rotor_tags + sb_stator_tags
if all_sb_tags:
    gmsh.model.mesh.field.add("Distance", 1)
    gmsh.model.mesh.field.setNumbers(1, "CurvesList", all_sb_tags)
    gmsh.model.mesh.field.setNumber(1, "Sampling", 300)

    gmsh.model.mesh.field.add("Threshold", 2)
    gmsh.model.mesh.field.setNumber(2, "InField",  1)
    gmsh.model.mesh.field.setNumber(2, "SizeMin",  lc_gap)
    gmsh.model.mesh.field.setNumber(2, "SizeMax",  lc_so)
    gmsh.model.mesh.field.setNumber(2, "DistMin",  0.3)
    gmsh.model.mesh.field.setNumber(2, "DistMax",  10.0)
    gmsh.model.mesh.field.setAsBackgroundMesh(2)

gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
gmsh.option.setNumber("Mesh.MeshSizeFromPoints",         0)
gmsh.option.setNumber("Mesh.MeshSizeFromCurvature",      0)
gmsh.option.setNumber("Mesh.Algorithm",  6)    # Frontal-Delaunay

# ═══════════════════════════════════════════════════════════════════════════════
#  Generate mesh and export
# ═══════════════════════════════════════════════════════════════════════════════
print("\nGenerating 2D mesh ...")
gmsh.model.mesh.generate(2)
gmsh.model.mesh.optimize("Netgen")

n_nodes = len(gmsh.model.mesh.getNodes()[0])
n_elems = sum(len(gmsh.model.mesh.getElements(2, t)[1][0])
              for _, t in gmsh.model.getEntities(2))
print(f"Mesh: {n_nodes} nodes, {n_elems} 2D elements")

out_msh = str(Path(__file__).parent / "mesh" / "motor.msh")
gmsh.write(out_msh)
print(f"Written: {out_msh}")

if "--gui" in sys.argv:
    gmsh.fltk.run()

gmsh.finalize()

# ═══════════════════════════════════════════════════════════════════════════════
#  Convert with ElmerGrid (14=gmsh msh → 2=Elmer native)
# ═══════════════════════════════════════════════════════════════════════════════
mesh_dir = str(Path(__file__).parent / "mesh")
cmd = ["ElmerGrid", "14", "2", out_msh, "-autoclean", "-out", mesh_dir]
print(f"\nRunning ElmerGrid ...")
result = subprocess.run(cmd, capture_output=True, text=True)
if result.returncode != 0:
    print("ElmerGrid FAILED:")
    print(result.stderr[-2000:])
    sys.exit(1)
# Show relevant lines
for line in result.stdout.splitlines():
    if any(k in line for k in ["nodes", "elements", "bodies", "boundary", "Error"]):
        print(" ", line)
print("ElmerGrid done.")
