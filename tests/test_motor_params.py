"""Tests for motor_params.py."""
import math
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from motor_params import MotorParams, suggest_slot


# ─────────────────────────────────────────────────────────────────────────────
#  Fixtures
# ─────────────────────────────────────────────────────────────────────────────
@pytest.fixture
def valid():
    """Default reference design — must pass validate()."""
    return MotorParams()


# ─────────────────────────────────────────────────────────────────────────────
#  Validate passes on good design
# ─────────────────────────────────────────────────────────────────────────────
def test_valid_design_passes(valid):
    p = valid.validate()
    assert p is valid   # returns self


def test_derived_radii(valid):
    assert math.isclose(valid.R_sb, valid.R_si - valid.g / 2)
    assert math.isclose(valid.R_ro, valid.R_si - valid.g)
    assert math.isclose(valid.R_mo, valid.R_ro - valid.h_bridge)
    assert math.isclose(valid.R_mi, valid.R_ro - valid.h_bridge - valid.h_m)


def test_sector_angle(valid):
    assert math.isclose(valid.θs, math.pi / 4)   # 45° for 8 poles


def test_slot_pitch_at_bore(valid):
    expected = 2 * math.pi * valid.R_si / valid.Qs
    assert math.isclose(valid.slot_pitch_arc, expected)


def test_fill_factor_range(valid):
    assert 0.25 < valid.fill_factor < 0.85


def test_carea_positive(valid):
    assert valid.Carea > 0


def test_hairpin_positive_dims(valid):
    assert valid.b_hp > 0
    assert valid.h_hp > 0


def test_tooth_widths_positive(valid):
    assert valid.tooth_tip_width > 0
    assert valid.tooth_body_width > 0


def test_back_iron_depth(valid):
    assert valid.back_iron_depth >= 2.0


def test_magnet_width_positive(valid):
    assert valid.w_mag > 0


def test_scale(valid):
    expected = valid.Qp * (valid.L_active * 1e-3)
    assert math.isclose(valid.SCALE, expected)


def test_j_peak(valid):
    assert math.isclose(valid.J_peak, valid.Is / valid.Carea)


# ─────────────────────────────────────────────────────────────────────────────
#  Topology errors
# ─────────────────────────────────────────────────────────────────────────────
def test_odd_pole_count_raises():
    with pytest.raises(ValueError, match="even"):
        MotorParams(Qp=7).validate()


def test_qs_not_divisible_by_qp_raises():
    with pytest.raises(ValueError, match="not divisible by Qp"):
        MotorParams(Qs=47, Qp=8).validate()


def test_q_not_integer_raises():
    # Qs=48, Qp=8, m=3 → q=2 is integer.  Use m=4 to break it.
    with pytest.raises(ValueError, match="Qp\\*m"):
        MotorParams(Qs=48, Qp=8, m=4).validate()


# ─────────────────────────────────────────────────────────────────────────────
#  Geometry errors
# ─────────────────────────────────────────────────────────────────────────────
def test_negative_airgap_raises():
    with pytest.raises(ValueError, match="manufacturing floor"):
        MotorParams(g=0.1).validate()


def test_tooth_tip_too_narrow_raises():
    # b1 slightly less than slot pitch → tooth tip effectively 0
    p = MotorParams()
    b1_bad = p.slot_pitch_arc - 0.1
    with pytest.raises(ValueError, match="[Tt]ooth"):
        MotorParams(b1=b1_bad).validate()


def test_tooth_body_too_narrow_raises():
    # b_slot almost equal to slot pitch at mid-radius
    p = MotorParams()
    r_mid = p.R_si + p.h1 + p.h_slot / 2
    slot_pitch_mid = 2 * math.pi * r_mid / p.Qs
    b_slot_bad = slot_pitch_mid - 0.5   # leaves < 1 mm tooth
    with pytest.raises(ValueError, match="[Tt]ooth"):
        MotorParams(b_slot=b_slot_bad).validate()


def test_back_iron_too_thin_raises():
    # Push slot so deep it eats the back iron
    with pytest.raises(ValueError, match="[Bb]ack iron"):
        MotorParams(h_slot=35.0).validate()


def test_radii_ordering_raises():
    with pytest.raises(ValueError, match="[Rr]adii"):
        MotorParams(R_si=80.0, R_so=70.0).validate()   # R_si > R_so


# ─────────────────────────────────────────────────────────────────────────────
#  Hairpin dimension errors
# ─────────────────────────────────────────────────────────────────────────────
def test_hairpin_too_wide_raises():
    # t_liner so large there's no room for conductor
    with pytest.raises(ValueError, match="[Hh]airpin width"):
        MotorParams(t_liner=4.0).validate()


def test_hairpin_too_tall_raises():
    # t_liner + t_enam so large per-layer height is negative
    with pytest.raises(ValueError, match="[Hh]airpin height"):
        MotorParams(n_hp=100, t_enam=0.5).validate()


# ─────────────────────────────────────────────────────────────────────────────
#  Fill factor bounds
# ─────────────────────────────────────────────────────────────────────────────
def test_fill_factor_too_low_raises():
    # Very thick liner → low fill
    with pytest.raises(ValueError, match="[Ff]ill factor"):
        MotorParams(t_liner=3.0).validate()


# ─────────────────────────────────────────────────────────────────────────────
#  Magnet errors
# ─────────────────────────────────────────────────────────────────────────────
def test_mag_frac_out_of_range_raises():
    with pytest.raises(ValueError, match="mag_frac"):
        MotorParams(mag_frac=1.2).validate()


def test_magnet_deeper_than_rotor_raises():
    # magnet depth > rotor outer - shaft outer
    with pytest.raises(ValueError, match="[Mm]agnet inner"):
        MotorParams(h_m=50.0).validate()


# ─────────────────────────────────────────────────────────────────────────────
#  suggest_slot
# ─────────────────────────────────────────────────────────────────────────────
def test_suggest_slot_returns_feasible():
    sug = suggest_slot(48, 8, 74.0, 110.0, target_fill=0.55)
    assert sug["b_slot"] > 0
    assert sug["h_slot"] > 0
    assert sug["tooth_tip"] > 0
    assert sug["tooth_body"] > 0
    assert sug["back_iron"] >= 2.0


def test_suggest_slot_fill_close_to_target():
    target = 0.55
    sug = suggest_slot(48, 8, 74.0, 110.0, target_fill=target)
    assert abs(sug["fill_factor"] - target) < 0.08   # within 8 pp


def test_suggest_slot_different_topologies():
    # 36-slot 6-pole
    sug = suggest_slot(36, 6, 90.0, 135.0, target_fill=0.60)
    assert sug["b_slot"] > 0
    assert sug["h_slot"] > 0
    # 72-slot 8-pole
    sug2 = suggest_slot(72, 8, 100.0, 160.0, target_fill=0.55)
    assert sug2["fill_factor"] > 0.2


# ─────────────────────────────────────────────────────────────────────────────
#  Summary / __str__
# ─────────────────────────────────────────────────────────────────────────────
def test_summary_contains_key_fields(valid):
    s = valid.summary()
    assert "48" in s       # Qs
    assert "8" in s        # Qp
    assert "Fill" in s
    assert "Hairpin" in s
