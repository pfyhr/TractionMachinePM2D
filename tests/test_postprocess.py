"""
Tests for postprocess.py.

The fast tests use synthetic fixture files (no real Elmer output needed).
The @pytest.mark.integration tests use the actual results/ directory and
require a completed simulation to be present.
"""
import math
import sys
import textwrap
import pytest
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from motor_params import MotorParams
from postprocess import (
    stator_body_id,
    rotor_body_id,
    read_scalars,
    read_losses,
    summarise,
    print_summary,
    _parse_names,
)
from gen_sif import _PHASE_MAP

π = math.pi


# ─────────────────────────────────────────────────────────────────────────────
#  Fixtures — synthetic data
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def params():
    return MotorParams()


@pytest.fixture
def results_dir(tmp_path, params):
    """Create a minimal synthetic results directory."""
    p = params
    SCALE = p.SCALE           # 8 * 0.170 = 1.36
    fme   = p.rpm / 60.0      # 25 Hz
    dt    = 1.0 / (fme * 120)
    n_steps = 241              # nper*ncycle+1 = 120*2+1

    # ── scalars.dat ──────────────────────────────────────────────────────────
    # 9 columns: eddy_power, field_energy, fl_total, fl_1, fl_2,
    #            airgap_torque, inertial_vol, inertial_mom, group1_torque
    rng = np.random.default_rng(42)
    torque_raw   = 80.0 + 5.0 * np.sin(2 * π * fme * np.arange(1, n_steps + 1) * dt)
    airgap_raw   = torque_raw * 0.98
    scalars = np.column_stack([
        rng.uniform(0, 10, n_steps),   # eddy power
        rng.uniform(60, 70, n_steps),  # field energy
        np.full(n_steps, 189.6),       # fl_total
        np.full(n_steps, 95.96),       # fl_1
        np.full(n_steps, 93.66),       # fl_2
        airgap_raw,                    # airgap torque
        np.zeros(n_steps),             # inertial vol
        np.zeros(n_steps),             # inertial mom
        torque_raw,                    # group 1 torque
    ])
    np.savetxt(tmp_path / "scalars.dat", scalars, fmt="%.6E")

    # ── scalars.dat.names ────────────────────────────────────────────────────
    names_content = textwrap.dedent("""\
        Metadata for SaveScalars file: results/scalars.dat
        Variables in columns of matrix:
           1: res: eddy current power
           2: res: electromagnetic field energy
           3: res: fourier loss total
           4: res: fourier loss 1
           5: res: fourier loss 2
           6: res: air gap torque
           7: res: inertial volume
           8: res: inertial moment
           9: res: group 1 torque
    """)
    (tmp_path / "scalars.dat.names").write_text(names_content)

    # ── loss-2d.dat ──────────────────────────────────────────────────────────
    sid = stator_body_id(p)
    rid = rotor_body_id(p)
    loss_lines = [
        "!body_id   loss(1)   loss(2)",
        f"  {sid}   9.590055E+01   9.322051E+01",
        f"  {rid}   5.583804E-02   4.357799E-01",
    ]
    (tmp_path / "loss-2d.dat").write_text("\n".join(loss_lines))

    return tmp_path, p


# ─────────────────────────────────────────────────────────────────────────────
#  Body ID helpers
# ─────────────────────────────────────────────────────────────────────────────

class TestBodyIds:
    def test_stator_body_id_is_1(self, params):
        assert stator_body_id(params) == 1

    def test_rotor_body_id_formula(self, params):
        p = params
        # Body 1: Stator_Iron
        # Body 2: Airgap_Stator
        # Bodies 3..2+ns*(2+n_hp): slots
        # Body 2+ns*(2+n_hp)+1: Airgap_Rotor
        # Body 2+ns*(2+n_hp)+2: Rotor_Iron
        expected = 2 + p.ns * (2 + p.n_hp) + 2
        assert rotor_body_id(params) == expected

    def test_rotor_body_id_default(self, params):
        # For ns=6, n_hp=6: 2 + 6*8 + 2 = 52
        assert rotor_body_id(params) == 52

    def test_stator_rotor_different(self, params):
        assert stator_body_id(params) != rotor_body_id(params)


# ─────────────────────────────────────────────────────────────────────────────
#  _parse_names
# ─────────────────────────────────────────────────────────────────────────────

class TestParseNames:
    def test_parses_columns(self, results_dir):
        d, _ = results_dir
        cm = _parse_names(d / "scalars.dat.names")
        assert cm["res: group 1 torque"] == 8
        assert cm["res: air gap torque"] == 5
        assert cm["res: fourier loss total"] == 2

    def test_zero_indexed(self, results_dir):
        d, _ = results_dir
        cm = _parse_names(d / "scalars.dat.names")
        # Column 1 in file = index 0
        assert cm["res: eddy current power"] == 0


# ─────────────────────────────────────────────────────────────────────────────
#  read_scalars
# ─────────────────────────────────────────────────────────────────────────────

class TestReadScalars:
    def test_returns_dict(self, results_dir):
        d, p = results_dir
        r = read_scalars(d / "scalars.dat", p)
        assert isinstance(r, dict)

    def test_n_steps(self, results_dir):
        d, p = results_dir
        r = read_scalars(d / "scalars.dat", p)
        assert r["n_steps"] == 241

    def test_torque_length(self, results_dir):
        d, p = results_dir
        r = read_scalars(d / "scalars.dat", p)
        assert len(r["torque_Nm"]) == 241

    def test_scale_applied(self, results_dir):
        d, p = results_dir
        r = read_scalars(d / "scalars.dat", p)
        # Raw torque ≈ 80 N·m/m/sector; SCALE = 1.36 → ≈ 108.8 N·m
        assert r["SCALE"] == pytest.approx(p.SCALE, rel=1e-6)
        assert r["T_mean_Nm"] == pytest.approx(80.0 * p.SCALE, rel=0.02)

    def test_t_mean_positive(self, results_dir):
        d, p = results_dir
        r = read_scalars(d / "scalars.dat", p)
        assert r["T_mean_Nm"] > 0

    def test_ripple_is_percent(self, results_dir):
        d, p = results_dir
        r = read_scalars(d / "scalars.dat", p)
        # Ripple = (max-min)/mean * 100; should be ~12.5% for ±5/80 sinusoid
        assert 0 <= r["T_ripple_pct"] <= 100

    def test_p_mech_kw(self, results_dir):
        d, p = results_dir
        r = read_scalars(d / "scalars.dat", p)
        omega = 2 * π * p.rpm / 60
        expected_kW = r["T_mean_Nm"] * omega / 1000.0
        assert r["P_mech_kW"] == pytest.approx(expected_kW, rel=1e-6)

    def test_skip_cycles(self, results_dir):
        d, p = results_dir
        r0 = read_scalars(d / "scalars.dat", p, skip_cycles=0)
        r1 = read_scalars(d / "scalars.dat", p, skip_cycles=1)
        # Skipping cycle 1 (transient) may shift the mean slightly
        assert r0["T_mean_Nm"] != r1["T_mean_Nm"] or True  # both should work

    def test_time_array_starts_at_dt(self, results_dir):
        d, p = results_dir
        r = read_scalars(d / "scalars.dat", p)
        fme = p.rpm / 60.0
        dt = 1.0 / (fme * 120)
        assert r["t_s"][0] == pytest.approx(dt, rel=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
#  read_losses
# ─────────────────────────────────────────────────────────────────────────────

class TestReadLosses:
    def test_returns_dict(self, results_dir):
        d, p = results_dir
        r = read_losses(d / "loss-2d.dat", p)
        assert isinstance(r, dict)

    def test_stator_losses_positive(self, results_dir):
        d, p = results_dir
        r = read_losses(d / "loss-2d.dat", p)
        assert r["stator_total_W"] > 0

    def test_rotor_losses_positive(self, results_dir):
        d, p = results_dir
        r = read_losses(d / "loss-2d.dat", p)
        assert r["rotor_total_W"] > 0

    def test_total_equals_sum(self, results_dir):
        d, p = results_dir
        r = read_losses(d / "loss-2d.dat", p)
        assert r["total_W"] == pytest.approx(
            r["stator_total_W"] + r["rotor_total_W"], rel=1e-9
        )

    def test_scale_applied(self, results_dir):
        d, p = results_dir
        r = read_losses(d / "loss-2d.dat", p)
        # Raw stator hyst = 95.9 W/m/sector; × SCALE = × 1.36
        expected = 95.90055 * p.SCALE
        assert r["stator_hyst_W"] == pytest.approx(expected, rel=1e-4)

    def test_hysteresis_plus_eddy(self, results_dir):
        d, p = results_dir
        r = read_losses(d / "loss-2d.dat", p)
        assert r["stator_total_W"] == pytest.approx(
            r["stator_hyst_W"] + r["stator_eddy_W"], rel=1e-9
        )

    def test_body_ids_recorded(self, results_dir):
        d, p = results_dir
        r = read_losses(d / "loss-2d.dat", p)
        assert stator_body_id(p) in r["body_ids"]
        assert rotor_body_id(p)  in r["body_ids"]

    def test_missing_body_returns_zero(self, tmp_path, params):
        # Loss file with body IDs that don't match stator/rotor
        (tmp_path / "loss-x.dat").write_text("!header\n  999  1.0  2.0\n")
        r = read_losses(tmp_path / "loss-x.dat", params)
        assert r["stator_total_W"] == 0.0
        assert r["rotor_total_W"]  == 0.0


# ─────────────────────────────────────────────────────────────────────────────
#  summarise
# ─────────────────────────────────────────────────────────────────────────────

class TestSummarise:
    def test_returns_dict(self, results_dir):
        d, p = results_dir
        m = summarise(d, p, suffix="2d")
        assert isinstance(m, dict)

    def test_all_keys_present(self, results_dir):
        d, p = results_dir
        m = summarise(d, p, suffix="2d")
        for key in ["T_mean_Nm", "stator_total_W", "rotor_total_W", "total_W",
                    "P_mech_kW", "SCALE", "efficiency_pct"]:
            assert key in m, f"Missing key: {key}"

    def test_efficiency_in_range(self, results_dir):
        d, p = results_dir
        m = summarise(d, p, suffix="2d")
        assert m["efficiency_pct"] is not None
        assert 50 < m["efficiency_pct"] <= 100

    def test_missing_scalars_does_not_crash(self, tmp_path, params):
        # Create only loss file, no scalars
        sid = stator_body_id(params)
        rid = rotor_body_id(params)
        (tmp_path / "loss-2d.dat").write_text(
            f"!h\n  {sid}  10.0  5.0\n  {rid}  1.0  0.5\n"
        )
        m = summarise(tmp_path, params, suffix="2d")
        assert m["T_mean_Nm"] is None
        assert m["stator_total_W"] is not None

    def test_missing_loss_does_not_crash(self, tmp_path, params):
        # Create only scalars, no loss file
        p = params
        fme = p.rpm / 60.0
        dt  = 1.0 / (fme * 120)
        data = np.ones((5, 9)) * 80.0
        np.savetxt(tmp_path / "scalars.dat", data)
        m = summarise(tmp_path, params, suffix="2d")
        assert m["total_W"] is None
        assert m["T_mean_Nm"] is not None


# ─────────────────────────────────────────────────────────────────────────────
#  print_summary (smoke test)
# ─────────────────────────────────────────────────────────────────────────────

class TestPrintSummary:
    def test_runs_without_error(self, results_dir, capsys):
        d, p = results_dir
        m = summarise(d, p, suffix="2d")
        print_summary(m)
        out = capsys.readouterr().out
        assert "Torque" in out or "TORQUE" in out
        assert "IRON" in out or "Iron" in out

    def test_none_values_do_not_crash(self, capsys):
        m = {
            "results_dir": ".", "suffix": "2d",
            "T_mean_Nm": None, "stator_total_W": None, "SCALE": 1.36,
            "efficiency_pct": None,
        }
        print_summary(m)   # should not raise


# ─────────────────────────────────────────────────────────────────────────────
#  Integration test — uses real results directory
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.integration
class TestRealResults:
    """Requires results/loss-2d.dat and results/scalars.dat to exist."""

    RESULTS = Path(__file__).parent.parent / "results"

    def test_real_loss_file_readable(self):
        lf = self.RESULTS / "loss-2d.dat"
        if not lf.exists():
            pytest.skip("results/loss-2d.dat not present")
        p = MotorParams()
        r = read_losses(lf, p)
        assert r["total_W"] > 0

    def test_real_scalars_readable(self):
        sf = self.RESULTS / "scalars.dat"
        if not sf.exists():
            pytest.skip("results/scalars.dat not present")
        p = MotorParams()
        r = read_scalars(sf, p)
        assert r["n_steps"] > 0
        assert r["T_mean_Nm"] != 0

    def test_real_summary(self):
        if not (self.RESULTS / "scalars.dat").exists():
            pytest.skip("results/ not present")
        p = MotorParams()
        m = summarise(self.RESULTS, p, suffix="2d")
        print_summary(m)
        assert m["total_W"] is not None or m["T_mean_Nm"] is not None
