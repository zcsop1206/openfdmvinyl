import math
import numpy as np
import pytest

from openvinyl.printer import get_profile
from openvinyl.groove import GrooveParams, build_groove, slope_limit
from openvinyl.dsp import test_signal as make_signal, condition


@pytest.fixture
def printer():
    return get_profile("p1s")


def test_extrusion_math(printer):
    assert printer.extrusion_width == pytest.approx(0.48)
    # 0.48 x 0.2 bead through 1.75 filament: ~0.037 mm filament per mm
    assert 0.03 < printer.e_per_mm() < 0.045


def test_pitch_formula(printer):
    g = GrooveParams(amplitude=0.1).resolve(printer)
    assert g.pitch == pytest.approx(1.1 * 0.48 + 0.48 + 0.2 + 0.48)
    assert g.land == pytest.approx(0.48)


def test_arc_inverse(printer):
    g = GrooveParams().resolve(printer)
    th = np.linspace(0, g.theta_end * 0.9, 50)
    s = g.arc_of_theta(th)
    assert np.allclose(g.theta_of_arc(s), th, atol=1e-9)


def test_smear_and_stylus_limits(printer):
    g = GrooveParams(rpm=78).resolve(printer)
    # 78 RPM, r=40: v = 326.7 mm/s, w_e = 0.48 -> 680 Hz
    assert g.f_smear(40.0) == pytest.approx(680.7, abs=1)
    assert g.f_stylus(40.0) == pytest.approx(326.7 / (2 * math.pi * 0.076), rel=1e-3)


def test_tone_maps_to_amplitude(printer):
    fs = 44100
    a = condition(make_signal("tone:440", 4.0, fs), fs, lpf=681)
    gp = GrooveParams(amplitude=0.1)
    p = build_groove(a, fs, gp, printer)
    assert np.abs(p.x).max() == pytest.approx(0.1, rel=0.02)
    assert p.stats["slope_limited_fraction"] == 0.0
    # constant arc length sampling
    ds = np.diff(p.geometry.arc_of_theta(p.theta))
    assert np.allclose(ds, gp.segment_length, atol=1e-6)
    # play time covers lead-in + audio + lead-out
    assert p.t[-1] == pytest.approx(4.0 + 2 * 60 / 78, abs=0.05)


def test_slope_limiter_engages(printer):
    fs = 44100
    # 600 Hz at 0.15 mm near the inner radius: slope ~60 deg, beyond 45 deg
    a = make_signal("tone:600", 2.0, fs)
    gp = GrooveParams(amplitude=0.15, outer_groove_radius=45.0, inner_groove_radius=40.0,
                      lead_in_turns=0.2, lead_out_turns=0.2)
    p = build_groove(a, fs, gp, printer)
    slope = np.abs(np.gradient(p.x, gp.segment_length)).max()
    assert slope <= math.tan(math.radians(45)) * 1.05
    assert p.stats["slope_limited_fraction"] > 0.1


def test_slope_limit_function():
    ds = 0.1
    x = 2.0 * np.sin(np.linspace(0, 40 * np.pi, 4000))   # peak slope 0.63 > 0.3
    y, frac = slope_limit(x, ds, 0.3, window=5)
    assert np.abs(np.gradient(y, ds)).max() <= 0.3 * 1.05
    assert frac > 0


def test_feedrate_within_limits(printer):
    fs = 44100
    a = condition(make_signal("sweep:300:680", 6.0, fs), fs, lpf=681)
    p = build_groove(a, fs, GrooveParams(amplitude=0.12), printer)
    assert p.speed.min() >= printer.groove_speed_min
    assert p.speed.max() <= printer.groove_speed_max


def test_truncation_reported(printer):
    fs = 8000
    a = np.zeros(int(60 * fs))
    p = build_groove(a, fs, GrooveParams(), printer)
    assert "truncated_audio_s" in p.stats
    assert p.r_spiral[-1] >= 40.0 - 1e-6


def test_crease_stays_open_on_steep_passages(printer):
    """Beads offset along the path normal keep the full spacing s even where
    the slope limiter is binding; a radial offset would give s*cos(slope)."""
    from openvinyl.groove import crease_checks
    fs = 44100
    a = condition(make_signal("tone:1000", 2.0, fs), fs, lpf=5000)
    gp = GrooveParams(amplitude=0.1, outer_groove_radius=60.0, inner_groove_radius=40.0)
    p = build_groove(a, fs, gp, printer)
    assert p.stats["curvature_limited_fraction"] > 0
    cr = crease_checks(p)
    s = p.geometry.bead_spacing
    assert cr["crease_gap_min_mm"] >= s - 0.005
    assert cr["inner_bead_reversals"] == 0
    # unlimited, this tone bends tighter than s/2 and the radial offset closes the crease
