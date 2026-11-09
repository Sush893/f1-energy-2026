"""Tests for the physics model: each term checked against a hand calculation."""
import numpy as np
import pandas as pd
import pytest

from src import physics as ph

P = ph.CarParams(mass_kg=800, cda_m2=1.0, rho=1.2, crr=0.0)


def test_constant_speed_power_is_drag_only():
    # 80 m/s, no acceleration: P = 0.5 * 1.2 * 1.0 * 80^3 = 307.2 kW
    assert ph.power_kw(80.0, 0.0, P) == pytest.approx(307.2)


def test_acceleration_term():
    # Zero drag: P = m * a * v = 800 * 5 * 50 = 200 kW
    no_drag = ph.CarParams(mass_kg=800, cda_m2=0.0, rho=1.2, crr=0.0)
    assert ph.power_kw(50.0, 5.0, no_drag) == pytest.approx(200.0)


def test_rolling_resistance_term():
    only_rolling = ph.CarParams(mass_kg=800, cda_m2=0.0, rho=1.2, crr=0.015)
    assert ph.power_kw(50.0, 0.0, only_rolling) == pytest.approx(800 * 9.81 * 0.015 * 50 / 1000)


def test_smoothing_recovers_known_acceleration_from_rounded_uneven_data():
    # Car accelerating at exactly 3 m/s^2, sampled unevenly and rounded to whole km/h,
    # like real telemetry.
    rng = np.random.default_rng(1)
    t = np.cumsum(rng.choice([0.16, 0.24, 0.28, 0.40], size=40))
    speed_kmh = np.round((50 + 3.0 * t) * 3.6)
    _, _, a = ph.smooth_speed(t, speed_kmh)
    middle = a[10:-10]
    assert np.median(middle) == pytest.approx(3.0, abs=0.3)


def _lap(throttle, brake, times=None):
    n = len(throttle)
    times = times if times is not None else np.arange(n) * 0.25
    return pd.DataFrame({"Session": "Q", "Driver": "TST", "Team": "T", "LapNumber": 1,
                         "Time": times, "Throttle": throttle, "Brake": brake,
                         "Speed": np.linspace(200, 300, n), "Distance": np.linspace(0, 1000, n)})


def test_find_straights_needs_full_throttle_for_long_enough():
    throttle = [100] * 30 + [40] * 5 + [100] * 8       # 7.25 s run, then a 1.75 s run
    lap = _lap(throttle, [False] * len(throttle))
    straights, skipped = ph.find_straights(lap)
    assert len(straights) == 1 and skipped == 0


def test_find_straights_skips_runs_with_a_data_hole():
    times = np.r_[np.arange(10) * 0.25, 4.0 + np.arange(20) * 0.25]   # 1.75 s hole
    lap = _lap([100] * 30, [False] * 30, times=times)
    straights, skipped = ph.find_straights(lap)
    assert straights == [] and skipped == 1


def test_matched_speed_delta_ignores_drag_area():
    # Same speeds, race car accelerates 1 m/s^2 less. The difference must not
    # depend on the drag area assumed.
    speeds = np.repeat(np.arange(250, 300, 5.0), 3)
    q = pd.DataFrame({"Speed_kmh": speeds, "Accel_ms2": 4.0})
    r = pd.DataFrame({"Speed_kmh": speeds, "Accel_ms2": 3.0})
    low = ph.matched_speed_delta(q, r, ph.CarParams(cda_m2=0.9), ph.CarParams(cda_m2=0.9))
    high = ph.matched_speed_delta(q, r, ph.CarParams(cda_m2=1.3), ph.CarParams(cda_m2=1.3))
    np.testing.assert_allclose(low["Delta_kW"], high["Delta_kW"])
    assert (low["Delta_kW"] < 0).all()


def test_match_straights_pairs_same_part_of_lap():
    q = pd.DataFrame({"StartFrac": [0.17, 0.50], "EndFrac": [0.31, 0.66]})
    r = pd.DataFrame({"StartFrac": [0.52, 0.18], "EndFrac": [0.67, 0.31]})
    pairs = ph.match_straights(q, r)
    assert ((0.17, 0.31), (0.18, 0.31)) in pairs and ((0.50, 0.66), (0.52, 0.67)) in pairs


def test_sample_params_stay_in_range():
    for p in ph.sample_params(200, seed=3):
        for name, (lo, hi) in ph.PARAM_RANGES.items():
            assert lo <= getattr(p, name) <= hi
