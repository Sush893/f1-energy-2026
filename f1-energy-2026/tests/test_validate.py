"""Tests for the data quality checks, built from problems seen in real 2026 data."""
import pandas as pd
import pytest

from src import validate as v


def make_lap(speeds, times=None, **extra):
    """A small, clean single lap; override channels to inject problems."""
    n = len(speeds)
    times = times if times is not None else [i * 0.24 for i in range(n)]
    data = {
        "Session": "Q", "Driver": "TST", "LapNumber": 1,
        "Time": times, "Speed": speeds,
        "RPM": [10000 + 37 * i for i in range(n)],
        "nGear": [8] * n, "Throttle": [100.0] * n, "Brake": [False] * n,
    }
    data.update(extra)
    return pd.DataFrame(data)


def test_clean_lap_has_no_hard_flags():
    lap = make_lap([280, 283, 286, 289, 292, 295])
    flagged = v.validate_car(lap)
    assert not flagged["flag_any"].any()


def test_gap_longer_than_limit_is_flagged():
    lap = make_lap([280, 283, 286, 289], times=[0.0, 0.24, 1.40, 1.64])
    assert v.flag_gaps(lap).tolist() == [False, False, True, False]


def test_frozen_run_is_flagged_like_the_real_alpine_case():
    # Real Monza example: speed stuck at 308 km/h with throttle 104 and brake on,
    # then a jump to 118 km/h.
    n_frozen = 4
    lap = make_lap(
        [300] + [308] * n_frozen + [118],
        RPM=[10500] + [10602] * n_frozen + [10436],
        Throttle=[100.0] + [104.0] * n_frozen + [60.0],
        Brake=[False] + [True] * n_frozen + [False],
    )
    flagged = v.validate_car(lap)
    assert flagged["flag_frozen"].tolist() == [False] + [True] * n_frozen + [False]
    # The jump out of the frozen run implies ~-220 m/s^2 (over 20 g).
    assert flagged["flag_impossible_accel"].iloc[-1]


def test_two_identical_samples_are_not_frozen():
    lap = make_lap([300, 300, 302], RPM=[10500, 10500, 10550])
    assert not v.flag_frozen(lap).any()


def test_realistic_hard_braking_is_not_flagged():
    # 5 g of braking over 0.24 s is about 42 km/h of speed lost.
    lap = make_lap([330, 288, 246])
    assert not v.flag_impossible_accel(lap).any()


def test_throttle_over_100_and_overlap():
    lap = make_lap([200, 202, 204], Throttle=[104.0, 50.0, 5.0], Brake=[False, True, True])
    assert v.flag_throttle_over_100(lap).tolist() == [True, False, False]
    assert v.flag_throttle_brake_overlap(lap).tolist() == [False, True, False]


def test_clean_car_drops_frozen_and_caps_throttle():
    lap = make_lap(
        [300, 308, 308, 308, 118, 120],
        RPM=[10500, 10602, 10602, 10602, 10436, 10500],
        Throttle=[104.0, 104.0, 104.0, 104.0, 60.0, 62.0],
        Brake=[False, True, True, True, False, False],
    )
    cleaned = v.clean_car(v.validate_car(lap))
    assert len(cleaned) == 2           # frozen run and the impossible jump removed
    assert cleaned["Throttle"].max() <= 100.0


def test_laps_are_checked_separately():
    # A big time difference between two different laps is not a gap.
    a = make_lap([280, 282], times=[0.0, 0.24])
    b = make_lap([280, 282], times=[50.0, 50.24]).assign(LapNumber=2)
    both = pd.concat([a, b], ignore_index=True)
    assert not v.flag_gaps(both).any()


def test_gps_jump_is_flagged():
    car = make_lap([216, 216, 216], times=[0.0, 0.25, 0.5])      # 60 m/s
    # 60 m/s for 0.25 s is 15 m; the last step jumps 100 m. X is in tenths of a metre.
    pos = pd.DataFrame({"Session": "Q", "Driver": "TST", "LapNumber": 1,
                        "Time": [0.0, 0.25, 0.5], "X": [0.0, 150.0, 1150.0], "Y": [0.0, 0.0, 0.0]})
    assert v.flag_gps_jumps(pos, car).tolist() == [False, False, True]


def test_lap_length_check():
    lap = make_lap([200, 200]).assign(Distance=[0.0, 5880.0])
    result = v.lap_length_check(lap, reference_m=5793.0)
    assert result["Error_pct"].iloc[0] == pytest.approx(1.50, abs=0.01)
    assert result["flag_lap_length"].iloc[0]


def test_summary_percentages():
    lap = make_lap([280, 283, 286, 289], times=[0.0, 0.24, 1.40, 1.64])
    table = v.summary(v.validate_car(lap))
    gap_row = table[table["Check"] == "gap"].iloc[0]
    assert gap_row["Flagged"] == 1 and gap_row["Percent"] == 25.0
