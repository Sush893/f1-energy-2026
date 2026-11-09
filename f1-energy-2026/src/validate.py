"""Data quality checks for FastF1 car and position telemetry.

Each check adds a boolean ``flag_*`` column instead of deleting rows, so the
effect of every check can be measured and reported. Thresholds are physical
limits or multiples of the observed sampling interval, and are documented
next to each constant.

Input: the tidy tables written by ``src.load.export_laps`` (times in seconds,
one or more laps identified by Session + Driver + LapNumber).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LAP_KEYS = ["Session", "Driver", "LapNumber"]

# Car data arrives roughly every 0.24 s (~4 Hz). A gap of more than twice that
# means at least one sample is missing.
MAX_GAP_S = 0.5

# Peak F1 braking is about 5-6 g and peak acceleration well under 3 g.
# Anything beyond these limits between two samples cannot be real driving.
MAX_DECEL_MS2 = 60.0   # ~6.1 g
MAX_ACCEL_MS2 = 30.0   # ~3.1 g

# A "frozen" signal repeats every channel exactly. RPM changes almost every
# sample on track, so identical speed, RPM, gear, throttle and brake for this
# many samples in a row means the feed stopped updating.
FROZEN_MIN_RUN = 3
FROZEN_CHANNELS = ["Speed", "RPM", "nGear", "Throttle", "Brake"]

# Throttle is a percentage. Values above 100 are a known scaling quirk.
THROTTLE_MAX = 100.0

# Throttle at or above this level with the brake on is unusual. Brief overlaps
# can be real, so this is flagged as suspicious rather than removed.
OVERLAP_THROTTLE = 10.0

# FastF1 X/Y positions are in tenths of a metre.
POS_UNITS_PER_M = 10.0
# A GPS step may exceed speed x time by this factor plus a small tolerance.
GPS_STEP_FACTOR = 1.5
GPS_STEP_TOL_M = 5.0

# Lap length built up from speed should be within this fraction of the
# official circuit length.
LAP_LENGTH_TOL = 0.01


def _grouped(df: pd.DataFrame):
    keys = [k for k in LAP_KEYS if k in df.columns]
    return df.groupby(keys, sort=False) if keys else df.groupby(lambda _: 0)


def implied_accel(car: pd.DataFrame) -> pd.Series:
    """Acceleration (m/s^2) between consecutive samples within each lap."""
    g = _grouped(car)
    dv = g["Speed"].diff() / 3.6
    dt = g["Time"].diff()
    return dv / dt


def flag_gaps(car: pd.DataFrame, max_gap: float = MAX_GAP_S) -> pd.Series:
    """Sample arrives more than `max_gap` seconds after the previous one."""
    return (_grouped(car)["Time"].diff() > max_gap).rename("flag_gap")


def flag_impossible_accel(car: pd.DataFrame,
                          max_decel: float = MAX_DECEL_MS2,
                          max_accel: float = MAX_ACCEL_MS2) -> pd.Series:
    """Speed change from the previous sample is physically impossible."""
    a = implied_accel(car)
    return ((a < -max_decel) | (a > max_accel)).rename("flag_impossible_accel")


def flag_frozen(car: pd.DataFrame, min_run: int = FROZEN_MIN_RUN,
                channels: list[str] = FROZEN_CHANNELS) -> pd.Series:
    """Sample belongs to a run of `min_run`+ samples with every channel identical."""
    cols = [c for c in channels if c in car.columns]
    flags = pd.Series(False, index=car.index)
    for _, lap in _grouped(car):
        same = (lap[cols] == lap[cols].shift()).all(axis=1)
        run_id = (~same).cumsum()
        run_len = run_id.map(run_id.value_counts())
        flags.loc[lap.index] = (run_len >= min_run).to_numpy()
    return flags.rename("flag_frozen")


def flag_throttle_over_100(car: pd.DataFrame) -> pd.Series:
    """Throttle reading above 100 %."""
    return (car["Throttle"] > THROTTLE_MAX).rename("flag_throttle_over_100")


def flag_throttle_brake_overlap(car: pd.DataFrame,
                                min_throttle: float = OVERLAP_THROTTLE) -> pd.Series:
    """Throttle at or above `min_throttle` % while the brake is on (suspicious)."""
    brake = car["Brake"].astype(bool)
    return ((car["Throttle"] >= min_throttle) & brake).rename("flag_throttle_brake_overlap")


def validate_car(car: pd.DataFrame) -> pd.DataFrame:
    """Return `car` with one flag column per check and an overall `flag_any`.

    `flag_any` excludes the throttle-brake overlap, which is suspicious but can
    be real driving.
    """
    out = car.copy()
    for check in (flag_gaps, flag_impossible_accel, flag_frozen,
                  flag_throttle_over_100, flag_throttle_brake_overlap):
        flag = check(car)
        out[flag.name] = flag.fillna(False).astype(bool)
    hard = ["flag_gap", "flag_impossible_accel", "flag_frozen", "flag_throttle_over_100"]
    out["flag_any"] = out[hard].any(axis=1)
    return out


def flag_gps_jumps(pos: pd.DataFrame, car: pd.DataFrame,
                   factor: float = GPS_STEP_FACTOR,
                   tol_m: float = GPS_STEP_TOL_M) -> pd.Series:
    """Position moves further between samples than the car's speed allows."""
    flags = pd.Series(False, index=pos.index, name="flag_gps_jump")
    keys = [k for k in LAP_KEYS if k in pos.columns and k in car.columns]
    car_laps = dict(tuple(car.groupby(keys, sort=False))) if keys else {0: car}
    pos_laps = pos.groupby(keys, sort=False) if keys else [(0, pos)]
    for key, lap in pos_laps:
        c = car_laps.get(key)
        if c is None or len(lap) < 2:
            continue
        speed_ms = np.interp(lap["Time"], c["Time"], c["Speed"]) / 3.6
        step_m = np.hypot(lap["X"].diff(), lap["Y"].diff()) / POS_UNITS_PER_M
        allowed = speed_ms * lap["Time"].diff() * factor + tol_m
        flags.loc[lap.index] = (step_m > allowed).fillna(False).to_numpy()
    return flags


def lap_length_check(car: pd.DataFrame, reference_m: float,
                     tol: float = LAP_LENGTH_TOL) -> pd.DataFrame:
    """Compare each lap's integrated distance with the official circuit length."""
    keys = [k for k in LAP_KEYS if k in car.columns]
    laps = car.groupby(keys, sort=False)["Distance"].max().rename("LapLength_m").reset_index()
    laps["Error_pct"] = (laps["LapLength_m"] / reference_m - 1.0) * 100.0
    laps["flag_lap_length"] = laps["Error_pct"].abs() > tol * 100.0
    return laps


def summary(flagged: pd.DataFrame) -> pd.DataFrame:
    """Count and percentage of samples flagged by each check, per session."""
    flag_cols = [c for c in flagged.columns if c.startswith("flag_")]
    by = "Session" if "Session" in flagged.columns else None
    groups = flagged.groupby(by) if by else [("all", flagged)]
    rows = []
    for name, df in groups:
        for col in flag_cols:
            n = int(df[col].sum())
            rows.append({"Session": name, "Check": col.removeprefix("flag_"),
                         "Flagged": n, "Samples": len(df),
                         "Percent": round(100.0 * n / len(df), 2) if len(df) else 0.0})
    return pd.DataFrame(rows)


def clean_car(flagged: pd.DataFrame) -> pd.DataFrame:
    """Drop frozen and impossible samples and cap throttle at 100 %.

    Gaps are kept but stay flagged: later steps skip straights that contain
    one, rather than interpolating across missing data.
    """
    keep = ~(flagged["flag_frozen"] | flagged["flag_impossible_accel"])
    out = flagged.loc[keep].copy()
    out["Throttle"] = out["Throttle"].clip(upper=THROTTLE_MAX)
    return out
