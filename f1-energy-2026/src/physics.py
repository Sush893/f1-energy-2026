"""Physics-based power estimation on full-throttle straights.

On a straight at full throttle the power reaching the wheels does three jobs:

    P = v * (m*a + 0.5*rho*CdA*v^2 + Crr*m*g)

    v    speed (m/s)                 a    acceleration (m/s^2)
    m    car + driver + fuel (kg)    rho  air density (kg/m^3)
    CdA  drag area (m^2)             Crr  rolling resistance coefficient

Speed comes from telemetry. Acceleration is the derivative of a smoothed
speed trace (Savitzky-Golay filter). Mass, drag area, air density and rolling
resistance are not public, so they are drawn from ranges in a Monte Carlo
simulation and results are reported as ranges.

Key design choice: comparing a driver's qualifying and race laps at the SAME
SPEED. At equal speed the drag and rolling terms are equal for the same car,
so the difference in power depends only on the difference in acceleration
(and a small mass difference from fuel), not on the unknown drag area.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

G = 9.81

# Full throttle: at or above this throttle % with the brake off.
FULL_THROTTLE = 98.0
# Only straights at least this long are analysed.
MIN_STRAIGHT_S = 4.0
# Skip straights with a hole in the data longer than this (s), e.g. where
# frozen samples were removed during cleaning.
MAX_GAP_IN_STRAIGHT_S = 1.0
# Resampling step and smoothing window for the speed trace.
RESAMPLE_DT = 0.1
SMOOTH_WINDOW_S = 1.1
SMOOTH_POLYORDER = 2
# Trim this much from each end of a straight: the filter is less reliable at
# the edges and the car is still settling at full throttle.
TRIM_S = 0.5


@dataclass(frozen=True)
class CarParams:
    mass_kg: float = 800.0     # 2026 minimum mass is 768 kg incl. driver, plus fuel
    cda_m2: float = 1.1        # drag area in low-drag straight-line mode (assumed)
    rho: float = 1.17          # air density, warm day near sea level
    crr: float = 0.015         # rolling resistance coefficient


# Ranges used in the Monte Carlo simulation (uniform draws).
PARAM_RANGES = {
    "mass_kg": (770.0, 830.0),
    "cda_m2": (0.9, 1.3),
    "rho": (1.13, 1.21),
    "crr": (0.010, 0.020),
}
# Extra fuel a car may carry on its fastest race lap compared with qualifying (kg).
RACE_EXTRA_FUEL_KG = (0.0, 20.0)


def power_kw(v_ms, a_ms2, p: CarParams) -> np.ndarray:
    """Power at the wheels (kW) from speed and acceleration."""
    v = np.asarray(v_ms, dtype=float)
    a = np.asarray(a_ms2, dtype=float)
    force = p.mass_kg * a + 0.5 * p.rho * p.cda_m2 * v**2 + p.crr * p.mass_kg * G
    return v * force / 1000.0


def smooth_speed(time_s, speed_kmh, dt: float = RESAMPLE_DT,
                 window_s: float = SMOOTH_WINDOW_S, polyorder: int = SMOOTH_POLYORDER):
    """Resample speed to an even time grid, then smooth and differentiate it.

    Returns (t, v_ms, a_ms2). Telemetry arrives unevenly (every 0.16-0.44 s)
    and speed is rounded to whole km/h, so differencing raw samples gives very
    noisy acceleration. A Savitzky-Golay filter fits a small polynomial in a
    sliding window, which smooths the noise while keeping real changes.
    """
    t_raw = np.asarray(time_s, dtype=float)
    v_raw = np.asarray(speed_kmh, dtype=float) / 3.6
    t = np.arange(t_raw[0], t_raw[-1] + 1e-9, dt)
    v = np.interp(t, t_raw, v_raw)
    window = int(round(window_s / dt)) | 1          # odd number of points
    if len(t) <= window:
        raise ValueError("segment too short to smooth")
    v_s = savgol_filter(v, window, polyorder)
    a_s = savgol_filter(v, window, polyorder, deriv=1, delta=dt)
    return t, v_s, a_s


def find_straights(lap: pd.DataFrame, min_duration: float = MIN_STRAIGHT_S,
                   max_gap: float = MAX_GAP_IN_STRAIGHT_S) -> tuple[list[pd.DataFrame], int]:
    """Full-throttle runs in one lap. Returns (straights, number skipped for gaps)."""
    full = (lap["Throttle"] >= FULL_THROTTLE) & ~lap["Brake"].astype(bool)
    run_id = (full != full.shift()).cumsum()
    straights, skipped = [], 0
    for _, run in lap[full].groupby(run_id[full]):
        if run["Time"].iloc[-1] - run["Time"].iloc[0] < min_duration:
            continue
        if run["Time"].diff().max() > max_gap:
            skipped += 1
            continue
        straights.append(run)
    return straights, skipped


def straight_profiles(car: pd.DataFrame, p: CarParams = CarParams()) -> tuple[pd.DataFrame, int]:
    """Smoothed speed, acceleration and power along every straight.

    One row per resampled point, with the straight's position as a fraction of
    the lap (fractions are used because integrated lap distance carries ~1 %
    error, see the data quality checks).
    """
    rows, skipped_total = [], 0
    for (session, driver, lap_no), lap in car.groupby(["Session", "Driver", "LapNumber"], sort=False):
        lap_len = lap["Distance"].max()
        straights, skipped = find_straights(lap)
        skipped_total += skipped
        for run in straights:
            t, v, a = smooth_speed(run["Time"], run["Speed"])
            keep = (t >= t[0] + TRIM_S) & (t <= t[-1] - TRIM_S)
            frac = np.interp(t, run["Time"], run["Distance"]) / lap_len
            start = float(run["Distance"].iloc[0] / lap_len)
            end = float(run["Distance"].iloc[-1] / lap_len)
            rows.append(pd.DataFrame({
                "Session": session, "Driver": driver, "Team": run["Team"].iloc[0],
                "LapNumber": lap_no, "StartFrac": round(start, 3), "EndFrac": round(end, 3),
                "Time": t[keep], "LapFrac": frac[keep],
                "Speed_kmh": v[keep] * 3.6, "Accel_ms2": a[keep],
                "Power_kW": power_kw(v[keep], a[keep], p),
            }))
    return (pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()), skipped_total


def power_decline(profile: pd.DataFrame) -> float:
    """Mean power in the last third of a straight minus the first third (kW).

    Negative = power fades while the driver stays at full throttle, the
    signature of the battery running short. Sensitive to the assumed drag
    area, so it is always reported with its Monte Carlo range.
    """
    p = profile["Power_kW"].to_numpy()
    k = len(p) // 3
    return float(p[-k:].mean() - p[:k].mean()) if k else float("nan")


def match_straights(q: pd.DataFrame, r: pd.DataFrame, min_overlap: float = 0.5) -> list[tuple]:
    """Pair each qualifying straight with the race straight covering the same part of the lap."""
    q_list = q[["StartFrac", "EndFrac"]].drop_duplicates().itertuples(index=False)
    r_list = list(r[["StartFrac", "EndFrac"]].drop_duplicates().itertuples(index=False))
    pairs = []
    for qs, qe in q_list:
        best, best_ov = None, 0.0
        for rs, re in r_list:
            overlap = min(qe, re) - max(qs, rs)
            shorter = min(qe - qs, re - rs)
            if shorter > 0 and overlap / shorter > best_ov:
                best, best_ov = (rs, re), overlap / shorter
        if best is not None and best_ov >= min_overlap:
            pairs.append(((qs, qe), best))
    return pairs


def matched_speed_delta(q: pd.DataFrame, r: pd.DataFrame, pq: CarParams, pr: CarParams,
                        bin_kmh: float = 5.0, min_points: int = 2) -> pd.DataFrame:
    """Race minus qualifying power at the same speeds, on one straight.

    Speeds are binned; only bins both laps pass through are compared.
    """
    def binned(df, p):
        power = power_kw(df["Speed_kmh"] / 3.6, df["Accel_ms2"], p)
        bins = (df["Speed_kmh"] // bin_kmh) * bin_kmh
        out = pd.DataFrame({"bin": bins, "P": power}).groupby("bin")["P"].agg(["mean", "size"])
        return out[out["size"] >= min_points]["mean"]

    bq, br = binned(q, pq), binned(r, pr)
    common = bq.index.intersection(br.index)
    return pd.DataFrame({"Speed_kmh": common + bin_kmh / 2,
                         "Q_kW": bq[common].to_numpy(), "R_kW": br[common].to_numpy(),
                         "Delta_kW": (br[common] - bq[common]).to_numpy()})


def sample_params(n: int, seed: int = 0) -> list[CarParams]:
    """Draw `n` parameter sets uniformly from PARAM_RANGES."""
    rng = np.random.default_rng(seed)
    draws = {k: rng.uniform(lo, hi, n) for k, (lo, hi) in PARAM_RANGES.items()}
    return [CarParams(**{k: float(draws[k][i]) for k in draws}) for i in range(n)]


def race_params(p: CarParams, rng: np.random.Generator) -> CarParams:
    """Same car in the race, possibly carrying a little more fuel."""
    return replace(p, mass_kg=p.mass_kg + rng.uniform(*RACE_EXTRA_FUEL_KG))


def summarise(values, low: float = 5, high: float = 95) -> tuple[float, float, float]:
    """Median and a 5-95 % range of Monte Carlo results."""
    arr = np.asarray(values, dtype=float)
    return (float(np.nanmedian(arr)), float(np.nanpercentile(arr, low)),
            float(np.nanpercentile(arr, high)))
