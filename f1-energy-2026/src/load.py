"""Load F1 sessions and lap telemetry with FastF1.

Everything that touches the FastF1 API lives here, so the validation and
physics code can work on plain pandas DataFrames (and be tested offline).
"""
from __future__ import annotations

from pathlib import Path

import fastf1
import pandas as pd
from fastf1.exceptions import DataNotLoadedError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / "cache"

# Circuits with long full-throttle straights, where energy limits show up most.
LONG_STRAIGHT_EVENTS = (
    "Italian",      # Monza
    "Mexico",
    "Azerbaijan",   # Baku
    "Saudi",        # Jeddah
    "Las Vegas",
    "Belgian",      # Spa
    "Bahrain",
)

CAR_COLUMNS = ["SessionTime", "Time", "Distance", "Speed", "RPM", "nGear",
               "Throttle", "Brake", "DRS"]
POS_COLUMNS = ["SessionTime", "Time", "X", "Y", "Z", "Status"]


def enable_cache(path: Path = CACHE_DIR) -> None:
    """Cache downloads locally so each session is only downloaded once."""
    path.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(path))


def _as_utc(series: pd.Series) -> pd.Series:
    times = pd.to_datetime(series)
    if times.dt.tz is None:
        return times.dt.tz_localize("UTC")
    return times.dt.tz_convert("UTC")


def completed_events(year: int) -> pd.DataFrame:
    """Race weekends in `year` whose race has already taken place."""
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    race_time = _as_utc(schedule["Session5DateUtc"])
    done = schedule.loc[race_time < pd.Timestamp.now(tz="UTC")]
    return done[["RoundNumber", "EventName", "Location", "EventDate"]]


def candidate_events(year: int, preferred: tuple[str, ...] = LONG_STRAIGHT_EVENTS) -> list[str]:
    """Completed events, long-straight circuits first, then the rest (newest first)."""
    done = completed_events(year)
    if done.empty:
        raise RuntimeError(f"No completed {year} events found in the schedule.")
    names = [str(n) for n in done["EventName"]]
    ordered = []
    for key in preferred:
        ordered += [n for n in names if key.lower() in n.lower() and n not in ordered]
    ordered += [n for n in reversed(names) if n not in ordered]
    return ordered


def pick_event(year: int, preferred: tuple[str, ...] = LONG_STRAIGHT_EVENTS) -> str:
    """Pick a completed event, preferring circuits with long straights."""
    return candidate_events(year, preferred)[0]


def has_laps(session) -> bool:
    """True if lap timing loaded (it can fail even when telemetry loads)."""
    try:
        return len(session.laps) > 0
    except DataNotLoadedError:
        return False


def load_session(year: int, event: str | int, session_type: str = "Q"):
    """Load laps and telemetry for one session (e.g. 'Q' qualifying, 'R' race)."""
    session = fastf1.get_session(year, event, session_type)
    session.load(laps=True, telemetry=True, weather=False, messages=False)
    return session


def fastest_laps(session, n_drivers: int = 10) -> dict:
    """Each driver's fastest lap, for the top `n_drivers` in the session results."""
    laps = {}
    for abbr in session.results["Abbreviation"].dropna().head(n_drivers):
        lap = session.laps.pick_drivers(abbr).pick_fastest()
        if lap is None or pd.isna(lap["LapTime"]):
            continue
        laps[abbr] = lap
    return laps


def _seconds(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Convert timedelta columns to float seconds (easier to save and model)."""
    out = df.copy()
    for col in columns:
        if col in out and pd.api.types.is_timedelta64_dtype(out[col]):
            out[col] = out[col].dt.total_seconds()
    return out


def lap_car_data(lap) -> pd.DataFrame:
    """Speed, throttle, brake etc. for one lap, with distance along the lap."""
    car = lap.get_car_data().add_distance()
    cols = [c for c in CAR_COLUMNS if c in car.columns]
    return _seconds(car[cols], ["SessionTime", "Time"])


def lap_pos_data(lap) -> pd.DataFrame:
    """X/Y/Z position samples for one lap."""
    pos = lap.get_pos_data()
    cols = [c for c in POS_COLUMNS if c in pos.columns]
    return _seconds(pos[cols], ["SessionTime", "Time"])


def export_laps(laps: dict, session_label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stack car and position data for several laps into two tidy tables."""
    car_frames, pos_frames = [], []
    for abbr, lap in laps.items():
        info = {
            "Session": session_label,
            "Driver": abbr,
            "Team": lap["Team"],
            "LapNumber": int(lap["LapNumber"]),
            "LapTime_s": lap["LapTime"].total_seconds(),
        }
        car_frames.append(lap_car_data(lap).assign(**info))
        pos_frames.append(lap_pos_data(lap).assign(**info))
    return pd.concat(car_frames, ignore_index=True), pd.concat(pos_frames, ignore_index=True)