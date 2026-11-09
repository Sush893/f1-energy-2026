"""Block 1: confirm 2026 data loads, inspect the telemetry, export it to CSV.

Run from the project folder:
    python block1_check.py
or pick the event yourself:
    python block1_check.py --event "Italian Grand Prix"

The first run downloads data and can take a few minutes.
"""
from __future__ import annotations

import argparse
import platform
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save figures without opening windows
import matplotlib.pyplot as plt
import pandas as pd
import fastf1

from src.load import (candidate_events, completed_events, enable_cache, export_laps,
                      fastest_laps, has_laps, load_session)

MAX_ATTEMPTS = 4

DATA_DIR = Path("data")
FIG_DIR = Path("figures")


def sample_rate(times_s: pd.Series) -> tuple[float, float]:
    """Median gap between samples (s) and the matching rate (Hz)."""
    gaps = times_s.diff().dropna()
    median_gap = float(gaps.median())
    return median_gap, (1.0 / median_gap if median_gap > 0 else float("nan"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--event", default=None, help="event name or round number")
    parser.add_argument("--session", default="Q", help="Q = qualifying, R = race")
    parser.add_argument("--drivers", type=int, default=10)
    parser.add_argument("--debug", action="store_true", help="show FastF1's detailed log")
    args = parser.parse_args()

    if args.debug:
        fastf1.set_log_level("DEBUG")

    print("=== 1. Environment ===")
    print(f"Python {platform.python_version()} | FastF1 {fastf1.__version__} | pandas {pd.__version__}")
    enable_cache()

    print(f"\n=== 2. Completed {args.year} events ===")
    done = completed_events(args.year)
    print(done.to_string(index=False))

    events = [args.event] if args.event else candidate_events(args.year)[:MAX_ATTEMPTS]
    session, event, failed = None, None, []
    for candidate in events:
        print(f"\n=== 3. Loading {args.year} {candidate}, session {args.session} ===")
        loaded = load_session(args.year, candidate, args.session)
        if has_laps(loaded):
            session, event = loaded, candidate
            break
        print(f"!! Timing data did not load for {candidate}, so laps are unavailable.")
        failed.append(candidate)

    if session is None:
        print("\nNo laps could be loaded for:", ", ".join(failed))
        print("Next step: run again with --debug and paste the full output,")
        print('or try another event, e.g.  python block1_check.py --event "Mexico City Grand Prix"')
        return

    if failed:
        print(f"Skipped (no timing data): {', '.join(failed)}")
    print(f"Using: {event} | Laps: {len(session.laps)} | Drivers: {len(session.drivers)}")

    laps = fastest_laps(session, args.drivers)
    print(f"\n=== 4. Fastest laps ({len(laps)} drivers) ===")
    for abbr, lap in laps.items():
        print(f"{abbr:4} {lap['Team']:28} {lap['LapTime'].total_seconds():8.3f} s  (lap {int(lap['LapNumber'])})")

    car, pos = export_laps(laps, session_label=args.session)

    print("\n=== 5. Telemetry contents ===")
    print("Car data columns:", list(car.columns))
    print("Position data columns:", list(pos.columns))
    print(f"Car data rows: {len(car)} | Position rows: {len(pos)}")
    print("\nCar data summary:")
    print(car[["Speed", "RPM", "nGear", "Throttle"]].describe().round(1).to_string())
    print("Brake values:", sorted(car["Brake"].dropna().unique().tolist()))
    if "DRS" in car:
        print("DRS values:", sorted(car["DRS"].dropna().unique().tolist()))

    print("\n=== 6. Sampling rate (first driver) ===")
    first = next(iter(laps))
    gap_car, hz_car = sample_rate(car.loc[car["Driver"] == first, "Time"])
    gap_pos, hz_pos = sample_rate(pos.loc[pos["Driver"] == first, "Time"])
    print(f"Car data: median gap {gap_car:.3f} s (~{hz_car:.1f} Hz)")
    print(f"Position: median gap {gap_pos:.3f} s (~{hz_pos:.1f} Hz)")

    DATA_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(exist_ok=True)
    tag = f"{args.year}_{str(event).replace(' ', '_')}_{args.session}"
    car_path = DATA_DIR / f"{tag}_car.csv"
    pos_path = DATA_DIR / f"{tag}_pos.csv"
    car.to_csv(car_path, index=False)
    pos.to_csv(pos_path, index=False)

    fig, ax = plt.subplots(figsize=(11, 4))
    for abbr in list(laps)[:3]:
        d = car[car["Driver"] == abbr]
        ax.plot(d["Distance"], d["Speed"], label=abbr, linewidth=1)
    ax.set_xlabel("Distance along lap (m)")
    ax.set_ylabel("Speed (km/h)")
    ax.set_title(f"{args.year} {event} – fastest laps, session {args.session}")
    ax.legend()
    fig.tight_layout()
    fig_path = FIG_DIR / f"block1_speed_{tag}.png"
    fig.savefig(fig_path, dpi=150)

    print("\n=== 7. Saved ===")
    print(car_path, pos_path, fig_path, sep="\n")
    print("\nDone. Paste this whole output back, and attach the two CSV files and the PNG.")


if __name__ == "__main__":
    main()