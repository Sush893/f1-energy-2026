"""Block 2: run the data quality checks on exported telemetry.

Run from the project folder after block1_check.py has exported the CSVs:
    python block2_validate.py
    python block2_validate.py --tag 2026_Italian_Grand_Prix --lap-length 5793
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src import validate as v

DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
FIG_DIR = Path("figures")

INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"


def load(tag: str, kind: str) -> pd.DataFrame:
    files = sorted(DATA_DIR.glob(f"{tag}_*_{kind}.csv"))
    if not files:
        raise SystemExit(f"No {kind} CSVs found for '{tag}' in {DATA_DIR}/")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def style(ax):
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
    ax.tick_params(colors=INK_2, labelsize=9)


def plot(flagged: pd.DataFrame, tag: str) -> Path:
    """Left: one lap with flagged samples marked. Right: where gaps happen."""
    frozen_laps = flagged[flagged["flag_frozen"]].groupby(v.LAP_KEYS, sort=False).size()
    key = frozen_laps.idxmax() if not frozen_laps.empty else tuple(flagged[v.LAP_KEYS].iloc[0])
    mask = (flagged[v.LAP_KEYS] == pd.Series(key, index=v.LAP_KEYS)).all(axis=1)
    lap = flagged[mask]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1.6, 1]})

    ax1.plot(lap["Distance"], lap["Speed"], color=BLUE, linewidth=1.5, label="Speed")
    frozen = lap[lap["flag_frozen"]]
    jumps = lap[lap["flag_impossible_accel"]]
    gaps = lap[lap["flag_gap"]]
    ax1.scatter(frozen["Distance"], frozen["Speed"], s=36, marker="s", color=ORANGE,
                edgecolor="white", linewidth=1, zorder=3, label="Frozen signal")
    ax1.scatter(jumps["Distance"], jumps["Speed"], s=56, marker="X", color=INK,
                edgecolor="white", linewidth=1, zorder=4, label="Impossible speed change")
    ax1.scatter(gaps["Distance"], gaps["Speed"], s=36, marker="^", facecolor="white",
                edgecolor=INK_2, linewidth=1.2, zorder=3, label="Gap > 0.5 s before sample")
    session, driver, lap_no = key
    ax1.set_title(f"{driver}, session {session}, lap {lap_no}: flagged samples",
                  loc="left", fontsize=11, color=INK)
    ax1.set_xlabel("Distance along lap (m)", color=INK_2)
    ax1.set_ylabel("Speed (km/h)", color=INK_2)
    ax1.legend(frameon=False, fontsize=8.5, loc="lower right")
    style(ax1)

    bins = list(range(60, 361, 20))
    all_speed = flagged["Speed"]
    gap_speed = flagged.loc[flagged["flag_gap"], "Speed"]
    ax2.hist(all_speed, bins=bins, density=True, color=GRID, edgecolor="white", label="All samples")
    ax2.hist(gap_speed, bins=bins, density=True, histtype="step", color=ORANGE, linewidth=2,
             label=f"Samples after a gap (n={len(gap_speed)})")
    ax2.set_title("Gaps are more common at high speed", loc="left", fontsize=11, color=INK)
    ax2.set_xlabel("Speed (km/h)", color=INK_2)
    ax2.set_ylabel("Share of samples (density)", color=INK_2)
    ax2.legend(frameon=False, fontsize=8.5, loc="upper left")
    style(ax2)

    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    path = FIG_DIR / f"block2_data_quality_{tag}.png"
    fig.savefig(path, dpi=150)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="2026_Italian_Grand_Prix")
    parser.add_argument("--lap-length", type=float, default=5793.0,
                        help="official circuit length in metres (Monza: 5793)")
    args = parser.parse_args()

    car = load(args.tag, "car")
    pos = load(args.tag, "pos")
    flagged = v.validate_car(car)
    flagged["flag_gps_jump"] = False  # car rows; GPS is checked on position rows below
    pos_flags = v.flag_gps_jumps(pos, car)

    print("=== Car data checks ===")
    table = v.summary(flagged.drop(columns="flag_gps_jump"))
    print(table.to_string(index=False))

    print("\n=== GPS check (position data) ===")
    print(f"{int(pos_flags.sum())} of {len(pos)} position samples jump further than speed allows "
          f"({100 * pos_flags.mean():.2f}%)")

    print("\n=== Frozen samples by driver ===")
    frozen = flagged[flagged["flag_frozen"]].groupby(["Team", "Driver", "Session"]).size()
    print(frozen.rename("FrozenSamples").to_string() if not frozen.empty else "None")

    print(f"\n=== Lap length vs official {args.lap_length:.0f} m ===")
    lengths = v.lap_length_check(car, args.lap_length)
    print(lengths.round(2).to_string(index=False))

    print("\n=== Speed where gaps happen ===")
    print(f"Median speed after a gap: {flagged.loc[flagged['flag_gap'], 'Speed'].median():.0f} km/h "
          f"(all samples: {flagged['Speed'].median():.0f} km/h)")

    cleaned = v.clean_car(flagged)
    print(f"\nCleaning removed {len(flagged) - len(cleaned)} of {len(flagged)} samples "
          f"({100 * (1 - len(cleaned) / len(flagged)):.2f}%) and capped throttle at 100%.")

    RESULTS_DIR.mkdir(exist_ok=True)
    table.to_csv(RESULTS_DIR / f"data_quality_summary_{args.tag}.csv", index=False)
    lengths.to_csv(RESULTS_DIR / f"lap_length_check_{args.tag}.csv", index=False)
    cleaned.to_csv(DATA_DIR / f"{args.tag}_car_clean.csv", index=False)
    fig_path = plot(flagged, args.tag)
    print(f"\nSaved results/ tables, data/{args.tag}_car_clean.csv and {fig_path}")


if __name__ == "__main__":
    main()
