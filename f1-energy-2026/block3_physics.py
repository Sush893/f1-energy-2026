"""Block 3: estimate power on full-throttle straights and compare qualifying with the race.

Run from the project folder after block2_validate.py:
    python block3_physics.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import physics as ph

DATA_DIR, RESULTS_DIR, FIG_DIR = Path("data"), Path("results"), Path("figures")
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"


def style(ax):
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
    ax.tick_params(colors=INK_2, labelsize=9)


def straight_label(start: float, end: float) -> str:
    return f"{start * 100:.0f}-{end * 100:.0f}% of lap"


def straights_table(prof: pd.DataFrame, draws: list[ph.CarParams]) -> pd.DataFrame:
    """One row per straight: speeds, nominal power and the power decline with its range."""
    rows = []
    for key, s in prof.groupby(["Session", "Driver", "Team", "StartFrac", "EndFrac"], sort=False):
        declines = []
        for p in draws:
            s2 = s.assign(Power_kW=ph.power_kw(s["Speed_kmh"] / 3.6, s["Accel_ms2"], p))
            declines.append(ph.power_decline(s2))
        med, lo, hi = ph.summarise(declines)
        rows.append(dict(zip(["Session", "Driver", "Team", "StartFrac", "EndFrac"], key),
                         Duration_s=round(s["Time"].iloc[-1] - s["Time"].iloc[0], 1),
                         SpeedStart_kmh=round(s["Speed_kmh"].iloc[0]),
                         SpeedEnd_kmh=round(s["Speed_kmh"].iloc[-1]),
                         MeanPower_kW=round(s["Power_kW"].mean()),
                         Decline_kW=round(med), Decline_low=round(lo), Decline_high=round(hi)))
    return pd.DataFrame(rows)


def q_vs_r(prof: pd.DataFrame, draws: list[ph.CarParams], seed: int = 0) -> pd.DataFrame:
    """Race minus qualifying power at matched speeds, per driver and straight."""
    rng = np.random.default_rng(seed)
    q_all, r_all = prof[prof["Session"] == "Q"], prof[prof["Session"] == "R"]
    rows = []
    for driver in sorted(set(q_all["Driver"]) & set(r_all["Driver"])):
        q, r = q_all[q_all["Driver"] == driver], r_all[r_all["Driver"] == driver]
        for (qs, qe), (rs, re) in ph.match_straights(q, r):
            qd = q[(q["StartFrac"] == qs) & (q["EndFrac"] == qe)]
            rd = r[(r["StartFrac"] == rs) & (r["EndFrac"] == re)]
            nominal = ph.matched_speed_delta(qd, rd, ph.CarParams(), ph.CarParams())
            if len(nominal) < 3:
                continue
            deltas = []
            for p in draws:
                d = ph.matched_speed_delta(qd, rd, p, ph.race_params(p, rng))
                deltas.append(d["Delta_kW"].mean())
            med, lo, hi = ph.summarise(deltas)
            rows.append({"Driver": driver, "Team": q["Team"].iloc[0],
                         "Straight": straight_label(qs, qe), "StartFrac": qs,
                         "SpeedFrom_kmh": nominal["Speed_kmh"].min(),
                         "SpeedTo_kmh": nominal["Speed_kmh"].max(),
                         "Delta_kW": round(med), "Delta_low": round(lo), "Delta_high": round(hi)})
    return pd.DataFrame(rows)


SECTION_EDGES = [0.0, 0.10, 0.30, 0.45, 0.60, 0.80, 1.0]


def add_section(table: pd.DataFrame) -> pd.DataFrame:
    """Group slightly different straight boundaries into common track sections."""
    idx = np.digitize(table["StartFrac"], SECTION_EDGES[1:-1])
    out = table.assign(Section=idx)
    labels = {}
    for i, grp in out.groupby("Section"):
        start, end = grp["StartFrac"].median(), grp["Straight"].str.extract(r"-(\d+)%")[0].astype(float).median()
        labels[i] = f"{start * 100:.0f}-{end:.0f}% of lap"
    return out.assign(SectionLabel=out["Section"].map(labels))


def plot_example(prof: pd.DataFrame, driver: str, start_q: float, tag: str) -> Path:
    """Power along one straight, qualifying vs race, by position on the lap."""
    q = prof[(prof["Session"] == "Q") & (prof["Driver"] == driver)]
    r = prof[(prof["Session"] == "R") & (prof["Driver"] == driver)]
    (qs, qe), (rs, re) = next(pair for pair in ph.match_straights(q, r) if pair[0][0] == start_q)
    qd = q[(q["StartFrac"] == qs) & (q["EndFrac"] == qe)]
    rd = r[(r["StartFrac"] == rs) & (r["EndFrac"] == re)]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for df, colour, name in ((qd, BLUE, "Qualifying"), (rd, ORANGE, "Race")):
        ax.plot(df["LapFrac"] * 100, df["Power_kW"], color=colour, linewidth=2, label=name)
        end = df.iloc[-1]
        ax.annotate(f"{end['Speed_kmh']:.0f} km/h", (end["LapFrac"] * 100, end["Power_kW"]),
                    xytext=(6, 0), textcoords="offset points", fontsize=8.5, color=INK_2, va="center")
    ax.set_title(f"{driver}: estimated power along the straight at {straight_label(qs, qe)} "
                 f"(full throttle throughout)", loc="left", fontsize=10.5, color=INK)
    ax.set_xlabel("Position on lap (%)", color=INK_2)
    ax.set_ylabel("Estimated power at the wheels (kW)", color=INK_2)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=9)
    style(ax)
    fig.tight_layout()
    path = FIG_DIR / f"block3_power_along_straight_{driver}_{tag}.png"
    fig.savefig(path, dpi=150)
    return path


def plot_deltas(table: pd.DataFrame, tag: str) -> Path:
    """Race minus qualifying power at the same speed, one row per track section."""
    t = add_section(table)
    sections = t.groupby("Section")["SectionLabel"].first().sort_index()
    fig, ax = plt.subplots(figsize=(8, 4.4))
    rng = np.random.default_rng(0)
    for row, (sec, label) in enumerate(sections.items()):
        pts = t[t["Section"] == sec]
        jitter = rng.uniform(-0.12, 0.12, len(pts))
        ax.hlines(row + jitter, pts["Delta_low"], pts["Delta_high"], color=INK_2, linewidth=1.5)
        ax.scatter(pts["Delta_kW"], row + jitter, s=40, color=ORANGE, edgecolor="white",
                   linewidth=1.5, zorder=3)
        med = pts["Delta_kW"].median()
        ax.scatter([med], [row], s=160, marker="|", color=INK, linewidth=2.5, zorder=4)
        ax.text(1.01, row, f"median {med:+.0f} kW ({len(pts)} drivers)", fontsize=8,
                color=INK_2, va="center", transform=ax.get_yaxis_transform())
    ax.axvline(0, color=INK_2, linewidth=1)
    ax.set_yticks(range(len(sections)), sections.to_list())
    ax.invert_yaxis()
    ax.set_xlabel("Race minus qualifying power at the same speed (kW)", color=INK_2)
    ax.set_title("Race vs qualifying power by track section (dot = driver, bar = median)",
                 loc="left", fontsize=10.5, color=INK)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=INK_2, labelsize=9)
    fig.tight_layout()
    path = FIG_DIR / f"block3_race_vs_quali_{tag}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="2026_Italian_Grand_Prix")
    parser.add_argument("--draws", type=int, default=300, help="Monte Carlo draws")
    parser.add_argument("--example", default="HAM", help="driver for the example chart")
    args = parser.parse_args()

    car = pd.read_csv(DATA_DIR / f"{args.tag}_car_clean.csv")
    prof, skipped = ph.straight_profiles(car)
    draws = ph.sample_params(args.draws)
    n_straights = prof.groupby(["Session", "Driver", "StartFrac"]).ngroups
    print(f"Straights analysed: {n_straights} | skipped for holes in the data: {skipped}")

    straights = straights_table(prof, draws)
    print("\n=== Power along each straight (nominal assumptions) ===")
    print(f"Mean estimated power: {straights['MeanPower_kW'].min()}-{straights['MeanPower_kW'].max()} kW")
    fading = straights[straights["Decline_high"] < 0]
    print(f"Straights where power fades even at the most favourable drag assumption: "
          f"{len(fading)} of {len(straights)}")
    print(straights.groupby("Session")[["Decline_kW", "Decline_low", "Decline_high"]]
          .median().round().to_string())

    table = q_vs_r(prof, draws)
    print("\n=== Race minus qualifying, same driver, same straight, same speed ===")
    print(table.to_string(index=False))
    print(f"\nPairs with less race power across the whole 5-95% range: "
          f"{int((table['Delta_high'] < 0).sum())} of {len(table)}")

    RESULTS_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(exist_ok=True)
    straights.to_csv(RESULTS_DIR / f"straights_{args.tag}.csv", index=False)
    table.to_csv(RESULTS_DIR / f"race_vs_quali_{args.tag}.csv", index=False)
    print("\n=== Median by track section ===")
    by_section = add_section(table).groupby("SectionLabel")["Delta_kW"].agg(["median", "size"])
    print(by_section.round().to_string())
    example_start = table.loc[table["Driver"] == args.example, "StartFrac"]
    paths = [plot_deltas(table, args.tag)]
    if not example_start.empty:
        paths.append(plot_example(prof, args.example, float(example_start.min()), args.tag))
    print("\nSaved results/ tables and", *paths, sep="\n")


if __name__ == "__main__":
    main()
