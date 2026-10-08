"""Publish the weather-enriched batted-ball dataset to HuggingFace.

Builds a clean, documented parquet from the enriched pool and uploads it to a
dataset repo. Requires a HuggingFace write token (``hf auth login`` or the
HF_TOKEN environment variable).

Usage:
    uv run python scripts/publish_hf.py --repo doyled-it/statcast-batted-balls-weather
    uv run python scripts/publish_hf.py --repo <...> --dry-run   # build locally only
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from frb import config as C
from frb import physics as P

# Published columns, renamed to clean public names.
RENAME = {
    "temp_f_use": "temperature_f",
    "rh_pct_use": "humidity_pct",
    "pressure_hpa_use": "pressure_hpa",
    "wind_along_mph": "wind_along_flight_mph",
    "player_name": "batter_name",
    "estimated_ba_using_speedangle": "xba",
    "estimated_woba_using_speedangle": "xwoba",
    "estimated_slg_using_speedangle": "xslg",
}
PUBLISH_COLS = [
    # identity and game context
    "game_pk",
    "game_date",
    "game_year",
    "game_type",
    "home_team",
    "away_team",
    "venue_id",
    "batter_name",
    "batter",
    "pitcher",
    "stand",
    "p_throws",
    "inning",
    "inning_topbot",
    "outs_when_up",
    "balls",
    "strikes",
    "on_1b",
    "on_2b",
    "on_3b",
    "at_bat_number",
    "pitch_number",
    "home_score",
    "away_score",
    "bat_score",
    "fld_score",
    "events",
    "description",
    "des",
    "bb_type",
    "hit_location",
    "if_fielding_alignment",
    "of_fielding_alignment",
    # batted ball and outcome
    "launch_speed",
    "launch_angle",
    "spray_deg",
    "hit_distance_sc",
    "launch_speed_angle",
    "is_barrel",
    "xba",
    "xwoba",
    "xslg",
    "woba_value",
    "babip_value",
    "iso_value",
    "delta_run_exp",
    "delta_home_win_exp",
    "carry_vs_expected_ft",
    # bat tracking (2024+)
    "bat_speed",
    "swing_length",
    "attack_angle",
    "attack_direction",
    "swing_path_tilt",
    "launch_minus_attack_deg",
    # the pitch that was hit
    "pitch_type",
    "pitch_name",
    "release_speed",
    "release_spin_rate",
    "effective_speed",
    "release_extension",
    "plate_x",
    "plate_z",
    "pfx_x",
    "pfx_z",
    "zone",
    "spin_axis",
    # player context
    "age_bat",
    "age_pit",
    "n_thruorder_pitcher",
    # conditions (weather)
    "roof_type",
    "roof_closed",
    "elevation_m",
    "temperature_f",
    "humidity_pct",
    "pressure_hpa",
    "air_density",
    "wind_mph",
    "wind_dir",
    "wind_along_flight_mph",
]

CARD = """---
license: other
task_categories:
- tabular-regression
tags:
- baseball
- statcast
- weather
- sports
pretty_name: Statcast Batted Balls with Weather and Air Density
size_categories:
- 1M<n<10M
---

# Statcast batted balls with per-game weather, air density, and field wind

Every regular-season and postseason batted ball with a measured exit velocity,
launch angle, and Statcast hit distance, 2015 through 2026, joined to the
conditions it was hit in. The base file covers the history; `data/updates/`
holds daily partitions appended through the season.

## What the join buys you

![Hotter air, more home runs](figures/temp_vs_hr.png)

![Retractable roofs, open vs closed](figures/roof_effect.png)

![Park carry factors](figures/park_carry_factors.png)

## What is here

Each row is one batted ball with its full Statcast context, plus the conditions
it was hit in:

- **Identity and game state**: batter and pitcher (names and ids), both teams,
  inning, count, outs, baserunners, score, and the play description.
- **Batted ball**: exit velocity, launch angle, derived spray angle, projected
  hit distance, the Statcast barrel code and an `is_barrel` flag, and bat
  tracking where available (2024+): bat speed, attack angle, swing path tilt,
  swing length, plus a derived `launch_minus_attack_deg` (positive = undercut/backspin,
  negative = overcut/topspin).
- **Expected outcomes** (Statcast models): `xba`, `xwoba`, `xslg`, plus
  `woba_value`, `babip_value`, `iso_value`, and the run- and win-expectancy
  deltas. A derived `carry_vs_expected_ft` gives how far the ball carried versus
  a physics model evaluated at its own air density.
- **The pitch that was hit**: type, release speed and spin, movement, plate
  location, zone, and spin axis.
- **Conditions**: roof state and a roof-closed flag; true **air density** from
  temperature, relative humidity, and surface pressure (official game-report
  temperature where present, humidity and pressure from Open-Meteo's hourly
  reanalysis at the park); and **wind** as MLB reports it plus
  `wind_along_flight_mph`, the component along each ball's own flight direction.
  Roof-closed games use controlled still air at 72 F with no wind.

## Key columns

| column | meaning |
|---|---|
| `batter_name`, `batter`, `pitcher` | batter name, batter id, pitcher id |
| `launch_speed`, `launch_angle` | exit velocity (mph), launch angle (deg) |
| `spray_deg` | spray angle; negative toward left field |
| `hit_distance_sc` | Statcast projected carry (ft) |
| `is_barrel` | True for a Statcast "barrel" |
| `xba`, `xwoba`, `xslg` | Statcast expected stats from exit velocity and angle |
| `carry_vs_expected_ft` | carry minus the physics model at this ball's air density |
| `launch_minus_attack_deg` | launch minus attack angle (2024+); + undercut/backspin, - overcut/topspin |
| `attack_angle`, `bat_speed` | bat tracking (2024+) |
| `pitch_type`, `release_speed`, `release_spin_rate` | the pitch that was hit |
| `air_density` | computed air density (kg/m^3) |
| `wind_along_flight_mph` | along-flight wind; positive aids carry |
| `roof_closed` | True when played in still, controlled air |

## Sources and caveats

- Statcast via Baseball Savant (2015-2025 through the
  `Jensen-holm/statcast-era-pitches` mirror, 2026 pulled directly from Savant).
- Game weather and roof from the MLB StatsAPI schedule.
- Humidity and pressure from the Open-Meteo historical reanalysis archive.

Caveats: the reported wind is a station or grid reading, not the wind inside the
stadium bowl, so `wind_along_flight_mph` overstates the wind the ball truly
feels. Roof state for retractable parks is best-effort from the game report.
`hit_distance_sc` is a Statcast projection, not a surveyed landing point. This is
derived data from public MLB Statcast; use it for research and credit the
sources above.

Built with https://github.com/doyled-it/big-eiffel
"""


def clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the public names, add derived columns, and select for publishing."""
    # Drop the raw Open-Meteo columns so the resolved *_use columns can take
    # their clean public names without colliding.
    df = df.drop(columns=[c for c in ("temp_f", "rh_pct", "pressure_hpa") if c in df.columns])
    df = df.rename(columns=RENAME)

    # Derived columns.
    if "launch_speed_angle" in df.columns:
        df["is_barrel"] = df["launch_speed_angle"].eq(6)  # Statcast barrel code
    if {"launch_angle", "attack_angle"} <= set(df.columns):
        # Signed vertical gap between the launch and the bat's path (2024+):
        # positive is an undercut (backspin), negative an overcut (topspin); the
        # magnitude scales the spin.
        df["launch_minus_attack_deg"] = df["launch_angle"] - df["attack_angle"]
    if {"launch_speed", "launch_angle", "air_density", "hit_distance_sc"} <= set(df.columns):
        # How far the ball went versus the fitted physics model at its own air
        # density (positive means it carried past expectation). No re-fit.
        exp = P.carry_from_knots(
            df["launch_speed"].to_numpy(float),
            df["launch_angle"].to_numpy(float),
            df["air_density"].to_numpy(float),
        )
        df["carry_vs_expected_ft"] = np.round(df["hit_distance_sc"].to_numpy(float) - exp, 1)

    cols = [c for c in PUBLISH_COLS if c in df.columns]
    return df[cols].copy()


def build_clean() -> pd.DataFrame:
    return clean_frame(pd.read_parquet(C.ENRICHED_PARQUET))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="doyled-it/statcast-batted-balls-weather")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    clean = build_clean()
    out = C.DATA / "statcast_batted_balls_weather.parquet"
    clean.to_parquet(out, index=False)
    card = C.DATA / "DATASET_CARD.md"
    card.write_text(CARD)
    print(f"built {len(clean):,} rows, {len(clean.columns)} cols -> {out}")

    if args.dry_run:
        print("dry run: not uploading")
        return

    from huggingface_hub import HfApi, get_token

    if not get_token():
        print("No HuggingFace token. Run `hf auth login` (or set HF_TOKEN), then re-run.")
        return
    api = HfApi()
    api.create_repo(args.repo, repo_type="dataset", exist_ok=True)
    api.upload_file(
        path_or_fileobj=str(out),
        path_in_repo="data/batted_balls_weather.parquet",
        repo_id=args.repo,
        repo_type="dataset",
    )
    # Card figures, if they have been generated.
    fig_dir = C.DATA / "card_figures"
    for name in ("temp_vs_hr.png", "roof_effect.png", "park_carry_factors.png"):
        fp = fig_dir / name
        if fp.exists():
            api.upload_file(
                path_or_fileobj=str(fp), path_in_repo=f"figures/{name}", repo_id=args.repo, repo_type="dataset"
            )
    api.upload_file(path_or_fileobj=str(card), path_in_repo="README.md", repo_id=args.repo, repo_type="dataset")
    print(f"published https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
