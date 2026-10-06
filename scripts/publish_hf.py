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

import pandas as pd

from frb import config as C

# Published columns, renamed to clean public names.
RENAME = {
    "temp_f_use": "temperature_f",
    "rh_pct_use": "humidity_pct",
    "pressure_hpa_use": "pressure_hpa",
    "wind_along_mph": "wind_along_flight_mph",
}
PUBLISH_COLS = [
    "game_pk",
    "game_date",
    "game_year",
    "game_type",
    "home_team",
    "venue_id",
    "stand",
    "events",
    "bb_type",
    "launch_speed",
    "launch_angle",
    "spray_deg",
    "hit_distance_sc",
    "bat_speed",
    "swing_length",
    "attack_angle",
    "swing_path_tilt",
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

- **Batted ball** (MLB Statcast): exit velocity, launch angle, derived spray
  angle, projected hit distance, batted-ball type, batter handedness, and the
  bat-tracking fields where available (2024+).
- **Game conditions** (MLB StatsAPI): roof type and a roof-closed flag, and the
  venue.
- **Air density** computed from temperature, relative humidity, and surface
  pressure. Temperature is the official game report where present; humidity and
  pressure come from Open-Meteo's hourly reanalysis at the park, matched to the
  game hour. Roof-closed games use controlled still air at 72 F.
- **Wind** as MLB reports it (speed and a field-relative direction such as "Out
  To CF"), plus `wind_along_flight_mph`, the component along each ball's own
  flight direction (positive is a tailwind). Zero for roof-closed games.

## Key columns

| column | meaning |
|---|---|
| `launch_speed`, `launch_angle` | exit velocity (mph), launch angle (deg) |
| `spray_deg` | spray angle; negative toward left field |
| `hit_distance_sc` | Statcast projected carry (ft) |
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

Built with https://github.com/doyled-it/france-roof-ball
"""


def clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the public column names and selection to an enriched frame."""
    # Drop the raw Open-Meteo columns so the resolved *_use columns can take
    # their clean public names without colliding.
    df = df.drop(columns=[c for c in ("temp_f", "rh_pct", "pressure_hpa") if c in df.columns])
    df = df.rename(columns=RENAME)
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
