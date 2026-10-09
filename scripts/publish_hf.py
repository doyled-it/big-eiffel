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
    "fielder_2": "catcher",
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
    "catcher",
    "stand",
    "p_throws",
    "batter_platoon_adv",
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
    "day_night",
    "hp_umpire",
    # batted ball and outcome
    "launch_speed",
    "launch_angle",
    "spray_deg",
    "hit_distance_sc",
    "launch_speed_angle",
    "is_barrel",
    "hard_hit",
    "sweet_spot",
    "batted_direction",
    "in_strike_zone",
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
    "intercept_ball_minus_batter_pos_x_inches",
    "intercept_ball_minus_batter_pos_y_inches",
    # the pitch that was hit
    "pitch_type",
    "pitch_name",
    "release_speed",
    "release_spin_rate",
    "effective_speed",
    "release_extension",
    "release_pos_x",
    "release_pos_y",
    "release_pos_z",
    "arm_angle",
    "vx0",
    "vy0",
    "vz0",
    "ax",
    "ay",
    "az",
    "plate_x",
    "plate_z",
    "sz_top",
    "sz_bot",
    "pfx_x",
    "pfx_z",
    "api_break_z_with_gravity",
    "api_break_x_arm",
    "api_break_x_batter_in",
    "zone",
    "spin_axis",
    # player context
    "age_bat",
    "age_pit",
    "n_thruorder_pitcher",
    "pitcher_days_since_prev_game",
    "batter_days_since_prev_game",
    # running, fielding, and park context (Savant season leaderboards)
    "batter_sprint_speed",
    "runner_1b_sprint_speed",
    "runner_2b_sprint_speed",
    "runner_3b_sprint_speed",
    "catcher_pop_time",
    "catcher_arm_strength",
    "park_factor_runs",
    "park_factor_hr",
    "park_factor_woba",
    # conditions (weather)
    "roof_type",
    "roof_closed",
    "elevation_m",
    "temperature_f",
    "humidity_pct",
    "dew_point_f",
    "pressure_hpa",
    "air_density",
    "wind_mph",
    "wind_dir",
    "wind_along_flight_mph",
    "wind_gust_mph",
    "precipitation_mm",
    "cloud_cover_pct",
    "humidor",
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
  inning, count, outs, baserunners, score, the play description, whether it was
  a day or night game, and the home-plate umpire.
- **Batted ball**: exit velocity, launch angle, derived spray angle, projected
  hit distance, the Statcast barrel code and an `is_barrel` flag, and bat
  tracking where available (2024+): bat speed, attack angle, swing path tilt,
  swing length, swing-intercept offsets, plus derived `launch_minus_attack_deg`
  (positive = undercut/backspin, negative = overcut/topspin), `hard_hit`
  (>= 95 mph), `sweet_spot` (8 to 32 deg), and `batted_direction` (pull/center/oppo).
- **Expected outcomes** (Statcast models): `xba`, `xwoba`, `xslg`, plus
  `woba_value`, `babip_value`, `iso_value`, and the run- and win-expectancy
  deltas. A derived `carry_vs_expected_ft` gives how far the ball carried versus
  a physics model evaluated at its own air density.
- **Running, fielding, and park** (season-level, from Savant leaderboards): the
  batter's and each baserunner's `sprint_speed` (ft/s), the catcher's
  `pop_time` to second (s) and `arm_strength` (mph), and the park's run,
  home-run, and wOBA factors (`park_factor_*`, 100 is neutral, Savant's 3-year
  rolling window through that season). These are the player's or park's number
  for the season, not a per-pitch measurement.
- **The pitch that was hit**: type, release speed and spin, the release point and
  `arm_angle`, the full trajectory (`vx0..az`), movement (`pfx`, `api_break_*`),
  plate location, the batter's strike zone (`sz_top`/`sz_bot`), zone, and spin
  axis. `in_strike_zone` flags whether that pitch was a strike by location (so a
  False is a chase). This is a batted-ball dataset, so there are no taken pitches
  and no umpire ball/strike calls to compare against.
- **Conditions**: roof state and a roof-closed flag; true **air density** from
  temperature, relative humidity, and surface pressure (official game-report
  temperature where present, humidity and pressure from Open-Meteo's hourly
  reanalysis at the park), plus a derived `dew_point_f` and a `humidor` park flag;
  and **wind** as MLB reports it plus `wind_along_flight_mph`, the component along
  each ball's own flight direction. Roof-closed games use controlled still air at
  72 F with no wind. Ambient `wind_gust_mph`, `precipitation_mm`, and
  `cloud_cover_pct` come from the Open-Meteo hour and describe the outdoor weather
  regardless of roof state (so a closed-roof game still reports the sky outside).

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
| `hard_hit`, `sweet_spot`, `batted_direction` | >= 95 mph; 8-32 deg; pull/center/oppo |
| `attack_angle`, `bat_speed` | bat tracking (2024+) |
| `pitch_type`, `release_speed`, `release_spin_rate` | the pitch that was hit |
| `arm_angle`, `release_pos_x/y/z` | arm slot and release point |
| `in_strike_zone` | True if the hit pitch was a strike by location (False = chase) |
| `batter_platoon_adv` | True when batter and pitcher throw opposite hands |
| `day_night`, `hp_umpire` | day or night game; home-plate umpire |
| `batter_sprint_speed`, `runner_*_sprint_speed` | sprint speed (ft/s), season level |
| `catcher_pop_time`, `catcher_arm_strength` | catcher pop time to 2B (s) and arm (mph), season level |
| `park_factor_runs`, `park_factor_hr`, `park_factor_woba` | park factors (100 = neutral, 3-year rolling) |
| `air_density`, `dew_point_f`, `humidor` | air density (kg/m^3), dew point (F), humidor park |
| `wind_along_flight_mph` | along-flight wind; positive aids carry |
| `wind_gust_mph`, `precipitation_mm`, `cloud_cover_pct` | ambient outdoor gusts, rain, cloud (Open-Meteo) |
| `roof_closed` | True when played in still, controlled air |

## Sources and caveats

- Statcast via Baseball Savant (2015-2025 through the
  `Jensen-holm/statcast-era-pitches` mirror, 2026 pulled directly from Savant).
- Game weather, roof, day/night, and umpire from the MLB StatsAPI schedule.
- Humidity and pressure from the Open-Meteo historical reanalysis archive.
- Sprint speed, pop time, and park factors from Baseball Savant leaderboards
  (season level; park factors are the 3-year rolling index through that season).

Caveats: the reported wind is a station or grid reading, not the wind inside the
stadium bowl, so `wind_along_flight_mph` overstates the wind the ball truly
feels. Roof state for retractable parks is best-effort from the game report.
`hit_distance_sc` is a Statcast projection, not a surveyed landing point. This is
derived data from public MLB Statcast; use it for research and credit the
sources above.

Built with https://github.com/doyled-it/big-eiffel
"""


def _dew_point_f(temp_f, rh_pct):
    """Dew point (F) from temperature (F) and relative humidity (%), Magnus formula."""
    tc = (np.asarray(temp_f, float) - 32.0) * 5.0 / 9.0
    rh = np.clip(np.asarray(rh_pct, float), 1.0, 100.0) / 100.0
    a, b = 17.625, 243.04
    gamma = np.log(rh) + a * tc / (b + tc)
    td_c = b * gamma / (a - gamma)
    return np.round(td_c * 9.0 / 5.0 + 32.0, 1)


def _humidor(team, year) -> bool:
    """Whether the home park used a ball humidor (documented cases only)."""
    try:
        y = int(year)
    except (TypeError, ValueError):
        return False
    if y >= 2022:  # MLB mandated humidors in all 30 parks in 2022
        return True
    if team == "COL":  # Coors Field since 2002
        return True
    if team == "ARI" and y >= 2018:  # Chase Field since 2018
        return True
    return False


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

    # Hitting flags.
    if "launch_speed" in df.columns:
        df["hard_hit"] = df["launch_speed"] >= 95.0
    if "launch_angle" in df.columns:
        df["sweet_spot"] = df["launch_angle"].between(8, 32)
    if {"spray_deg", "stand"} <= set(df.columns):
        # Positive means pulled, accounting for handedness (RH pulls to LF, negative spray).
        pull = np.where(df["stand"].to_numpy() == "R", -1.0, 1.0) * df["spray_deg"].to_numpy(float)
        df["batted_direction"] = np.select([pull > 15, pull < -15], ["pull", "oppo"], default="center")
        df.loc[df["spray_deg"].isna() | df["stand"].isna(), "batted_direction"] = None
    if {"stand", "p_throws"} <= set(df.columns):
        pa = pd.Series(df["stand"].to_numpy() != df["p_throws"].to_numpy(), index=df.index, dtype="boolean")
        df["batter_platoon_adv"] = pa.mask(df["stand"].isna() | df["p_throws"].isna())  # opposite hands = advantage
    if {"plate_x", "plate_z", "sz_top", "sz_bot"} <= set(df.columns):
        # Was the pitch that was hit in the rulebook zone (a strike by location)?
        px, pz = df["plate_x"].to_numpy(float), df["plate_z"].to_numpy(float)
        zt, zb = df["sz_top"].to_numpy(float), df["sz_bot"].to_numpy(float)
        iz = pd.Series((np.abs(px) <= 0.83) & (pz >= zb) & (pz <= zt), index=df.index, dtype="boolean")
        miss = df["plate_x"].isna() | df["plate_z"].isna() | df["sz_top"].isna() | df["sz_bot"].isna()
        df["in_strike_zone"] = iz.mask(miss)

    # Conditions.
    if {"temperature_f", "humidity_pct"} <= set(df.columns):
        df["dew_point_f"] = _dew_point_f(df["temperature_f"].to_numpy(float), df["humidity_pct"].to_numpy(float))
    if {"home_team", "game_year"} <= set(df.columns):
        df["humidor"] = [_humidor(t, y) for t, y in zip(df["home_team"], df["game_year"])]

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
