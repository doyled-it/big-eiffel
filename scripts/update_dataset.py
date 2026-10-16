"""Incrementally update the HuggingFace dataset with recent games.

Designed to run daily (see .github/workflows/update-dataset.yml). For the last
few days it pulls the new batted balls, joins the game weather and roof, matches
the Open-Meteo hour where available, and uploads one small partition file per
date to the dataset under data/updates/. Re-running a date overwrites its
partition, so it is idempotent, and the rolling window backfills the precise
humidity and pressure once the Open-Meteo reanalysis (which lags a few days)
catches up.

Usage:
    uv run python scripts/update_dataset.py --days 8 --repo doyled-it/statcast-batted-balls-weather
"""

from __future__ import annotations

import argparse
import datetime as dt
import time

import numpy as np
import pandas as pd
import requests

from frb import config as C
from frb import context as ctx
from frb import data as D
from frb import weather as W
from scripts.publish_hf import clean_frame


def _mlb_games_for_date(date_str: str) -> pd.DataFrame:
    params = {
        "sportId": 1,
        "date": date_str,
        "gameType": "R,F,D,L,W",
        "hydrate": "weather,venue(location,fieldInfo),officials",
    }
    r = requests.get(W.MLB_SCHEDULE, params=params, timeout=120, headers=W.UA)
    r.raise_for_status()
    rows = []
    for d in r.json().get("dates", []):
        for g in d.get("games", []):
            v = g.get("venue", {})
            loc = v.get("location", {}) or {}
            coords = loc.get("defaultCoordinates", {}) or {}
            w = g.get("weather", {}) or {}
            spd, wdir = W.parse_wind(w.get("wind"))
            rows.append(
                {
                    "game_pk": g.get("gamePk"),
                    "game_hour": pd.to_datetime(g.get("gameDate"), utc=True, errors="coerce"),
                    "venue_id": v.get("id"),
                    "lat": coords.get("latitude"),
                    "lon": coords.get("longitude"),
                    "elevation_m": (loc.get("elevation") or 0) * W.FT_TO_M,
                    "roof_type": (v.get("fieldInfo", {}) or {}).get("roofType"),
                    "mlb_temp_f": pd.to_numeric(w.get("temp"), errors="coerce"),
                    "mlb_condition": w.get("condition"),
                    "wind_mph": spd,
                    "wind_dir": wdir,
                    "day_night": g.get("dayNight"),
                    "hp_umpire": W.home_plate_umpire(g.get("officials")),
                }
            )
    g = pd.DataFrame(rows).dropna(subset=["game_pk"])
    if len(g):
        g["game_pk"] = g["game_pk"].astype("int64")
        g["game_hour"] = g["game_hour"].dt.floor("h")
    return g


def _openmeteo_for_date(games: pd.DataFrame, date_str: str) -> pd.DataFrame:
    frames = []
    for vid, row in (
        games.dropna(subset=["lat", "lon"])
        .groupby("venue_id")
        .agg(lat=("lat", "first"), lon=("lon", "first"))
        .iterrows()
    ):
        params = {
            "latitude": row.lat,
            "longitude": row.lon,
            "start_date": date_str,
            "end_date": date_str,
            "hourly": W.OPENMETEO_HOURLY,
            "temperature_unit": "fahrenheit",
            "windspeed_unit": "mph",
            "timezone": "UTC",
        }
        try:
            r = requests.get(W.OPEN_METEO, params=params, timeout=120, headers=W.UA)
            if r.status_code != 200:
                continue
            h = r.json().get("hourly", {})
            frames.append(
                pd.DataFrame(
                    {
                        "venue_id": int(vid),
                        "game_hour": pd.to_datetime(h["time"], utc=True),
                        "temp_f": h["temperature_2m"],
                        "rh_pct": h["relative_humidity_2m"],
                        "pressure_hpa": h["surface_pressure"],
                        "wind_gust_mph": h["wind_gusts_10m"],
                        "precipitation_mm": h["precipitation"],
                        "cloud_cover_pct": h["cloud_cover"],
                    }
                )
            )
        except requests.RequestException:
            continue
    cols = [
        "venue_id",
        "game_hour",
        "temp_f",
        "rh_pct",
        "pressure_hpa",
        "wind_gust_mph",
        "precipitation_mm",
        "cloud_cover_pct",
    ]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=cols)


def enrich_date(date_str: str) -> pd.DataFrame | None:
    """Pull and enrich one day of batted balls, or None if there were no games."""
    balls = D.pull_savant_batted(date_str, date_str, window_days=1)
    if balls.empty:
        return None
    games = _mlb_games_for_date(date_str)
    if games.empty:
        return None
    om = _openmeteo_for_date(games, date_str)
    df = balls.merge(games, left_on="game_pk", right_on="game_pk", how="left")
    df = df.merge(om, on=["venue_id", "game_hour"], how="left", suffixes=("", "_om"))

    closed = np.array([W._roof_closed(rt, c) for rt, c in zip(df["roof_type"], df["mlb_condition"])])
    std_p = np.asarray(W._std_pressure_hpa(df["elevation_m"].fillna(0)), float)
    temp = df["mlb_temp_f"].where(df["mlb_temp_f"].notna(), df["temp_f"]).fillna(70.0).to_numpy(float)
    rh = df["rh_pct"].fillna(50.0).to_numpy(float)
    press = df["pressure_hpa"].to_numpy(float)
    press = np.where(np.isfinite(press), press, std_p)
    temp = np.where(closed, 72.0, temp)
    rh = np.where(closed, 50.0, rh)
    press = np.where(closed, std_p, press)
    df["roof_closed"] = closed
    df["temp_f_use"] = temp
    df["rh_pct_use"] = rh
    df["pressure_hpa_use"] = press
    df["air_density"] = W.air_density_obs(temp, rh, press)
    along = W.wind_along_flight(
        df["wind_mph"].fillna(0).to_numpy(), df["wind_dir"].fillna("Calm").to_numpy(), df["spray_deg"].to_numpy()
    )
    df["wind_along_mph"] = np.where(closed, 0.0, along)
    # Running, fielding, and park context; refresh the in-season leaderboard.
    df = ctx.attach_context(df, refresh_current_year=True)
    return clean_frame(df)


def main() -> None:
    from huggingface_hub import HfApi, get_token

    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=8, help="rolling window of recent days to (re)build")
    ap.add_argument("--repo", default="doyled-it/statcast-batted-balls-weather")
    args = ap.parse_args()

    if not get_token():
        raise SystemExit("No HuggingFace token. Set HF_TOKEN or run `hf auth login`.")
    api = HfApi()
    updir = C.DATA / "updates"
    updir.mkdir(exist_ok=True)
    today = dt.date.today()
    base_last = dt.date.fromisoformat(C.DATASET_BASE_LAST_DATE)
    for k in range(1, args.days + 1):
        day = today - dt.timedelta(days=k)
        if day <= base_last:
            continue  # covered by the published base file; do not duplicate
        date_str = day.isoformat()
        try:
            part = enrich_date(date_str)
        except Exception as e:  # noqa: BLE001
            print(f"[update] {date_str}: error {e}")
            continue
        if part is None or part.empty:
            print(f"[update] {date_str}: no games")
            continue
        local = updir / f"{date_str}.parquet"
        part.to_parquet(local, index=False)
        for attempt in range(4):
            try:
                api.upload_file(
                    path_or_fileobj=str(local),
                    path_in_repo=f"data/updates/{date_str}.parquet",
                    repo_id=args.repo,
                    repo_type="dataset",
                )
                print(f"[update] {date_str}: {len(part):,} balls uploaded")
                break
            except Exception as e:  # noqa: BLE001
                print(f"[update] {date_str}: upload attempt {attempt + 1} failed ({e}); retrying")
                time.sleep(10 * (attempt + 1))
        else:
            print(f"[update] {date_str}: upload failed after retries")


if __name__ == "__main__":
    main()
