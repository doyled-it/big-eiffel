"""Gather per-game weather and attach it to each batted ball.

Two sources:

- MLB StatsAPI schedule, per game: roof state, official temperature, condition,
  and wind as speed plus a field-relative direction ("8 mph, Out To CF"), plus
  the venue's coordinates, elevation, and orientation. This is the authoritative
  game report and the only source for roof state and field-relative wind.
- Open-Meteo historical reanalysis, per park and hour: temperature, relative
  humidity, surface pressure, wind speed and direction. The only source for
  humidity and pressure, which set air density precisely.

From these we compute, per ball, the true air density and the along-flight wind
component, and flag roof-closed games (still air, controlled temperature).
"""

from __future__ import annotations

import math
import time

import numpy as np
import pandas as pd
import requests

from . import config as C

MLB_SCHEDULE = "https://statsapi.mlb.com/api/v1/schedule"
OPEN_METEO = "https://archive-api.open-meteo.com/v1/archive"
UA = {"User-Agent": "Mozilla/5.0 (france-roof-ball research)"}

# Field-relative wind direction -> unit vector in field coordinates
# (x toward right field, y toward center field), as MLB reports it.
_S = math.sin(math.radians(45))
WIND_DIR_VEC = {
    "Out To CF": (0.0, 1.0),
    "In From CF": (0.0, -1.0),
    "Out To LF": (-_S, _S),
    "In From LF": (_S, -_S),
    "Out To RF": (_S, _S),
    "In From RF": (-_S, -_S),
    "L To R": (1.0, 0.0),
    "R To L": (-1.0, 0.0),
    "Calm": (0.0, 0.0),
    "Varies": (0.0, 0.0),
}


def parse_wind(s: str) -> tuple[float, str]:
    """Parse an MLB wind string into (speed_mph, field_direction).

    Arguments:
        s: Wind string such as "8 mph, Out To CF" or "0 mph, None".

    Returns:
        (speed in mph, direction label). Calm when no usable direction.
    """
    if not isinstance(s, str) or "mph" not in s:
        return (np.nan, "Unknown")
    try:
        spd = float(s.split("mph")[0].strip())
    except ValueError:
        spd = np.nan
    parts = s.split(",", 1)
    direction = parts[1].strip() if len(parts) > 1 else "Calm"
    if direction in ("None", "", "Calm") or spd == 0:
        direction = "Calm"
    return (spd, direction)


def wind_along_flight(speed_mph, direction, spray_deg) -> np.ndarray:
    """Tailwind component (mph, positive aids carry) along the ball's flight.

    Arguments:
        speed_mph: Wind speed.
        direction: Field-relative direction label.
        spray_deg: Ball spray angle (deg, negative toward left field).

    Returns:
        Along-flight wind component in mph (vectorized).
    """
    speed = np.asarray(speed_mph, dtype=float)
    spray = np.radians(np.asarray(spray_deg, dtype=float))
    fx, fy = np.sin(spray), np.cos(spray)
    vx = np.array([WIND_DIR_VEC.get(d, (0.0, 0.0))[0] for d in np.atleast_1d(direction)])
    vy = np.array([WIND_DIR_VEC.get(d, (0.0, 0.0))[1] for d in np.atleast_1d(direction)])
    return speed * (vx * fx + vy * fy)


def air_density_obs(temp_f, rh_pct, pressure_hpa) -> np.ndarray:
    """Air density (kg/m^3) from observed temperature, humidity, and pressure.

    Arguments:
        temp_f: Temperature in degrees Fahrenheit.
        rh_pct: Relative humidity in percent (0-100).
        pressure_hpa: Surface pressure in hectopascals.

    Returns:
        Air density in kg/m^3 (vectorized).
    """
    tk = (np.asarray(temp_f, float) - 32) * 5 / 9 + 273.15
    tc = tk - 273.15
    p = np.asarray(pressure_hpa, float) * 100.0
    psat = 610.94 * np.exp(17.625 * tc / (tc + 243.04))
    pv = np.clip(np.asarray(rh_pct, float) / 100.0, 0, 1) * psat
    pd_ = p - pv
    return pd_ / (287.05 * tk) + pv / (461.495 * tk)


def fetch_mlb_games(start_year: int = 2015, end_year: int = 2026, force: bool = False) -> pd.DataFrame:
    """Pull per-game weather, roof, and venue data from MLB StatsAPI.

    One schedule request per season. Cached to MLB_GAMES_PARQUET.

    Arguments:
        start_year: First season (inclusive).
        end_year: Last season (inclusive).
        force: Re-pull even if cached.

    Returns:
        One row per game with weather, roof, venue, and parsed wind.
    """
    if C.MLB_GAMES_PARQUET.exists() and not force:
        print(f"[weather] using cached {C.MLB_GAMES_PARQUET}")
        return pd.read_parquet(C.MLB_GAMES_PARQUET)

    rows = []
    for year in range(start_year, end_year + 1):
        params = {
            "sportId": 1,
            "startDate": f"{year}-03-01",
            "endDate": f"{year}-11-30",
            "gameType": "R",
            "hydrate": "weather,venue(location,fieldInfo)",
        }
        r = requests.get(MLB_SCHEDULE, params=params, timeout=120, headers=UA)
        r.raise_for_status()
        data = r.json()
        n = 0
        for dt in data.get("dates", []):
            for g in dt.get("games", []):
                v = g.get("venue", {})
                loc = v.get("location", {})
                coords = loc.get("defaultCoordinates", {}) or {}
                w = g.get("weather", {}) or {}
                spd, wdir = parse_wind(w.get("wind"))
                rows.append(
                    {
                        "game_pk": g.get("gamePk"),
                        "game_datetime_utc": g.get("gameDate"),
                        "venue_id": v.get("id"),
                        "venue_name": v.get("name"),
                        "lat": coords.get("latitude"),
                        "lon": coords.get("longitude"),
                        "elevation_m": loc.get("elevation"),
                        "azimuth_deg": loc.get("azimuthAngle"),
                        "roof_type": (v.get("fieldInfo", {}) or {}).get("roofType"),
                        "mlb_temp_f": pd.to_numeric(w.get("temp"), errors="coerce"),
                        "mlb_condition": w.get("condition"),
                        "wind_mph": spd,
                        "wind_dir": wdir,
                    }
                )
                n += 1
        print(f"[weather] MLB {year}: {n} games")
        time.sleep(0.5)
    df = pd.DataFrame(rows).dropna(subset=["game_pk"])
    df["game_pk"] = df["game_pk"].astype("int64")
    df.to_parquet(C.MLB_GAMES_PARQUET, index=False)
    print(f"[weather] cached {len(df):,} games to {C.MLB_GAMES_PARQUET}")
    return df


def fetch_openmeteo_for_venues(games: pd.DataFrame, force: bool = False) -> pd.DataFrame:
    """Hourly humidity and pressure per venue from Open-Meteo.

    One request per distinct venue over the full date range. Cached per venue
    under data/openmeteo_<venue_id>.parquet and returned concatenated.

    Arguments:
        games: The MLB games table (for venue coordinates).
        force: Re-pull even if cached.

    Returns:
        Long table: venue_id, time_utc (hourly), temp_f, rh_pct, pressure_hpa,
        wind_mph, wind_dir_deg.
    """
    venues = games.dropna(subset=["lat", "lon"]).groupby("venue_id").agg(lat=("lat", "first"), lon=("lon", "first"))
    frames = []
    for vid, row in venues.iterrows():
        cache = C.DATA / f"openmeteo_{int(vid)}.parquet"
        if cache.exists() and not force:
            frames.append(pd.read_parquet(cache))
            continue
        params = {
            "latitude": row.lat,
            "longitude": row.lon,
            "start_date": "2015-03-01",
            "end_date": "2026-09-30",  # archive reanalysis lags a few days; season ends late Sept
            # Only the variables that set air density; wind comes from MLB (field-relative).
            "hourly": "temperature_2m,relative_humidity_2m,surface_pressure",
            "temperature_unit": "fahrenheit",
            "timezone": "UTC",
        }
        # The archive weighs long hourly ranges heavily, so retry 429 with backoff.
        for attempt in range(6):
            r = requests.get(OPEN_METEO, params=params, timeout=180, headers=UA)
            if r.status_code == 429:
                wait = 65
                print(f"[weather] Open-Meteo 429 on venue {int(vid)}, waiting {wait}s (attempt {attempt + 1})")
                time.sleep(wait)
                continue
            r.raise_for_status()
            break
        else:
            raise RuntimeError(f"Open-Meteo kept returning 429 for venue {int(vid)}")
        h = r.json().get("hourly", {})
        part = pd.DataFrame(
            {
                "venue_id": int(vid),
                "time_utc": pd.to_datetime(h["time"], utc=True),
                "temp_f": h["temperature_2m"],
                "rh_pct": h["relative_humidity_2m"],
                "pressure_hpa": h["surface_pressure"],
            }
        )
        part.to_parquet(cache, index=False)
        frames.append(part)
        print(f"[weather] Open-Meteo venue {int(vid)}: {len(part):,} hours")
        time.sleep(6)
    return pd.concat(frames, ignore_index=True)


FT_TO_M = 0.3048
CLOSED_CONDITIONS = {"Roof Closed", "Dome", "Indoor", "Controlled"}


def _roof_closed(roof_type, condition) -> bool:
    """Whether a game was played with no wind and controlled air."""
    if roof_type == "Dome":
        return True
    if roof_type == "Retractable":
        return condition in CLOSED_CONDITIONS
    return False


def _std_pressure_hpa(elev_m) -> np.ndarray:
    """Standard-atmosphere surface pressure (hPa) at an elevation."""
    return 101325 * (1 - 2.25577e-5 * np.asarray(elev_m, float)) ** 5.2559 / 100.0


def build_enriched(force: bool = False) -> pd.DataFrame:
    """Join weather onto every batted ball and compute density and wind.

    Each ball is matched to its game (MLB report: roof, official temperature,
    field-relative wind, venue elevation) and to its park's Open-Meteo hour at
    first pitch (humidity, pressure). Roof-closed games are treated as still air
    at a controlled temperature. The result adds true air density and the
    along-flight wind component per ball.

    Arguments:
        force: Rebuild even if cached.

    Returns:
        The enriched batted-ball table.
    """
    from . import data as D

    if C.ENRICHED_PARQUET.exists() and not force:
        print(f"[weather] using cached {C.ENRICHED_PARQUET}")
        return pd.read_parquet(C.ENRICHED_PARQUET)

    balls = D.load_full()
    games = fetch_mlb_games(2015, 2026)
    om = fetch_openmeteo_for_venues(games)

    g = games.drop_duplicates("game_pk").set_index("game_pk")
    g_dt = pd.to_datetime(g["game_datetime_utc"], utc=True, errors="coerce")
    meta = pd.DataFrame(
        {
            "venue_id": g["venue_id"],
            "game_hour": g_dt.dt.floor("h"),
            "roof_type": g["roof_type"],
            "elevation_m": g["elevation_m"].astype(float) * FT_TO_M,
            "mlb_temp_f": g["mlb_temp_f"],
            "mlb_condition": g["mlb_condition"],
            "wind_mph": g["wind_mph"],
            "wind_dir": g["wind_dir"],
        }
    )
    df = balls.merge(meta, left_on="game_pk", right_index=True, how="left")

    # Attach Open-Meteo humidity/pressure/temp at the game hour.
    omk = om.rename(columns={"time_utc": "game_hour"})[["venue_id", "game_hour", "temp_f", "rh_pct", "pressure_hpa"]]
    df = df.merge(omk, on=["venue_id", "game_hour"], how="left", suffixes=("", "_om"))

    df["roof_closed"] = [_roof_closed(rt, cond) for rt, cond in zip(df["roof_type"], df["mlb_condition"])]

    # Resolve the air-density inputs. Open-air: MLB temp if present else
    # Open-Meteo, Open-Meteo humidity/pressure. Closed roof: controlled.
    temp = df["mlb_temp_f"].where(df["mlb_temp_f"].notna(), df["temp_f"])
    rh = df["rh_pct"]
    press = df["pressure_hpa"]
    std_p = np.asarray(_std_pressure_hpa(df["elevation_m"].fillna(0)), dtype=float)
    press = press.where(press.notna(), pd.Series(std_p, index=df.index))
    rh = rh.fillna(50.0)
    temp = temp.fillna(70.0)

    closed = df["roof_closed"].fillna(False).to_numpy()
    temp = np.where(closed, 72.0, temp.to_numpy(float))
    rh = np.where(closed, 50.0, rh.to_numpy(float))
    press = np.where(closed, std_p, press.to_numpy(float))

    df["temp_f_use"] = temp
    df["rh_pct_use"] = rh
    df["pressure_hpa_use"] = press
    df["air_density"] = air_density_obs(temp, rh, press)

    along = wind_along_flight(
        df["wind_mph"].fillna(0).to_numpy(), df["wind_dir"].fillna("Calm").to_numpy(), df["spray_deg"].to_numpy()
    )
    df["wind_along_mph"] = np.where(closed, 0.0, along)

    df.to_parquet(C.ENRICHED_PARQUET, index=False)
    matched = df["air_density"].notna().mean()
    print(f"[weather] enriched {len(df):,} balls ({matched:.1%} with density); cached {C.ENRICHED_PARQUET}")
    return df
