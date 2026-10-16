"""Season-level running, fielding, and park context from Baseball Savant.

Three Savant leaderboards, each one request per season and cached locally:

- Sprint speed, per player: the batter's and each baserunner's top-end speed.
- Pop time, per catcher: time to second base and arm strength.
- Park factors, per venue: the run, home-run, and wOBA index of the park
  (100 is neutral), on Savant's 3-year rolling window through that season.

:func:`attach_context` joins all of these onto a batted-ball frame by
``(game_year, player_id)`` and ``(game_year, venue_id)``. It runs before
:func:`scripts.publish_hf.clean_frame`, so the columns it adds carry through
the public dataset. The join keys (``batter``, ``on_1b``/``on_2b``/``on_3b``,
``fielder_2``, ``venue_id``, ``game_year``) are the raw Statcast names present
on the enriched frame, not the renamed public ones.
"""

from __future__ import annotations

import io
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from . import config as C

UA = {"User-Agent": "Mozilla/5.0 (france-roof-ball research)"}

SPRINT_URL = "https://baseballsavant.mlb.com/leaderboard/sprint_speed"
POPTIME_URL = "https://baseballsavant.mlb.com/leaderboard/poptime"
PARK_URL = "https://baseballsavant.mlb.com/leaderboard/statcast-park-factors"

# Statcast tracking (and these leaderboards) begin in 2015.
FIRST_CONTEXT_YEAR = 2015


def _cache(name: str) -> Path:
    return C.DATA / f"context_{name}.parquet"


def _get(url: str, params: dict) -> requests.Response:
    """GET with a short backoff on transient failures."""
    last: Exception | None = None
    for attempt in range(4):
        try:
            r = requests.get(url, params=params, timeout=120, headers=UA)
            r.raise_for_status()
            return r
        except requests.RequestException as e:  # noqa: PERF203
            last = e
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"request failed after retries: {url} ({last})")


def sprint_speed(year: int, force: bool = False) -> pd.DataFrame:
    """Per-player sprint speed for one season.

    Arguments:
        year: Season.
        force: Re-pull even if cached.

    Returns:
        Columns ``year``, ``player_id``, ``sprint_speed`` (ft/s), ``hp_to_1b``.
    """
    cache = _cache(f"sprint_{year}")
    if cache.exists() and not force:
        return pd.read_parquet(cache)
    r = _get(SPRINT_URL, {"year": year, "position": "", "team": "", "min": "0", "csv": "true"})
    df = pd.read_csv(io.StringIO(r.text))
    out = pd.DataFrame(
        {
            "year": year,
            "player_id": pd.to_numeric(df["player_id"], errors="coerce").astype("Int64"),
            "sprint_speed": pd.to_numeric(df.get("sprint_speed"), errors="coerce"),
            "hp_to_1b": pd.to_numeric(df.get("hp_to_1b"), errors="coerce"),
        }
    ).dropna(subset=["player_id"])
    out.to_parquet(cache, index=False)
    print(f"[context] sprint_speed {year}: {len(out)} players")
    return out


def pop_time(year: int, force: bool = False) -> pd.DataFrame:
    """Per-catcher pop time for one season.

    Arguments:
        year: Season.
        force: Re-pull even if cached.

    Returns:
        Columns ``year``, ``catcher_id``, ``pop_2b`` (s), ``arm_strength`` (mph).
    """
    cache = _cache(f"poptime_{year}")
    if cache.exists() and not force:
        return pd.read_parquet(cache)
    r = _get(POPTIME_URL, {"year": year, "team": "", "min2b": "1", "min3b": "0", "csv": "true"})
    df = pd.read_csv(io.StringIO(r.text))
    out = pd.DataFrame(
        {
            "year": year,
            "catcher_id": pd.to_numeric(df["entity_id"], errors="coerce").astype("Int64"),
            "pop_2b": pd.to_numeric(df.get("pop_2b_sba"), errors="coerce"),
            "arm_strength": pd.to_numeric(df.get("maxeff_arm_2b_3b_sba"), errors="coerce"),
        }
    ).dropna(subset=["catcher_id"])
    out.to_parquet(cache, index=False)
    print(f"[context] pop_time {year}: {len(out)} catchers")
    return out


def park_factors(year: int, force: bool = False) -> pd.DataFrame:
    """Per-venue park factors for one season (3-year rolling, Savant default).

    The leaderboard has no CSV export; the table ships as a ``var data = [...]``
    JSON blob in the page, which is parsed here.

    Arguments:
        year: Season.
        force: Re-pull even if cached.

    Returns:
        Columns ``year``, ``venue_id``, ``park_factor_runs``, ``park_factor_hr``,
        ``park_factor_woba`` (each an index where 100 is neutral).
    """
    cache = _cache(f"park_{year}")
    if cache.exists() and not force:
        return pd.read_parquet(cache)
    r = _get(
        PARK_URL,
        {
            "type": "year",
            "year": year,
            "batSide": "",
            "stat": "index_wOBA",
            "condition": "All",
            "rolling": "",
            "parks": "mlb",
        },
    )
    m = re.search(r"var\s+data\s*=\s*(\[.*?\]);", r.text, re.DOTALL)
    if not m:
        print(f"[context] park_factors {year}: no data blob, skipping")
        out = pd.DataFrame(columns=["year", "venue_id", "park_factor_runs", "park_factor_hr", "park_factor_woba"])
        out.to_parquet(cache, index=False)
        return out
    rows = json.loads(m.group(1))
    df = pd.DataFrame(rows)
    out = pd.DataFrame(
        {
            "year": year,
            "venue_id": pd.to_numeric(df["venue_id"], errors="coerce").astype("Int64"),
            "park_factor_runs": pd.to_numeric(df["index_runs"], errors="coerce"),
            "park_factor_hr": pd.to_numeric(df["index_hr"], errors="coerce"),
            "park_factor_woba": pd.to_numeric(df["index_woba"], errors="coerce"),
        }
    ).dropna(subset=["venue_id"])
    out.to_parquet(cache, index=False)
    print(
        f"[context] park_factors {year}: {len(out)} parks ({df['year_range'].iloc[0] if 'year_range' in df else year})"
    )
    return out


def _load_all(fetch, years: list[int], refresh_years: set[int]) -> pd.DataFrame:
    """Concatenate a per-season fetch across years, refreshing the given years."""
    frames = [fetch(y, force=(y in refresh_years)) for y in years]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _map_by_year_id(df: pd.DataFrame, id_col: str, lookup: pd.DataFrame, id_key: str, value_col: str) -> np.ndarray:
    """Look a value up by ``(game_year, id_col)`` against a per-season table.

    Arguments:
        df: The batted-ball frame (needs ``game_year`` and ``id_col``).
        id_col: The id column on ``df`` to join (batter, runner, catcher).
        lookup: A per-season table with ``year``, ``id_key``, and ``value_col``.
        id_key: The id column name in ``lookup``.
        value_col: The value column to return.

    Returns:
        The looked-up values aligned to ``df`` (NaN where there is no match or
        the id is missing, as for an empty base).
    """
    series = lookup.dropna(subset=[id_key]).set_index(["year", id_key])[value_col]
    series = series[~series.index.duplicated(keep="first")]
    years = pd.to_numeric(df["game_year"], errors="coerce").astype("Int64")
    ids = pd.to_numeric(df[id_col], errors="coerce").astype("Int64")
    idx = pd.MultiIndex.from_arrays([years, ids])
    return series.reindex(idx).to_numpy()


def attach_context(df: pd.DataFrame, refresh_current_year: bool = False) -> pd.DataFrame:
    """Join sprint speed, pop time, and park factors onto a batted-ball frame.

    Adds, per row: the batter's and each baserunner's sprint speed, the
    catcher's pop time and arm strength, and the park's run/HR/wOBA factors.
    Missing ids (an empty base, a player below the leaderboard minimum) yield
    NaN, which is correct.

    Arguments:
        df: An enriched batted-ball frame with raw Statcast keys (``game_year``,
            ``batter``, ``on_1b``/``on_2b``/``on_3b``, ``fielder_2``,
            ``venue_id``).
        refresh_current_year: Re-pull the latest season's leaderboards instead
            of using the cache (the in-progress season changes daily).

    Returns:
        The frame with the context columns added.
    """
    years = sorted({int(y) for y in pd.to_numeric(df["game_year"], errors="coerce").dropna().unique()})
    years = [y for y in years if y >= FIRST_CONTEXT_YEAR]
    if not years:
        return df
    refresh = {max(years)} if refresh_current_year else set()

    sprint = _load_all(sprint_speed, years, refresh)
    pops = _load_all(pop_time, years, refresh)
    parks = _load_all(park_factors, years, refresh)

    df = df.copy()
    if not sprint.empty:
        df["batter_sprint_speed"] = _map_by_year_id(df, "batter", sprint, "player_id", "sprint_speed")
        for base in ("1b", "2b", "3b"):
            col = f"on_{base}"
            if col in df.columns:
                df[f"runner_{base}_sprint_speed"] = _map_by_year_id(df, col, sprint, "player_id", "sprint_speed")
    if not pops.empty and "fielder_2" in df.columns:
        df["catcher_pop_time"] = _map_by_year_id(df, "fielder_2", pops, "catcher_id", "pop_2b")
        df["catcher_arm_strength"] = _map_by_year_id(df, "fielder_2", pops, "catcher_id", "arm_strength")
    if not parks.empty and "venue_id" in df.columns:
        for value_col in ("park_factor_runs", "park_factor_hr", "park_factor_woba"):
            df[value_col] = _map_by_year_id(df, "venue_id", parks, "venue_id", value_col)
    return df
