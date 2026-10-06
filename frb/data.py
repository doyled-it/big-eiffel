"""Pull, cache, and load the Statcast batted-ball data.

Primary source is the HuggingFace parquet mirror, queried with DuckDB so the
filter and column projection push down and only batted balls transfer. A
pybaseball season-by-season pull is available as a fallback. The cached
product is every regular-season batted ball from 2015 on that has a measured
exit velocity, launch angle, and Statcast hit distance, with a derived spray
angle.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from . import config as C


def spray_angle(hc_x: pd.Series | np.ndarray, hc_y: pd.Series | np.ndarray) -> np.ndarray:
    """Spray angle in degrees from Statcast hit coordinates.

    Negative is toward left field, positive toward right field, from the
    catcher's perspective.

    Arguments:
        hc_x: Statcast horizontal hit coordinate.
        hc_y: Statcast vertical hit coordinate.

    Returns:
        Spray angle in degrees.
    """
    return np.degrees(np.arctan2(np.asarray(hc_x) - C.HC_X0, C.HC_Y0 - np.asarray(hc_y)))


def download_source_parquet() -> str:
    """Download the single source parquet from HuggingFace (resumable, cached).

    Returns:
        Local filesystem path to the downloaded parquet.
    """
    from huggingface_hub import hf_hub_download

    print(f"[data] downloading {C.HF_REPO}:{C.HF_FILE} (~826 MB, resumable) ...")
    path = hf_hub_download(repo_id=C.HF_REPO, filename=C.HF_FILE, repo_type="dataset")
    print(f"[data] source parquet at {path}")
    return path


def _filter_local(source_path: str) -> pd.DataFrame:
    """Filter a local Statcast parquet down to usable batted balls."""
    import duckdb

    con = duckdb.connect()
    con.execute("PRAGMA threads=4;")
    cols = ", ".join(C.PULL_COLUMNS)
    query = f"""
        SELECT {cols},
               degrees(atan2(hc_x - {C.HC_X0}, {C.HC_Y0} - hc_y)) AS spray_deg
        FROM read_parquet('{source_path}')
        WHERE game_type IN ('R', 'F', 'D', 'L', 'W')
          AND launch_speed IS NOT NULL
          AND launch_angle IS NOT NULL
          AND hit_distance_sc IS NOT NULL
    """
    df = con.execute(query).df()
    con.close()
    return df


def pull_raw(force: bool = False) -> pd.DataFrame:
    """Pull and cache the filtered batted-ball table from HuggingFace.

    Downloads the single source parquet once (resumable), then filters it
    locally with DuckDB. Downloading up front avoids the many HTTP range
    requests that trigger rate limiting when filtering directly over httpfs.

    Arguments:
        force: Re-pull even if the cached parquet already exists.

    Returns:
        The filtered batted-ball DataFrame.
    """
    if C.RAW_PARQUET.exists() and not force:
        print(f"[data] using cached {C.RAW_PARQUET}")
        return pd.read_parquet(C.RAW_PARQUET)

    source_path = download_source_parquet()
    df = _filter_local(source_path)
    print(f"[data] filtered to {len(df):,} batted balls, {df.game_year.min()}-{df.game_year.max()}")
    C.RAW_PARQUET.parent.mkdir(exist_ok=True)
    df.to_parquet(C.RAW_PARQUET, index=False)
    print(f"[data] cached to {C.RAW_PARQUET}")
    _write_sample(df)
    return df


def pull_raw_pybaseball(force: bool = False) -> pd.DataFrame:
    """Fallback: pull season by season with pybaseball and cache.

    Used only if the HuggingFace path is unavailable. Produces the same
    schema as :func:`pull_raw`.

    Arguments:
        force: Re-pull even if the cached parquet already exists.

    Returns:
        The filtered batted-ball DataFrame.
    """
    if C.RAW_PARQUET.exists() and not force:
        print(f"[data] using cached {C.RAW_PARQUET}")
        return pd.read_parquet(C.RAW_PARQUET)

    from pybaseball import statcast

    frames = []
    for year in range(2015, 2026):
        print(f"[data] pybaseball statcast {year} ...")
        part = statcast(start_dt=f"{year}-03-01", end_dt=f"{year}-11-30")
        part = part[part["game_type"].isin(C.GAME_TYPES)]
        part = part.dropna(subset=["launch_speed", "launch_angle", "hit_distance_sc"])
        keep = [c for c in C.PULL_COLUMNS if c in part.columns]
        part = part[keep].copy()
        part["spray_deg"] = spray_angle(part["hc_x"], part["hc_y"])
        frames.append(part)
    df = pd.concat(frames, ignore_index=True)
    df.to_parquet(C.RAW_PARQUET, index=False)
    print(f"[data] cached to {C.RAW_PARQUET}")
    _write_sample(df)
    return df


def pull_savant_batted(start_dt: str, end_dt: str, window_days: int = 18) -> pd.DataFrame:
    """Pull batted balls directly from the Baseball Savant CSV endpoint.

    Faster than day-by-day pybaseball: the endpoint filters to batted-ball types
    server side, so each multi-day window transfers only the balls in play. The
    endpoint truncates a single query at 25000 rows, so windows are kept small.

    Arguments:
        start_dt: Start date (YYYY-MM-DD).
        end_dt: End date (YYYY-MM-DD).
        window_days: Days per request window.

    Returns:
        A filtered batted-ball DataFrame with the same schema as the HF pull.
    """
    import io

    import requests

    frames = []
    cur = pd.Timestamp(start_dt)
    end = pd.Timestamp(end_dt)
    while cur <= end:
        w_end = min(cur + pd.Timedelta(days=window_days - 1), end)
        params = {
            "all": "true",
            "hfGT": "R|PO|",
            "hfBBT": "fly_ball|ground_ball|line_drive|popup|",
            "player_type": "batter",
            "game_date_gt": cur.strftime("%Y-%m-%d"),
            "game_date_lt": w_end.strftime("%Y-%m-%d"),
            "min_pitches": "0",
            "min_results": "0",
            "type": "details",
            "sort_col": "pitches",
            "player_event_sort": "api_p_release_speed",
            "sort_order": "desc",
            "min_pas": "0",
        }
        r = requests.get(C.SAVANT_CSV, params=params, timeout=180, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        part = pd.read_csv(io.StringIO(r.text))
        if len(part) >= 25000:
            print(f"[data] WARNING: window {cur.date()}..{w_end.date()} hit the 25000 row cap; shrink window_days")
        print(f"[data] savant {cur.date()}..{w_end.date()}: {len(part):,} batted balls")
        if len(part):
            frames.append(part)
        cur = w_end + pd.Timedelta(days=1)
    if not frames:  # off-day or empty range
        return pd.DataFrame(columns=C.PULL_COLUMNS + ["spray_deg"])
    df = pd.concat(frames, ignore_index=True)
    if "game_type" in df.columns:
        df = df[df["game_type"].isin(C.GAME_TYPES)]
    # Coerce numeric columns (empty windows can promote a column to object dtype).
    for col in ["launch_speed", "launch_angle", "hit_distance_sc", "hc_x", "hc_y"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["launch_speed", "launch_angle", "hit_distance_sc"])
    keep = [c for c in C.PULL_COLUMNS if c in df.columns]
    out = df[keep].copy()
    for c in C.PULL_COLUMNS:
        if c not in out.columns:
            out[c] = np.nan
    out["spray_deg"] = spray_angle(out["hc_x"], out["hc_y"])
    return out[C.PULL_COLUMNS + ["spray_deg"]]


def pull_2026(force: bool = False) -> pd.DataFrame:
    """Pull and cache 2026 batted balls through today (not in the HF mirror yet).

    The end date tracks the current day so a forced re-pull always brings the
    pool up to the latest completed games, including the postseason.
    """
    import datetime as dt

    if C.BALLS_2026_PARQUET.exists() and not force:
        print(f"[data] using cached {C.BALLS_2026_PARQUET}")
        return pd.read_parquet(C.BALLS_2026_PARQUET)
    today = dt.date.today().isoformat()
    df = pull_savant_batted("2026-03-15", today)
    df.to_parquet(C.BALLS_2026_PARQUET, index=False)
    print(f"[data] cached {len(df):,} 2026 batted balls to {C.BALLS_2026_PARQUET}")
    return df


def load_full() -> pd.DataFrame:
    """Load the full 2015-2026 batted-ball pool (HF 2015-2025 plus Savant 2026)."""
    frames = [pd.read_parquet(C.RAW_PARQUET)]
    if C.BALLS_2026_PARQUET.exists():
        frames.append(pd.read_parquet(C.BALLS_2026_PARQUET))
    df = pd.concat(frames, ignore_index=True)
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df


def _write_sample(df: pd.DataFrame, n: int = 5000, seed: int = 0) -> None:
    """Write a small, committable sample of the data for repo browsing."""
    sample = df.sample(n=min(n, len(df)), random_state=seed).sort_index()
    sample.to_parquet(C.SAMPLE_PARQUET, index=False)
    print(f"[data] wrote {len(sample):,}-row sample to {C.SAMPLE_PARQUET}")


def load(sample_ok: bool = False) -> pd.DataFrame:
    """Load the cached batted-ball table.

    Arguments:
        sample_ok: If the full pull is missing, fall back to the committed
            sample rather than raising.

    Returns:
        The batted-ball DataFrame.
    """
    if C.RAW_PARQUET.exists():
        return pd.read_parquet(C.RAW_PARQUET)
    if sample_ok and C.SAMPLE_PARQUET.exists():
        print("[data] WARNING: full pull missing, using the small sample")
        return pd.read_parquet(C.SAMPLE_PARQUET)
    raise FileNotFoundError(
        f"{C.RAW_PARQUET} not found. Run `python -m frb.data` (or `uv run python -m frb.data`) to pull it."
    )


if __name__ == "__main__":
    force = "--force" in sys.argv
    if "--2026" in sys.argv:
        pull_2026(force=force)
    elif "--pybaseball" in sys.argv:
        pull_raw_pybaseball(force=force)
    else:
        pull_raw(force=force)
