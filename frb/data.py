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
        WHERE game_type = 'R'
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
        part = part[part["game_type"] == "R"]
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
    use_fallback = "--pybaseball" in sys.argv
    force = "--force" in sys.argv
    if use_fallback:
        pull_raw_pybaseball(force=force)
    else:
        pull_raw(force=force)
