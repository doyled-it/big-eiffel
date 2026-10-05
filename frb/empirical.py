"""Method 1: empirical pooling.

The model-free baseline. Pool every comparable batted ball and read the
carry distribution straight off the data, both in fixed (EV, LA) windows and
with a Gaussian kernel centered exactly on the France profile.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config as C
from .stats_utils import effective_n, weighted_quantile, wilson_ci


@dataclass
class WindowResult:
    """Carry statistics for one (EV, LA) window."""

    label: str
    ev_lo: float
    ev_hi: float
    la_lo: float
    la_hi: float
    n: int
    mean_carry: float
    median_carry: float
    k_clear: int
    frac_clear: float
    ci_low: float
    ci_high: float
    fence_ft: float

    def as_row(self) -> dict:
        return {
            "label": self.label,
            "ev_lo": self.ev_lo,
            "ev_hi": self.ev_hi,
            "la_lo": self.la_lo,
            "la_hi": self.la_hi,
            "n": self.n,
            "mean_carry": round(self.mean_carry, 1),
            "median_carry": round(self.median_carry, 1),
            "k_clear": self.k_clear,
            "frac_clear": round(self.frac_clear, 4),
            "ci_low": round(self.ci_low, 4),
            "ci_high": round(self.ci_high, 4),
            "fence_ft": self.fence_ft,
        }


def window_stats(
    df: pd.DataFrame,
    ev_lo: float,
    ev_hi: float,
    la_lo: float,
    la_hi: float,
    fence_ft: float = C.FENCE_FT,
    label: str = "",
) -> WindowResult:
    """Carry statistics for balls inside an (EV, LA) box.

    Arguments:
        df: Batted-ball table with launch_speed, launch_angle, hit_distance_sc.
        ev_lo: Lower exit-velocity bound (inclusive).
        ev_hi: Upper exit-velocity bound (inclusive).
        la_lo: Lower launch-angle bound (inclusive).
        la_hi: Upper launch-angle bound (inclusive).
        fence_ft: Distance the ball must reach to count as a clear.
        label: Human-readable label for the window.

    Returns:
        A :class:`WindowResult`.
    """
    m = df.launch_speed.between(ev_lo, ev_hi) & df.launch_angle.between(la_lo, la_hi)
    sub = df.loc[m, "hit_distance_sc"].to_numpy(dtype=float)
    n = int(sub.size)
    if n == 0:
        return WindowResult(label, ev_lo, ev_hi, la_lo, la_hi, 0, np.nan, np.nan, 0, np.nan, np.nan, np.nan, fence_ft)
    k = int(np.sum(sub >= fence_ft))
    lo, hi = wilson_ci(k, n)
    return WindowResult(
        label=label,
        ev_lo=ev_lo,
        ev_hi=ev_hi,
        la_lo=la_lo,
        la_hi=la_hi,
        n=n,
        mean_carry=float(np.mean(sub)),
        median_carry=float(np.median(sub)),
        k_clear=k,
        frac_clear=k / n,
        ci_low=lo,
        ci_high=hi,
        fence_ft=fence_ft,
    )


# The validation gate plus a set of widening windows around 105.3 / 49.
WINDOWS = [
    ("gate 104-106.5 / 48-50", C.GATE_EV_LO, C.GATE_EV_HI, C.GATE_LA_LO, C.GATE_LA_HI),
    ("+-0.5 / +-0.5", 104.8, 105.8, 48.5, 49.5),
    ("+-1.0 / +-1.0", 104.3, 106.3, 48.0, 50.0),
    ("+-1.5 / +-1.5", 103.8, 106.8, 47.5, 50.5),
    ("+-2.0 / +-2.0", 103.3, 107.3, 47.0, 51.0),
    ("+-2.5 / +-3.0", 102.8, 107.8, 46.0, 52.0),
]


def windows_table(df: pd.DataFrame, fence_ft: float = C.FENCE_FT) -> pd.DataFrame:
    """Build the windows table around the France profile.

    Arguments:
        df: Batted-ball table.
        fence_ft: Clear distance.

    Returns:
        DataFrame, one row per window.
    """
    rows = [
        window_stats(df, elo, ehi, llo, lhi, fence_ft=fence_ft, label=label).as_row()
        for (label, elo, ehi, llo, lhi) in WINDOWS
    ]
    return pd.DataFrame(rows)


@dataclass
class KernelResult:
    """Gaussian-kernel-weighted carry estimate at an exact (EV, LA) point."""

    ev0: float
    la0: float
    sigma_ev: float
    sigma_la: float
    eff_n: float
    mean_carry: float
    median_carry: float
    frac_clear: float
    fence_ft: float
    carry_samples: np.ndarray = field(repr=False, default=None)
    weights: np.ndarray = field(repr=False, default=None)

    def as_row(self) -> dict:
        return {
            "ev0": self.ev0,
            "la0": self.la0,
            "sigma_ev": self.sigma_ev,
            "sigma_la": self.sigma_la,
            "eff_n": round(self.eff_n, 1),
            "mean_carry": round(self.mean_carry, 1),
            "median_carry": round(self.median_carry, 1),
            "frac_clear": round(self.frac_clear, 4),
            "fence_ft": self.fence_ft,
        }


def kernel_weighted(
    df: pd.DataFrame,
    ev0: float = C.FRANCE_EV,
    la0: float = C.FRANCE_LA,
    sigma_ev: float = 1.5,
    sigma_la: float = 1.5,
    fence_ft: float = C.FENCE_FT,
    truncate: float = 4.0,
) -> KernelResult:
    """Gaussian-kernel-weighted carry estimate centered on (ev0, la0).

    Weights each ball by a 2D Gaussian in (EV, LA) so the estimate is anchored
    exactly at the France profile rather than forced into a bin.

    Arguments:
        df: Batted-ball table.
        ev0: Exit velocity to center on.
        la0: Launch angle to center on.
        sigma_ev: Gaussian bandwidth in exit velocity (mph).
        sigma_la: Gaussian bandwidth in launch angle (deg).
        fence_ft: Clear distance.
        truncate: Drop points beyond this many sigma in either axis for speed.

    Returns:
        A :class:`KernelResult`.
    """
    ev = df.launch_speed.to_numpy(dtype=float)
    la = df.launch_angle.to_numpy(dtype=float)
    carry = df.hit_distance_sc.to_numpy(dtype=float)

    near = (np.abs(ev - ev0) <= truncate * sigma_ev) & (np.abs(la - la0) <= truncate * sigma_la)
    ev, la, carry = ev[near], la[near], carry[near]

    w = np.exp(-0.5 * ((ev - ev0) / sigma_ev) ** 2 - 0.5 * ((la - la0) / sigma_la) ** 2)
    wsum = w.sum()
    mean_carry = float(np.sum(w * carry) / wsum)
    median_carry = weighted_quantile(carry, w, 0.5)
    frac_clear = float(np.sum(w * (carry >= fence_ft)) / wsum)
    return KernelResult(
        ev0=ev0,
        la0=la0,
        sigma_ev=sigma_ev,
        sigma_la=sigma_la,
        eff_n=effective_n(w),
        mean_carry=mean_carry,
        median_carry=median_carry,
        frac_clear=frac_clear,
        fence_ft=fence_ft,
        carry_samples=carry,
        weights=w,
    )
