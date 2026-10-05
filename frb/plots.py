"""Figures for the analysis.

Four plots:
1. carry vs launch angle, empirical means against the fitted physics curve
2. the carry distribution at the France profile, with the 344 ft line
3. the LightGBM predictive distribution at the France input
4. a top-down map of the park wall distance vs spray angle
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as C
from . import geometry as G
from . import physics as P

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.dpi": 130,
        "font.size": 11,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

INK = "#1b2a4a"
ACCENT = "#c1432b"
BLUE = "#2b6cb0"
GREEN = "#2f855a"


def plot_carry_vs_la(df: pd.DataFrame, fit: P.LiftFit, path, ev: float = 105.0) -> None:
    """Empirical mean carry vs launch angle against the fitted physics curve."""
    evd = df.launch_speed.to_numpy(float)
    lad = df.launch_angle.to_numpy(float)
    hd = df.hit_distance_sc.to_numpy(float)
    la_centers = np.arange(10, 56, 2.0)
    means, sems, las_ok = [], [], []
    for a in la_centers:
        m = (np.abs(evd - ev) <= 1.5) & (np.abs(lad - a) <= 1.0)
        if m.sum() >= 30:
            d = hd[m]
            means.append(d.mean())
            sems.append(d.std() / np.sqrt(m.sum()))
            las_ok.append(a)
    las_ok = np.array(las_ok)
    rho_mlb = P.rho(C.MLB_TEMP_F, C.MLB_ELEV_M, C.MLB_RH)
    la_fine = np.arange(10, 56, 1.0)
    model = P.carry_vs_la(ev, fit, rho_mlb, la_fine)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(
        las_ok,
        means,
        yerr=sems,
        fmt="o",
        color=INK,
        ms=5,
        capsize=2,
        label=f"empirical mean carry (EV {ev:.0f} +/- 1.5)",
    )
    ax.plot(la_fine, model, "-", color=ACCENT, lw=2, label="fitted physics model")
    ax.axvline(C.FRANCE_LA, color=GREEN, ls="--", lw=1.5, alpha=0.8)
    ax.annotate("France 49 deg", xy=(C.FRANCE_LA, 300), xytext=(C.FRANCE_LA + 0.6, 360), color=GREEN, fontsize=10)
    ax.set_xlabel("launch angle (deg)")
    ax.set_ylabel("carry distance (ft)")
    ax.set_title("Carry vs launch angle at ~105 mph: data and fitted physics model")
    ax.legend(loc="lower center")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_france_distribution(df: pd.DataFrame, path) -> None:
    """Carry distribution at the France profile, with the 344 ft line marked."""
    m = df.launch_speed.between(C.GATE_EV_LO, C.GATE_EV_HI) & df.launch_angle.between(C.GATE_LA_LO, C.GATE_LA_HI)
    carry = df.loc[m, "hit_distance_sc"].to_numpy(float)
    n = carry.size
    frac = float(np.mean(carry >= C.FENCE_FT))

    fig, ax = plt.subplots(figsize=(8, 5))
    # Anchor bins so the 344 ft fence lands exactly on a bin edge, keeping the
    # cleared (red) region cleanly separated from the rest.
    width = 8.0
    lo = C.FENCE_FT - width * np.ceil((C.FENCE_FT - carry.min()) / width)
    hi = C.FENCE_FT + width * np.ceil((carry.max() - C.FENCE_FT) / width) + width
    bins = np.arange(lo, hi, width)
    ax.hist(
        carry, bins=bins, color=BLUE, alpha=0.55, edgecolor="white", label=f"n = {n} balls (104-106.5 mph, 48-50 deg)"
    )
    over = carry[carry >= C.FENCE_FT]
    ax.hist(
        over,
        bins=bins,
        color=ACCENT,
        alpha=0.85,
        edgecolor="white",
        label=f"cleared 344 ft: {over.size}/{n} = {frac:.1%}",
    )
    ax.axvline(C.FENCE_FT, color=INK, lw=2, ls="--")
    ax.axvline(carry.mean(), color=GREEN, lw=1.5)
    ax.annotate(
        "344 ft fence",
        xy=(C.FENCE_FT, ax.get_ylim()[1] * 0.9),
        xytext=(C.FENCE_FT + 3, ax.get_ylim()[1] * 0.9),
        color=INK,
    )
    ax.annotate(
        f"mean {carry.mean():.0f} ft",
        xy=(carry.mean(), ax.get_ylim()[1] * 0.75),
        xytext=(carry.mean() - 70, ax.get_ylim()[1] * 0.75),
        color=GREEN,
    )
    ax.set_xlabel("carry distance (ft)")
    ax.set_ylabel("count")
    ax.set_title("Carry distribution at the France profile (empirical pool)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_ml_distribution(quantile_values: dict, path, p_clear: float | None = None) -> None:
    """LightGBM predictive distribution at the France input, from its quantiles."""
    qs = np.array(sorted(quantile_values))
    vals = np.array([quantile_values[q] for q in qs])

    fig, ax = plt.subplots(figsize=(8, 5))
    # Predictive CDF from the quantile function.
    ax.plot(vals, qs, "o-", color=BLUE, lw=2, label="predicted carry CDF (direct quantile model)")
    ax.axvline(C.FENCE_FT, color=INK, ls="--", lw=2)
    ax.annotate("344 ft fence", xy=(C.FENCE_FT, 0.5), xytext=(C.FENCE_FT - 1, 0.42), color=INK, ha="right")
    if p_clear is not None:
        tau = 1 - p_clear
        ax.axhline(tau, color=ACCENT, ls=":", lw=1.2)
        ax.annotate(
            f"P(carry >= 344) = {p_clear:.1%}",
            xy=(vals.min(), tau),
            xytext=(vals.min(), min(tau + 0.06, 0.95)),
            color=ACCENT,
        )
    ax.set_xlabel("carry distance (ft)")
    ax.set_ylabel("cumulative probability")
    ax.set_title("Model predictive carry distribution at the France input")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_park_map(path) -> None:
    """Top-down map of the park wall distance as a function of spray angle."""
    spray = np.linspace(C.FAIR_MIN_DEG, C.FAIR_MAX_DEG, 200)
    wall = G.wall_distance(spray)
    # Top-down: x to the right, y toward center field.
    rad = np.radians(spray)
    wx = wall * np.sin(rad)
    wy = wall * np.cos(rad)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(wx, wy, "-", color=INK, lw=2.5, label="outfield wall")
    # Foul lines.
    for s in (C.FAIR_MIN_DEG, C.FAIR_MAX_DEG):
        r = np.radians(s)
        d = G.wall_distance(s)
        ax.plot([0, d * np.sin(r)], [0, d * np.cos(r)], color="#999", lw=1)
    # Anchors.
    for s, d in C.PARK_WALL_ANCHORS.items():
        r = np.radians(s)
        ax.plot(d * np.sin(r), d * np.cos(r), "o", color=BLUE, ms=5)
        ax.annotate(
            f"{d:.0f}",
            xy=(d * np.sin(r), d * np.cos(r)),
            xytext=(d * np.sin(r) + 4, d * np.cos(r) + 4),
            fontsize=8,
            color=BLUE,
        )
    # France direction.
    rf = np.radians(C.FRANCE_SPRAY)
    df_wall = float(G.wall_distance(C.FRANCE_SPRAY))
    ax.plot(
        [0, 360 * np.sin(rf)],
        [0, 360 * np.cos(rf)],
        color=ACCENT,
        lw=2,
        ls="--",
        label=f"France line ({C.FRANCE_SPRAY:.0f} deg, wall {df_wall:.0f} ft)",
    )
    ax.plot(0, 0, "s", color="k", ms=7)
    ax.annotate("home", xy=(0, 0), xytext=(6, -14), fontsize=9)
    ax.set_aspect("equal")
    ax.set_xlabel("feet (toward right field +)")
    ax.set_ylabel("feet (toward center field)")
    ax.set_title("American Family Field wall distance by spray angle")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
