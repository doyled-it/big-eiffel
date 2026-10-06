"""Figures for the analysis.

Plots:
1. carry vs launch angle, empirical means against the fitted physics curve
2. the carry distribution at the France profile, with the 344 ft line
3. the LightGBM predictive distribution at the France input
4. a top-down map of the park wall distance vs spray angle
5. the France ball's carry across air densities (the altitude and weather effect)
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
    rho_ref = fit.rho_fit  # pool-mean air density, matching the empirical means
    la_fine = np.arange(10, 56, 1.0)
    model = P.carry_vs_la(ev, fit, rho_ref, la_fine)

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


def plot_trajectory_profile(
    fit: P.LiftFit,
    path,
    carry_p05: float | None = None,
    carry_p95: float | None = None,
    spin_rpm: tuple[float, float] | None = None,
    landing_samples=None,
) -> None:
    """Headline profile: the free-flight arc and the cable-interrupted ball.

    A true-scale side view from home plate. The solid arc is the fitted physics
    model's free flight for France's launch at the closed-roof dome air density
    (the backspin that lifts a 49 deg ball is already in the calibrated lift, so
    the arc has a real fly ball's climb and descent, not a bare parabola). A ball
    cannot clip a cable higher than its own apex, so "near the apex" fixes the
    strike to the top of the arc. A cable only shortens a flight, so the dashed
    path after the strike comes down just inside the free flight, deep in left
    field short of the wall, consistent with the ball's Statcast distance. The
    344 ft fence (8 ft tall) is drawn to scale, and a rug shows the model's
    landing distribution, red where it reaches the fence.

    Arguments:
        fit: The fitted physics model.
        path: Output image path.
        carry_p05: 5th percentile of the landing distribution (ft).
        carry_p95: 95th percentile of the landing distribution (ft).
        spin_rpm: (low, high) estimated backspin range, for the annotation.
        landing_samples: Sample of model landing distances for the rug.
    """
    rho_dome = P.rho(C.DOME_TEMP_F, C.DOME_ELEV_M, C.DOME_RH)
    cl0 = float(fit.cl(C.FRANCE_LA))
    cd0 = float(fit.cd(C.FRANCE_LA))
    tr = P.integrate(C.FRANCE_EV, C.FRANCE_LA, cl0, rho_dome, cd=cd0)
    apex_i = int(np.argmax(tr.z_ft))
    apex_x, apex_z = float(tr.x_ft[apex_i]), float(tr.z_ft[apex_i])
    land = float(tr.range_ft)
    catch = 315.0  # cable-clipped ball: deep left, just inside the free-flight spot

    # Aerodynamic envelope: sample temperature and lift (spin), min/max height per distance.
    rng = np.random.default_rng(0)
    xg = np.linspace(0, land, 240)
    lo = np.full_like(xg, np.inf)
    hi = np.full_like(xg, -np.inf)
    for _ in range(40):
        t = rng.uniform(60.0, 80.0)
        s = float(np.clip(rng.normal(1.0, 0.35 / 2.0), 0.4, 1.9))
        trj = P.integrate(C.FRANCE_EV, C.FRANCE_LA, cl0 * s, P.rho(t, C.DOME_ELEV_M, C.DOME_RH), cd=cd0)
        z = np.interp(xg, trj.x_ft, trj.z_ft, right=np.nan)
        lo = np.fmin(lo, z)
        hi = np.fmax(hi, z)

    fig, ax = plt.subplots(figsize=(11, 5.4))
    ax.set_facecolor("#f7f8fa")
    ax.axhline(0, color="#8a8f99", lw=1)
    ax.plot(0, 0, "D", color="k", ms=7)
    ax.annotate("home plate", xy=(0, 0), xytext=(5, 7), fontsize=9, color=INK)

    ax.fill_between(xg, np.nan_to_num(lo), np.nan_to_num(hi), color=ACCENT, alpha=0.13, lw=0,
                    label="plausible range (temperature + spin)")
    ax.plot(tr.x_ft, tr.z_ft, "-", color=ACCENT, lw=2.6, label=f"model free flight → {land:.0f} ft")

    # Interrupted path: from the cable (apex) down to the catch, hugging the free flight.
    tt = np.linspace(0, 1, 60)
    xi = apex_x + (catch - apex_x) * tt
    zi = apex_z * (1 - tt) ** 1.35
    ax.plot(xi, np.clip(zi, 0, None), ":", color="#6a6a6a", lw=1.9, label="path after the cable (estimated)")

    # The 344 ft fence, 8 ft tall, to scale; labels to its right.
    ax.add_patch(plt.Rectangle((C.FENCE_FT, 0), 2.4, C.WALL_HEIGHT_FT, color=INK))
    ax.annotate(f"{C.FENCE_FT:.0f} ft fence", xy=(C.FENCE_FT + 6, 14), fontsize=9.5, color=INK, fontweight="bold")
    ax.annotate(f"{C.WALL_HEIGHT_FT:.0f} ft high", xy=(C.FENCE_FT + 6, 5), fontsize=9, color=INK)

    # Cable strike at the apex (best estimate), in gold so it reads as the cable.
    ax.plot([apex_x - 13, apex_x + 13], [apex_z, apex_z], "-", color="#d99a2b", lw=3)
    ax.plot(apex_x, apex_z, "o", color="#d99a2b", ms=8, markeredgecolor=INK, markeredgewidth=0.6)
    ax.annotate(
        f"clipped the roof cable near its apex\nbest estimate: ~{apex_z:.0f} ft up, {apex_x:.0f} ft out",
        xy=(apex_x, apex_z),
        xytext=(apex_x - 168, apex_z + 8),
        fontsize=9,
        color="#333",
        arrowprops=dict(arrowstyle="->", color="#333", lw=1),
    )

    # Catch marker: where the clipped ball came down (deep left, short).
    ax.plot(catch, 3, "o", color=INK, ms=6)

    # Shortfall from the free-flight landing to the fence.
    ax.annotate("", xy=(C.FENCE_FT, 27), xytext=(land, 27), arrowprops=dict(arrowstyle="<->", color=GREEN, lw=1.4))
    ax.annotate(f"{C.FENCE_FT - land:.0f} ft short", xy=((land + C.FENCE_FT) / 2, 27),
                xytext=((land + C.FENCE_FT) / 2 - 14, 32), fontsize=9, color=GREEN)

    # Landing distribution rug.
    if landing_samples is not None and len(landing_samples):
        ss = np.asarray(landing_samples, float)
        cols = np.where(ss >= C.FENCE_FT, ACCENT, BLUE)
        ax.vlines(ss, -9, -4, colors=cols, lw=1, alpha=0.5)
        frac = float(np.mean(ss >= C.FENCE_FT))
        lab = "model landing distribution"
        if carry_p05 is not None:
            lab += f": 5th-95th pct {carry_p05:.0f}-{carry_p95:.0f} ft"
        lab += f";  {frac * 100:.0f}% reach the fence (red)"
        ax.annotate(lab, xy=(np.median(ss), -9), xytext=(55, -20), fontsize=8.5, color=BLUE)

    sub = "" if spin_rpm is None else f"   ·   estimated backspin ≈ {spin_rpm[0]:.0f}–{spin_rpm[1]:.0f} rpm"
    ax.set_title("Uninterrupted, the ball still comes down short of the fence" + sub, fontsize=12)
    ax.set_xlabel("distance from home plate (ft)")
    ax.set_ylabel("height (ft)")
    ax.set_xlim(-12, 392)
    ax.set_ylim(-27, apex_z + 38)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_density_effect(fit: P.LiftFit, path) -> None:
    """The France ball's carry across air densities (altitude and weather)."""
    rhos = np.linspace(0.95, 1.30, 80)
    cl = float(fit.cl(C.FRANCE_LA))
    cd = float(fit.cd(C.FRANCE_LA))
    carries = P.carry_vec(
        np.full_like(rhos, C.FRANCE_EV),
        np.full_like(rhos, C.FRANCE_LA),
        np.full_like(rhos, cl),
        rhos,
        np.full_like(rhos, cd),
    )
    marks = [
        ("Coors Field, 1580 m", 0.990, ACCENT),
        ("American Family Field dome", float(P.rho(C.DOME_TEMP_F, C.DOME_ELEV_M, C.DOME_RH)), GREEN),
        ("cold, dense, sea level", 1.260, BLUE),
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(rhos, carries, "-", color=INK, lw=2.5)
    ax.axhline(C.FENCE_FT, color=ACCENT, ls="--", lw=1.8)
    ax.annotate("344 ft fence", xy=(0.96, C.FENCE_FT), xytext=(0.96, C.FENCE_FT + 3), color=ACCENT, fontsize=10)
    for label, rr, col in marks:
        cc = float(P.carry(C.FRANCE_EV, C.FRANCE_LA, cl, rr, cd=cd))
        ax.plot(rr, cc, "o", color=col, ms=8)
        ax.annotate(f"{label}\n{cc:.0f} ft", xy=(rr, cc), xytext=(rr, cc - 18), color=col, fontsize=9, ha="center")
    ax.set_xlabel("air density (kg/m^3)  ·  thinner air to the left")
    ax.set_ylabel("carry distance (ft)")
    ax.set_title("The same France ball carries 344 ft only in thin air")
    ax.invert_xaxis()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
