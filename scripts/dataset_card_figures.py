"""Generate the explanatory figures shown in the HuggingFace dataset card.

Three views that the weather join makes easy:
1. home-run rate rises with temperature (open-air games),
2. retractable roofs open vs closed change carry,
3. each park's air gives it a carry factor (Coors on top).

Saves PNGs to data/card_figures/.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from frb import config as C

INK, ACCENT, BLUE, GREEN = "#15181c", "#df4f27", "#36627f", "#2f855a"
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

OUT = C.DATA / "card_figures"


def _hr(g):
    return float((g["events"] == "home_run").mean())


def temp_vs_hr(df, path):
    """Home-run rate by temperature for open-air, well-struck balls."""
    sub = df[(~df.roof_closed) & df.launch_speed.between(95, 110) & df.launch_angle.between(18, 42)].copy()
    edges = np.arange(40, 101, 5.0)
    sub["tb"] = pd.cut(sub.temperature_f, edges)
    grp = sub.groupby("tb", observed=True)
    rate = grp.apply(_hr)
    n = grp.size()
    centers = [iv.mid for iv in rate.index]
    keep = n.to_numpy() >= 200
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(np.array(centers)[keep], rate.to_numpy()[keep] * 100, "o-", color=ACCENT, lw=2.3, ms=5)
    ax.set_xlabel("game temperature (F)")
    ax.set_ylabel("home-run rate (%)")
    ax.set_title("Hotter air, more home runs")
    ax.text(
        0.02, 0.96, "open-air games, 95-110 mph and 18-42 deg", transform=ax.transAxes, va="top", color=INK, fontsize=9
    )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def roof_effect(df, path):
    """Mean carry open vs closed at retractable-roof parks, by park."""
    retract = df[(df.roof_type == "Retractable") & df.launch_speed.between(95, 105) & df.launch_angle.between(22, 35)]
    rows = []
    for team, g in retract.groupby("home_team"):
        opn = g[~g.roof_closed]
        cl = g[g.roof_closed]
        if len(opn) >= 200 and len(cl) >= 200:
            rows.append((team, opn.hit_distance_sc.mean(), cl.hit_distance_sc.mean()))
    rows.sort(key=lambda r: r[1] - r[2], reverse=True)
    teams = [r[0] for r in rows]
    x = np.arange(len(teams))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - 0.2, [r[1] for r in rows], 0.4, label="roof open", color=BLUE)
    ax.bar(x + 0.2, [r[2] for r in rows], 0.4, label="roof closed", color=GREEN)
    ax.set_xticks(x)
    ax.set_xticklabels(teams)
    ax.set_ylabel("mean carry (ft)")
    ax.set_ylim(bottom=min(min(r[1], r[2]) for r in rows) - 8)
    ax.set_title("Retractable roofs, open vs closed")
    ax.text(
        0.02,
        0.96,
        "95-105 mph, 22-35 deg; closed games are still air near 72 F",
        transform=ax.transAxes,
        va="top",
        color=INK,
        fontsize=9,
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def park_carry_factors(df, path):
    """Each park's carry factor for a fixed launch band, vs league average."""
    band = df[df.launch_speed.between(95, 105) & df.launch_angle.between(22, 35)]
    league = band.hit_distance_sc.mean()
    by = band.groupby("home_team").agg(
        carry=("hit_distance_sc", "mean"), dens=("air_density", "mean"), n=("hit_distance_sc", "size")
    )
    by = by[by.n >= 500]
    by["factor"] = by.carry - league
    by = by.sort_values("factor")
    colors = [ACCENT if f > 0 else BLUE for f in by.factor]
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.barh(by.index, by.factor, color=colors)
    ax.axvline(0, color=INK, lw=1)
    ax.set_xlabel("carry vs league average (ft), fixed launch band")
    ax.set_title("Park carry factors: the air each park gives")
    ax.text(
        0.98,
        0.04,
        "95-105 mph, 22-35 deg, 2015-2026 (regular + postseason)",
        transform=ax.transAxes,
        ha="right",
        color=INK,
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    df = pd.read_parquet(C.ENRICHED_PARQUET).rename(columns={"temp_f_use": "temperature_f"})
    temp_vs_hr(df, OUT / "temp_vs_hr.png")
    roof_effect(df, OUT / "roof_effect.png")
    park_carry_factors(df, OUT / "park_carry_factors.png")
    print(f"wrote 3 card figures to {OUT}")


if __name__ == "__main__":
    main()
