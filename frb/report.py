"""Assemble the cross-method results table and summary.

Pulls P(clear) from all three methods under four fence definitions and writes
both a machine-readable JSON and a human-readable Markdown table.
"""

from __future__ import annotations

import json

import numpy as np

from . import config as C
from . import geometry as G
from .stats_utils import wilson_ci


def empirical_clear_probs(carry: np.ndarray, descent_deg: float, wall_park: float) -> dict:
    """P(clear) from a pool of carries under the four fence definitions.

    Arguments:
        carry: Carry distances (ft) of the pooled comparable balls.
        descent_deg: Descent angle (deg) used for the 8 ft wall penalty.
        wall_park: Milwaukee wall distance (ft) at the France spray angle.

    Returns:
        Dict of probabilities and Wilson intervals, with sample size.
    """
    n = carry.size
    pen = G.wall_height_distance_penalty(descent_deg)

    def pack(thresh):
        k = int(np.sum(carry >= thresh))
        lo, hi = wilson_ci(k, n)
        return {"k": k, "n": n, "p": k / n, "ci": [lo, hi], "threshold": round(float(thresh), 1)}

    return {
        "generic_344_raw": pack(C.FENCE_FT),
        "generic_344_wall8": pack(C.FENCE_FT + pen),
        "park_raw": pack(wall_park),
        "park_wall8": pack(wall_park + pen),
    }


def physics_clear_probs(carries: np.ndarray, descent_deg: float, wall_park: float) -> dict:
    """P(clear) from the physics Monte Carlo carry distribution."""
    pen = G.wall_height_distance_penalty(descent_deg)

    def pack(thresh):
        return {"p": float(np.mean(carries >= thresh)), "threshold": round(float(thresh), 1)}

    return {
        "generic_344_raw": pack(C.FENCE_FT),
        "generic_344_wall8": pack(C.FENCE_FT + pen),
        "park_raw": pack(wall_park),
        "park_wall8": pack(wall_park + pen),
    }


def fmt_pct(p: float) -> str:
    return f"{p * 100:.1f}%"


def fmt_ci(ci) -> str:
    if ci is None or (isinstance(ci, (list, tuple)) and (ci[0] != ci[0])):
        return ""
    return f"[{ci[0] * 100:.1f}, {ci[1] * 100:.1f}]"


def results_markdown(results: dict) -> str:
    """Render the master comparison table as Markdown."""
    lines = []
    lines.append("## P(clear) by method and fence definition\n")
    lines.append("All probabilities are P(carry clears the fence). 'raw' uses the")
    lines.append("ground distance; 'wall8' additionally requires the ball to be above")
    lines.append("the 8 ft wall when it arrives. The France ball's measured spray")
    lines.append("(-44 deg) is fair, so fairness does not reduce these further.\n")
    header = "| method | central carry (ft) | generic 344 raw | generic 344 + 8ft wall | Milwaukee park raw | Milwaukee park + 8ft wall |"
    sep = "|---|---|---|---|---|---|"
    lines.append(header)
    lines.append(sep)
    for r in results["rows"]:
        lines.append(
            "| {name} | {carry} | {g_raw} | {g_wall} | {p_raw} | {p_wall} |".format(
                name=r["name"],
                carry=r.get("central_carry", ""),
                g_raw=r["generic_344_raw"],
                g_wall=r["generic_344_wall8"],
                p_raw=r["park_raw"],
                p_wall=r["park_wall8"],
            )
        )
    return "\n".join(lines)


def save_results(results: dict, json_path, md_path) -> None:
    """Write results to JSON and Markdown."""
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=_json_default)
    with open(md_path, "w") as f:
        f.write(results_markdown(results))


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(str(type(o)))
