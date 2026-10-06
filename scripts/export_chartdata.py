"""Export the data behind the web page's charts to outputs/chartdata.json.

Reads the final results.json and the enriched pool, and emits small arrays the
page renders with: the carry-vs-launch-angle curve, the France carry
distribution, the model predictive CDF, the park wall map, the air-density
effect, the wind effect, and the results table.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

from frb import config as C
from frb import geometry as G
from frb import physics as P


def main() -> None:
    res = json.load(open(C.OUTPUTS / "results.json"))
    df = pd.read_parquet(C.ENRICHED_PARQUET)
    out = {}

    out["rows"] = res["rows"]
    out["gate_validation"] = res["gate_validation_2015_2025"]
    out["gate"] = res["gate"]
    out["data"] = res["data"]
    out["event"] = res["event"]
    out["physics"] = {
        "france_dome_carry": res["physics"]["france_dome_carry"],
        "shortfall_ft": res["physics"]["shortfall_ft"],
        "rmse": res["physics"]["fit_rmse_ft"],
        "wind": res["physics"]["wind_effect"],
        "pool_mean_density": res["physics"]["pool_mean_density"],
    }

    # Reconstruct the fitted Cl(LA) and drag kd(LA) from the stored knots.
    cl_k = res["physics"]["cl_knots"]
    kd_k = res["physics"]["kd_knots"]
    kla = np.array(sorted(float(k) for k in cl_k))
    cli = PchipInterpolator(kla, np.array([cl_k[str(int(k))] for k in kla]), extrapolate=True)
    kdi = PchipInterpolator(kla, np.array([kd_k[str(int(k))] for k in kla]), extrapolate=True)
    clamp = lambda a: float(np.clip(a, kla.min(), kla.max()))  # noqa: E731
    rho_ref = res["physics"]["pool_mean_density"]

    # Carry vs launch angle: empirical means and the model curve.
    ev = df.launch_speed.to_numpy(float)
    la = df.launch_angle.to_numpy(float)
    hd = df.hit_distance_sc.to_numpy(float)
    emp_la, emp_mean = [], []
    for a in np.arange(10, 56, 2.0):
        m = (np.abs(ev - 105) <= 1.5) & (np.abs(la - a) <= 1.0)
        if m.sum() >= 30:
            emp_la.append(float(a))
            emp_mean.append(round(float(hd[m].mean()), 1))
    la_fine = np.arange(10, 56, 1.0)
    model = [
        round(float(P.carry(105, float(a), float(cli(clamp(a))), rho_ref, cd=C.CD * float(kdi(clamp(a))))), 1)
        for a in la_fine
    ]
    out["carry_vs_la"] = {
        "emp_la": emp_la,
        "emp_mean": emp_mean,
        "model_la": [float(x) for x in la_fine],
        "model": model,
    }

    # France carry distribution (gate window, full 2015-2026 pool).
    gm = df.launch_speed.between(104, 106.5) & df.launch_angle.between(48, 50)
    carry = sorted(round(float(x), 1) for x in df.loc[gm, "hit_distance_sc"])
    out["france_hist"] = {
        "carry": carry,
        "fence": 344,
        "mean": round(float(np.mean(carry)), 1),
        "cleared": int(sum(c >= 344 for c in carry)),
    }

    # ML predictive quantiles.
    q = res["ml"]["direct"]["quantiles"]
    out["ml_cdf"] = {"q": [float(k) for k in q], "v": [q[k] for k in q], "p_clear": res["ml"]["direct"]["p_clear_344"]}

    # Park map.
    spray = np.linspace(-45, 45, 181)
    out["park"] = {
        "spray": [round(float(s), 1) for s in spray],
        "wall": [round(float(G.wall_distance(float(s))), 1) for s in spray],
        "anchors": {str(k): v for k, v in C.PARK_WALL_ANCHORS.items()},
        "france_spray": -44.0,
        "france_wall": round(float(G.wall_distance(-44)), 1),
    }

    # Air-density effect at the France profile.
    rhos = np.round(np.linspace(0.95, 1.30, 50), 4)
    cl49, cd49 = float(cli(clamp(49))), C.CD * float(kdi(clamp(49)))
    dcarry = [round(float(P.carry(C.FRANCE_EV, C.FRANCE_LA, cl49, float(r), cd=cd49)), 1) for r in rhos]
    dome_rho = round(float(P.rho(C.DOME_TEMP_F, C.DOME_ELEV_M, C.DOME_RH)), 4)
    out["density_curve"] = {
        "rho": [float(r) for r in rhos],
        "carry": dcarry,
        "markers": [
            {
                "label": "Coors Field",
                "rho": 0.99,
                "carry": round(float(P.carry(C.FRANCE_EV, C.FRANCE_LA, cl49, 0.99, cd=cd49)), 1),
            },
            {
                "label": "AmFam dome",
                "rho": dome_rho,
                "carry": round(float(P.carry(C.FRANCE_EV, C.FRANCE_LA, cl49, dome_rho, cd=cd49)), 1),
            },
            {
                "label": "cold sea level",
                "rho": 1.26,
                "carry": round(float(P.carry(C.FRANCE_EV, C.FRANCE_LA, cl49, 1.26, cd=cd49)), 1),
            },
        ],
    }

    json.dump(out, open(C.OUTPUTS / "chartdata.json", "w"))
    print(f"wrote {C.OUTPUTS / 'chartdata.json'}: keys {list(out)}")
    print(f"carry-vs-la emp pts {len(emp_la)} | hist n {len(carry)} cleared {out['france_hist']['cleared']}")


if __name__ == "__main__":
    main()
