"""Reproduce the whole analysis end to end.

Usage:
    uv run python run.py                 # full run (pulls data if needed)
    uv run python run.py --bootstrap 12  # add a bootstrap CI for the ML model
    uv run python run.py --no-plots      # skip figures

Outputs land in outputs/ (tables, JSON) and figures/ (PNGs).
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from frb import config as C
from frb import empirical as E
from frb import geometry as G
from frb import ml as ML
from frb import physics as P
from frb import plots as PL
from frb import report as R
from frb import spin as SP
from frb import weather as W


def banner(msg: str) -> None:
    print("\n" + "=" * 72 + f"\n{msg}\n" + "=" * 72)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap", type=int, default=0, help="bootstrap reps for the ML P(clear) CI")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--force-pull", action="store_true")
    ap.add_argument(
        "--smoke", action="store_true", help="subsample + skip the exact gate assert, to validate the code path fast"
    )
    args = ap.parse_args()
    t0 = time.time()

    banner("DATA")
    df = W.build_enriched()
    print(f"batted balls: {len(df):,}  seasons {int(df.game_year.min())}-{int(df.game_year.max())}  (weather-enriched)")
    print(f"roof-closed: {df.roof_closed.mean():.1%}  mean air density: {df.air_density.mean():.4f} kg/m^3")
    if args.smoke:
        df = df.sample(n=min(250_000, len(df)), random_state=0).reset_index(drop=True)
        print(f"[smoke] subsampled to {len(df):,} rows (gate assert disabled)")

    # ---------------------------------------------------------------
    banner("METHOD 1: EMPIRICAL POOLING")
    # The validation gate is pinned to regular-season 2015-2025 (the established
    # baseline that proved the pipeline). The reported empirical estimate, and
    # the rest of the France analysis, use the full pool: regular season plus
    # postseason, 2015-2026.
    df_2025 = df[(df.game_type == "R") & (df.game_year <= 2025)]
    gatev = E.window_stats(df_2025, C.GATE_EV_LO, C.GATE_EV_HI, C.GATE_LA_LO, C.GATE_LA_HI, label="gate-2025")
    gate_ok = (
        gatev.n == C.GATE_N
        and round(gatev.mean_carry, 1) == C.GATE_MEAN
        and round(gatev.median_carry, 1) == C.GATE_MEDIAN
        and gatev.k_clear == C.GATE_CLEAR_344
    )
    print(
        f"GATE (2015-2025 checkpoint) n={gatev.n} mean={gatev.mean_carry:.1f} median={gatev.median_carry:.1f} "
        f"k>=344={gatev.k_clear} -> {'PASS' if gate_ok else 'FAIL'}"
    )
    if not args.smoke:
        assert gate_ok, "Validation gate failed: data or filter bug."

    gate = E.window_stats(df, C.GATE_EV_LO, C.GATE_EV_HI, C.GATE_LA_LO, C.GATE_LA_HI, label="gate")
    print(
        f"empirical (2015-2026 pool) n={gate.n} mean={gate.mean_carry:.1f} median={gate.median_carry:.1f} "
        f"k>=344={gate.k_clear} frac={gate.frac_clear:.4f} CI=[{gate.ci_low:.3f},{gate.ci_high:.3f}]"
    )

    windows = E.windows_table(df)
    print("\nwidening windows:")
    print(windows.to_string(index=False))

    kernels = {
        "sigma_1.0": E.kernel_weighted(df, sigma_ev=1.0, sigma_la=1.0),
        "sigma_1.5": E.kernel_weighted(df, sigma_ev=1.5, sigma_la=1.5),
        "sigma_2.0": E.kernel_weighted(df, sigma_ev=2.0, sigma_la=2.0),
    }
    print("\nGaussian-kernel-weighted estimates centered at (105.3, 49):")
    for k, v in kernels.items():
        print(
            f"  {k}: eff_n={v.eff_n:.0f} mean={v.mean_carry:.1f} median={v.median_carry:.1f} "
            f"P(>=344)={v.frac_clear:.4f}"
        )

    # Spray-conditioned pool (down the LF line).
    sp = df[df.spray_deg.between(-45, -40)]
    gate_sp = E.window_stats(sp, C.GATE_EV_LO, C.GATE_EV_HI, C.GATE_LA_LO, C.GATE_LA_HI, label="gate+spray[-45,-40]")
    print(
        f"\nspray-conditioned [-45,-40] at gate EV/LA: n={gate_sp.n} mean={gate_sp.mean_carry:.1f} "
        f"P(>=344)={gate_sp.frac_clear:.4f} CI=[{gate_sp.ci_low:.3f},{gate_sp.ci_high:.3f}]"
    )

    # ---------------------------------------------------------------
    banner("METHOD 2: PHYSICS TRAJECTORY MODEL")
    fit = P.fit_lift_curve(df)
    print(
        f"density-aware fit: {fit.density_aware}  grid bins: {len(fit.grid_obs)}  pool-mean density: {fit.rho_fit:.4f}"
    )
    print(f"fit RMSE = {fit.rmse():.2f} ft")
    print("effective Cl knots:", {int(a): round(float(c), 3) for a, c in zip(fit.knot_la, fit.knot_cl)})
    print("effective drag kd knots:", {int(a): round(float(c), 3) for a, c in zip(fit.knot_la, fit.knot_kd)})
    rho_ref = fit.rho_fit  # pool-mean air density, where the empirical curve lives
    la_fine = np.arange(20, 51, 2.0)
    cv = P.carry_vs_la(105.0, fit, rho_ref, la_fine)
    pk = int(np.argmax(cv))
    print(f"carry-vs-LA @105 mph peak = {cv[pk]:.1f} ft at {la_fine[pk]:.0f} deg (target ~405-410 at 28-32)")

    fp = P.evaluate_point(fit, C.FRANCE_EV, C.FRANCE_LA, C.DOME_TEMP_F, C.DOME_ELEV_M, C.DOME_RH)
    print(
        f"\nFRANCE dome point: carry={fp.carry_ft:.1f} ft  apex={fp.apex_ft:.0f} ft  "
        f"descent={fp.descent_deg:.1f} deg  shortfall vs 344 = {fp.shortfall_ft:.1f} ft"
    )

    # Temperature sensitivity.
    print("\ntemperature sweep (fitted Cl, dome elevation):")
    for tf in (60, 65, 70, 75, 80):
        pt = P.evaluate_point(fit, C.FRANCE_EV, C.FRANCE_LA, tf, C.DOME_ELEV_M, C.DOME_RH)
        print(f"  {tf} F: carry={pt.carry_ft:.1f} ft")

    # Per-ball spin scatter: carry spread of comparable balls at near-dome conditions.
    near = df[df.launch_speed.between(103, 107) & df.launch_angle.between(47, 51) & df.air_density.between(1.14, 1.19)]
    spin_sigma = float(near.hit_distance_sc.std()) if len(near) >= 30 else 20.0
    print(f"spin scatter at France profile (near-dome conditions, n={len(near)}): {spin_sigma:.1f} ft")
    sens = P.sensitivity_mc(fit, spin_sigma_ft=spin_sigma)
    descent = fp.descent_deg
    wall_pen = G.wall_height_distance_penalty(descent)
    wall_park = float(G.wall_distance(C.FRANCE_SPRAY))
    print(
        f"\nsensitivity MC (temp 60-80 F, Cl +/-35%): "
        f"carry mean={sens['mean']:.1f} median={sens['median']:.1f} "
        f"[p05={sens['p05']:.1f}, p95={sens['p95']:.1f}]"
    )
    phys_probs = R.physics_clear_probs(sens["carries"], descent, wall_park)
    print(
        f"physics P(clear 344 raw)={phys_probs['generic_344_raw']['p']:.3f}  "
        f"P(344+8ft wall)={phys_probs['generic_344_wall8']['p']:.3f}"
    )
    print(
        f"8 ft wall distance penalty at descent {descent:.0f} deg = {wall_pen:.1f} ft; "
        f"Milwaukee wall at {C.FRANCE_SPRAY:.0f} deg = {wall_park:.1f} ft"
    )

    # Learn the wind effect from the data (open-air games).
    wind = P.learn_wind_effect(df, fit)
    if wind.get("available"):
        print(
            f"\nwind effect (learned, n={wind['n']:,}): "
            f"{wind['slope_ft_per_mph']:.2f} +/- {wind['slope_se']:.2f} ft per mph of reported tailwind; "
            f"model field sensitivity {wind['model_field_sensitivity_ft_per_mph']:.2f} ft/mph; "
            f"effective field-wind fraction {wind['effective_wind_fraction']:.2f}"
        )

    # ---------------------------------------------------------------
    banner("SPIN FROM THE MEASURED SWING")
    # France's actual swing was measured by Statcast (game_pk 849830): a 75.8 mph
    # barrel on a 21.9 deg uppercut, launching the ball at 49 deg. That is a ~27
    # deg undercut, so a backspin-heavy ball. We cannot measure batted-ball spin
    # (Statcast does not publish it), so we bound it from the collision geometry
    # and read its carry effect empirically from comparable 2024+ balls.
    spin_geo = SP.collision_geometry()
    print(
        f"measured swing: bat {C.FRANCE_BAT_SPEED} mph, attack {C.FRANCE_ATTACK_ANGLE} deg; "
        f"undercut {spin_geo['undercut_deg']} deg, collision obliquity {spin_geo['obliquity_deg']} deg"
    )
    print(
        f"backspin: full-grip ceiling {spin_geo['spin_ceiling_rpm']} rpm; "
        f"estimated {spin_geo['spin_estimate_rpm_lo']:.0f}-{spin_geo['spin_estimate_rpm_hi']:.0f} rpm "
        f"(sliding contact reaches a fraction of the ceiling)"
    )
    attack_slope = SP.empirical_attack_slope(df)
    if attack_slope:
        print(
            f"empirical carry vs attack angle (n={attack_slope['n']}, EV {attack_slope['ev_band']}, "
            f"LA {attack_slope['la_band']}): {attack_slope['slope_ft_per_deg']} ft per deg; "
            f"France's {C.FRANCE_ATTACK_ANGLE} deg -> {attack_slope['carry_at_france_attack']} ft vs "
            f"{attack_slope['carry_at_median_attack']} ft at the band median {attack_slope['median_attack_deg']} deg"
        )
    spin_rng = (spin_geo["spin_estimate_rpm_lo"], spin_geo["spin_estimate_rpm_hi"])

    # ---------------------------------------------------------------
    banner("METHOD 3: PHYSICS-INFORMED MACHINE LEARNING")
    dm = ML.prep_model_frame(df)
    print(f"modeling rows: {len(dm):,}")
    direct = ML.direct_estimate(dm, descent, wall_pen, wall_park, bootstrap=args.bootstrap)
    resid = ML.residual_estimate(dm, fit, descent, wall_pen, wall_park)
    bat = ML.bat_tracking_estimate(df, descent, wall_pen, wall_park)

    for est in [direct, resid, bat]:
        if est is None:
            print("\nbat-tracking model: insufficient 2024+ rows, skipped")
            continue
        print(f"\n[{est.name}] median carry={est.median:.1f} ft")
        print("  quantiles:", {q: round(v, 1) for q, v in est.quantile_values.items()})
        print(
            f"  P(clear 344 raw)={est.p_clear_344:.3f}  +8ft wall={est.p_clear_344_wall:.3f}  "
            f"park={est.p_clear_park:.3f}  park+8ft={est.p_clear_park_wall:.3f}"
        )
        if est.p_clear_ci[0] == est.p_clear_ci[0]:
            print(f"  bootstrap 80% CI on P(clear 344) = [{est.p_clear_ci[0]:.3f}, {est.p_clear_ci[1]:.3f}]")
        if est.coverage:
            print("  holdout interval coverage:", est.coverage)
        if est.importance:
            print("  feature importance:", est.importance)
        if est.extra:
            print("  extra:", est.extra)

    # ---------------------------------------------------------------
    banner("RESULTS TABLE")
    gate_carry = df.loc[
        df.launch_speed.between(C.GATE_EV_LO, C.GATE_EV_HI) & df.launch_angle.between(C.GATE_LA_LO, C.GATE_LA_HI),
        "hit_distance_sc",
    ].to_numpy(float)
    emp_probs = R.empirical_clear_probs(gate_carry, descent, wall_park)

    def prob_cell(d):
        p = d["p"]
        ci = d.get("ci")
        return f"{p * 100:.1f}%" + (f" {R.fmt_ci(ci)}" if ci else "")

    rows = [
        {
            "name": f"Method 1 empirical (2015-2026 pool, n={gate.n})",
            "central_carry": f"{gate.mean_carry:.0f}",
            "generic_344_raw": prob_cell(emp_probs["generic_344_raw"]),
            "generic_344_wall8": prob_cell(emp_probs["generic_344_wall8"]),
            "park_raw": prob_cell(emp_probs["park_raw"]),
            "park_wall8": prob_cell(emp_probs["park_wall8"]),
        },
        {
            "name": "Method 1 kernel (sigma 1.5)",
            "central_carry": f"{kernels['sigma_1.5'].mean_carry:.0f}",
            "generic_344_raw": f"{kernels['sigma_1.5'].frac_clear * 100:.1f}%",
            "generic_344_wall8": "",
            "park_raw": "",
            "park_wall8": "",
        },
        {
            "name": "Method 2 physics (sensitivity MC)",
            "central_carry": f"{fp.carry_ft:.0f}",
            "generic_344_raw": f"{phys_probs['generic_344_raw']['p'] * 100:.1f}%",
            "generic_344_wall8": f"{phys_probs['generic_344_wall8']['p'] * 100:.1f}%",
            "park_raw": f"{phys_probs['park_raw']['p'] * 100:.1f}%",
            "park_wall8": f"{phys_probs['park_wall8']['p'] * 100:.1f}%",
        },
        {
            "name": "Method 3 ML direct",
            "central_carry": f"{direct.median:.0f}",
            "generic_344_raw": f"{direct.p_clear_344 * 100:.1f}%"
            + (f" {R.fmt_ci(direct.p_clear_ci)}" if direct.p_clear_ci[0] == direct.p_clear_ci[0] else ""),
            "generic_344_wall8": f"{direct.p_clear_344_wall * 100:.1f}%",
            "park_raw": f"{direct.p_clear_park * 100:.1f}%",
            "park_wall8": f"{direct.p_clear_park_wall * 100:.1f}%",
        },
        {
            "name": "Method 3 ML physics-residual",
            "central_carry": f"{resid.median:.0f}",
            "generic_344_raw": f"{resid.p_clear_344 * 100:.1f}%",
            "generic_344_wall8": f"{resid.p_clear_344_wall * 100:.1f}%",
            "park_raw": f"{resid.p_clear_park * 100:.1f}%",
            "park_wall8": f"{resid.p_clear_park_wall * 100:.1f}%",
        },
    ]
    if bat is not None:
        rows.append(
            {
                "name": "Method 3 ML + bat-tracking (2024+)",
                "central_carry": f"{bat.median:.0f}",
                "generic_344_raw": f"{bat.p_clear_344 * 100:.1f}%",
                "generic_344_wall8": f"{bat.p_clear_344_wall * 100:.1f}%",
                "park_raw": f"{bat.p_clear_park * 100:.1f}%",
                "park_wall8": f"{bat.p_clear_park_wall * 100:.1f}%",
            }
        )

    results = {
        "event": {
            "ev": C.FRANCE_EV,
            "la": C.FRANCE_LA,
            "spray": C.FRANCE_SPRAY,
            "park": C.FRANCE_PARK,
            "fence_ft": C.FENCE_FT,
            "wall_ft": C.WALL_HEIGHT_FT,
        },
        "data": {
            "n_balls": int(len(df)),
            "year_min": int(df.game_year.min()),
            "year_max": int(df.game_year.max()),
            "roof_closed_frac": round(float(df.roof_closed.mean()), 4),
            "mean_air_density": round(float(df.air_density.mean()), 4),
        },
        "gate_validation_2015_2025": gatev.as_row(),
        "gate": gate.as_row(),
        "gate_pass": bool(gate_ok),
        "windows": windows.to_dict(orient="records"),
        "kernels": {k: v.as_row() for k, v in kernels.items()},
        "spray_conditioned": gate_sp.as_row(),
        "physics": {
            "france_dome_carry": round(fp.carry_ft, 1),
            "shortfall_ft": round(fp.shortfall_ft, 1),
            "descent_deg": round(descent, 1),
            "peak_ft": round(float(cv[pk]), 1),
            "peak_la": float(la_fine[pk]),
            "fit_rmse_ft": round(fit.rmse(), 2),
            "density_aware": bool(fit.density_aware),
            "pool_mean_density": round(float(fit.rho_fit), 4),
            "sensitivity": {k: (round(v, 1) if isinstance(v, float) else v) for k, v in sens.items() if k != "carries"},
            "clear_probs": phys_probs,
            "wind_effect": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in wind.items()},
            "cl_knots": {int(a): round(float(c), 3) for a, c in zip(fit.knot_la, fit.knot_cl)},
            "kd_knots": {int(a): round(float(c), 3) for a, c in zip(fit.knot_la, fit.knot_kd)},
        },
        "empirical_clear_probs": emp_probs,
        "spin": {
            "measured_swing": {
                "bat_speed_mph": C.FRANCE_BAT_SPEED,
                "attack_angle_deg": C.FRANCE_ATTACK_ANGLE,
                "swing_path_tilt_deg": C.FRANCE_SWING_PATH_TILT,
                "swing_length_ft": C.FRANCE_SWING_LENGTH,
                "pitch_mph": C.FRANCE_PITCH_MPH,
                "statcast_hit_distance_ft": C.FRANCE_STATCAST_DIST,
            },
            "collision": spin_geo,
            "empirical_attack_slope": attack_slope,
        },
        "ml": {
            "direct": {
                "median": round(direct.median, 1),
                "quantiles": {q: round(v, 1) for q, v in direct.quantile_values.items()},
                "p_clear_344": round(direct.p_clear_344, 4),
                "p_clear_344_wall8": round(direct.p_clear_344_wall, 4),
                "p_clear_park": round(direct.p_clear_park, 4),
                "p_clear_park_wall8": round(direct.p_clear_park_wall, 4),
                "p_clear_ci": [round(x, 4) if x == x else None for x in direct.p_clear_ci],
                "coverage": direct.coverage,
                "importance": direct.importance,
            },
            "residual": {
                "median": round(resid.median, 1),
                "quantiles": {q: round(v, 1) for q, v in resid.quantile_values.items()},
                "p_clear_344": round(resid.p_clear_344, 4),
                "p_clear_344_wall8": round(resid.p_clear_344_wall, 4),
                "coverage": resid.coverage,
                "importance": resid.importance,
                "extra": resid.extra,
            },
            "bat_tracking": None
            if bat is None
            else {
                "median": round(bat.median, 1),
                "p_clear_344": round(bat.p_clear_344, 4),
                "importance": bat.importance,
                "extra": bat.extra,
            },
        },
        "geometry": {"wall_at_france_spray": round(wall_park, 1), "wall_height_penalty_ft": round(wall_pen, 1)},
        "rows": rows,
    }
    R.save_results(results, C.OUTPUTS / "results.json", C.OUTPUTS / "results_table.md")
    print(R.results_markdown(results))
    print(f"\nwrote {C.OUTPUTS / 'results.json'} and {C.OUTPUTS / 'results_table.md'}")

    # ---------------------------------------------------------------
    if not args.no_plots:
        banner("FIGURES")
        PL.plot_trajectory_profile(
            fit,
            C.FIGURES / "trajectory_profile.png",
            carry_p05=sens["p05"],
            carry_p95=sens["p95"],
            spin_rpm=spin_rng,
        )
        PL.plot_carry_vs_la(df, fit, C.FIGURES / "carry_vs_launch_angle.png")
        PL.plot_france_distribution(df, C.FIGURES / "france_carry_distribution.png")
        PL.plot_ml_distribution(
            direct.quantile_values, C.FIGURES / "ml_predictive_distribution.png", p_clear=direct.p_clear_344
        )
        PL.plot_park_map(C.FIGURES / "park_wall_map.png")
        PL.plot_density_effect(fit, C.FIGURES / "density_effect.png")
        print("saved 6 figures to", C.FIGURES)

    banner("DONE")
    print(f"total time {time.time() - t0:.0f}s")
    print(
        f"\nHEADLINE: expected carry ~{fp.carry_ft:.0f}-{gate.mean_carry:.0f} ft; "
        f"empirical P(reach 344) = {gate.frac_clear:.1%} (Wilson {gate.ci_low:.0%}-{gate.ci_high:.0%}); "
        f"ML P(clear 344) ~ {direct.p_clear_344:.1%}; most likely falls short, but a real ~1-in-8 chance."
    )


if __name__ == "__main__":
    main()
