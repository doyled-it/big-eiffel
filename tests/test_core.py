"""Unit tests for the geometry, stats, physics, and ML helpers.

The heavy data-dependent validation gate is tested separately and skips when
the cached pull is absent.
"""

from __future__ import annotations

import numpy as np
import pytest

from frb import config as C
from frb import data as D
from frb import geometry as G
from frb import ml
from frb import physics as P
from frb.stats_utils import weighted_quantile, wilson_ci

# --------------------------------------------------------------------------
# Spray angle
# --------------------------------------------------------------------------


def test_spray_sign_and_scale():
    # A ball straight up the middle lands at x = HC_X0, y < HC_Y0 -> ~0 deg.
    assert abs(float(D.spray_angle(C.HC_X0, C.HC_Y0 - 100))) < 1e-6
    # Pulled to left field (smaller x) -> negative spray.
    assert D.spray_angle(C.HC_X0 - 100, C.HC_Y0 - 100) < 0
    # Pushed to right field (larger x) -> positive spray.
    assert D.spray_angle(C.HC_X0 + 100, C.HC_Y0 - 100) > 0
    # 45 deg to left field.
    assert abs(float(D.spray_angle(C.HC_X0 - 100, C.HC_Y0 - 100)) + 45) < 1e-6


# --------------------------------------------------------------------------
# Wilson interval and weighted quantile
# --------------------------------------------------------------------------


def test_wilson_matches_known_value():
    lo, hi = wilson_ci(31, 250)
    assert abs(lo - 0.089) < 0.005
    assert abs(hi - 0.171) < 0.005


def test_wilson_degenerate():
    assert wilson_ci(0, 0) == (0.0, 0.0)


def test_weighted_quantile_uniform():
    v = np.arange(0, 101, dtype=float)
    w = np.ones_like(v)
    assert abs(weighted_quantile(v, w, 0.5) - 50) < 1.0


# --------------------------------------------------------------------------
# Park geometry
# --------------------------------------------------------------------------


def test_wall_anchors_interpolated():
    assert abs(float(G.wall_distance(-45)) - 344) < 1e-6
    assert abs(float(G.wall_distance(0)) - 400) < 1e-6
    # The France line sits just inside the foul pole, near the foul-line value.
    assert 343 <= float(G.wall_distance(C.FRANCE_SPRAY)) <= 360


def test_fairness():
    assert bool(G.is_fair(-44))
    assert not bool(G.is_fair(-46))
    assert not bool(G.is_fair(50))


def test_wall_height_penalty_positive_and_monotone():
    steep = G.wall_height_distance_penalty(60)
    shallow = G.wall_height_distance_penalty(40)
    assert steep > 0
    # A steeper descent needs less extra distance to clear the same wall.
    assert steep < shallow


def test_clears_logic():
    # Carry past the wall and fair -> clears.
    assert bool(G.clears_generic(360, spray_deg=-44, fence_ft=344))
    # Short of the fence -> does not clear.
    assert not bool(G.clears_generic(300, spray_deg=-44, fence_ft=344))
    # Foul -> does not clear even if far enough.
    assert not bool(G.clears_generic(400, spray_deg=-50, fence_ft=344))


# --------------------------------------------------------------------------
# Physics
# --------------------------------------------------------------------------


def test_rho_cold_is_denser():
    assert P.rho(40, 0, 0.5) > P.rho(100, 0, 0.5)


def test_rho_altitude_thins_air():
    assert P.rho(70, 0, 0.5) > P.rho(70, 1600, 0.5)  # Denver-ish


def test_carry_increases_with_ev_at_optimal_angle():
    rho_ = P.rho(70, 150, 0.5)
    assert P.carry(100, 28, 0.15, rho_) < P.carry(108, 28, 0.15, rho_)


def test_carry_reasonable_magnitude():
    rho_ = P.rho(70, 150, 0.5)
    d = P.carry(105, 30, 0.15, rho_)
    assert 380 < d < 450  # a well-struck ball at ~30 deg


def test_trajectory_height_at_consistency():
    rho_ = P.rho(70, 150, 0.5)
    tr = P.integrate(105, 30, 0.15, rho_)
    assert tr.height_at(tr.range_ft + 10) == 0.0
    assert tr.height_at(0) > 0
    assert tr.apex_ft > 50


# --------------------------------------------------------------------------
# ML quantile-to-probability
# --------------------------------------------------------------------------


def test_p_at_least_interpolation():
    qs = [0.25, 0.5, 0.75]
    vals = [300.0, 320.0, 340.0]
    # Threshold at the median -> P(>=) = 0.5.
    assert abs(ml.p_at_least(qs, vals, 320.0) - 0.5) < 1e-6
    # Threshold above all quantiles -> clamps to 1 - top quantile.
    assert abs(ml.p_at_least(qs, vals, 400.0) - (1 - 0.75)) < 1e-6
    # Monotone: a higher threshold gives a lower probability.
    assert ml.p_at_least(qs, vals, 330.0) < ml.p_at_least(qs, vals, 310.0)


# --------------------------------------------------------------------------
# Data-dependent validation gate (skips without the cached pull)
# --------------------------------------------------------------------------


@pytest.mark.skipif(not C.RAW_PARQUET.exists(), reason="raw data not pulled")
def test_validation_gate():
    import pandas as pd

    df = pd.read_parquet(C.RAW_PARQUET)
    # The gate is pinned to regular-season 2015-2025; the pool also holds postseason.
    m = (
        (df.game_type == "R")
        & (df.game_year <= 2025)
        & df.launch_speed.between(C.GATE_EV_LO, C.GATE_EV_HI)
        & df.launch_angle.between(C.GATE_LA_LO, C.GATE_LA_HI)
    )
    carry = df.loc[m, "hit_distance_sc"].to_numpy(float)
    assert carry.size == C.GATE_N
    assert round(float(carry.mean()), 1) == C.GATE_MEAN
    assert round(float(np.median(carry)), 1) == C.GATE_MEDIAN
    assert int((carry >= C.FENCE_FT).sum()) == C.GATE_CLEAR_344
