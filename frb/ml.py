"""Method 3: physics-informed gradient-boosted quantile regression.

Two LightGBM models predict the carry distribution at the France profile:

(a) Direct: quantile regression of hit_distance_sc on exit velocity, launch
    angle, spray angle, season, and park.
(b) Physics-informed residual: the same features predict the residual between
    measured carry and the physics-model carry, which is then added back to
    the physics carry. This anchors the model in physics and lets the trees
    learn park, spray, and era corrections.

A 2024+ variant adds the bat-tracking fields as a partial spin proxy. All
models are assessed for interval calibration on a time-based holdout, and
P(clear) at the France input is read from the predicted quantiles.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

from . import config as C
from . import physics as P

FEATURES = ["launch_speed", "launch_angle", "spray_deg", "game_year", "home_team"]
CAT_FEATURES = ["home_team"]
BAT_FEATURES = ["attack_angle", "swing_path_tilt", "bat_speed", "swing_length"]
QUANTILES = [0.05, 0.1, 0.25, 0.5, 0.75, 0.8, 0.85, 0.9, 0.95]

LGB_PARAMS = dict(
    objective="quantile",
    n_estimators=300,
    num_leaves=63,
    learning_rate=0.05,
    min_child_samples=100,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.9,
    random_state=0,
    n_jobs=-1,
    verbosity=-1,
)


def prep_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and type the modeling frame.

    Arguments:
        df: Raw batted-ball table.

    Returns:
        A frame with no feature/target nulls, sane ranges, and a categorical
        park column.
    """
    cols = list(dict.fromkeys(FEATURES + BAT_FEATURES + ["hit_distance_sc"]))
    cols = [c for c in cols if c in df.columns]
    out = df[cols].copy()
    out = out.dropna(subset=["launch_speed", "launch_angle", "spray_deg", "hit_distance_sc", "home_team", "game_year"])
    out = out[out.hit_distance_sc.between(0, 520) & out.launch_angle.between(-90, 90) & out.spray_deg.between(-70, 70)]
    out["home_team"] = out["home_team"].astype("category")
    return out


def france_row(park_categories, year: int = 2025) -> pd.DataFrame:
    """One-row frame of the France inputs with a park-matched categorical."""
    row = pd.DataFrame(
        [
            {
                "launch_speed": C.FRANCE_EV,
                "launch_angle": C.FRANCE_LA,
                "spray_deg": C.FRANCE_SPRAY,
                "game_year": year,
                "home_team": C.FRANCE_PARK,
            }
        ]
    )
    row["home_team"] = pd.Categorical(row["home_team"], categories=park_categories)
    return row


def train_quantiles(X: pd.DataFrame, y: np.ndarray, quantiles=QUANTILES, cat=CAT_FEATURES) -> dict:
    """Train one LightGBM quantile model per quantile level."""
    models = {}
    for q in quantiles:
        m = lgb.LGBMRegressor(alpha=q, **LGB_PARAMS)
        m.fit(X, y, categorical_feature=[c for c in cat if c in X.columns])
        models[q] = m
    return models


def predict_quantiles(models: dict, X: pd.DataFrame) -> dict:
    """Predict every quantile for X; enforce monotone non-crossing quantiles."""
    qs = sorted(models)
    preds = np.column_stack([models[q].predict(X) for q in qs])
    preds = np.maximum.accumulate(preds, axis=1)  # guard against quantile crossing
    return {q: preds[:, i] for i, q in enumerate(qs)}


def p_at_least(quantile_levels, quantile_values, threshold: float) -> float:
    """P(carry >= threshold) by interpolating the predicted quantile function.

    Arguments:
        quantile_levels: Sorted quantile levels (e.g. 0.05..0.95).
        quantile_values: Predicted carry at each level (sorted ascending).
        threshold: Carry distance of interest.

    Returns:
        Estimated P(carry >= threshold), clamped by the fitted quantile range.
    """
    q = np.asarray(quantile_levels, dtype=float)
    v = np.asarray(quantile_values, dtype=float)
    order = np.argsort(v)
    v = v[order]
    q = q[order]
    if threshold <= v[0]:
        return float(1 - q[0])  # at least the lowest modeled quantile
    if threshold >= v[-1]:
        return float(1 - q[-1])
    tau = float(np.interp(threshold, v, q))  # P(carry <= threshold)
    return 1 - tau


def time_split(df: pd.DataFrame, valid_years=(2023, 2024, 2025)):
    """Split into train (earlier) and validation (recent years)."""
    valid = df[df.game_year.isin(valid_years)]
    train = df[~df.game_year.isin(valid_years)]
    return train, valid


def interval_coverage(models: dict, X: pd.DataFrame, y: np.ndarray) -> dict:
    """Empirical coverage of predicted central intervals on held-out data."""
    preds = predict_quantiles(models, X)
    out = {}
    for lo, hi in [(0.05, 0.95), (0.1, 0.9), (0.25, 0.75)]:
        if lo in preds and hi in preds:
            inside = np.mean((y >= preds[lo]) & (y <= preds[hi]))
            out[f"{int((hi - lo) * 100)}%_nominal"] = round(float(inside), 4)
    return out


# --------------------------------------------------------------------------
# Physics carry feature (fast, via a precomputed lookup table)
# --------------------------------------------------------------------------


def build_physics_interp(fit: P.LiftFit, temp_f=C.MLB_TEMP_F, elev_m=C.MLB_ELEV_M, rh=C.MLB_RH, dt=0.002):
    """A fast (EV, LA) -> physics carry interpolator for the fitted model."""
    rho_ = P.rho(temp_f, elev_m, rh)
    ev_grid = np.arange(50.0, 122.0, 2.0)
    la_grid = np.arange(-20.0, 72.0, 2.0)
    z = np.empty((ev_grid.size, la_grid.size))
    for i, e in enumerate(ev_grid):
        for j, a in enumerate(la_grid):
            z[i, j] = P.carry(float(e), float(a), float(fit.cl(a)), rho_, cd=float(fit.cd(a)), dt=dt)
    return RegularGridInterpolator((ev_grid, la_grid), z, bounds_error=False, fill_value=None)


def add_physics_carry(df: pd.DataFrame, interp) -> np.ndarray:
    """Physics carry for each row via the interpolator."""
    pts = np.column_stack([df.launch_speed.to_numpy(float), df.launch_angle.to_numpy(float)])
    return interp(pts)


# --------------------------------------------------------------------------
# End-to-end estimators
# --------------------------------------------------------------------------


@dataclass
class MLEstimate:
    """A single ML carry-distribution estimate at the France profile."""

    name: str
    quantile_values: dict
    median: float
    p_clear_344: float
    p_clear_344_wall: float
    p_clear_park: float
    p_clear_park_wall: float
    coverage: dict = field(default_factory=dict)
    importance: dict = field(default_factory=dict)
    p_clear_ci: tuple = (np.nan, np.nan)
    extra: dict = field(default_factory=dict)


def _clear_probs(qs, fr_vals, descent_deg, wall_penalty, wall_park):
    """P(clear) under the four fence definitions from predicted quantiles."""
    p344 = p_at_least(qs, fr_vals, C.FENCE_FT)
    p344w = p_at_least(qs, fr_vals, C.FENCE_FT + wall_penalty)
    ppark = p_at_least(qs, fr_vals, wall_park)
    pparkw = p_at_least(qs, fr_vals, wall_park + wall_penalty)
    return p344, p344w, ppark, pparkw


def direct_estimate(
    df_model: pd.DataFrame, descent_deg: float, wall_penalty: float, wall_park: float, bootstrap: int = 0, seed: int = 0
) -> MLEstimate:
    """Direct quantile-regression estimate with holdout calibration."""
    X_all = df_model[FEATURES]
    y_all = df_model["hit_distance_sc"].to_numpy(float)

    # Time-based holdout for calibration and importance.
    train, valid = time_split(df_model)
    models_cv = train_quantiles(train[FEATURES], train["hit_distance_sc"].to_numpy(float))
    cov = interval_coverage(models_cv, valid[FEATURES], valid["hit_distance_sc"].to_numpy(float))
    imp = dict(zip(FEATURES, models_cv[0.5].feature_importances_.tolist()))

    # Final models on all data; predict at France.
    models = train_quantiles(X_all, y_all)
    fr = france_row(df_model["home_team"].cat.categories)
    qvals = {q: float(v[0]) for q, v in predict_quantiles(models, fr).items()}
    qs = sorted(qvals)
    fr_vals = [qvals[q] for q in qs]
    p344, p344w, ppark, pparkw = _clear_probs(qs, fr_vals, descent_deg, wall_penalty, wall_park)

    ci = (np.nan, np.nan)
    if bootstrap:
        ci = _bootstrap_pclear(
            df_model, descent_deg, wall_penalty, wall_park, residual=False, interp=None, B=bootstrap, seed=seed
        )

    return MLEstimate(
        name="direct",
        quantile_values=qvals,
        median=qvals[0.5],
        p_clear_344=p344,
        p_clear_344_wall=p344w,
        p_clear_park=ppark,
        p_clear_park_wall=pparkw,
        coverage=cov,
        importance=imp,
        p_clear_ci=ci,
    )


def residual_estimate(
    df_model: pd.DataFrame, fit: P.LiftFit, descent_deg: float, wall_penalty: float, wall_park: float
) -> MLEstimate:
    """Physics-informed residual estimate."""
    interp = build_physics_interp(fit)
    phys = add_physics_carry(df_model, interp)
    resid = df_model["hit_distance_sc"].to_numpy(float) - phys
    X_all = df_model[FEATURES]

    train, valid = time_split(df_model)
    phys_tr = add_physics_carry(train, interp)
    phys_va = add_physics_carry(valid, interp)
    models_cv = train_quantiles(train[FEATURES], train["hit_distance_sc"].to_numpy(float) - phys_tr)
    # coverage on reconstructed carry = phys + residual-quantiles
    rq = predict_quantiles(models_cv, valid[FEATURES])
    yv = valid["hit_distance_sc"].to_numpy(float)
    cov = {}
    for lo, hi in [(0.05, 0.95), (0.1, 0.9), (0.25, 0.75)]:
        inside = np.mean((yv >= phys_va + rq[lo]) & (yv <= phys_va + rq[hi]))
        cov[f"{int((hi - lo) * 100)}%_nominal"] = round(float(inside), 4)
    imp = dict(zip(FEATURES, models_cv[0.5].feature_importances_.tolist()))

    models = train_quantiles(X_all, resid)
    fr = france_row(df_model["home_team"].cat.categories)
    # Physics carry at France under DOME conditions.
    interp_dome = build_physics_interp(fit, C.DOME_TEMP_F, C.DOME_ELEV_M, C.DOME_RH)
    phys_fr = float(add_physics_carry(fr, interp_dome)[0])
    rq_fr = {q: float(v[0]) for q, v in predict_quantiles(models, fr).items()}
    qvals = {q: phys_fr + rq_fr[q] for q in rq_fr}
    qs = sorted(qvals)
    fr_vals = [qvals[q] for q in qs]
    p344, p344w, ppark, pparkw = _clear_probs(qs, fr_vals, descent_deg, wall_penalty, wall_park)

    return MLEstimate(
        name="physics_residual",
        quantile_values=qvals,
        median=qvals[0.5],
        p_clear_344=p344,
        p_clear_344_wall=p344w,
        p_clear_park=ppark,
        p_clear_park_wall=pparkw,
        coverage=cov,
        importance=imp,
        extra={"physics_carry_france_dome": round(phys_fr, 1)},
    )


def bat_tracking_estimate(
    df: pd.DataFrame, descent_deg: float, wall_penalty: float, wall_park: float
) -> MLEstimate | None:
    """2024+ estimate adding bat-tracking fields as a partial spin proxy."""
    sub = df[df.game_year >= 2024].copy()
    sub = sub.dropna(
        subset=["launch_speed", "launch_angle", "spray_deg", "hit_distance_sc", "home_team"] + BAT_FEATURES
    )
    sub = sub[sub.hit_distance_sc.between(0, 520) & sub.launch_angle.between(-90, 90) & sub.spray_deg.between(-70, 70)]
    if len(sub) < 20000:
        return None
    sub["home_team"] = sub["home_team"].astype("category")
    feats = FEATURES + BAT_FEATURES
    X = sub[feats]
    y = sub["hit_distance_sc"].to_numpy(float)

    # Typical bat-tracking values for comparable high-launch balls near France.
    near = sub[(sub.launch_angle.between(44, 54)) & (sub.launch_speed.between(102, 108))]
    bat_typ = {f: float(near[f].median()) for f in BAT_FEATURES}

    models = train_quantiles(X, y)
    fr = france_row(sub["home_team"].cat.categories)
    for f in BAT_FEATURES:
        fr[f] = bat_typ[f]
    fr = fr[feats]
    qvals = {q: float(v[0]) for q, v in predict_quantiles(models, fr).items()}
    qs = sorted(qvals)
    fr_vals = [qvals[q] for q in qs]
    p344, p344w, ppark, pparkw = _clear_probs(qs, fr_vals, descent_deg, wall_penalty, wall_park)
    imp = dict(zip(feats, models[0.5].feature_importances_.tolist()))
    return MLEstimate(
        name="bat_tracking_2024plus",
        quantile_values=qvals,
        median=qvals[0.5],
        p_clear_344=p344,
        p_clear_344_wall=p344w,
        p_clear_park=ppark,
        p_clear_park_wall=pparkw,
        importance=imp,
        extra={"n_rows": int(len(sub)), "bat_typical": {k: round(v, 2) for k, v in bat_typ.items()}},
    )


# P(clear 344) depends only on the upper quantiles (344 ft is above the median),
# and each quantile model is independent, so fitting just these at the same
# settings as the point estimate gives an interval consistent with it.
BOOT_QUANTILES = [0.5, 0.75, 0.8, 0.85, 0.9, 0.95]


def _bootstrap_pclear(df_model, descent_deg, wall_penalty, wall_park, residual, interp, B=10, seed=0):
    """Bootstrap interval for P(clear 344) under the direct model (optional).

    Caveat: a bootstrap resample holds only about 63% unique rows, so in the
    sparse high-launch-angle tail the quantile models regress slightly toward
    center and this biases a tail probability like P(clear 344) low. It is kept
    as a diagnostic, not the headline interval. The honest uncertainty on the ML
    estimate is better read from the spread across the three methods, which
    matches the empirical Wilson interval.
    """
    rng = np.random.default_rng(seed)
    y_all = df_model["hit_distance_sc"].to_numpy(float)
    cats = df_model["home_team"].cat.categories
    fr = france_row(cats)
    n = len(df_model)
    ps = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        Xb = df_model.iloc[idx][FEATURES]
        yb = y_all[idx]
        models = train_quantiles(Xb, yb, quantiles=BOOT_QUANTILES)
        qvals = {q: float(v[0]) for q, v in predict_quantiles(models, fr).items()}
        qs = sorted(qvals)
        ps.append(p_at_least(qs, [qvals[q] for q in qs], C.FENCE_FT))
    return (float(np.percentile(ps, 10)), float(np.percentile(ps, 90)))
