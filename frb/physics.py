"""Method 2: a calibrated drag-plus-Magnus trajectory model.

A quadratic-drag, Magnus-lift point-mass integrator. The drag coefficient is
fixed; an effective lift coefficient is fit as a smooth function of launch
angle so the model reproduces the empirical mean-carry surface in average MLB
conditions (not calibrated at a single point). The fitted model is then
evaluated at the France profile under closed-roof dome conditions, with a
sensitivity sweep over temperature and over a plausible (unmeasured) spin
range.

A note on the effective lift coefficient. Below the distance-optimal launch
angle (~30 deg) more backspin lift means more carry, so the fit lands on a
low, physical Cl. Above it the Magnus force points increasingly backward on
the long steep ascent, so more lift means less carry, and reproducing the
short empirical carry of steeply hit balls requires a larger effective Cl.
The constant-drag model structurally overshoots high-angle carry, so the
high-angle effective Cl absorbs that deficit. It is an effective parameter,
not a measured spin.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import PchipInterpolator

from . import config as C


def rho(temp_f: float, elev_m: float = C.DOME_ELEV_M, rh: float = 0.55) -> float:
    """Air density (kg/m^3) for temperature, elevation, and humidity.

    Arguments:
        temp_f: Temperature in degrees Fahrenheit.
        elev_m: Elevation in meters.
        rh: Relative humidity (0-1).

    Returns:
        Air density in kg/m^3.
    """
    tk = (temp_f - 32) * 5 / 9 + 273.15
    tc = tk - 273.15
    p = 101325 * (1 - 2.25577e-5 * elev_m) ** 5.2559
    psat = 610.94 * math.exp(17.625 * tc / (tc + 243.04))
    pv = rh * psat
    pd = p - pv
    return pd / (287.05 * tk) + pv / (461.495 * tk)


@dataclass
class Trajectory:
    """A simulated batted-ball flight."""

    x_ft: np.ndarray
    z_ft: np.ndarray
    range_ft: float
    apex_ft: float
    flight_s: float
    descent_angle_deg: float

    def height_at(self, dist_ft: float) -> float:
        """Trajectory height (ft) at a horizontal distance, 0 beyond landing."""
        if dist_ft >= self.range_ft:
            return 0.0
        return float(np.interp(dist_ft, self.x_ft, self.z_ft))


def carry(
    ev_mph: float, la_deg: float, cl: float, rho_: float, z0: float = C.LAUNCH_HEIGHT_M, dt: float = 0.001
) -> float:
    """Carry distance in feet (fast path, no trajectory stored).

    Arguments:
        ev_mph: Exit velocity in mph.
        la_deg: Launch angle in degrees.
        cl: Effective lift coefficient.
        rho_: Air density in kg/m^3.
        z0: Contact height in meters.
        dt: Integration time step in seconds.

    Returns:
        Carry (landing) distance in feet.
    """
    drag = 0.5 * rho_ * C.CD * C.BALL_AREA_M2 / C.BALL_MASS_KG
    lift = 0.5 * rho_ * cl * C.BALL_AREA_M2 / C.BALL_MASS_KG
    v = ev_mph * C.MPH_TO_MS
    th = math.radians(la_deg)
    vx = v * math.cos(th)
    vz = v * math.sin(th)
    x = 0.0
    z = z0
    xp = x
    zp = z
    g = C.GRAVITY
    for _ in range(200000):
        xp, zp = x, z
        sp = math.hypot(vx, vz)
        ax = -drag * sp * vx - lift * sp * vz
        az = -g - drag * sp * vz + lift * sp * vx
        vx += ax * dt
        vz += az * dt
        x += vx * dt
        z += vz * dt
        if z <= 0:
            break
        if x < -1:
            break
    if zp != z:
        frac = zp / (zp - z)
        x = xp + frac * (x - xp)
    return x * C.M_TO_FT


def integrate(
    ev_mph: float,
    la_deg: float,
    cl: float,
    rho_: float,
    z0: float = C.LAUNCH_HEIGHT_M,
    dt: float = 0.001,
) -> Trajectory:
    """Integrate a full batted-ball trajectory (stores the path).

    Arguments:
        ev_mph: Exit velocity in mph.
        la_deg: Launch angle in degrees.
        cl: Effective lift coefficient.
        rho_: Air density in kg/m^3.
        z0: Contact height in meters.
        dt: Integration time step in seconds.

    Returns:
        A :class:`Trajectory` (distances and heights in feet).
    """
    drag = 0.5 * rho_ * C.CD * C.BALL_AREA_M2 / C.BALL_MASS_KG
    lift = 0.5 * rho_ * cl * C.BALL_AREA_M2 / C.BALL_MASS_KG
    v = ev_mph * C.MPH_TO_MS
    th = np.radians(la_deg)
    vx = v * np.cos(th)
    vz = v * np.sin(th)
    x, z, t = 0.0, z0, 0.0
    xs, zs = [x], [z]
    for _ in range(200000):
        sp = np.hypot(vx, vz)
        ax = -drag * sp * vx - lift * sp * vz
        az = -C.GRAVITY - drag * sp * vz + lift * sp * vx
        vx += ax * dt
        vz += az * dt
        x += vx * dt
        z += vz * dt
        t += dt
        xs.append(x)
        zs.append(z)
        if z <= 0 or x < -1:
            break
    xs = np.array(xs) * C.M_TO_FT
    zs = np.array(zs) * C.M_TO_FT
    if zs[-1] <= 0 and len(zs) > 1:
        z1, z2 = zs[-2], zs[-1]
        x1, x2 = xs[-2], xs[-1]
        frac = z1 / (z1 - z2) if z1 != z2 else 1.0
        range_ft = float(x1 + frac * (x2 - x1))
    else:
        range_ft = float(xs[-1])
    descent = float(np.degrees(np.arctan2(-vz, vx)))
    return Trajectory(
        x_ft=xs, z_ft=zs, range_ft=range_ft, apex_ft=float(zs.max()), flight_s=t, descent_angle_deg=descent
    )


# --------------------------------------------------------------------------
# Calibrating the effective lift against the empirical mean-carry surface
# --------------------------------------------------------------------------

CL_KNOTS_LA = np.array([10.0, 14.0, 18.0, 22.0, 26.0, 30.0, 34.0, 38.0, 42.0, 46.0, 50.0])


@dataclass
class LiftFit:
    """A fitted effective-lift-vs-launch-angle model."""

    knot_la: np.ndarray
    knot_cl: np.ndarray
    rho_fit: float
    grid_pts: np.ndarray
    grid_obs: np.ndarray
    grid_model: np.ndarray
    grid_n: np.ndarray

    def _interp(self) -> PchipInterpolator:
        return PchipInterpolator(self.knot_la, self.knot_cl, extrapolate=True)

    def cl(self, la_deg: float | np.ndarray) -> np.ndarray:
        """Effective lift coefficient at a launch angle (clamped to knot range)."""
        la = np.clip(np.asarray(la_deg, dtype=float), self.knot_la.min(), self.knot_la.max())
        return self._interp()(la)

    def rmse(self) -> float:
        """Count-weighted RMSE of model vs empirical mean carry over the grid."""
        w = self.grid_n / self.grid_n.sum()
        return float(np.sqrt(np.sum(w * (self.grid_model - self.grid_obs) ** 2)))


GRID_EV_CENTERS = np.arange(92, 112, 3.0)
GRID_LA_CENTERS = np.arange(10, 51, 2.0)


def mean_carry_grid(
    df,
    ev_centers=None,
    la_centers=None,
    half_ev: float = 1.5,
    half_la: float = 1.5,
    min_n: int = 30,
):
    """Empirical mean carry on an (EV, LA) grid.

    Arguments:
        df: Batted-ball table.
        ev_centers: Exit-velocity bin centers (defaults to GRID_EV_CENTERS).
        la_centers: Launch-angle bin centers (defaults to GRID_LA_CENTERS).
        half_ev: Half-width of each EV bin.
        half_la: Half-width of each LA bin.
        min_n: Minimum count for a bin to be used.

    Returns:
        (points, means, counts) arrays.
    """
    ev_centers = GRID_EV_CENTERS if ev_centers is None else ev_centers
    la_centers = GRID_LA_CENTERS if la_centers is None else la_centers
    ev = df.launch_speed.to_numpy(dtype=float)
    la = df.launch_angle.to_numpy(dtype=float)
    d = df.hit_distance_sc.to_numpy(dtype=float)
    pts, means, counts = [], [], []
    for e in ev_centers:
        for a in la_centers:
            m = (np.abs(ev - e) <= half_ev) & (np.abs(la - a) <= half_la)
            n = int(m.sum())
            if n >= min_n:
                pts.append((e, a))
                means.append(float(d[m].mean()))
                counts.append(n)
    return np.array(pts), np.array(means), np.array(counts)


def fit_lift_curve(df, smooth: float = 1.0e4, dt: float = 0.004) -> LiftFit:
    """Fit a smooth effective Cl(LA) to the empirical mean-carry surface.

    The lift coefficient is parameterized by its value at a set of launch-angle
    knots. Those values are optimized jointly to minimize the squared error
    between model carry and empirical mean carry across the whole grid, with a
    light curvature penalty for smoothness. Each grid bin is weighted equally
    (every bin has at least ``min_n`` balls), so the sparse high-launch-angle
    region that the France profile lives in is fit as well as the dense core,
    rather than being swamped by the high-count bins near the distance peak.
    Fitting is done in average-MLB atmosphere.

    Note on scope: a constant-drag lift-only model reproduces carry well up to
    about 35 deg (including the ~405 ft peak near 30 deg) but structurally
    overshoots the steep 36-44 deg descent regime, where carry-vs-lift is
    non-monotone and the effective lift is poorly identified. The France
    profile (49 deg) sits in that hard regime, so the physics point there
    carries extra model uncertainty and is treated as a cross-check, not the
    primary estimate.

    Arguments:
        df: Batted-ball table.
        smooth: Curvature penalty weight (larger is smoother).
        dt: Integration step used during fitting (coarser for speed).

    Returns:
        A :class:`LiftFit`.
    """
    from scipy.optimize import minimize

    rho_fit = rho(C.MLB_TEMP_F, C.MLB_ELEV_M, C.MLB_RH)
    pts, obs, counts = mean_carry_grid(df)
    evs = pts[:, 0]
    las = pts[:, 1]
    w = np.ones_like(counts, dtype=float)

    def model_carry(knot_cl):
        interp = PchipInterpolator(CL_KNOTS_LA, knot_cl, extrapolate=True)
        cls = interp(las)
        return np.array([carry(e, a, max(cl, 0.0), rho_fit, dt=dt) for e, a, cl in zip(evs, las, cls)])

    def objective(knot_cl):
        model = model_carry(knot_cl)
        data_term = np.sum(w * (model - obs) ** 2)
        curv = np.diff(knot_cl, 2)
        pen = smooth * np.sum(curv**2)
        return data_term + pen

    x0 = np.full(CL_KNOTS_LA.size, 0.15)
    bounds = [(0.02, 0.6)] * CL_KNOTS_LA.size
    res = minimize(objective, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 400, "ftol": 1e-9})
    knot_cl = res.x
    # Recompute model carry at the fitted knots with the fit-time dt for reporting.
    grid_model = model_carry(knot_cl)
    return LiftFit(
        knot_la=CL_KNOTS_LA,
        knot_cl=knot_cl,
        rho_fit=rho_fit,
        grid_pts=pts,
        grid_obs=obs,
        grid_model=grid_model,
        grid_n=counts,
    )


def carry_vs_la(ev_mph: float, fit: LiftFit, rho_: float, la_values: np.ndarray, dt: float = 0.001) -> np.ndarray:
    """Model carry across launch angles at a fixed exit velocity."""
    return np.array([carry(ev_mph, a, float(fit.cl(a)), rho_, dt=dt) for a in la_values])


# --------------------------------------------------------------------------
# Evaluating at the France profile
# --------------------------------------------------------------------------


@dataclass
class PhysicsPoint:
    """Physics-model result at a single (EV, LA) point and atmosphere."""

    ev: float
    la: float
    cl: float
    temp_f: float
    rho: float
    carry_ft: float
    apex_ft: float
    descent_deg: float
    shortfall_ft: float


def evaluate_point(
    fit: LiftFit,
    ev: float,
    la: float,
    temp_f: float,
    elev_m: float,
    rh: float,
    cl_scale: float = 1.0,
    fence_ft: float = C.FENCE_FT,
) -> PhysicsPoint:
    """Evaluate the fitted model at a point under given atmosphere.

    Arguments:
        fit: The fitted lift model.
        ev: Exit velocity (mph).
        la: Launch angle (deg).
        temp_f: Temperature (F).
        elev_m: Elevation (m).
        rh: Relative humidity (0-1).
        cl_scale: Multiplier on the fitted Cl (spin sensitivity).
        fence_ft: Fence distance for the shortfall.

    Returns:
        A :class:`PhysicsPoint`.
    """
    rho_ = rho(temp_f, elev_m, rh)
    cl = float(fit.cl(la)) * cl_scale
    tr = integrate(ev, la, cl, rho_)
    return PhysicsPoint(
        ev=ev,
        la=la,
        cl=cl,
        temp_f=temp_f,
        rho=rho_,
        carry_ft=tr.range_ft,
        apex_ft=tr.apex_ft,
        descent_deg=tr.descent_angle_deg,
        shortfall_ft=fence_ft - tr.range_ft,
    )


def sensitivity_mc(
    fit: LiftFit,
    ev: float = C.FRANCE_EV,
    la: float = C.FRANCE_LA,
    elev_m: float = C.DOME_ELEV_M,
    rh: float = C.DOME_RH,
    temp_lo: float = 60.0,
    temp_hi: float = 80.0,
    cl_rel: float = 0.35,
    n: int = 4000,
    seed: int = 0,
) -> dict:
    """Crude Monte Carlo carry distribution from temperature and spin uncertainty.

    Temperature is sampled uniformly over a plausible dome range and the
    effective lift coefficient is scaled by a normal factor whose 2-sigma spans
    +/- ``cl_rel`` (standing in for the unmeasured backspin).

    Arguments:
        fit: The fitted lift model.
        ev: Exit velocity (mph).
        la: Launch angle (deg).
        elev_m: Elevation (m).
        rh: Relative humidity.
        temp_lo: Low temperature (F).
        temp_hi: High temperature (F).
        cl_rel: Relative spin range (2-sigma).
        n: Number of Monte Carlo draws.
        seed: RNG seed.

    Returns:
        Dict with the carry samples and summary statistics.
    """
    rng = np.random.default_rng(seed)
    temps = rng.uniform(temp_lo, temp_hi, n)
    cl_scales = rng.normal(1.0, cl_rel / 2.0, n).clip(0.3, 2.0)
    cl0 = float(fit.cl(la))
    carries = np.empty(n)
    for i in range(n):
        rho_ = rho(float(temps[i]), elev_m, rh)
        tr_cl = cl0 * float(cl_scales[i])
        carries[i] = carry(ev, la, tr_cl, rho_)
    base = integrate(ev, la, cl0, rho(float(np.mean([temp_lo, temp_hi])), elev_m, rh))
    return {
        "carries": carries,
        "cl0": cl0,
        "mean": float(carries.mean()),
        "median": float(np.median(carries)),
        "p05": float(np.percentile(carries, 5)),
        "p95": float(np.percentile(carries, 95)),
        "descent_deg": base.descent_angle_deg,
    }
